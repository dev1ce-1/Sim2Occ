"""Candidate retrieval, balanced/dustbin OT, and real transport alignment."""
import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .protocol import (
    GEOMETRY_WEIGHT,
    MethodSpec,
    SINKHORN_EPSILON,
    SINKHORN_MARGINAL_TOLERANCE,
    SINKHORN_MAX_ITER,
)
from .tokens import TokenDescriptor, TokenRecord


@dataclass
class TransportResult:
    query_indices: List[int]
    candidate_indices: List[int]
    plan: torch.Tensor
    values: torch.Tensor
    match_ratio: torch.Tensor
    row_residual: float
    col_residual: float
    entropy: float
    dustbin_mass: float
    candidate_changed: bool
    shuffle_changed_fraction: float
    value_change_l2: float


def _stable_seed(*parts) -> int:
    text = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big") % (2 ** 63 - 1)


def feature_distance(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q.float(), dim=-1, eps=1e-12)
    k = F.normalize(k.float(), dim=-1, eps=1e-12)
    return (1.0 - q @ k.transpose(0, 1)).clamp(-1e-6, 2.0 + 1e-6) / 2.0


def coarse_distance(query: Sequence[TokenDescriptor], source: Sequence[TokenDescriptor], device) -> torch.Tensor:
    q = torch.stack([d.g0 for d in query]).to(device=device, dtype=torch.float32)
    s = torch.stack([d.g0 for d in source]).to(device=device, dtype=torch.float32)
    return (q[:, None, :] - s[None, :, :]).abs().mean(dim=-1).clamp(0.0, 1.0)


def fine_distance(query: Sequence[TokenDescriptor], source: Sequence[TokenDescriptor], device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qs = torch.stack([d.support for d in query]).to(device=device, dtype=torch.float32)
    ss = torch.stack([d.support for d in source]).to(device=device, dtype=torch.float32)
    qp = torch.stack([d.occupancy for d in query]).to(device=device, dtype=torch.float32)
    sp = torch.stack([d.occupancy for d in source]).to(device=device, dtype=torch.float32)
    qv = torch.stack([d.occupancy_valid for d in query]).to(device=device)
    sv = torch.stack([d.occupancy_valid for d in source]).to(device=device)
    w = torch.minimum(qs[:, None, :], ss[None, :, :])
    common_valid = qv[:, None, :] & sv[None, :, :]
    w = w * common_valid.float()
    denom = w.sum(dim=-1)
    dist = (w * (qp[:, None, :] - sp[None, :, :]).abs()).sum(dim=-1) / denom.clamp_min(1e-12)
    has_common = denom > 0
    dist = torch.where(has_common, dist, torch.ones_like(dist)).clamp(0.0, 1.0)
    overlap_denom = torch.maximum(qs[:, None, :], ss[None, :, :]).sum(dim=-1)
    overlap = torch.minimum(qs[:, None, :], ss[None, :, :]).sum(dim=-1) / overlap_denom.clamp_min(1e-12)
    overlap = torch.where(overlap_denom > 0, overlap, torch.zeros_like(overlap)).clamp(0.0, 1.0)
    return dist, overlap, has_common


def log_sinkhorn(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float = SINKHORN_EPSILON,
    max_iter: int = SINKHORN_MAX_ITER,
    tolerance: float = SINKHORN_MARGINAL_TOLERANCE,
    allowed: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float, float, int]:
    """Log-domain balanced Sinkhorn with actual marginal residual stopping."""
    if cost.ndim != 2 or a.ndim != 1 or b.ndim != 1:
        raise ValueError("cost/a/b dimensions are invalid")
    if cost.shape != (a.numel(), b.numel()):
        raise ValueError("marginal dimensions do not match cost")
    if not torch.isfinite(cost).all() or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError("nonfinite transport input")
    if (a <= 0).any() or (b <= 0).any() or abs(float(a.sum() - b.sum())) > 1e-5:
        raise ValueError("marginals must be positive and equal-mass")
    log_k = -cost.float() / float(epsilon)
    if allowed is not None:
        if allowed.shape != cost.shape:
            raise ValueError("allowed mask shape mismatch")
        log_k = torch.where(allowed, log_k, torch.full_like(log_k, -1e9))
    log_a, log_b = a.log(), b.log()
    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)
    row_residual = col_residual = float("inf")
    iterations = max_iter
    row_projected = None
    for iteration in range(max_iter):
        log_u = log_a - torch.logsumexp(log_k + log_v[None, :], dim=1)
        # Retain the row-normalised half iterate from the same Sinkhorn
        # iteration.  Sparse dustbin problems can converge from opposite
        # sides: the row half iterate has exact rows, while the following
        # column half iterate has exact columns.  If the latter narrowly
        # misses the frozen residual gate after the final iteration, their
        # ergodic average is a support-preserving numerical stabilisation
        # formed entirely from the prescribed iteration budget.
        row_projected = torch.exp(log_u[:, None] + log_k + log_v[None, :])
        log_v = log_b - torch.logsumexp(log_k + log_u[:, None], dim=0)
        log_p = log_u[:, None] + log_k + log_v[None, :]
        plan = torch.exp(log_p)
        row_residual = float((plan.sum(dim=1) - a).abs().max().detach())
        col_residual = float((plan.sum(dim=0) - b).abs().max().detach())
        if max(row_residual, col_residual) <= tolerance:
            iterations = iteration + 1
            break
    plan = torch.exp(log_u[:, None] + log_k + log_v[None, :])
    if not torch.isfinite(plan).all():
        raise FloatingPointError("nonfinite transport plan")
    if max(row_residual, col_residual) > tolerance:
        averaged = 0.5 * (row_projected + plan)
        averaged_row_residual = float((averaged.sum(dim=1) - a).abs().max().detach())
        averaged_col_residual = float((averaged.sum(dim=0) - b).abs().max().detach())
        if max(averaged_row_residual, averaged_col_residual) <= tolerance:
            plan = averaged
            row_residual = averaged_row_residual
            col_residual = averaged_col_residual
        else:
            # Versioned numerical recovery: keep the original accepted path
            # exactly unchanged. Continue the same dual problem only on failure.
            initial_residuals = [row_residual, col_residual,
                                 averaged_row_residual, averaged_col_residual]
            log_k64 = -cost.double() / float(epsilon)
            if allowed is not None:
                log_k64 = torch.where(allowed, log_k64, torch.full_like(log_k64, -float("inf")))
            a64, b64 = a.double(), b.double()
            u64, v64 = log_u.double(), log_v.double()
            converged = False
            for iteration in range(max_iter, max_iter + 1900):
                u64 = a64.log() - torch.logsumexp(log_k64 + v64[None, :], dim=1)
                v64 = b64.log() - torch.logsumexp(log_k64 + u64[:, None], dim=0)
                # Gate the actual returned float32 plan, not just its float64 precursor.
                candidate = torch.exp(u64[:, None] + log_k64 + v64[None, :]).float()
                if not torch.isfinite(candidate).all():
                    raise FloatingPointError("nonfinite transport recovery plan")
                rr = float((candidate.sum(1) - a).abs().max().detach())
                cr = float((candidate.sum(0) - b).abs().max().detach())
                if max(rr, cr) <= tolerance:
                    plan, row_residual, col_residual = candidate, rr, cr
                    iterations = iteration + 1
                    converged = True
                    break
            # Record every exceptional solve, including the original problem,
            # for direct replay against the old and repaired solvers.
            import os, json, time
            from pathlib import Path
            audit = os.environ.get("LANE_SINKHORN_AUDIT_DIR")
            if audit:
                folder = Path(audit); folder.mkdir(parents=True, exist_ok=True)
                ident = "%d_%d" % (os.getpid(), time.time_ns())
                snapshot = folder / (ident + ".pth")
                torch.save(dict(cost=cost.detach().cpu(), a=a.detach().cpu(), b=b.detach().cpu(),
                                allowed=None if allowed is None else allowed.detach().cpu(),
                                epsilon=epsilon, max_iter=max_iter, tolerance=tolerance), str(snapshot))
                with (folder / "events.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(version="same_tolerance_continuation_v1",
                        time=time.time(), snapshot=str(snapshot), initial_residuals=initial_residuals,
                        converged=converged, iterations=iterations if converged else max_iter+1900,
                        row_residual=rr, col_residual=cr, tolerance=tolerance)) + "\n")
            if not converged:
                raise RuntimeError("Sinkhorn recovery exhausted without meeting unchanged tolerance: row=%g col=%g tolerance=%g" % (rr, cr, tolerance))
    return plan, row_residual, col_residual, iterations


def _candidate_lists(
    spec: MethodSpec,
    queries: Sequence[TokenDescriptor],
    query_keys: torch.Tensor,
    bank: Sequence[TokenRecord],
    seed: int,
    event: int,
    k: int = 16,
) -> Tuple[List[List[int]], bool]:
    lists = []
    any_changed = False
    for qi, query in enumerate(queries):
        eligible = [
            idx for idx, row in enumerate(bank)
            if row.descriptor.class_id == query.class_id and
            not (spec.bank_domain == "real" and row.descriptor.scene_token == query.scene_token)
        ]
        if not eligible:
            lists.append([])
            continue
        keys = torch.stack([bank[idx].key for idx in eligible]).to(query_keys.device)
        df = feature_distance(query_keys[qi:qi + 1], keys)[0]
        source_desc = [bank[idx].descriptor for idx in eligible]
        dg0 = coarse_distance([query], source_desc, query_keys.device)[0]
        base_score = 0.5 * df + 0.5 * dg0
        if spec.module_a:
            dg1, overlap, _ = fine_distance([query], source_desc, query_keys.device)
            score = 0.5 * df + 0.5 * dg1[0] + 0.25 * (1.0 - overlap[0])
        else:
            score = base_score
        base_order = sorted(range(len(eligible)), key=lambda j: (float(base_score[j]), eligible[j]))[:k]
        if spec.random_candidates:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_stable_seed(seed, event, query.identity, "random"))
            order = torch.randperm(len(eligible), generator=generator).tolist()[:k]
        else:
            order = sorted(range(len(eligible)), key=lambda j: (float(score[j]), eligible[j]))[:k]
        chosen = [eligible[j] for j in order]
        lists.append(chosen)
        # Candidate quality is a set-level claim.  A mere rank permutation does
        # not alter the dense OT union and must not be counted as an A change.
        any_changed |= set(chosen) != {eligible[j] for j in base_order}
    return lists, any_changed


def _entropy(plan: torch.Tensor) -> float:
    positive = plan[plan > 0]
    return float((-(positive * positive.log()).sum()).detach()) if positive.numel() else 0.0


def solve_block(
    spec: MethodSpec,
    query_indices: Sequence[int],
    descriptors: Sequence[TokenDescriptor],
    query_keys: torch.Tensor,
    bank: Sequence[TokenRecord],
    seed: int,
    event: int,
    retrieval_top_k: int = 16,
    dustbin_cost: float = 0.5,
) -> Optional[TransportResult]:
    block_desc = [descriptors[i] for i in query_indices]
    block_keys = query_keys[list(query_indices)]
    lists, candidate_changed = _candidate_lists(spec, block_desc, block_keys, bank, seed, event, k=retrieval_top_k)
    union = sorted({idx for rows in lists for idx in rows})
    if not union:
        return None
    source_desc = [bank[idx].descriptor for idx in union]
    source_keys = torch.stack([bank[idx].key for idx in union]).to(block_keys.device)
    source_values = torch.stack([bank[idx].value for idx in union]).to(block_keys.device)
    df = feature_distance(block_keys, source_keys)
    dg0 = coarse_distance(block_desc, source_desc, block_keys.device)
    if spec.module_b:
        dg1, overlap, common = fine_distance(block_desc, source_desc, block_keys.device)
        real_cost = (1.0 - GEOMETRY_WEIGHT) * df + GEOMETRY_WEIGHT * dg1 + 0.25 * (1.0 - overlap)
        n, m = real_cost.shape
        cost = real_cost.new_full((n + 1, m + 1), float(dustbin_cost))
        cost[:n, :m] = real_cost
        cost[n, m] = 0.0
        allowed = torch.ones_like(cost, dtype=torch.bool)
        allowed[:n, :m] = common
        a = torch.cat((torch.full((n,), 1.0 / n, device=cost.device), cost.new_tensor([1.0])))
        b = torch.cat((torch.full((m,), 1.0 / m, device=cost.device), cost.new_tensor([1.0])))
        full, rr, cr, _ = log_sinkhorn(cost, a, b, allowed=allowed)
        plan = full[:n, :m]
        row_mass = plan.sum(dim=1)
        ratio = row_mass / a[:n]
        # The absolute row-marginal tolerance implies a relative tolerance of
        # tol/a_i for r_i.  Using a tighter unrelated constant would reject an
        # otherwise accepted plan, especially for four-query blocks (a_i=.25).
        ratio_tolerance = SINKHORN_MARGINAL_TOLERANCE / float(a[:n].min()) + 1e-6
        if (ratio < -ratio_tolerance).any() or (ratio > 1.0 + ratio_tolerance).any():
            raise RuntimeError(
                "dustbin match ratio outside residual-derived tolerance: "
                "min=%g max=%g tol=%g row_residual=%g" %
                (float(ratio.min()), float(ratio.max()), ratio_tolerance, rr)
            )
        ratio = ratio.clamp(0.0, 1.0)
        dustbin_mass = float((full[:n, m].sum() + full[n, :m].sum()).detach())
    else:
        cost = 0.5 * df + 0.5 * dg0
        n, m = cost.shape
        a = torch.full((n,), 1.0 / n, device=cost.device)
        b = torch.full((m,), 1.0 / m, device=cost.device)
        plan, rr, cr, _ = log_sinkhorn(cost, a, b)
        row_mass = plan.sum(dim=1)
        ratio = torch.ones(n, device=cost.device)
        dustbin_mass = 0.0

    aligned = plan @ source_values / row_mass[:, None].clamp_min(1e-12)
    original_aligned = aligned
    shuffle_fraction = 0.0
    value_change = 0.0
    if spec.shuffle_values:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_seed(seed, event, *(d.identity for d in block_desc), "shuffle"))
        perm = torch.randperm(source_values.shape[0], generator=generator).to(source_values.device)
        shuffled = source_values[perm]
        aligned = plan @ shuffled / row_mass[:, None].clamp_min(1e-12)
        shuffle_fraction = float((perm != torch.arange(len(perm), device=perm.device)).float().mean())
        value_change = float((aligned - original_aligned).norm(dim=1).mean().detach())
    return TransportResult(
        query_indices=list(query_indices),
        candidate_indices=union,
        plan=plan,
        values=aligned,
        match_ratio=ratio,
        row_residual=rr,
        col_residual=cr,
        entropy=_entropy(plan),
        dustbin_mass=dustbin_mass,
        candidate_changed=candidate_changed,
        shuffle_changed_fraction=shuffle_fraction,
        value_change_l2=value_change,
    )


def align_queries(
    spec: MethodSpec,
    descriptors: Sequence[TokenDescriptor],
    query_keys: torch.Tensor,
    bank: Sequence[TokenRecord],
    seed: int,
    event: int,
    retrieval_top_k: int = 16,
    dustbin_cost: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
    """Return aligned values, ratios, valid flags in original query order."""
    count, channels = query_keys.shape
    aligned = query_keys.new_zeros((count, channels))
    ratios = query_keys.new_zeros(count)
    accepted = torch.zeros(count, dtype=torch.bool, device=query_keys.device)
    diagnostics = []
    if spec.prototype:
        for idx, desc in enumerate(descriptors):
            rows = [r.value for r in bank if r.descriptor.class_id == desc.class_id]
            if rows:
                aligned[idx] = torch.stack(rows).to(query_keys.device).mean(dim=0)
                ratios[idx] = 1.0
                accepted[idx] = True
        return aligned, ratios, accepted, diagnostics

    by_class = {}
    for idx, desc in enumerate(descriptors):
        by_class.setdefault(desc.class_id, []).append(idx)
    for class_id, indices in sorted(by_class.items()):
        for start in range(0, len(indices), 4):
            block = indices[start:start + 4]
            result = solve_block(spec, block, descriptors, query_keys, bank, seed, event, retrieval_top_k=retrieval_top_k, dustbin_cost=dustbin_cost)
            if result is None:
                continue
            for local, global_idx in enumerate(result.query_indices):
                if float(result.match_ratio[local]) > 1e-12:
                    aligned[global_idx] = result.values[local]
                    ratios[global_idx] = result.match_ratio[local]
                    accepted[global_idx] = True
            diagnostics.append({
                "class_id": class_id,
                "queries": len(block),
                "candidate_union": len(result.candidate_indices),
                "requested_top_k": int(retrieval_top_k),
                "dustbin_cost": float(dustbin_cost),
                "geometry_weight": GEOMETRY_WEIGHT,
                "row_residual": result.row_residual,
                "col_residual": result.col_residual,
                "entropy": result.entropy,
                "dustbin_mass": result.dustbin_mass,
                "match_ratio": [float(x) for x in result.match_ratio.detach().cpu()],
                "candidate_changed": result.candidate_changed,
                "shuffle_changed_fraction": result.shuffle_changed_fraction,
                "value_change_l2": result.value_change_l2,
            })
    return aligned, ratios, accepted, diagnostics


def transfer_loss(
    student_tokens: torch.Tensor,
    aligned_values: torch.Tensor,
    match_ratio: torch.Tensor,
    accepted: torch.Tensor,
    n_all: int,
    eta: float = 0.5,
) -> torch.Tensor:
    if n_all <= 0:
        return student_tokens.sum() * 0.0
    z = F.normalize(student_tokens, dim=-1, eps=1e-12)
    v = F.normalize(aligned_values.detach(), dim=-1, eps=1e-12)
    r = match_ratio.detach().clamp(0.0, 1.0)
    target_raw = (1.0 - eta * r[:, None]) * z.detach() + eta * r[:, None] * v
    target_norm = target_raw.norm(dim=-1)
    valid = accepted & (target_norm > 1e-12)
    if not valid.any():
        return z.sum() * 0.0
    target = F.normalize(target_raw[valid], dim=-1, eps=1e-12).detach()
    terms = 1.0 - F.cosine_similarity(z[valid], target, dim=-1, eps=1e-12)
    return terms.sum() / float(n_all)
