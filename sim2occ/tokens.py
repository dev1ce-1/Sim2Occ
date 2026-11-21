"""Deterministic occupancy-token construction and serializable memory banks."""
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

WINDOW = 5
HALF = WINDOW // 2
Z_BINS = 4
XY_BINS = 2
DESC_BINS = XY_BINS * XY_BINS * Z_BINS


def _stable_int(*parts) -> int:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _coordinate_order(coords: torch.Tensor, seed: int, frame_token: str,
                      class_id: int, limit: int) -> List[Tuple[int, int]]:
    """Return the lowest fixed-hash coordinates without Python hashing every cell.

    A dense road-surface class can contain tens of thousands of columns.  Calling
    hashlib once per column retained large temporary allocator arenas during a
    full 171-frame bank build.  This vectorized SplitMix64 permutation preserves
    the frozen hash-sampling meaning while retaining only the at-most-32 entries
    that the image-level round robin can ever consume.
    """
    if coords.numel() == 0 or limit <= 0:
        return []
    xy = coords.detach().cpu().numpy().astype(np.uint64, copy=False)
    base = np.uint64(_stable_int(seed, frame_token, class_id, "coordinate"))
    with np.errstate(over="ignore"):
        values = base ^ (xy[:, 0] * np.uint64(0x9E3779B97F4A7C15))
        values ^= xy[:, 1] * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        values ^= values >> np.uint64(31)
    order = np.lexsort((xy[:, 1], xy[:, 0], values))[:limit]
    return [(int(xy[i, 0]), int(xy[i, 1])) for i in order]


@dataclass
class TokenDescriptor:
    identity: str
    frame_token: str
    scene_token: str
    class_id: int
    x: int
    y: int
    g0: torch.Tensor
    support: torch.Tensor
    occupancy: torch.Tensor
    occupancy_valid: torch.Tensor
    boundary: torch.Tensor

    def as_dict(self) -> Dict:
        return {
            "identity": self.identity,
            "frame_token": self.frame_token,
            "scene_token": self.scene_token,
            "class_id": int(self.class_id),
            "x": int(self.x),
            "y": int(self.y),
            "g0": self.g0.detach().cpu(),
            "support": self.support.detach().cpu(),
            "occupancy": self.occupancy.detach().cpu(),
            "occupancy_valid": self.occupancy_valid.detach().cpu(),
            "boundary": self.boundary.detach().cpu(),
        }

    @classmethod
    def from_dict(cls, row: Dict):
        return cls(**row)


@dataclass
class TokenRecord:
    descriptor: TokenDescriptor
    key: torch.Tensor
    value: torch.Tensor

    def as_dict(self) -> Dict:
        row = self.descriptor.as_dict()
        row.update(key=self.key.detach().cpu(), value=self.value.detach().cpu())
        return row

    @classmethod
    def from_dict(cls, row: Dict):
        desc_keys = {
            "identity", "frame_token", "scene_token", "class_id", "x", "y",
            "g0", "support", "occupancy", "occupancy_valid", "boundary",
        }
        desc = TokenDescriptor.from_dict({k: row[k] for k in desc_keys})
        return cls(desc, row["key"], row["value"])


def _bounds(x: int, y: int, nx: int, ny: int) -> Tuple[int, int, int, int]:
    return max(0, x - HALF), min(nx, x + HALF + 1), max(0, y - HALF), min(ny, y + HALF + 1)


def _boundary_stats(labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Counts known 6-neighbour geometry/class boundaries and known interiors."""
    occupied = labels < 17
    geom = torch.zeros_like(valid, dtype=torch.bool)
    semantic = torch.zeros_like(valid, dtype=torch.bool)
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        left, right = tuple(left), tuple(right)
        supported = valid[left] & valid[right]
        dg = supported & (occupied[left] != occupied[right])
        ds = supported & occupied[left] & occupied[right] & (labels[left] != labels[right])
        geom[left] |= dg
        geom[right] |= dg
        semantic[left] |= ds
        semantic[right] |= ds
    semantic &= ~geom
    interior = valid & ~(geom | semantic)
    return torch.tensor(
        [int(geom.sum()), int(semantic.sum()), int(interior.sum())], dtype=torch.int64
    )


def describe_token(
    semantics: torch.Tensor,
    valid: torch.Tensor,
    frame_token: str,
    scene_token: str,
    class_id: int,
    x: int,
    y: int,
) -> TokenDescriptor:
    """Build g0/g1 from a [X,Y,Z] label grid without treating unknown as free."""
    if semantics.ndim != 3 or valid.shape != semantics.shape:
        raise ValueError("expected aligned [X,Y,Z] semantics and valid mask")
    nx, ny, nz = semantics.shape
    x0, x1, y0, y1 = _bounds(x, y, nx, ny)
    local = semantics[x0:x1, y0:y1, :]
    known = valid[x0:x1, y0:y1, :].bool() & local.ne(255)
    cls = known & local.eq(int(class_id))
    coords = torch.nonzero(cls, as_tuple=False)
    if coords.numel() == 0:
        raise ValueError("token center class has no known occupied support")
    z = coords[:, 2].float()
    mean_h = (z.mean() + 0.5) / float(nz)
    std_h = z.std(unbiased=False) / float(nz)
    span_x = (coords[:, 0].max() - coords[:, 0].min() + 1).float() / float(WINDOW)
    span_y = (coords[:, 1].max() - coords[:, 1].min() + 1).float() / float(WINDOW)
    g0 = torch.stack((mean_h, std_h, span_x, span_y)).to(torch.float32)

    support = torch.zeros(DESC_BINS, dtype=torch.float32)
    occupancy = torch.zeros(DESC_BINS, dtype=torch.float32)
    occupancy_valid = torch.zeros(DESC_BINS, dtype=torch.bool)
    cursor = 0
    lx, ly, _ = local.shape
    for bx in range(XY_BINS):
        xa = (bx * lx) // XY_BINS
        xb = ((bx + 1) * lx) // XY_BINS
        for by in range(XY_BINS):
            ya = (by * ly) // XY_BINS
            yb = ((by + 1) * ly) // XY_BINS
            for bz in range(Z_BINS):
                za = (bz * nz) // Z_BINS
                zb = ((bz + 1) * nz) // Z_BINS
                k = known[xa:xb, ya:yb, za:zb]
                c = cls[xa:xb, ya:yb, za:zb]
                possible = k.numel()
                count = int(k.sum())
                support[cursor] = 0.0 if possible == 0 else count / float(possible)
                if count:
                    occupancy[cursor] = float(c.sum()) / float(count)
                    occupancy_valid[cursor] = True
                cursor += 1
    identity = "%s:%02d:%03d:%03d" % (frame_token, class_id, x, y)
    return TokenDescriptor(
        identity=identity,
        frame_token=frame_token,
        scene_token=scene_token,
        class_id=int(class_id),
        x=int(x),
        y=int(y),
        g0=g0,
        support=support,
        occupancy=occupancy,
        occupancy_valid=occupancy_valid,
        boundary=_boundary_stats(local, known),
    )


def select_descriptors(
    semantics: torch.Tensor,
    valid: torch.Tensor,
    frame_token: str,
    scene_token: str,
    seed: int,
    max_tokens: int = 32,
) -> Tuple[List[TokenDescriptor], int]:
    """Select occupied tokens by class round-robin and stable identity hash."""
    if semantics.device.type != "cpu":
        semantics = semantics.detach().cpu()
    if valid.device.type != "cpu":
        valid = valid.detach().cpu()
    valid = valid.bool() & semantics.ne(255)
    queues = {}
    present = set(int(x) for x in torch.unique(semantics[valid & semantics.lt(17)]).tolist())
    for class_id in sorted(present):
        xy = torch.nonzero((valid & semantics.eq(class_id)).any(dim=2), as_tuple=False)
        coords = _coordinate_order(xy, seed, frame_token, class_id, max_tokens)
        if coords:
            queues[class_id] = deque(coords)
    selected = []
    class_order = sorted(queues, key=lambda c: _stable_int(seed, frame_token, "class", c))
    while class_order and len(selected) < max_tokens:
        next_order = []
        for class_id in class_order:
            if len(selected) >= max_tokens:
                break
            q = queues[class_id]
            x, y = q.popleft()
            selected.append(describe_token(semantics, valid, frame_token, scene_token, class_id, x, y))
            if q:
                next_order.append(class_id)
        class_order = next_order
    return selected, len(selected)


def gather_features(feature: torch.Tensor, descriptors: Sequence[TokenDescriptor]) -> torch.Tensor:
    """Mean BEV features at columns containing the token class; feature is [C,Y,X]."""
    if feature.ndim != 3:
        raise ValueError("feature must be [C,Y,X]")
    _, ny, nx = feature.shape
    vectors = []
    for desc in descriptors:
        x0, x1, y0, y1 = _bounds(desc.x, desc.y, nx, ny)
        # Descriptor construction guarantees the centre class occupies at least the
        # centre column. Use the full 5x5 local feature support only at occupied
        # class columns; callers pass the aligned column mask separately when needed.
        # With no label tensor here, retain the deterministic centre-column vector.
        # extract_feature_records supplies the exact occupied-column mean below.
        vectors.append(feature[:, desc.y, desc.x])
    if not vectors:
        return feature.new_zeros((0, feature.shape[0]))
    return torch.stack(vectors, dim=0)


def gather_occupied_column_features(
    feature: torch.Tensor,
    semantics: torch.Tensor,
    valid: torch.Tensor,
    descriptors: Sequence[TokenDescriptor],
) -> torch.Tensor:
    """Exact 5x5 occupied-column feature mean for each descriptor."""
    if feature.ndim != 3:
        raise ValueError("feature must be [C,Y,X]")
    nx, ny, _ = semantics.shape
    vectors = []
    for desc in descriptors:
        x0, x1, y0, y1 = _bounds(desc.x, desc.y, nx, ny)
        columns = (valid[x0:x1, y0:y1, :].bool() & semantics[x0:x1, y0:y1, :].eq(desc.class_id)).any(dim=2)
        xy = torch.nonzero(columns, as_tuple=False)
        if xy.numel() == 0:
            raise RuntimeError("descriptor lost occupied feature columns")
        ys = xy[:, 1].to(feature.device) + y0
        xs = xy[:, 0].to(feature.device) + x0
        vectors.append(feature[:, ys, xs].mean(dim=1))
    if not vectors:
        return feature.new_zeros((0, feature.shape[0]))
    return torch.stack(vectors, dim=0)


def limit_bank(records: Iterable[TokenRecord], seed: int, per_class=64, total: int = 1024) -> List[TokenRecord]:
    """Class cap with deterministic cross-frame round-robin."""
    by_class_frame = defaultdict(lambda: defaultdict(list))
    for record in records:
        d = record.descriptor
        by_class_frame[d.class_id][d.frame_token].append(record)
    kept = []
    for class_id in sorted(by_class_frame):
        frames = by_class_frame[class_id]
        for values in frames.values():
            values.sort(key=lambda r: _stable_int(seed, r.descriptor.identity))
        frame_order = sorted(frames, key=lambda f: _stable_int(seed, class_id, f))
        queues = {f: deque(frames[f]) for f in frame_order}
        class_kept = []
        class_cap = int(per_class.get(class_id, 0) if isinstance(per_class, dict) else per_class)
        while frame_order and len(class_kept) < class_cap:
            next_order = []
            for frame in frame_order:
                if len(class_kept) >= class_cap:
                    break
                class_kept.append(queues[frame].popleft())
                if queues[frame]:
                    next_order.append(frame)
            frame_order = next_order
        kept.extend(class_kept)
    kept.sort(key=lambda r: (r.descriptor.class_id, _stable_int(seed, r.descriptor.identity)))
    return kept[:total]


def bank_summary(records: Sequence[TokenRecord]) -> Dict:
    per_class = {}
    for class_id in range(17):
        rows = [r for r in records if r.descriptor.class_id == class_id]
        per_class[str(class_id)] = {
            "tokens": len(rows),
            "frames": len({r.descriptor.frame_token for r in rows}),
            "scenes": len({r.descriptor.scene_token for r in rows}),
        }
    return {"tokens": len(records), "per_class": per_class}
