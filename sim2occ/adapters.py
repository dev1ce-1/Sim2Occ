"""Native dense-feature sampling and auxiliary-gradient budgeting."""
import math

import torch

def occupied_features(feature,descriptors,semantics,valid):
    if tuple(feature.shape[1:])!=tuple(semantics.shape):
        raise ValueError('Feature and label coordinate grids differ')
    out=[];nx,ny,nz=semantics.shape
    for d in descriptors:
        x0,x1=max(0,d.x-2),min(nx,d.x+3)
        y0,y1=max(0,d.y-2),min(ny,d.y+3)
        support=valid[x0:x1,y0:y1].bool() & semantics[x0:x1,y0:y1].eq(d.class_id)
        assert support.any()
        out.append(feature[:,x0:x1,y0:y1,:][:,support.to(feature.device)].mean(-1))
    return torch.stack(out)


def gradients_norm(grads) -> float:
    terms = [g.detach().float().pow(2).sum(dtype=torch.float64) for g in grads if g is not None]
    if not terms:
        return 0.0
    return float(torch.sqrt(torch.stack(terms).sum()))


def _gradient_dot(left, right):
    dot = None
    shared = 0
    for a, b in zip(left, right):
        if a is not None and b is not None:
            term = (a.detach().float() * b.detach().float()).sum(dtype=torch.float64)
            dot = term if dot is None else dot + term
            shared += 1
    return (0.0 if dot is None else float(dot), shared)


def merge_gradients(params, main_grads, aux_grads, maximum_weight):
    main_norm = gradients_norm(main_grads)
    aux_norm = gradients_norm(aux_grads)
    dot, shared = _gradient_dot(main_grads, aux_grads)
    cosine = None if main_norm <= 1e-12 or aux_norm <= 1e-12 else dot / (main_norm * aux_norm)
    if not all(math.isfinite(x) for x in (main_norm, aux_norm, dot)) or (cosine is not None and not math.isfinite(cosine)):
        raise FloatingPointError("nonfinite Phase2G gradient statistics")
    if main_norm <= 1e-12:
        raise RuntimeError("main gradient norm is zero")
    weight = 0.0 if aux_norm <= 1e-12 else min(float(maximum_weight), 0.10 * main_norm / aux_norm)
    for parameter, main, auxiliary in zip(params, main_grads, aux_grads):
        if main is None:
            parameter.grad = None if auxiliary is None or weight == 0 else auxiliary.detach().mul(weight)
        elif auxiliary is None or weight == 0:
            parameter.grad = main.detach()
        else:
            parameter.grad = main.detach().add(auxiliary.detach(), alpha=weight)
    return {
        "main_norm": main_norm,
        "aux_norm": aux_norm,
        "dot": dot,
        "cosine": cosine,
        "shared_parameter_tensors": shared,
        "applied_weight": weight,
        "weighted_aux_ratio": 0.0 if main_norm <= 1e-12 else weight * aux_norm / main_norm,
    }
