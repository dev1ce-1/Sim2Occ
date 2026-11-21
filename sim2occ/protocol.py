"""Default parameters of retrieval-guided partial alignment (MatchMass-Occ)."""
from dataclasses import asdict, dataclass

GEOMETRY_WEIGHT = 0.5
SINKHORN_EPSILON = 0.05
SINKHORN_MAX_ITER = 100
SINKHORN_MARGINAL_TOLERANCE = 0.01

@dataclass(frozen=True)
class MethodSpec:
    name: str
    memory: bool
    replay: bool = False
    prototype: bool = False
    module_a: bool = False
    module_b: bool = False
    random_candidates: bool = False
    shuffle_values: bool = False
    bank_domain: str = "none"

    def to_dict(self):
        return asdict(self)


RPA = MethodSpec("B", memory=True, module_b=True, bank_domain="source")
