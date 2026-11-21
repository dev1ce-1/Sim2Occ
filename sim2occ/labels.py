"""Candidate observed-occupancy labels; pure resolver, no file or training edits.

Input evidence must come from synchronized, calibrated CARLA measurements.
Free-ray traversal, semantic mapping and visibility masks are kept unchanged.
"""
import numpy as np

VERSION = 'carla_observed_occ_v0.3_surface_priority_candidate'


def resolve(grid):
    """Accept an EvidenceGrid-compatible object and return labels plus audit arrays.

    A known, unambiguous surface endpoint wins over a free traversal in the same
    coarse voxel. Unknown endpoints and multiple known classes remain ignored.
    Diagnostic conflict flags describe raw evidence, not final label validity.
    """
    old = grid.finish()
    result = {key: value.copy() for key, value in old.items()}
    bits = np.asarray(grid.bits)
    occupied = np.asarray(grid.occupied, dtype=bool)
    ambiguous = np.asarray(grid.ambiguous, dtype=bool)
    if bits.dtype != np.uint32:
        raise ValueError('Expected uint32 semantic bitset')
    if not (bits.shape == occupied.shape == ambiguous.shape == old['semantics'].shape):
        raise ValueError('Evidence shapes do not agree')
    if np.any(bits & np.uint32(~((1 << 17) - 1) & 0xffffffff)):
        raise ValueError('Only occupied semantic IDs 0..16 are accepted')
    if np.any(((bits != 0) | ambiguous) & ~occupied):
        raise ValueError('Semantic evidence without a surface endpoint')
    classes_conflict = (bits & (bits - np.uint32(1))) != 0
    eligible = occupied & (bits != 0) & ~classes_conflict & ~ambiguous
    semantic = np.full(bits.shape, 255, dtype=np.uint8)
    semantic[np.asarray(grid.free, dtype=bool) & ~occupied] = 17
    for class_id in range(17):
        semantic[eligible & (bits == np.uint32(1 << class_id))] = class_id
    result['semantics'] = semantic
    result['mask_label_valid'] = semantic != 255
    changed = semantic != old['semantics']
    if np.any(changed & ((old['semantics'] != 255) | (semantic >= 17))):
        raise AssertionError('Resolver may only recover ignored occupied endpoints')
    indices = np.argwhere(changed).astype(np.int32)
    audit = {
        'indices': indices,
        'old_semantics': old['semantics'][changed].copy(),
        'new_semantics': semantic[changed].copy(),
        'reason_code': np.ones(int(changed.sum()), dtype=np.uint8),
    }
    return result, audit
