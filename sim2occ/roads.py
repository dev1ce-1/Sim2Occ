"""Match a reference BEV road and sidewalk raster to a library of CARLA anchors."""
import numpy as np


def rank_anchors(grid, reference, rasters, anchors, limit=48):
    """Return diverse candidates using the road and sidewalk Dice scores.

    grid contains ego-frame XY cell centers in metres, with shape [N, 2].
    reference has shape [N, 2] and rasters [A, N, 2]. Channels are road and sidewalk.
    Each anchor supplies map, xyz in metres, and yaw in degrees.
    """
    grid = np.asarray(grid)
    reference = np.asarray(reference, dtype=np.float32)
    rasters = np.asarray(rasters, dtype=np.float32)
    if (grid.ndim != 2 or grid.shape[1] != 2 or reference.shape != grid.shape
            or rasters.shape != (len(anchors), *grid.shape)):
        raise ValueError("Grid, two-channel rasters and anchor count must agree")
    if limit < 1:
        raise ValueError("The candidate limit must be positive")
    weights = (.5 + 1.5 * np.exp(-np.sum(grid**2, axis=1) / (2 * 20**2))).astype(np.float32)
    weighted = rasters * weights[None, :, None]
    intersections = np.stack([reference[:, c] @ weighted[:, :, c].T for c in range(2)], axis=-1)
    denominator = (reference * weights[:, None]).sum(0) + weighted.sum(1)
    dice = 2 * intersections / np.maximum(denominator, 1)
    scores = .7 * dice[:, 0] + .3 * dice[:, 1]
    selected, per_map = [], {}
    for i in np.argsort(-scores):
        anchor = anchors[i]
        town = anchor["map"]
        if per_map.get(town, 0) >= 8:
            continue
        if any(old["map"] == town
               and np.linalg.norm(np.asarray(anchor["xyz"][:2]) - old["xyz"][:2]) < 18
               and abs((anchor["yaw"] - old["yaw"] + 180) % 360 - 180) < 20
               for old in selected):
            continue
        selected.append(dict(anchor, anchor_index=int(i), score=float(scores[i]),
                             road_dice=float(dice[i, 0]), sidewalk_dice=float(dice[i, 1])))
        per_map[town] = per_map.get(town, 0) + 1
        if len(selected) == limit:
            break
    return selected
