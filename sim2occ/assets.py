"""Rank same-class CARLA assets against the reference camera projections."""
import math
import numpy as np

from .projection import compare

SETTINGS={'center_3d_tolerance_m':.02,'center_px_floor':3.,'center_px_diagonal_fraction':.01,'center_px_cap':8.,'footprint_iou_min':.50,'footprint_bbox_center_floor_px':6.,'footprint_bbox_center_diagonal_fraction':.08,'image_edge_sliver_max_px':32.,'strict_reference_iou':.70,'source_resolution':[1600,900]}

def diag(bb):
 return float(math.hypot(bb[2]-bb[0],bb[3]-bb[1])) if bb else 0.


def edge_sliver(bb):
 if not bb:return False
 x0,y0,x1,y1=bb
 return (x0<=.01 or y0<=.01 or x1>=1599.99 or y1>=899.99) and min(x1-x0,y1-y0)<=SETTINGS['image_edge_sliver_max_px']


def view_review(v):
 a,b=v['source'],v['rendered'];active=a['in_image'] or b['in_image'];dd=diag(a.get('bbox_xyxy'))
 tol=min(SETTINGS['center_px_cap'],max(SETTINGS['center_px_floor'],SETTINGS['center_px_diagonal_fraction']*dd))
 ce=v.get('center_error_source_px');warnings=[];issues=[]
 center_ok=not active or (ce is not None and math.isfinite(ce) and ce<=tol)
 if not center_ok:issues.append('IMAGE_PROJECTED_CENTER_DRIFT_OR_UNVERIFIED')
 if not active:shape=True
 elif a['in_image']!=b['in_image']:
  # A shape-dependent sliver at an image boundary is not projected-center drift.
  sliver=edge_sliver(a.get('bbox_xyxy') or b.get('bbox_xyxy'));shape=bool(sliver and center_ok)
  (warnings if shape else issues).append('IMAGE_EDGE_SHAPE_VISIBILITY_DIFFERENCE' if shape else 'LARGE_FOOTPRINT_VISIBILITY_DIFFERENCE')
 else:
  iou=v.get('bbox_iou');bc=v.get('bbox_center_error_source_px');bt=max(SETTINGS['footprint_bbox_center_floor_px'],SETTINGS['footprint_bbox_center_diagonal_fraction']*dd)
  shape=iou is not None and math.isfinite(iou) and iou>=SETTINGS['footprint_iou_min'] and bc is not None and math.isfinite(bc) and bc<=bt
  if not shape:issues.append('LARGE_ASSET_SHAPE_OR_POSE_DIFFERENCE')
  elif iou<SETTINGS['strict_reference_iou'] or not v.get('footprint_pass',True):warnings.append('ACCEPTED_APPROXIMATE_ASSET_SHAPE_OR_POSE')
 if center_ok and active and ce is not None and ce>1.:warnings.append('ACCEPTED_SMALL_PROJECTED_CENTER_ERROR')
 return {'position_pass':bool(center_ok),'footprint_pass':bool(shape),'center_tolerance_source_px':tol,'issues':issues,'warnings':warnings}


def assess_asset(obj, asset, cameras):
    model = dict(obj, half_size_lwh=asset['box']['extent'])
    views = {}
    old_error = 0.0
    source_views = 0
    for name, cam in cameras.items():
        projection = compare(obj, model, cam)
        review = view_review(projection)
        views[name] = {
            'source_in_image': projection['source']['in_image'],
            'rendered_in_image': projection['rendered']['in_image'],
            'bbox_iou': projection['bbox_iou'],
            'bbox_center_error_source_px': projection['bbox_center_error_source_px'],
            'position_pass': review['position_pass'],
            'footprint_pass': review['footprint_pass'],
            'issues': review['issues'], 'warnings': review['warnings'],
        }
        if projection['source']['in_image']:
            old_error += (1 - (projection['bbox_iou'] or 0)) * 10 + (projection['bbox_center_error_source_px'] or 0) / 10
            source_views += 1
    size = float(np.sum(np.abs(np.log(np.maximum(asset['box']['extent'], .01) / np.maximum(obj['half_size_lwh'], .01)))))
    failures = sum(not (v['position_pass'] and v['footprint_pass']) for v in views.values())
    score = old_error / max(source_views, 1) + size * .1
    return {
        'type_id': asset['type_id'], 'kind': asset['kind'],
        'failed_view_count': failures, 'all_views_pass': failures == 0,
        'legacy_average_score': score,
        'asset_half_size_lwh': list(asset['box']['extent']),
        'views': views,
    }


def ranking_key(review):
    return (review['failed_view_count'], review['legacy_average_score'])


def rank_assets(obj, choices, cameras):
    scored = [(a, assess_asset(obj, a, cameras)) for a in choices]
    scored.sort(key=lambda item: ranking_key(item[1]))
    return scored
