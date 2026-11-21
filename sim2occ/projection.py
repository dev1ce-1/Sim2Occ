"""Pixel geometry contract. All errors expressed at original 1600x900 resolution."""
import math,itertools
import numpy as np
H=np.diag([1.,-1.,1.])
C=np.array([[0.,0.,1.],[1.,0.,0.],[0.,-1.,0.]])
CAMERAS=['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']
SIGNS=np.array(list(itertools.product([-1.,1.],repeat=3)))
EDGES=[(i,j) for i in range(8) for j in range(i+1,8) if np.sum(SIGNS[i]!=SIGNS[j])==1]
def quat(q):
 q=np.asarray(q,dtype=float);q=q/np.linalg.norm(q);w,x,y,z=q
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def pose(translation,rotation):
 t=np.eye(4);t[:3,:3]=quat(rotation);t[:3,3]=translation;return t
def inv(t):
 t=np.asarray(t);r=np.eye(4);r[:3,:3]=t[:3,:3].T;r[:3,3]=-r[:3,:3]@t[:3,3];return r
def points(t,p):return np.asarray(p)@np.asarray(t)[:3,:3].T+np.asarray(t)[:3,3]
def camera_mount(cam):
 q=np.asarray(cam['T_reference_ego_from_optical']);t=np.eye(4);t[:3,:3]=H@q[:3,:3]@C.T;t[:3,3]=H@q[:3,3];assert np.linalg.det(t[:3,:3])>.999999;return t
def corners(center,rotation,half):return SIGNS*np.asarray(half)@np.asarray(rotation).T+np.asarray(center)
def projected(center,rotation,half,cam):
 t=inv(cam['T_reference_ego_from_optical']);v=points(t,corners(center,rotation,half));p=points(t,[center])[0];K=np.asarray(cam['cam_intrinsic']);near=.05
 uv=(K@p)[:2]/p[2] if p[2]>near else None
 visible=[x for x in v if x[2]>=near]
 for i,j in EDGES:
  a,b=v[i],v[j]
  if (a[2]>=near)!=(b[2]>=near):
   visible.append(a+(b-a)*(near-a[2])/(b[2]-a[2]))
 bb=None
 if visible:
  vv=np.array(visible);xy=vv@K.T;xy=xy[:,:2]/xy[:,2,None]
  lo=np.maximum(xy.min(0),[0,0]);hi=np.minimum(xy.max(0),cam['image_size'])
  if np.all(hi>lo):bb=np.r_[lo,hi]
 return {'center_uv':None if uv is None else uv.tolist(),'center_depth_m':float(p[2]),'bbox_xyxy':None if bb is None else bb.tolist(),'in_image':bb is not None}
def compare(source,rendered,cam,render_cam=None):
 a=projected(source['center_ego_rh'],source['rotation_ego_rh'],source['half_size_lwh'],cam)
 b=projected(rendered['center_ego_rh'],rendered['rotation_ego_rh'],rendered['half_size_lwh'],render_cam or cam)
 ce=None
 if a['center_uv'] is not None and b['center_uv'] is not None:
  ce=float(np.linalg.norm(np.array(a['center_uv'])-b['center_uv']))
 iou=bc=None;tol=None
 if a['bbox_xyxy'] is not None and b['bbox_xyxy'] is not None:
  x,y=np.array(a['bbox_xyxy']),np.array(b['bbox_xyxy']);area=lambda z:float(np.prod(np.maximum(z[2:]-z[:2],0)))
  intersection=float(np.prod(np.maximum(np.minimum(x[2:],y[2:])-np.maximum(x[:2],y[:2]),0)));iou=intersection/max(area(x)+area(y)-intersection,1e-9)
  bc=float(np.linalg.norm((x[:2]+x[2:]-y[:2]-y[2:])/2));tol=max(3.,.03*float(np.linalg.norm(x[2:]-x[:2])))
 position=(not a['in_image'] and not b['in_image']) or (a['in_image']==b['in_image'] and ce is not None and ce<=1.0)
 footprint=(not a['in_image'] and not b['in_image']) or (iou is not None and iou>=.70 and bc<=tol)
 return {'source':a,'rendered':b,'center_error_source_px':ce,'bbox_center_error_source_px':bc,'bbox_center_tolerance_source_px':tol,'bbox_iou':iou,'position_pass':bool(position),'footprint_pass':bool(footprint),'pass':bool(position and footprint)}
