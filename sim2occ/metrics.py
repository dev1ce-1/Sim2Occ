import numpy as np

def metrics(cm):
 tp=np.diag(cm);gt=cm.sum(1);pred=cm.sum(0);den=gt+pred-tp
 iou=np.divide(tp,den,out=np.full(18,np.nan),where=den>0)*100
 ids=[4,7,2];core=[float(iou[i]) if gt[i]>0 and np.isfinite(iou[i]) else None for i in ids]
 return {'miou17_percent':float(np.nanmean(iou[:17])),'class_iou_percent':[float(x) if np.isfinite(x) else None for x in iou],'gt_voxels':gt.tolist(),'pred_voxels':pred.tolist(),'valid_voxels':int(cm.sum()),'core3_order':['car','pedestrian','bicycle'],'core3_iou_percent':core,'core3_miou_percent':float(np.mean(core)) if all(x is not None for x in core) else None}

def confusion(truth,pred,mask):return np.bincount(truth[mask].astype(np.int64)*18+pred[mask].astype(np.int64),minlength=324).reshape(18,18)
