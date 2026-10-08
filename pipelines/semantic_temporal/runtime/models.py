"""Lazy, local-only segmentation models; no downloads during inference."""
from pathlib import Path
import os
import cv2
import numpy as np
import torch


def window_starts(length, window, stride):
    if length <= window: return [0]
    starts=list(range(0,length-window+1,stride))
    if starts[-1]!=length-window: starts.append(length-window)
    return starts


class ModelBundle:
    def __init__(self, directory, *, device='cpu', window=512, overlap=128, tta=True):
        self.directory=Path(directory)
        if device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable in this PyTorch environment; use --device cpu')
        if window<32 or not 0<=overlap<window: raise ValueError('Inference window >=32 and 0<=overlap<window required')
        self.device=device;self.window=int(window);self.overlap=int(overlap);self.tta=tta
        self._semantic=None;self._parcel=None

    def semantic_model(self):
        if self._semantic is None:
            from transformers import AutoModelForSemanticSegmentation
            self._semantic=AutoModelForSemanticSegmentation.from_pretrained(
                str(self.directory/'satlens-segformer'),local_files_only=True).to(self.device).eval().float()
            if self._semantic.config.num_labels!=9: raise ValueError('Expected nine OpenEarthMap model channels')
        return self._semantic

    @torch.inference_mode()
    def probabilities(self,image,scale):
        import torch.nn.functional as F
        model=self.semantic_model();h,w=image.shape[:2];H,W=round(h*scale),round(w*scale)
        rgb=cv2.resize(image,(W,H),interpolation=cv2.INTER_CUBIC)
        acc=np.zeros((9,H,W),np.float32);weight_sum=np.zeros((H,W),np.float32)
        ramp=np.maximum(np.hanning(self.window),.05).astype(np.float32);weight=ramp[:,None]*ramp[None,:]
        mean=torch.tensor([.485,.456,.406],device=self.device)[None,:,None,None]
        std=torch.tensor([.229,.224,.225],device=self.device)[None,:,None,None]
        for y in window_starts(H,self.window,self.window-self.overlap):
            for x in window_starts(W,self.window,self.window-self.overlap):
                crop=rgb[y:y+self.window,x:x+self.window];ch,cw=crop.shape[:2]
                tensor=torch.from_numpy(crop.copy()).permute(2,0,1)[None].float().to(self.device)/255
                tensor=(tensor-mean)/std
                # SegFormer patch embedding needs sufficient height and width on tiny crops.
                tensor=F.pad(tensor,(0,max(0,32-cw),0,max(0,32-ch)),mode='replicate')
                prob=0;flips=[None,(-1,),(-2,),(-1,-2)] if self.tta else [None]
                for flip in flips:
                    inp=tensor if flip is None else torch.flip(tensor,flip)
                    logits=model(pixel_values=inp).logits
                    logits=F.interpolate(logits,size=tensor.shape[-2:],mode='bilinear',align_corners=False)
                    if flip is not None: logits=torch.flip(logits,flip)
                    prob=prob+logits.softmax(1)
                values=(prob[0,:,:ch,:cw]/len(flips)).cpu().numpy()
                wt=weight[:ch,:cw];acc[:,y:y+ch,x:x+cw]+=values*wt;weight_sum[y:y+ch,x:x+cw]+=wt
        return acc/weight_sum[None]

    def parcel_model(self):
        if self._parcel is None:
            cache=self.directory/'.ultralytics';cache.mkdir(parents=True,exist_ok=True)
            os.environ['YOLO_CONFIG_DIR']=str(cache)
            from ultralytics import YOLO
            self._parcel=YOLO(str(self.directory/'DelineateAnythingv2.pt'))
        return self._parcel

    def _instances(self,image,shape,conf):
        result=self.parcel_model().predict(image[:,:,::-1].copy(),device=self.device,conf=conf,
                                          retina_masks=True,verbose=False)[0]
        if result.masks is None: return [],[]
        masks=[cv2.resize(m,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>.5
               for m in result.masks.data.cpu().numpy()]
        return masks,result.boxes.conf.cpu().numpy()

    def parcels(self,image,shape,conf):
        from .segmentation.parcels import bbox_truncated
        h,w=image.shape[:2];H,W=shape;sy,sx=H/h,W/w;all_masks=[];scores=[];windows=[];win=0
        window=self.window;stride=max(1,window-min(50,self.overlap))
        for y in window_starts(h,window,stride):
            for x in window_starts(w,window,stride):
                crop=image[y:y+window,x:x+window];ch,cw=crop.shape[:2]
                y0,y1=round(y*sy),round((y+ch)*sy);x0,x1=round(x*sx),round((x+cw)*sx)
                outshape=(y1-y0,x1-x0);up=cv2.resize(crop,(outshape[1],outshape[0]),interpolation=cv2.INTER_CUBIC)
                exempt=(y==0,y+ch==h,x==0,x+cw==w);win+=1
                for sample in (up,crop):
                    masks,confs=self._instances(sample,outshape,conf)
                    for mask,score in zip(masks,confs):
                        if not mask.any() or bbox_truncated(mask,exempt=exempt): continue
                        full=np.zeros(shape,bool);full[y0:y1,x0:x1]=mask
                        all_masks.append(full);scores.append(float(score));windows.append(win)
        keep=[]
        for i in np.argsort(scores,kind='stable')[::-1]:
            if any(windows[i]!=windows[j] and np.logical_and(all_masks[i],all_masks[j]).sum()/max(1,np.logical_or(all_masks[i],all_masks[j]).sum())>.5 for j in keep): continue
            keep.append(i)
        keep.sort()
        return (np.stack([all_masks[i] for i in keep]),np.array([scores[i] for i in keep])) if keep else (np.zeros((0,H,W),bool),np.zeros(0))


def load_models(context):
    root=Path(context.options.get('project_root') or Path(__file__).resolve().parents[3])
    directory=Path(context.options.get('semantic_models') or root/'checkpoints/semantic_temporal')
    return ModelBundle(directory,device=context.options.get('device') or 'cpu',
                       window=context.options.get('inference_window') or 512,
                       overlap=context.options.get('inference_overlap',128),tta=context.options.get('tta',True))
