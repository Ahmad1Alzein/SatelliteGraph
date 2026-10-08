import cv2
import numpy as np

OEM_TO_CLASS={1:6,2:4,3:7,4:2,5:3,6:8,7:5,8:1}


def semantics(image,models):
    p2=models.probabilities(image,2)
    average=p2.copy()
    maximum=p2.copy()
    for scale in (1.,1.5):
        values=models.probabilities(image,scale)
        resized=np.stack([cv2.resize(v,p2.shape[:0:-1],interpolation=cv2.INTER_LINEAR) for v in values])
        average+=resized
        maximum=np.maximum(maximum,resized)
    average/=3
    canonical=np.zeros_like(average)
    canonical[0]=average[0]
    for source,target in OEM_TO_CLASS.items(): canonical[target]=average[source]
    shape=np.zeros_like(p2);maxima=np.zeros_like(p2)
    shape[0]=p2[0];maxima[0]=maximum[0]
    for source,target in OEM_TO_CLASS.items(): shape[target]=p2[source];maxima[target]=maximum[source]
    return p2,canonical,shape,maxima
