"""Image-relative landmarks without invented map coordinates or ground resolution."""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint


def fit_pixel_registration(moving_xy,reference_xy,*,max_residual_m=2.0):
    """Image-relative affine fit; pixel scale can differ between acquisitions.

    The argument keeps the existing call interface; its value is in pixels.
    Metric world-coordinate registration retains its separate strict scale gate.
    """
    a=np.asarray(moving_xy,float);b=np.asarray(reference_xy,float)
    if a.ndim!=2 or a.shape!=b.shape or a.shape[1:]!=(2,) or len(a)<6 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Pixel registration requires six finite paired 2D controls')
    if not np.isfinite(max_residual_m) or max_residual_m<=0:raise ValueError('Pixel residual tolerance must be positive')
    for points in (a,b):
        if len(np.unique(points,axis=0))<6 or np.linalg.matrix_rank(points-points.mean(0))<2:
            raise ValueError('Pixel controls must be distinct and distributed in two dimensions')
    ao=a.mean(0);bo=b.mean(0);cv2.setRNGSeed(42)
    matrix,mask=cv2.estimateAffine2D(a-ao,b-bo,method=cv2.RANSAC,
        ransacReprojThreshold=float(max_residual_m),maxIters=5000,confidence=.999,refineIters=20)
    if matrix is None or mask is None:raise ValueError('Unable to fit pixel alignment')
    keep=mask.ravel().astype(bool);singular=np.linalg.svd(matrix[:,:2],compute_uv=False)
    if keep.sum()<6 or keep.mean()<.5 or singular.min()<.25 or singular.max()>4 or np.linalg.det(matrix[:,:2])<=0:
        raise ValueError('Pixel alignment has insufficient support or implausible scale/reflection')
    if any(np.linalg.matrix_rank(p[keep]-p[keep].mean(0))<2 for p in (a,b)):
        raise ValueError('Pixel inliers must be distributed in two dimensions')
    matrix[:,2]+=bo-matrix[:,:2]@ao
    residual=np.linalg.norm(a@matrix[:,:2].T+matrix[:,2]-b,axis=1);rmse=float(np.sqrt(np.mean(residual[keep]**2)))
    if rmse>max_residual_m:raise ValueError('Pixel residual exceeds tolerance')
    return dict(method='pixel_affine_ransac',affine=matrix.ravel().tolist(),point_count=len(a),
        inlier_count=int(keep.sum()),inlier_ratio=float(keep.mean()),inlier_rmse_pixels=rmse,
        inlier_p95_pixels=float(np.percentile(residual[keep],95)),inlier_mask=keep.tolist(),
        inlier_hull_area_pixel2=float(MultiPoint(a[keep]).convex_hull.area),
        accuracy_status='in_sample_only_requires_independent_validation')


def pixel_controls(moving,reference,moving_valid,reference_valid):
    h=max(moving.shape[0],reference.shape[0]);w=max(moving.shape[1],reference.shape[1])
    factor=min(1.,2048/max(h,w));size=(round(w*factor),round(h*factor))
    def prepare(rgb,valid):
        canvas=np.zeros((h,w,3),np.uint8);mask=np.zeros((h,w),np.uint8)
        canvas[:rgb.shape[0],:rgb.shape[1]]=rgb;mask[:rgb.shape[0],:rgb.shape[1]]=valid
        gray=cv2.cvtColor(cv2.resize(canvas,size),cv2.COLOR_RGB2GRAY)
        gray=cv2.createCLAHE(2.,(8,8)).apply(gray)
        return gray,cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST)
    mg,mv=prepare(moving,moving_valid);rg,rv=prepare(reference,reference_valid)
    if np.array_equal(moving,reference) and np.array_equal(moving_valid,reference_valid):
        ys,xs=np.where(moving_valid)
        if len(xs)<16:raise ValueError('Insufficient valid image coverage')
        points=np.array([(x,y) for y in np.linspace(ys.min(),ys.max(),3) for x in np.linspace(xs.min(),xs.max(),3)])+.5
        return points,points.copy(),{'method':'identical_original_pixels'}
    if min(size)<32:raise ValueError('Image registration needs texture over at least 32 pixels')
    common=mv.astype(bool)&rv.astype(bool)
    shift,response=cv2.phaseCorrelate(mg.astype(np.float32)*common,rg.astype(np.float32)*common,
        cv2.createHanningWindow(size,cv2.CV_32F))
    detector=cv2.SIFT_create(nfeatures=12000,contrastThreshold=.008)
    mk,md=detector.detectAndCompute(mg,cv2.erode(mv,np.ones((7,7),np.uint8))*255)
    rk,rd=detector.detectAndCompute(rg,cv2.erode(rv,np.ones((7,7),np.uint8))*255)
    if md is None or rd is None:raise ValueError('Image registration has insufficient texture')
    mp=np.array([k.pt for k in mk],np.float32);rp=np.array([k.pt for k in rk],np.float32)
    candidates=[]
    if np.isfinite(shift).all() and response>=.015 and max(abs(shift[0])/size[0],abs(shift[1])/size[1])<=.3:
        tree=cKDTree(rp);radius=max(8.,min(14.,.025*max(size)))
        for i,p in enumerate(mp):
            ids=tree.query_ball_point(p+shift,radius)
            if len(ids)<2:continue
            delta=rd[ids]-md[i];dist=np.einsum('ij,ij->i',delta,delta)
            a,b=np.argsort(dist)[:2]
            if dist[a]<.85**2*dist[b]:candidates.append((float(dist[a]),i,ids[a]))
    # Compare both fits. A plausible translation can be spurious when the real
    # pair has a different resolution; candidate count alone cannot select it.
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    def matches(a,b):
        return {p[0].queryIdx:(p[0].trainIdx,p[0].distance) for p in matcher.knnMatch(a,b,k=2)
            if len(p)==2 and p[0].distance<.75*p[1].distance}
    forward=matches(md,rd);backward=matches(rd,md)
    descriptor=[(distance,i,j) for i,(j,distance) in forward.items() if backward.get(j,(-1,))[0]==i]
    fits=[]
    for method,choices in [('phase_guided',candidates),('mutual_descriptor',descriptor)]:
        pairs=[];used=set()
        for _,i,j in sorted(choices):
            if j not in used:pairs.append((i,j));used.add(j)
        if len(pairs)<12:continue
        a=mp[[i for i,j in pairs]];b=rp[[j for i,j in pairs]];cv2.setRNGSeed(42)
        matrix,mask=cv2.estimateAffine2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=5000,confidence=.999)
        if matrix is None or mask is None:continue
        keep=mask.ravel().astype(bool)
        hull=MultiPoint(a[keep]).convex_hull.area/max(1.,float(np.prod(np.ptp(a,axis=0))))
        singular=np.linalg.svd(matrix[:,:2],compute_uv=False)
        if keep.sum()<12 or keep.mean()<.08 or hull<.12 or singular.min()<.25 or singular.max()>4 or np.linalg.det(matrix[:,:2])<=0:continue
        fits.append((int(keep.sum()),hull,method,a,b,keep,len(pairs)))
    if not fits:raise ValueError('Image alignment landmarks are too weak, too local or geometrically implausible')
    _,hull,method,a,b,keep,pair_count=max(fits,key=lambda v:(v[0],v[1]))
    return (a[keep]+.5)/factor,(b[keep]+.5)/factor,{
        'method':'pixel_'+method+'_sift_ransac','phase_response':float(response),
        'phase_shift_pixels':(np.asarray(shift)/factor).tolist(),'inlier_count':int(keep.sum()),
        'candidate_count':pair_count,'inlier_hull_fraction':hull,
        'accuracy_status':'in_sample_landmarks_not_independent_ground_truth'}
