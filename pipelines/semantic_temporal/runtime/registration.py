"""Current-image alignment and real shared-grid preparation."""
from dataclasses import replace
import math
import cv2
import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import reproject, transform, transform_bounds, calculate_default_transform
from rasterio.enums import Resampling
from shapely.geometry import Polygon
from .contracts import GridSpec
from .inputs import read_tile
from .storage import atomic_json, read_json
from src.stage5.registration import fit_registration


def _corners(tile):
    h,w=tile.shape_hw; affine=Affine(*tile.grid.affine)
    return [affine*p for p in ((0,0),(w,0),(w,h),(0,h))]


def _world_points(tile, points, context, *, registered=True):
    points=np.asarray(points,float)
    affine=Affine(*tile.grid.affine)
    xy=np.array([affine*tuple(p) for p in points])
    target=context.options.get('target_crs')
    if tile.grid.crs and target and tile.grid.crs!=target:
        x,y=transform(tile.grid.crs,target,xy[:,0],xy[:,1]); xy=np.column_stack([x,y])
    elif not tile.grid.crs and tile.grid.physical_pixel_size:
        xy*=np.asarray(tile.grid.physical_pixel_size)
    registration=context.options.get('registration')
    if registered and registration and tile.year==context.years[0].year:
        a=np.array(registration['affine']).reshape(2,3)
        xy=xy@a[:,:2].T+a[:,2]
    return xy


def coverage_pairs(context):
    if len(context.years)!=2: return []
    footprints={}
    for year in context.years:
        for tile in year.tiles:
            h,w=tile.shape_hw
            footprints[tile.key]=Polygon(_world_points(tile,[(0,0),(w,0),(w,h),(0,h)],context))
    return [(a.key,b.key) for a in context.years[0].tiles for b in context.years[1].tiles
            if footprints[a.key].intersection(footprints[b.key]).area>0]


def registration_pairs(context,limit=8):
    """Prefer substantial shared coverage, distributing the budget across source tiles."""
    footprints={tile.key:Polygon(_world_points(tile,[(0,0),(tile.shape_hw[1],0),
        (tile.shape_hw[1],tile.shape_hw[0]),(0,tile.shape_hw[0])],context))
        for year in context.years for tile in year.tiles}
    ranked=sorted(coverage_pairs(context),key=lambda pair:(
        -footprints[pair[0]].intersection(footprints[pair[1]]).area,
        pair[0],pair[1]))
    first=[];remaining=[];seen=set()
    for pair in ranked:
        if pair[0] not in seen:first.append(pair);seen.add(pair[0])
        else:remaining.append(pair)
    return (first+remaining)[:limit]


def _features(tile, context):
    rgb,mask=read_tile(tile); h,w=mask.shape
    factor=min(1.0,2048/max(h,w)); H=max(1,round(h*factor)); W=max(1,round(w*factor))
    gray=cv2.cvtColor(cv2.resize(rgb,(W,H)),cv2.COLOR_RGB2GRAY)
    valid=cv2.resize(mask.astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)*255
    kp,desc=cv2.SIFT_create(nfeatures=6000).detectAndCompute(cv2.createCLAHE(2.0,(8,8)).apply(gray),valid)
    if desc is None or len(kp)<6: raise ValueError('Image registration has insufficient texture; supply control points')
    points=[((p.pt[0]+.5)*w/W,(p.pt[1]+.5)*h/H) for p in kp]
    return _world_points(tile,points,context,registered=False),desc


def _matched_controls(moving,reference,context,details=None):
    if not moving.grid.crs and not reference.grid.crs:
        from .pixel_registration import pixel_controls
        ma,mv=read_tile(moving);ra,rv=read_tile(reference)
        mp,rp,diagnostics=pixel_controls(ma,ra,mv,rv)
        if details is not None:details.append({'moving_tile':moving.key,'reference_tile':reference.key,**diagnostics})
        return _world_points(moving,mp,context,registered=False),_world_points(reference,rp,context,registered=False)
    if moving.grid.crs and reference.grid.crs:
        from src.stage5.registration import staged_tile_control_points
        try:
            mp,rp,diagnostics=staged_tile_control_points(moving.image,moving.grid.affine,moving.grid.crs,
                reference.image,reference.grid.affine,reference.grid.crs,context.options.get('target_crs') or reference.grid.crs)
            if details is not None:details.append({'moving_tile':moving.key,'reference_tile':reference.key,**diagnostics})
            return mp,rp
        except ValueError:
            # Approximate maps with little common coverage can still have distributed descriptor matches.
            pass
    mp,md=_features(moving,context); rp,rd=_features(reference,context)
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    def ratio(a,b):
        return {p[0].queryIdx:p[0].trainIdx for p in matcher.knnMatch(a,b,k=2)
                if len(p)==2 and p[0].distance<.75*p[1].distance}
    forward=ratio(md,rd); backward=ratio(rd,md)
    pairs=[(a,b) for a,b in forward.items() if backward.get(b)==a]
    if len(pairs)<6: raise ValueError('Insufficient mutual registration landmarks; supply control points')
    if details is not None:details.append({'moving_tile':moving.key,'reference_tile':reference.key,'method':'mutual_sift'})
    return mp[[a for a,b in pairs]],rp[[b for a,b in pairs]]


def prepare_comparison(context):
    options=dict(context.options)
    tiles=[tile for year in context.years for tile in year.tiles]
    has_map=[bool(t.grid.crs) for t in tiles]
    if any(has_map) and not all(has_map): raise ValueError('Cannot compare mapped imagery with an unrelated pixel coordinate frame')
    if all(has_map):
        reference=context.years[-1].tiles[0]
        target=options.get('target_crs')
        if not target:
            crs=CRS.from_user_input(reference.grid.crs)
            if crs.is_projected and math.isclose(crs.linear_units_factor[1],1): target=crs.to_string()
            else:
                centre=Affine(*reference.grid.affine)*(reference.shape_hw[1]/2,reference.shape_hw[0]/2)
                lon,lat=transform(reference.grid.crs,'EPSG:4326',[centre[0]],[centre[1]])
                zone=min(60,max(1,int((lon[0]+180)//6)+1))
                target=f'EPSG:{(32600 if lat[0]>=0 else 32700)+zone}'
        crs=CRS.from_user_input(target)
        if not crs.is_projected or not math.isclose(crs.linear_units_factor[1],1): raise ValueError('Target CRS must use projected metres')
        options['target_crs']=crs.to_string(); options['coordinate_units']='metres'
    else:
        physical=[bool(t.grid.physical_pixel_size) for t in tiles]
        if any(physical) and not all(physical): raise ValueError('Supply pixel size for each original image frame')
        options['coordinate_units']='metres' if all(physical) else 'pixels'
        if len(context.years)==2 and not options.get('registration_json'):
            options['register_images']=True
        if not all(physical):
            options['measurement_policy']='image_relative_pixels; area thresholds use pixel squared, distances use pixels'
    reference=context.years[-1].tiles[0]
    ref_affine=Affine(*reference.grid.affine)
    if reference.grid.crs and options.get('target_crs')!=reference.grid.crs:
        corners=np.array(_corners(reference)); bounds=(*corners.min(0),*corners.max(0))
        ref_affine,_,_=calculate_default_transform(reference.grid.crs,options['target_crs'],
                                                   reference.shape_hw[1],reference.shape_hw[0],*bounds)
    elif not reference.grid.crs and reference.grid.physical_pixel_size:
        ref_affine=Affine.scale(*reference.grid.physical_pixel_size)*ref_affine
    resolution=float(options.get('target_resolution') or math.sqrt(abs(ref_affine.determinant)))
    if not math.isfinite(resolution) or resolution<=0: raise ValueError('Target resolution must be positive')
    options['shared_resolution']=resolution
    options['shared_anchor']=[ref_affine.c,ref_affine.f]
    prepared=replace(context,options=options)
    if len(context.years)==2:
        fit=fit_registration
        if options['coordinate_units']=='pixels':
            from .pixel_registration import fit_pixel_registration
            fit=fit_pixel_registration
        registration=None
        moving=[]; reference=[]; control_keys=[];feature_diagnostics=[]
        if options.get('registration_json'):
            controls=read_json(options['registration_json'])
            moving=controls['moving_xy']; reference=controls['reference_xy']
            registration=fit(controls['moving_xy'],controls['reference_xy'],
                                          max_residual_m=float(controls.get('max_residual_m',2.0)))
        elif options.get('register_images'):
            pairs=registration_pairs(prepared)
            if not pairs: raise ValueError('Yearly image coverage does not overlap; registration cannot be estimated')
            by_key={t.key:t for t in tiles}; moving=[];reference=[]
            for a,b in pairs:
                try: m,r=_matched_controls(by_key[a],by_key[b],prepared,feature_diagnostics)
                except ValueError: continue
                try:fit(m,r,max_residual_m=options.get('registration_max_residual',2.0))
                except ValueError:
                    feature_diagnostics[-1]['fit_status']='rejected_insufficient_local_support'
                    continue
                moving.extend(m);reference.extend(r)
                control_keys.extend([(a,b)]*len(m))
            if len(moving)<6: raise ValueError('Registration lacks distributed landmarks; supply --registration-json')
            m=np.array(moving);r=np.array(reference)
            _,unique=np.unique(np.round(m,6),axis=0,return_index=True)
            unique=np.sort(unique)
            moving=m[unique];reference=r[unique]
            control_keys=[control_keys[i] for i in unique]
            tolerance=options.get('registration_max_residual',2.0)
            registration=fit(moving,reference,max_residual_m=tolerance)
            registration['max_residual']=float(tolerance)
        if registration:
            if options['coordinate_units']=='pixels':
                registration={(k[:-3]+'_pixel2' if k.endswith('_m2') else k[:-2]+'_pixels' if k.endswith('_m') else k):v for k,v in registration.items()}
            registration['units']=options['coordinate_units']; options['registration']=registration
            matrix=np.asarray(registration['affine']).reshape(2,3)
            residuals=np.linalg.norm(np.asarray(moving)@matrix[:,:2].T+matrix[:,2]-reference,axis=1)
            def native_location(point,year,key=None):
                candidates=[t for t in tiles if t.year==year and (key is None or t.key==key)]
                for tile in candidates:
                    xy=np.asarray(point,float)
                    if tile.grid.crs and options.get('target_crs')!=tile.grid.crs:
                        x,y=transform(options['target_crs'],tile.grid.crs,[xy[0]],[xy[1]])
                        xy=np.array([x[0],y[0]])
                    elif not tile.grid.crs and tile.grid.physical_pixel_size:
                        xy=xy/np.asarray(tile.grid.physical_pixel_size)
                    pixel=(~Affine(*tile.grid.affine))*tuple(xy)
                    h,w=tile.shape_hw
                    if key is not None or (0<=pixel[0]<=w and 0<=pixel[1]<=h):
                        return tile.key,list(pixel)
                return None,None
            pairs=[]
            for index,(m,r,residual,inlier) in enumerate(zip(moving,reference,residuals,registration['inlier_mask'])):
                keys=control_keys[index] if control_keys else (None,None)
                mk,mp=native_location(m,context.years[0].year,keys[0])
                rk,rp=native_location(r,context.years[1].year,keys[1])
                pairs.append({'id':f'control-{index+1:06d}','moving_tile':mk,'reference_tile':rk,
                              'moving_native_xy':mp,'reference_native_xy':rp,
                              'moving_xy':np.asarray(m).tolist(),'reference_xy':np.asarray(r).tolist(),
                              'inlier':bool(inlier),'residual':float(residual)})
            atomic_json(context.output/'prepared/registration_controls.json',{
                'method':'supplied_controls' if options.get('registration_json') else 'current_image_features_affine_ransac',
                'feature_diagnostics':feature_diagnostics,
                'units':options['coordinate_units'],'years':[y.year for y in context.years],
                'pairs':pairs,'diagnostics':registration,
                'accuracy_status':'in_sample_only_requires_independent_validation'})
        elif not all(has_map):
            # Distinct source pixels require explicit frame alignment, not a guessed identity.
            raise ValueError('Original-pixel comparison needs --register-images or --registration-json')
        else:
            atomic_json(context.output/'prepared/registration_controls.json',{
                'method':'georeference_only','units':options['coordinate_units'],
                'years':[y.year for y in context.years],'pairs':[],
                'status':'Feature registration was not performed'})
    prepared=replace(context,options=options)
    atomic_json(context.output/'context.json',prepared)
    from .context import seal_prepared_context
    seal_prepared_context(prepared)
    return prepared


def prepare_tile(tile,context,*,model=True):
    entry=context.options.get('prepared_inputs',{}).get(tile.key)
    if entry:
        import rasterio
        with rasterio.open(entry['model_rgb' if model else 'original_rgb']) as src:
            rgb=src.read().transpose(1,2,0)
        with rasterio.open(entry['valid']) as src: valid=src.read(1)>0
        value=entry['grid']
        grid=value if isinstance(value,GridSpec) else GridSpec(value['crs'],tuple(value['affine']),value['units'],tuple(value['physical_pixel_size']) if value.get('physical_pixel_size') else None)
        return rgb,valid,grid
    rgb,valid=read_tile(tile); h,w=valid.shape
    affine=Affine(*tile.grid.affine); crs=tile.grid.crs; target=context.options.get('target_crs')
    units=context.options.get('coordinate_units',tile.grid.units)
    if crs and target and crs!=target:
        corners=np.array(_corners(tile)); bounds=(*corners.min(0),*corners.max(0))
        destination,W,H=calculate_default_transform(crs,target,w,h,*bounds,
                          resolution=context.options.get('target_resolution'))
        warped=np.zeros((H,W,3),np.uint8); observed=np.zeros((H,W),np.uint8)
        for band in range(3):
            reproject(rgb[:,:,band],warped[:,:,band],src_transform=affine,src_crs=crs,
                      dst_transform=destination,dst_crs=target,resampling=Resampling.bilinear)
        reproject(valid.astype(np.uint8),observed,src_transform=affine,src_crs=crs,
                  dst_transform=destination,dst_crs=target,resampling=Resampling.nearest)
        rgb,valid,affine,crs=warped,observed>0,destination,target
    elif not crs and tile.grid.physical_pixel_size:
        sx,sy=tile.grid.physical_pixel_size; affine=Affine.scale(sx,sy)*affine
    registration=context.options.get('registration')
    if registration and tile.year==context.years[0].year:
        affine=Affine(*registration['affine'])*affine
    if not math.isfinite(affine.determinant) or abs(affine.determinant)<1e-15: raise ValueError('Singular image affine')
    resolution=context.options.get('shared_resolution')
    if resolution:
        ax,ay=context.options['shared_anchor']; H,W=valid.shape
        corners=np.array([affine*p for p in ((0,0),(W,0),(W,H),(0,H))])
        lo=corners.min(0); hi=corners.max(0); sy=-1 if crs else 1
        x=ax+math.floor((lo[0]-ax)/resolution+1e-7)*resolution
        y=(ay+math.ceil((hi[1]-ay)/resolution-1e-7)*resolution) if sy<0 else (ay+math.floor((lo[1]-ay)/resolution+1e-7)*resolution)
        dest=Affine(resolution,0,x,0,sy*resolution,y)
        width=max(1,math.ceil((hi[0]-x)/resolution-1e-7))
        height=max(1,math.ceil(((y-lo[1]) if sy<0 else (hi[1]-y))/resolution-1e-7))
        if (width,height)!=(W,H) or not np.allclose(tuple(dest)[:6],tuple(affine)[:6],rtol=0,atol=1e-8):
            warped=np.zeros((height,width,3),np.uint8); observed=np.zeros((height,width),np.uint8)
            if crs:
                for band in range(3):
                    reproject(rgb[:,:,band],warped[:,:,band],src_transform=affine,src_crs=crs,
                              dst_transform=dest,dst_crs=crs,resampling=Resampling.bilinear)
                reproject(valid.astype(np.uint8),observed,src_transform=affine,src_crs=crs,
                          dst_transform=dest,dst_crs=crs,resampling=Resampling.nearest)
            else:
                matrix=np.array(tuple(Affine.translation(-.5,-.5)*(~dest)*affine*Affine.translation(.5,.5))[:6]).reshape(2,3)
                warped=cv2.warpAffine(rgb,matrix,(width,height),flags=cv2.INTER_LINEAR)
                observed=cv2.warpAffine(valid.astype(np.uint8),matrix,(width,height),flags=cv2.INTER_NEAREST)
            rgb,valid=warped,observed>0
        affine=dest
    physical=(math.hypot(affine.a,affine.d),math.hypot(affine.b,affine.e)) if units=='metres' else None
    rgb[~valid]=0
    return rgb,valid,GridSpec(crs,tuple(affine)[:6],units,physical)
