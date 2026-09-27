"""Windowed GeoTIFF resampling and fixed-size, mask-aware Stage 1 tiling."""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import re
import tempfile

import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, transform as transform_coordinates
from rasterio.windows import Window

from src.stage2.artifacts import atomic_json, now


def _metric_target(source, requested):
    if source.crs is None:
        raise ValueError("Metre resolution requires a georeferenced source with a CRS")
    if requested is not None:
        target = CRS.from_user_input(requested)
    elif source.crs.is_projected and np.isclose(source.crs.linear_units_factor[1],1):
        target = source.crs
    else:
        centre = source.transform*(source.width/2,source.height/2)
        lon,lat = transform_coordinates(source.crs,"EPSG:4326",[centre[0]],[centre[1]])
        if not -80<=lat[0]<=84:
            raise ValueError("Outside UTM coverage: provide a suitable projected metre target_crs")
        zone = max(1,min(60,math.floor((lon[0]+180)/6)+1))
        target = CRS.from_epsg((32600 if lat[0]>=0 else 32700)+zone)
    if not target.is_projected or not np.isclose(target.linear_units_factor[1],1):
        raise ValueError("Target CRS must be projected with metre units")
    return target


def _pixel_size_metres(source):
    """Pixel-axis lengths at image centre; geographic grids vary with location."""
    if source.crs is None:
        raise ValueError("Metre resolution requires a georeferenced source with a CRS")
    if source.crs.is_projected:
        factor = source.crs.linear_units_factor[1]
        return tuple(value*factor for value in source.res)
    centre = (source.width/2,source.height/2)
    points = [source.transform*p for p in [centre,(centre[0]+1,centre[1]),(centre[0],centre[1]+1)]]
    lon,lat = transform_coordinates(source.crs,"EPSG:4326",[points[0][0]],[points[0][1]])
    local = CRS.from_string(f"+proj=aeqd +lat_0={lat[0]} +lon_0={lon[0]} +datum=WGS84 +units=m")
    x,y = transform_coordinates(source.crs,local,[p[0] for p in points],[p[1] for p in points])
    return math.hypot(x[1]-x[0],y[1]-y[0]),math.hypot(x[2]-x[0],y[2]-y[0])


def _grid_starts(length,tile_size,stride,edge_mode):
    if edge_mode=="pad":
        return list(range(0,max(1,length-tile_size+stride),stride))
    if length<tile_size:
        raise ValueError("Raster is smaller than tile_size; reduce tile_size or explicitly use edge_mode='pad'")
    starts = list(range(0,length-tile_size+1,stride))
    if starts[-1]!=length-tile_size:
        starts.append(length-tile_size)
    return starts


def _resample(source_path,destination,resolution,target_crs,resampling,progress,resolution_mode):
    destination.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary = tempfile.mkstemp(dir=destination.parent,suffix=".partial.tif")
    os.close(fd)
    try:
        with rasterio.Env(GDAL_CACHEMAX=256*1024*1024,GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(source_path) as source:
                if resolution_mode=="preserve":
                    pixel_x,pixel_y = _pixel_size_metres(source)
                    width = max(1,round(source.width*pixel_x/resolution))
                    height = max(1,round(source.height*pixel_y/resolution))
                    target = source.crs
                    transform = source.transform*source.transform.scale(source.width/width,source.height/height)
                else:
                    target = _metric_target(source,target_crs)
                    transform,width,height = calculate_default_transform(source.crs,target,source.width,source.height,
                                                                           *source.bounds,resolution=resolution)
                profile = dict(driver="GTiff",width=width,height=height,count=3,dtype=source.dtypes[0],
                               crs=target,transform=transform,tiled=True,blockxsize=512,blockysize=512,
                               compress="deflate",zlevel=4,BIGTIFF="IF_SAFER",nodata=None)
                with WarpedVRT(source,crs=target,transform=transform,width=width,height=height,
                               resampling=Resampling[resampling],add_alpha=ColorInterp.alpha not in source.colorinterp,
                               warp_mem_limit=128) as warped, rasterio.open(temporary,"w",**profile) as output:
                    output.colorinterp = (ColorInterp.red,ColorInterp.green,ColorInterp.blue)
                    total = math.ceil(width/1024)*math.ceil(height/1024)
                    completed = 0
                    for y in range(0,height,1024):
                        for x in range(0,width,1024):
                            window = Window(x,y,min(1024,width-x),min(1024,height-y))
                            output.write(warped.read([1,2,3],window=window),window=window)
                            output.write_mask(warped.dataset_mask(window=window),window=window)
                            completed += 1
                        if progress:
                            print(f"Resampling: {completed}/{total} blocks ({completed/total:.0%})",flush=True)
                    output.update_tags(SATELLITEGRAPH_ORIGINAL_SOURCE=str(source_path),
                                       SATELLITEGRAPH_TARGET_RESOLUTION_M=str(resolution),
                                       SATELLITEGRAPH_RESOLUTION_MODE=resolution_mode,
                                       SATELLITEGRAPH_RESAMPLING=resampling)
        os.replace(temporary,destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_raster_tiles(source_path,data_directory,*,target_resolution=None,target_crs=None,
                         image_name=None,tile_size=512,overlap=50,resampling="average",progress=False,
                         resolution_mode="preserve",edge_mode="shift"):
    """Save the complete optionally resized raster, then tile it without dropping pixels.

    None preserves the source grid. RGB uint8/uint16 imagery is supported without
    radiometric normalization. Existing raster/tile outputs are never overwritten.
    Preserve mode retains the exact footprint/CRS; metre resolution is approximate
    at the centre for geographic grids. Projected mode explicitly changes CRS.
    Shift mode anchors the final windows to the edges, increasing their overlap.
    Tile alpha is authoritative source validity; no tiles are filtered out.
    """
    for name,value in (("tile_size",tile_size),("overlap",overlap)):
        if isinstance(value,bool) or not isinstance(value,int) or value < (1 if name=="tile_size" else 0):
            raise ValueError(f"Invalid {name}")
    if overlap>=tile_size:
        raise ValueError("overlap must be smaller than tile_size")
    if target_resolution is not None and (isinstance(target_resolution,bool)
            or not isinstance(target_resolution,(int,float)) or not math.isfinite(target_resolution) or target_resolution<=0):
        raise ValueError("target_resolution must be a finite positive metre value or None")
    if target_resolution is None and target_crs is not None:
        raise ValueError("A target CRS requires target_resolution")
    if resolution_mode not in {"preserve","projected"} or edge_mode not in {"shift","pad"}:
        raise ValueError("Invalid resolution_mode or edge_mode")
    if target_crs is not None and resolution_mode!="projected":
        raise ValueError("target_crs requires resolution_mode='projected'; preserve mode retains the original CRS")
    if resampling not in {"average","bilinear","cubic","nearest"}:
        raise ValueError("Unsupported resampling method")
    source_path = Path(source_path).resolve()
    data_directory = Path(data_directory).resolve()
    suffix = "" if target_resolution is None else f"_{target_resolution:g}m".replace(".","p")
    name = image_name or source_path.stem+suffix
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",name):
        raise ValueError("image_name must be a safe source identifier")
    tiles_directory = data_directory/"tiles"/name
    metadata_path = data_directory/"metadata"/f"{name}.csv"
    manifest_path = data_directory/"metadata"/f"{name}.stage1.json"
    raster_path = data_directory/"raw"/f"{name}.tif" if target_resolution is not None else source_path
    if tiles_directory.exists() or metadata_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Stage 1 outputs already exist for {name}; choose a new image_name")
    if target_resolution is not None and raster_path.exists():
        raise FileExistsError(f"Output raster already exists: {raster_path}")
    if target_resolution is None and (source_path.parent!=data_directory/"raw" or name!=source_path.stem):
        raise ValueError("Without resampling the source must be data_directory/raw/<image_name>.tif")
    with rasterio.open(source_path) as source:
        if source.count<3 or len(set(source.dtypes[:3]))!=1 or source.dtypes[0] not in {"uint8","uint16"}:
            raise ValueError("Stage 1 PNG tiling requires at least three uint8 or uint16 RGB bands")
        if target_resolution is not None:
            if resolution_mode=="projected":
                _metric_target(source,target_crs)
            else:
                _pixel_size_metres(source)
        original = {"path":str(source_path),"width":source.width,"height":source.height,
                    "crs":str(source.crs),"transform":list(source.transform)[:6],"file_size":source_path.stat().st_size}
    manifest = {"schema_version":"satellitegraph.stage1.tiles.v3","status":"running","started_at":now(),
                "original_source":original,"image_name":name,"target_resolution_m":target_resolution,
                "tile_size":tile_size,"overlap":overlap,"stride":tile_size-overlap,
                "resolution_mode":resolution_mode if target_resolution is not None else None,"edge_mode":edge_mode,
                "resampling":resampling if target_resolution is not None else None,
                "source_raster":str(raster_path),"tiles_directory":str(tiles_directory),"metadata_path":str(metadata_path)}
    atomic_json(manifest_path,manifest)
    try:
        if target_resolution is not None:
            if progress:
                print("Step 1/2: converting the complete raster before creating any tiles",flush=True)
            _resample(source_path,raster_path,target_resolution,target_crs,resampling,progress,resolution_mode)
        if progress:
            print(f"Step 2/2: tiling the saved raster: {raster_path}",flush=True)
        tiles_directory.mkdir(parents=True)
        stride = tile_size-overlap
        columns = ["tile_id","tile_name","x0","y0","x1","y1","width","height",
                   "valid_width","valid_height","alpha_band","valid_pixel_count"]
        valid_total,empty_count = 0,0
        with rasterio.open(raster_path) as raster, metadata_path.open("w",newline="",encoding="utf-8") as stream:
            writer = csv.DictWriter(stream,fieldnames=columns)
            writer.writeheader()
            xs = _grid_starts(raster.width,tile_size,stride,edge_mode)
            ys = _grid_starts(raster.height,tile_size,stride,edge_mode)
            count = 0
            for y in ys:
                for x in xs:
                    width,height = min(tile_size,raster.width-x),min(tile_size,raster.height-y)
                    window = Window(x,y,width,height)
                    rgb = raster.read([1,2,3],window=window).transpose(1,2,0)
                    valid = np.all(raster.read_masks([1,2,3],window=window)>0,axis=0)
                    rgba = np.zeros((tile_size,tile_size,4),dtype=rgb.dtype)
                    rgba[:height,:width,:3] = rgb
                    rgba[:height,:width,3] = valid.astype(rgb.dtype)*np.iinfo(rgb.dtype).max
                    filename = f"tile_{count:05d}.png"
                    if not cv2.imwrite(str(tiles_directory/filename),cv2.cvtColor(rgba,cv2.COLOR_RGBA2BGRA)):
                        raise OSError(f"Could not write {filename}")
                    valid_count = int(valid.sum())
                    valid_total += valid_count
                    empty_count += int(valid_count==0)
                    writer.writerow(dict(tile_id=count,tile_name=filename,x0=x,y0=y,x1=x+tile_size,y1=y+tile_size,
                                         width=tile_size,height=tile_size,valid_width=width,valid_height=height,
                                         alpha_band=4,valid_pixel_count=valid_count))
                    count += 1
                if progress:
                    print(f"Tiling: {count}/{len(xs)*len(ys)} tiles",flush=True)
            manifest.update(width=raster.width,height=raster.height,crs=str(raster.crs),
                            transform=list(raster.transform)[:6],resolution=list(raster.res),
                            tile_count=count,tile_columns=len(xs),tile_rows=len(ys),
                            grid_x_offsets=xs,grid_y_offsets=ys,
                            pixel_size_metres_at_centre=list(_pixel_size_metres(raster)) if raster.crs else None,
                            fully_invalid_tile_count=empty_count,valid_pixels_including_overlap=valid_total)
        manifest.update(status="complete",completed_at=now())
        atomic_json(manifest_path,manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        atomic_json(manifest_path,manifest)
        raise
