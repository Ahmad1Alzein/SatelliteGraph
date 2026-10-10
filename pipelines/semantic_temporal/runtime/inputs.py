"""Validate supplied placement, or explicitly use the requested left-to-right row."""
from pathlib import Path
import csv
import math
import re
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window
from affine import Affine
from .contracts import GridSpec, TileInput, YearInput


def _path(value, root):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else root/p).resolve()


def _integer(value, label, *, minimum=0):
    try: number=float(value)
    except (TypeError,ValueError): raise ValueError(f'Invalid {label}: {value}') from None
    if not math.isfinite(number) or not number.is_integer() or number<minimum:
        raise ValueError(f'Invalid {label}: {value}')
    return int(number)


def _rgb(array):
    if array.dtype==np.uint16: return np.round(array.astype(np.float64)*(255/65535)).astype(np.uint8)
    if array.dtype!=np.uint8: raise ValueError('RGB input must be uint8 or uint16')
    return array


def _read_image(path):
    if path.suffix.lower() in ('.tif','.tiff'):
        with rasterio.open(path) as src:
            if src.count<3: raise ValueError(f'RGB raster requires three bands: {path}')
            return _rgb(src.read([1,2,3]).transpose(1,2,0)), src.dataset_mask()>0
    with Image.open(path) as image:
        if image.mode not in ('RGB','RGBA'): raise ValueError(f'RGB/RGBA image required: {path}')
        array=np.asarray(image)
    return array[:,:,:3].copy(), array[:,:,3]>0 if array.shape[2]==4 else np.ones(array.shape[:2],bool)


def _grid(transform, crs, scale=None):
    if crs is not None:
        crs=rasterio.crs.CRS.from_user_input(crs)
        metric=crs.is_projected and math.isclose(crs.linear_units_factor[1],1.0)
        physical=(math.hypot(transform.a,transform.d),math.hypot(transform.b,transform.e)) if metric else None
        return GridSpec(crs.to_string(),tuple(transform)[:6],'metres' if metric else 'pixels',physical)
    return GridSpec(None,tuple(transform)[:6],'pixels',(scale,scale) if scale else None)


def _one_year(folder, year, *, selected=None, metadata=None, raster=None, scale=None):
    if not folder.is_dir(): raise ValueError(f'Image folder not found: {folder}')
    def natural(path):
        return [int(s) if s.isdigit() else s.casefold() for s in re.split(r'(\d+)',path.name)]
    images=[selected] if selected else sorted((p for p in folder.iterdir() if p.suffix.lower()=='.png'),key=natural)
    if not images:
        images=sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.tif','.tiff'))
    if not images: raise ValueError(f'No RGB images in {folder}')
    if any(not p.is_file() for p in images): raise ValueError('Selected image does not exist')
    if all(p.suffix.lower() in ('.tif','.tiff') for p in images):
        tiles=[]
        for image in images:
            rgb,valid=_read_image(image)
            with rasterio.open(image) as src:
                if src.crs is None: raise ValueError('GeoTIFF image requires georeferencing')
                grid=_grid(src.transform,src.crs)
            key=f'{year}/{image.name}'
            tiles.append(TileInput(key,year,image,None,str(image),(0,0),rgb.shape[:2],grid,rgb.shape[:2]))
        return YearInput(year,tuple(tiles),None)
    meta=metadata or folder/'metadata.csv'
    if not meta.is_file():
        if metadata or raster:
            raise ValueError('Explicit source placement requires metadata.csv with tile_name, x0 and y0')
        tiles=[];x=0
        for image in images:
            rgb,valid=_read_image(image);h,w=rgb.shape[:2]
            tiles.append(TileInput(f'{year}/{image.name}',year,image,None,str(folder),
                (x,0),(h,w),_grid(Affine.translation(x,0),None,scale),(h,w)))
            x+=w
        return YearInput(year,tuple(tiles),None)
    with meta.open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        if not {'tile_name','x0','y0'}.issubset(reader.fieldnames or []):
            raise ValueError('Placement metadata needs tile_name, x0 and y0')
        rows=list(reader)
    names=[r['tile_name'].casefold() for r in rows]
    if len(names)!=len(set(names)): raise ValueError('duplicate image names in metadata')
    by_name={r['tile_name'].casefold():r for r in rows}
    selected_names={p.name.casefold() for p in images}
    if (selected and not selected_names.issubset(by_name)) or (not selected and selected_names!=set(by_name)):
        raise ValueError('metadata must contain every selected image exactly once')
    if raster is None:
        candidates=sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.tif','.tiff'))
        if len(candidates)>1: raise ValueError('Ambiguous source rasters; supply --source-rasters')
        raster=candidates[0] if candidates else None
    source=rasterio.open(raster) if raster else None
    try:
        if source and (source.crs is None or source.count<3): raise ValueError('Source raster requires georeferencing and RGB bands')
        tiles=[]
        for image in images:
            rgb,valid=_read_image(image); h,w=rgb.shape[:2]; row=by_name[image.name.casefold()]
            x=_integer(row['x0'],'x0'); y=_integer(row['y0'],'y0')
            for name,expected in (('width',w),('height',h),('x1',x+w),('y1',y+h)):
                if row.get(name) and _integer(row[name],name)!=expected: raise ValueError(f'{name} disagrees with image dimensions')
            vh=_integer(row.get('valid_height') or h,'valid_height',minimum=1)
            vw=_integer(row.get('valid_width') or w,'valid_width',minimum=1)
            if vh>h or vw>w or valid[vh:,:].any() or valid[:,vw:].any(): raise ValueError('Invalid padded tile extent or alpha mask')
            if row.get('valid_pixel_count') and _integer(row['valid_pixel_count'],'valid_pixel_count')!=int(valid.sum()):
                raise ValueError('metadata valid pixel count disagrees with alpha mask')
            if source:
                if x+vw>source.width or y+vh>source.height: raise ValueError('Tile lies outside original source raster')
                window=Window(x,y,vw,vh)
                actual=_rgb(source.read([1,2,3],window=window).transpose(1,2,0))
                mask=valid[:vh,:vw]
                if not np.array_equal(rgb[:vh,:vw][mask],actual[mask]): raise ValueError('Tile pixels do not match source window metadata')
                grid=_grid(source.window_transform(window),source.crs)
            else:
                affine_fields=[f'tile_affine_{k}' for k in ('a','b','c','d','e','f')]
                present=[bool(row.get(k)) for k in affine_fields]
                if any(present):
                    if not all(present) or not row.get('target_crs'): raise ValueError('Incomplete affine/CRS metadata')
                    grid=_grid(Affine(*(float(row[k]) for k in affine_fields)),row['target_crs'])
                else: grid=_grid(Affine.translation(x,y),None,scale)
            tiles.append(TileInput(f'{year}/{image.name}',year,image,meta,
                                   str(raster or row.get('source_image_id') or folder),(x,y),(h,w),grid,(vh,vw)))
        return YearInput(year,tuple(tiles),raster)
    finally:
        if source: source.close()


def inspect_inputs(options: dict, project_root: Path) -> tuple[YearInput,...]:
    root=Path(project_root).resolve()
    if options.get('folders') and options.get('images'): raise ValueError('--folders and --images are mutually exclusive')
    items=options.get('images') or options.get('folders')
    if not items or len(items) not in (1,2): raise ValueError('Supply one or two --folders or --images')
    paths=[_path(p,root) for p in items]; singles=bool(options.get('images'))
    years=options.get('years')
    if years is None:
        try: years=[int(p.parent.name if singles else p.name) for p in paths]
        except ValueError: years=list(range(1,len(paths)+1))
        if len(set(years))!=len(years): years=list(range(1,len(paths)+1))
    if len(years)!=len(paths) or any(not isinstance(y,int) or isinstance(y,bool) or not 1<=y<=9999 for y in years):
        raise ValueError('Supply one valid year per input')
    if len(years)==2 and years[0]>=years[1]: raise ValueError('Years must be distinct and chronological')
    for key in ('metadata','source_rasters','pixel_size'):
        if options.get(key) is not None and len(options[key])!=len(paths): raise ValueError(f'{key} requires one value per input')
    scales=options.get('pixel_size') or [None]*len(paths)
    if any(s is not None and (not math.isfinite(s) or s<=0) for s in scales): raise ValueError('pixel size must be positive metres per pixel')
    output=[]
    for index,(p,year) in enumerate(zip(paths,years)):
        def override(key):
            values=options.get(key)
            return _path(values[index],root) if values and values[index] else None
        output.append(_one_year(p.parent if singles else p,year,selected=p if singles else None,
                                metadata=override('metadata'),raster=override('source_rasters'),scale=scales[index]))
    return tuple(output)


def read_tile(tile: TileInput):
    rgb,valid=_read_image(tile.image)
    if rgb.shape[:2]!=tile.shape_hw: raise ValueError('Image dimensions changed after validation')
    if tile.metadata and tile.grid.crs and tile.source_id and Path(tile.source_id).is_file():
        with rasterio.open(tile.source_id) as src:
            vh,vw=tile.valid_extent_hw; x,y=tile.offset_xy
            mask=src.dataset_mask(window=Window(x,y,vw,vh))>0
            valid[:vh,:vw]&=mask
    return rgb,valid
