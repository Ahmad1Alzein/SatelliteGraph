"""Shared original-raster sampling, geometry drawing, and readable report layout."""
from pathlib import Path
import math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from affine import Affine
from ..comparison import warp_array
from ..visualization import COLORS
from ..change_contracts import CLASS_LABELS

EDGE_STYLES={'touches':('#00a6b5','solid'),'near':('#ec8b20','dashed'),
    'context_near':('#9b42d2','dotted'),'t_primary_fwd':('#00aa88','solid'),
    't_primary_bwd':('#00aa88','solid'),'t_overlap_fwd':('#ed9b24','dashed'),
    't_overlap_bwd':('#ed9b24','dashed'),'change_conversion':('#ee3c83','dotted')}


def font(size=16):
    for name in ('DejaVuSans.ttf','arial.ttf','C:/Windows/Fonts/arial.ttf'):
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    try:return ImageFont.load_default(size=size)
    except TypeError:return ImageFont.load_default()


def class_color(klass):
    return tuple(COLORS[int(klass)]) if int(klass) in range(9) else (235,235,240)


def frame_bounds(frame):
    h,w=frame['shape_hw'];a=Affine(*frame['grid'].affine)
    points=np.array([a*p for p in ((0,0),(w,0),(w,h),(0,h))])
    return (*points.min(0),*points.max(0))


def fit_bounds(bounds,size):
    left,bottom,right,top=map(float,bounds);w,h=size
    extent_x=max(right-left,1e-8);extent_y=max(top-bottom,1e-8)
    scale=max(extent_x/w,extent_y/h);cx=(left+right)/2;cy=(bottom+top)/2
    return (cx-scale*w/2,cy-scale*h/2,cx+scale*w/2,cy+scale*h/2)


def view_affine(bounds,size,*,flip_y=True):
    left,bottom,right,top=fit_bounds(bounds,size);w,h=size
    return Affine((right-left)/w,0,left,0,(-1 if flip_y else 1)*(top-bottom)/h,top if flip_y else bottom)


def world_pixel(x,y,bounds,size,*,flip_y=True):
    return (~view_affine(bounds,size,flip_y=flip_y))*(x,y)


def sample_rgb(frame,year,bounds,size):
    data=frame['data'][year];grid=frame['grid'];flip_y=bool(grid.crs) or grid.affine[4]<0
    affine=view_affine(bounds,size,flip_y=flip_y);source=Affine(*grid.affine);shape_hw=size[::-1]
    valid=warp_array(data['valid'].astype(np.uint8),source,affine,shape_hw,nearest=True)>0
    weight=warp_array(data['valid'].astype(np.float32),source,affine,shape_hw)
    rgb=warp_array(data['rgb_original'].transpose(2,0,1).astype(np.float32)*data['valid'],source,affine,shape_hw)
    rgb=np.divide(rgb,weight,out=np.zeros_like(rgb),where=weight>1e-8)
    rgb=np.clip(np.rint(rgb.transpose(1,2,0)),0,255).astype(np.uint8);rgb[~valid]=0
    return rgb,valid


def display_rgb(rgb,valid):
    result=rgb.copy();yy,xx=np.indices(valid.shape);shade=np.where((xx//8+yy//8)%2,220,240).astype(np.uint8)
    result[~valid]=shade[~valid,None]
    return Image.fromarray(result)


def sample_classes(frame,year,bounds,size):
    grid=frame['grid'];affine=view_affine(bounds,size,flip_y=bool(grid.crs) or grid.affine[4]<0)
    return warp_array(frame['data'][year]['classes'],Affine(*grid.affine),affine,size[::-1],nearest=True)


def paint_geometry(image,geometry,bounds,color,*,alpha=0,flip_y=True,width=2):
    if geometry.is_empty:return
    parts=[geometry] if geometry.geom_type=='Polygon' else list(geometry.geoms)
    mask=Image.new('L',image.size,0);md=ImageDraw.Draw(mask);rings=[]
    for part in parts:
        if part.geom_type!='Polygon':continue
        exterior=[world_pixel(x,y,bounds,image.size,flip_y=flip_y) for x,y in part.exterior.coords]
        md.polygon(exterior,fill=alpha);rings.append(exterior)
        for hole in part.interiors:
            ring=[world_pixel(x,y,bounds,image.size,flip_y=flip_y) for x,y in hole.coords]
            md.polygon(ring,fill=0);rings.append(ring)
    if alpha:image.paste(Image.new('RGB',image.size,color),(0,0),mask)
    draw=ImageDraw.Draw(image)
    for ring in rings:draw.line(ring,fill=color,width=width,joint='curve')


def styled_line(draw,start,end,color,style='solid',width=2,arrow=False):
    start=np.asarray(start,float);end=np.asarray(end,float);delta=end-start;length=float(np.linalg.norm(delta))
    if style=='solid' or length<1:draw.line([tuple(start),tuple(end)],fill=color,width=width)
    else:
        dash,gap=(8,5) if style=='dashed' else (2,4)
        for position in np.arange(0,length,dash+gap):
            a=start+delta*(position/length);b=start+delta*(min(position+dash,length)/length)
            draw.line([tuple(a),tuple(b)],fill=color,width=width)
    if arrow and length>=6:
        unit=delta/length;normal=np.array([-unit[1],unit[0]])
        draw.polygon([tuple(end),tuple(end-unit*8+normal*4),tuple(end-unit*8-normal*4)],fill=color)


def wrap_text(text,width,*,size=16):
    draw=ImageDraw.Draw(Image.new('RGB',(1,1)));face=font(size);lines=[];current=''
    for word in str(text).split():
        candidate=(current+' '+word).strip()
        if draw.textlength(candidate,font=face)<=width:current=candidate;continue
        if current:lines.append(current);current=''
        # Break long IDs/filenames as well as ordinary words.
        for character in word:
            if current and draw.textlength(current+character,font=face)>width:lines.append(current);current=''
            current+=character
    if current:lines.append(current)
    return lines or ['']


def compose(body,title,legend,*,minimum_width=640):
    """Return image plus measurable body/legend bounds; no text covers imagery."""
    width=max(minimum_width,body.width+24)
    title_lines=wrap_text(title,width-32,size=20);header=24+len(title_lines)*26
    legend_lines=[]
    for entry in legend:
        text,color,style=entry if isinstance(entry,tuple) else (entry,None,None)
        for n,line in enumerate(wrap_text(text,width-76)):
            legend_lines.append((line,color if n==0 else None,style))
    legend_y=header+body.height+18;height=legend_y+len(legend_lines)*23+16
    image=Image.new('RGB',(width,height),'white');image.paste(body,(12,header));draw=ImageDraw.Draw(image)
    for i,line in enumerate(title_lines):draw.text((16,12+i*26),line,font=font(20),fill='#182331')
    for i,(text,color,style) in enumerate(legend_lines):
        y=legend_y+i*23
        if color is not None:
            if style:styled_line(draw,(18,y+9),(42,y+9),color,style,3)
            else:draw.rectangle((20,y+3,38,y+17),fill=color,outline='#444444')
        draw.text((50,y),text,font=font(),fill='#182331')
    return image,{'image_bounds':[12,header,12+body.width,header+body.height],
                  'legend_bounds':[16,legend_y,width-16,height-16],'canvas_size':[width,height]}


def horizontal_panels(panels,captions):
    cells=[max(p.width,260) for p in panels]
    caption_lines=[wrap_text(caption,width-8) for caption,width in zip(captions,cells)]
    header=max(len(lines) for lines in caption_lines)*23+5
    width=sum(cells)+16*(len(panels)-1);height=max(p.height for p in panels)+header
    body=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(body);x=0
    for panel,lines,cell in zip(panels,caption_lines,cells):
        for line,text in enumerate(lines):draw.text((x+4,2+line*23),text,font=font(),fill='#182331')
        body.paste(panel,(x+(cell-panel.width)//2,header));x+=cell+16
    return body
