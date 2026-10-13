"""Shared pipeline choices and command construction for the thesis workspace."""
from pathlib import Path
import sys

METHOD_LABELS = {
    'segmentation': {'new':'New · SegFormer + Delineate Anything', 'old':'Old · SAM 2.1 + OpenEarthMap'},
    'graph': {'new':'New · Adjacency + nearby relations', 'old':'Old · Adjacency only'},
    'change': {'gnn':'GNN · Original thesis building ST-GNN', 'classical':'Old · Rule-based change detection'},
}


def pipeline_command(output, segmentation, graph, change, inputs, *, input_kind='images',
                     years=(2013, 2023), device='cpu', pixel_size=None,
                     register_images=False, harmonize_images=False):
    from src.pipeline_select import check_combination
    check_combination(segmentation, graph, change)
    if input_kind not in ('images', 'folders') or len(inputs) != 2:
        raise ValueError('Choose two images or two tile folders')
    if len(years) != 2 or years[0] >= years[1]:
        raise ValueError('The before year must precede the after year')
    if not all(Path(p).is_file() if input_kind == 'images' else Path(p).is_dir() for p in inputs):
        raise ValueError('An input path does not exist or has the wrong type')
    command = [sys.executable, '-u', '-m', 'src', 'pipeline', 'run',
        '--segmentation', segmentation, '--graph', graph, '--change', change,
        '--'+input_kind, *map(str, inputs), '--years', *map(str, years),
        '--output', str(output), '--device', device, '--max-report-pixels', '256000000']
    if pixel_size is not None:
        if pixel_size <= 0:raise ValueError('Pixel size must be positive')
        command += ['--pixel-size', str(pixel_size), str(pixel_size)]
    if register_images:command.append('--register-images')
    if harmonize_images:command.append('--harmonize-images')
    return command
