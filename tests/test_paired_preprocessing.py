import json
from dataclasses import replace
import cv2
import numpy as np
import pytest
from tests.semantic_fixtures import make_year


def prepared(tmp_path):
    from pipelines.semantic_temporal.runtime.context import build_context
    from pipelines.semantic_temporal.runtime.registration import prepare_comparison
    folders = [make_year(tmp_path, year, names=('a.png', 'b.png'), offsets=((3, 2), (10, 2)))
               for year in (2013, 2023)]
    return prepare_comparison(build_context({'folders': list(map(str, folders)),
        'harmonize_images': True}, tmp_path/'run'))


def test_masked_pair_preserves_overlap(tmp_path):
    from pipelines.semantic_temporal.runtime.preprocessing import prepare_model_inputs
    from pipelines.semantic_temporal.runtime.registration import prepare_tile
    context = prepare_model_inputs(prepared(tmp_path))
    a, b = context.years[0].tiles
    left, _, _ = prepare_tile(a, context)
    right, _, _ = prepare_tile(b, context)
    assert np.array_equal(left[:, 7:], right[:, :11])
    assert len(context.options['prepared_inputs']) == 4


def test_original_rgb_is_preserved(tmp_path):
    from pipelines.semantic_temporal.runtime.preprocessing import prepare_model_inputs
    from pipelines.semantic_temporal.runtime.registration import prepare_tile
    context = prepared(tmp_path)
    tile = context.years[0].tiles[0]
    before, _, _ = prepare_tile(tile, context)
    context = prepare_model_inputs(context)
    after, _, _ = prepare_tile(tile, context, model=False)
    assert np.array_equal(before, after)


def test_invalid_pixels_do_not_change_histogram_or_sharpness():
    from pipelines.semantic_temporal.runtime.preprocessing import harmonize_images
    rng = np.random.default_rng(23)
    ref = rng.integers(50, 200, (48, 64, 3), dtype=np.uint8)
    mov = cv2.GaussianBlur(ref, (0, 0), 2.)
    valid = np.ones(ref.shape[:2], bool); valid[:8] = False
    a, b, info = harmonize_images(ref, mov, valid, valid)
    ref2 = ref.copy(); mov2 = mov.copy(); ref2[:8] = 255; mov2[:8] = 0
    a2, b2, info2 = harmonize_images(ref2, mov2, valid, valid)
    assert np.array_equal(a[valid], a2[valid])
    assert np.array_equal(b[valid], b2[valid])
    assert info['blur_sigma_reference'] > 0
    assert info['blur_sigma_reference'] == info2['blur_sigma_reference']


def test_cached_inputs_are_hash_verified(tmp_path):
    from pipelines.semantic_temporal.runtime.preprocessing import prepare_model_inputs
    from pathlib import Path
    context = prepare_model_inputs(prepared(tmp_path))
    path = Path(next(iter(context.options['prepared_inputs'].values()))['model_rgb'])
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Artifact changed|hash|integrity'):
        prepare_model_inputs(replace(context, options={k:v for k,v in context.options.items() if k != 'prepared_inputs'}))


def test_saved_controls_match_registration(tmp_path):
    from pipelines.semantic_temporal.runtime.context import build_context
    from pipelines.semantic_temporal.runtime.registration import prepare_comparison
    folders = [make_year(tmp_path, y) for y in (2013, 2023)]
    points = np.array([[0,0],[10,0],[0,10],[10,10],[20,0],[0,20]], float) + [500000,3700000]
    path = tmp_path/'controls.json'
    path.write_text(json.dumps({'moving_xy': points.tolist(), 'reference_xy': (points+[3,-1.5]).tolist()}))
    context = prepare_comparison(build_context({'folders': list(map(str,folders)), 'registration_json': str(path)}, tmp_path/'run'))
    controls = json.loads((context.output/'prepared/registration_controls.json').read_text())
    assert controls['units'] == 'metres'
    assert len(controls['pairs']) == 6
    assert all(row['inlier'] and row['residual'] < 1e-5 for row in controls['pairs'])


def test_harmonization_option_inherits_and_invalidates_upstream(tmp_path):
    from src.pipeline_select import resolve_options
    from src.pipeline_state import input_fingerprint
    base = {'harmonize_images': True}
    assert resolve_options(base, {'harmonize_images': None}, initial=False, project_root=tmp_path)['harmonize_images']
    assert not resolve_options(base, {'harmonize_images': False}, initial=False, project_root=tmp_path)['harmonize_images']
    assert input_fingerprint({}) == input_fingerprint({'harmonize_images': False})
    assert input_fingerprint(base) != input_fingerprint({'harmonize_images': False})
