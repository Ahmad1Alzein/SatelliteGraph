import numpy as np
import pytest
from PIL import Image
from tests.test_semantic_temporal import temporal_fixture
from pipelines.semantic_temporal.runtime.storage import read_json


def test_arbitrary_year_rectangular_previews_and_artifact_hashes(tmp_path):
    from pipelines.semantic_temporal.runtime.visualization import render_outputs,validate_run
    from pipelines.semantic_temporal.runtime.temporal import detect_classical
    artifact,context=temporal_fixture(tmp_path);detect_classical(artifact,context)
    result=validate_run(context,'change');assert result['status']=='pass'
    paths=render_outputs(context,'change');assert paths
    index=read_json(context.output/'preview_index.json')
    assert index['years']==[2017,2023] and index['units']==['metres']
    for path in paths.values():
        with Image.open(path) as image: assert image.width>image.height
    with artifact.paths['tensors'].open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError,match='changed'): validate_run(context,'change')
