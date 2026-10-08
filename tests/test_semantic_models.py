from pathlib import Path
import numpy as np
import pytest


def test_device_and_window_validation(tmp_path):
    from pipelines.semantic_temporal.runtime.models import ModelBundle
    with pytest.raises(ValueError,match='window'): ModelBundle(tmp_path,window=16)
    import torch
    if not torch.cuda.is_available():
        with pytest.raises(ValueError,match='CUDA'): ModelBundle(tmp_path,device='cuda')
    assert ModelBundle(tmp_path,device='auto').device==('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.mark.real_semantic
def test_actual_cpu_models_on_rectangular_image():
    from pipelines.semantic_temporal.runtime.models import ModelBundle
    path=Path(__file__).resolve().parents[1]/'checkpoints/semantic_temporal'
    if not (path/'DelineateAnythingv2.pt').exists(): pytest.skip('Local verified checkpoints required')
    models=ModelBundle(path,window=64,overlap=16,tta=False)
    image=np.full((35,51,3),120,np.uint8)
    probabilities=models.probabilities(image,1.5)
    assert probabilities.shape==(9,52,76)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(0),1,atol=1e-5)
    masks,confidence=models.parcels(image,(70,102),.2)
    assert masks.shape[1:]==(70,102)
    assert len(masks)==len(confidence)
