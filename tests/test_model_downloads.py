import hashlib
import pytest


def test_valid_assets_are_reused_and_existing_differing_files_preserved(tmp_path,monkeypatch):
    import scripts.download_pipeline_models as download
    content=b'test checkpoint';source=tmp_path/'source.pt';source.write_bytes(content)
    asset={'path':'new/model.pt','method':'new','url':source.as_uri(),'size':len(content),'sha256':hashlib.sha256(content).hexdigest()}
    monkeypatch.setattr(download,'assets',lambda:[asset])
    destination=tmp_path/'checkpoints';first=download.ensure_models(destination,methods=('new',))
    assert first['assets'][0]['status']=='downloaded'
    assert download.ensure_models(destination,methods=('new',))['assets'][0]['status']=='verified_existing'
    target=destination/asset['path'];target.write_bytes(b'custom weights')
    with pytest.raises(ValueError,match='preserved'): download.ensure_models(destination,methods=('new',))
    assert target.read_bytes()==b'custom weights'


def test_failed_hash_never_promotes_partial_file(tmp_path,monkeypatch):
    import scripts.download_pipeline_models as download
    source=tmp_path/'source.pt';source.write_bytes(b'wrong')
    monkeypatch.setattr(download,'assets',lambda:[{'path':'model.pt','method':'new','url':source.as_uri(),'size':5,'sha256':'0'*64}])
    with pytest.raises(ValueError,match='hash'): download.ensure_models(tmp_path/'out',methods=('new',))
    assert not (tmp_path/'out/model.pt').exists()
    assert not (tmp_path/'out/model.pt.part').exists()
