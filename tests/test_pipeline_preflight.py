def test_preflight_reports_missing_models_without_downloading(tmp_path,monkeypatch):
    from pipelines.semantic_temporal.runtime.contracts import RunContext
    from pipelines.semantic_temporal.runtime.preflight import check_environment
    context=RunContext(tmp_path/'run',(),{'project_root':str(tmp_path)}, {},'fixture')
    report=check_environment(context)
    assert report['status']=='needs_setup'
    assert len(report['models'])==7 and all(m['status']=='missing' for m in report['models'])
    assert not (tmp_path/'checkpoints').exists()
    assert report['general_requires_osgeo'] is False


def test_unused_publisher_demo_dependency_does_not_block_native_preflight(monkeypatch):
    from pipelines.semantic_temporal.runtime import preflight
    original=preflight.importlib.util.find_spec
    monkeypatch.setattr(preflight.importlib.util,'find_spec',lambda module:None if module=='easydict' else original(module))
    monkeypatch.setattr(preflight,'read_json',lambda path:{'assets':[]})
    report=preflight.check_environment()
    assert not report['libraries']['easydict']['available']
    assert report['status']=='ready'
