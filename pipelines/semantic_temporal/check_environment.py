from pathlib import Path
import argparse,json,sys,importlib.util
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from pipelines.semantic_temporal.runtime.preflight import check_environment

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Read-only dependency and checkpoint verification')
    parser.add_argument('--reproduction',action='store_true',help='Also check the historic GDAL requirement')
    args=parser.parse_args();report=check_environment()
    if args.reproduction:
        report['osgeo_available']=importlib.util.find_spec('osgeo') is not None
        if not report['osgeo_available']: report['status']='needs_setup'
    print(json.dumps(report,indent=2));raise SystemExit(0 if report['status']=='ready' else 1)
