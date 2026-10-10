"""Fetch a reproducible annotated subset using official archive byte ranges.

Usage: python scripts/download_lsscd_subset.py --output data/external_training/lsscd_subset.zip
The archive is CC BY-NC-SA 4.0, academic use only, subject to Google Earth terms.
"""
from pathlib import Path
import argparse
import hashlib
import json
import random
import struct
import time
import urllib.request
import urllib.error
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

URL='https://zenodo.org/api/records/17765755/files/256.zip/content'
SIZE=934763849
DATASET='https://doi.org/10.5281/zenodo.17765755'


def fetch(start,end):
    for attempt in range(5):
        try:
            request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}','User-Agent':'SatelliteGraph research dataset downloader'})
            with urllib.request.urlopen(request,timeout=90) as response:
                if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{SIZE}':
                    raise ValueError('Official archive byte range/size changed')
                value=response.read()
            if len(value)!=end-start+1:raise ValueError('Truncated dataset download')
            return value
        except (urllib.error.URLError,TimeoutError) as exc:
            if attempt==4:raise
            time.sleep(min(60,2**attempt*3))


def directory(data):
    index={};offset=0
    while data[offset:offset+4]==b'PK\x01\x02':
        v=struct.unpack('<4s6H3I5H2I',data[offset:offset+46]);name=data[offset+46:offset+46+v[10]].decode('utf8')
        index[name]=dict(crc=v[7],compressed=v[8],size=v[9],offset=v[-1],method=v[4])
        offset+=46+sum(v[10:13])
    if len(index)!=44820:raise ValueError('Official archive index changed')
    return index


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-pairs',type=int,default=24);parser.add_argument('--validation-pairs',type=int,default=10)
    parser.add_argument('--test-pairs',type=int,default=10);args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cache=args.output.parent/'zip_directory.bin'
    data=cache.read_bytes() if cache.is_file() else fetch(929698745,SIZE-1)
    index=directory(data);rng=random.Random(1729);selection=[];scenes={}
    for official in ('train','test'):
        names=[n.rsplit('/',1)[1] for n in index if n.startswith(f'{official}/2013/images/') and n.endswith('.png')]
        by_parent={}
        for name in names:by_parent.setdefault(Path(name).stem.split('_')[1],[]).append(name)
        if len(by_parent)!=(70 if official=='train' else 30) or any(len(v)!=64 for v in by_parent.values()):
            raise ValueError('Unexpected parent scene/chip structure')
        scenes[official]=by_parent
    if set(scenes['train'])&set(scenes['test']):raise ValueError('Official parent scenes overlap')
    train_parents=sorted(scenes['train'],key=int);rng.shuffle(train_parents)
    test_parents=sorted(scenes['test'],key=int);rng.shuffle(test_parents)
    if args.train_pairs+args.validation_pairs>70 or args.test_pairs>30 or min(args.train_pairs,args.validation_pairs,args.test_pairs)<1:
        parser.error('Requested subset exceeds independent parent scenes')
    sets=[('train','train',train_parents[:args.train_pairs]),
          ('validation','train',train_parents[args.train_pairs:args.train_pairs+args.validation_pairs]),
          ('test','test',test_parents[:args.test_pairs])]
    for split,official,parents in sets:
        for parent in parents:
            name=rng.choice(sorted(scenes[official][parent]))
            selection.append(dict(split=split,official_split=official,name=name,parent_scene=parent,
                selection='seed 1729 random parent and chip; one chip per parent'))
    wanted=[f'{item["official_split"]}/{year}/{folder}/{item["name"]}' for item in selection
        for year in (2013,2015) for folder in ('images','semantic_masks')]
    def member(name):
        info=index[name];start=info['offset']
        # Local file header + name + bounded extra field. Re-request precisely if
        # the header uses a larger extra field; never fetch an entire archive.
        header=fetch(start,start+29);v=struct.unpack('<4s5H3I2H',header)
        if v[0]!=b'PK\x03\x04' or v[3]!=8:raise ValueError('Unexpected archive compression')
        payload=fetch(start+30+v[-2]+v[-1],start+29+v[-2]+v[-1]+info['compressed'])
        content=zlib.decompress(payload,-15)
        if len(content)!=info['size'] or zlib.crc32(content)&0xffffffff!=info['crc']:raise ValueError('Dataset member CRC mismatch')
        return name,content
    hashes={}
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED) as archive,ThreadPoolExecutor(max_workers=3) as pool:
        for number,(name,content) in enumerate(pool.map(member,wanted),1):
            archive.writestr(name,content);hashes[name]=hashlib.sha256(content).hexdigest()
            print(f'Downloaded {number}/{len(wanted)}: {name}',flush=True)
    manifest=dict(dataset=DATASET,source=URL,official_archive_size=SIZE,
        official_archive_md5='feb0e1e8a79bcd673318131671af9f8a',official_full_archive_md5_verified=False,
        directory_sha256=hashlib.sha256(data).hexdigest(),member_sha256=hashes,
        subset_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),selection=selection,
        license='CC BY-NC-SA 4.0; academic purposes only; Google Earth terms apply')
    args.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print('Verified subset ready: '+str(args.output.resolve()),flush=True)


if __name__=='__main__':main()
