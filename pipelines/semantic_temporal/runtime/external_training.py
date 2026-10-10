"""Supervision from actual LsSCD-Ex annotations; never from model change guesses."""
from pathlib import Path
import argparse
import copy
import re
import hashlib
import io
import zipfile
from collections import Counter
from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import shape
from .change_contracts import TASKS,KNOWN_CLASSES,RELATIONS,RELATION_PROFILE,CHANGE_SCHEMA
from .gnn_model import RGNN
from .gnn_training import normalization,tensor_inputs,_metrics,_average_precision
from .storage import atomic_json,read_json,sha256

DATASET_URL='https://doi.org/10.5281/zenodo.17765755'
ARCHIVE_MD5='feb0e1e8a79bcd673318131671af9f8a'
LABEL_MAP=np.array([6,4,7,2,3,8,5,1],np.uint8)


def canonical_labels(values):
    values=np.asarray(values)
    if not np.isin(values,range(8)).all():raise ValueError('LsSCD-Ex semantic labels must be integer classes 0..7')
    return LABEL_MAP[values]


def semantic_metrics(confusion):
    confusion=np.asarray(confusion)
    if confusion.shape!=(8,9):raise ValueError('Semantic confusion must be 8 truth classes by 9 predicted classes')
    metrics={}
    for klass,name in KNOWN_CLASSES.items():
        tp=int(confusion[klass-1,klass]);union=int(confusion[klass-1].sum()+confusion[:,klass].sum()-tp)
        metrics[name]=tp/union if union else None
    values=[v for v in metrics.values() if v is not None]
    correct=sum(int(confusion[k-1,k]) for k in KNOWN_CLASSES)
    return dict(pixel_accuracy=correct/max(1,int(confusion.sum())),class_iou=metrics,
        mean_iou=float(np.mean(values)) if values else None,unknown_prediction_pixels=int(confusion[:,0].sum()),
        truth_pixels=int(confusion.sum()))


def parent_scene(name):
    # The official archive uses <chip 0..63>_<parent scene 0..99>.png.
    # Confirmed: every training parent has 64 chips; 70 train / 30 test parents.
    match=re.fullmatch(r'\d+_(\d+)',Path(name).stem)
    if match:return match[1]
    return re.sub(r'_\d+_\d+$','',Path(name).stem)


def split_training_names(names):
    parents=sorted({parent_scene(n) for n in names})
    if len(parents)<2:raise ValueError('Need at least two independent training parent scenes')
    validation=set(parents[::4])
    return ([n for n in names if parent_scene(n) not in validation],
            [n for n in names if parent_scene(n) in validation])


def annotation_targets(document,before,after,*,include_false_class_negatives=False):
    """Label predicted graph nodes only where annotation agrees with their class.

    A positive needs >=70% class replacement, a negative >=95% persistence.
    Intermediate shape/size variation is masked, not taught as construction.
    Optional hard negatives teach rejection of predicted regions whose current
    class is absent from complete known-class annotation, not missing data.
    """
    if before.shape!=after.shape or before.ndim!=2:raise ValueError('Annotations need an identical 2D grid')
    labels=np.full((len(document['nodes']),len(TASKS)),-1,np.int64)
    eligible=np.zeros(labels.shape,bool);first,last=document['years']
    for i,node in enumerate(document['nodes']):
        klass=node['class_id']
        if klass not in KNOWN_CLASSES:continue
        current,other=(after,before) if node['year']==last else (before,after)
        selected=rasterize([(shape(node['geometry']),1)],out_shape=current.shape,transform=Affine.identity(),dtype='uint8')>0
        total=shape(node['geometry']).area
        if selected.sum()<16 or selected.sum()/max(total,1)<.95:continue
        agreement=selected&(current==klass)
        task=KNOWN_CLASSES[klass]+('_gain' if node['year']==last else '_loss');j=TASKS.index(task)
        current_fraction=agreement.sum()/selected.sum()
        if (include_false_class_negatives and current_fraction<=.05
                and np.isin(current[selected],list(KNOWN_CLASSES)).mean()>=.95):
            eligible[i,j]=True;labels[i,j]=0
            continue
        if current_fraction<.8:continue
        eligible[i,j]=True
        fraction=float((agreement&(other!=klass)).sum()/agreement.sum())
        if fraction>=.7:labels[i,j]=1
        elif fraction<=.05:labels[i,j]=0
    return labels,eligible


def validation_score(labels,probabilities,mask,bce,metric):
    if metric not in ('bce','macro_ap'):raise ValueError('Unknown validation selection metric')
    if metric=='macro_ap':
        values=[_average_precision(labels[mask[:,j],j],probabilities[mask[:,j],j])
            for j in range(labels.shape[1]) if all((labels[mask[:,j],j]==v).sum()>=3 for v in (0,1))]
        if values:return (float(np.mean(values)),-bce)
    return (-bce,)


def fit_supervised(x,relations,labels,eligible,splits,settings):
    """One supervised fit; validation picks thresholds, official test stays held out."""
    splits=np.asarray(splits);known=eligible&(labels>=0)
    train=known&(splits=='train')[:,None];val=known&(splits=='validation')[:,None]
    supported=[j for j in range(len(TASKS)) if set(labels[train[:,j],j])=={0,1}]
    if not supported:raise ValueError('No class has both annotated positive and negative training nodes')
    norm=normalization(x,relations,splits=='train');tensor,r=tensor_inputs(x,relations,norm,splits)
    torch.manual_seed(0);torch.set_num_threads(4)
    model=RGNN(x.shape[1],r,hidden=settings['hidden'],edge_dim=10,tasks=TASKS,
        message_mode=settings.get('message_mode','legacy'))
    optimizer=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=.0005)
    truth=torch.tensor(np.maximum(labels,0),dtype=torch.float32)
    weights={j:torch.tensor(float((labels[train[:,j],j]==0).sum()/max(1,(labels[train[:,j],j]==1).sum()))) for j in supported}
    best=None;best_score=None;best_epoch=None
    ranking_mask=val.copy();ranking_mask[:,[j for j in range(len(TASKS)) if j not in supported]]=False
    for epoch in range(settings['epochs']):
        model.train();optimizer.zero_grad();logits=model(tensor,r)
        loss=torch.stack([F.binary_cross_entropy_with_logits(logits[train[:,j],j],truth[train[:,j],j],pos_weight=weights[j]) for j in supported]).mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite supervised loss')
        loss.backward();optimizer.step();model.eval()
        with torch.no_grad():
            logits=model(tensor,r)
            measures=[F.binary_cross_entropy_with_logits(logits[val[:,j],j],truth[val[:,j],j]) for j in supported if val[:,j].any()]
            value=float(torch.stack(measures).mean()) if measures else float(loss)
        score=validation_score(labels,torch.sigmoid(logits).numpy(),ranking_mask,value,settings.get('selection_metric','bce'))
        if best_score is None or score>best_score:best_score=score;best=copy.deepcopy(model.state_dict());best_epoch=epoch+1
        if settings.get('progress') and (epoch==0 or (epoch+1)%25==0 or epoch+1==settings['epochs']):
            print(f'GNN epoch {epoch+1}/{settings["epochs"]}: training loss {float(loss.detach()):.4f}; validation loss {value:.4f}',flush=True)
    model.load_state_dict(best);model.eval()
    with torch.no_grad():prob=torch.sigmoid(model(tensor,r)).numpy()
    heads={};evaluation={}
    for j,task in enumerate(TASKS):
        counts={s:{'positive':int(((splits==s)&known[:,j]&(labels[:,j]==1)).sum()),
                   'negative':int(((splits==s)&known[:,j]&(labels[:,j]==0)).sum())} for s in ('train','validation','test')}
        threshold=None
        if j in supported and counts['validation']['positive']>=3 and counts['validation']['negative']>=3:
            options=[];mask=val[:,j]
            for t in np.unique(np.r_[.5,prob[mask,j]]):
                metrics=_metrics(labels[mask,j],prob[mask,j]>=t)
                if metrics['precision']>=.85 and int(((prob[mask,j]>=t)&(labels[mask,j]==1)).sum())>=3:
                    options.append((metrics['recall'],metrics['precision'],float(t)))
            if options:threshold=max(options)[2]
        available=threshold is not None
        heads[task]=dict(available=available,status='supervised_external' if available else 'insufficient_validation',
            positive_targets=counts['train']['positive'],negative_targets=counts['train']['negative'],threshold=threshold,
            reason='Annotation-supervised; validation precision >=0.85 on a selected external subset' if available else 'Needs supported training and validation positives/negatives',counts=counts)
        if available:
            test=(splits=='test')&known[:,j]
            evaluation[task]=dict(threshold=threshold,**_metrics(labels[test,j],prob[test,j]>=threshold),
                positive=counts['test']['positive'],negative=counts['test']['negative'],
                status='held_out_annotated_subset' if set(labels[test,j])=={0,1} else 'insufficient_test_support')
    return dict(model=model,normalization=norm,heads=heads,evaluation=evaluation,best_epoch=best_epoch,best_validation_score=best_score,
                probabilities=prob,settings=settings)


def archive_selection(archive,counts):
    """Select a bounded class-balanced subset, with no parent-scene leakage.

    This is a selected challenge subset, not a claim to the full benchmark score.
    Selection uses annotation coverage only, never predictions or performance.
    """
    with zipfile.ZipFile(archive) as z:
        train=[Path(n).name for n in z.namelist() if n.startswith('train/2013/images/') and n.endswith('.png')]
        test=[Path(n).name for n in z.namelist() if n.startswith('test/2013/images/') and n.endswith('.png')]
        if set(map(parent_scene,train))&set(map(parent_scene,test)):raise ValueError('Official train/test scenes overlap')
        if len(set(map(parent_scene,train)))!=70 or len(set(map(parent_scene,test)))!=30:
            raise ValueError('Unexpected official LsSCD-Ex scene split')
        training,validation=split_training_names(train)
        output=[]
        for split,names in [('train',training),('validation',validation),('test',test)]:
            official='test' if split=='test' else 'train';summaries=[]
            for name in sorted(names):
                masks=[canonical_labels(np.asarray(Image.open(io.BytesIO(z.read(f'{official}/{y}/semantic_masks/{name}'))))) for y in (2013,2015)]
                a,b=masks;signature=[]
                for klass in KNOWN_CLASSES:
                    for current,other in ((b,a),(a,b)):
                        changed=int(((current==klass)&(other!=klass)).sum());stable=int(((current==klass)&(other==klass)).sum())
                        signature.extend([int(changed>=100),int(stable>=100)])
                summaries.append((name,np.array(signature),int((a!=b).sum())))
            seen=np.zeros(32);used=set()
            for _ in range(counts[split]):
                choices=[s for s in summaries if parent_scene(s[0]) not in used]
                if not choices:break
                selected=max(choices,key=lambda s:(float((s[1]/(1+seen)).sum()),min(s[2],12000),s[0]))
                name,signature,changed=selected;used.add(parent_scene(name));seen+=signature
                output.append(dict(split=split,official_split=official,name=name,parent_scene=parent_scene(name),
                    changed_annotation_pixels=changed,selection='greedy annotation-class coverage; one chip per parent'))
        return output


def build_training_arrays(archive,output,selection,root,*,include_false_class_negatives=False):
    """Use the same full segmentation and graph methods as deployment."""
    from .contracts import RunContext,YearInput,TileInput,GridSpec
    from .models import load_models
    from .segmentation import segment_tile
    from .graphs import build_tile_graph,combine_year_graphs
    from .temporal import build_temporal_graph
    from .storage import load_segmentation,verify_manifest
    from .runner import _graph
    output=Path(output);grid=GridSpec(None,(1,0,0,0,1,0),'pixels',None)
    options=dict(project_root=str(root),device='cpu',coordinate_units='pixels',inference_window=512,
        inference_overlap=128,tta=True,max_comparison_pixels=4_000_000)
    initial=RunContext(output,(),options,{},'external_annotation_training');models=load_models(initial)
    arrays=[];schemas=[];quality=[]
    with zipfile.ZipFile(archive) as z:
        for number,item in enumerate(selection,1):
            print(f'External pair {number}/{len(selection)}: {item["split"]} {item["name"]}',flush=True)
            pair=output/'pairs'/item['split']/Path(item['name']).stem;years=[];truth=[]
            for year in (2013,2015):
                prefix=f'{item["official_split"]}/{year}';path=pair/'inputs'/f'{year}.png';path.parent.mkdir(parents=True,exist_ok=True)
                content=z.read(f'{prefix}/images/{item["name"]}')
                if path.is_file() and path.read_bytes()!=content:raise ValueError('Cached external image changed')
                if not path.is_file():path.write_bytes(content)
                pixels=np.asarray(Image.open(io.BytesIO(content)).convert('RGB'))
                truth.append(canonical_labels(np.asarray(Image.open(io.BytesIO(z.read(f'{prefix}/semantic_masks/{item["name"]}'))))))
                tile=TileInput(f'{year}/external.png',year,path,None,item['parent_scene'],(0,0),pixels.shape[:2],grid,pixels.shape[:2])
                years.append(YearInput(year,(tile,)))
            context=RunContext(pair,tuple(years),options,{},'external_annotation_training');annual=[]
            for year in years:
                tile=year.tiles[0];directory=pair/'segmentation'/str(tile.year)/tile.key
                if (directory/'artifact.json').is_file():verify_manifest(directory);seg=load_segmentation(directory/'artifact.json')
                else:seg=segment_tile(tile,context,models=models)
                graph=build_tile_graph(seg,'new',context);annual.append(combine_year_graphs([graph],context))
            graph=build_temporal_graph(annual,context);doc=read_json(graph.paths['graph'])
            labels,eligible=annotation_targets(doc,*truth,include_false_class_negatives=include_false_class_negatives)
            confusion=np.zeros((8,9),np.int64)
            for year,annotation in zip((2013,2015),truth):
                predicted=rasterize([(shape(n['geometry']),n['class_id']) for n in doc['nodes'] if n['year']==year],
                    out_shape=annotation.shape,transform=Affine.identity(),dtype='uint8')
                confusion+=np.bincount((annotation.astype(np.int64)*9+predicted).ravel(),minlength=81).reshape(9,9)[1:]
            quality.append(dict(split=item['split'],name=item['name'],confusion=confusion.tolist()))
            with np.load(graph.paths['tensors'],allow_pickle=False) as data:
                # Graph document and tensor rows are checked before assigning GT.
                if list(data['node_ids'])!=[n['id'] for n in doc['nodes']]:raise ValueError('External target node ordering mismatch')
                arrays.append(dict(x=data['x'],ei=data['edge_index'],ea=data['edge_attr'],et=data['edge_type'],
                    labels=labels,eligible=eligible,split=np.full(len(labels),item['split']),
                    group=np.full(len(labels),item['split']+':'+item['parent_scene'])))
            schemas.append(read_json(graph.paths['features']))
            atomic_json(pair/'supervision.json',dict(dataset=DATASET_URL,selection=item,
                annotation_mapping=LABEL_MAP.tolist(),labels=labels.tolist(),eligible=eligible.tolist(),
                image_sha256=[sha256(y.tiles[0].image) for y in years],
                target_definition='>=70% class replacement; <=5% replacement is unchanged; other cases masked',
                include_false_class_negatives=include_false_class_negatives))
    if any(s!=schemas[0] for s in schemas):raise ValueError('Mixed external graph schemas')
    offset=0;indices=[]
    for a in arrays:indices.append(a['ei']+offset);offset+=len(a['x'])
    combined={key:np.concatenate([a[key] for a in arrays]) for key in ('x','ea','et','labels','eligible','split','group')}
    combined['ei']=np.concatenate(indices,axis=1)
    np.savez_compressed(output/'training_arrays.npz',**combined);atomic_json(output/'feature_spec.json',schemas[0])
    atomic_json(output/'segmentation_quality.json',{split:semantic_metrics(sum(
        (np.asarray(q['confusion']) for q in quality if q['split']==split),np.zeros((8,9),np.int64)))
        for split in ('train','validation','test')})
    return combined,schemas[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selection-manifest',type=Path,help='Verified bounded archive produced by scripts/download_lsscd_subset.py')
    parser.add_argument('--train-pairs',type=int,default=24);parser.add_argument('--validation-pairs',type=int,default=10)
    parser.add_argument('--test-pairs',type=int,default=10);parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--message-mode',choices=['legacy','temporal_difference.v1'],default='legacy')
    parser.add_argument('--selection-metric',choices=['bce','macro_ap'],default='macro_ap')
    parser.add_argument('--include-false-class-negatives',action='store_true',
        help='Teach rejection when <=5%% of a predicted region has its current class in complete reference annotation')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if min(args.train_pairs,args.validation_pairs,args.test_pairs,args.epochs)<1:parser.error('Counts and epochs must be positive')
    checksum=hashlib.md5()
    with args.archive.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):checksum.update(block)
    if args.selection_manifest:
        downloaded=read_json(args.selection_manifest)
        if downloaded.get('dataset')!=DATASET_URL or downloaded.get('subset_sha256')!=sha256(args.archive):
            raise ValueError('External subset integrity/provenance mismatch')
        selection=downloaded['selection']
        parents={s:{item['parent_scene'] for item in selection if item['split']==s} for s in ('train','validation','test')}
        if any(parents[a]&parents[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
            raise ValueError('External training scenes overlap')
    else:
        if checksum.hexdigest()!=ARCHIVE_MD5:raise ValueError('Official dataset archive MD5 mismatch')
        selection=archive_selection(args.archive,dict(train=args.train_pairs,validation=args.validation_pairs,test=args.test_pairs))
    manifest=dict(dataset=DATASET_URL,archive_md5=ARCHIVE_MD5,license='CC BY-NC-SA 4.0; academic use only; Google Earth terms apply',
        archive_md5_verified=not bool(args.selection_manifest),subset_sha256=sha256(args.archive),
        selection=selection,include_false_class_negatives=args.include_false_class_negatives,
        scope='Selected external optical-image subset; not the full benchmark and not user-imagery accuracy')
    atomic_json(args.output/'dataset_manifest.json',manifest)
    data,schema=build_training_arrays(args.archive,args.output,selection,Path(__file__).resolve().parents[3],
        include_false_class_negatives=args.include_false_class_negatives)
    relations={name:(data['ei'][:,data['et']==j],data['ea'][data['et']==j]) for j,name in enumerate(RELATIONS)}
    settings=dict(hidden=64,edge_dim=10,epochs=args.epochs,seed=0,progress=True,message_mode=args.message_mode,selection_metric=args.selection_metric)
    trained=fit_supervised(data['x'],relations,data['labels'],data['eligible'],data['split'],settings)
    from .gnn import save_gnn_checkpoint
    checkpoint=args.output/'model.pt'
    save_gnn_checkpoint(checkpoint,trained['model'],schema,trained['normalization'],settings)
    metadata=read_json(checkpoint.with_suffix('.json'))
    metadata.update(target_schema=CHANGE_SCHEMA,task_order=list(TASKS),relation_profile=RELATION_PROFILE,
        head_status=trained['heads'],target='Official annotation-supervised major class gains/losses',
        external_evaluation=trained['evaluation'],
        inference_eligibility='observed_known_class_node.v1',
        dataset=manifest,validation_policy='Threshold selected only on validation nodes, precision >=.85 and >=3 positives',
        domain='Optical RGB; external 0.6 m dataset; pixel-unit graphs; transfer accuracy unverified')
    atomic_json(checkpoint.with_suffix('.json'),metadata)
    report=dict(checkpoint=str(checkpoint.resolve()),head_status=trained['heads'],test_metrics=trained['evaluation'],
        segmentation_quality=read_json(args.output/'segmentation_quality.json'),
        best_epoch=trained['best_epoch'],dataset=manifest,
        counts=Counter(data['split'].tolist()),independent_of_user_images=True,user_imagery_accuracy='not measured',
        evaluation_unit=('Annotation-assessed predicted nodes including false-class negatives' if args.include_false_class_negatives
            else 'Annotation-agreeing predicted graph nodes')+' on selected held-out parents; not pixel segmentation accuracy')
    atomic_json(args.output/'training_report.json',report)
    print('Training complete: '+str(checkpoint.resolve()),flush=True)
    print({task:head['available'] for task,head in trained['heads'].items()},flush=True)


if __name__=='__main__':main()
