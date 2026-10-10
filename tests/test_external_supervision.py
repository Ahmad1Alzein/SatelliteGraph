import numpy as np
from shapely.geometry import box,mapping


def test_annotation_mapping_preserves_bare_ground_zero():
    from pipelines.semantic_temporal.runtime.external_training import canonical_labels
    np.testing.assert_array_equal(canonical_labels(np.arange(8,dtype=np.uint8)),[6,4,7,2,3,8,5,1])


def test_official_labels_ignore_size_variations_and_segmentation_errors():
    from pipelines.semantic_temporal.runtime.external_training import annotation_targets
    before=np.full((50,60),4,np.uint8);after=before.copy()
    before[5:20,5:20]=1;after[5:20,5:18]=1
    after[25:40,30:45]=1
    after[5:20,40:55]=2
    nodes=[dict(id='stable',year=2013,class_id=1,geometry=mapping(box(5,5,20,20))),
        dict(id='new',year=2015,class_id=1,geometry=mapping(box(30,25,45,40))),
        dict(id='wrong',year=2015,class_id=1,geometry=mapping(box(40,5,55,20)))]
    labels,eligible=annotation_targets({'years':[2013,2015],'nodes':nodes},before,after)
    from pipelines.semantic_temporal.runtime.change_contracts import TASKS
    assert labels[0,TASKS.index('building_loss')]==-1  # small shrinkage is ambiguous
    assert labels[1,TASKS.index('building_gain')]==1
    assert not eligible[2].any()  # predicted building is annotated road
    assert not eligible[:,TASKS.index('road_gain')].any()  # no road node, never fabricate road labels


def test_patch_split_keeps_parent_scenes_together():
    from pipelines.semantic_temporal.runtime.external_training import parent_scene,split_training_names
    names=['scene_a_1_0.png','scene_a_1_1.png','scene_b_2_0.png','scene_c_3_0.png','scene_d_4_0.png']
    train,val=split_training_names(names)
    assert {parent_scene(n) for n in train}.isdisjoint({parent_scene(n) for n in val})
    assert parent_scene('20_71.png')==parent_scene('63_71.png')=='71'


def test_test_labels_never_affect_training_or_thresholds():
    import torch
    from pipelines.semantic_temporal.runtime.external_training import fit_supervised
    from pipelines.semantic_temporal.runtime.change_contracts import TASKS,RELATIONS
    x=np.random.default_rng(4).normal(size=(18,3)).astype(np.float32)
    relations={name:(np.empty((2,0),np.int64),np.empty((0,10),np.float32)) for name in RELATIONS}
    labels=np.full((18,len(TASKS)),-1,np.int64);labels[:,0]=np.tile([0,1],9)
    eligible=labels>=0;splits=np.repeat(['train','validation','test'],6)
    settings=dict(hidden=8,epochs=2,edge_dim=10)
    a=fit_supervised(x,relations,labels,eligible,splits,settings)
    labels[12:,0]=1-labels[12:,0]
    b=fit_supervised(x,relations,labels,eligible,splits,settings)
    np.testing.assert_allclose(a['normalization']['node_mean'],x[:6].mean(0))
    assert a['heads']==b['heads']
    for name,value in a['model'].state_dict().items():assert torch.equal(value,b['model'].state_dict()[name])


def test_semantic_quality_counts_unknown_predictions_as_errors():
    from pipelines.semantic_temporal.runtime.external_training import semantic_metrics
    confusion=np.zeros((8,9),int);confusion[5,0]=100
    metrics=semantic_metrics(confusion)
    assert metrics['pixel_accuracy']==0
    assert metrics['class_iou']['bare_soil']==0
    assert metrics['unknown_prediction_pixels']==100
    assert metrics['class_iou']['building'] is None


def test_validation_ranking_can_prefer_rare_changes_over_constant_background():
    from pipelines.semantic_temporal.runtime.external_training import validation_score
    truth=np.zeros((100,1),int);truth[:3]=1;mask=np.ones(truth.shape,bool)
    constant=np.full(truth.shape,.03);ranked=np.full(truth.shape,.3);ranked[:3]=.4
    def bce(p):return float(-(truth*np.log(p)+(1-truth)*np.log(1-p)).mean())
    assert bce(constant)<bce(ranked)
    assert validation_score(truth,ranked,mask,bce(ranked),'macro_ap')>validation_score(truth,constant,mask,bce(constant),'macro_ap')


def test_false_class_hard_negatives_cover_every_class_and_both_directions():
    from pipelines.semantic_temporal.runtime.external_training import annotation_targets
    from pipelines.semantic_temporal.runtime.change_contracts import TASKS,KNOWN_CLASSES
    for klass,name in KNOWN_CLASSES.items():
        other=1 if klass!=1 else 2
        truth=np.full((30,30),other,np.uint8)
        nodes=[dict(id=str(year),year=year,class_id=klass,geometry=mapping(box(5,5,25,25))) for year in (2013,2015)]
        labels,eligible=annotation_targets(dict(years=[2013,2015],nodes=nodes),truth,truth,include_false_class_negatives=True)
        for index,direction in enumerate(('loss','gain')):
            task=TASKS.index(name+'_'+direction)
            assert labels[index,task]==0 and eligible[index,task]
            assert eligible[index].sum()==1


def test_hard_negatives_do_not_turn_unknown_or_mixed_truth_into_unchanged():
    from pipelines.semantic_temporal.runtime.external_training import annotation_targets
    from pipelines.semantic_temporal.runtime.change_contracts import TASKS
    before=np.full((30,30),2,np.uint8);after=before.copy()
    node=dict(id='building',year=2015,class_id=1,geometry=mapping(box(5,5,25,25)))
    doc=dict(years=[2013,2015],nodes=[node])
    after[5:25,5:25]=0
    labels,eligible=annotation_targets(doc,before,after,include_false_class_negatives=True)
    assert labels[0,TASKS.index('building_gain')]==-1 and not eligible.any()
    after[5:25,5:25]=2;after[5:15,5:25]=1
    labels,eligible=annotation_targets(doc,before,after,include_false_class_negatives=True)
    assert labels[0,TASKS.index('building_gain')]==-1 and not eligible.any()
