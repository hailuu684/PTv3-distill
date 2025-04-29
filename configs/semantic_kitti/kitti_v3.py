weight = None
resume = True
evaluate = True
test_only = False
seed = 28024989
save_path = 'exp/dataset_type/kitti/kitti-train-teacher'
miou_result_path = f"/home/dingz@bgsu.edu/ptv3/ptv3_distill/PTv3-distill/{save_path}/miou_result"

# num_worker = 2 #run on muti gpu
# batch_size = 24  # takes 79 Gb RAMs on A100
# batch_size_val = 24
num_worker = 2 #run on muti gpu muti nodes
batch_size = 24  # takes 79 Gb RAMs on A100
batch_size_val = 24
batch_size_test = None
epoch = 50
eval_epoch = 50
##############1
clip_grad = None
sync_bn = False
##############2
sync_bn = False
enable_amp = True
empty_cache = False
###########1
empty_cache_per_epoch = False
###########2
find_unused_parameters = True
mix_prob = 0.8
param_dicts = [dict(keyword='block', lr=0.0002)]

hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=1),
    dict(type='PreciseEvaluator', test_last=False, keywords="module.", replacement="")
]


train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)




model = dict(
    type='DefaultSegmentorV2',
    num_classes=19,
    backbone_out_channels=64,
    backbone=dict(
        type='PT-v3-train-teacher',
        in_channels=4,
        order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=('nuScenes', 'SemanticKITTI', 'Waymo')
    ),
    criteria=[dict(
        type='CrossEntropyLoss',
        weight=[
            3.1557, 8.7029, 7.8281, 6.1354, 6.3161, 7.9937, 8.9704,
            10.1922, 1.6155, 4.2187, 1.9385, 5.5455, 2.0198, 2.6261,
            1.3212, 5.1102, 2.5492, 5.8585, 7.3299
        ],
        loss_weight=1.0,
        ignore_index=-1,
    ),
    dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1),]
)







optimizer = dict(type='AdamW', lr=0.002, weight_decay=0.005)
scheduler = dict(
    type='OneCycleLR',
    max_lr=optimizer["lr"],
    pct_start=0.04,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0)
dataset_type = 'SemanticKITTIDataset'
data_root = '/home/dingz@bgsu.edu/ptv3/ptv3_distill/PTv3-distill/data/semantic_kitti'
ignore_index = -1
names = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]
data = dict(
    num_classes=19,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        # split=["train", "val"],
        split="train",
        data_root=data_root,
        transform=[
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'strength', 'segment'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('coord', 'strength'))
        ],
        test_mode=False,
        ignore_index=ignore_index,
        loop=1 # why loop = 1?
        ),
    val=dict(
        type=dataset_type,
        split='val',
        data_root=data_root,
        transform=[
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'strength', 'segment'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('coord', 'strength'))
        ],
        test_mode=False,
        ignore_index=ignore_index),
    test=dict(
        type=dataset_type,
        split='val',
        data_root=data_root,
        # transform=[
        #     dict(type='Copy', keys_dict=dict(segment='origin_segment')),
        #     dict(
        #         type='GridSample',
        #         grid_size=0.025,
        #         hash_type='fnv',
        #         mode='train',
        #         keys=('coord', 'strength', 'segment'),
        #         return_inverse=True)
        # ],
        transform=[], ## in the examples of semantic_kitti dataset, the original v2 config sets this in empty []
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True,
                keys=('coord', 'strength')),
            crop=None,
            post_transform=[
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index'),
                    feat_keys=('coord', 'strength'))
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
            ],
                           ),
        ignore_index=ignore_index))