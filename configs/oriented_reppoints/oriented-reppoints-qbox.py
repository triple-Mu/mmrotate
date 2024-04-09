_base_ = ['../_base_/default_runtime.py']

dataset_type = 'DOTADataset'
data_root = 'data/DOTA/'
file_client_args = dict(backend='disk')

metainfo = {
    'classes': ('A', 'B', 'C', 'D', 'E', 'F'),
    'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                (138, 43, 226), (255, 128, 0)]
}

batch_size = 16
num_workers = 8

lr = 0.001
max_epochs = 300

load_from = None
resume = False

norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='mmdet.RepPointsDetector',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='OrientedRepPointsHead',
        num_classes=6,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.3,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=2,
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='ConvexGIoULoss', loss_weight=0.375),
        loss_bbox_refine=dict(type='ConvexGIoULoss', loss_weight=1.0),
        loss_spatial_init=dict(type='SpatialBorderLoss', loss_weight=0.05),
        loss_spatial_refine=dict(type='SpatialBorderLoss', loss_weight=0.1),
        init_qua_weight=0.2,
        top_ratio=0.4),
    # training and testing settings
    train_cfg=dict(
        init=dict(
            assigner=dict(type='ConvexAssigner', scale=4, pos_num=1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=dict(
            assigner=dict(
                type='MaxConvexIoUAssigner',
                pos_iou_thr=0.1,
                neg_iou_thr=0.1,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.4),
        max_per_img=2000))

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=num_workers > 0,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annfiles/',
        data_prefix=dict(img_path='train/images/'),
        img_shape=(1024, 1024),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=num_workers // 2,
    persistent_workers=num_workers > 0,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        img_shape=(1024, 1024),
        test_mode=True,
        pipeline=val_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    # dict(
    #     type='LinearLR',
    #     start_factor=1.0 / 3,
    #     by_epoch=False,
    #     begin=0,
    #     end=500),
    # dict(
    #     type='MultiStepLR',
    #     begin=0,
    #     end=40,
    #     by_epoch=True,
    #     milestones=[24, 32, 38],
    #     gamma=0.1)
    dict(
        type='CosineAnnealingLR',
        eta_min=lr * 0.01,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=lr, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

_base_.default_hooks['checkpoint'] = dict(
    type='CheckpointHook',
    interval=1,
    by_epoch=True,
    max_keep_ckpts=10,
    save_best='dota/mAP',
    rule='greater',
)
