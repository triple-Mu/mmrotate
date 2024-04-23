_base_ = [
    './_base_/default_runtime.py', './_base_/schedule_3x.py',
    './_base_/dota_rr.py'
]
checkpoint = 'https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_l_syncbn_fast_8xb32-300e_coco/yolov6_l_syncbn_fast_8xb32-300e_coco_20221109_183156-91e3c447.pth'  # noqa
custom_imports = dict(imports=['mmyolo.models'], allow_failed_imports=True)
num_classes = 6
angle_version = 'le90'
model = dict(
    _scope_='mmyolo',
    type='mmyolo.YOLODetector',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True,
        boxtype2tensor=False,
        batch_augments=None),
    backbone=dict(
        _scope_='mmyolo',
        type='mmyolo.YOLOv6EfficientRep',
        deepen_factor=1,
        widen_factor=1,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU'),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(
        _scope_='mmyolo',
        type='mmyolo.YOLOv6RepPAFPN',
        deepen_factor=1,
        widen_factor=1,
        in_channels=[256, 512, 1024],
        out_channels=[256, 256, 256],
        num_csp_blocks=12,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU'),
    ),
    bbox_head=dict(
        _scope_='mmrotate',
        type='mmrotate.RotatedRTMDetSepBNHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        angle_version=angle_version,
        anchor_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        use_hbbox_loss=False,
        scale_angle=False,
        loss_angle=None,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='ReLU')),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.DynamicSoftLabelAssigner',
            iou_calculator=dict(type='RBboxOverlaps2D'),
            topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000),
)

# batch_size = (2 GPUs) x (4 samples per GPU) = 8
train_dataloader = dict(batch_size=4, num_workers=4)
