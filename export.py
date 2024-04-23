# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn

from mmdet.apis import init_detector

from mmrotate.utils import register_all_modules

MERGE_STRIDE = False
MERGE_MEAN_STD = False
FP16 = False


class ExportNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        if MERGE_STRIDE:
            strides = self.model.bbox_head.prior_generator.strides
            rtm_regs = self.model.bbox_head.rtm_reg
            for stride, rtm_reg in zip(strides, rtm_regs):
                rtm_reg.bias.data.copy_(rtm_reg.bias.data + math.log(stride[0]))

    def bbox_head_forward(self, feats) -> tuple:
        _self = self.model.bbox_head
        cls_scores = []
        bbox_preds = []
        angle_preds = []
        for idx, (x, stride) in enumerate(
                zip(feats, _self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x

            for cls_layer in _self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_score = _self.rtm_cls[idx](cls_feat)

            for reg_layer in _self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)

            if _self.exp_on_reg:
                # reg_dist = _self.rtm_reg[idx](reg_feat)
                if MERGE_STRIDE:
                    reg_dist = _self.rtm_reg[idx](reg_feat).exp()
                else:
                    reg_dist = _self.rtm_reg[idx](reg_feat).exp() * stride[0]
            else:
                reg_dist = _self.rtm_reg[idx](reg_feat) * stride[0]

            angle_pred = _self.rtm_ang[idx](reg_feat)

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            angle_preds.append(angle_pred)
        return tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds)

    def forward(self, x):
        # n,h,w,c
        if FP16:
            x = x.half()
        if MERGE_MEAN_STD:
            x = x - (self.model.data_preprocessor.mean.permute(1, 2, 0))
            x = x / (self.model.data_preprocessor.std.permute(1, 2, 0))
            x = x.permute([0, 3, 1, 2]).contiguous()
        # x = x - self.model.data_preprocessor.mean
        # x = x / self.model.data_preprocessor.std
        x = self.model.extract_feat(x)
        if MERGE_STRIDE:
            x = self.bbox_head_forward(x)
        else:
            x = self.model.bbox_head(x)
        results = []
        for conf, box, angle in zip(*x):
            conf = conf.sigmoid()
            cat = torch.cat([conf, box, angle], dim=1)
            cat = cat.permute(0, 2, 3, 1)
            if FP16:
                cat = cat.float()
            results.append(cat)
        return tuple(results)


@torch.inference_mode()
def main():
    # register all modules in mmrotate into the registries
    register_all_modules()

    config = 'work_dirs/rotated_rtmdet_s-300e-aug/rotated_rtmdet_s-300e-aug.py'
    # checkpoint = 'epoch_300.pth'
    checkpoint = 'work_dirs/rotated_rtmdet_s-300e-aug/epoch_294.pth'
    f = 'rotated_rtmdet_s-294-db'
    if MERGE_STRIDE:
        f += '-mstride'
    if MERGE_MEAN_STD:
        f += '-mmeanstd'
    if FP16:
        f += '-fp16'
    else:
        f += '-fp32'
    f += '.onnx'
    # build the model from a config file and a checkpoint file
    model = init_detector(
        config, checkpoint, palette='dota', device='cuda:0')

    export_net = ExportNet(model)
    export_net.eval()
    if FP16:
        export_net.half()

    x = torch.randn((1, 1024, 1024, 3) if MERGE_MEAN_STD else (1, 3, 1024, 1024), device='cuda:0')
    export_net(x)

    torch.onnx.export(
        export_net,
        x,
        f,
        opset_version=17,
        input_names=['image'],
        output_names=['large', 'medium', 'small'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'large': {0: 'batch_size'},
            'medium': {0: 'batch_size'},
            'small': {0: 'batch_size'}
        }
    )


if __name__ == '__main__':
    main()
