import numpy as np
from mmdet.datasets.transforms.loading import \
    LoadAnnotations as MMDET_LoadAnnotations
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MMROTATE_LoadAnnotations(MMDET_LoadAnnotations):

    def _load_regressions(self, results: dict) -> None:
        gt_bboxes_regressions = []
        for instance in results.get('instances', []):
            gt_bboxes_regressions.append(instance['bbox_regression'])
        # TODO: Inconsistent with mmcv, consider how to deal with it later.
        results['gt_bboxes_regressions'] = np.array(
            gt_bboxes_regressions, dtype=np.float32)

    def transform(self, results: dict) -> dict:
        super().transform(results)
        self._load_regressions(results)
        return results
