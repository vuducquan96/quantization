import torch
from core.ops.iou3d import nms_gpu
from pdb import set_trace as bp


def rotate_nms_gpu(rbboxes,
                   scores,
                   pre_max_size=None,
                   post_max_size=None,
                   iou_threshold=0.5):
    """Nms function with gpu implementation.    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (int): Threshold.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.    Returns:
        torch.Tensor: Indexes after nms.
    """

    new_rbboxes = torch.zeros_like(rbboxes)
    new_rbboxes[:, 0] = rbboxes[:, 0] - (rbboxes[:, 2] / 2)
    new_rbboxes[:, 1] = rbboxes[:, 1] - (rbboxes[:, 3] / 2)
    new_rbboxes[:, 2] = rbboxes[:, 0] + (rbboxes[:, 2] / 2)
    new_rbboxes[:, 3] = rbboxes[:, 1] + (rbboxes[:, 3] / 2)
    new_rbboxes[:, 4] = rbboxes[:, 4]
    #new_rbboxes = new_rbboxes[torch.argsort(scores, descending=True)]
    #bp()
    res = nms_gpu(new_rbboxes, scores, iou_threshold, pre_maxsize=pre_max_size, post_max_size=post_max_size)
    return res
