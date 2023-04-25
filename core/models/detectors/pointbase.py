import time
import copy
import torch
from pdb import set_trace as bp
from core.models.registry import HEAD, MODELS
from torch import nn

def coor_to_point(coords, detection_range, voxel_size):

    coords = coords.float()
    coords[:,0] = (coords[:,0] * voxel_size[0]) + detection_range[0]
    coords[:,1] = (coords[:,1] * voxel_size[1]) + detection_range[1]
    coords[:,2] = (coords[:,2] * voxel_size[2]) + detection_range[2]
    points = coords
    return points


def voxel_sampling(point, detection_range):

    # detection_range = [-80, -80, -3, 80, 80, 3]
    voxel_step = [0.34, 0.34, 0.34]
    # point = point[0]

    point[:, 0] = (point[:, 0] - detection_range[0]) / voxel_step[0]
    point[:, 1] = (point[:, 1] - detection_range[1]) / voxel_step[1]
    point[:, 2] = (point[:, 2] - detection_range[2]) / voxel_step[2]

    coords = point.int()

    n_coor, inverse = torch.unique(coords, return_inverse=True, dim=0)
    sampling_point = coor_to_point(n_coor, detection_range, voxel_step)

    return sampling_point


@MODELS.register_module()
class cpupointbase(nn.Module):

    def __init__(self, config_box):

        super(cpupointbase, self).__init__()
        # Take configuration
        config = config_box["model_config"]
        data_config = config_box["data_config"]
        grid_size = config_box["grid_size"]

        self.output_shape = [30, 800, 800]

        self.num_classes = config["num_classes"]

        self.name_to_id = data_config["CLASS_ENCODE"]
        self.id_to_name = dict([(v, k) for k, v in self.name_to_id.items()])

        self.detection_range = data_config["POINT_CLOUD_RANGE"]
        self.post_center_range = copy.copy(self.detection_range)

        self.layer_0_0 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

        self.layer_0_1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

        self.training = False

        self._nms_score_thresholds = [0.2, 0.2, 0.2]
        self._use_rotate_nms = True
        self._nms_pre_max_sizes = [1000, 1000, 1000]
        self._nms_post_max_sizes = [1200, 1200, 1200]
        self._nms_iou_thresholds = [0.01, 0.01, 0.01]

        self.loc_loss = config_box["loc_loss"]
        self.cls_loss = config_box["cls_loss"]

    def forward(self, data):

        # points = data["show_points"][0]
        features = data["sparse_voxels"]
        coors = data["coordinates"]
        num_voxels = data["num_points"]

        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        points = points_mean.squeeze().cpu()
        # bp()
        for i in range(10):
            now = time.time()
            center_point = voxel_sampling(points, self.detection_range)
            print((time.time() - now) * 1000.0)
        # bp()
        loss = {}
        return loss
