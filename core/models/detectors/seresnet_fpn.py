import copy
import numpy as np
from torch import nn
from numpy import log
from pdb import set_trace as bp
from core.models.registry import HEAD, MODELS
from configs.system_config import *
from core.gaussian import gaussian_radius, draw_heatmap_gaussian
import math
import torch
from torch import nn
from segmentation_models_pytorch.base import modules as md
from torch.nn import functional as F
from core.ops.iou3d_nms import iou3d_nms_utils

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel//reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel//reduction, channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResidualUnit(nn.Module):
    def __init__(self, n_input, n_output, downsample=False):
        """
        Residual Unit consisting of two convolutional layers and an identity mapping
        :param n_input: number of input channels
        :param n_output: number of output channels
        :param downsample: downsample the output by a factor of 2
        """
        super(ResidualUnit, self).__init__()
        self.conv1 = nn.Conv2d(n_input, n_output, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(n_output, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_output, n_output, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(n_output, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # down-sampling: use stride two for convolutional kernel and create 1x1 kernel for down-sampling of input
        self.downsample = None
        if downsample:
            self.conv1 = nn.Conv2d(n_input, n_output, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.downsample = nn.Sequential(nn.Conv2d(n_input, n_output, kernel_size=(1, 1), stride=(2, 2), bias=False),
                                            nn.BatchNorm2d(n_output, eps=1e-05, momentum=0.1, affine=True,
                                                           track_running_stats=True))
        else:
            self.identity_channels = nn.Conv2d(n_input, n_output, kernel_size=(1, 1), bias=False)

    def forward(self, x):

        # store input for skip-connection
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # downsample input to match output dimensions
        if self.downsample is not None:
            identity = self.downsample(identity)
        else:
            identity = self.identity_channels(identity)

        # skip-connection
        x += identity

        # apply ReLU activation
        x = self.relu(x)

        return x

class ResidualBlock(nn.Module):
    """
        Residual Block containing specified number of residual layers
        """

    def __init__(self, n_input, n_output, n_res_units):
        super(ResidualBlock, self).__init__()

        # use down-sampling only in the first residual layer of the block
        first_unit = True

        # specific channel numbers
        if n_res_units == 3:
            inputs = [n_input, n_output//4, n_output//4]
            outputs = [n_output//4, n_output//4, n_output]
        else:
            inputs = [n_input, n_output // 4, n_output // 4, n_output // 4, n_output // 4, n_output]
            outputs = [n_output // 4, n_output // 4, n_output // 4, n_output // 4, n_output, n_output]

        # create residual units
        units = []
        for unit_id in range(n_res_units):
            if first_unit:
                units.append(ResidualUnit(inputs[unit_id], outputs[unit_id], downsample=True))
                first_unit = False
            else:
                units.append(ResidualUnit(inputs[unit_id], outputs[unit_id]))
        self.res_block = nn.Sequential(*units)

    def forward(self, x):
        x = self.res_block(x)
        return x


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x

class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(policy))
        self.policy = policy

    def forward(self, x):
        if self.policy == "add":
            return sum(x)
        elif self.policy == "cat":
            return torch.cat(x, dim=1)
        else:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy))


class FPNDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        encoder_depth=5,
        pyramid_channels=256,
        segmentation_channels=128,
        dropout=0.2,
        merge_policy="add",
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[: encoder_depth + 1]

        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

    def forward(self, features):
        # A = features[-4:]
        c1, c2, c3, c4, c5 = features[-5:]

        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)
        return p2

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        blocks = []
        self.config =[{"res": [32, 48, 3],
                       "se": 48},
                      {"res": [48, 64, 3],
                       "se": 64},
                      {"res": [64, 128, 3],
                       "se": 128},
                      {"res": [128, 256, 3],
                       "se": 256}]
        self.base_block = nn.Sequential(
                nn.Conv2d(in_channels=30, out_channels=32, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.BatchNorm2d(32)
            )

        for module_config in self.config:
            block = []
            for k, v in module_config.items():

                if "res" in k:
                    mod = ResidualBlock(n_input=v[0], n_output=v[1], n_res_units=v[2])
                if "se" in k:
                    mod = SELayer(channel = v)
                block.append(mod)
            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.decoder = FPNDecoder(
            encoder_channels=[48, 64, 128, 256],
            encoder_depth=4,
            pyramid_channels=128,
            segmentation_channels=128,
            dropout=0.2,
            merge_policy="add",
        )
        self._num_anchor_per_loc = 1
        self._num_class = 3
        self._box_code_size = 6

        self.conv_cls = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=True),
        )

        self.conv_box = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 6, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):

        x = self.base_block(x)
        features = [x]
        for mod_id in range(len(self.blocks)):
            # x = mod(x)
            x = self.blocks[mod_id](x)
            features.append(x)
        out = self.decoder(features)

        box_preds = self.conv_box(out)
        cls_preds = self.conv_cls(out)
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   self._box_code_size, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()

        pred = {}
        pred["cls_preds"] = cls_preds
        pred["box_preds"] = box_preds

        return pred

def from_2d_box_to_3d(dd_box):
    ddd_box = torch.zeros((dd_box.shape[0], 7)).float()
    ddd_box[:,0] = dd_box[:, 0]
    ddd_box[:,1] = dd_box[:, 1]
    ddd_box[:,3] = dd_box[:, 2]
    ddd_box[:,4] = dd_box[:, 3]
    ddd_box[:,6] = dd_box[:, 4]

    return ddd_box
def my_rotate_nms(box_preds_class, top_scores_class, top_labels_class, iou_thres, top_object=2000):
    order = top_scores_class.sort(descending=True)[1]
    box_preds_class = box_preds_class[order]
    top_scores_class = top_scores_class[order]
    top_labels_class = top_labels_class[order]
    # bp()
    box_preds_class_3d = from_2d_box_to_3d(box_preds_class)
    iou_matrix = iou3d_nms_utils.boxes_bev_iou_cpu(box_preds_class_3d, box_preds_class_3d)
    iou_matrix.fill_diagonal_(0.0)
    total_box = box_preds_class.shape[0]
    check_ = torch.zeros(total_box)
    selected_index = []
    for i in range(total_box):
        if check_[i] == 0:
            selected_index.append(i)
            mask = iou_matrix[i] > iou_thres
            check_[mask] = 1

    box_preds_class = box_preds_class[selected_index]
    top_scores_class = top_scores_class[selected_index]
    top_labels_class = top_labels_class[selected_index]
    return box_preds_class, top_scores_class, top_labels_class

@MODELS.register_module()
class seresnetfpn(nn.Module):

    def __init__(self, config_box):

        super(seresnetfpn, self).__init__()
        # Take configuration
        config = config_box["model_config"]
        data_config = config_box["data_config"]
        grid_size = config_box["grid_size"]
        try:
            self.attention = config.MODEL.attention
        except:
            self.attention = ""

        # Process data

        self.output_shape = [30, 800, 800]

        self.num_classes = config["num_classes"]

        self.name_to_id = data_config["CLASS_ENCODE"]
        self.id_to_name = dict([(v, k) for k, v in self.name_to_id.items()])

        self.detection_range = data_config["POINT_CLOUD_RANGE"]
        self.post_center_range = copy.copy(self.detection_range)

        self.out_size_factor = 4
        self.feature_x = int(np.ceil(grid_size[0] / self.out_size_factor))
        self.feature_y = int(np.ceil(grid_size[1] / self.out_size_factor))

        self.pre_define_vector = torch.zeros((self.feature_y, self.feature_x, 2))
        for i in range(self.feature_y):
            self.pre_define_vector[i,:,0] = torch.arange(0, self.feature_x, 1)

        for i in range(self.feature_x):
            self.pre_define_vector[:,i,1] = torch.arange(0, self.feature_y, 1)

        self.pre_define_vector = self.pre_define_vector.cuda()
        self.voxel_size = data_config["VOXEL_GENERATOR"]["VOXEL_SIZE"]
        self.gaussian_overlap = 0.1
        self.min_radius = 2
        self.max_objs = 600

        self.backbone = MyNet()

        self.training = False

        self._nms_score_thresholds = [0.2, 0.4, 0.3]
        self._use_rotate_nms = True
        self._nms_pre_max_sizes = [1000, 1000, 1000]
        self._nms_post_max_sizes = [1200, 1200, 1200]
        self._nms_iou_thresholds = [0.01, 0.01, 0.01]
        

        self.loc_loss = config_box["loc_loss"]
        self.cls_loss = config_box["cls_loss"]
        self.sparse_shape = [30, 800, 800]
    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor): Mask of the feature map with the shape
                of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def forward(self, data):

        voxel_features = data['voxels']
        pred = self.backbone(voxel_features)
        if self.training == False:
            rr = self.predict1(data, pred)
            return rr

        cls_preds = pred["cls_preds"].contiguous()
        box_preds = pred["box_preds"].contiguous()
        device = cls_preds.get_device()

        B, _, X, Y, _ = box_preds.shape
        heatmap_label = torch.zeros(cls_preds.shape).to(device)
        heatmap_ious_label = torch.zeros(cls_preds.shape).to(device)

        anno_box = torch.zeros((B, self.max_objs, 6)).to(device)
        index_matrix = torch.zeros((B, self.max_objs), dtype=torch.int64).to(device)
        masks_obj = torch.zeros((B, self.max_objs), dtype=torch.uint8).to(device)
        masks_class = torch.zeros((B, self.max_objs), dtype=torch.uint8).to(device)

        total_length_x = int(((self.detection_range[3] - self.detection_range[0]) / self.voxel_size[0]) / self.out_size_factor)
        total_length_y = int(((self.detection_range[4] - self.detection_range[1]) / self.voxel_size[1]) / self.out_size_factor)

        for batch_id in range(len(data["gt_boxes"])):
            single_gt_box = data["gt_boxes"][batch_id, :, :]
            filter_gt = np.sum(single_gt_box, axis=1)
            single_gt_box = single_gt_box[filter_gt != 0]
            single_gt_class = torch.from_numpy(single_gt_box[:, -1])
            single_gt_bbox = torch.from_numpy(single_gt_box[:,:-1])
            obj_id = 0

            for id_map in range(self.num_classes):
                index_cls = np.where(single_gt_class == (id_map + 1))[0]
                for idxx in index_cls:
                    bbox = single_gt_bbox[idxx]
                    x, y, z = bbox[0], bbox[1], bbox[2]
                    dx, dy = bbox[3], bbox[4]

                    dx = dx / self.voxel_size[0] / self.out_size_factor
                    dy = dy / self.voxel_size[1] / self.out_size_factor

                    coor_x = ((x - self.detection_range[0]) / self.voxel_size[0]) / self.out_size_factor
                    coor_y = ((y - self.detection_range[1]) / self.voxel_size[1]) / self.out_size_factor

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.to(torch.int32)
                    
                    radius = gaussian_radius((dy, dx), min_overlap=self.gaussian_overlap)
                    radius = max(self.min_radius, int(radius))
                    draw_gaussian = draw_heatmap_gaussian
                    draw_gaussian(heatmap_label[batch_id, 0, :, :, id_map], center_int, radius)

                    xx, yy = center_int[0], center_int[1]

                    obj_index = yy * Y + xx
                    index_matrix[batch_id, obj_id] = obj_index
                    masks_obj[batch_id, obj_id] = 1
                    masks_class[batch_id, obj_id] = id_map

                    rx, ry, rz = bbox[3], bbox[4], bbox[5]
                    rot = bbox[6]
                    anno_box[batch_id, obj_id, 0] = coor_x - xx
                    anno_box[batch_id, obj_id, 1] = coor_y - yy

                    anno_box[batch_id, obj_id, 2] = log(rx)
                    anno_box[batch_id, obj_id, 3] = log(ry)

                    anno_box[batch_id, obj_id, 4] = torch.sin(2 * rot)
                    anno_box[batch_id, obj_id, 5] = torch.cos(2 * rot)

                    obj_id += 1

        box_preds = box_preds.view(box_preds.size(0), -1, box_preds.size(-1)).contiguous()
        box_preds = self._gather_feat(box_preds, index_matrix)
        masks_obj = masks_obj.unsqueeze(2).expand_as(anno_box).float()

        loss_dict = {}
        loss_heat_map_1 = self.cls_loss(cls_preds[:, :, :, :, 0].clone(), heatmap_label[:, :, :, :, 0])
        loss_heat_map_2 = self.cls_loss(cls_preds[:, :, :, :, 1].clone(), heatmap_label[:, :, :, :, 1])
        loss_heat_map_3 = self.cls_loss(cls_preds[:, :, :, :, 2].clone(), heatmap_label[:, :, :, :, 2])

        cls_loss = loss_heat_map_1.sum() + loss_heat_map_2.sum() + loss_heat_map_3.sum()
        loc_loss = self.loc_loss(box_preds, anno_box, masks_obj).sum()

        loss = cls_loss + 2 * loc_loss
        loss_dict["loss"] = loss
        loss_dict["_cls_loss"] = cls_loss
        loss_dict["_loc_loss"] = loc_loss
        loss_dict["_iou_loss"] = 0

        return loss_dict
    def predict1(self, example, preds_dict):

        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        num_class_with_bg = self.num_classes

        predictions_dicts = []

        idd = -1
        for box_preds, cls_preds in zip(batch_box_preds, batch_cls_preds):
            
            idd += 1
            box_preds = box_preds.float()
            box_preds[0, :, :, 0] += self.pre_define_vector[:, :, 0]
            box_preds[0, :, :, 1] += self.pre_define_vector[:, :, 1]

            box_preds[0, :, :, 0] = box_preds[0, :, :, 0] * self.out_size_factor * self.voxel_size[0] + \
                                    self.detection_range[0]
            box_preds[0, :, :, 1] = box_preds[0, :, :, 1] * self.out_size_factor * self.voxel_size[1] + \
                                    self.detection_range[1]
            box_preds[0, :, :, 2] = torch.exp(box_preds[0, :, :, 2])
            box_preds[0, :, :, 3] = torch.exp(box_preds[0, :, :, 3])
            box_preds[0, :, :, 4] = torch.atan2(box_preds[0, :, :, 4], box_preds[0, :, :, 5]) / 2.0
            
            box_preds = box_preds[:, :, :, :-1]
            box_preds = box_preds.view(-1, 5)
            cls_preds = cls_preds.float()
            
            total_scores = torch.sigmoid(cls_preds)
            total_scores = total_scores.view(-1,3).float()


            if num_class_with_bg == 1:
                top_scores = total_scores.squeeze(-1)
                top_labels = torch.zeros(
                    total_scores.shape[0],
                    device=total_scores.device,
                    dtype=torch.long)
            else:
                top_scores, top_labels = torch.max(total_scores, dim=-1)

            # nnn = time.time()

            all_box_prediction = []
            all_score_prediction = []
            all_label_prediction = []
            
            for classid, (score_threshold, nms_pre, nms_post, iou_thres) in enumerate(
                    zip(self._nms_score_thresholds, self._nms_pre_max_sizes, self._nms_post_max_sizes,
                        self._nms_iou_thresholds)):
                
                labels_keep = (top_labels == classid)
                top_scores_class = top_scores.masked_select(labels_keep)
                top_labels_class = top_labels.masked_select(labels_keep)

                box_preds_class = box_preds[labels_keep]

                top_scores_keep = top_scores_class >= score_threshold
                top_scores_class = top_scores_class.masked_select(top_scores_keep)
                top_labels_class = top_labels_class.masked_select(top_scores_keep)

                box_preds_class = box_preds_class[top_scores_keep]

                #if classid == 0:
                #    area_filter = (box_preds_class[:, 2] * box_preds_class[:, 3]) > 4.5
                #    box_preds_class = box_preds_class[area_filter]
                #    top_scores_class = top_scores_class.masked_select(area_filter)
                #    top_labels_class = top_labels_class.masked_select(area_filter)

                #if classid == 2:
                #    area_filter = (box_preds_class[:, 2] * box_preds_class[:, 3]) > 1
                #    box_preds_class = box_preds_class[area_filter]
                #    top_scores_class = top_scores_class.masked_select(area_filter)
                #    top_labels_class = top_labels_class.masked_select(area_filter)

                if box_preds_class.shape[0] != 0:
                    selected_boxes,  selected_top_scores_class, selected_top_labels_class = my_rotate_nms(box_preds_class,  top_scores_class, top_labels_class, iou_thres)
                    # bp()
                else:
                    # bp()
                    selected_boxes = torch.zeros((0,5)).cuda()
                    selected_top_scores_class = torch.zeros(0).cuda()
                    selected_top_labels_class = torch.zeros(0).cuda()
                    selected = []

                all_box_prediction.append(selected_boxes)
                all_score_prediction.append(selected_top_scores_class)
                all_label_prediction.append(selected_top_labels_class)

            all_box_prediction = torch.cat(all_box_prediction, dim=0)
            all_score_prediction = torch.cat(all_score_prediction, dim=0)
            all_label_prediction = torch.cat(all_label_prediction, dim=0)
            

            if len(all_box_prediction) > 0:
                if "kernel" not in example:

                    predictions_dict = {
                        "box3d_lidar": all_box_prediction,
                        "scores": all_score_prediction,
                        "label_preds": all_label_prediction,
                    }
                else:
                    predictions_dict = {
                        "box3d_lidar": all_box_prediction,
                        "scores": all_score_prediction,
                        "label_preds": all_label_prediction,
                        "predict_heatmap": batch_cls_preds[idd]
                    }

            else:
                dtype = batch_box_preds.dtype
                device = batch_box_preds.device
                predictions_dict = {
                    "box3d_lidar":
                        torch.zeros([0, box_preds.shape[-1]],
                                    dtype=dtype,
                                    device=device),
                    "scores":
                        torch.zeros([0], dtype=dtype, device=device),
                    "label_preds":
                        torch.zeros([0], dtype=top_labels.dtype, device=device),
                }
            predictions_dicts.append(predictions_dict)

        return predictions_dicts

