import math
import torch
import time
import copy
import numpy as np
from torch import nn
from numpy import log
from maskconv import MaskConv
from pdb import set_trace as bp
from configs.system_config import *
from torch.nn import functional as F
from typing import Any, List, Optional
from core.models.registry import HEAD, MODELS
from segmentation_models_pytorch.base import modules as md

from core.gaussian import gaussian_radius, draw_heatmap_gaussian
from core.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw

                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def ResNet34_hc():
    return ResNet(BasicBlock, [3, 4, 6, 3], hc=2)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        x = F.interpolate(x, size=(skip.shape[-2], skip.shape[-1]), mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)


    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i]
            x = decoder_block(x, skip)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Unet_Fam1(nn.Module):
    def __init__(self,):

        super().__init__()
        self.encoder = ResNet34_hc()
        self.decoder = UnetDecoder(
            encoder_channels=[30, 32, 32, 64, 128, 256],
            decoder_channels=[128, 64, 32, 16],
            n_blocks=4,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        self._num_anchor_per_loc = 1
        self._num_class = 3
        self._box_code_size = 6
        self.conv_cls = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)
        self.conv_box = nn.Conv2d(in_channels=16, out_channels=6, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(features)

        box_preds = self.conv_box(decoder_output)
        cls_preds = self.conv_cls(decoder_output)
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   self._box_code_size, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        
        preds = {}
        preds["box_preds"] = box_preds
        preds["cls_preds"] = cls_preds
        return preds

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, hc=1):
        super(ResNet, self).__init__()

        self.in_planes = 64//hc
        self.bn1 = nn.BatchNorm2d(64//hc)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.layer1 = self._make_layer(block, 64//hc, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128//hc, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256//hc, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512//hc, num_blocks[3], stride=2)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        features = [x]
        x = self.conv1(x)
        out = F.relu(self.bn1(x))
        features.append(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        features.append(out)
        out = self.layer2(out)
        features.append(out)
        out = self.layer3(out)
        features.append(out)
        out = self.layer4(out)
        features.append(out)
        return features


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
class resnetunet(nn.Module):

    def __init__(self, config_box):

        super(resnetunet, self).__init__()
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

        self.out_size_factor = 2
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

        self.backbone = Unet_Fam1()

        self.training = False

        self._nms_score_thresholds = [0.2, 0.2, 0.2]
        self._use_rotate_nms = True
        self._nms_pre_max_sizes = [1000, 1000, 1000]
        self._nms_post_max_sizes = [1200, 1200, 1200]
        self._nms_iou_thresholds = [0.01, 0.01, 0.01]
        

        self.loc_loss = config_box["loc_loss"]
        self.cls_loss = config_box["cls_loss"]
        self.sparse_shape = [30, 800, 800]

        self.segmentation_mask = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)
        )
        # bp()
    def voxelize(self, point_cloud):
        """
        Transform a continuous point cloud into a discrete voxelized grid that serves as the network input
        :param point_cloud: continuous point cloud | dim_0: all points, dim_1: [x, y, z, reflection]
        :return: voxelized point cloud | shape: [INPUT_DIM_0, INPUT_DIM_1, INPUT_DIM_2]
        """
        dim_x = int((self.detection_range[3] - self.detection_range[0]) / self.voxel_size[0])
        dim_y = int((self.detection_range[4] - self.detection_range[1]) / self.voxel_size[1])
        dim_z = int((self.detection_range[5] - self.detection_range[2]) / self.voxel_size[2])

        points = point_cloud
        device = points.get_device()
        cell_inds = torch.zeros((len(points), 3), dtype=torch.long).to(device)
        cell_inds[:, 1] = ((points[:, 1] - self.detection_range[1]) /
                           self.voxel_size[1]).long()
        cell_inds[:, 0] = ((points[:, 0] - self.detection_range[0]) /
                           self.voxel_size[0]).long()
        cell_inds[:, 2] = ((points[:, 2] - self.detection_range[2]) /
                           self.voxel_size[2]).long()

        valid = torch.logical_and(torch.logical_and(
            torch.logical_and(cell_inds[:, 0] > -1, cell_inds[:, 0] < dim_x),
            torch.logical_and(cell_inds[:, 1] > -1, cell_inds[:, 1] < dim_y)),
            torch.logical_and(cell_inds[:, 2] > -1, cell_inds[:, 2] < dim_z))

        valid_cell_inds = cell_inds[valid]

        voxel = torch.zeros((dim_x, dim_y, dim_z), dtype=torch.float)
        voxel[valid_cell_inds[:, 1], valid_cell_inds[:, 0],
        valid_cell_inds[:, 2]] = 1
        voxel = voxel.permute(2, 0, 1).unsqueeze(0)

        return voxel

    def point_to_mask(self, data):
        list_points = data["show_points"]
        device = data["points"].get_device()
        batch_size = len(list_points)
        masks = []
        for batch_id in range(batch_size):
            points = torch.from_numpy(list_points[batch_id]).unsqueeze(0).float().to(device)
            single_gt_box = data["gt_boxes"][batch_id, :, :]
            filter_gt = np.sum(single_gt_box, axis=1)
            single_gt_box = single_gt_box[filter_gt != 0]
            single_gt_box = torch.from_numpy(single_gt_box[:,:-1]).to(device).unsqueeze(0).float()
            index = points_in_boxes_gpu(points[:,:,:3], single_gt_box)
            index = torch.where(index[0] != -1)[0]
            object_points = points[0][index.long()]
            # bp()
            voxel = self.voxelize(object_points)
            voxel = (torch.mean(voxel[0], 0) > 0).float().unsqueeze(0)
            masks.append(voxel)
            # bp()
            # bp()
        # bp()
        masks = torch.stack(masks)
        # bp()
        return masks

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
        # bp()
        voxel_features = {}
        # voxel_features
        voxel_features["voxel_features"] = data['voxels']
        voxel_features["mask"] = data["list_mask"][0]
        voxel_features["indice"] = data["coordinates"][:,1:].cpu()
        # aa =voxel_features["voxel_features"] 
        # bp()
        # for i in range(len(voxel_features["indice"])):
        #     zz = voxel_features["indice"][i][0]
        #     yy = voxel_features["indice"][i][1]
        #     xx = voxel_features["indice"][i][2]
        #     aa[0, zz,yy,xx] = 0
        # bp()
        pred = self.backbone(voxel_features)
        if self.training == False:
            rr = self.predict1(data, pred)
            return rr

        decoder_output = pred["decoder_output"]
        segmask = self.segmentation_mask(decoder_output)

        cls_preds = pred["cls_preds"].contiguous()
        box_preds = pred["box_preds"].contiguous()
        device = cls_preds.get_device()

        gt_masks = self.point_to_mask(data)
        gt_masks = F.interpolate(gt_masks, scale_factor=1 / self.out_size_factor, mode='nearest').float().to(device)

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

        bceloss = nn.BCEWithLogitsLoss(reduction='sum')
        seg_loss = bceloss(segmask, gt_masks)
        # bp()
        loss_heat_map_1 = self.cls_loss(cls_preds[:, :, :, :, 0].clone(), heatmap_label[:, :, :, :, 0])
        loss_heat_map_2 = self.cls_loss(cls_preds[:, :, :, :, 1].clone(), heatmap_label[:, :, :, :, 1])
        loss_heat_map_3 = self.cls_loss(cls_preds[:, :, :, :, 2].clone(), heatmap_label[:, :, :, :, 2])

        cls_loss = loss_heat_map_1.sum() + loss_heat_map_2.sum() + loss_heat_map_3.sum()
        loc_loss = self.loc_loss(box_preds, anno_box, masks_obj).sum()

        loss = cls_loss + 2 * loc_loss + seg_loss
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

