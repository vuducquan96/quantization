import copy
import time
import torch
import numpy as np
from torch import nn
from numpy import log
from pdb import set_trace as bp
from core.models.registry import HEAD, MODELS
from configs.system_config import *
from core.gaussian import gaussian_radius, draw_heatmap_gaussian
from einops import rearrange
from einops.layers.torch import Rearrange
import torch
from torch import nn
from core.ops.iou3d_nms import iou3d_nms_utils
from core.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

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

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.out_channels = out_channels

        self.linear = torch.nn.Linear(in_channels, self.out_channels, bias=False)
        self.norm = torch.nn.BatchNorm1d(self.out_channels, eps=1e-3, momentum=0.01)

    def forward(self, inputs):

        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = torch.nn.functional.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        return x_max.squeeze(1)

class PALayer(nn.Module):
    def __init__(self, dim_pa, reduction_pa):
        super(PALayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_pa, dim_pa // reduction_pa),
            nn.ReLU(inplace=True),
            nn.Linear(dim_pa // reduction_pa, dim_pa)
        )

    def forward(self, x):
        b, w, _ = x.size()
        y = torch.max(x, dim=2, keepdim=True)[0].view(b, w)
        out1 = self.fc(y).view(b, w, 1)
        return out1


# Channel-wise attention for each voxel
class CALayer(nn.Module):
    def __init__(self, dim_ca, reduction_ca):
        super(CALayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_ca, dim_ca // reduction_ca),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ca // reduction_ca, dim_ca)
        )

    def forward(self, x):
        b, _, c = x.size()
        y = torch.max(x, dim=1, keepdim=True)[0].view(b, c)
        y = self.fc(y).view(b, 1, c)
        return y


# Point-wise attention for each voxel
class PACALayer(nn.Module):
    def __init__(self, dim_ca, dim_pa, reduction_r):
        super(PACALayer, self).__init__()
        self.pa = PALayer(dim_pa,  dim_pa // reduction_r)
        self.ca = CALayer(dim_ca,  dim_ca // reduction_r)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        pa_weight = self.pa(x)
        ca_weight = self.ca(x)
        paca_weight = torch.mul(pa_weight, ca_weight)
        paca_normal_weight = self.sig(paca_weight)
        out = torch.mul(x, paca_normal_weight)
        return out, paca_normal_weight

# Voxel-wise attention for each voxel
class VALayer(nn.Module):
    def __init__(self, c_num, p_num):
        super(VALayer, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_num + 3, 1),
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(p_num, 1),    ########################
            nn.ReLU(inplace=True)
        )

        self.sigmod = nn.Sigmoid()

    def forward(self, voxel_center, paca_feat):
        '''
        :param voxel_center: size (K,1,3)
        :param SACA_Feat: size (K,N,C)
        :return: voxel_attention_weight: size (K,1,1)
        '''
        voxel_center_repeat = voxel_center.repeat(1, paca_feat.shape[1], 1)
        # print(voxel_center_repeat.shape)
        voxel_feat_concat = torch.cat([paca_feat, voxel_center_repeat], dim=-1)  # K,N,C---> K,N,(C+3)

        feat_2 = self.fc1(voxel_feat_concat)  # K,N,(C+3)--->K,N,1
        feat_2 = feat_2.permute(0, 2, 1).contiguous()  # K,N,1--->K,1,N

        voxel_feat_concat = self.fc2(feat_2)  # K,1,N--->K,1,1

        voxel_attention_weight = self.sigmod(voxel_feat_concat)  # K,1,1

        return voxel_attention_weight



class VoxelFeature_TA(nn.Module):
    def __init__(self,dim_ca=6, dim_pa=100,
                 reduction_r = 3, boost_c_dim = 32,
                 use_paca_weight = False):
        super(VoxelFeature_TA, self).__init__()
        self.PACALayer1 = PACALayer(dim_ca=dim_ca, dim_pa=dim_pa, reduction_r=reduction_r)
        self.PACALayer2 = PACALayer(dim_ca=boost_c_dim, dim_pa=dim_pa, reduction_r=reduction_r)
        self.voxel_attention1 = VALayer(c_num=dim_ca, p_num=dim_pa)
        self.voxel_attention2 = VALayer(c_num=boost_c_dim, p_num=dim_pa)
        self.use_paca_weight = use_paca_weight
        self.FC1 = nn.Sequential(
            nn.Linear(2*dim_ca, boost_c_dim),
            nn.ReLU(inplace=True),
        )
        self.FC2 = nn.Sequential(
            nn.Linear(boost_c_dim, boost_c_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, voxel_center, x):
        paca1,paca_normal_weight1 = self.PACALayer1(x)
        voxel_attention1 = self.voxel_attention1(voxel_center, paca1)
        if self.use_paca_weight:
            paca1_feat = voxel_attention1 * paca1 * paca_normal_weight1
        else:
            paca1_feat = voxel_attention1 * paca1
        out1 = torch.cat([paca1_feat, x], dim=2)
        out1 = self.FC1(out1)

        paca2,paca_normal_weight2 = self.PACALayer2(out1)
        voxel_attention2 = self.voxel_attention2(voxel_center, paca2)
        if self.use_paca_weight:
            paca2_feat = voxel_attention2 * paca2 * paca_normal_weight2
        else:
            paca2_feat = voxel_attention2 * paca2
        out2 = out1 + paca2_feat
        out = self.FC2(out2)


        return out

def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator

class PointPillarsScatter(nn.Module):
    def __init__(self,num_input_features=32):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.ny = 400
        self.nx = 400
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size):

        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
                                 device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx)

        return batch_canvas

class simplenet(nn.Module):
    def __init__(self):
        super(simplenet, self).__init__()
        self.bev = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.skip_net = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.cls_preds = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.box_preds = nn.Conv2d(in_channels=32, out_channels=6, kernel_size=3, stride=1, padding=1)

    def forward(self, inp):
        x = self.bev(inp)
        skip = self.skip_net(inp)
        x = x + skip
        cls_preds = self.cls_preds(x)
        box_preds = self.box_preds(x)
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, 1,6, H, W).permute(0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, 1,3, H, W).permute(0, 1, 3, 4, 2).contiguous()
        pred = {}
        pred["cls_preds"] = cls_preds
        pred["box_preds"] = box_preds
        return pred

@MODELS.register_module()
class pillarset(nn.Module):

    def __init__(self, config_box):

        super(pillarset, self).__init__()
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

        self.out_size_factor = 4
        self.feature_x = int(np.ceil(grid_size[0] / self.out_size_factor))
        self.feature_y = int(np.ceil(grid_size[1] / self.out_size_factor))

        self.pre_define_vector = torch.zeros((self.feature_y, self.feature_x, 2))

        for i in range(self.feature_y):
            self.pre_define_vector[i, :, 0] = torch.arange(0, self.feature_x, 1)

        for i in range(self.feature_x):
            self.pre_define_vector[:, i, 1] = torch.arange(0, self.feature_y, 1)

        self.pre_define_vector = self.pre_define_vector.cuda()
        self.voxel_size = data_config["VOXEL_GENERATOR"]["VOXEL_SIZE"]
        self.gaussian_overlap = 0.1
        self.min_radius = 2
        self.max_objs = 600
        self.rpn1 = HEAD.get(config.MODEL.HEAD.name)(config.MODEL.HEAD)

        self.VoxelFeature_TA = VoxelFeature_TA(dim_pa=200)
        self.pfn_layers = PFNLayer(32, 32)
        self.to_pillar = PointPillarsScatter(32)
        self.unet = simplenet()

        self.training = False

        self._nms_score_thresholds = [0.2, 0.2, 0.2]
        self._use_rotate_nms = True
        self._nms_pre_max_sizes = [1000, 1000, 1000]
        self._nms_post_max_sizes = [1200, 1200, 1200]
        self._nms_iou_thresholds = [0.01, 0.01, 0.01]

        self.loc_loss = config_box["loc_loss"]
        self.cls_loss = config_box["cls_loss"]
        self.sparse_shape = [30, 800, 800]

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)

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

        # voxel_features = data['voxels']
        points = data["show_points"][0]

        features = data["sparse_voxels"]
        coors = data["coordinates"]
        num_voxels = data["num_points"]

        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        # f_center = torch.zeros_like(features[:, :, :2])
        # bp()
        # f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)
        # f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)
        # bp()
        # Combine together feature decorations
        features_ls = [features, f_cluster]
        # if self._with_distance:
        #     points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
        #     features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask
        features = self.VoxelFeature_TA(points_mean, features)
        features = self.pfn_layers(features)
        # bp()
        input_pillar = self.to_pillar(features, coors, 1)
        pred = self.unet(input_pillar)
        # bp()
        # pred = {}
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

        total_length_x = int(
            ((self.detection_range[3] - self.detection_range[0]) / self.voxel_size[0]) / self.out_size_factor)
        total_length_y = int(
            ((self.detection_range[4] - self.detection_range[1]) / self.voxel_size[1]) / self.out_size_factor)

        for batch_id in range(len(data["gt_boxes"])):
            single_gt_box = data["gt_boxes"][batch_id, :, :]
            filter_gt = np.sum(single_gt_box, axis=1)
            single_gt_box = single_gt_box[filter_gt != 0]
            single_gt_class = torch.from_numpy(single_gt_box[:, -1])
            single_gt_bbox = torch.from_numpy(single_gt_box[:, :-1])
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
            total_scores = total_scores.view(-1, 3).float()

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

                # if classid == 0:
                #    area_filter = (box_preds_class[:, 2] * box_preds_class[:, 3]) > 4.5
                #    box_preds_class = box_preds_class[area_filter]
                #    top_scores_class = top_scores_class.masked_select(area_filter)
                #    top_labels_class = top_labels_class.masked_select(area_filter)

                # if classid == 2:
                #    area_filter = (box_preds_class[:, 2] * box_preds_class[:, 3]) > 1
                #    box_preds_class = box_preds_class[area_filter]
                #    top_scores_class = top_scores_class.masked_select(area_filter)
                #    top_labels_class = top_labels_class.masked_select(area_filter)

                if box_preds_class.shape[0] != 0:
                    selected_boxes, selected_top_scores_class, selected_top_labels_class = my_rotate_nms(
                        box_preds_class, top_scores_class, top_labels_class, iou_thres)
                    # bp()
                else:
                    # bp()
                    selected_boxes = torch.zeros((0, 5)).cuda()
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


