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
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from core.ops.iou3d_nms import iou3d_nms_utils

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
    ):
        super().__init__()
        self.conv1 = conv_nxn_bn(in_channels + skip_channels,out_channels)
        self.conv2 = conv_nxn_bn(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        x = F.interpolate(x, size=(skip.shape[-2], skip.shape[-1]), mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

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

        self.center = nn.Identity()

        # combine decoder keyword arguments
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features) -> torch.Tensor:

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i]
            x = decoder_block(x, skip)
            
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                nn.SiLU()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # SELayer(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                SELayer(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, hidden_dim, depth, channel, kernel_size, mlp_dim):
        super().__init__()

        self.depth = depth
        self.mlp_dim = mlp_dim
        dropout = 0.0

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, hidden_dim)

        self.transformer = Transformer(hidden_dim, depth, 1, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(hidden_dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        b, c, h, w = x.shape

        x = x.view(b, c, -1).unsqueeze(-1).permute((0, 3, 2, 1)).contiguous()
        x = self.transformer(x)
        x = x.permute((0, 3, 2, 1)).view(b, c, -1).view(b, c, h, w).contiguous()

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    def __init__(self,):
        super().__init__()

        L = [2, 4, 3]
        expansion = 4
        self.base_block = nn.Sequential(
                          nn.Conv2d(30, 32, kernel_size=7, stride=4, padding=3, bias=False),
                          nn.BatchNorm2d(32),
                          nn.SiLU())

        self.block1 = nn.ModuleList([])
        self.block1.append(MV2Block(32, 48, 1, expansion=2))
        self.block1.append(MV2Block(48, 48, 2, expansion=2))

        self.block2 = nn.ModuleList([])
        self.block2.append(MV2Block(48, 64, 1, expansion=1))
        self.block2.append(MV2Block(64, 64, 2, expansion=2))
        # hidden_dim, depth, channel, kernel_size, mlp_dim):
        hidden_dim2 = 48
        self.vit2 = MobileViTBlock(hidden_dim2, 2, 64, 3, hidden_dim2 * 3)

        self.block3 = nn.ModuleList([])
        self.block3.append(MV2Block(64, 80, 1, expansion=1))
        self.block3.append(MV2Block(80, 80, 2, expansion=2))
        hidden_dim3 = 60
        self.vit3 = MobileViTBlock(hidden_dim3, 2, 80, 3, hidden_dim3 * 3)

        self.block4 = nn.ModuleList([])
        self.block4.append(MV2Block(80, 100, 1, expansion=1))
        self.block4.append(MV2Block(100, 100, 2, expansion=2))
        hidden_dim4 = 60
        self.vit4 = MobileViTBlock(hidden_dim4, 2, 100, 3, hidden_dim4 * 3)

        self.decoder = UnetDecoder(
            encoder_channels=[30, 32, 48, 64, 80, 100],
            decoder_channels=[256, 128, 64, 48],
            n_blocks=4,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

    def forward(self, x):
        base = self.base_block(x)

        layer1 = self.block1[0](base)
        layer1 = self.block1[1](layer1)

        layer2 = self.block2[0](layer1)
        layer2 = self.block2[1](layer2)
        layer2 = self.vit2(layer2)

        layer3 = self.block3[0](layer2)
        layer3 = self.block3[1](layer3)
        layer3 = self.vit3(layer3)

        layer4 = self.block4[0](layer3)
        layer4 = self.block4[1](layer4)
        layer4 = self.vit4(layer4)

        features = [x, base, layer1, layer2, layer3, layer4]
        out = self.decoder(features)

        return out

def from_2d_box_to_3d(dd_box):
    ddd_box = torch.zeros((dd_box.shape[0], 7)).float()
    ddd_box[:, 0] = dd_box[:, 0]
    ddd_box[:, 1] = dd_box[:, 1]
    ddd_box[:, 3] = dd_box[:, 2]
    ddd_box[:, 4] = dd_box[:, 3]
    ddd_box[:, 6] = dd_box[:, 4]

    return ddd_box


def my_rotate_nms(box_preds_class, top_scores_class, top_labels_class, iou_thres):
    order = top_scores_class.sort(descending=True)[1]
    box_preds_class = box_preds_class[order]
    top_scores_class = top_scores_class[order]
    top_labels_class = top_labels_class[order]
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
class basemobvit(nn.Module):

    def __init__(self, config_box):

        super(basemobvit, self).__init__()
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
            self.pre_define_vector[i, :, 0] = torch.arange(0, self.feature_x, 1)

        for i in range(self.feature_x):
            self.pre_define_vector[:, i, 1] = torch.arange(0, self.feature_y, 1)

        self.pre_define_vector = self.pre_define_vector.cuda()
        self.voxel_size = data_config["VOXEL_GENERATOR"]["VOXEL_SIZE"]
        self.gaussian_overlap = 0.1
        self.min_radius = 2
        self.max_objs = 600

        # Detection Header
        #        self.header = DetectionHeader(n_input=96, n_output=96)
        # self.BEVNet1 = BBONES2D.get(config.MODEL.BBONES2D.name)(config.MODEL.BBONES2D)
        self.rpn1 = HEAD.get(config.MODEL.HEAD.name)(config.MODEL.HEAD)

        # self.iou_aware = IOU_AWARE(config.MODEL.HEAD.num_input_features, 1)
        self.backbone = MobileViT()
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

        voxel_features = data['voxels']
        spatial_features = self.backbone(voxel_features)
        pred = self.rpn1(spatial_features)

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

                if classid == 0:
                    area_filter = (box_preds_class[:, 2] * box_preds_class[:, 3]) > 4.5
                    box_preds_class = box_preds_class[area_filter]
                    top_scores_class = top_scores_class.masked_select(area_filter)
                    top_labels_class = top_labels_class.masked_select(area_filter)

                if classid == 2:
                    area_filter = (box_preds_class[:, 2] * box_preds_class[:, 3]) > 1
                    box_preds_class = box_preds_class[area_filter]
                    top_scores_class = top_scores_class.masked_select(area_filter)
                    top_labels_class = top_labels_class.masked_select(area_filter)

                if box_preds_class.shape[0] != 0:
                    selected_boxes, selected_top_scores_class, selected_top_labels_class = my_rotate_nms(
                        box_preds_class, top_scores_class, top_labels_class, iou_thres)
                else:
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

