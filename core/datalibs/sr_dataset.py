import os
import json
import time
import copy
import glob
import pptk
import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from numba import int64, jit
from termcolor import colored
# import spconv
from pdb import set_trace as bp
from collections import defaultdict
from configs.system_config import *
from core.ops.iou3d_nms import iou3d_nms_utils
from core.datalibs.box_utils import boxes_to_corners_3d
from core.datalibs.augmentor.data_augmentor import DataAugmentor
from core.visualize_lib import debug_draw, add_show_future_position


def sample_convert_to_torch(example, dtype, device=None) -> dict:

    if example is None:
        return None

    device = device or torch.device("cuda")
    example_torch = {}

    float_names = [
        "voxels", "anchors", "reg_targets", "importance", "iou_mask", "fix_size_points", "points", "voxel"
    ]
    batch_size = example["batch_size"]

    assert not batch_size > 100, "Too big batch size or BUG :) !!!"

    for k, v in example.items():
        if k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            if not( v[0] is None):
                example_torch[k] = torch.tensor(v, dtype=dtype).cuda(non_blocking=True)
            else:
                example_torch[k] = v

        elif k in ["previous_voxel", "previous_coor", "previous_num_point", "2d_previous_voxel"]:
            example_torch[k] = []
            # bp()
            for iii in range(len(example[k])):
                if ("coor" in k) or ("num_point" in k): 
                    example_torch[k].append(torch.tensor(example[k][iii], dtype=torch.int32).cuda(non_blocking=True))
                else:
                    example_torch[k].append(torch.tensor(example[k][iii], dtype=dtype).cuda(non_blocking=True))
        elif k in ["coordinates", "labels", "num_points"]:
            if not( v[0] is None):
                example_torch[k] = torch.tensor(v, dtype=torch.int32).cuda(non_blocking=True)
            else:
                example_torch[k] = v
        elif k == "num_voxels":
            if not( v[0] is None):
                example_torch[k] = torch.tensor(v)
            else:
                example_torch[k] = None
        elif k == "batch_size":
            example_torch[k] = v

        elif k in ['gt_boxes']:
            val = v
            key = k
            max_gt = max([len(x) for x in val])
            batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
            for k in range(batch_size):
                batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
            example_torch[key] = batch_gt_boxes3d
        else:
            if not( v[0] is None):
                example_torch[k] = v
            else:
                example_torch[k] = None
    return example_torch

def rotate_along_z(points, theta):

    matrix = np.zeros((3, 3), dtype = np.float32)
    matrix[0][0] = np.cos(theta)
    matrix[0][1] = -np.sin(theta)
    matrix[1][0] = np.sin(theta)
    matrix[1][1] = np.cos(theta)
    matrix[2][2] = 1

    points = np.matmul(points, matrix)
    
    return points

def get_num(in_str):
    out_str = ""
    for i in range(len(in_str)):
        if in_str[i].isdigit() == True:
            out_str += in_str[i]

    return int(out_str)

def get_max_id(reduce_list):
    max_id = 0

    for check_path in reduce_list:
        name = check_path.split("/")[-1]
        frame_id = int(name.replace(".bin",""))
        if frame_id > max_id:
            max_id = frame_id

    return max_id

def filter_list_with_minid(reduce_list, min_id, pre_obs):
    filtered_list = []
    counter = 0
    max_id = get_max_id(reduce_list)

    for check_path in reduce_list:
        name = check_path.split("/")[-1]
        frame_id = int(name.replace(".bin",""))
        if frame_id >= min_id:
            if frame_id <= max_id - pre_obs:
                filtered_list.append(check_path)

    return filtered_list

def get_dataset_list(mode, config, root):

    #############################################
    ## This function return a list of dataset
    ## Mode: "train" or "val"
    ## Config: Dataconfig 
    ## Root: root path of dataset
    #############################################
    
    if root[-1] != "/":
        root += "/"

    the_list = []
    mode = mode.upper()
    path_list_testing_file = config[mode]["SINGLE_FRAME"]
    for the_file in path_list_testing_file:
        the_list += glob.glob(root + the_file)

    print("Mode:", mode)
    print("==========")
    
    if "SEQUENCE_FRAME" in config[mode]:
        for key in config[mode]["SEQUENCE_FRAME"]:
            list_bin = []
            print(key)
            base_search = "/*/lidar/*.bin"
            is_fine = False
            for i in range(3):
                data_list = root + config[mode]["SEQUENCE_FRAME"][key]["DIR"] + base_search
                data_list = data_list.replace("//","/")
                if len(glob.glob(data_list)) > 0:
                    is_fine = True
                    break
                base_search = "/*"+base_search 

            print(data_list)
            if is_fine == False:
                print(key,":", 0)
                continue
            # if is_fine == False:
            first_root = data_list[:data_list.find("*")]
            second_root = data_list[data_list.find("*")+2:]

            # telta = config[mode]["SEQUENCE_FRAME"][key]["SEQ"]
            ss,ee = config[mode]["SEQUENCE_FRAME"][key]["SEQ"]

            list_scene = sorted(glob.glob(first_root + "*/"))
            
            for folder in list_scene: 
                flag = False
                for idx in range(ss, ee+1):
                    if idx == get_num(folder.split("/")[-2]):
                        flag = True
                        break

                if flag == True:
                    glob_cmd = folder + second_root
                    list_bin += glob.glob(glob_cmd)

            reduce_list = []
            for idx in range(len(list_bin)):
                if idx % config[mode]["SEQUENCE_FRAME"][key]["STEP"] == 0:
                    reduce_list.append(list_bin[idx])

            if "MIN_ID" in config[mode]["SEQUENCE_FRAME"][key]:
                reduce_list = filter_list_with_minid(reduce_list, config[mode]["SEQUENCE_FRAME"][key]["MIN_ID"], config[mode]["SEQUENCE_FRAME"][key]["PRE_OBS"])
            print(key,":", len(reduce_list))
            the_list += reduce_list

    print("==========")
    return the_list

def batch_merge(batch_list):
    for x in batch_list:
        if x is None:
            return None
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    ret["batch_size"] = len(batch_list)
    
    for key, elems in example_merged.items():

        if key in [
            'voxels', 'num_points'
        ]:
            try:
                ret[key] = np.concatenate(elems, axis=0)
            except:
                ret[key] = elems

        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        elif key == 'points':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        elif not (key in ['show_points', '2d_previous_voxel', 'previous_num_point', 'previous_voxel', 'previous_coor', 'points', 'gt_names', 'gt_class', 'gt_bbox', 'gt_boxes', "future_position"]):
            ret[key] = np.stack(elems, axis=0)
        else:
            ret[key] = elems

    return ret

def seoul_snow_augment(points, detection_range, numb_points, p):

    VOX_X_MIN, VOX_Y_MIN, VOX_Z_MIN, VOX_X_MAX, VOX_Y_MAX, VOX_Z_MAX = detection_range
    if random.random() < p:
        
        snow_points = np.zeros((numb_points,points.shape[1]), dtype=np.float32)

        snow_points[:, 0] = np.random.uniform(low=VOX_X_MIN, high=VOX_X_MAX, size=(numb_points,))
        snow_points[:, 1]= np.random.uniform(low=VOX_Y_MIN, high=VOX_Y_MAX, size=(numb_points,))
        snow_points[:, 2]= np.random.uniform(low=(VOX_Z_MIN/2), high=VOX_Z_MAX, size=(numb_points,))
        points = np.concatenate((points, snow_points), axis=0)

    return points

def augmenation_function(gt_bbox, points):

    global_scaling_noise = (0.95, 1.00)
#    global_scaling_noise = (0.5, 2)
    global_rotation_noise = (-np.pi / 8, np.pi / 8)

    gt_bbox, points = aug.random_flip(gt_bbox, points, 0.5, False, True)
    gt_bbox, points = aug.global_scaling_v2(gt_bbox, points, *global_scaling_noise)

    return gt_bbox, points

def point_filter_range(points, detection_range, delta = 1e-4):

    VOX_X_MIN, VOX_Y_MIN, VOX_Z_MIN, VOX_X_MAX, VOX_Y_MAX, VOX_Z_MAX = detection_range
    idx = np.where(points[:, 0] > (VOX_X_MIN + delta))
    points = points[idx]
    idx = np.where(points[:, 0] < (VOX_X_MAX - delta))
    points = points[idx]
    idx = np.where(points[:, 1] > (VOX_Y_MIN + delta))
    points = points[idx]
    idx = np.where(points[:, 1] < (VOX_Y_MAX - delta))
    points = points[idx]
    idx = np.where(points[:, 2] > (VOX_Z_MIN + delta))
    points = points[idx]
    idx = np.where(points[:, 2] < (VOX_Z_MAX - delta))
    points = points[idx]

    return points

def merge_point_list(past_point_list, voxelization):

    list_previous_point = {}
    voxel_list = []
    coor_list = []
    num_point_list = []
    counter_previous_point = 0
    for raw_point in past_point_list:
        l_point = raw_point[:,:3]
        l_voxel_output = voxelization.generate(l_point)

        if isinstance(l_voxel_output, dict):
            l_voxels, l_coordinates, l_num_points_per_voxel = \
                l_voxel_output['voxels'], l_voxel_output['coordinates'], l_voxel_output['num_points_per_voxel']
        else:
            l_voxels, l_coordinates, l_num_points_per_voxel = l_voxel_output
        the_shape = np.zeros((l_coordinates.shape[0], 4))
        the_shape[:,1:] = l_coordinates
        the_shape[:,0] = counter_previous_point
        counter_previous_point += 1
        coor_list.append(the_shape)
        voxel_list.append(l_voxels)
        num_point_list.append(l_num_points_per_voxel)
    previous_voxel = np.concatenate(voxel_list, axis=0)
    previous_coor = np.concatenate(coor_list, axis=0)
    previous_num_point = np.concatenate(num_point_list, axis=0)

    return previous_voxel, previous_coor, previous_num_point

def auto_detect_ground(points, gt_bbox):

    if (gt_bbox.shape[0] == 0) or (gt_bbox is None):
        if points.shape[0] > 0:
            testing_points = copy.copy(points)
            square = 20
            min_height = -1.6
            VOX_X_MIN, VOX_Y_MIN, VOX_Z_MIN, VOX_X_MAX, VOX_Y_MAX, VOX_Z_MAX = [-square, -square, -100, square, square, 100]
            idx = np.where(testing_points[:, 0] > VOX_X_MIN)
            testing_points = testing_points[idx]
            idx = np.where(testing_points[:, 0] < VOX_X_MAX)
            testing_points = testing_points[idx]
            idx = np.where(testing_points[:, 1] > VOX_Y_MIN)
            testing_points = testing_points[idx]
            idx = np.where(testing_points[:, 1] < VOX_Y_MAX)
            testing_points = testing_points[idx]
            idx = np.where(testing_points[:, 2] > VOX_Z_MIN)
            testing_points = testing_points[idx]
            idx = np.where(testing_points[:, 2] < VOX_Z_MAX)
            testing_points = testing_points[idx]
            ground_length = min(np.min(testing_points[:,2]), min_height)
        else:
            ground_length = -1.6
    else:
        ground_length = np.mean(gt_bbox[:,2])

    return ground_length

def unify_label(pre_label):
    map_label = {
            "CAR" : ["van","jeep", "car", "truck", "bus", "emergency_vehicle" , "construction_vehicle", "other_vehicle"],
            "CYC" : ["cyc", "moto"],
            "PED" : ["ped"]
    }
    pre_label = pre_label.lower()
    for key in map_label.keys():
        for label in map_label[key]:
            if label in pre_label:
                return key  
    return "E"

def read_bin_file(file_name, col):
    """
    Reading the .bin lidar file
    Kitti, Lyft, Waymo, ...
    col: number of feature of .bin file. (x, y, z, intensity )
    """
    shape =  (-1, col)
    bin_data = np.fromfile(file_name, dtype=np.float32)
    bin_data = bin_data.reshape(shape)
    return bin_data

def read_points(file_name):
    scan = np.fromfile(file_name, dtype=np.float32)
    if ("5col" in file_name.lower()) or ("nuscene" in file_name.lower())or ("koril" in file_name.lower()) or ("scale" in file_name.lower()):
        shape = (-1, 5)
        points = read_bin_file(file_name, 5)[:,:4]

    elif ("lyft" not in file_name.lower()) and ("waymo" not in file_name.lower())and ("3col" not in file_name.lower()):
        points = read_bin_file(file_name, 4)
    elif "waymo" in file_name.lower():
        base_name = os.path.basename(file_name)
        folder_name = file_name.replace(base_name,"").replace("lidar","other_lidars")
        file_name_1 = os.path.join(folder_name, base_name.replace(".bin","_1.bin"))
        file_name_2 = os.path.join(folder_name, base_name.replace(".bin","_2.bin"))
        file_name_3 = os.path.join(folder_name, base_name.replace(".bin","_3.bin"))
        file_name_4 = os.path.join(folder_name, base_name.replace(".bin","_4.bin"))

        points = read_bin_file(file_name, 4)
        points_1 = read_bin_file(file_name_1, 4)
        points_2 = read_bin_file(file_name_2, 4)
        points_3 = read_bin_file(file_name_3, 4)
        points_4 = read_bin_file(file_name_4, 4)
        points = np.concatenate([points, points_1, points_2, points_3, points_4])
    else:
        points = read_bin_file(file_name, 3)

    return points

def get_past_point_cloud_frame(file_name, numb_pass_frame):
    frame_id = int(file_name.split("/")[-1].replace(".bin",""))
    root_path = file_name.replace(file_name.split("/")[-1], "") + "{:04d}.bin"
    list_point = []
    for i in range(frame_id-1, frame_id - numb_pass_frame - 1, -3):
        previous_file_point = root_path.format(i)
        list_point.append(read_points(previous_file_point))
    return list_point

class lidar_dataset(torch.utils.data.Dataset):

    def __init__(self, false_sampling_offline,
                       list_of_file,
                       mode,
                       root,
                       config,
                       voxel_generator,
                       data_base_sampler,
                       data_base_sampler_1,
                       anchor_box_function,
                       logger,
                       num_gpus=1,
                       partition_id=None,
                       new_augments=False):

        ###  Initailize the dataset
        #    self.point_cloud_path is lidar pointcloud path
        #    self.gt_path is label path
        ###

        self.root = root
        self.config = config
        # bp()
        self.db_sampler = data_base_sampler
        self.db_sampler_1 = data_base_sampler_1
        self.mode = mode
        self.point_cloud_path = []
        self.gt_bbox_path = []
        self.gt_point_segmentation_path = []
        self.detection_class = config["DETECTION_CLASS"]
        self.new_augments=False
        self.false_sampling_offline = false_sampling_offline

        self.class_encode = config["CLASS_ENCODE"]
        self.detection_range = config["POINT_CLOUD_RANGE"]

        self.point_feature_encoder = PointFeatureEncoder(
            self.config.POINT_FEATURE_ENCODING,
            point_cloud_range=np.array(self.detection_range, dtype=np.float32)
        )

        if config.DATA_AUGMENTOR.IS_USED == True:
            self.data_augmentor = DataAugmentor(
                root_path=Path(config.DATA_AUGMENTOR.ROOT_PATH),
                augmentor_configs=config.DATA_AUGMENTOR,
                class_names=config.DETECTION_CLASS,
                logger=logger
            )
        else:
            self.data_augmentor = None

        self.min_num_points_in_box = config["MIN_POINTS_IN_BOX"]
        self.anchor_box_function = anchor_box_function
        self.delta_detection_range = 1e-4

        # feature_map_size = grid_size[:2] // out_size_factor
        # feature_map_size = [*feature_map_size, 1][::-1]
        
        # self.feature_map_size = [1, 200, 176]

        self.FIX_POINT_SAMPLING = config.FIX_POINT_SAMPLING

        self.voxel_size =  [0.1, 0.1, 0.15]
        self.out_size_factor = config["out_size_factor"]
        if self.anchor_box_function != None:
            self.feature_map_size = voxel_generator.grid_size[:2] // out_size_factor
            self.feature_map_size = [*self.feature_map_size, 1][::-1]
            ret = self.anchor_box_function.generate_anchors(self.feature_map_size)
            self.anchors = ret["anchors"]
            self.anchors = self.anchors.reshape([-1, self.anchor_box_function.box_ndim])
            self.anchors_dict = self.anchor_box_function.generate_anchors_dict(self.feature_map_size)
            self.matched_thresholds = ret["matched_thresholds"]
            self.unmatched_thresholds = ret["unmatched_thresholds"]
        else:
            self.anchors = None
            self.anchors_dict = None
            self.matched_thresholds = None 
            self.unmatched_thresholds = None 

        if config.VOXEL_GENERATOR.IS_USED == True:
            self.voxelization = voxel_generator
        else:
            self.voxelization = None

        list_for_testing = []
        temp = []
        print(">>>>> Start loading label")

        all_lidar_bin = self.small_part(list_of_file, num_gpus, partition_id)

        for lidar_path in tqdm(all_lidar_bin):
            label_path = lidar_path.replace("/lidar/","/label/").replace(".bin",".txt")
            if os.path.isfile(label_path) and os.path.isfile(lidar_path):
                self.point_cloud_path.append(lidar_path)
                self.gt_bbox_path.append(label_path)
                point_trust_list = ["/points_index_inside_gt_boxes/", "/point_idx/"]
                for point_file_name in point_trust_list:
                    point_path = lidar_path.replace("/lidar/",point_file_name).replace(".bin",".txt")
                    if os.path.isfile(point_path) == True:
                        self.gt_point_segmentation_path.append(point_path)
                        break

        assert (len(self.point_cloud_path) == len(self.gt_bbox_path) == len(self.gt_point_segmentation_path))
        print("Total file:", len(self.point_cloud_path))
    def voxelize(self, point_cloud):
        """
        Transform a continuous point cloud into a discrete voxelized grid that serves as the network input
        :param point_cloud: continuous point cloud | dim_0: all points, dim_1: [x, y, z, reflection]
        :return: voxelized point cloud | shape: [INPUT_DIM_0, INPUT_DIM_1, INPUT_DIM_2]
        """
        VOX_Y_MIN = self.detection_range[1]
        VOX_Y_MAX = self.detection_range[4]

        VOX_X_MIN = self.detection_range[0]
        VOX_X_MAX = self.detection_range[3]

        VOX_Z_MIN = self.detection_range[2]
        VOX_Z_MAX = self.detection_range[5]

        # transformation from m to voxels
        VOX_X_DIVISION = self.voxel_size[0]
        VOX_Y_DIVISION = self.voxel_size[1]
        VOX_Z_DIVISION = self.voxel_size[2]
        #bp()

        # dimensionality of network input (voxelized point cloud)
        INPUT_DIM_0 = int((VOX_X_MAX-VOX_X_MIN) // VOX_X_DIVISION) + 1
        INPUT_DIM_1 = int((VOX_Y_MAX-VOX_Y_MIN) // VOX_Y_DIVISION) + 1
        # + 1 for average reflectance value of the points in the respective voxel
        INPUT_DIM_2 = int((VOX_Z_MAX-VOX_Z_MIN) // VOX_Z_DIVISION) + 1 + 1
        # remove all points outside the pre-specified FOV
        idx = np.where(point_cloud[:, 0] > VOX_X_MIN)
        point_cloud = point_cloud[idx]
        idx = np.where(point_cloud[:, 0] < VOX_X_MAX)
        point_cloud = point_cloud[idx]
        idx = np.where(point_cloud[:, 1] > VOX_Y_MIN)
        point_cloud = point_cloud[idx]
        idx = np.where(point_cloud[:, 1] < VOX_Y_MAX)
        point_cloud = point_cloud[idx]
        idx = np.where(point_cloud[:, 2] > VOX_Z_MIN)
        point_cloud = point_cloud[idx]
        idx = np.where(point_cloud[:, 2] < VOX_Z_MAX)
        point_cloud = point_cloud[idx]

        # create separate vectors for x, y, z coordinates and the reflectance value
        pxs = point_cloud[:, 0]
        pys = point_cloud[:, 1]
        pzs = point_cloud[:, 2]

        # convert velodyne coordinates to voxel
        qxs = ((pxs - VOX_X_MIN) // VOX_X_DIVISION).astype(np.int32)
        qys = ((pys - VOX_Y_MIN) // VOX_Y_DIVISION).astype(np.int32)
        qzs = ((pzs - VOX_Z_MIN) // VOX_Z_DIVISION).astype(np.int32)
        quantized = np.dstack((qxs, qys, qzs)).squeeze()

        # create empty voxel grid and reflectance image
        voxel_grid = np.zeros(shape=(INPUT_DIM_1, INPUT_DIM_0, INPUT_DIM_2-1), dtype=np.float32)
        reflectance_image = np.zeros(shape=(INPUT_DIM_0, INPUT_DIM_1), dtype=np.float32)
        reflectance_count = np.zeros(shape=(INPUT_DIM_0, INPUT_DIM_1), dtype=np.float32)

        for point_id, point in enumerate(quantized):
            point = point.astype(np.int32)
            voxel_grid[point[1], point[0], point[2]] = 1

        voxel_output = voxel_grid

        return voxel_output

    def small_part(self, all_lidar_bin, numbs, id):
        all_lidar_bin = sorted(all_lidar_bin)
        numb_in_list = int(len(all_lidar_bin) / numbs)
        start_id = id * numb_in_list
        end_id = (id + 1) * numb_in_list
        return all_lidar_bin[start_id: end_id]

    def checking_gt_with_detection_range(self, one_bbox):
        corner = boxes_to_corners_3d(one_bbox)
        corner = corner[0,:,:]

        DET_X_MIN, DET_Y_MIN, DET_Z_MIN, DET_X_MAX, DET_Y_MAX, DET_Z_MAX = self.detection_range
        x_min, x_max = one_bbox[0], one_bbox[0]
        y_min, y_max = one_bbox[1], one_bbox[1]
        z_min, z_max = one_bbox[2], one_bbox[2]

        if x_min < DET_X_MIN or y_min < DET_Y_MIN:
            return False

        if x_max > DET_X_MAX or y_max > DET_Y_MAX:
            return False

        if z_min < DET_Z_MIN:
            return False

        if z_max > DET_Z_MAX:
            return False
        return True

    def out_of_range_filter(self, gt_bbox, gt_name, gt_importance, gt_id):

        filtered_gt_bbox = []
        filtered_gt_name = []
        filtered_gt_importance = []
        filtered_gt_id = []
        for bbox, name, imp, id_box in zip(gt_bbox, gt_name, gt_importance, gt_id):
            if self.checking_gt_with_detection_range(bbox) == True:
                filtered_gt_bbox.append(bbox)
                filtered_gt_name.append(name)
                filtered_gt_importance.append(imp)
                filtered_gt_id.append(id_box)

        filtered_gt_bbox = np.array(filtered_gt_bbox, dtype=np.float64)
        filtered_gt_importance = np.array(filtered_gt_importance, dtype=np.float64)
        filtered_gt_name = np.array(filtered_gt_name, dtype='<U100')
        filtered_gt_id = np.array(filtered_gt_id, dtype='<U100')

        return filtered_gt_bbox, filtered_gt_name, filtered_gt_importance, filtered_gt_id

    def finding_character_split(self, data):
        sp_split = [',',';','/']
        
        for cc in sp_split:
            if cc in data:
                return cc
        return ','

    def read_gt_bbox(self, file_name_gt_bbox, file_name_gt_point_idx, global_x, global_y, plus_z):
        file_gt_bbox = open(file_name_gt_bbox, "r", encoding="utf-8")
        lines_gt_bbox = file_gt_bbox.readlines()


        file_json = file_name_gt_bbox.replace("/label/","/meta/").replace(".txt", ".json")
        if os.path.isfile(file_json) == True:
            f = open(file_json, "r")
            data = json.load(f)
            matrix_A = np.zeros((4,4), dtype=np.float32)
            for ii in range(4):
                matrix_A[ii,:] = data["pose"][ii]
        else:
            matrix_A = None

        file_gt_point_idx = open(file_name_gt_point_idx, "r", encoding="utf-8")
        lines_point_idx = file_gt_point_idx.readlines()

        gt_bbox = []
        gt_name = []
        gt_id = []
        gt_class = []
        gt_importance = []

        keep_point_list = []
        ignore_point_list = []

        count_point = 0
        id_test = 0
        
        for line_gt_bbox, line_point_idx in zip(lines_gt_bbox, lines_point_idx):
            split_character = self.finding_character_split(line_gt_bbox)
            data = line_gt_bbox.split(split_character)
            split_character = self.finding_character_split(line_point_idx)

            numb_of_point = len(line_point_idx.split(split_character)) - 1
            class_name_in_data = str(data[len(data) - 1]).rstrip()
            label = unify_label(class_name_in_data)
            id_box = data[0]

            one_bbox = [float(data[i]) for i in range(1,8)]
            one_bbox[0] += global_x
            one_bbox[1] += global_y
            one_bbox[2] += float(one_bbox[5] / 2.0) + plus_z

            if ((label in self.detection_class) 
               and (numb_of_point > self.min_num_points_in_box[label]) 
                 and (self.checking_gt_with_detection_range(one_bbox) == True)) or (self.config.NO_FILTER_GROUND_TRUTH == True):
                    
                gt_bbox.append(one_bbox)
                gt_name.append(label)
                gt_id.append(id_box)
                gt_class.append(int(self.class_encode[label]))
                gt_importance.append(int(1))

                for x in line_point_idx.split(split_character)[2:]:
                    try:
                        keep_point_list.append(int(x.replace("\n","")))
                    except:
                        print("Error")
                        pass
            else:
                for x in line_point_idx.split(split_character)[2:]:
                    try:
                        ignore_point_list.append(int(x.replace("\n","")))
                    except:
                        print("Error")
                        pass
        
        gt_bbox = np.array(gt_bbox, dtype=np.float64)
        gt_class = np.array(gt_class, dtype=np.float64)
        gt_importance = np.array(gt_importance, dtype=np.float64)
        gt_name = np.array(gt_name, dtype='<U100')
        gt_id = np.array(gt_id, dtype='<U100')

        return gt_bbox, gt_name, gt_class, gt_importance, gt_id, keep_point_list, ignore_point_list, matrix_A

    def normal_name(self, gt_name):
        for id in range(len(gt_name)):
            name = gt_name[id]
            if name == "Vehicle":
                gt_name[id] = "CAR"
        
            if name == "Pedestrian":
                gt_name[id] = "PED"
            
            if name == "Cyclist":
                gt_name[id] = "CYC"
        return gt_name
    
    def get_position(self, label_path, gt_bbox, gt_id, matrix_A, global_x, global_y, plus_z):
        frame_id = int(label_path.split("/")[-1].replace(".txt","").replace(".bin",""))
        root_path = label_path.replace(label_path.split("/")[-1], "") + "{:04d}.txt"
        root_path_point = label_path.replace(label_path.split("/")[-1], "").replace("/label/","/points_index_inside_gt_boxes/") + "{:04d}.txt"
        center_position_gt = np.ones((gt_bbox.shape[0], 4))

        try:
            center_position_gt[:,:3] = gt_bbox[:,:3]
        except:
            return None

        center_position_gt = center_position_gt.transpose()
        global_center_position_gt = np.matmul(matrix_A, center_position_gt).transpose()
        delta_future = np.zeros((gt_bbox.shape[0], 19, 2))
        delta_future_mask = np.zeros((gt_bbox.shape[0], 19))
        counter = 0
        for ii in range(frame_id + 1,frame_id + 20):
            future_path_label = root_path.format(ii)
            future_point_path = root_path_point.format(ii)
            f_gt_bbox, f_gt_name, _, f_gt_importance, f_gt_id, keep_point_list, ignore_point_list, f_matrix_A  = self.read_gt_bbox(future_path_label, future_point_path, global_x, global_y, plus_z)

            f_center_position_gt = np.ones((f_gt_bbox.shape[0], 4))

            #print("Debug 1:", "f_center_position_gt.shape ", f_gt_bbox.shape)
            try:
                f_center_position_gt[:,:3] = f_gt_bbox[:,:3]
            except:
                return None

            f_center_position_gt = f_center_position_gt.transpose()
            f_global_center_position_gt = np.matmul(f_matrix_A, f_center_position_gt).transpose()
            checking_dict = {}
            for ii, box_id in enumerate(f_gt_id):
                checking_dict[box_id] = f_global_center_position_gt[ii,:2]
            
            for ixi in range(gt_bbox.shape[0]):
                # print("ixi:  ", ixi)
                gt_box_id = gt_id[ixi]
                if gt_box_id in checking_dict:
                    temp_point = np.zeros((1, 3))
                    temp_point[0, 0] = checking_dict[gt_box_id][0] - global_center_position_gt[ixi,0]
                    temp_point[0, 1] = checking_dict[gt_box_id][1] - global_center_position_gt[ixi,1]
                    temp_point = rotate_along_z(temp_point, np.pi/2)
                    dx = temp_point[0, 0]
                    dy = temp_point[0, 1]
                    delta_future[ixi, counter, 0] = dx
                    delta_future[ixi, counter, 1] = dy
#                    delta_future[ixi,counter, 0] = checking_dict[gt_box_id][0] - global_center_position_gt[ixi,0]
#                    delta_future[ixi,counter, 1] = checking_dict[gt_box_id][1] - global_center_position_gt[ixi,1]
                    # delta_future[ixi,counter, 1] = 6.0
                    # delta_future[ixi,counter, 1] = f_matrix_A[1, -1] - matrix_A[1, -1]
                    delta_future_mask[ixi, counter] = 1
            counter += 1

        for jj in range(len(gt_id)):
            for ii in range(1, 19):
                if delta_future_mask[jj,ii] == 0:
                    delta_future[jj,ii,0] = delta_future[jj,ii-1,0]
                    delta_future[jj,ii,1] = delta_future[jj,ii-1,1]

        return delta_future


    def fix_point_sampling(self, point, maximum_point):
        fix_point = np.zeros((maximum_point, 3), dtype=np.float32)
        mean_point = copy.copy(point)
        mean_point[:, 0] = (mean_point[:, 0] - self.detection_range[0]) / (self.detection_range[3] - self.detection_range[0])
        mean_point[:, 1] = (mean_point[:, 1] - self.detection_range[1]) / (self.detection_range[4] - self.detection_range[1])
        mean_point[:, 2] = (mean_point[:, 2] - self.detection_range[2]) / (self.detection_range[5] - self.detection_range[2])
        numb_of_point = point.shape[0]

        # get_point 
        if numb_of_point > maximum_point:
            fix_point[:maximum_point,:] = mean_point[:maximum_point,:]
        else:
            fix_point[:numb_of_point,:] = mean_point[:numb_of_point,:]
        return fix_point

    def __getitem__(self, index):
        # debugging=True
        sample = {}
        ### 1/ Reading gt_bbox
        ### 2/ Remove gt_bbox out of detection range
        ### 3/ Remove gt_bbox have few points inside
        plus_z = 0
        lidar_path = self.point_cloud_path[index]
        label_path = self.gt_bbox_path[index]
        point_path = self.gt_point_segmentation_path[index]

        #if ("scaleai" in lidar_path.lower()) or ("hesai64" in lidar_path.lower()) or ("hesai40" in lidar_path.lower()) or ("vld" in lidar_path.lower()):
        if ("scaleai" in lidar_path.lower()) or ("hesai64" in lidar_path.lower()) or ("hesai40" in lidar_path.lower()):
            plus_z = 2.0

        if self.config.USING_MULTI_FRAME.IS_USED == True:
            past_point_list = get_past_point_cloud_frame(lidar_path, self.config.USING_MULTI_FRAME.NUMBER_OF_FRAME)

        points = read_points(lidar_path)

        sample["points"] = points
        sample = self.point_feature_encoder.forward(sample)
        points = sample["points"]
        points[:,2] += plus_z

        global_x = random.uniform(self.config["global_sampling_x"][0], self.config["global_sampling_x"][1])
        global_y = random.uniform(self.config["global_sampling_y"][0], self.config["global_sampling_y"][1])

        if "train" not in self.mode.lower():
            global_x, global_y = 0, 0

        global_x, global_y = 0, 0

        points[:, 0] += global_x
        points[:, 1] += global_y

        gt_bbox, gt_name, _, gt_importance, gt_id, keep_point_list, ignore_point_list, matrix_A  = self.read_gt_bbox(label_path, point_path, global_x, global_y, plus_z)

        if self.config.USING_MULTI_FRAME.IS_USED == True:
            future_position = self.get_position(label_path, gt_bbox, gt_id, matrix_A, global_x, global_y, plus_z)
            if future_position is None:
                return None
        points = np.delete(points, ignore_point_list, axis=0)
        #points = points[keep_point_list]

        points = point_filter_range(points, self.detection_range)

        if ("train" in self.mode.lower()) or (self.config.NO_FILTER_GROUND_TRUTH == True):
            gt_bbox, gt_name, gt_importance, gt_id = self.out_of_range_filter(gt_bbox, gt_name, gt_importance, gt_id)

        ground_length = auto_detect_ground(points, gt_bbox)

        if "train" in self.mode.lower():
            points, gt_bbox, gt_name, gt_importance, gt_id = self.db_sampler.insert(points, gt_bbox, gt_name, gt_importance, ground_length, gt_id)
            # points = self.false_sampling_offline.get(points, gt_bbox, ground_length)
        if ("train" in self.mode.lower()) or (self.config.NO_FILTER_GROUND_TRUTH == True):
            gt_bbox, gt_name, gt_importance, gt_id = self.out_of_range_filter(gt_bbox, gt_name, gt_importance, gt_id)


        if self.data_augmentor is not None:
            if (gt_bbox.shape[0] != 0) and ("train" in self.mode.lower()):
            
                data_dict = {}
                data_dict["gt_boxes"] = gt_bbox
                data_dict["gt_names"] = gt_name
                data_dict["gt_boxes_mask"] = np.ones_like(gt_name, dtype=np.bool)
                great_point = np.zeros((points.shape[0], 5))
                great_point[:,:3] = points
                data_dict["points"] = great_point
                data_dict = self.data_augmentor.forward(data_dict)
                
                points = data_dict["points"][:,:3]
                gt_bbox = data_dict["gt_boxes"]
                gt_name = self.normal_name(data_dict["gt_names"])
                points = point_filter_range(points, self.detection_range)
                gt_importance = np.ones_like(gt_name, dtype=np.float32)
                gt_bbox, gt_name, gt_importance, gt_id = self.out_of_range_filter(gt_bbox, gt_name, gt_importance, gt_id)

            # draw_scenes(pointcloud, gt_boxes=gt_bbox, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True)
        
        ## Convert 
        gt_class = np.array(
                [self.class_encode[n] + 1 for n in gt_name])

        anchors_mask = None

        if time_check == True:
            assign_time = time.time()
        
        if ("train" in self.mode.lower()) and (self.anchor_box_function != None):
            targets_dict = self.anchor_box_function.assign(
                self.anchors,
                self.anchors_dict,
                gt_bbox,
                anchors_mask,
                gt_classes=gt_class,
                gt_names=gt_name,
                matched_thresholds=self.matched_thresholds,
                unmatched_thresholds=self.unmatched_thresholds,
                importance=gt_importance)
        else:
            targets_dict = None

        if time_check == True:
            show_time = "Time for assign box: {:.5f} s".format(time.time() - assign_time)

        points = points[:,:self.config["numb_input_feature"]]

        if debugging == True:
            print("Voxel shape:", voxels.shape)
            
        if "train" in self.mode.lower():
           now = time.time()
           
           total_length_x = int(((self.detection_range[3] - self.detection_range[0]) / self.voxel_size[0]) / self.out_size_factor)
           total_length_y = int(((self.detection_range[4] - self.detection_range[1]) / self.voxel_size[1]) / self.out_size_factor)

           iou_mask = np.zeros((total_length_y,total_length_x), dtype=np.float32)
           counter = 0
           for gt in range(len(gt_bbox)):

               check_box = gt_bbox[gt]
               delta_range = int((max(check_box[3], check_box[4])/ 0.1) // self.out_size_factor + 1)
               x, y = check_box[0], check_box[1]
               coor_x = int(((x - self.detection_range[0]) / self.voxel_size[0]) / self.out_size_factor)
               coor_y = int(((y - self.detection_range[1]) / self.voxel_size[1]) / self.out_size_factor)
               fix_x = coor_x * self.out_size_factor * self.voxel_size[0] + self.detection_range[0]
               fix_y = coor_y * self.out_size_factor * self.voxel_size[1] + self.detection_range[1] 
               gtbox = np.zeros((1,7))
               newbox = np.zeros((1,7))
               gtbox[0,:] = copy.copy(check_box)
               gtbox[0,0] = fix_x
               gtbox[0,1] = fix_y
               newbox[0,:] = copy.copy(check_box)
               for ddx in range(-delta_range, (delta_range + 1), 1):
                   for ddy in range(-delta_range, (delta_range + 1), 1):
                       new_x = coor_x + ddx
                       new_y = coor_y + ddy
                       if (new_x >= 0) and (new_y >= 0):
                           if (new_x < total_length_x) and (new_y < total_length_y):
                               real_x = new_x * self.out_size_factor * self.voxel_size[0] + self.detection_range[0]
                               real_y = new_y * self.out_size_factor * self.voxel_size[1] + self.detection_range[1]
                               newbox[0, 0] = real_x
                               newbox[0, 1] = real_y
                               iou = iou3d_nms_utils.boxes_bev_iou_cpu(newbox , gtbox)[0, 0]
                               iou_mask[new_y, new_x] = max(iou_mask[new_y, new_x], iou)
                               
        if gt_bbox.shape[0] > 0:
            gt_boxes = np.zeros((gt_bbox.shape[0], gt_bbox.shape[1] + 1))
            gt_boxes[:, :-1] = gt_bbox
            gt_boxes[:, -1] = gt_class
        else:
            gt_boxes = None
            gt_bbox = None
            gt_class = None
            # sample["gt_bbox"] = None
            # sample["gt_names"] = None
            # sample["gt_boxes"] = None
            # return sample
        

        ######################
        # Debugging mode
        ######################


        if (debugging == True):
            if ("val" in self.mode.lower()) or ("train" in self.mode.lower()):
                print("<><><><>")
                print(lidar_path)
                print(label_path)
                print("Debug:", debugging)
                print("==== Doing statistic")
                print("Total CAR:", len(gt_name[gt_name=="CAR"]))
                print("Total PED:", len(gt_name[gt_name=="PED"]))
                print("Total CYC:", len(gt_name[gt_name=="CYC"]))
                print("============")
                detection_range = np.array([[0, 0, 0, 70.4, 80, 20, 0], [0, 0, 0, 70.4, 80, 20, 0]])
                pointcloud, colors = debug_draw(points[:,:3], points[:, 2], 0, 1, gt_bbox, gt_name, detection_range)
                if self.config.USING_MULTI_FRAME.IS_USED == True:
                    pointcloud, colors = add_show_future_position(pointcloud, colors, gt_bbox, future_position)

                v = pptk.viewer(pointcloud)
                v.attributes(colors / 255.)
                v.set(point_size=0.01, phi=3.141, theta=0.785, lookat=[1, 0, 0])
                print("<><><>")

        # if self.new_augments and "train" in self.mode.lower():
        #     pa_aug = PartAwareAugmentation(points, gt_bbox, gt_name, class_names=['CAR', 'PED', 'CYC'])
        #     points, gt_boxes_mask = pa_aug.augment(pa_aug_param="dropout_p02_swap_p02_mix_p02_sparse40_p01_noise10_p01")
        #     gt_bbox = gt_bbox[gt_boxes_mask]

        sample["path_lidar"] = lidar_path
        
        if gt_name is not None:
            sample["gt_names"] = gt_name
        else:
            sample["gt_names"] = np.array([], dtype='U<100')
            
        if gt_bbox is not None:
            sample["gt_bbox"] = gt_bbox
        else:
            sample["gt_bbox"] = np.zeros((0, 7), dtype=np.float32)

        if "train" in self.mode.lower():
            sample["iou_mask"] = iou_mask

        if gt_boxes is not None:
            sample["gt_boxes"] = gt_boxes 
        else:
            sample["gt_boxes"] = np.zeros((0,8), dtype=np.float32)

        if (gt_class is not None):
            sample["gt_class"]= gt_class
        else:
            sample["gt_class"] = np.array([], dtype=np.float32)
            
        sample["points"] = points
        sample["show_points"] = points
        
        return sample

    def __len__(self):
        return len(self.gt_bbox_path)

    def get_path(self, idx):
        return self.point_cloud_path[idx]

