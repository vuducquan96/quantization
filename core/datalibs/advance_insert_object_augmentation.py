import os
import copy
import glob
import pptk
import random
import pickle
import numpy as np
from numba import jit
from tqdm import tqdm
from termcolor import colored
from pdb import set_trace as bp
from configs.system_config import *
from core.ops.iou3d_nms import iou3d_nms_utils
from core.tools.config_loader import config_loader
from core.datalibs.box_utils import boxes_to_corners_3d
from core.datalibs.point_box_op import rotate_points, rotate_3dboxes
from core.datalibs.processor.point_feature_encoder import PointFeatureEncoder

CFG = {
    'no_point_collide_checked': 0.05,
    'cut_out:': 0.4
}

@jit(nopython=True)
def contain_point(points, 
                  center_x,
                  center_y,
                  base_z,
                  size_x,
                  size_y,
                  size_z,
                  yaw):
    num_points = points.shape[0]
    diagonal = np.sqrt(size_x**2 + size_y**2) / 2;
    half_x = size_x / 2;
    half_y = size_y / 2;
    cos_yaw = np.cos(yaw);
    sin_yaw = np.sin(yaw);

    for i in range(num_points):
        d = np.sqrt((points[i, 0]-center_x)**2 + (points[i, 1]-center_y)**2)
        if d>diagonal:
            continue
        if (points[i,2] < base_z) or (points[i,2]> (base_z + size_z)):
            continue
        rot_x = points[i, 0] - center_x
        rot_y = points[i, 1] - center_y
        rot_x = cos_yaw * rot_x - sin_yaw * rot_y
        rot_y = sin_yaw * rot_x + cos_yaw * rot_y
        if (rot_x < half_x) and (rot_x > -half_x) and (rot_y < half_y) and (rot_y > -half_y):
            return True
 
    return False


def get_base_name(data_folder, lidar_file):
    base_name = lidar_file.replace(data_folder, "")
    if base_name[0] == "/":
        base_name = base_name[1:]
    base_name = base_name.replace("/", "%").replace(".bin", "")
    return base_name

def vibrate_xy(points):

    delta = 0.04
    x = np.random.uniform(low=-delta, high=delta, size=(len(points),))
    y = np.random.uniform(low=-delta, high=delta, size=(len(points),))

    points[:, 0] += x
    points[:, 1] += y

    return points


def random_drop(points):

    ratio = random.uniform(1.0, 1.0)
    indice = np.random.choice(len(points), int(len(points) * ratio))

    return points[indice]


def cluster_cut_out_cyc(points):

    x_min, y_min, z_min = np.min(points[:, 0]), np.min(
        points[:, 1]), np.min(points[:, 2])
    x_max, y_max, z_max = np.max(points[:, 0]), np.max(
        points[:, 1]), np.max(points[:, 2])

    mid_z = (z_max - z_min) / 4 + z_min

    idx = np.where(points[:, 2] > mid_z)
    test_points = copy.copy(points[idx])

    if len(test_points) > 60:
        points = test_points

    return points


def cluster_cut_out(points):

    x_min, y_min, z_min = np.min(points[:, 0]), np.min(
        points[:, 1]), np.min(points[:, 2])
    x_max, y_max, z_max = np.max(points[:, 0]), np.max(
        points[:, 1]), np.max(points[:, 2])

    mid_z = (z_max + z_min)/2.0

    idx = np.where(points[:, 2] > mid_z)
    test_points = copy.copy(points[idx])

    if len(test_points) > 60:
        points = test_points

    return points


class AdvanceInsert():
    def __init__(self, root, config, reset_cluster_database):

        self.root = root
        self.config = config
        self.class_map = config["CLASS_ENCODE"]
        self.insert_classes = config["OBJ_INSERT_AUGMENT"]["INSERT_CLASSES"]
        self.max_num_insert = config["OBJ_INSERT_AUGMENT"]["MAX_NUM_INSERT"]
        self.min_cluster_points = config["OBJ_INSERT_AUGMENT"]["MIN_POINT_AUGMENT"]
        self.cluster_folders = config["CLUSTER_FOLDER"]
        self.detection_range = config["POINT_CLOUD_RANGE"]

        self.cluster_files = {}
        self.control_id_class = {}

        for cls in self.insert_classes:
            self.cluster_files[cls] = []
            self.control_id_class[cls] = 0

        self.point_feature_encoder = PointFeatureEncoder(
            self.config.POINT_FEATURE_ENCODING,
            point_cloud_range=np.array(self.detection_range, dtype=np.float32)
        )

        print(">>> Starting loading cluster file <<<")
        for type in self.cluster_folders:
            print("Cluster data:", type)
            cluster_folders = os.path.join(root, self.cluster_folders[type])

            if os.path.exists(os.path.join(cluster_folders, "list.pkl")) == False or reset_cluster_database == True:
                saving_file = {}
                for cls in self.insert_classes:
                    folder = os.path.join(cluster_folders, cls)
                    all_file = glob.glob(folder + "/*")
                    self.cluster_files[cls] += all_file
                    saving_file[cls] = all_file

                if os.path.isdir(cluster_folders) == True:
                    with open(os.path.join(cluster_folders, "list.pkl"), "wb") as fp:
                        pickle.dump(saving_file, fp)
                else:
                    print("No folder for cluster dataset !!")
                    print("Path:", cluster_folders)
                    exit()
            else:
                with open(os.path.join(cluster_folders, "list.pkl"), "rb") as fp:
                    list_file = pickle.load(fp)

                for cls in self.insert_classes:
                    all_file = list_file[cls]
                    self.cluster_files[cls] += all_file

        if os.path.exists("cluster_dir_list.pkl") == False or reset_cluster_database == True:
            for obj in self.insert_classes:
                filter_cluster_class = []
                cluster_files = self.cluster_files[obj]
                print("Filter class:", obj)
                print("Total cluster in ", obj, ":", len(cluster_files))
                for check_file in tqdm(cluster_files):
                    cluster = np.load(check_file)
                    if cluster["points"].shape[0] > self.min_cluster_points[obj]:
                        filter_cluster_class.append(check_file)

                print("After filter cluster in ", obj,
                      ":", len(filter_cluster_class))
                print("<><><><>")
                self.cluster_files[obj] = filter_cluster_class

            with open("cluster_dir_list.pkl", "wb") as fp:
                pickle.dump(self.cluster_files, fp)
        else:
            with open("cluster_dir_list.pkl", "rb") as fp:
                self.cluster_files = pickle.load(fp)

        for obj in self.insert_classes:
            cluster_files = len(self.cluster_files[obj])
            if cluster_files < 1:
                print("So little number of cluster file")
                exit()
            else:
                show_str = obj + ":" + str(cluster_files)
                if obj == "CAR":
                    color = "on_cyan"
                elif obj == "PED":
                    color = "on_blue"
                else:
                    color = "on_red"

                print(colored(show_str, "white", color))

    def get_excluded_file_name(self, exclude_data_list, data_dir):
        is_sequencial = {}
        for dtype, directory in data_dir.items():
            if directory.endswith("lidar") or directory.endswith("lidar/"):
                is_sequencial[dtype] = False
            else:
                is_sequencial[dtype] = True
        excluded_file_names = {}
        for dtype, data_list in exclude_data_list.items():
            lidar_paths = data_list.lidar_paths
            if is_sequencial[dtype]:
                seq = []
                for lidar_path in lidar_paths:
                    splitted = lidar_path.split("/")
                    seq.append(splitted[splitted.index("lidar") - 1])
                excluded_file_names[dtype] = set(seq)
            else:
                base_names = []
                for lidar_path in lidar_paths:
                    base_name = get_base_name(
                        data_folder=data_dir[dtype], lidar_file=lidar_path)
                    base_names.append(base_name)
                excluded_file_names[dtype] = set(base_names)

        return excluded_file_names, is_sequencial

    def is_excluded(self, cluster_file, obj_cls):
        base_name = cluster_file.replace(".npz", "")
        splitted = base_name.split(";")
        num_points = int(splitted[-1])
        if num_points < self.min_cluster_points[obj_cls]:
            print(num_points, " ", self.min_cluster_points[obj_cls])
            return True
        print("Error:", num_points, " ", self.min_cluster_points[obj_cls])
        return False

    def checking_gt_with_detection_range(self, one_bbox, ground_length):
        #        corner = get_box_corners_2(one_bbox)
        corner = boxes_to_corners_3d(one_bbox)
        corner = corner[0, :, :]
        # print(self.detection_range)
        DET_X_MIN, DET_Y_MIN, DET_Z_MIN, DET_X_MAX, DET_Y_MAX, DET_Z_MAX = self.detection_range
        x_min, y_min, z_min = np.min(corner[:, 0]), np.min(
            corner[:, 1]), np.min(corner[:, 2])
        x_max, y_max, z_max = np.max(corner[:, 0]), np.max(
            corner[:, 1]), np.max(corner[:, 2])

        # print("====")
        # # print(x_min, " ", y_min, " ", x_max, " ", y_max)
        # print(corner)
        # print(DET_X_MIN, " ", DET_Y_MIN, " ", DET_X_MAX, " ", DET_Y_MAX)
        # print("====")

        if x_min < DET_X_MIN or x_max > DET_X_MAX:
            return False

        if y_min < DET_Y_MIN or y_max > DET_Y_MAX:
            return False

        if z_max > DET_Z_MAX or z_min < DET_Z_MIN:
            return False

        return True

    def is_valid_cluster(self, points, gt_boxes, box, ground_length):
        eps = 1e-20
        testbox = [box[0], box[1], box[2], box[3], box[4], box[5], box[6]]

        if not self.checking_gt_with_detection_range(testbox, ground_length):
            return False

        if gt_boxes is not None:
            expand_gt_boxes = copy.copy(gt_boxes)

            test_box = copy.copy(box)
            test_box = test_box[np.newaxis, ...]

            if len(gt_boxes) == 0:
                return True

            iou = iou3d_nms_utils.boxes_bev_iou_cpu(test_box, expand_gt_boxes)
            if np.max(iou) > eps:

                if debugging == True:
                    print("NoNONONO")

                return False

        ## Apply checking point in bbox or not
        if random.random() > CFG['no_point_collide_checked']:
            if contain_point(points, box[0], box[1], box[2]-10, box[3], box[4], box[5]*10, box[6]):
                # print(">>>> contain <<<")
                return False
        return True

    def unify_label(self, mylabel):
        if mylabel == "BUS":

            return "CAR"

        return mylabel

    def insert(self, points, gt_bbox, gt_name, gt_importance, ground_length, gt_id):
        ######
        ## Points: Point inside box.
        ## gt_bbox: bounding box information.
        ## gt_name: Class name.
        ######

        count = {}
        my_min_cluster_points = copy.copy(self.min_cluster_points)
        for obj in self.insert_classes:
            cluster_files = self.cluster_files[obj]
            num_cluster_files = len(cluster_files)
            num_obj = min(len(cluster_files), int(
                2.5 * self.max_num_insert[obj]))
            chosen_indices = np.random.choice(
                num_cluster_files, size=num_obj, replace=False)

            chosen_cluster_files = [cluster_files[i] for i in chosen_indices]
            count_obj = 0
            for cluster_path in chosen_cluster_files:
                # if not self.is_excluded(cluster_path, obj):
                # cluster_path = cluster_file
                # bp()
                DET_X_MIN, DET_Y_MIN, DET_Z_MIN, DET_X_MAX, DET_Y_MAX, DET_Z_MAX = self.detection_range
                rot_angle = np.random.uniform(low=0, high=2 * np.pi)
                # print(cluster_path)
                cluster = np.load(cluster_path)

                cluster_points = rotate_points(cluster["points"], rot_angle)

                # print("cluster points:", cluster_points.shape)
                # exit()

                box = np.expand_dims(cluster["box"][:7], axis=0)
                box[:, 2] += float(box[:, 5]/2)
                box = rotate_3dboxes(box, rot_angle)[0]

                if obj.lower() == "ped":
                    if (box[3] > 1.) or (box[4] > 1.):
                        continue

                if obj.lower() == "cyc":
                    if box[3] < 0.9 and box[4] < 0.9:
                        continue

                if (cluster_points.shape[0] > my_min_cluster_points[obj]):

                    xr, yr, zr = box[3], box[4], box[5]

                    cx = random.uniform(
                        self.detection_range[0] + (xr*2 + 4), self.detection_range[3] - (xr*2 + 4))
                    cy = random.uniform(
                        self.detection_range[1] + (yr*2 + 4), self.detection_range[4] - (yr*2 + 4))
                    # cz = random.uniform(self.detection_range[2] + (zr + 0.2), self.detection_range[5] - (zr+0.2))
                    delta = float(box[5]/2) + 0.2
                    # bp()
                    cz = min(max(DET_Z_MIN + delta, random.uniform(ground_length - 0.2, ground_length)), DET_Z_MAX - delta)

                    xx = copy.copy(box)
                    delta_x = cx - box[0]
                    delta_y = cy - box[1]
                    delta_z = cz - box[2]

                    cluster_points[:, 0] += delta_x
                    cluster_points[:, 1] += delta_y
                    cluster_points[:, 2] += delta_z

                    box[0] += delta_x
                    box[1] += delta_y
                    box[2] += delta_z

                    # if gt_bbox.shape[0] == 0:
                    #     gt_boxes_2d = None
                    # else:
                    #     gt_boxes_2d = gt_bbox[:,[0,1,3,4,6]]

                    ########################
                    ## Checking the bbox is valid or not
                    ##
                    ########################

                    if not self.is_valid_cluster(points, gt_bbox, box, ground_length):
                        continue
                    
                    ########################
                    ## cutout augment
                    ## add noise to point augment.
                    ########################
                    
                    if random.random() > 0.3:
                        cluster_points = vibrate_xy(cluster_points)
                    
                    if len(cluster_points) > 160 and ("car" in obj.lower() or "ped" in obj.lower()):
                        if random.random() > 0.5:
                            cluster_points = cluster_cut_out(cluster_points)

                    if len(cluster_points) > 160 and "cyc" in obj.lower():
                        if random.random() > 0.5:
                            cluster_points = cluster_cut_out_cyc(
                                cluster_points)

                    data = {}
                    data["points"] = cluster_points

                    data = self.point_feature_encoder.forward(data)
                    points = np.concatenate([points, data["points"]], axis=0)

                    box = np.array(
                        [[box[0], box[1], box[2], box[3], box[4], box[5], box[6]]], dtype=np.float32)

                    obj = self.unify_label(obj)

                    if gt_bbox.shape[0] == 0:
                        gt_bbox = box
                        gt_importance = np.array([1], dtype=np.float)
                        gt_name = np.array([obj], dtype='<U100')
                        gt_id = np.array(["clone_obj"], dtype='<U100')
                    else:
                        gt_bbox = np.concatenate([gt_bbox, box], axis=0)
                        gt_importance = np.append(
                            gt_importance, np.array([1], dtype=np.float), axis=0)
                        gt_name = np.append(gt_name, np.array(
                            [obj], dtype='<U100'), axis=0)
                        gt_id = np.append(gt_id, np.array(
                            ["clone_obj"], dtype='<U100'), axis=0)
                    count_obj += 1

                if count_obj >= self.max_num_insert[obj]:
                    break

            count[obj] = count_obj

        if debugging == True:
            print(count)

        return points, gt_bbox, gt_name, gt_importance, gt_id

    def generate_human_inside_area(self, area_limit, object_name):
        xmin, ymin, xmax, ymax = area_limit[0], area_limit[1], area_limit[2], area_limit[3], area_limit[4]
        start_x = xmin
        start_y = ymin
        while start_x < xmax:
            while start_y < ymax:
                id_file = self.control_id_class[object_name]
                self.control_id_class[object_name] += 1

    def insert_v2(self, points, gt_bbox, gt_name, gt_importance, ground_length):
        count = {}
        my_min_cluster_points = copy.copy(self.min_cluster_points)


if __name__ == "__main__":

    data_config_path = "/home/seoulrobotics/quan/lazy_boy/configs/dataset/waymo_scale_2k_big4_val_small_y.json"
    root = "/home/seoulrobotics/quan/dataset"
    config = config_loader(data_config_path)

    simple_insert_object = AdvanceInsert(
        root=root,
        config=config,
        cluster_folders=config["cluster_folders"],
        insert_classes=config["obj_insert_augment"]["insert_classes"],
        max_num_insert=config["obj_insert_augment"]["max_num_insert"],
        class_map=config["class_map"],
        min_cluster_points=config["obj_insert_augment"]["min_point_augment"],
        detection_range=config["detection_range"],
        reset_cluster_database=reset_cluster_database)
