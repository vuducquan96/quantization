import numpy as np
import random
import glob
from pdb import set_trace as bp
from core.visualize_lib import debug_draw
from core.helpers import ious_cc
import pptk


def random_flip(gt_boxes, points, probability=0.5, random_flip_x=True, random_flip_y=True):

    flip_x = np.random.choice([False, True],
                              replace=False,
                              p=[1 - probability, probability])
    flip_y = np.random.choice([False, True],
                              replace=False,
                              p=[1 - probability, probability])
    if flip_y and random_flip_y:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6] + np.pi
        if gt_boxes.shape[1] == 9:
            gt_boxes[:, 8] = -gt_boxes[:, 8]
        points[:, 1] = -points[:, 1]
    if flip_x and random_flip_x:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        if gt_boxes.shape[1] == 9:
            gt_boxes[:, 7] = -gt_boxes[:, 7]
        points[:, 0] = -points[:, 0]

    return gt_boxes, points


def check_collison(box, gt_boxes):
    eps = 1e-8
#    gt_boxes_2d.reshape(len(gt_boxes_2d), 1)
    if len(gt_boxes) == 0:
        return False

#    bp()
    iou = ious_cc.rbbox_iou(np.expand_dims(
        box[[0, 1, 3, 4, 6]], axis=0), gt_boxes[:, [0, 1, 3, 4, 6]])
    if np.max(iou) > eps:
        return True

    return False


def global_rotation_v2(gt_boxes, points, min_rad=-np.pi / 4,
                       max_rad=np.pi / 4):

    noise_rotation = np.random.uniform(min_rad, max_rad)
    points[:, :3] = box_np_ops.rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2)
    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(
        gt_boxes[:, :3], noise_rotation, axis=2)
    gt_boxes[:, 6] += noise_rotation

    if gt_boxes.shape[1] == 9:
        # rotate velo vector
        rot_cos = np.cos(noise_rotation)
        rot_sin = np.sin(noise_rotation)
        rot_mat_T = np.array(
            [[rot_cos, -rot_sin], [rot_sin, rot_cos]],
            dtype=points.dtype)

        gt_boxes[:, 7:9] = gt_boxes[:, 7:9] @ rot_mat_T

    return gt_boxes, points


def rotate_points_y(points):
    """counterclockwise z_axis rotation of an angle theta
    Args:
      points: N*(2 or 3 or 4) numpy array of lidar points
      theta: float - angle of rotation (radian)
      flip: bool - flip the points throgh Oxz plan
    Return:
      Rotated (flipped if flip=True) points
    """

    theta = np.random.uniform(low=0, high=2 * np.pi)
    u = np.cos(theta)
    v = np.sin(theta)

    points = np.copy(points)
    x_mean, y_mean = np.mean(points[:, 0]), np.mean(points[:, 1])

    points[:, 0] -= x_mean
    points[:, 1] -= y_mean
    rot_points = np.copy(points)

    rot_points[:, 0] = u * points[:, 0] - v * points[:, 1]
#    if not flip:
    rot_points[:, 1] = v * points[:, 0] + u * points[:, 1]

    rot_points[:, 0] += x_mean
    rot_points[:, 1] += y_mean

    return rot_points


class False_Collect_Offline:
    def __init__(self, detection_range, folder_path, max_sample):

        self.detection_range = detection_range
#        self.min_x = self.detection_range[0]
#        self.min_y = self.detection_range[1]
#        self.max_x = self.detection_range[3]
#        self.max_y = self.detection_range[4]
        self.objects = glob.glob(folder_path + "/*")
        self.max_fake_gen = max_sample
        self.id = 0

    def get_max_min_xy(self, data):
        x_min = np.min(data[:, 0])
        x_max = np.max(data[:, 0])

        y_min = np.min(data[:, 1])
        y_max = np.max(data[:, 1])

        z_min = np.min(data[:, 2])
        z_max = np.max(data[:, 2])

        length_x = x_max - x_min
        length_y = y_max - y_min
        length_z = z_max - z_min

        x, y, z = np.mean(data[:, 0]), np.mean(data[:, 1]), np.mean(data[:, 2])

#        return np.array([x, y, length_x, length_y, 0], dtype= np.float32)
#        return np.array([[x], [y], [length_x], [length_y], [0]], dtype= np.float32)
        return np.array([x, y, z, length_x, length_y, length_z, 0], dtype=np.float32)

    def get(self, points, gt_boxes, ground_length):

        for x in range(self.max_fake_gen):
            self.id += 1
            get_id = self.id % len(self.objects)
            false_object = read_bin_file(self.objects[get_id], 3)
            false_object = rotate_points_y(false_object)

            delta_z = np.min(false_object[:, 2]) - ground_length
            false_object[:, 2] -= delta_z
            check_box = self.get_max_min_xy(false_object)

            xr, yr, zr = check_box[3], check_box[4], check_box[5]

            cx = random.uniform(
                self.detection_range[0] + (xr*2 + 1), self.detection_range[3] - (xr*2 + 1))
            cy = random.uniform(
                self.detection_range[1] + (yr*2 + 1), self.detection_range[4] - (yr*2 + 1))
#            bp()

            delta_x = cx - check_box[0]
            delta_y = cy - check_box[1]

            false_object[:, 0] += delta_x
            false_object[:, 1] += delta_y

            check_box[0] += delta_x
            check_box[1] += delta_y

            is_collide = check_collison(check_box, gt_boxes)
            if is_collide == False:
                points = np.concatenate((points, false_object), axis=0)
#                debug_draw(points, points[:,2],-3,3,bounding_box=check_box.reshape(1,-1))
#            pptk.viewer(points)


#            bp()

        return points


AUG_FUNC = {"RANDOM_FLIP": random_flip,
            "GLOBAL_ROTATE": global_rotation_v2}
