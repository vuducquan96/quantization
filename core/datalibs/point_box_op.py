import numpy as np


def rotate_points(points, theta, flip=False):
    """counterclockwise z_axis rotation of an angle theta
    Args:
      points: N*(2 or 3 or 4) numpy array of lidar points
      theta: float - angle of rotation (radian)
      flip: bool - flip the points throgh Oxz plan
    Return:
      Rotated (flipped if flip=True) points
    """
    u = np.cos(theta)
    v = np.sin(theta)
    rot_points = np.copy(points)
    rot_points[:, 0] = u * points[:, 0] - v * points[:, 1]
    if not flip:
        rot_points[:, 1] = v * points[:, 0] + u * points[:, 1]
    else:
        rot_points[:, 1] = - v * points[:, 0] - u * points[:, 1]
    return rot_points


def rotate_boxes(gt_boxes: np.ndarray, theta: float, flip=False):
    """
    gt_boxes: np.ndarray (N, >5) - center_x, center_y, size_x, size_y, yaw, ..
    """
    if len(gt_boxes) == 0:
        return gt_boxes
    gt_boxes[:, :2] = rotate_points(gt_boxes[:, :2], theta, flip)
    gt_boxes[:, 4] += theta
    if flip:
        gt_boxes[:, 4] *= -1
    return gt_boxes


def rotate_3dboxes(gt_boxes: np.ndarray, theta: float, flip=False):
    """
    gt_boxes: np.ndarray (N, >7) - center_x, center_y, center_z, size_x, size_y, size_z, yaw, ..
    """

    if len(gt_boxes) == 0:
        return gt_boxes
    gt_boxes[:, :2] = rotate_points(gt_boxes[:, :2], theta, flip)
    gt_boxes[:, 6] += theta
    if flip:
        gt_boxes[:, 6] *= -1
    return gt_boxes


def scale_points(points, scale):
    points[:, :3] *= scale
    return points


def scale_boxes(gt_boxes, scale):
    """
    gt_boxes: np.ndarray (N, >5) - center_x, center_y, size_x, size_y, yaw, ..
    """
    gt_boxes[:, :4] *= scale
    return gt_boxes


def scale_3dboxes(gt_boxes, scale):
    """
    gt_boxes: np.ndarray (N, >7) - center_x, center_y, center_z, size_x, size_y, size_z, yaw, ..
    """
    gt_boxes[:, :6] *= scale
    return gt_boxes


def z_rotate_boxes_dict(gt_boxes: list, theta: float, flip: bool):
    """
    gt_boxes: list of dictionary of keys: center_x:float, center_y:float, yaw:float
    """
    xy_pos = []
    for box in gt_boxes:
        xy_pos.append([box["center_x"], box["center_y"]])
        box["yaw"] += theta
        if flip:
            box["yaw"] = -box["yaw"]

    xy_pos = np.array(xy_pos)
    xy_pos = rotate_points(xy_pos, theta, flip)

    for i, box in enumerate(gt_boxes):
        box["center_x"] = xy_pos[i, 0]
        box["center_y"] = xy_pos[i, 1]


def to_axis_aligned_boxes(poses, sizes, yaws, zoom=None):
    half_size_x = sizes[:, [0]] / 2
    half_size_y = sizes[:, [1]] / 2

    cos = np.cos(yaws)
    sin = np.sin(yaws)

    v1 = half_size_x * np.stack([cos, sin], axis=1)
    v2 = half_size_y * np.stack([sin, cos], axis=1)

    if zoom is not None and zoom != 1:
        v1 *= zoom
        v2 *= zoom

    box_v1 = poses + v1 + v2
    box_v2 = poses + v1 - v2
    box_v3 = poses - v1 - v2
    box_v4 = poses - v1 + v2

    max_x = np.maximum(
        np.maximum(box_v1[:, 0], box_v2[:, 0]),
        np.maximum(box_v3[:, 0], box_v4[:, 0]))

    max_y = np.maximum(
        np.maximum(box_v1[:, 1], box_v2[:, 1]),
        np.maximum(box_v3[:, 1], box_v4[:, 1]))

    min_x = 2 * poses[:, 0] - max_x
    min_y = 2 * poses[:, 1] - max_y

    return np.stack([min_x, min_y, max_x, max_y], axis=1)


def feat_pixel_in_boxes(gt_box, aligned_gt_box, down_scale, pos_neg_box_scale,
                        feat_row, feat_col, ignored_cls=False):
    """
    feat_row, feat_col : size of feature output
    downscale: int (power of 2) ratio of input size and output size
    box: ndarray (6,) or (5,) in bev_image coordinate include
      x, y, sx, sy, yaw, class (the last colume 'class' is optional)
    """

    yaw = gt_box[4]
    r0, c0, r1, c1 = aligned_gt_box[0], aligned_gt_box[1], aligned_gt_box[2], aligned_gt_box[3]

    r0 = max(int((r0 - 0.5) / down_scale), 0)
    c0 = max(int((c0 - 0.5) / down_scale), 0)
    r1 = min(int((r1 - 0.5) / down_scale), feat_row - 1) + 1
    c1 = min(int((c1 - 0.5) / down_scale), feat_col - 1) + 1
    # Build a grid in output feature grid whose cells in aligned gt_box
    feat_grids = np.array([[i, j] for i in range(r0, r1)
                          for j in range(c0, c1)])
    if len(feat_grids) == 0:
        if ignored_cls:
            return None
        else:
            return None, None
    # Build an associated grid in pillar coordinate
    input_grids = (feat_grids + np.array([[.5, .5]])) * down_scale
    zero_centered_grids = input_grids - np.expand_dims(gt_box[:2], axis=0)

    rotated_grids = rotate_points(zero_centered_grids, -yaw, flip=False)
    abs_rotated_grids = np.abs(rotated_grids)
    # Ignored region
    ign_grid_mask = np.logical_and(abs_rotated_grids[:, 0] <= gt_box[2] * pos_neg_box_scale[1] / 2,
                                   abs_rotated_grids[:, 1] <= gt_box[3] * pos_neg_box_scale[1] / 2)
    ign_grids = feat_grids[ign_grid_mask]
    if ignored_cls:
        return ign_grids

    pos_grid_mask = np.logical_and(abs_rotated_grids[:, 0] <= gt_box[2] * pos_neg_box_scale[0] / 2,
                                   abs_rotated_grids[:, 1] <= gt_box[3] * pos_neg_box_scale[0] / 2)
    pos_grids = feat_grids[pos_grid_mask]

    return ign_grids, pos_grids


def interior_point_in_circle(cx, cy, radius, down_scale, pos_neg_box_scale,
                             feat_row, feat_col):
    r0 = max(int((cx - radius - 0.5) / down_scale), 0)
    c0 = max(int((cy - radius - 0.5) / down_scale), 0)
    r1 = min(int((cx + radius - 0.5) / down_scale), feat_row - 1) + 1
    c1 = min(int((cy + radius - 0.5) / down_scale), feat_col - 1) + 1
    # Build a grid in output feature grid whose cells in aligned gt_box
    feat_grids = np.array([[i, j] for i in range(r0, r1)
                          for j in range(c0, c1)])
    if len(feat_grids) == 0:
        return None, None
    # Build an associated grid in pillar coordinate
    input_grids = (feat_grids + np.array([[.5, .5]])) * down_scale
    zero_centered_grids = input_grids - np.array([[cx, cy]])
    dist_grids = np.sqrt(np.sum(np.square(zero_centered_grids), axis=1))
    # Ignored region
    ign_grid_mask = dist_grids <= radius * pos_neg_box_scale[1]
    ign_grids = feat_grids[ign_grid_mask]

    pos_grid_mask = dist_grids <= radius * pos_neg_box_scale[0]
    pos_grids = feat_grids[pos_grid_mask]
    return ign_grids, pos_grids


def get_box_corners(box):
    corners = np.array([[box["size_x"] / 2, box["size_y"] / 2, box["size_z"]],
                        [-box["size_x"] / 2, box["size_y"] / 2, box["size_z"]],
                        [-box["size_x"] / 2, -box["size_y"] / 2, box["size_z"]],
                        [box["size_x"] / 2, -box["size_y"] / 2, box["size_z"]],
                        [box["size_x"] / 2, box["size_y"] / 2, 0],
                        [-box["size_x"] / 2, box["size_y"] / 2, 0],
                        [-box["size_x"] / 2, -box["size_y"] / 2, 0],
                        [box["size_x"] / 2, -box["size_y"] / 2, 0]])
    corners = rotate_points(corners, box["yaw"])
    corners = corners + \
        np.array([[box["center_x"], box["center_y"], box["base_z"]]])
    return corners


def get_box_corners_2(box):
    corners = np.array([[box[3] / 2, box[4] / 2, 0],
                        [-box[3] / 2, box[4] / 2, 0],
                        [-box[3] / 2, -box[4] / 2, 0],
                        [box[3] / 2, -box[4] / 2, 0],
                        [box[3] / 2, box[4] / 2, box[5]],
                        [-box[3] / 2, box[4] / 2, box[5]],
                        [-box[3] / 2, -box[4] / 2, box[5]],
                        [box[3] / 2, -box[4] / 2, box[5]]])
    corners = rotate_points(corners, box[6])
    corners = corners + np.array([[box[0], box[1], box[2]]])
    return corners


def get_2dbox_corners(box, z_box=0):
    corners = np.array([[box[2] / 2, box[3] / 2, z_box],
                        [-box[2] / 2, box[3] / 2, z_box],
                        [-box[2] / 2, -box[3] / 2, z_box],
                        [box[2] / 2, -box[3] / 2, z_box]])
    corners = rotate_points(corners, box[4])
    corners = corners + np.array([[box[0], box[1], 0]])
    return corners


def pol2cart(points):
    flattened_points = points.reshape(-1, points.shape[-1])
    points_z = flattened_points[:, 0] * np.exp(1j * flattened_points[:, 1])
    x = np.expand_dims(np.real(points_z), axis=1)
    y = np.expand_dims(np.imag(points_z), axis=1)

    return np.concatenate([x, y], axis=1).reshape(points.shape)


def cart2pol(points):
    flattened_points = points.reshape(-1, points.shape[-1])
    points_z = flattened_points[:, 0] + (1j * flattened_points[:, 1])

    r, t = np.abs(points_z).reshape(-1, 1), np.angle(points_z).reshape(-1, 1)
    t[t < 0] += 2 * np.pi
    polar_points = np.concatenate([r, t], axis=1)

    return polar_points.reshape(points.shape)
