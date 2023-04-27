import math
import pptk
import numpy as np
from core.datalibs.box_utils import boxes_to_corners_3d


line_list = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5],
             [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

line_list_ground_truth = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5],
                          [5, 6], [6, 7], [4, 7],
                          [0, 4], [1, 5], [2, 6], [3, 7], [5, 7], [2, 7]]

color_list = {}
color_list["red"] = [255, 0, 0]
color_list["red1"] = [255, 150, 100]

color_list["gray"] = [125, 125, 125]

color_list["green"] = [0, 255, 0]
color_list["green1"] = [100, 200, 0]

color_list["blue"] = [0, 0, 255]
color_list["blue1"] = [100, 100, 255]

color_list["white"] = [255, 255, 255]
color_list["purple"] = [255, 0, 255]

color_list["yellow"] = [255, 255, 0]
color_list["yellow1"] = [200, 255, 100]

def add_show_future_position_curve_vector(pointcloud, colors, gt_box, curve_vector, moving_vector, velocity_vector):
    oz = 0.5
    draw_step = 12
    range_of_each_step = 0.5
    for idid in range(gt_box.shape[0]):
        box = gt_box[idid]
        center_x = box[0]
        center_y = box[1]
        yaw_angle = box[6]
        aa = curve_vector[idid, 0]
        bb = curve_vector[idid, 1]
        cc = curve_vector[idid, 2]
        OX, OY, OZ = [], [], []
        if moving_vector[idid] > 0:
            t = 0
            draw_step = velocity_vector[idid]
            while t < draw_step:
                dx = t
                dy = aa * t * t + bb * t + cc
                rotate_x, rotate_y = rotate((dx, dy), yaw_angle)
                OX.append(rotate_x + center_x)
                OY.append(rotate_y + center_y)
                OZ.append(0.5)
                t += range_of_each_step

        if len(OX) > 0:
            start_x = OX[0]
            start_y = OY[0]
            eett = 0
            for xx, yy, zz in zip(OX, OY, OZ):
                eett += 1
                vector_length = 1
                vector_x = (xx - start_x) / vector_length
                vector_y = (yy - start_y) / vector_length
                #rotate_vector
                rotate_vector_x = - vector_y
                rotate_vector_y = vector_x
                new_xx = rotate_vector_x + xx
                new_yy = rotate_vector_y + yy
                if eett == int((draw_step / range_of_each_step) * 0.8):
                    x1 = new_xx
                    y1 = new_yy

                    x2 = 2 * xx - new_xx
                    y2 = 2 * yy - new_yy
#                    pointcloud, colors = append_points(pointcloud, colors, new_xx, new_yy ,zz + float(float(dd)/10.0))
#                    pointcloud, colors = append_points(pointcloud, colors, 2*xx - new_xx, 2 * yy - new_yy ,zz + float(float(dd)/10.0))

                
                delta_height = box[2] + 1
                A = np.ones((3,), dtype=np.float32)
                B = np.ones((3,), dtype=np.float32)
                A[0] = start_x
                A[1] = start_y
                A[2] = delta_height

                B[0] = xx 
                B[1] = yy
                B[2] = delta_height
                line, color = line_point_list_slow(A, B, "green")
                pointcloud = np.concatenate((pointcloud, line), axis=0)
                colors = np.concatenate((colors, color))

                start_x = xx
                start_y = yy

            A[0] = x1
            A[1] = y1
            A[2] = delta_height

            B[0] = OX[-1]
            B[1] = OY[-1]
            B[2] = delta_height
            line, color = line_point_list_slow(A, B, "green")
            pointcloud = np.concatenate((pointcloud, line), axis=0)
            colors = np.concatenate((colors, color))

            A[0] = x2
            A[1] = y2
            A[2] = delta_height

            B[0] = OX[-1]
            B[1] = OY[-1]
            B[2] = delta_height
            line, color = line_point_list_slow(A, B, "green")
            pointcloud = np.concatenate((pointcloud, line), axis=0)
            colors = np.concatenate((colors, color))

    return pointcloud, colors

def add_show_future_position(pointcloud, colors, gt_box, future_position):
    for i in range(len(future_position)):
        for j in range(0,9):
            ox = gt_box[i,0] + future_position[i, j, 0]
            oy = gt_box[i,1] + future_position[i, j, 1]
            oz = gt_box[i,2]
            print(future_position[i,j,1])
            for dd in range(0, 30, 3):
                pointcloud, colors = append_points(pointcloud, colors, ox, oy ,oz + float(float(dd)/10.0))

    return pointcloud, colors


def line_point_list_slow(A, B, color):
    density = 180

    X = np.linspace(A, B, density).astype(np.float)

    color_point = []
    for x in range(density):
        color_point.append(color_list[color])
    color_point = np.array(color_point, dtype=np.uint8)

    return X, color_point

def inte_to_rgb(pc_inte):
    minimum, maximum = np.min(pc_inte), np.max(pc_inte)
    ratio = 2 * (pc_inte-minimum) / (maximum - minimum)
    b = (np.maximum((1 - ratio), 0))
    r = (np.maximum((ratio - 1), 0))
    g = 1 - b - r
    return np.stack([r, g, b, np.ones_like(r)]).transpose()


def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b


def debug_draw_2d(pointcloud, intensity, height, gt_box=None, gt_name=None, prediction_box=None, prediction_name=None):

    colors = []
    min_i, max_i = np.min(intensity), np.max(intensity)

    global line_list, line_list_ground_truth
    for i in range(pointcloud.shape[0]):
        r, g, b = rgb(min_i, max_i, intensity[i])
        colors.append([r, g, b])

    colors = np.array(colors, dtype=np.uint32)
    draw_box, draw_label, draw_line_list, color_name_gt_pred = [gt_box,prediction_box], [gt_name, prediction_name], [line_list_ground_truth, line_list], ["","1"]

#    draw_box, draw_label, draw_line_list, color_name_gt_pred = [
#        prediction_box], [prediction_name], [line_list], ["1"]

    for gt_box, gt_name, line_list, color_add in zip(draw_box, draw_label, draw_line_list, color_name_gt_pred):
        for id in range(gt_box.shape[0]):
            label = gt_box[id, :]
            if len(label) == 7:
                cx, cy, cz, hx, hy, hz, yaw = label
                gt_label = [float(cx), float(cy), float(cz), float(
                    hx), float(hy), float(hz), float(yaw)]
            else:
                cx, cy, hx, hy, yaw = label
                gt_label = [float(cx), float(cy), height,
                            float(hx), float(hy), 0, float(yaw)]

            corner = boxes_to_corners_3d(gt_label)
            corner = corner[0, :, :]

            for id_line in line_list:
                color_name = "blue"
                if gt_name[id] == "CYC":
                    color_name = "yellow"

                if gt_name[id] == "PED":
                    color_name = "red"

                color_name = color_name + color_add

                line, color = line_point_list_slow(
                    corner[int(id_line[0])], corner[int(id_line[1])], color_name)
                pointcloud = np.concatenate((pointcloud, line), axis=0)
                colors = np.concatenate((colors, color))


            center = np.mean(corner, 0)
            head_point = np.copy(center)
            yaw = gt_label[-1]
            max_size = max(gt_label[3], gt_label[4])
            head_vector = np.array([np.cos(yaw) * max_size, np.sin(yaw)*max_size, 0.0], dtype=np.float32)
            head_point = head_point + head_vector

            line, color = line_point_list_slow(
                center, head_point, "purple")

            pointcloud = np.concatenate((pointcloud, line), axis=0)
            colors = np.concatenate((colors, color))

    return pointcloud, colors


def debug_draw(pointcloud, intensity, gt_box=None, gt_name=None, bounding_box=None):

    colors = []
    min_i = np.min(intensity)
    max_i = np.max(intensity)
    for i in range(pointcloud.shape[0]):
        r, g, b = rgb(min_i, max_i, intensity[i])
        colors.append([r, g, b])

    colors = np.array(colors, dtype=np.uint32)

    print(colors.shape)
    for id in range(gt_box.shape[0]):

        corner = boxes_to_corners_3d(gt_box[id:(id+1), :])
        new_corner = np.zeros((1, 10, 3), dtype=np.float32)
        new_corner[:, :8, :] = corner
        new_corner[0, 8, 0] = np.mean(corner[0, :, 0])
        new_corner[0, 8, 1] = np.mean(corner[0, :, 1])
        new_corner[0, 8, 2] = np.mean(corner[0, :, 2])

        new_corner[0, 9, 0] = np.mean(
            (corner[0, 4:6, 0] + corner[0, 0:2, 0])/2)
        new_corner[0, 9, 1] = np.mean(
            (corner[0, 4:6, 1] + corner[0, 0:2, 1])/2)
        new_corner[0, 9, 2] = np.mean(
            (corner[0, 4:6, 2] + corner[0, 0:2, 2])/2)

        vector = (new_corner[0, 9, :] - new_corner[0, 8, :]) * 0.8
        new_corner[0, 9, :] += vector
        corner = new_corner


        line_list = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5],
                     [5, 6], [6, 7], [4, 7],
                     [0, 4], [1, 5], [2, 6], [3, 7], [8, 9], [0, 9], [1, 9]]

        for id_line in line_list:

            color_name = "blue"
            if gt_name[id] == "CYC":
                color_name = "yellow"

            if gt_name[id] == "PED":
                color_name = "red"

            if gt_name[id] == "GT":
                color_name = "white"

            line, color = line_point_list_slow(
                corner[0, int(id_line[0]), :], corner[0, int(id_line[1]), :], color_name)
            pointcloud = np.concatenate((pointcloud, line), axis=0)
            colors = np.concatenate((colors, color))

    print(bounding_box)
    if bounding_box is not None:
        for id in range(bounding_box.shape[0]):

            corner = boxes_to_corners_3d(bounding_box[id:(id+1), :])
            corner = corner[0, :, :]

            line_list = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5],
                         [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]

            for id_line in line_list:
                line, color = line_point_list_slow(
                    corner[int(id_line[0])], corner[int(id_line[1])], "red")
                pointcloud = np.concatenate((pointcloud, line), axis=0)
                colors = np.concatenate((colors, color))

    return pointcloud, colors



def debug_draw_ros(pointcloud, intensity, min_i, max_i, gt_box=None, gt_name=None, bounding_box=None):

    colors = []
    for i in range(pointcloud.shape[0]):
        r, g, b = rgb(min_i, max_i, intensity[i])
        colors.append([r, g, b])

    colors = np.array(colors, dtype=np.uint32)

    print(colors.shape)
    for id in range(gt_box.shape[0]):

        corner = boxes_to_corners_3d(gt_box[id:(id+1), :])
        corner = corner[0, :, :]
        new_corner = np.zeros((9,3)).astype(np.float32)
        new_corner[:8,:] = corner
        new_corner[8, 0] = gt_box[id:(id+1), :][0,0]
        new_corner[8, 1] = gt_box[id:(id+1), :][0,1]
        new_corner[8, 2] = gt_box[id:(id+1), :][0,2]
        corner = new_corner


        line_list = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5],
                     [5, 6], [6, 7], [4, 7],
                     [0, 4], [1, 5], [2, 6], [3, 7], [0, 8]]

        for id_line in line_list:

            color_name = "blue"
            if gt_name[id] == "CYC":
                color_name = "yellow"

            if gt_name[id] == "PED":
                color_name = "red"

            line, color = line_point_list_slow(
                corner[int(id_line[0])], corner[int(id_line[1])], color_name)
            pointcloud = np.concatenate((pointcloud, line), axis=0)
            colors = np.concatenate((colors, color))

    print(bounding_box)
    try:
        for id in range(bounding_box.shape[0]):

            corner = boxes_to_corners_3d(gt_box[id:(id+1), :])
            corner = corner[0, :, :]

            line_list = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5],
                         [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]

            for id_line in line_list:
                line, color = line_point_list_slow(
                    corner[int(id_line[0])], corner[int(id_line[1])], "red")
                pointcloud = np.concatenate((pointcloud, line), axis=0)
                colors = np.concatenate((colors, color))
    except:
        pass

    return pointcloud, colors


def append_points(point, color,x,y,z):
    temp_point = np.zeros((point.shape[0]+1, 3))
    temp_point[:-1,:] = point[:,:3]
    temp_point[-1,0] = x
    temp_point[-1,1] = y
    temp_point[-1,2] = z
    point = temp_point

    temp_point = np.zeros((color.shape[0]+1, 3))
    temp_point[:-1,:] = color
    temp_point[-1,0] = 1.0
    color = temp_point

    return point, color

def rotate(point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = 0, 0
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy

def debug_draw_ros_with_prediction_vector(pointcloud, intensity, min_i, max_i, gt_box=None, gt_name=None, bounding_box=None):

    colors = []
    for i in range(pointcloud.shape[0]):
        r, g, b = rgb(min_i, max_i, intensity[i])
        colors.append([r, g, b])

    colors = np.array(colors, dtype=np.uint32)

    print(colors.shape)
    delta_moving = 0.0
    for id in range(gt_box.shape[0]):

        corner = boxes_to_corners_3d(gt_box[id:(id+1), :7])
        corner = corner[0, :, :]
        new_corner = np.zeros((9,3)).astype(np.float32)
        new_corner[:8,:] = corner
        new_corner[8, 0] = gt_box[id:(id+1), :][0,0]
        new_corner[8, 1] = gt_box[id:(id+1), :][0,1]
        new_corner[8, 2] = gt_box[id:(id+1), :][0,2]
        corner = new_corner

        line_list = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5],
                     [5, 6], [6, 7], [4, 7],
                     [0, 4], [1, 5], [2, 6], [3, 7], [0, 8]]

        for id_line in line_list:
            color_name = "blue"
            if gt_box[id, -2] < gt_box[id, -1]:
                color_name = "red"

#            if gt_name[id] == "CYC":
#                color_name = "yellow"
#
#            if gt_name[id] == "PED":
#                color_name = "red"

            line, color = line_point_list_slow(
                corner[int(id_line[0])], corner[int(id_line[1])], color_name)
            pointcloud = np.concatenate((pointcloud, line), axis=0)
            colors = np.concatenate((colors, color))

    print(bounding_box)
    try:
        for id in range(bounding_box.shape[0]):

            corner = boxes_to_corners_3d(gt_box[id:(id+1), :7])
            corner = corner[0, :, :]

            line_list = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5],
                         [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]

            for id_line in line_list:
                line, color = line_point_list_slow(
                    corner[int(id_line[0])], corner[int(id_line[1])], "red")
                pointcloud = np.concatenate((pointcloud, line), axis=0)
                colors = np.concatenate((colors, color))
    except:
        pass

    oz = 0.5
    draw_step = 12
    range_of_each_step = 0.5

    for idid in range(gt_box.shape[0]):
        box = gt_box[idid]
        center_x = box[0]
        center_y = box[1]
        yaw_angle = box[6]
        aa = box[7]
        bb = box[8]
        cc = box[9]
        OX, OY, OZ = [], [], []
        delta = 0.5
        if gt_box[idid, -2] < gt_box[idid, -1]:
#            print("testing: ", gt_box[idid, -1])
            t = 0
            draw_step = gt_box[idid, -3]
            while t < draw_step:
                dx = t
                dy = aa * t * t + bb * t + cc
                rotate_x, rotate_y = rotate((dx, dy), yaw_angle)
                OX.append(rotate_x + center_x)
                OY.append(rotate_y + center_y)
                OZ.append(0.5)
                t += range_of_each_step

        if len(OX) > 0:
            start_x = OX[0]
            start_y = OY[0]
            eett = 0
            for xx, yy, zz in zip(OX, OY, OZ):
                eett += 1
                #for dd in range(0, 30, 3):
                    #pointcloud, colors = append_points(pointcloud, colors, xx, yy ,zz + float(float(dd)/10.0))
                    #vector_length = np.sqrt((xx-start_x) ** 2 + (yy-start_y) ** 2)
                vector_length = 1
                vector_x = (xx - start_x) / vector_length
                vector_y = (yy - start_y) / vector_length
                #rotate_vector
                rotate_vector_x = - vector_y
                rotate_vector_y = vector_x
                new_xx = rotate_vector_x + xx
                new_yy = rotate_vector_y + yy
                if eett >= int((draw_step / range_of_each_step) * 0.8):
                    x1 = new_xx
                    y1 = new_yy

                    x2 = 2 * xx - new_xx
                    y2 = 2 * yy - new_yy
#                    pointcloud, colors = append_points(pointcloud, colors, new_xx, new_yy ,zz + float(float(dd)/10.0))
#                    pointcloud, colors = append_points(pointcloud, colors, 2*xx - new_xx, 2 * yy - new_yy ,zz + float(float(dd)/10.0))

                
                delta_height = box[2] + 1
                A = np.ones((3,), dtype=np.float32)
                B = np.ones((3,), dtype=np.float32)
                A[0] = start_x
                A[1] = start_y
                A[2] = delta_height

                B[0] = xx 
                B[1] = yy
                B[2] = delta_height
                line, color = line_point_list_slow(A, B, "green")
                pointcloud = np.concatenate((pointcloud, line), axis=0)
                colors = np.concatenate((colors, color))

                start_x = xx
                start_y = yy

            A[0] = x1
            A[1] = y1
            A[2] = delta_height

            B[0] = OX[-1]
            B[1] = OY[-1]
            B[2] = delta_height
            line, color = line_point_list_slow(A, B, "green")
            pointcloud = np.concatenate((pointcloud, line), axis=0)
            colors = np.concatenate((colors, color))

            A[0] = x2
            A[1] = y2
            A[2] = delta_height

            B[0] = OX[-1]
            B[1] = OY[-1]
            B[2] = delta_height
            line, color = line_point_list_slow(A, B, "green")
            pointcloud = np.concatenate((pointcloud, line), axis=0)
            colors = np.concatenate((colors, color))



    return pointcloud, colors

def debug_draw_ros_with_prediction(pointcloud, intensity, min_i, max_i, gt_box=None, gt_name=None, bounding_box=None):

    colors = []
    for i in range(pointcloud.shape[0]):
        r, g, b = rgb(min_i, max_i, intensity[i])
        colors.append([r, g, b])

    colors = np.array(colors, dtype=np.uint32)

    print(colors.shape)
    for id in range(gt_box.shape[0]):

        corner = boxes_to_corners_3d(gt_box[id:(id+1), :7])
        corner = corner[0, :, :]
        new_corner = np.zeros((9,3)).astype(np.float32)
        new_corner[:8,:] = corner
        new_corner[8, 0] = gt_box[id:(id+1), :][0,0]
        new_corner[8, 1] = gt_box[id:(id+1), :][0,1]
        new_corner[8, 2] = gt_box[id:(id+1), :][0,2]
        corner = new_corner

        line_list = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5],
                     [5, 6], [6, 7], [4, 7],
                     [0, 4], [1, 5], [2, 6], [3, 7], [0, 8]]

        for id_line in line_list:

            color_name = "blue"
            if gt_name[id] == "CYC":
                color_name = "yellow"

            if gt_name[id] == "PED":
                color_name = "red"

            line, color = line_point_list_slow(
                corner[int(id_line[0])], corner[int(id_line[1])], color_name)
            pointcloud = np.concatenate((pointcloud, line), axis=0)
            colors = np.concatenate((colors, color))
            # intensity = np.concatenate((intensity, color), axis=0)

#    try:
#    except:
#        pass

    print(bounding_box)
    try:
        for id in range(bounding_box.shape[0]):

            corner = boxes_to_corners_3d(gt_box[id:(id+1), :7])
            corner = corner[0, :, :]

            line_list = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5],
                         [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]

            for id_line in line_list:
                line, color = line_point_list_slow(
                    corner[int(id_line[0])], corner[int(id_line[1])], "red")
                pointcloud = np.concatenate((pointcloud, line), axis=0)
                colors = np.concatenate((colors, color))
    except:
        pass

    oz = 0.5

    for id in range(gt_box.shape[0]):
        ox = gt_box[id,0] + gt_box[id, 7]
        oy = gt_box[id,1] + gt_box[id, 8]
        A = np.ones((3,), dtype=np.float32)
        B = np.ones((3,), dtype=np.float32)
        delta_height = 1.5
        A[0] = gt_box[id,0]
        A[1] = gt_box[id,1]
        A[2] = delta_height
        B[0] = ox
        B[1] = oy
        B[2] = delta_height

        line, color = line_point_list_slow(A, B, "red")

        pointcloud = np.concatenate((pointcloud, line), axis=0)
        colors = np.concatenate((colors, color))
        for z in range(7, 12, 2):

            A = np.ones((3,), dtype=np.float32)
            B = np.ones((3,), dtype=np.float32)
            A[0] = gt_box[id,0] + gt_box[id, z]
            A[1] = gt_box[id,1] + gt_box[id, z+1]
            A[2] = delta_height

            B[0] = gt_box[id,0] + gt_box[id, z+2]
            B[1] = gt_box[id,1] + gt_box[id, z+3]
            B[2] = delta_height

            line, color = line_point_list_slow(A, B, "red")

            pointcloud = np.concatenate((pointcloud, line), axis=0)
            colors = np.concatenate((colors, color))

        oz = 0.5
        for dd in range(0, 30, 3):
            pointcloud, colors = append_points(pointcloud, colors, ox, oy ,oz + float(float(dd)/10.0))

        ox = gt_box[id,0] + gt_box[id, 9]
        oy = gt_box[id,1] + gt_box[id, 10]
        for dd in range(0, 30, 3):
            pointcloud, colors = append_points(pointcloud, colors, ox, oy ,oz + float(float(dd)/10.0))
        ox = gt_box[id,0] + gt_box[id, 11]
        oy = gt_box[id,1] + gt_box[id, 12]
        for dd in range(0, 30, 3):
            pointcloud, colors = append_points(pointcloud, colors, ox, oy ,oz + float(float(dd)/10.0))

        ox = gt_box[id,0] + gt_box[id, 13]
        oy = gt_box[id,1] + gt_box[id, 14]
        for dd in range(0, 30, 3):
            pointcloud, colors = append_points(pointcloud, colors, ox, oy ,oz + float(float(dd)/10.0))
    return pointcloud, colors

def vis_lr(optimizer, scheduler):
    from torch.optim.swa_utils import AveragedModel, SWALR
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import matplotlib.pyplot as plt

    list_lr = []
    swa_start = 74
    swa_scheduler = SWALR(optimizer, anneal_strategy="cos", anneal_epochs=10, swa_lr=0.00001)
    for epoch in range(100):
        for input in range(400):
            optimizer.zero_grad()
            # loss_fn(model(input), target).backward()
            list_lr.append(get_lr(optimizer))
            optimizer.step()
            if epoch > swa_start:
                # swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()
    # print(list_lr)
    plt.scatter(x=range(len(list_lr)),y=list_lr, s=1 ,alpha=1)
    plt.show()
