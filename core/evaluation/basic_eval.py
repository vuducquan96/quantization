import torch
import numpy as np
from termcolor import colored
from pdb import set_trace as bp
from core.ops.iou3d_nms import iou3d_nms_utils

class DebugMetric():
    def __init__(self, classes):

        self.num_classes = len(classes)
        self.classes = classes
        self.iou_threshold = [0.7, 0.4, 0.5]
        print("IOU score:", self.iou_threshold)
        self.number_object_ground_truth = np.zeros(
            (self.num_classes), dtype=np.int32)
        self.number_object_prediction = np.zeros(
            (self.num_classes), dtype=np.int32)

        self.correct_prediction = np.zeros((self.num_classes), dtype=np.int32)

        self.correct_object_by_threshold = np.zeros((self.num_classes, 11), dtype=np.float32)
        self.wrong_object_by_threshold = np.zeros((self.num_classes, 11), dtype=np.float32)
        self.miss_object_by_threshold = np.zeros((self.num_classes, 11), dtype=np.float32) 

        self.distance = [0, 30, 50, 500]
        self.object_by_distance_of_ground_truth = np.zeros(
            (len(self.distance) - 1), dtype=np.int32)
        self.object_by_distance_of_prediction = np.zeros(
            (len(self.distance) - 1), dtype=np.int32)

    def reset(self):
        self.number_object_ground_truth *= 0
        self.number_object_prediction *= 0
        self.correct_prediction *= 0
        self.object_by_distance_of_ground_truth *= 0
        self.object_by_distance_of_prediction *= 0

    def to_bev_box(self, box):
        if box.shape[1] >= 8:
            return box[:, [0, 1, 3, 4, 6]]

    def bev_to_3d(self, box):
        box3d = torch.zeros((len(box), 7))
        box3d[:,0:2] = box[:,0:2]
        box3d[:,3:5] = box[:,2:4]
        box3d[:,-1] = box[:,-1]
        return box3d

    def calculate_box_by_distance(self, ground_truth, prediction):
        for i in range(len(self.distance) - 1):
            min_distance = self.distance[i]
            max_distance = self.distance[i + 1]
            for obj in range(len(ground_truth)):
                distance = np.sqrt(
                    ground_truth[obj, 0] ** 2 + ground_truth[obj, 1] ** 2 + ground_truth[obj, 2] ** 2)
                if (distance < max_distance) and (distance >= min_distance):
                    self.object_by_distance_of_ground_truth[i] += 1

            for obj in range(len(prediction)):
                distance = np.sqrt(
                    prediction[obj, 0] ** 2 + prediction[obj, 1] ** 2 + prediction[obj, 2] ** 2)
                if (distance < max_distance) and (distance >= min_distance):
                    self.object_by_distance_of_prediction[i] += 1

    def update_object_result_by_threshold(self, ground_truth, prediction, threshold):
        pass

    def update_result(self, ground_truth, prediction):
        """[calculate number of object in ground truth and prediction]

        Args:
            ground_truth ([matrix]): [(N,8), x,y,z,dx,dy,dz,yaw,class]
            prediction ([matrix]): [(N,9), x,y,z,dx,dy,dz,yaw,class,score]
        """

        assert ground_truth.shape[1] == 8, "ground truth should be (N, 8)"
        assert prediction.shape[1] == 9, "prediction should be (N, 9)"
        
        for i in range(11):
            threshold_testing = 0.08 * i
            self.update_object_result_by_threshold(ground_truth, prediction, threshold_testing)

        ground_truth_object = ground_truth[:, -1] - 1
        if torch.is_tensor(prediction) == True:
            prediction = prediction.cpu().numpy()

        ground_truth = ground_truth.astype(np.float32)
        prediction = prediction.astype(np.float32)

        self.calculate_box_by_distance(ground_truth, prediction)
        counter = 0
        for object_id in ground_truth_object:
            if object_id >= 0:
                counter += 1
            else:
                break
            self.number_object_ground_truth[int(object_id)] += 1
        ground_truth = ground_truth[:counter, :]

        prediction_object = prediction[:, -2] - 1

        for object_id in prediction_object:
            self.number_object_prediction[int(object_id)] += 1

        for object_id in range(self.num_classes):
            iou_threshold = self.iou_threshold[object_id]
            class_ground_truth = ground_truth[(
                ground_truth[:, -1] - 1) == object_id]
            class_prediction = prediction[(prediction[:, -2] - 1) == object_id]
            ground_truth_2d = self.to_bev_box(class_ground_truth)
            prediction_2d = self.to_bev_box(class_prediction)
            if len(prediction_2d)*len(ground_truth_2d) > 0:
                # ious = riou_cc(ground_truth_2d, prediction_2d)
                # ious = iou3d_nms_utils.boxes_bev_iou_cpu(class_prediction[:,:-2], class_ground_truth[:,:-1])
                ious = iou3d_nms_utils.boxes_bev_iou_cpu( class_ground_truth[:,:-1], class_prediction[:,:-2])
                for i in range(len(ground_truth_2d)):
                    max_iou = np.max(ious[i, :])
                    if max_iou >= iou_threshold:
                        self.correct_prediction[object_id] += 1

    def number_object_show(self):
        """
        Show number of object in each class
        """

        print("<><><><><><><><><><><><><><><><><><><><><><><><>")
        print(colored("Ground truths:", "white", "on_blue"))
        for cls_id in range(self.num_classes):
            print(self.classes[cls_id], " ",
                  self.number_object_ground_truth[cls_id])

        print(colored("Prediction:", "white", "on_blue"))
        for cls_id in range(self.num_classes):
            print(self.classes[cls_id], " ",
                  self.number_object_prediction[cls_id])

        print(colored("Correction:", "white", "on_red"))
        for cls_id in range(self.num_classes):
            print(self.classes[cls_id], " ", self.correct_prediction[cls_id])

        print(colored("Precision:", "red", "on_white"))
        for cls_id in range(self.num_classes):
            print("AP : ", self.classes[cls_id], " -> ",
                  self.correct_prediction[cls_id] / self.number_object_prediction[cls_id])

        print(colored("Recall:", "red", "on_white"))
        for cls_id in range(self.num_classes):
            print("AR : ", self.classes[cls_id], " -> ",
                  self.correct_prediction[cls_id] / self.number_object_ground_truth[cls_id])

    def show_object_by_distance(self):
        print("<><><><><><><><><><><><><><><><><><><><><><><><>")
        for i in range(len(self.distance) - 1):

            print("Object in range: [", self.distance[i],
                  " <-> ", self.distance[i + 1], "]")

            print("Total number [Groud truth]:",
                  self.object_by_distance_of_ground_truth[i])

            print("Total number [Prediction]:",
                  self.object_by_distance_of_prediction[i])

if __name__ == "__main__":
    my_metric = DebugMetric(["Car", "Ped", "Cyc"])
    X = np.ones((2,8))
    X[0,0] = 1
    X[0,1] = 0
    X[0,3] = np.sqrt(2) * 2
    X[0,4] = np.sqrt(2) * 2
    X[0,6] = np.pi/4.0
    X[0,-1] = 1
    X[1,:3] = 3
    Y = np.ones((1,9))
    Y[0,0] = 1
    Y[0,1] = 0
    Y[0,3] = np.sqrt(2) * 2
    Y[0,4] = np.sqrt(2) * 2
    Y[0,6] = np.pi/4.0 +0.1
    Y[0,-2] = 1
    my_metric.update_result(X,Y)
    my_metric.number_object_show()
