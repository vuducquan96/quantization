import numpy as np
from core.ops.iou3d_nms import iou3d_nms_utils

def f1(prec, rec):
    if prec == 0 or rec == 0:
        return 0
    else:
        return 2 * prec * rec / (prec + rec)
def f1_score(ap, ar):
    f1 = float(2 * ((ap * ar) / (ap + ar)))
    return f1

def log_eval_result(eval_results, classes=["Car", "Ped", "Cyc"]):

    to_print = "Detection eval\n"
    for i, cls in enumerate(classes):
        Pre, Rec, precs, recs = eval_results[i][0], eval_results[i][1], eval_results[i][2], eval_results[i][3]
        prec50 = precs[4]
        rec50 = recs[4]
        AveragePre, AverageRec = np.mean(Pre), np.mean(Rec)
        AveragePre1, AverageRec1 = np.mean(Pre[:-2]), np.mean(Rec[:-2])
        to_print += cls + "\n"
        to_print += "Average prec : {0:05.3f} - average recall {1:05.3f}".format(AveragePre, AverageRec) + "\n"
        to_print += "Average prec [0.1 -> 0.7] : {0:05.3f} - average recall [0.1 -> 0.7] {1:05.3f}".format(AveragePre1, AverageRec1) + "\n"
        to_print += "@scores >= 0.5 - prec : {0:05.3f} - recall {1:05.3f}".format(prec50, rec50) + "\n"
        to_print += "Prec @ score   .1:{0:05.3f}|.2:{1:05.3f}|.3:{2:05.3f}|.4:{3:05.3f}|.5:{4:05.3f}|.6:{5:05.3f}|.7:{6:05.3f}|.8:{7:05.3f}|.9:{8:05.3f}".format(
            precs[0], precs[1], precs[2], precs[3], precs[4], precs[5], precs[6], precs[7], precs[8]) + "\n"
        to_print += "Recall @ score .1:{0:05.3f}|.2:{1:05.3f}|.3:{2:05.3f}|.4:{3:05.3f}|.5:{4:05.3f}|.6:{5:05.3f}|.7:{6:05.3f}|.8:{7:05.3f}|.9:{8:05.3f}".format(
            recs[0], recs[1], recs[2], recs[3], recs[4], recs[5], recs[6], recs[7], recs[8]) + "\n"
        to_print += "F1 @ score     .1:{0:05.3f}|.2:{1:05.3f}|.3:{2:05.3f}|.4:{3:05.3f}|.5:{4:05.3f}|.6:{5:05.3f}|.7:{6:05.3f}|.8:{7:05.3f}|.9:{8:05.3f}".format(
            f1(precs[0], recs[0]), f1(precs[1], recs[1]), f1(precs[2], recs[2]), f1(precs[3], recs[3]),
            f1(precs[4], recs[4]), f1(precs[5], recs[5]),
            f1(precs[6], recs[6]), f1(precs[7], recs[7]), f1(precs[8], recs[8])) + "\n"
        to_print += "Prec at recall @1:{0:05.3f}|@2:{1:05.3f}|@3:{2:05.3f}|@4:{3:05.3f}|@5:{4:05.3f}" \
                    "|@6:{5:05.3f}|@7:{6:05.3f}|@8:{7:05.3f}|@9:{8:05.3f}|@10:{9:05.3f}".format(Pre[1], Pre[2], \
                                                                                                Pre[3], Pre[4], Pre[5],
                                                                                                Pre[6], Pre[7], Pre[8],
                                                                                                Pre[9], Pre[10]) + "\n"
        to_print += "Recall at prec @1:{0:05.3f}|@2:{1:05.3f}|@3:{2:05.3f}|@4:{3:05.3f}|@5:{4:05.3f}" \
                    "|@6:{5:05.3f}|@7:{6:05.3f}|@8:{7:05.3f}|@9:{8:05.3f}|@10:{9:05.3f}".format(Rec[1], Rec[2], \
                                                                                                Rec[3], Rec[4], Rec[5],
                                                                                                Rec[6], Rec[7], Rec[8],
                                                                                                Rec[9], Rec[10]) + "\n"

    print(to_print)

def cls_true_positive_false_positive(ious, iou_thresh):
    true_positives = []
    for i in range(len(ious), 0, -1):
        iou = ious[0]
        if len(iou) == 0:
            true_positives = [0] * len(ious)
            break

        ind = np.argmax(iou)
        if iou[ind] >= iou_thresh:
            true_positives.append(1)
            ious = np.delete(np.delete(ious, 0, axis=0), ind, axis=1)
        else:
            true_positives.append(0)
            ious = np.delete(ious, 0, axis=0)
        if ious.shape[1] == 0:
            true_positives += [0] * (i - 1)
            break
    return true_positives


def calculate_precision_recall(cls_detections, num_gt_boxes):
    precisons = []
    recalls = []

    tp = 0
    for i in range(len(cls_detections)):
        tp += cls_detections[i, 1]
        precisons.append(tp / (i + 1))
        recalls.append(tp / num_gt_boxes)
    return np.array(precisons), np.array(recalls)


def _interpolated_precision(precisons, recalls):
    '''
    calculates list of precisions @ recalls = [0, 0.1, .., 1]
    '''
    if len(precisons) == 0:
        return [1] + [0] * 10

    interpolated_precision = [1]

    for i in [0.1 * k for k in range(1, 11)]:
        upper_prec = precisons[recalls >= i]

        interpolated_precision.append(
            0 if len(upper_prec) == 0 else max(upper_prec))
    return interpolated_precision


def _interpolated_recall(precisons, recalls):
    '''
    calculates list of recalls @ precitions = [0, 0.1, .., 1]
    '''
    if len(recalls) == 0:
        return [1] + [0] * 10
    interpolated_recall = [1]

    for i in [0.1 * k for k in range(1, 11)]:
        upper_recall = recalls[precisons >= i]

        interpolated_recall.append(
            0 if len(upper_recall) == 0 else max(upper_recall))
    return interpolated_recall


def calc_precison_recall_at_thresh(detections, thresh, num_gt_boxes):
    '''
    params:
      detections: ndarray N*3:
        1st colume: frame ind (float)
        2nd colume: 1/0 true positive/false postive (float)
        3rd colume: scores (float)
    return:
      precision and recall counted: with boxes having scores >= thresh
    '''
    considered_detections = detections[detections[:, 2] >= thresh]
    if len(considered_detections) == 0:
        return 0., 0.
    tp = np.sum(considered_detections[:, 1])
    num_dectect = len(considered_detections)
    prec = tp / num_dectect
    recall = tp / num_gt_boxes
    return prec, recall


class EvaluationMetric():
    def __init__(self, positive_iou_thresh, num_classes=3):

        self.num_classes = num_classes
        # positive_iou_thresh = 
        self.iou_thresh = positive_iou_thresh
        print("Iou threshold: ", positive_iou_thresh)
        self.num_objects = {}

        self.num_objects[0] = 0
        self.num_objects[1] = 0
        self.num_objects[2] = 0
        self.idx = 1
        self.detection_table = []

    # def update_frame(self, idx, gt_boxes, detections, use_ignore=True):
    def update_frame(self, gt_boxes, detections, use_ignore=False):
        '''update detection table: [idx, tp/fp, score, cls] and num_boxes'''
        # Update detection tables
        if len(detections) > 0:
            for cls in range(0, self.num_classes):
                cls_gt_boxes = gt_boxes[cls]
                cls_detections = detections[cls]
                self.num_objects[cls] += cls_gt_boxes.shape[0]
                num_boxes = len(cls_detections)
                if num_boxes > 0 and len(cls_gt_boxes) > 0:

                    # ious = ious_cc.rbbox_iou(cls_detections, cls_gt_boxes)
                    newbox =np.zeros((cls_detections.shape[0], 7), dtype=np.float32)
                    gtbox =np.zeros((cls_gt_boxes.shape[0], 7), dtype=np.float32)
                    newbox[:,[0,1,3,4,-1]] = cls_detections[:,[0,1,2,3,4]]
                    gtbox[:,[0,1,3,4,-1]] = cls_gt_boxes[:,[0,1,2,3,4]]

                    ious = iou3d_nms_utils.boxes_bev_iou_cpu(newbox , gtbox)
                    # ious = riou_cc(cls_detections, cls_gt_boxes)

                    true_positives = cls_true_positive_false_positive(ious, self.iou_thresh)
                    # print(true_positives)
                    table = np.stack([[self.idx] * num_boxes, true_positives, cls_detections[:, -1], [cls] * num_boxes],
                                     axis=1)
                    self.detection_table.append(table)
                    self.idx += 1
                elif len(cls_gt_boxes) == 0 and num_boxes > 0:
                    table = np.stack(
                        [[self.idx] * num_boxes, [0] * num_boxes, cls_detections[:, -1], [cls] * num_boxes], axis=1)
                    self.detection_table.append(table)
                    self.idx += 1

    def reset(self):
        self.idx = 1
        self.detection_table = []
        self.num_objects = {}

        self.num_objects[0] = 0
        self.num_objects[1] = 0
        self.num_objects[2] = 0

    def get_total_object(self):
        print("Oject 0:", self.num_objects[0])
        print("Oject 1:", self.num_objects[1])
        print("Oject 2:", self.num_objects[2])

    def average_precision_recall(self, print_time=False):
        if len(self.detection_table) == 0:
            return [[[0] * 11, [0] * 11, [0] * 9, [0] * 9],
                    [[0] * 11, [0] * 11, [0] * 9, [0] * 9],
                    [[0] * 11, [0] * 11, [0] * 9, [0] * 9]]
        eval_result = []
        detection_table = np.concatenate(self.detection_table, axis=0)
        for cls in range(0, self.num_classes):
            cls_detections = detection_table[detection_table[:, -1] == cls]
            num_gt_boxes = self.num_objects[cls]

            if num_gt_boxes == 0:
                if len(cls_detections) == 0:
                    ipres = np.array([1] * 11, dtype=np.float32)
                    irecs = np.array([1] * 11, dtype=np.float32)
                    precs = np.array([1] * 9, dtype=np.float32)
                    recs = np.array([1] * 9, dtype=np.float32)
                else:
                    ipres = np.array([0] * 11, dtype=np.float32)
                    irecs = np.array([0] * 11, dtype=np.float32)
                    precs = np.array([0] * 9, dtype=np.float32)
                    recs = np.array([0] * 9, dtype=np.float32)
                eval_result.append([ipres, irecs, precs, recs])
            else:
                cls_detections = cls_detections[np.argsort(
                    -cls_detections[:, -2])]
                precisons, recalls = calculate_precision_recall(
                    cls_detections, num_gt_boxes)
                ipres = _interpolated_precision(precisons, recalls)
                irecs = _interpolated_recall(precisons, recalls)
                precs = []
                recs = []
                for score_thres in np.linspace(start=0.1, stop=0.9, num=9, endpoint=True):
                    prec, rec = calc_precison_recall_at_thresh(cls_detections,
                                                               score_thres, num_gt_boxes)
                    precs.append(prec)
                    recs.append(rec)

                eval_result.append([ipres, irecs, precs, recs])

        return eval_result
