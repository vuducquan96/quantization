from operator import gt
import os
import time
import torch
import numpy as np
import argparse
from tqdm import tqdm

from core.models.utils import VoxelGenerator
from core.evaluation.sr_detection_eval import EvaluationMetric, log_eval_result
from core.visualize_lib import debug_draw, debug_draw_ros, debug_draw_2d
from core.datalibs.sr_dataset import sample_convert_to_torch
from core.datalibs.builders import building_dataset
from core.models.builders import model_builder
from termcolor import colored
from core.tools.config_loader import config_loader
from configs.system_config import *
from pdb import set_trace as bp
import pptk


def parse_config():
    # pass
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model', type=str, default="configs/models/x1_no_dir.toml",
                        help='specify the config for testing')
    parser.add_argument('--dataset', type=str, default="configs/dataset/base.yaml",
                        help='specify the config for testing')
    parser.add_argument('--checkpoint', type=str, default="", help='pretrained_model')
    parser.add_argument('--batch_size', type=int, default=1, help='pretrained_model')
    parser.add_argument('--model_type', type=str, default="original", help='pretrained_model')
    parser.add_argument('--eval_mode', type=str, default="eval", help='pretrained_model')
    parser.add_argument('--show', type=bool, default=False, help='pretrained_model')
    args = parser.parse_args()
    dataset_config_path = args.dataset
    model_config_path = args.model
    rosbag_path = None
    eval_mode = args.eval_mode
    testing_file = args.checkpoint
    model_type = args.model_type
    batch_size = args.batch_size
    is_show = args.show
    return model_config_path, dataset_config_path, rosbag_path, eval_mode, testing_file, model_type, batch_size, is_show

def f1_score(ap, ar):
    f1 = float(2 * ((ap * ar) / (ap + ar)))
    return f1


def visualize(test_dataloader, model, config):
    output_visualize_folder = "output/"
    if os.path.isdir(output_visualize_folder) == False:
        os.mkdir(output_visualize_folder)

    model.eval()
    label_to_class = {}
    label_to_class[0] = "CAR"
    label_to_class[1] = "PED"
    label_to_class[2] = "CYC"

    total_time = 0
    data_id = 0
    for batch_idx, sample in tqdm(enumerate(test_dataloader)):
        #        if batch_idx <= 110:
        #            continue
        sample = sample_convert_to_torch(sample, dtype=torch.float32, device="cuda:0")
        if "kernel" not in sample:
            sample["kernel"] = 0

        batch_size = 1
        alll = []
        with torch.no_grad():
            nownow = time.time()
            prediction = model(sample)
            torch.cuda.synchronize()
            alll.append(time.time() - nownow)

            for batch_id, (preds, gt_bboxes, gt_class, points) in enumerate(
                    zip(prediction, sample["gt_bbox"], sample["gt_class"], sample["show_points"])):
                single_gt_box = sample["gt_boxes"][batch_id, :, :-1]
                single_gt_name = sample["gt_boxes"][batch_id, :, -1]
                filter_gt = np.sum(single_gt_box, axis=1)
                gt_bboxes = single_gt_box[filter_gt != 0]
                gt_class = single_gt_name[filter_gt != 0]

                if "predict_heatmap" in preds:
                    heatmap = torch.sigmoid(preds["predict_heatmap"])
                else:
                    heatmap = None

                saving_result_path = os.path.join(output_visualize_folder, str(data_id))
                if os.path.isdir(saving_result_path) == False:
                    os.mkdir(saving_result_path)

                point_path = os.path.join(saving_result_path, "points.npy")
                color_path = os.path.join(saving_result_path, "colors.npy")

                gt_bbox, gt_name, pr_bbox, pr_label = [], [], [], []
                for pred, label, score in zip(preds["box3d_lidar"].cpu().numpy(), preds["label_preds"].cpu().numpy(),
                                              preds["scores"].cpu().numpy()):
                    if (label == 1) and (score > 0.2):
                        pr_bbox.append(pred)
                        pr_label.append(label_to_class[label])

                    if (label == 0) and (score > 0.2):
                        pr_bbox.append(pred)
                        pr_label.append(label_to_class[label])

                    if (label == 2) and (score > 0.2):
                        pr_bbox.append(pred)
                        pr_label.append(label_to_class[label])

                pr_bbox = np.array(pr_bbox, dtype=np.float32)
                pr_label = np.array(pr_label, dtype="<U100")

                for id in range(gt_bboxes.shape[0]):
                    cls = gt_class[id] - 1
                    gt_bbox.append(
                        [gt_bboxes[id, 0], gt_bboxes[id, 1], gt_bboxes[id, 2], gt_bboxes[id, 3], gt_bboxes[id, 4], 2,
                         gt_bboxes[id, 6]])
                    gt_name.append(label_to_class[cls])

                gt_bbox = np.array(gt_bbox, dtype=np.float32)
                gt_name = np.array(gt_name, dtype="<U100")
                pp, cc = debug_draw_2d(points, points[:, 2], 2, gt_bbox, gt_name, pr_bbox, pr_label)
                v = pptk.viewer(pp)
                v.set(point_size=0.01)
                v.attributes(cc)
                bp()

def evaluate(test_dataloader, model, config):
    model.eval()

    class_encode = config["CLASS_ENCODE"]
    eval = EvaluationMetric(positive_iou_thresh=config["VALIDATE"]["POSITIVE_IOU_THRESH"], num_classes=3)
    eval.reset()
    alll = []
    for batch_idx, sample in tqdm(enumerate(test_dataloader)):
        sample = sample_convert_to_torch(sample, dtype=torch.float32, device="cuda:0")
        with torch.no_grad():
            nownow1 = time.time()
            prediction = model(sample)
            torch.cuda.synchronize()
            alll.append(time.time() - nownow1)

            for batch_id, preds in enumerate(prediction):
                detections = {}
                gt_2d = {}
                single_gt_box = sample["gt_boxes"][batch_id, :, :-1]
                single_gt_name = sample["gt_boxes"][batch_id, :, -1]
                filter_gt = np.sum(single_gt_box, axis=1)
                gt_bboxes = single_gt_box[filter_gt != 0]
                gt_class = single_gt_name[filter_gt != 0]

                detections[0], detections[1], detections[2] = [], [], []
                gt_2d[0], gt_2d[1], gt_2d[2] = [], [], []

                for pred, label, score in zip(preds["box3d_lidar"].cpu().numpy(), preds["label_preds"].cpu().numpy(),
                                              preds["scores"].cpu().numpy()):
                    if len(pred) == 5:
                        detections[label].append([pred[0], pred[1], pred[2], pred[3], pred[4], label, score])
                    else:
                        detections[label].append([pred[0], pred[1], pred[3], pred[4], pred[6], label, score])

                for id in range(gt_bboxes.shape[0]):
                    cls = gt_class[id] - 1
                    gt_2d[cls].append(
                        [gt_bboxes[id, 0], gt_bboxes[id, 1], gt_bboxes[id, 3], gt_bboxes[id, 4], gt_bboxes[id, 6], cls])

                for x in gt_2d.keys():
                    gt_2d[x] = np.array(gt_2d[x], dtype=np.float32)

                for x in detections.keys():
                    detections[x] = np.array(detections[x], dtype=np.float32)
                eval.update_frame(gt_2d, detections)

    eval_result = eval.average_precision_recall()
    log_eval_result(eval_result)
    result_dict = {}

    for cls in config["DETECTION_CLASS"]:
        AP = np.mean(eval_result[class_encode[cls]][0])
        AR = np.mean(eval_result[class_encode[cls]][1])
        result_dict[cls] = {
            "AP": AP,
            "AR": AR,
            "F1": f1_score(AP, AR)
        }

    print("Mean time:", np.mean(alll) / sample['batch_size'])
    return result_dict


def root_checking(root_list):
    root = ""
    for root_dir in root_list:
        if os.path.isdir(root_dir) == True:
            return root_dir
    return root


def main():

    model_config, dataset_config_path, rosbag_path, running_mode, testing_file, type_of_model, batch_size, is_show = parse_config()
    assert type_of_model in ["scripted", "original"]

    config = config_loader(dataset_config_path)
    data_config = config
    model_config = config_loader(model_config)

    print("=======")
    print(colored(">>> Start testing <<<", "white", "on_blue"))
    print("Model name:", model_config.experiment_name)
    print("Data config:", dataset_config_path)
    print("=======")

    data_config = config
    if "root_list" in data_config:
        root = root_checking(data_config["root_list"])
    else:
        root = None

    logger = None
    distributed_training = False
    test_dataloader, data_sampler = building_dataset(root, data_config, model_config, "val", distributed_training,
                                                     logger, batch_size)
    voxel_generator = VoxelGenerator(
        voxel_size=config["VOXEL_GENERATOR"]["VOXEL_SIZE"],
        point_cloud_range=config["POINT_CLOUD_RANGE"],
        max_num_points_per_voxel=config["VOXEL_GENERATOR"]["MAX_NUMBER_OF_POINTS_PER_VOXEL"],
        max_num_voxels=config["VOXEL_GENERATOR"]["MAX_VOXELS"]
    )
    print(">>> Let build the model <<<")
    if "script" in type_of_model:
        scripted_model = torch.jit.load(testing_file, map_location="cpu").to("cuda")
        model = get_scripted_model(model_config=model_config,
                                   data_config=data_config,
                                   voxel_generator=voxel_generator,
                                   scripted_model=scripted_model)
    else:
        model = model_builder(model_config=model_config,
                              data_config=data_config)
        print("Testing file:", testing_file)
        checkpoint = torch.load(testing_file)
        model.load_state_dict(checkpoint["net"])
    # bp()

    model.cuda()

    mode = "val"

    if is_show == True:
        visualize(test_dataloader, model, config)
        return 0

    result_dict = evaluate(test_dataloader, model, config)
    print(result_dict)

if __name__ == "__main__":
    main()

