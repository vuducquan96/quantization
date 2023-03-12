from operator import gt
import os
import time
import torch
import numpy as np
import argparse
from tqdm import tqdm

from core.models.utils import VoxelGenerator
from core.evaluation.sr_detection_eval import EvaluationMetric, log_eval_result
from core.evaluation.basic_eval import DebugMetric
from core.visualize_lib import debug_draw, debug_draw_ros, debug_draw_2d
from core.datalibs.sr_dataset import sample_convert_to_torch
from core.datalibs.builders import building_dataset
from core.models.builders import model_builder
from termcolor import colored
from core.tools.config_loader import config_loader
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from configs.system_config import *
from pdb import set_trace as bp
from core.script.template_resunet import resnetunet

def parse_config():
    # pass
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model', type=str, default="configs/models/x1_no_dir.toml",
                        help='specify the config for testing')
    parser.add_argument('--dataset', type=str, default="configs/dataset/base.yaml",
                        help='specify the config for testing')
    parser.add_argument('--checkpoint', type=str, default="", help='pretrained_model')
    parser.add_argument('--out', type=str, default="script_model.pth", help='pretrained_model')
    parser.add_argument('--batch_size', type=int, default=1, help='pretrained_model')
    parser.add_argument('--model_type', type=str, default="original", help='pretrained_model')
    parser.add_argument('--eval_mode', type=str, default="eval", help='pretrained_model')
    parser.add_argument('--show', type=bool, default=False, help='pretrained_model')
    args = parser.parse_args()
    dataset_config_path = args.dataset
    out_file = args.out
    model_config_path = args.model
    eval_mode = args.eval_mode
    testing_file = args.checkpoint
    model_type = args.model_type
    batch_size = args.batch_size
    is_show = args.show
    return model_config_path, dataset_config_path, eval_mode, testing_file, model_type, batch_size, is_show, out_file

def f1_score(ap, ar):
    f1 = float(2 * ((ap * ar) / (ap + ar)))
    return f1

def root_checking(root_list):
    root = ""
    for root_dir in root_list:
        if os.path.isdir(root_dir) == True:
            return root_dir
    return root


def main():

    model_config, dataset_config_path, running_mode, testing_file, type_of_model, batch_size, is_show, out_file = parse_config()
    assert type_of_model in ["scripted", "original"]

    config = config_loader(dataset_config_path)
    data_config = config
    model_config = config_loader(model_config)

    model = model_builder(model_config=model_config,
                            data_config=data_config)
    print("Testing file:", testing_file)
    model.eval().float()
    checkpoint = torch.load(testing_file)
    model.load_state_dict(checkpoint["net"])
    script_model = resnetunet(backbone=model.backbone)
    
    input_tensor = torch.rand((1,30, 512, 512)).float()
    with torch.no_grad():
        for i in range(10):
            now = time.time()
            script_model(input_tensor)
            print("Time:", (time.time() - now) * 1000.0)
    
    traced = torch.jit.trace(script_model, (input_tensor))
    traced.save(out_file)

    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {"": qconfig}
    # `prepare_fx` inserts observers in the model based on the configuration in `qconfig_dict`
    model_prepared = prepare_fx(script_model, qconfig_dict)

    calibration_data = [torch.zeros(1, 30, 500, 500)]
    calibration_data.append(torch.ones(1, 30, 500, 500))
    fix = torch.ones(1, 30, 500, 500)
    fix[:,:,:,:250] = 0
    calibration_data.append(fix)
    for i in range(len(calibration_data)):
        model_prepared(calibration_data[i])
    model_quantized = convert_fx(model_prepared)
    traced = torch.jit.trace(model_quantized, (input_tensor))
    traced.save(out_file + ".quant")
    # benchmark

if __name__ == "__main__":
    main()

