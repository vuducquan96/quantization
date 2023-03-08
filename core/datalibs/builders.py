import os
import copy
import time
import torch
from termcolor import colored
from pdb import set_trace as bp
from configs.system_config import *
from .sr_dataset import lidar_dataset
from .sr_dataset_2d import lidar_dataset_2d
from core.box_coders import GroundBox3dCoder
from core.sample_ops import DataBaseSamplerV2
from core.datalibs.sr_dataset import batch_merge
from core.datalibs.sr_dataset_2d import batch_merge_2d
from core.datalibs.waymo_dataset import WaymoDataset
from core.datalibs.kitti_dataset import KittiDataset
from core.datalibs.sr_dataset import get_dataset_list
from core.datalibs.augment_points_cloud import False_Collect_Offline
# from spconv.utils import PointToVoxel as VoxelGenerator
from core.visualize_lib import debug_draw
from core.models.utils import VoxelGenerator
from core.target_assigner_builder import build as target_assigner_builder
from core.datalibs.advance_insert_object_augmentation import AdvanceInsert
from core.datalibs.datasets import build_dataloader, build_dataset
from core.datalibs.semi_dataset import semi_dataset
from core.datalibs.waymo_seq_dataset import WaymoSeqDataset, batch_merge_seq


def building_semi_dataset(root, config, batch_size):

    print(">>> Semi dataset <<<")

    voxel_generator = VoxelGenerator(
        voxel_size=config["VOXEL_GENERATOR"]["VOXEL_SIZE"],
        point_cloud_range=config["POINT_CLOUD_RANGE"],
        # num_point_features=3, 
        max_num_points_per_voxel=config["VOXEL_GENERATOR"]["MAX_NUMBER_OF_POINTS_PER_VOXEL"],
        max_num_voxels=config["VOXEL_GENERATOR"]["MAX_VOXELS"]
    )

    dataset = semi_dataset(root, config, voxel_generator) 

    the_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=number_worker_per_gpu,
        drop_last=True,
        pin_memory=True,
        collate_fn=batch_merge)
    
    return the_dataloader

def building_dataset(root, config, model_config, mode, dist, logger, batch_size, kfold_mode=False):

    voxel_generator = VoxelGenerator(
            voxel_size=config["VOXEL_GENERATOR"]["VOXEL_SIZE"],
            point_cloud_range=config["POINT_CLOUD_RANGE"],
            # point_cloud_range
            # num_point_features=3, 
            max_num_points_per_voxel=config["VOXEL_GENERATOR"]["MAX_NUMBER_OF_POINTS_PER_VOXEL"],
            max_num_voxels=config["VOXEL_GENERATOR"]["MAX_VOXELS"]
        )

    if config.DATASET == "DEBUG":
        print(">>> Start building debug dataset <<<")
        dataset = build_dataset(config.data.val)
        the_dataloader = build_dataloader(
            dataset,
            batch_size=config.data.samples_per_gpu,
            workers_per_gpu=config.data.workers_per_gpu,
            dist=dist,
            shuffle=False,
        )
        if dist == True:
            data_sampler = torch.utils.data.distributed.DistributedSampler(
                the_dataset, shuffle=True)
        else:
            data_sampler = None

        return the_dataloader, data_sampler

    if config.DATASET == "WaymoSequential":
        print(">>> Start build waymo dataset <<<")
        db_sampler_1 = AdvanceInsert(
            root=root,
            config=config,
            reset_cluster_database=reset_cluster_database)
        the_dataset = WaymoSeqDataset(db_sampler=db_sampler_1, 
                                      mode=mode)
        the_dataset[0]

        if dist == True:
            data_sampler = torch.utils.data.distributed.DistributedSampler(
                the_dataset, shuffle=True)
        else:
            data_sampler = None

        the_dataloader = torch.utils.data.DataLoader(
            the_dataset,
            batch_size=batch_size,
            shuffle=("train" in mode.lower()) and (data_sampler is None),
            num_workers=number_worker_per_gpu,
            drop_last=True,
            pin_memory=True,
            collate_fn=batch_merge_seq)

        if debugging == True:
            if mode == "train" or "val" in mode.lower():
                show_list = [109, 110, 111, 0, 1, 2, 3, 4, 5]
                for i in show_list:
                    if i < len(the_dataset):
                        now = time.time()
                        sample = the_dataset[i]
                        gt_boxes = sample["gt_boxes"]
                        gt_class = []
                        for i in range(len(gt_boxes)):
                            if gt_boxes[i, -1] == 1:
                                gt_class.append("CAR")

                            if gt_boxes[i, -1] == 2:
                                gt_class.append("PED")

                            if gt_boxes[i, -1] == 3:
                                gt_class.append("CYC")
                        debug_draw(pointcloud=sample["points"][:, :3], intensity=sample["points"]
                                   [:, 2], min_i=-2, max_i=4, gt_box=gt_boxes, gt_name=gt_class)
                        total_time = time.time() - now
                        show_str = "Data_id: " + \
                            str(i) + "-> " + "{:.3f}".format(total_time)
                        print(colored(show_str, "white", "on_blue"))
                    else:
                        print(">>> Out of dataset <<<")

                exit()

        return the_dataloader, data_sampler

    if config.DATASET == "WaymoDataset":
        print(">>> Start build waymo dataset <<<")
        the_dataset = WaymoDataset(
            dataset_cfg=config,
            class_names=config.CLASS_NAMES,
            training=("train" in mode.lower()),
            logger=logger
        )

        if dist == True:
            data_sampler = torch.utils.data.distributed.DistributedSampler(
                the_dataset, shuffle=True)
        else:
            data_sampler = None

        the_dataloader = torch.utils.data.DataLoader(
            the_dataset,
            batch_size=batch_size,
            shuffle=("train" in mode.lower()) and (data_sampler is None),
            num_workers=number_worker_per_gpu,
            drop_last=True,
            pin_memory=True,
            collate_fn=the_dataset.collate_batch,
            sampler=data_sampler)

        if debugging == True:
            if mode == "train" or "val" in mode.lower():
                show_list = [109, 110, 111, 0, 1, 2, 3, 4, 5]

                for i in show_list:
                    if i < len(the_dataset):
                        now = time.time()
                        sample = the_dataset[i]
                        gt_boxes = sample["gt_boxes"]
                        gt_class = []
                        for i in range(len(gt_boxes)):
                            if gt_boxes[i, -1] == 1:
                                gt_class.append("CAR")

                            if gt_boxes[i, -1] == 2:
                                gt_class.append("PED")

                            if gt_boxes[i, -1] == 3:
                                gt_class.append("CYC")
                        debug_draw(pointcloud=sample["points"][:, :3], intensity=sample["points"]
                                   [:, 2], min_i=-2, max_i=4, gt_box=gt_boxes, gt_name=gt_class)
                        total_time = time.time() - now
                        show_str = "Data_id: " + \
                            str(i) + "-> " + "{:.3f}".format(total_time)
                        print(colored(show_str, "white", "on_blue"))
                    else:
                        print(">>> Out of dataset <<<")

                exit()

    if config.DATASET == "KittiDataset":
        print(">>> Start building kitti dataset <<<")
        the_dataset = KittiDataset(
            dataset_cfg=config,
            class_names=config.CLASS_NAMES,
            training=("train" in mode.lower()),
            logger=logger
        )
        if dist == True:
            data_sampler = torch.utils.data.distributed.DistributedSampler(
                the_dataset, shuffle=True)
        else:
            data_sampler = None

        the_dataloader = torch.utils.data.DataLoader(
            the_dataset,
            batch_size=batch_size,
            shuffle=("train" in mode.lower()) and (data_sampler is None),
            num_workers=number_worker_per_gpu,
            drop_last=True,
            pin_memory=True,
            collate_fn=the_dataset.collate_batch,
            sampler=data_sampler)

        if debugging == True:
            if mode == "train" or "val" in mode.lower():
                show_list = [0, 1, 2, 3, 4, 5]

                for i in show_list:
                    if i < len(the_dataset):
                        now = time.time()
                        sample = the_dataset[i]
                        gt_boxes = sample["gt_boxes"]
                        gt_class = []
                        for i in range(len(gt_boxes)):
                            if gt_boxes[i, -1] == 1:
                                gt_class.append("CAR")

                            if gt_boxes[i, -1] == 2:
                                gt_class.append("PED")

                            if gt_boxes[i, -1] == 3:
                                gt_class.append("CYC")
                        bp()
                        debug_draw(pointcloud=sample["points"][:, :3], intensity=sample["points"]
                                   [:, 2], min_i=-2, max_i=4, gt_box=gt_boxes, gt_name=gt_class)
                        total_time = time.time() - now
                        show_str = "Data_id: " + \
                            str(i) + "-> " + "{:.3f}".format(total_time)
                        print(colored(show_str, "white", "on_blue"))
                    else:
                        print(">>> Out of dataset <<<")

                exit()
        # bp()

    if config.DATASET == "SeoulRobotics2D":
        print(">>> Start building seoul robotic dataset <<<")
        list_of_file = get_dataset_list(mode + "_data", config, root)
        box_coders = copy.deepcopy(GroundBox3dCoder())
        if "anchor" in model_config.TARGET_ASSIGNER.name.lower():
            anchor_box_function = copy.deepcopy(target_assigner_builder(config, box_coders))
        else:
            anchor_box_function = None
        if "train" in mode.lower():
            db_sampler = []
            db_sampler = None
            false_sampling_offline = copy.copy(False_Collect_Offline(detection_range=config["POINT_CLOUD_RANGE"],
                                                   folder_path=os.path.join(root,config["FALSE_SAMPLE"]["FOLDER_PATH"]),
                                                   max_sample= config["FALSE_SAMPLE"]["MAX_SAMPLE"]))
            db_sampler_1 = AdvanceInsert(
                root=root,
                config=config,
                reset_cluster_database=reset_cluster_database)
        else:
            db_sampler_1 = None
            db_sampler = None
            false_sampling_offline = None
        # bp()
        the_dataset = lidar_dataset_2d(false_sampling_offline = false_sampling_offline,
                                    list_of_file=list_of_file,
                                    mode=mode,
                                    root = root,
                                    config=config,
                                    voxel_generator=voxel_generator,
                                    data_base_sampler=db_sampler_1,
                                    data_base_sampler_1=db_sampler,
                                    anchor_box_function=anchor_box_function,
                                    logger=logger,
                                    num_gpus=1,
                                    partition_id=0
                                    )
        if dist == True:
            data_sampler = torch.utils.data.distributed.DistributedSampler(the_dataset, shuffle=True)
        else:
            data_sampler = None

        the_dataloader = torch.utils.data.DataLoader(
            the_dataset,
            batch_size=batch_size,
            shuffle=("train" in mode.lower()) and (data_sampler is None),
            num_workers=number_worker_per_gpu,
            drop_last=True,
            pin_memory=True,
            collate_fn=batch_merge_2d,
            sampler=data_sampler)
    else:
        print(">>> Start building seoul robotic dataset <<<")
        list_of_file = get_dataset_list(mode + "_data", config, root)
        box_coders = copy.deepcopy(GroundBox3dCoder())
        if "TARGET_ASSIGNER" in model_config:
            anchor_box_function = copy.deepcopy(target_assigner_builder(config, box_coders))
        else:
            anchor_box_function = None
        if "train" in mode.lower():
            db_sampler = []
            db_sampler = None
            false_sampling_offline = copy.copy(False_Collect_Offline(detection_range=config["POINT_CLOUD_RANGE"],
                                                   folder_path=os.path.join(root,config["FALSE_SAMPLE"]["FOLDER_PATH"]),
                                                   max_sample= config["FALSE_SAMPLE"]["MAX_SAMPLE"]))
            db_sampler_1 = AdvanceInsert(
                root=root,
                config=config,
                reset_cluster_database=reset_cluster_database)
            # bp()
        else:
            db_sampler_1 = None
            db_sampler = None
            false_sampling_offline = None
        the_dataset = lidar_dataset(false_sampling_offline = false_sampling_offline,
                                    list_of_file=list_of_file,
                                    mode=mode,
                                    root = root,
                                    config=config,
                                    voxel_generator=voxel_generator,
                                    data_base_sampler=db_sampler_1,
                                    data_base_sampler_1=db_sampler,
                                    anchor_box_function=anchor_box_function,
                                    logger=logger,
                                    num_gpus=1,
                                    partition_id=0
                                    )
        the_dataset[0]
        
        if dist == True:
            data_sampler = torch.utils.data.distributed.DistributedSampler(the_dataset, shuffle=True)
        else:
            data_sampler = None


        if kfold_mode == True:
            return the_dataset, data_sampler

        the_dataloader = torch.utils.data.DataLoader(
            the_dataset,
            batch_size=batch_size,
            shuffle=("train" in mode.lower()) and (data_sampler is None),
            num_workers=number_worker_per_gpu,
            drop_last=True,
            pin_memory=True,
            collate_fn=batch_merge,
            sampler=data_sampler)
    if debugging == True:
        if mode == "train" or "val" in mode.lower():
            show_list = [100, 200, 1000, 100, 200, 300, 2, 3, 4, 5, 60]

            for i in show_list:
                if i < len(the_dataset):
                    now = time.time()
                    the_dataset[i]
                    total_time = time.time() - now
                    show_str = "Data_id: " + \
                        str(i) + "-> " + "{:.3f}".format(total_time)
                    print(colored(show_str, "white", "on_blue"))
                else:
                    print(">>> Out of dataset <<<")

            exit()
    
        
    return the_dataloader, data_sampler


def building_partition_dataset(list_of_file, root, config, model_config, gpu_id, mode):

    box_coders = copy.deepcopy(GroundBox3dCoder())
    if "TARGET_ASSIGNER" in model_config:
        anchor_box_function = copy.deepcopy(
            target_assigner_builder(config, box_coders))
    else:
        anchor_box_function = None
    
    voxel_generator = copy.deepcopy(VoxelGenerator(
        voxel_size=config["voxel_generator"]["voxel_size"],
        # num_point_features=3, 
        point_cloud_range=config["detection_range"],
        max_num_points_per_voxel=config["voxel_generator"]["max_number_of_points_per_voxel"],
        max_num_voxels=config["voxel_generator"]["MAX_VOXELS"]
    ))

    if "train" in mode.lower():
        min_num_points_in_box = config["min_num_points_in_box"]
        min_points_in_box_augment = config["data_dir"]["min_points"]
        detection_range = config["detection_range"]
        db_sampler = []
        for db_config in config["data_dir"]["data_base_sampler"]:
            db_sampler.append(None)
#            db_sampler.append(DataBaseSamplerV2(root,
#                                          db_config,
#                                          min_points_in_box_augment,
#                                          detection_range
#                                          ))
        false_sampling_offline = copy.copy(False_Collect_Offline(detection_range=config["detection_range"],
                                                                 folder_path=os.path.join(
                                                                     root, config["false_sample"]["folder_path"]),
                                                                 max_sample=config["false_sample"]["max_sample"]))
        simple_insert_object = copy.deepcopy(AdvanceInsert(root=root,
                                                           config=config,
                                                           reset_cluster_database=reset_cluster_database))
    else:
        simple_insert_object = None
        db_sampler = None
        false_sampling_offline = None

    

    train_dataset = lidar_dataset(false_sampling_offline=false_sampling_offline,
                                  list_of_file=list_of_file,
                                  mode=mode,
                                  root=root,
                                  config=config,
                                  voxel_generator=voxel_generator,
                                  data_base_sampler=simple_insert_object,
                                  data_base_sampler_1=db_sampler,
                                  anchor_box_function=anchor_box_function,
                                  detection_class=config["detection_class"],
                                  class_encode=config["class_encode"],
                                  out_size_factor=config["target_assigner"]["out_size_factor"],
                                  num_gpus=numb_gpus,
                                  partition_id=gpu_id
                                  )

    if debugging == True and "train" in mode.lower():
        if mode == "train":
            show_list = [1, 2, 3, 4, 5, 20, 1500, 1200,
                         1000, 212, 400, 423, 212, 300, 211]
#            show_list = [1,2,3,4,5]

            for i in show_list:
                if i < len(train_dataset):
                    now = time.time()
                    train_dataset[i]
                    total_time = time.time() - now
                    show_str = "Data_id: " + \
                        str(i) + "-> " + "{:.3f}".format(total_time)
                    print(colored(show_str, "white", "on_blue"))
                else:
                    print(">>> Out of dataset <<<")

            exit()

    the_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=("train" in mode.lower()),
        num_workers=number_worker_per_gpu,
        pin_memory=True,
        collate_fn=batch_merge)

    return the_dataloader
