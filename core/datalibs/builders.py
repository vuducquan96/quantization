import os
import copy
import time
import torch
from termcolor import colored
from configs.system_config import *
from .sr_dataset import lidar_dataset
from core.datalibs.sr_dataset import batch_merge
from core.datalibs.sr_dataset import get_dataset_list
from core.models.utils import VoxelGenerator
from core.datalibs.advance_insert_object_augmentation import AdvanceInsert

def building_dataset(root, config, model_config, mode, dist, logger, batch_size, kfold_mode=False):

    voxel_generator = VoxelGenerator(
            voxel_size=config["VOXEL_GENERATOR"]["VOXEL_SIZE"],
            point_cloud_range=config["POINT_CLOUD_RANGE"],
            # point_cloud_range
            # num_point_features=3, 
            max_num_points_per_voxel=config["VOXEL_GENERATOR"]["MAX_NUMBER_OF_POINTS_PER_VOXEL"],
            max_num_voxels=config["VOXEL_GENERATOR"]["MAX_VOXELS"]
        )

    print(">>> Start building seoul robotic dataset <<<")
    list_of_file = get_dataset_list(mode + "_data", config, root)
    if "train" in mode.lower():
        db_sampler = []
        db_sampler = None
        false_sampling_offline = None
        db_sampler_1 = AdvanceInsert(
            root=root,
            config=config,
            reset_cluster_database=reset_cluster_database)
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
                                anchor_box_function=None,
                                logger=logger,
                                num_gpus=1,
                                partition_id=0
                                )
    the_dataset[0]
        
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
