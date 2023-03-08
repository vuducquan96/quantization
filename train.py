import os
import time
import copy
import torch
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
from termcolor import colored
from core import common_utils
from pdb import set_trace as bp
from core.torchplus import get_lr
from configs.system_config import *
import torch.backends.cudnn as cudnn
from core.tools import resume_model
from core.box_coders import GroundBox3dCoder
from core.models.builders import model_builder
from core.tools.config_loader import config_loader
from core.optimizer import build_optimizer_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from core.common_utils import init_dist_pytorch
from core.datalibs.sr_dataset import sample_convert_to_torch
from core.datalibs.builders import building_dataset


def parse_config():
    #----------------
    # Argument Parser
    #----------------

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model', type=str, default="configs/models/bev.toml", help='specify the config for training')
    parser.add_argument('--dataset', nargs='+', default=["configs/dataset/base.yaml"], help='specify the config for training')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=8888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')

    args = parser.parse_args()
    dataset_config = []

    for dataset_config_path in args.dataset:
        dataset_config.append(config_loader(dataset_config_path))

    model_config_path = args.model
    model_config = config_loader(model_config_path)
    return dataset_config, model_config, args

def Training(train_dataloader, model, optimizer, scheduler, epoch, rank, logger, semi_set, swa_mode, num_epoch):

    ##########################################################
    ## This function is for single epoch training
    ## Args:
    ## + Train_dataloader: Dataloader for training
    ## + Model: target model
    ## + Optimizer: target optimizer
    ## + Scheduler: scheduler learning rate
    ## + Epoch: current epochW
    ## + Rank: Current rank id of process
    ##########################################################

    if rank == 0:
        print("Start training epoch:", epoch)

    model.train()
    train_loss, train_loc_loss, train_cls_loss = 0, 0, 0
    batch_loss = 0
    starting_epoch = time.time()
    counter = 0
    now = time.time()
    total_data = len(train_dataloader)
    if semi_set is not None:
        semi_dataloader = semi_set["semi_dataloader"]
        dataiter = iter(semi_dataloader)
        
    for g in optimizer.param_groups:
        g['lr'] = 0.0001

    for batch_idx, sample in enumerate(tqdm(train_dataloader)):
        if sample is None:
            continue
        counter += 1
        if semi_set is not None:
            try:
                semidata = dataiter.next()
            except:
                print(">>> reset semi <<<")
                dataiter = iter(semi_dataloader)
                semidata = dataiter.next()
        sample = sample_convert_to_torch(sample, dtype = torch.float32,device="cuda")
        if semi_set is not None:
            semidata = sample_convert_to_torch(semidata, dtype=torch.float32,device="cuda")
            with torch.no_grad():
                semidata["semidata"] = copy.copy(semi_set["teacher_model"](semidata, "semi"))


        loss_dict = model(sample)
        if semi_set is not None:
            semi_loss = model(semidata, "semi_train")
        if semi_set is not None:
            loss = loss_dict["loss"] + semi_loss["loss"]
        else:
            loss = loss_dict["loss"]

        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
        optimizer.step()
        optimizer.zero_grad()
        show_lr = copy.copy(get_lr(optimizer))


        #if swa_mode:
        #    if epoch > swa_start:
        #        # print(">>> Start SWA model <<<")
        #        swa_model.update_parameters(model)
        #        swa_scheduler.step()
        #    else:
        #        scheduler.step()
        #else:    
        #    scheduler.step()


        train_loss += loss.cpu().detach().numpy()
        batch_loss += loss.mean().cpu().detach().numpy()
        if "_loc_loss" in loss_dict:
            train_loc_loss += loss_dict["_loc_loss"].mean().cpu().detach().numpy()

        if "_cls_loss" in loss_dict:
            train_cls_loss += loss_dict["_cls_loss"].mean().cpu().detach().numpy()
        # batch_loss += loss_dict["_loc_loss"].mean().cpu().detach().numpy() + loss_dict["_cls_loss"].mean().cpu().detach().numpy()

        if rank == 0:
            if (batch_idx + 1 ) % showing_frequency == 0:
                loss_now = batch_loss / counter
                loc_loss_now = train_loc_loss / counter
                cls_loss_now = train_cls_loss / counter
                time_now = time.time() - now
                show_loss = loss_now.item()
                if "_loc_loss" in loss_dict:
                    show_loc_loss = loc_loss_now.item()
                else:
                    show_loc_loss = 0
                if "_cls_loss" in loss_dict:
                    show_cls_loss = cls_loss_now.item()
                else:
                    show_cls_loss = 0

                printing_loss = "Data: {} / {} Loss:{:.6f} = [ Loc_loss:{:.6f} Cls_loss:{:.4f} ] LR:{:.5f} Time:{:.2f} s".format(batch_idx,
                                    total_data,
                                    loss_now,
                                    show_loc_loss,
                                    show_cls_loss,
                                    show_lr,
                                    time_now)
                logger.info(printing_loss)
                now = time.time()
                counter = 0
                batch_loss, train_loc_loss, train_cls_loss= 0, 0, 0

    if rank == 0:
        epoch_loss = train_loss / len(train_dataloader)
        epoch_loss = epoch_loss.item()
        
        logger.info("Loss of epoch: {:.5f}".format(epoch_loss))
        logger.info("Time per epoch: {:.2f} s".format(time.time() - starting_epoch))
    if swa_mode and epoch==num_epoch:
        torch.optim.swa_utils.update_bn(train_dataloader, swa_model)
        return swa_model, epoch > swa_start
    

def root_checking(root_list):
    ##########################################################
    ## Automatic detection root of data
    ## Args:
    ## + root_list: List of data
    ## Return root of data (string)
    ##########################################################

    root = ""
    for root_dir in root_list:
        if os.path.isdir(root_dir) == True:
            return root_dir
    return root


def main():
    #--------------
    # Main function
    #--------------
    cudnn.benchmark = True
    data_config, model_config, args = parse_config()
    if args.local_rank is None:
        distributed_training = False
        rank = 0
    else:
        numb_gpus, rank = init_dist_pytorch(args.tcp_port, args.local_rank)
        distributed_training = True

    working_place =  os.path.join("working_path", model_config.experiment_name)
    working_place = Path(working_place)
    working_place.mkdir(parents=True, exist_ok=True)

    #### Logger ####
    if "root_list" in data_config[0]:

        root = root_checking(data_config[0]["root_list"])
    else:
        root = None 
    log_file = working_place / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=rank)
    
    semi_dataloader = None
    teacher_model = None

    semi_set = {}
    semi_set["semi_dataloader"] = semi_dataloader
    semi_set["teacher_model"] = teacher_model
    semi_set = None
    list_data_loader = []
    list_data_sampler = []
    
    for data_cf in data_config:
        data_loader, data_sampler = building_dataset(root, data_cf, model_config, "train", distributed_training, logger, args.batch_size, kfold_mode=False)

        list_data_loader.append(data_loader)
        list_data_sampler.append(data_sampler)

    model = model_builder(model_config= model_config,
                            data_config=data_config[0],
                            box_coders=GroundBox3dCoder(),
                            anchor_box_function=None)
    model.cuda()
    optimizer, scheduler = build_optimizer_scheduler(model_config, data_config[0], model, rank)

    if pretrain in ['resume', 'load']:
        model, optimizer , _ , first_epoch = resume_model.resume(model, optimizer, None, working_place, rank, logger, scheduler, pretrain)
    else:
        first_epoch = 0
        if rank == 0:
            print(colored(">>> No Loaded pretrained <<<", "red"))
    
    if distributed_training == True:
        model = DDP(model, device_ids=[rank])

    for epoch in range(first_epoch, int(model_config["SCHEDULER"]["num_epochs"])):
        for data_loader, data_sampler in zip(list_data_loader, list_data_sampler):
            if distributed_training == True:
                data_sampler.set_epoch(epoch)

            Training(data_loader, model, optimizer, scheduler, epoch, rank, logger, semi_set, swa_mode=False, num_epoch=int(model_config["SCHEDULER"]["num_epochs"]))

        if rank == 0:
            checkpoint = resume_model.checkpoint_state(model=model, optimizer=optimizer, epoch=epoch, scheduler=scheduler)
            file_name = model_config["base"] + "_" + f"{epoch}.pth"
            saving_file = working_place / file_name
            print("Saving model: ", saving_file)
            torch.save(checkpoint, saving_file)


if __name__ == "__main__":
    main()
