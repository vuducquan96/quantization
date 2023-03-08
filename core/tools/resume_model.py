import torch
import os
import re 
import glob
from termcolor import colored
from pdb import set_trace as bp

def load_params_from_file(model, filename, logger, to_cpu=False):

    if logger is not None:
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
    else:
        print('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
    loc_type = torch.device('cpu') if to_cpu else None
    checkpoint = torch.load(filename, map_location=loc_type)
    model_state_disk = checkpoint['net']

    update_model_state = {}
    for key, val in model_state_disk.items():
        if key in model.state_dict() and model.state_dict()[key].shape == model_state_disk[key].shape:
            update_model_state[key] = val
            # logger.info('Update weight %s: %s' % (key, str(val.shape)))

    state_dict = model.state_dict()
    state_dict.update(update_model_state)
    model.load_state_dict(state_dict)

    for key in state_dict:
        if key not in update_model_state:

            if logger is not None:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))
            else:
                print('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

    if logger is not None:
        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(model.state_dict())))
    else:
        print('==> Done (loaded %d/%d)' % (len(update_model_state), len(model.state_dict())))
    return model


def checkpoint_state(model=None, optimizer=None, epoch=None, scheduler=None):
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    checkpoint_data = {'epoch': epoch, 
                       'net': model_state,
                       'optimizer': optimizer_state,
                       'scheduler': scheduler.state_dict()}
    # import pdb; pdb.set_trace()
    return checkpoint_data

def get_epoch(name):
    return int(re.sub("[^0-9]", "", name))

def resume(model, optimizer, amp, working_place, rank, logger, scheduler, mode):
    all_resume_file = glob.glob(str(working_place)+"/*.pth")
    start_epoch = 0
    if len(all_resume_file) > 0:
        newest_file = all_resume_file[0]
        for check_file in all_resume_file:
           if get_epoch(check_file) > get_epoch(newest_file):
            # if get_epoch(check_file) == 9:
                newest_file = check_file
        
        print("Resume from file:", newest_file)
        checkpoint = torch.load(newest_file, map_location="cuda")
        model_load = False
        if model != None:
            try:
                model = load_params_from_file(model, newest_file, logger, False)
                model_load = True
                if rank == 0:
                    print(colored(">>> Resume training <<<", "yellow"))
            except:
                if rank == 0:
                    print(colored("Can't load model state !!!", "blue"))
        if mode == 'resume':
            if optimizer is not None:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                except:
                    if rank == 0:
                        print(colored("Can't load optimizer state !!!","red"))
            
            if scheduler is not None: 
                try:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                except:
                    if rank == 0:
                        print(colored("Can't load scheduler state !!!","red"))

            if amp is not None:
                try:
                    amp.load_state_dict(checkpoint["amp"])
                except:
                    if rank == 0:
                        print(colored("Can't load apex state !!!","green"))
            
            if model_load == True:
                try:
                    start_epoch = int(checkpoint["epoch"]) + 1
                except:
                    start_epoch = 0
                    if rank == 0:
                        print(">>> Can't resume epoch <<<")
            else:
                start_epoch = 0
                if rank == 0:
                    print(">>> Can't resume epoch <<<")
        elif mode == 'load':
            start_epoch=0
    else:
        if rank == 0:
            print(">>> No file for resume in training folder <<<")

    return model, optimizer, amp, start_epoch 
    
