from torch import optim
import torch
from pdb import set_trace as bp


def build_optimizer_scheduler(config, train_dataloader, model, rank):

    name_optim = config.OPTIMIZER.name
    max_lr = config.SCHEDULER.max_lr
    if "adamw" == name_optim.lower():
        optimizer = optim.AdamW(
            model.parameters(), lr=max_lr, weight_decay=0.001)

    if "rmsprop" == name_optim.lower():
        optimizer = optim.RMSprop(
            model.parameters(), lr=max_lr, weight_decay=0.001)

    if "sgd" == name_optim.lower():
        optimizer = optim.SGD(model.parameters(),
                              lr=max_lr, weight_decay=0.001)

    if "adadelta" == name_optim.lower():
        optimizer = optim.Adadelta(
            model.parameters(), lr=max_lr, weight_decay=0.001)

    if "adam" == name_optim.lower():
        optimizer = optim.Adam(
            model.parameters(), lr=max_lr, weight_decay=0.001)

    if rank == 0:
        print(">>>>> Opimizer:", name_optim.lower())

    if "onecycle" in config.SCHEDULER.name.lower():
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=max_lr, steps_per_epoch=len(train_dataloader), epochs=config.SCHEDULER.num_epochs)

    if "steplr" in config.SCHEDULER.name.lower():
        scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.SCHEDULER.num_epochs*800//3, gamma=0.1)
        
    return optimizer, scheduler
