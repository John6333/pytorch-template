'''
Implemented:
    1.  Multiple GPUs training
    2.  track and save best model on validation set
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from progress.bar import Bar
from utils.logger import Logger
from utils.misc import save_model
from utils.osutils import mkdir_p, isfile, isdir, join
from utils.loss import AverageMeter, Criterion
from utils.opts import opt

import pdb
import time

from dataloader.data_loader import get_dataloaders
from model.model import DemoNet, weight_init

from pprint import pprint
print("\n==================Options=================")
pprint(vars(opt), indent=4)
print("==========================================\n")

def train_model(model, dataloaders, optimizer, scheduler, num_epochs=1):
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = Logger(join(opt.output, f'log.txt'))
    logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss'])
    best_loss = 1e5 # set to some large enough value

    criterion = Criterion()
    epoch_loss  = dict()
    for epoch in range(num_epochs):
        scheduler.step()
        lr = scheduler.get_lr()[-1]
        print(f'Epoch: {epoch+1}/{num_epochs} LR: {lr:.3E}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # Iterate over data.
            batch_time  = AverageMeter()
            data_time   = AverageMeter()
            loss_meter  = AverageMeter()

            end = time.time()
            bar_name = 'Training' if phase=='train' else 'Testing '
            num_batch = len(dataloaders[phase])
            bar = Bar(bar_name, max=num_batch)
            # Iterate over data.
            for i,(inputs, targets) in enumerate(dataloaders[phase]):
                # measure data loading time
                data_time.update(time.time() - end)

                # move data to GPU
                inputs  = inputs.to(device).float()
                targets = targets.to(device).float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss, _ = criterion.eval(outputs, targets)

                # measure accuracy and record loss
                loss_meter.update(loss.item(), inputs.shape[0])
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix  = f'({i+1:04d}/{num_batch:04d}) Data: {data_time.val:.6f}s | Batch: {batch_time.val:.3f}s | Total: {bar.elapsed_td:} | ETA: {bar.eta_td:} | Loss: {loss_meter.avg:.4f}'
                bar.next()
            bar.finish()
            epoch_loss[phase]  = loss_meter.avg

            # save last model 
            if phase == 'train':
                save_model(model, join(opt.output, f'last.pth'))
            else:
                is_best = epoch_loss['val'] < best_loss
                if is_best:
                    best_loss = epoch_loss['val']
                    save_model(model, join(opt.output, f'best.pth'))
        # append logger file
        logger.append(f'{epoch+1:>3d}\t{lr:.3E}\t{epoch_loss["train"]:.6f}\t{epoch_loss["val"]:.6f}')

if __name__ == '__main__':
    # set random seed for reproduction
    torch.manual_seed(int('0xABCDEF',16))

    # set True for possible higher speed. set False for BN stability
    cudnn.benchmark = True

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create Model and DataParallel
    net = DemoNet(nin=128,nout=128)

    # print the number of parameters
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1.0e6))

    # initialize network
    net.apply(weight_init)

    # load a pretrained model if required
    #net.load_state_dict(torch.load('path/to/pretrain.pth'))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)

    # choose an optimizer
    #optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    optimizer = optim.SGD( net.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.wd, nesterov=True)
    #optimizer = optim.RMSprop(net.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    # choose a learning rate scheduler
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.gamma)
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95**epoch)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

    dataloaders = get_dataloaders(batch_size=opt.batch_size, num_workers=opt.workers)

    # create checkpoint dir
    #opt.output = f'{opt.output}'
    if not isdir(opt.output):
        mkdir_p(opt.output)

    # train model
    train_model(net, dataloaders, optimizer, scheduler, num_epochs=opt.epoch)
