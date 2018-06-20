'''
Implemented Functions:
    1.  Multiple GPUs training
    2.  track and save best model on validation set
TODO:
    1.  running mean of model
    2.  plot the validation loss and training loss
    3.  hyper-papameter random search (on a mini sub-set)
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn

import os
import pdb
import copy
import argparse
import time
from datetime import datetime

from data_loader import *
from model import *
from loss  import *

parser = argparse.ArgumentParser()
# training parameters
parser.add_argument('--lr',type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--momentum',type=float, default=0.9, help='SGD momemtum')
parser.add_argument('--weight_decay',type=float, default=1e-4, help='weight decay')
parser.add_argument('--epoch_num', type=int, default=100, help='number of epoches to train')
parser.add_argument('--pretrain', type=str, default='', help='pretrained model to load')
# dataloader parameters
parser.add_argument('--workers', type=int, default=16, help='number of data loading workers')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--subsize', type=float, default=1, help='the propotion of dataset to use')
# model parameters
# other parameters
parser.add_argument('--display', type=int, default=1, help='how often to output results')

opt = parser.parse_args()
print(opt)

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=1):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e5 # set to some large enough value

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
                eva = Evaluation(criterion, accumulate=False)
            else:
                model.eval()   # Set model to evaluate mode
                eva = Evaluation(criterion, accumulate=True)

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i,(inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device).float() # requires_grad=False by default
                labels = labels.to(device).float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = eva.get_performance(outputs, labels)
                    info = eva.get_info()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                info = 'Epoch: {:0>3d} {:0>5d}/{:0>5d} '.format(epoch, i, len(dataloaders[phase])) + info
                print(info)

            # deep copy the model
            if phase == 'val':
                epoch_loss = eva.get_accumulated_loss()
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # saving model
                    torch.save(best_model_wts, 'model.pth')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    # set random seed
    torch.manual_seed(int('0xABCDFE9',16))

    # set True for possible higher speed. Comment it out if running out of memery
    cudnn.benchmark = True

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create Model and DataParallel
    net = get_model(nin=128,nout=128).float()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)

    # print the network
    print(net)

    # load a pretrained model if required
    if (opt.pretrain != ''):
        net.load_state_dict(torch.load(opt.pretrain))

    # choose a loss function
    criterion = nn.MSELoss()

    # choose an optimizer
    #optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    optimizer = optim.SGD( net.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=True)
    #optimizer = optim.RMSprop(net.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    # choose a learning rate scheduler
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.gamma)
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95**epoch)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

    dataloaders = get_dataloaders(batch_size=opt.batch_size, num_workers=opt.workers, subsize=opt.subsize)

    # train model
    train_model(net, dataloaders, criterion, optimizer, scheduler, num_epochs=opt.epoch_num)

