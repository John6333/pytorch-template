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
# dataloader parameters
parser.add_argument('--workers', type=int, default=16, help='number of data loading workers')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--subsize', type=float, default=1, help='the propotion of dataset to use')
# model parameters
parser.add_argument('--pretrain', type=str, default='', help='pretrained model')

# other parameters
parser.add_argument('--display', type=int, default=1, help='how often to output results')

opt = parser.parse_args()
print(opt)

def test_model(model, dataloaders, criterion):
    since = time.time()
    model.eval()   # Set model to evaluate mode
    eva = Evaluation(criterion, accumulate=True)

    phase = 'val'
    # Iterate over data.
    for i,(inputs, labels) in enumerate(dataloaders[phase]):
        inputs = inputs.to(device).float() # requires_grad=False by default
        labels = labels.to(device).float()

        # forward
        with torch.no_grad():
            outputs = model(inputs)
            loss = eva.get_performance(outputs, labels)
            info = eva.get_info()

        if i%opt.display==0:
            info = 'Epoch: {:0>3d} {:0>5d}/{:0>5d} '.format(epoch, i, len(dataloaders[phase])) + info
            print(info)

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    # set random seed
    torch.manual_seed(int('0xABCDFE9',16))

    # set True for possible higher speed. Comment it out if running out of memery
    cudnn.benchmark = True

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create Model and DataParallel
    net = get_model(nin=128,nout=128).float()

    # load a pretrained model if required
    if (opt.pretrain != ''):
        net.load_state_dict(torch.load(opt.pretrain))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)

    # print the network
    print(net)

    # choose a loss function
    criterion = nn.MSELoss()

    dataloaders = get_dataloaders(batch_size=opt.batch_size, num_workers=opt.workers, subsize=opt.subsize)

    # test model
    test_model(net, dataloaders, criterion)

