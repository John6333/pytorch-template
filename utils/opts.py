import argparse
parser = argparse.ArgumentParser(description='PyTorch training parser')
# training parameters
parser.add_argument('--lr',             type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--momentum',       type=float, default=0.9, help='SGD momemtum')
parser.add_argument('--wd',             type=float, default=0, help='weight decay')
parser.add_argument('--epoch',          type=int, default=100, help='number of epoches to train')
# dataloader parameters
parser.add_argument('--workers',        type=int, default=16, help='number of data loading workers')
parser.add_argument('--batch_size',     type=int, default=64, help='input batch size')
# model parameters
# other parameters
parser.add_argument('--output','-o',    type=str, default='exp/exp1', help='pretrained model to load')

opt = parser.parse_args()

import torch
opt.device = None
if torch.cuda.is_available():
    opt.device = torch.device('cuda')
else:
    opt.device = torch.device('cpu')
