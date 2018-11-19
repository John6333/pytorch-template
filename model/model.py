import torch
import torch.nn as nn

__all__ = ['DemoNet', 'weight_init']

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)

class DemoNet(nn.Module):
    def __init__(self,nin=128,nout=1024):
        super(DemoNet,self).__init__()
        self.linear = nn.Linear(nin,nout)
        self.bn = nn.BatchNorm1d(nout)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self._weight_init()

    def _weight_init(self):
        gain = nn.init.calculate_gain('relu')
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight,gain=gain)
            elif isinstance(m,nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal(m.weight,gain=gain)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0,0.02)
                m.bias.data.fill_(0)

    def forward(self,x):
        return self.dropout(self.relu(self.bn(self.linear(x))))
