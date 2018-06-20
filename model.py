import torch
import torch.nn as nn

__all__ = ['get_model']

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

def get_model(nin=128, nout=256):
    return DemoNet(nin=nin, nout=nout)

# testing model
import numpy as np
import torch.optim as optim
from torch.utils.data import  Dataset,DataLoader

class DummyDataset(Dataset):
    def __init__(self):
        self.x = np.random.rand(1000,128)
        self.y = np.random.rand(1000,256)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self,item):
        return self.x[item], self.y[item]

if __name__ == '__main__':
    critical = nn.MSELoss()
    dataset = DummyDataset()
    dataLoader = DataLoader(dataset=dataset,batch_size=64,shuffle=True)
    net = DemoNet(nin=128, nout=256).cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epoch = 5
    display = 1
    for e in range(epoch):
        for i,(x,y) in enumerate(dataLoader):
            x = torch.tensor(x,requires_grad=False).float().cuda()
            y = torch.tensor(y,requires_grad=False).float().cuda()
            y_pred = net.forward(x)
            loss = critical(y_pred,y)
            if i%display==0:
                print('loss : {}'.format(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
