import torch

__all__ = ['Criterion', 'AverageMeter']

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count

def _loss_criterion(pred, gt):
    criterion = torch.nn.MSELoss()
    return criterion(pred, gt)

def _accuracy(pred, gt):
    with torch.no_grad():
        ac = (pred==gt).sum().float()/gt.numel()
    return ac

class Criterion():
    def __init__(self):
        pass

    def eval(self, pred, gt):
        loss = _loss_criterion(pred, gt)
        ac   = _accuracy(pred, gt)
        return loss, ac

