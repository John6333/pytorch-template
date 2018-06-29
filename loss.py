
__all__ = ['LossMeter']

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg_meter = self.sum / self.count

class LossMeter():
    def __init__(self, criterion, accumulate=False):
        self.criterion = criterion
        self.accumulate = accumulate
        if self.accumulate:
            self.avg_meter = AverageMeter()
        self.info = ''

    def update(self, y_pred,y):
        loss = self.criterion(y_pred, y)
        self.info = f'loss: {loss.item():.6f}'
        if self.accumulate:
            self.avg_meter.update(loss.item())
        return loss

    def get_info(self):
        if self.accumulate:
            self.info += ' avg_loss: {self.avg_meter.avg:.6f}'
        return self.info

    def get_avg(self):
        if self.accumulate:
            return self.avg_meter.avg
        else:
            return 0
