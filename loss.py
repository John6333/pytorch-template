__all__ = ['Evaluation']

class Evaluation():
    def __init__(self, criterion, accumulate=False):
        self.current_loss = 0
        self.criterion = criterion
        self.accumulate = accumulate
        if self.accumulate:
            self.accumulated_loss = 0
            self.count = 0

    def get_performance(self, y_pred,y):
        self.current_loss = self.criterion(y_pred, y)
        if self.accumulate:
            self.accumulated_loss += self.current_loss.item()
            self.count += 1
        return self.current_loss

    def get_info(self):
        info = 'loss: {:.6f}'.format(self.current_loss.item())
        if self.accumulate:
            info += ' avg_loss: {:.6f}'.format(self.accumulated_loss/self.count)
        return info

    def get_accumulated_loss(self):
        if self.accumulate:
            return self.accumulated_loss/self.count
        else:
            return None
