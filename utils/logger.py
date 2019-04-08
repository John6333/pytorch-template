import torch

def save_model(model, path):
    if hasattr(model, 'module'): # wrapped by nn.DataParallel
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger():
    def __init__(self, fpath, head_info='', nsf=4, gap=2):
        self.nsf = nsf            # number of significant figures, 1.234e-04, nsf+5 ch
        self.fpath = fpath
        self.data_dict = dict()
        self.lines = 0
        self.keys = []
        self.width_per_col = []
        self.gap = gap
        with open(self.fpath, 'w') as f:
            f.write(f'head: {head_info}\n')

    def add_title(self):
        with open(self.fpath, 'a') as f:
            title = ''
            for key in self.data_dict.keys():
                self.keys.append(key)
                width = max(self.nsf+5+self.gap, len(key)+self.gap)
                self.width_per_col.append(width)
                title  += key + ' '*(width-len(key))
            f.write(title+'\n')

    def update(self, value_dict):
        self.data_dict.update(value_dict)
        self.log()

    def log(self):
        with open(self.fpath, 'a') as f:
            if self.lines==0:
                self.add_title()
            info = ''
            for key, width in zip(self.keys, self.width_per_col):
                val = self.data_dict[key]
                str_val = self.to_str(val)
                info += str_val + ' '*(width-len(str_val))
            f.write(info+'\n')
        self.lines += 1

    def to_str(self, val):
        if type(val) is int:
            str_val = f'{val:0>{self.nsf}d}'
        elif val>=1e3 or val<=1:
            str_val = f'{val:.{self.nsf-1}e}'
        elif val>=1e2:
            str_val = f'{val:.{self.nsf-3}f}'
        elif val>=10:
            str_val = f'{val:.{self.nsf-2}f}'
        else:
            str_val = f'{val:.{self.nsf-1}f}'
        return str_val



if __name__=='__main__':
    logger = Logger('exp/test_log.txt', nsf=8, gap=4)
    data_dict = dict()
    for i in range(10):
        data_dict['Epoch'] = i
        data_dict['LR'] = 1e-4
        data_dict['Train loss'] = 0.05343213
        data_dict['Train acc and basadavfaveaf'] = 14.35343213
        data_dict['val Loss'] = 0.000000124405835
        logger.update(data_dict)

