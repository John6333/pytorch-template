import numpy as np
from torch.utils.data import Dataset, DataLoader

__all__ = ['get_dataloaders']

class BasicDataset(Dataset):
    def __init__(self,training=True, subsize=None):
        '''
        subsize: the proportion in (0,1] to downsample the dataset
        '''
        if training:
            self.x = np.random.rand(100,128)
            self.y = np.random.rand(100,128)
        else:
            self.x = np.random.rand(50,128)
            self.y = np.random.rand(50,128)
        # get a subset if size is not None
        if subsize!=None:
            num_all = self.x.shape[0]
            num = int(num_all*subsize)
            sample_index = np.linspace(0, num_all-1, num, dtype=int)
            self.x = self.x[sample_index]
            self.y = self.y[sample_index]


    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class AugmentedDataset(BasicDataset):
    def __init__(self, training, subsize=None):
        super(AugmentedDataset,self).__init__(training=training, subsize=subsize)
        # ADD INITIALIZATION CODE HERE

    def __len__(self):
        return super(AugmentedDataset,self).__len__()

    def __getitem__(self,i):
        x,y = super(AugmentedDataset,self).__getitem__(i)
        # ADD ANY AUGMENTATION OPERATION HERE
        return x,y

def get_dataloaders(batch_size, num_workers, subsize=None):
    # load data
    train_dataset = AugmentedDataset(training=True,  subsize=subsize)
    test_dataset  = AugmentedDataset(training=False, subsize=subsize)
    # create dataloader
    dataloaders = dict()
    dataloaders['train'] = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    dataloaders['val']  = DataLoader(dataset= test_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    return dataloaders

if __name__ == '__main__':
    import pdb
    dataloaders = get_dataloaders(2,2,0.1)
    for i, (x, y)  in enumerate(dataloaders['train']):
        print(x.size())
        print(y.size())
        pdb.set_trace()

