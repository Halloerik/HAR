from torch.utils import data
import torch


#Origninal code is from:
#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel#disqus-thread
class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dataset, labels, sliding_window_size, sliding_window_step):
        'Initialization'
        self.labels = labels
        self.data = dataset
        self.sliding_window_size = sliding_window_size
        self.sliding_window_step = sliding_window_step
        
    def __len__(self):
        'Denotes the total number of samples'
        return int(self.data.shape()[0]/self.sliding_window_size)

    def __getitem__(self, index):
        'Generates one segment using the index-th sliding_window position'
        #Segment bounds
        lowerbound = index      * self.sliding_window_step
        upperbound = lowerbound + self.sliding_window_size
        # Load segment and get label
        segment = self.data[lowerbound:upperbound, :]
        label = torch.mode(self.labels[lowerbound:upperbound])

        return segment, label
    
    
