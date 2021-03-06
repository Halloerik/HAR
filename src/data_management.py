from torch.utils.data import Dataset
import torch

import numpy as np
from scipy.spatial import distance

#Origninal code is from:
#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel#disqus-thread
class Sliding_Window_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, gpudevice, sliding_window_size, sliding_window_step):
        super(Sliding_Window_Dataset, self).__init__()
        'Initialization'
        self.gpudevice = gpudevice
        self.data, self.labels = data
        t,d = self.data.shape
        
        #print("t:{} d:{}".format(t,d))
        
        self.data = torch.from_numpy(self.data)
        self.data = self.data.float()
        self.data = self.data.reshape(1,t,d)
        #self.data = self.data.to(device = self.gpudevice)
        
        
        self.labels = torch.from_numpy(self.labels)
        self.labels = self.labels.int()
        #self.labels = self.labels.to(device = self.gpudevice)
        
        self.sliding_window_size = sliding_window_size
        self.sliding_window_step = sliding_window_step
        

        
    def __len__(self):
        'Denotes the total number of samples'
        return int(self.data.shape[1]/self.sliding_window_size)

    def __getitem__(self, index):
        'Generates one segment using the index-th sliding_window position'
        #Segment bounds
        lowerbound = index      * self.sliding_window_step
        upperbound = lowerbound + self.sliding_window_size
        # Load segment and get label
        segment = self.data[:, lowerbound:upperbound, :]
        label = torch.mode(self.labels[lowerbound:upperbound])[0].item()
        
        return segment,label
    
    
class Attribute_Representation():
    def __init__(self,dataset=None, n_classes=None,n_attributes=None):
        if dataset is None:
            self.n_attributes = n_attributes
            self.n_classes = n_classes
            self.attributes = torch.zeros((n_classes,n_attributes))
            self.diagonal_matrix()
        elif dataset is "pamap2":
            self.n_attributes = 24
            self.n_classes = 12               
            self.attributes = torch.tensor([ [1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,0,0,1,1],
                                             [0,1,1,1,0,0,0,1,1,1,1,1,1,1,0,1,0,0,1,1,1,1,0,1],
                                             [1,0,0,1,1,1,0,0,0,0,1,1,1,0,1,0,0,1,1,1,0,0,0,0],
                                             [1,0,1,1,0,1,0,0,0,1,0,0,1,0,0,0,1,1,1,1,0,1,1,0],
                                             [0,1,1,0,0,1,1,1,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,1],
                                             [1,0,1,1,1,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
                                             [0,1,0,1,1,1,1,0,1,0,1,1,0,1,0,1,0,0,0,0,1,0,0,1],
                                             [0,0,0,1,1,0,0,0,1,0,0,0,0,0,1,0,1,1,0,1,0,0,0,0],
                                             [1,1,0,1,0,1,0,0,1,1,0,1,1,0,0,1,0,1,0,0,0,1,0,0],
                                             [1,0,1,0,0,1,0,0,1,1,0,0,1,1,1,1,0,0,1,1,1,1,1,1],
                                             [0,1,0,1,1,0,1,1,1,0,1,0,1,1,1,1,1,0,0,0,1,0,0,1],
                                             [1,0,1,1,1,1,0,1,1,0,1,1,0,0,0,1,1,1,0,0,0,0,1,1] ])
            #print(self.attributes)
        elif dataset is "gestures":
            self.n_attributes = 32
            self.n_classes = 18
            self.attributes = torch.tensor([ [1,0,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,0,1,1,0,0],
                                            [1,0,1,0,1,1,1,1,0,0,0,1,1,0,1,1,0,0,1,0,1,0,0,0,0,1,1,0,0,1,0,1],
                                            [1,1,0,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,0,0,0,1,0,1,0,1,0,1,0,0,1],
                                            [1,0,1,0,0,0,1,0,0,0,1,0,1,1,1,0,0,1,1,0,0,1,0,1,0,1,0,1,1,0,0,1],
                                            [1,1,0,0,1,0,0,0,1,1,0,0,1,0,1,1,1,0,0,1,1,0,1,0,0,0,0,0,0,1,0,0],
                                            [0,1,0,0,0,0,1,1,0,1,0,1,0,0,1,1,0,1,0,0,1,0,1,1,0,1,1,1,0,1,0,1],
                                            [1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1],
                                            [0,1,1,1,1,0,0,1,0,1,0,0,0,1,0,0,1,1,1,0,1,1,1,1,1,0,1,0,0,1,0,0],
                                            [0,0,0,0,0,1,0,0,1,0,1,1,0,0,1,1,0,1,1,0,1,1,0,0,0,1,0,1,1,0,0,1],
                                            [0,1,1,0,0,0,0,1,1,0,1,1,1,0,0,0,1,0,0,0,0,1,0,1,0,1,0,1,0,0,0,1],
                                            [1,1,0,1,0,0,0,1,1,0,0,1,0,0,1,1,1,0,1,1,1,1,0,0,1,1,0,0,0,1,1,1],
                                            [1,0,0,0,0,1,1,0,1,1,1,1,1,1,0,1,1,0,1,1,0,0,0,1,0,0,1,1,1,0,0,0],
                                            [1,1,1,1,1,0,1,0,0,1,0,0,0,1,0,1,1,1,1,0,0,1,1,0,0,1,0,0,1,0,1,0],
                                            [0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,0,0,1,0,1,0,0],
                                            [0,1,1,0,0,0,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1],
                                            [1,0,1,1,0,0,1,1,0,0,0,0,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,0,0,1,0,0],
                                            [1,0,0,1,1,1,1,0,1,1,1,0,1,0,1,0,1,0,0,1,1,0,0,0,1,0,0,1,0,0,1,0],
                                            [1,1,0,1,0,0,0,1,1,1,0,1,0,1,1,1,1,0,1,0,1,1,0,0,0,0,0,0,0,1,1,1] ])
        elif dataset is "locomotion":
            self.n_attributes = 10
            self.n_classes = 5
            self.attributes = torch.tensor([ [1,0,0,0,0,1,0,0,1,1],
                                             [0,1,1,0,1,1,1,1,1,1],  
                                             [0,1,0,0,0,0,0,0,1,1],
                                             [0,0,0,0,1,1,0,0,1,1],
                                             [1,1,1,0,1,0,0,1,1,0] ])
    def diagonal_matrix(self):
        if self.n_attributes is self.n_classes:
            for i in range(self.n_classes):
                self.attributes[i][i] = 1
    
    def closest_class(self,attributeVector, distancemetric="cosine"):
        batch_size = attributeVector.shape[0]
        closestClass = torch.zeros(batch_size)
        
        for batch in range(batch_size):
            
            attrVector = attributeVector[batch,:]
            classVector = self.attributes[0, :]
            
            #print("attrVector {}".format(attrVector.shape))
            #print("classVector {}".format(classVector.shape))
            
            closestDistance = self.distance(attrVector,classVector,distancemetric)
            closestClass[batch] = 0
                
            for i in range(1,self.n_classes):
                attrVector = attributeVector[batch,:]
                classVector = self.attributes[i, :]
                newDistance = self.distance(attrVector,classVector,distancemetric)
                if  newDistance <= closestDistance:
                    closestDistance = newDistance
                    closestClass[batch] = i
            
        return closestClass
        
        
    def distance(self, u,v,distancemetric):
        if distancemetric is "cosine":
            return distance.cosine(u, v)
        elif distancemetric is "euclidean":
            return distance.euclidean(u, v)
        elif distancemetric is "cityblock":
            return distance.cityblock(u, v)
        elif distancemetric is "braycurtis":
            return distance.braycurtis(u, v)
        
            
    def attributevector_of_class(self, class_indeces):
        #print(class_index)
        batch_size = class_indeces.shape[0]
        class_attributes = torch.zeros(batch_size,self.n_attributes)
        for i in range(batch_size):
            class_index = int(class_indeces[i].item())
            class_attributes[i,:] = self.attributes[class_index, :] 
        return class_attributes
    
    
