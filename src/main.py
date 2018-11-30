import pickle
import numpy as np
import torch

import preprocessing_pamap2 as pamap2
import neuralnetwork
import data_management
from data_management import Attribute_Representation
#torch.set_default_tensor_type('torch.FloatTensor')

#pamap2.generate_data("../datasets/","../datasets/pamap.dat")


file = open("../datasets/pamap.dat", 'rb')
data = pickle.load(file)

gpudevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

attr_representation = Attribute_Representation(12,12)
attr_representation.diagonal_matrix()
#print(attr_representation.attributes)

sliding_window_size = 50
sliding_window_step = 22

training_set    = data_management.Sliding_Window_Dataset(data[0], gpudevice, sliding_window_size, sliding_window_step,attr_representation) 
validation_set  = data_management.Sliding_Window_Dataset(data[1], gpudevice, sliding_window_size, sliding_window_step,attr_representation) 
test_set        = data_management.Sliding_Window_Dataset(data[2], gpudevice, sliding_window_size, sliding_window_step,attr_representation) 

training_loader     = torch.utils.data.DataLoader(training_set  , batch_size=4, shuffle=True, num_workers=0)
validation_loader   = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=True, num_workers=0)
test_loader         = torch.utils.data.DataLoader(test_set      , batch_size=4, shuffle=False, num_workers=0)





numberOfClasses = 12
numberOfAttributes = 24

#0 is heartmonitor
#1-13 is imu1
#14-26 is imu2
#27-39 is imu3
imulist = [1,13,13,13]

net = neuralnetwork.Net(imulist, sliding_window_size, 12, gpudevice, uncertaintyForwardPasses=100)


#training_data   = neuralnetwork.shapeinputs(imulist, training_data   ,gpudevice)
#validation_data = neuralnetwork.shapeinputs(imulist, validation_data ,gpudevice)
#test_data       = neuralnetwork.shapeinputs(imulist, test_data       ,gpudevice)


#
neuralnetwork.train(net, training_loader, validation_loader, attr_representation, "cosine", epochs=10)
neuralnetwork.test(net, test_loader,"test" ,attr_representation)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    