import pickle
import numpy as np
import torch

import preprocessing_pamap2 as pamap2
import neuralnetwork
from data_management import Sliding_Window_Dataset
from data_management import Attribute_Representation

#pamap2.generate_data("../datasets/","../datasets/pamap.dat")
    
    
def load_data(dataset,gpudevice,batch_size,sliding_window_size,sliding_window_step):
    
    
    if dataset is "pamap2":
        file = open("../datasets/pamap.dat", 'rb')
        imulist = [1,13,13,13]
    data = pickle.load(file)

    training_set    = Sliding_Window_Dataset(data[0], gpudevice, sliding_window_size, sliding_window_step) 
    validation_set  = Sliding_Window_Dataset(data[1], gpudevice, sliding_window_size, sliding_window_step) 
    test_set        = Sliding_Window_Dataset(data[2], gpudevice, sliding_window_size, sliding_window_step) 

    training_loader     = torch.utils.data.DataLoader(training_set  , batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader   = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader         = torch.utils.data.DataLoader(test_set      , batch_size=batch_size, shuffle=False, num_workers=0)
    
    return training_loader, validation_loader, test_loader, imulist


def get_optimiser(network, optimizer, learning_rate, weight_decay,momentum):
    if optimizer is "SGD":
        return torch.optim.SGD(network.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        

def main():
    data_set = ["pamap2"]
    batch_size = [128]
    sliding_window_size = 100
    sliding_window_step = 22
    
    #Training Parameters
    epochs = 10
    learning_rate = [0.0001]
    weight_decay = [0.0001]
    momentum = [0.9]
    loss_critereon = [torch.nn.CrossEntropyLoss()]
    optimizer = ["SGD"]
    
    
    #Network parameters
    kernelsize = [(5,1)]
    
    
    
    gpu_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #Attribute Representation
    #number_of_attributes = [12]
    #Uncertainty
    #uncertainty_forward_passes = [100]
    
    
    
    for ds in data_set:
        for b in batch_size:
            training_loader,validation_loader,test_loader, imu_list = load_data(ds,gpu_device,b,sliding_window_size,sliding_window_step)
            for lr in learning_rate:
                for wd in weight_decay:
                    for m in momentum:
                        for ks in kernelsize:
                            for opt in optimizer:
                                for criterion in loss_critereon:
                                    network = neuralnetwork.Net(imu_list, sliding_window_size, 12, gpu_device, ks)
                                    opt = get_optimiser(network, opt, lr, wd, m)
                                    
                                    
                                    
                                    neuralnetwork.train(network, training_loader, validation_loader, criterion, opt, epochs=epochs)
                                    #neuralnetwork.test(net, test_loader,"test")
  
                                    
                
                
                
                
    
if __name__ == '__main__':
    
    
    main()   
    