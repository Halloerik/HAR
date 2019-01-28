import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt


import neuralnetwork
from data_management import Sliding_Window_Dataset
from data_management import Attribute_Representation



    
def load_data(dataset,gpudevice,batch_size,sliding_window_size,sliding_window_step):
    
    
    if dataset is "pamap2":
        file = open("../datasets/pamap.dat", 'rb')
        #imulist = [1,13,13,13]
        imulist = [40]
    elif dataset is "gestures":
        file = open("../datasets/opportunity_gestures.dat", 'rb')
        #imulist = [3,3,3,3, 3,3,3,3, 3,3,3,3, 9,9,9,9,9, 16,16]
        imulist = [113]
    elif dataset is "locomotion":
        file = open("../datasets/opportunity_locomotion.dat", 'rb')
        #imulist = [3,3,3,3, 3,3,3,3, 3,3,3,3, 9,9,9,9,9, 16,16]
        imulist = [113]
        
    attr_rep = Attribute_Representation(dataset=dataset)
    
    data = pickle.load(file)
    file.close()
    
    training_set    = Sliding_Window_Dataset(data[0], gpudevice, sliding_window_size, sliding_window_step) 
    validation_set  = Sliding_Window_Dataset(data[1], gpudevice, sliding_window_size, sliding_window_step) 
    test_set        = Sliding_Window_Dataset(data[2], gpudevice, sliding_window_size, sliding_window_step) 
    
    #print(np.unique(data[0][1]).shape)
    
    training_loader     = torch.utils.data.DataLoader(training_set  , batch_size=batch_size, shuffle=True , num_workers=0, pin_memory = True)
    validation_loader   = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True , num_workers=0, pin_memory = True)
    test_loader         = torch.utils.data.DataLoader(test_set      , batch_size=batch_size, shuffle=False, num_workers=0, pin_memory = True)
    
    return training_loader, validation_loader, test_loader, imulist, attr_rep

def get_optimiser(network, optimizer, learning_rate, weight_decay,momentum):
    #print("get_optimizer")
    if optimizer is "SGD":
        #print("optimizer should be sgd")
        return torch.optim.SGD(network.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

def plot_run_stats(name,data, epochs):
    fig, ax_lst = plt.subplots(3, 1)
    fig.suptitle('Performance during Trainingphase')
    
    x = np.arange(epochs)
    
    ax_lst[0].set_title("Loss")
    ax_lst[0].set_xlabel("Epochs")
    ax_lst[0].set_ylabel("")
    y1 = data[0]
    y2 = data[1]
    ax_lst[0].plot(x,y1, label="Training")
    ax_lst[0].plot(x,y2, label="Validation")
    #ax_lst[0].legend()
    
    ax_lst[1].set_title("Accuracy")
    ax_lst[1].set_xlabel("Epochs")
    ax_lst[1].set_ylabel("")
    y1 = data[2]
    y2 = data[3]
    ax_lst[1].plot(x,y1, label="Training")
    ax_lst[1].plot(x,y2, label="Validation")
    #ax_lst[1].legend()
    
    ax_lst[2].set_title("F1")
    ax_lst[2].set_xlabel("Epochs")
    ax_lst[2].set_ylabel("")
    y1 = data[4]
    y2 = data[5]
    ax_lst[2].plot(x,y1, label="Training")
    ax_lst[2].plot(x,y2, label="Validation")
    #ax_lst[2].legend()
    
    #plt.show()
    
    f= open("../../performance/{}.png".format(name), 'w+b')
    plt.savefig(f, facecolor='w', edgecolor='w')
    f.close()
    
def save_run_stats(name, data,comment):
    f= open("../../performance/{}.txt".format(name), 'w+t')
    
    table = np.stack(data, 1)
    
    np.savetxt(f, table, delimiter=' ', newline='\n', 
               header='loss_trn loss_val accurary_trn accuracy_val f1_trn f1_val',
               comments='# {}\n'.format(comment) )
    
    f.close()

def main():
    config = {
    'data_set' : ["pamap2","gestures","locomotion"],
    'batch_size' : [128],
    'sliding_window_size' : 100,
    'sliding_window_step' : 22,
    
    #Training Parameters
    'epochs' : 200,
    'learning_rate' : [0.01,0.001,0.0001],
    'weight_decay' : [0.0001],
    'momentum' : [0.9],
    'loss_critereon' : [torch.nn.BCELoss()],
    'optimizer' : ["SGD"],
    
    #Network parameters
    'kernelsize' : [(5,1)],
    
    'gpu_device' : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    
    #Attribute Representation
    'n_attributes' : {"pamap2" : 12, "locomotion" : 10, "gestures" : 18},
    'distance_metric' : ["braycurtis","cosine"],
    
    #Uncertainty
    'uncertainty_forward_passes' : 100
    
    }
    
    run_number = 0
    
    #for i in range(9):
    #    print(current_config_str(config, i))
    
    for ds in config['data_set']:
        for b in config['batch_size']:
            training_loader,validation_loader,test_loader, imu_list, attr_rep = load_data(
                ds,config['gpu_device'],b,config['sliding_window_size'],config['sliding_window_step'])
            for lr in config['learning_rate']:
                for wd in config['weight_decay']:
                    for m in config['momentum']:
                        for ks in config['kernelsize']:
                            for opt in config['optimizer']:
                                for criterion in config['loss_critereon']:
                                    for dist_metric in config['distance_metric']:
                                    
                                        network = neuralnetwork.Net(imu_list,config['sliding_window_size'], config['n_attributes'][ds],
                                                                 config['gpu_device'], ks, config['uncertainty_forward_passes']).cuda(config['gpu_device'])
                                                                 
                                        optimizer = get_optimiser(network, opt, lr, wd, m)
                                        
                                        data = neuralnetwork.train(network, training_loader, validation_loader, criterion, optimizer, config['epochs'], config['gpu_device'],attr_rep,dist_metric)
                                            
                                        plot_run_stats("train run {}".format(run_number), data, config['epochs'])
                                        save_run_stats("train run {}".format(run_number),data,current_config_str(config, run_number))
                                    
                                        run_number += 1
                
def current_config_str(config, current_run):
    run_number = 0
    for ds in config['data_set']:
        for b in config['batch_size']:
            for lr in config['learning_rate']:
                for wd in config['weight_decay']:
                    for m in config['momentum']:
                        for ks in config['kernelsize']:
                            for opt in config['optimizer']:
                                for criterion in config['loss_critereon']:
                                    for dist_metric in config['distance_metric']:
                                            
                                        if run_number is current_run:
                                            settings = [('data_set',ds),('batch_size',b),('sliding_window_size',config['sliding_window_size']),
                                                        ('sliding_window_step',config['sliding_window_step']), ('epochs',config['epochs']),
                                                        ('learning_rate',lr),('weight_decay',wd),('momentum',m),('loss_critereon',criterion),
                                                        ('optimizer',opt),('kernelsize',ks),('distance_metric',dist_metric)]
                                            current_config = ''
                                            for setting in settings:
                                                current_config += '{}: {}, '.format(setting[0],setting[1])
                                            return(current_config[0:-2])
                                        else:
                                            run_number += 1
    
    
if __name__ == '__main__':
    main()    