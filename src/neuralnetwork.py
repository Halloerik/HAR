import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from data_management import Attribute_Representation


class Net(nn.Module):
    
    def __init__(self, imusizes, segmentsize, numberOfAttributes, gpudevice, uncertaintyForwardPasses=1,kernelsize=5 ):
        super(Net, self).__init__()
        #torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        #torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        #torch.nn.Sigmoid
        #torch.nn.Linear(in_features, out_features, bias=True)
        #torch.nn.ReLU(inplace=False)
        self.gpudevice = gpudevice
        self.cuda(device=self.gpudevice)
        self.uncertaintyForwardPasses = uncertaintyForwardPasses
        self.imusizes = imusizes
        self.numberOfIMUs = len(imusizes)
        self.numberOfAttributes = numberOfAttributes

        self.imunets = []
        for sensorcount in imusizes:
            self.imunets.append(IMUnet(sensorcount, segmentsize,kernelsize, self.gpudevice))
        self.dropout1 = nn.Dropout(0,5)
        self.fc1 = nn.Linear(512*self.numberOfIMUs, 512, bias=True)
        self.dropout2 = nn.Dropout(0,5)
        self.fc2 = nn.Linear(512, self.numberOfAttributes, bias=True)
    
    def train(self,mode=True):
        super(Net, self).train(mode)
        for imu in self.imunets:
            imu.train()
        
    def eval(self):
        super(Net, self).eval() 
        for imu in self.imunets:
            imu.eval()
        
    def forward(self,input):
        #input ist ein tensor mit shape[n,C_in,t,d] 
        # n ist batch size
        # C_in = 1
        # t = sensor inputs
        # d = sensoren anzahl
        
        y = []
        firstsensor = 0
        for i in range(self.numberOfIMUs):
            lastsensor = firstsensor + self.imusizes[i]
            x = input[:,:,:,firstsensor:lastsensor]
            y.append(self.imunets[i](x))
        
        extractedFeatures = y[0]
        for tensor in range(1,len(y)):
            extractedFeatures = torch.cat((extractedFeatures,y[tensor]),dim=1)
        
        
        extractedFeatures = torch.reshape(extractedFeatures, (extractedFeatures.shape[0],-1) )

        #Compute single forwardpass without dropout
        z = torch.tensor(extractedFeatures)
        z = F.relu( self.fc1(self.dropout1(z)))
        z = torch.sigmoid( self.fc2(self.dropout2(z)))
        result = z
        
        if(self.training is True):
            return result
        else:
            ##TODO: do this...
            #results = torch.zeros((self.uncertaintyForwardPasses,self.numberOfAttributes))

            #for i in range(self.uncertaintyForwardPasses):
            #    z = torch.tensor(extractedFeatures)
            #    z = F.dropout(F.relu( self.fc1(z)),0.5,training=True)
            #    z = torch.sigmoid( self.fc2(z))
            #    results[i, :] = z

            ##torch.var(input, dim, keepdim=False, unbiased=True, out=None)
            ##torch.mean(input, dim, keepdim=False, out=None)
            #attribute_mean = torch.mean(results,0)
            #attribute_st_deviation = torch.sqrt( torch.var(results,0))

            #optimistic_prediction = attribute_mean.add(1,attribute_st_deviation)
            #pessimistic_prediction = attribute_mean.sub(1,attribute_st_deviation)

            #return [result,pessimistic_prediction,attribute_mean,optimistic_prediction]
            return result

class IMUnet(nn.Module): #defines a parrallel convolutional block
    def __init__(self,numberOfSensors, segmentsize,kernelsize, gpudevice):
        super(IMUnet, self).__init__()
        self.gpudevice = gpudevice
        self.cuda(device=self.gpudevice)

        padding = 0
        self.numberOfSensors = numberOfSensors
        self.measurementLength = segmentsize
        
        self.dropout1 = nn.Dropout(0,5)
        self.conv1 = nn.Conv2d( in_channels=1 , out_channels=64, kernel_size=(kernelsize,1), stride=1, padding=padding, bias=True)
        self.dropout2 = nn.Dropout(0,5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(kernelsize,1), stride=1, padding=padding, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,1), stride=1, padding=0, return_indices=False, ceil_mode=False)
        self.dropout3 = nn.Dropout(0,5)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(kernelsize,1), stride=1, padding=padding, bias=True)
        self.dropout4 = nn.Dropout(0,5)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(kernelsize,1), stride=1, padding=padding, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,1), stride=1, padding=0, return_indices=False, ceil_mode=False) 
        self.dropout5 = nn.Dropout(0,5)
        
        #This is supposed to find out the inputsize of the final fullyconnected layer
        #segmentsize = segmentsize-4 #conv
        #segmentsize = segmentsize-4 #conv
        #segmentsize = segmentsize-1 #maxpool
        #segmentsize = segmentsize-4 #conv
        #segmentsize = segmentsize-4 #conv
        #segmentsize = segmentsize-1 #maxpool
        #number of sensors is constant
        #and C_out is 64 while C_in was 1
        neurons = (segmentsize-18)*numberOfSensors*64  
        self.fc1 = nn.Linear(int(neurons), 512, bias=True)
        
        
        
        
        
    def forward(self, input):
        input = F.relu( self.conv1( self.dropout1(input)))
        input = F.relu( self.conv2( self.dropout2(input)))
        input = self.pool1(input)
        input = F.relu( self.conv3( self.dropout3(input)))
        input = F.relu( self.conv4( self.dropout4(input)))
        input = self.pool2(input)
        batchsize = input.shape[0]
        input = torch.reshape(input,(batchsize,-1))
        input = F.relu( self.fc1( self.dropout5(input)))
        return input


def train(network, training_loader, validation_loader, attr_representation, distance_metric="cosine", criterion=None, optimizer=None, epochs=2): 
    
    network.train(True)

    if criterion is None:
        criterion = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='elementwise_mean')
        #criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9) #TODO: adapt lr and momentum 

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(training_loader, 0):
            # get the inputs
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, attr_representation.attributevector_of_class(labels))
            loss.backward()
            optimizer.step()
            #print("progress {}".format(i))
            # print statistics
            #running_loss += loss.item()
            if i % 1000 == 999:    # print every 1000 mini-batches
                #print('[{}, {}] loss: {}'.format(epoch + 1, i + 1, running_loss / 100))
                #running_loss = 0.0
                test(network,training_loader, "training" , attr_representation,distance_metric,)
                test(network,validation_loader, "validation" , attr_representation,distance_metric)
                network.train(True)

    print('Finished Training')


def test(network,data_loader, data_name, attr_representation,distance_metric="cosine"): #TODO: adapt this tutorial method for my purpose
    network.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            #print(inputs.shape)
            outputs = network(inputs)
            #_, predicted = torch.max(outputs.data, 1)
            predicted = attr_representation.closest_class(outputs, distance_metric)
            total += labels.size(0)
            correct += (predicted.float() == labels.float()).sum().item()

    print('Accuracy of the network on the {}_set: {} %'.format( data_name,100 * correct / total ))
    
    #class_correct = list(0. for i in range(10))
    #class_total = list(0. for i in range(10))
    #with torch.no_grad():
    #    for data in testloader:
    #        inputs, labels = data
    #        outputs = network(inputs)
    #        #_, predicted = torch.max(outputs, 1)
    #        predicted = attr_representation.closest_class(outputs, distance_metric)
    #        c = (predicted == labels).squeeze()
    #        for i in range(4):
    #            label = labels[i]
    #            class_correct[label] += c[i].item()
    #            class_total[label] += 1


    #for i in range(10):
    #    print('Accuracy of %5s : %2d %%' % (
    #        classes[i], 100 * class_correct[i] / class_total[i]))
        

    
    
    
    
    
    
    
    
    
    
    
    