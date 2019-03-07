import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from sklearn.metrics import f1_score
import scipy

class Net(nn.Module):
    
    def __init__(self, imusizes, segmentsize, n_attributes, gpudevice, kernelsize, uncertaintyforwardpasses):
        super(Net, self).__init__()
        self.gpudevice = gpudevice
        self.cuda(device=self.gpudevice)
        self.imusizes = imusizes
        self.numberOfIMUs = len(imusizes)
        self.n_attributes = n_attributes
        self.uncertaintyforwardpasses = uncertaintyforwardpasses
        
        self.imu_list = []
        for i, sensorcount in enumerate(imusizes):
            imunet = IMUnet(sensorcount, segmentsize,kernelsize, self.gpudevice)
            self.add_module("IMUnet{}".format(i),imunet)
            self.imu_list.append(imunet)
            
        self.dropout1 = nn.Dropout(0,5)
        self.fc1 = nn.Linear(512*self.numberOfIMUs, 512, bias=True)
        self.dropout2 = nn.Dropout(0,5)
        self.fc2 = nn.Linear(512, self.n_attributes, bias=True)
        #self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self,input):
        #input ist ein tensor mit shape[n,C_in,t,d] 
        # n ist batch size
        # C_in = 1
        # t = sensor inputs
        # d = sensoren anzahl
        
        extractedFeatures = torch.zeros(0,dtype=torch.float,device =self.gpudevice)
        firstsensor = 0
        for i, imu in enumerate(self.imu_list):
            lastsensor = firstsensor + self.imusizes[i]
            x = input[:,:,:,firstsensor:lastsensor].cuda(self.gpudevice)
            x = imu(x)
            extractedFeatures = torch.cat((extractedFeatures,x),dim=1)
            firstsensor = lastsensor
        
        #Compute single forwardpass without dropout
        single_forward_pass = torch.tensor(extractedFeatures).cuda(device = self.gpudevice)
        single_forward_pass = F.relu(self.fc1(self.dropout1(single_forward_pass)))
        single_forward_pass = self.fc2(self.dropout2(single_forward_pass))
        single_forward_pass = self.sigmoid(single_forward_pass)
        
        
        if(self.training):
            return single_forward_pass.to(device = "cpu")
        else:
            
            forward_passes = torch.zeros((self.uncertaintyforwardpasses, single_forward_pass.shape[0], single_forward_pass.shape[1]))
            
            for i in range(self.uncertaintyforwardpasses):
                forward_pass = torch.tensor(extractedFeatures).cuda(device = self.gpudevice)
                forward_pass = F.relu(self.fc1(F.dropout(forward_pass,0.5,training=True)))
                forward_pass = self.fc2(F.dropout(forward_pass,0.5,training=True))
                forward_pass = self.sigmoid(forward_pass)
                
                #print("forward pass {}: {}".format(i,forward_pass[0]))
                
                forward_passes[i, :, :] = forward_pass
            
            
            #scipy.stats.norm.ppf(0.97725)
            #torch.var(input, dim, keepdim=False, unbiased=True, out=None)
            #torch.mean(input, dim, keepdim=False, out=None)
            
            #TODO: Multiply mean and var by modelprecision to get predictive equvalents
            attribute_mean = torch.mean(forward_passes,0)
            attribute_variance = torch.var(forward_passes,0)
            attribute_st_deviation = torch.sqrt( attribute_variance)
                        
            
            
            
            
            
            
            factor = torch.div(attribute_variance, self.uncertaintyforwardpasses )
            factor = torch.sqrt(factor)
            factor = torch.mul(factor, scipy.stats.norm.ppf(0.97725))
            
            
            
            optimistic_prediction  = attribute_mean + factor
            optimistic_prediction[optimistic_prediction>1] = 1
            
            pessimistic_prediction = attribute_mean - factor
            pessimistic_prediction[pessimistic_prediction<0] = 0
            
            
            results = torch.zeros(4,single_forward_pass.shape[0], single_forward_pass.shape[1])
            results[0,:,:] = single_forward_pass
            results[1,:,:] = attribute_mean
            results[2,:,:] = pessimistic_prediction
            results[3,:,:] = optimistic_prediction
            return results
                
                
class IMUnet(nn.Module): #defines a parrallel convolutional block
    def __init__(self,numberOfSensors, segmentsize,kernelsize, gpudevice):
        super(IMUnet, self).__init__()
        self.gpudevice = gpudevice
        self.cuda(device=self.gpudevice)
        
        padding = 0
        self.numberOfSensors = numberOfSensors
        self.measurementLength = segmentsize
        
        self.dropout1 = nn.Dropout(0,5)
        self.conv1 = nn.Conv2d( in_channels=1 , out_channels=64, kernel_size=kernelsize, stride=1, padding=padding, bias=True)
        self.dropout2 = nn.Dropout(0,5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernelsize, stride=1, padding=padding, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,1), stride=1, padding=0, return_indices=False, ceil_mode=False)
        self.dropout3 = nn.Dropout(0,5)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernelsize, stride=1, padding=padding, bias=True)
        self.dropout4 = nn.Dropout(0,5)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernelsize, stride=1, padding=padding, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,1), stride=1, padding=0, return_indices=False, ceil_mode=False) 
        self.dropout5 = nn.Dropout(0,5)
        
        #print("number of sensors {}".format(numberOfSensors))
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
        #print("shape of input {}".format(input.shape))
        input = F.relu( self.fc1( self.dropout5(input)))
        return input

class smallnet(nn.Module):
    def __init__(self, segmentsize, n_attributes, gpudevice, kernelsize):
        super(smallnet, self).__init__()
        self.gpudevice = gpudevice
        self.cuda(device=self.gpudevice)
        self.n_attributes = n_attributes
        
        imunet = IMUnet(40, segmentsize,kernelsize, self.gpudevice)
        self.add_module("IMUnet",imunet)
            
        self.dropout1 = nn.Dropout(0,5)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.dropout2 = nn.Dropout(0,5)
        self.fc2 = nn.Linear(512, self.n_attributes, bias=True)
        
    def forward(self,input):
        imunets = self.named_children()
        
        z = next(imunets)[1](input)
        z = F.relu( self.fc1(self.dropout1(z)))
        z = self.fc2(self.dropout2(z))
        z = torch.softmax( z, dim=1)        
        return z

def train(network, training_loader, validation_loader, criterion, optimizer, epochs, gpudevice, attr_rep, dist_metric): 
    
    
    loss = torch.zeros(epochs)
    accuracy = torch.zeros(epochs)
    f1 = torch.zeros(epochs)
    
    loss_val = torch.zeros(epochs)
    accuracy_val = torch.zeros(epochs)
    f1_val = torch.zeros(epochs)
    
    network.train(True)
    
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        print("Epoch:{}/{}".format(epoch+1,epochs))
        outputs = None
        labels= None
        iter_loss = None
        for _, data in enumerate(training_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda(device = gpudevice)
            #labels = labels.cuda(device = gpudevice)
            
            label_vectors = attr_rep.attributevector_of_class(labels)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = network(inputs)
            #print("outputs shape {}".format(outputs.shape))
            #print("labelvectors shape {}".format(label_vectors.shape))
            
            
            iter_loss = criterion(outputs, label_vectors)
            iter_loss.backward()
            optimizer.step()
        
        #_, predicted = torch.max(outputs.data, 1)
        predicted = attr_rep.closest_class(outputs.detach(), dist_metric)
        
        #print("label shape: {} predicted shape: {}".format(labels.shape,predicted.shape))
        
        
        total = labels.size(0)
        correct = (predicted.float() == labels.float()).sum().item()
        
        loss[epoch] = iter_loss.detach()
        accuracy[epoch] = correct / total
        f1[epoch] = f1_score(labels, predicted, average='weighted')
        
        #print("loss of training: {}".format(loss[epoch]))
        #print("accuracy of training: {}".format(accuracy[epoch]))
        #print("f1 score of training: {}".format(f1[epoch]))
        
        
        loss_val[epoch], accuracy_val[epoch], f1_val[epoch] = test(network,validation_loader, criterion,gpudevice, attr_rep, dist_metric, training=True)
        
        #print("loss of validation: {}".format(loss_val[epoch]))
        #print("accuracy of validation: {}".format(accuracy_val[epoch]))
        #print("f1 score of validation: {}".format(f1_val[epoch]))
        
        network.train(True)

    print('Finished Training')
    
    return(loss.detach().numpy(),
           loss_val.detach().numpy(),
           accuracy.detach().numpy(),
           accuracy_val.detach().numpy(),
           f1.detach().numpy(),
           f1_val.detach().numpy()) 


def test(network,data_loader, criterion,gpudevice, attr_rep, dist_metric, training):
    
    if training:
        network.train()
    else:
        network.eval()

    correct = 0
    total = 0
    
    #predictions = torch.zeros(0,dtype=torch.float)
    labels = torch.zeros(0,dtype=torch.long)
    outputs = torch.zeros(0,dtype=torch.float)
    
    with torch.no_grad():
        if training:
            for data in data_loader:
                inputs, label = data
                inputs.cuda(device = gpudevice)
                label.to(device = gpudevice)
            
                output = network(inputs)
                
                #print("output {}".format(output.shape))
                
                outputs = torch.cat((outputs,output),0)
                labels = torch.cat((labels,label),0)

            #_, predicted = torch.max(outputs.data, 1)
            predicted = attr_rep.closest_class(outputs, dist_metric)
            
            #print("output {}".format(outputs))    
            #print("predicted {}".format(predicted))
            #print("label {}".format(labels))
            
            total = labels.shape[0]
            correct = (predicted.float() == labels.float()).sum().item()    
        
        
            label_vectors = attr_rep.attributevector_of_class(labels)
        
            loss = criterion(outputs, label_vectors)
            accuracy = correct / total
            f1 = f1_score(labels, predicted, average='weighted')
    
            return loss,accuracy,f1
        else:
            
            for data in data_loader:
                inputs, label = data
                inputs.cuda(device = gpudevice)
                label.to(device = gpudevice)
            
                output = network(inputs)
                outputs = torch.cat((outputs,output),1)
                labels = torch.cat((labels,label),0)
                
                
            
            print("outputs {}".format(outputs.shape))
            print("labels {}".format(labels.shape))
            loss = torch.zeros(4,dtype=torch.float)
            accuracy = torch.zeros(4,dtype=torch.float)
            f1 = torch.zeros(4,dtype=torch.float)
            f1_per_class = None
            
            for i in range(4):
                
                print("prediction {}/4".format(i+1))
                
                predicted = attr_rep.closest_class(outputs[i,:,:], dist_metric)
                
                #print("predicted: {}".format(predicted) )
                
                total = labels.shape[0]
                correct = (predicted.float() == labels.float()).sum().item()  
                label_vectors = attr_rep.attributevector_of_class(labels)
        
                loss[i] = criterion(outputs[i,:,:], label_vectors)
                
                #print("loss {}: {}".format(i+1,loss[i]))
                
                accuracy[i] = correct / total
                f1[i] = f1_score(labels, predicted, average='weighted')
                if f1_per_class is None: 
                    f1_per_class = f1_score(labels, predicted, average=None)
                    f1_per_class = numpy.expand_dims(f1_per_class, axis=1)
                    #print(f1_per_class)
                else:
                    f = f1_score(labels, predicted, average=None)
                    f = numpy.expand_dims(f,axis=1)
                    f1_per_class = numpy.concatenate((f1_per_class,f),1)
                    
            f1_per_class = torch.from_numpy(f1_per_class)
            return loss,accuracy,f1,f1_per_class
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    