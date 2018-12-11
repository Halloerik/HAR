import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import f1_score

class Net(nn.Module):
    
    def __init__(self, imusizes, segmentsize, numberOfAttributes, gpudevice, kernelsize):
        super(Net, self).__init__()
        self.gpudevice = gpudevice
        self.cuda(device=self.gpudevice)
        self.imusizes = imusizes
        self.numberOfIMUs = len(imusizes)
        self.numberOfAttributes = numberOfAttributes

        self.imu_list = []
        for i, sensorcount in enumerate(imusizes):
            imunet = IMUnet(sensorcount, segmentsize,kernelsize, self.gpudevice)
            self.add_module("IMUnet{}".format(i),imunet)
            self.imu_list.append(imunet)
            
        self.dropout1 = nn.Dropout(0,5)
        self.fc1 = nn.Linear(512*self.numberOfIMUs, 512, bias=True)
        self.dropout2 = nn.Dropout(0,5)
        self.fc2 = nn.Linear(512, self.numberOfAttributes, bias=True)
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
        single_forward_pass = torch.tensor(extractedFeatures)
        single_forward_pass = F.relu(self.fc1(self.dropout1(single_forward_pass)))
        single_forward_pass = self.fc2(self.dropout2(single_forward_pass))
        single_forward_pass = self.sigmoid(single_forward_pass)
        
        return single_forward_pass.to(device = "cpu")

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

class smallnet(nn.Module):
    def __init__(self, segmentsize, numberOfAttributes, gpudevice, kernelsize):
        super(smallnet, self).__init__()
        self.gpudevice = gpudevice
        self.cuda(device=self.gpudevice)
        self.numberOfAttributes = numberOfAttributes
        
        imunet = IMUnet(40, segmentsize,kernelsize, self.gpudevice)
        self.add_module("IMUnet",imunet)
            
        self.dropout1 = nn.Dropout(0,5)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.dropout2 = nn.Dropout(0,5)
        self.fc2 = nn.Linear(512, self.numberOfAttributes, bias=True)
        
    def forward(self,input):
        imunets = self.named_children()
        
        z = next(imunets)[1](input)
        z = F.relu( self.fc1(self.dropout1(z)))
        z = self.fc2(self.dropout2(z))
        z = torch.softmax( z, dim=1)        
        return z

def train(network, training_loader, validation_loader, criterion, optimizer, epochs, gpudevice): 
    
    
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
        for i, data in enumerate(training_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda(device = gpudevice)
            #labels = labels.cuda(device = gpudevice)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = network(inputs)
            iter_loss = criterion(outputs, labels)
            iter_loss.backward()
            optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted.float() == labels.float()).sum().item()
        
        loss[epoch] = iter_loss.detach()
        accuracy[epoch] = correct / total
        f1[epoch] = f1_score(labels, predicted, average='weighted')
        
        print("loss of training: {}".format(loss[epoch]))
        print("accuracy of training: {}".format(accuracy[epoch]))
        print("f1 score of training: {}".format(f1[epoch]))
        
        
        loss_val[epoch], accuracy_val[epoch], f1_val[epoch] = test(network,validation_loader, criterion,gpudevice)
        
        print("loss of validation: {}".format(loss_val[epoch]))
        print("accuracy of validation: {}".format(accuracy_val[epoch]))
        print("f1 score of validation: {}".format(f1_val[epoch]))
        
        network.train(True)

    print('Finished Training')
    
    return(loss.detach().numpy(),
           loss_val.detach().numpy(),
           accuracy.detach().numpy(),
           accuracy_val.detach().numpy(),
           f1.detach().numpy(),
           f1_val.detach().numpy()) 


def test(network,data_loader, criterion,gpudevice): #TODO: adapt this tutorial method for my purpose
    
    network.eval()

    correct = 0
    total = 0
    
    #predictions = torch.zeros(0,dtype=torch.float)
    labels = torch.zeros(0,dtype=torch.long)
    outputs = torch.zeros(0,dtype=torch.float)
    
    with torch.no_grad():
        for data in data_loader:
            inputs, label = data
            inputs.cuda(device = gpudevice)
            label.to(device = gpudevice)
            
            output = network(inputs)
            outputs = torch.cat((outputs,output),0)
            
            labels = torch.cat((labels,label),0)

        _, predicted = torch.max(outputs.data, 1)
        
        
        #print("labels shape {}".format(labels.shape))    
        #print("outputs shape {}".format(outputs.shape))
        #print("predicted shape {}".format(predicted.shape))
        
        total = labels.shape[0]
        correct = (predicted.float() == labels.float()).sum().item()    
        
        #print("total: {}".format(total))
        #print("correct: {}".format(correct))
        
        loss = criterion(outputs, labels)
        accuracy = correct / total
        f1 = f1_score(labels, predicted, average='weighted')
    
        return loss,accuracy,f1
    
    
    
    
    
    