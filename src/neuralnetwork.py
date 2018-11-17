import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    
    def __init__(self, imusizes, numberOfAttributes, gpudevice, uncertaintyForwardPasses=1):
        super(Net, self).__init__()
        #torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        #torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        #torch.nn.Sigmoid
        #torch.nn.Linear(in_features, out_features, bias=True)
        #torch.nn.ReLU(inplace=False)
        padding = 0
        self.gpudevice = gpudevice
        self.cuda(device=self.gpudevice)
        self.uncertaintyForwardPasses = uncertaintyForwardPasses
        self.numberOfIMUs = len(imusizes)
        self.numberOfAttributes = numberOfAttributes

        self.imunets = []
        for sensorcount in imusizes:
            self.imunets.append(IMUnet(sensorcount))
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
        #input ist ein liste von tensoren [n,C_in,t,d] 
        # listen element pro imu
        # n ist batch size
        # C_in = 1
        # t = sensor inputs
        # d = sensoren anzahl
        y = []

        for i in range(self.numberOfIMUs):
            x = input[i]
            y.append(self.imunets[i].forward(x))

        extractedFeatures = torch.cat(y,2)
        extractedFeatures = torch.reshape(extractedFeatures, (extractedFeatures.shape[0],-1) )

        #Compute single forwardpass without dropout
        z = extractedFeatures.copy()
        z = F.relu( self.fc1(self.dropout1(z)))
        z = F.sigmoid( self.fc2(self.dropout2(z)))
        result = z
        

        results = torch.tensor.new_empty(self.uncertaintyForwardPasses,self.numberOfAttributes)

        for i in range(self.uncertaintyForwardPasses): #TODO apply dropout.
            z = extractedFeatures.copy()
            z = F.dropout(F.relu( self.fc1(z)),0.5,training=True)
            z = F.sigmoid( self.fc2(z))
            results[i, :] = z

        #torch.var(input, dim, keepdim=False, unbiased=True, out=None)
        #torch.mean(input, dim, keepdim=False, out=None)
        attribute_mean = torch.mean(results,0)
        attribute_st_deviation = torch.sqrt( torch.var(results,0))

        optimistic_prediction = attribute_mean.add(1,attribute_st_deviation)
        pessimistic_prediction = attribute_mean.sub(1,attribute_st_deviation)

        return [result,pessimistic_prediction,optimistic_prediction]


class IMUnet(nn.Module): #defines a parrallel convolutional block
    def __init__(self,numberOfSensors, measurementLength, gpudevice):
        super(Net, self).__init__()
        #torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        #torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        #torch.nn.Sigmoid
        #torch.nn.Linear(in_features, out_features, bias=True)
        #torch.nn.ReLU(inplace=False)

        self.gpudevice = gpudevice
        self.cuda(device=self.gpudevice)

        padding = 0
        self.numberOfSensors = numberOfSensors
        self.measurementLength = measurementLength
        neurons = numberOfSensors*measurementLength*64 # This is supposed to find out the inputsize of the final fullyconnected layer
        self.dropout1 = nn.Dropout(0,5)
        self.conv1 = nn.Conv2d( in_channels=1 , out_channels=64, kernel_size=(5,1), stride=1, padding=padding, bias=True)
        neurons = neurons - (numberOfSensors* 4) # since kernel of 5 with 0 padding loses 4 inputs per sensor
        self.dropout2 = nn.Dropout(0,5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,1), stride=1, padding=padding, bias=True)
        neurons = neurons - (numberOfSensors* 4)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,1), stride=None, padding=0, return_indices=False, ceil_mode=False)
        neurons = neurons/2 # maxpool with 2 halves measurementlength
        self.dropout3 = nn.Dropout(0,5)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,1), stride=1, padding=padding, bias=True)
        neurons = neurons - (numberOfSensors* 4)
        self.dropout4 = nn.Dropout(0,5)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,1), stride=1, padding=padding, bias=True)
        neurons = neurons - (numberOfSensors* 4)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,1), stride=None, padding=0, return_indices=False, ceil_mode=False) 
        neurons = neurons/2
        self.dropout5 = nn.Dropout(0,5)
        self.fc1 = nn.Linear(neurons, 512, bias=True) 
        
        
    def forward(self, input):
        input = F.relu( self.conv1( self.dropout1(input)))
        input = F.relu( self.conv2( self.dropout2(input)))
        input = self.pool1(input)
        input = F.relu( self.conv3( self.dropout3(input)))
        input = F.relu( self.conv4( self.dropout4(input)))
        input = self.pool2(input)

        input = torch.reshape(input,(-1,))

        input = F.relu( self.fc1( self.dropout5(input)))
        return input



def train(network,dataset,criterion=None,optimizer=None,epochs=2): #TODO adapt this tutorial method for my purpose
    network.train()

    if criterion is None:
       criterion = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='elementwise_mean')
    if optimizer is None:
        optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9) #TODO adapt lr and momentum 

    for epoch in range(epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0): #TODO
            # get the inputs
            inputs, labels = data #TODO

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


def test(network,dataset): #TODO adapt this tutorial method for my purpose
    network.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))