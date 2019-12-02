# Source code for the paper: "A Deep Learning approach to Photoacoustic Wavefront Localization in Deep-Tissue Medium"



import time
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat, savemat
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import dsntnn
from scipy import ndimage
import h5py
import os

#-------------------------------------------------
seed1 = np.random.randint(low = 0,high = 100)
seed2 = np.random.randint(low = 0,high = 100)
seeds = (seed1,seed2)

torch.manual_seed(seed1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(seed2)
#-----------------------------------------------------------------------------
#          -Load Data-
#-----------------------------------------------------------------------------
'''
data_path = '/home/kfj5051/Desktop/Matt/Deepit_data/';
X_Test = loadmat(''.join([data_path,'X_Test_deepit.mat']));
X_test = X_Test['X_Test']
X_test = torch.from_numpy(X_test)

X_Test=torch.Tensor(1201,1,256,2048);
X_Test[:,0,:,:]=X_test[:,:,0:2048];

X_Train = loadmat(''.join([data_path,'X_Train_deepit.mat']));
X_train = X_Train['X_Train']
X_train = torch.from_numpy(X_train)

X_Train=torch.Tensor(16240,1,256,2048);
X_Train[:,0,:,:]=X_train[:,:,0:2048];


Y_Te = loadmat(''.join([data_path,'Y_Test_deepit.mat']));
Y_Te = Y_Te['Y_Test']
Y_Te = torch.from_numpy(Y_Te)

Y_Test = torch.Tensor(1201,2)
lat = Y_Te[:,0]/50
ax = (Y_Te[:,1]-50)/50
Y_Test[:,0] = ax
Y_Test[:,1] = lat
Y_Test = Y_Test.unsqueeze(1).float()


Y_Tr = loadmat(''.join([data_path,'Y_Train_deepit.mat']));
Y_Tr = Y_Tr['Y_Train']
Y_Tr = torch.from_numpy(Y_Tr)

Y_Train = torch.Tensor(16240,2)
lat = Y_Tr[:,0]/50
ax = (Y_Tr[:,1]-50)/50
Y_Train[:,0] = ax
Y_Train[:,1] = lat
Y_Train = Y_Train.unsqueeze(1).float()


bs = 20
train = torch.utils.data.TensorDataset(X_Train, Y_Train)
trainloader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=False)

test = torch.utils.data.TensorDataset(X_Test, Y_Test)
testloader = torch.utils.data.DataLoader(test, batch_size=bs, shuffle=False)

print('Displaying Channel Data from random training set image')
n = np.random.randint(0,16240)
channel_Data = X_train[n,:,:].numpy()
plt.imshow(channel_Data)
plt.show()
print Y_Train[n]
'''

data_path = '/home/kfj5051/Desktop/Matt/Deepit_data/';


Load_Data = True
if Load_Data == True:
    # Load sensor testing data
    with h5py.File(''.join([data_path,'X_Test_20_Noisy_-9dB.mat']), 'r') as data:
        X_Test = data['X_Test'][:]
    X_Test = np.transpose(X_Test)
    X_Test = torch.from_numpy(X_Test)
    
    # Load sensor training data
    with h5py.File(''.join([data_path,'X_Train_20_Noisy_-9dB.mat']), 'r') as data:
        X_Train = data['X_Train'][:]
    X_Train = np.transpose(X_Train)
    X_Train = torch.from_numpy(X_Train)
    X_Train[7006,0,:,:] = X_Test[4060,0,:]   #7006 was blank
    X_Test = X_Test[0:4060,:,:,:]  #4061 was blank and 4060 is transferred to the training set
#    for e in range(13):
#        maindir = '/home/kfj5051/Machine_Learning/Experimental/MachineLearningData/extracted_data/X_50MHz_'+'{:02d}'.format(e)
#        X_Test[e,0,:,:] = torch.from_numpy(loadmat(maindir)['data'])[:,600:1624]
        
    #X_Test[12,0,:,:] = torch.from_numpy(X[:,600:1624])
    #X_Test[13,0,:,:] = torch.from_numpy(X[:,700:1724])
    #X_Test[14,0,:,:] = torch.from_numpy(X[:,800:1824])
    #X_Test[15,0,:,:] = torch.from_numpy(X[:,900:1924])
    #X_Test[16,0,:,:] = torch.from_numpy(X[:,1000:2024])
    # Load label data for test set
    with h5py.File(''.join([data_path,'Y_Test_20.mat']), 'r') as data:
        Y_te = data['Y_Test'][:]
    Y_te = np.transpose(Y_te)
    Y_Te = torch.from_numpy(Y_te)
    Y_Test = torch.Tensor(4062,2)
    max_ax = 50
    min_ax = 10
    max_normalized_depth = 0.9
    ax = (Y_Te[:,1]-(max_ax-min_ax)/2-min_ax)/((max_ax-min_ax)/2/max_normalized_depth)
    lat = Y_Te[:,0]/((max_ax-min_ax)/2/max_normalized_depth)
    Y_Test[:,0] = ax
    Y_Test[:,1] = lat
    #Y_Test = Y_Test.unsqueeze(1).float()
    
    # Load label data for training set
    with h5py.File(''.join([data_path,'Y_Train_20.mat']), 'r') as data:
        Y_tr = data['Y_Train'][:]
    Y_tr = np.transpose(Y_tr)
    Y_Tr = torch.from_numpy(Y_tr)
    Y_Train = torch.Tensor(16240,2)
    ax = (Y_Tr[:,1]-(max_ax-min_ax)/2-min_ax)/((max_ax-min_ax)/2/max_normalized_depth)
    lat = Y_Tr[:,0]/((max_ax-min_ax)/2/max_normalized_depth)
    
    Y_Train[:,0] = ax
    Y_Train[:,1] = lat
    Y_Train[7006,:] = Y_Test[4060,:]  #again, 7006 was blank
    Y_Test = Y_Test[0:4060,:]   #same as above
    Y_Train = Y_Train.unsqueeze(1).float()
    Y_Test = Y_Test.unsqueeze(1).float()
    
    


bs = 8
train = torch.utils.data.TensorDataset(X_Train, Y_Train)
trainloader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=False)

test = torch.utils.data.TensorDataset(X_Test, Y_Test)
testloader = torch.utils.data.DataLoader(test, batch_size=bs, shuffle=False)


#-----------------------------------------------------------------------------
#          -ResNet Implementation-
#-----------------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, expanding=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.expanding = expanding
        self.planes = planes
        self.resconv = conv3x3(inplanes, planes)
        

    def forward(self, x):
        if self.expanding:
            residual = x[:,0:self.planes,:,:]
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3,stride=1,padding=1)

    def forward(self, present, former):
        present = self.up(present)
        present = self.conv(present)
        x = torch.cat([present, former], dim=1)
        return x


class ResNet(nn.Module):

    def __init__(self, block, num_classes=9):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(7,7), stride=(1,2), dilation=(1,4),padding=(3,12), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool1d = nn.MaxPool2d(kernel_size=(1,2))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, 16, blocks=1)
        self.layer2 = self._make_layer(block, 16, 32, blocks=1, stride=2)
        self.layer3 = self._make_layer(block, 32, 64, blocks=1, stride=2)
        self.layer4 = self._make_layer(block, 64, 128, blocks=1, stride=2)
        self.layer5 = self._make_layer(block, 128, 256, blocks=1, stride=2)
        self.classify = nn.Conv2d(256, 256, kernel_size=(5,5), padding=(2,2))
        self.bn2 = nn.BatchNorm2d(256)
        self.layer6 = self._make_layer(block, 256, 256, blocks=1, stride=2) #subsample w/o adding features
        self.classify2 = nn.Conv2d(256, 256, kernel_size=(5,5),padding=(2,2)) #8x8
        self.bn3 = nn.BatchNorm2d(256)
        self.layer7 = self._make_layer(block, 256, 256, blocks=1, stride=1)
        self.layer8 = self._make_layer(block, 256, 256, blocks=1, stride=1)
        self.up1 = up(256, 256)
        self.layer9= self._make_layer(block, 512, 256, blocks=1, stride=1, expanding=True)
        self.up2 = up(256, 128)
        self.layer10 = self._make_layer(block, 256, 128, blocks=1, stride=1, expanding=True)
        self.up3 = up(128, 64)
        self.layer11 = self._make_layer(block, 128, 64, blocks=1, stride=1, expanding=True)
        self.up4 = up(64, 32)
        self.layer12 = self._make_layer(block, 64, 32, blocks=1, stride=1, expanding=True)
        self.up5 = up(32, 16)
        self.layer13 = self._make_layer(block, 32, 16, blocks=1, stride=1, expanding=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, expanding=False):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )            
        layers = []
        layers.append(block(inplanes, planes, stride, downsample, expanding))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
            
        return nn.Sequential(*layers)
    
    def display(self, x, vis):
        if vis == True:
            plt.imshow(x[0,0,:,:].detach().cpu().numpy()) #first in the batch
            plt.show()
            
    def forward(self, x, remember=False):
        x1 = self.conv1(x)
        xhist = {'x1': x}
        x2 = self.bn1(x1)
        xhist['x2'] = x2
        x3 = self.relu(x2)
        xhist['x3'] = x3
        x4 = self.pool1d(x3)       #256x256x16  <-outputs
        xhist['x4'] = x4
        x5 = self.layer1(x4)       #256x256x16
        xhist['x5'] = x5
        x6 = self.layer2(x5)       #128x128x32
        xhist['x6'] = x6
        x7 = self.layer3(x6)       #64x64x64
        xhist['x7'] = x7
        x8 = self.layer4(x7)       #32x32x128
        xhist['x8'] = x8
        x9 = self.layer5(x8)       #16x16x256
        xhist['x9'] = x9
        x10 = self.classify(x9)    #16x16x256
        xhist['x10'] = x10
        x11 = self.bn2(x10)        #BN
        xhist['x11'] = x11
        x12 = self.relu(x11)       #ReLU
        xhist['x12'] = x12
        x13 = self.layer6(x12)     #8x8x256
        xhist['x13'] = x13
        x14 = self.classify2(x13)  #8x8x256
        xhist['x14'] = x14
        x15 = self.bn3(x14)        #BN
        xhist['x15'] = x15
        x16 = self.relu(x15)       #ReLU
        xhist['x16'] = x16
        x17 = self.layer7(x16)     #8x8x256
        xhist['x17'] = x17
        x18 = self.layer8(x17)     #8x8x256
        xhist['x18'] = x18
        x19 = self.up1(x18,x10)    #16x16x512
        xhist['x19'] = x19
        x20 = self.layer9(x19)     #16x16x256
        xhist['x20'] = x20
        x21 = self.up2(x20,x8)     #32x32x256
        xhist['x21'] = x21
        x22 = self.layer10(x21)    #32x32x128
        xhist['x22'] = x22
        x23 = self.up3(x22,x7)     #64x64x128
        xhist['x23'] = x23
        x24 = self.layer11(x23)    #64x64x64
        xhist['x24'] = x24
        x25 = self.up4(x24,x6)     #128x128x64
        xhist['x25'] = x25
        x26 = self.layer12(x25)    #128x128x32
        xhist['x26'] = x26
        x27 = self.up5(x26,x5)     #256x256x32
        xhist['x27'] = x27
        x28 = self.layer13(x27)    #256x256x16
        xhist['x28'] = x28
        return x28, xhist


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, **kwargs)
    return model

class CoordRegressionNetwork(nn.Module):
    def __init__(self,n_locations):
        super(CoordRegressionNetwork, self).__init__()
        self.fcn = resnet18()
        self.hm_conv = nn.Conv2d(16, n_locations, kernel_size=1,bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, images):
        fcn_out, xhist = self.fcn(images)
        unnormalized_heatmaps = self.hm_conv(fcn_out)
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        coords = dsntnn.dsnt(heatmaps)
        return coords, heatmaps, xhist

model = CoordRegressionNetwork(n_locations=1).cuda()


optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
#optimizer = optim.Adam(model.parameters(),lr=0.0002,betas=(0.9,0.999),eps=1e-8)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)


load = True
#model_path = '/home/kfj5051/Desktop/Matt/Deepit_data/original.pth'
#model_new_path = '/home/kfj5051/Desktop/Matt/Deepit_data/original.pth'
#model_path = '/home/kfj5051/Desktop/Matt/Deepit_data/original13.pth'
#model_path = '/home/kfj5051/Desktop/Matt/Deepit_data/original26.pth'
#model_path = '/home/kfj5051/Desktop/Matt/savedmodels/Noisuy.pth'
model_path = '/home/kfj5051/Machine_Learning/First_Paper/Reproducible/noisy6/noisy_99/model.pth'
#model_path = '/home/kfj5051/Machine_Learning/First_Paper/Reproducible/noisy/noisy_99/model.pth'

if load == True:
    model.load_state_dict(torch.load(model_path))
model = model.cuda()

#-----------------------------------------------------------------------------
#          -Training-
#-----------------------------------------------------------------------------
Train = False
start = time.time() #Time the training
if Train:
    train_size = X_Train.size(0)
    test_size = X_Test.size(0)
    n = 100#number of epochs
    Train_loss_hist = torch.Tensor(n).numpy()
    Test_loss_hist = torch.Tensor(n).numpy()
    Train_acc_hist = torch.Tensor(n).numpy()
    Test_acc_hist = torch.Tensor(n).numpy()
    start = time.time() #Time the training
    for epoch in range(n):   
        model.train()
        print('Epoch: %d' % epoch)
        running_train_loss = 0.0
        running_test_loss = 0.0
        train_correct = 0 #running count of correct predictions on training set
        test_correct = 0 #running count of correct predictions on test set
        total = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # load to GPU
            inputs = inputs.cuda()
            labels = labels.cuda()
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            #forward pass
            coords, heatmaps, xhist = model(inputs)
            #Per-location euclidean losses
            euc_losses = dsntnn.euclidean_losses(coords, labels)
            #Per-location regularization losses
            reg_losses = dsntnn.js_reg_losses(heatmaps,labels,sigma_t=1.0)
            #Combine losses into an overall loss
            loss = dsntnn.average_loss(euc_losses+reg_losses)
            #Calculate gradients
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)
            loss.backward()
            #Update model parameters
            optimizer.step()
            # accrue loss
            running_train_loss += loss.data[0]
            
        for data in testloader:
            testin, testlabels = data
            # Load to GPU
            testin = testin.cuda()
            testlabels = testlabels.cuda()
            # Wrap in Variable
            testin = Variable(testin)
            testlabels = Variable(testlabels)
            # Evaluate
            model.eval()
            coords, heatmaps, xhist = model(testin)
            euc_losses = dsntnn.euclidean_losses(coords, testlabels)
            reg_losses = dsntnn.js_reg_losses(heatmaps,testlabels, sigma_t=1.0)
            loss = dsntnn.average_loss(euc_losses+reg_losses)
            running_test_loss += loss.data[0]
            
            
        
        Train_loss_hist[epoch] = running_train_loss
        Test_loss_hist[epoch] = running_test_loss
        scheduler.step(running_test_loss)
        print('loss: %.10f' % running_train_loss)
        if (epoch+1 >= 50):
            dir_name = '/home/kfj5051/Machine_Learning/First_Paper/Reproducible/noisy6/noisy_'+str(epoch)
            os.mkdir(dir_name)
            torch.save(model.state_dict(), dir_name+'/model.pth')
            savemat(dir_name+'/Train_loss_hist',mdict = {'Train_loss_hist':Train_loss_hist})
            savemat(dir_name+'/Test_loss_hist_',mdict = {'Test_loss_hist':Test_loss_hist})
    
    
stop = time.time()
print('Finished Training')
print('Elapsed Time: %d seconds' % (stop-start))


#-----------------------------------------------------------------------------
#                   -Evaluation-
#-----------------------------------------------------------------------------


if not load:
    print('\n\n Training Loss vs Epoch')
    plt.figure(2)
    plt.plot(Train_loss_hist)
    plt.ylabel('Training Loss')
    plt.show()
    print('\n\n Test Loss vs Epoch')
    plt.figure(3)
    plt.plot(Test_loss_hist)
    plt.ylabel('Test Loss')
    plt.show()


#-----------------------------------------------------------------------------
#                   -Visualization-
#-----------------------------------------------------------------------------
for param_group in optimizer.param_groups:
    print(param_group['lr'])
    
batch = 0
errors = torch.Tensor(4060,6)
heat = torch.Tensor(4060,256,256)
nplabels = torch.Tensor(bs,1,2)
npcoords = torch.Tensor(bs,1,2)
bw = bs #batch width to decrease at last batch



for data in testloader:
    images, labels = data    
    coords, heatmaps, xhist = model(images.cuda()) #imagehist is a dictionary with keys 'x0','x1','x2','x3'
    if batch*bs+bw >= errors.shape[0]:
        bw = errors.shape[0]-batch*bs
        nplabels = torch.Tensor(bw,1,2)
        npcoords = torch.Tensor(bw,1,2)
    
    for i in range(bw):
        print(1+i+batch*bs)
        #plt.imshow(ndimage.rotate(images[i,0,:,:],90))
        #plt.show()
        #plt.imshow(ndimage.rotate(heatmaps[i,0,:,0:11201].detach().cpu().numpy(),90))
        #plt.show()
        
        nplabels[i,0,0] = labels[i,0,0].detach().cpu()*((max_ax-min_ax)/2/max_normalized_depth)+((max_ax-min_ax)/2+min_ax)
        nplabels[i,0,1] = labels[i,0,1].detach().cpu()*((max_ax-min_ax)/2/max_normalized_depth)
        npcoords[i,0,0] = coords.detach().cpu()[i,0,0]*((max_ax-min_ax)/2/max_normalized_depth)+((max_ax-min_ax)/2+min_ax)
        npcoords[i,0,1] = coords.detach().cpu()[i,0,1]*((max_ax-min_ax)/2/max_normalized_depth)
        #        coords[i,0,0] = coords[i,0,0]*50+50
        #        coords[i,0,1] = coords[i,0,1]*50
        print(nplabels[i])
        print(npcoords[i])
        print(bw)
            
    heat[batch*bs:batch*bs+bw,:,:] = heatmaps[:,0,:,:].detach().cpu()
    errors[batch*bs:batch*bs+bw,0:2] = nplabels[:,0,:]
    errors[batch*bs:batch*bs+bw,2:4] = npcoords[:,0,:]
    errors[batch*bs:batch*bs+bw,4:6] = abs(npcoords[:,0,:]-nplabels[:,0,:])
    batch += 1


#plt.plot(Test_loss_hist)
latmean = np.mean(errors[:,5].numpy())
latstd = np.std(errors[:,5].numpy())
axmean = np.mean(errors[:,4].numpy())
axstd = np.std(errors[:,4].numpy())
eucmean = np.mean(np.sqrt(np.power(errors[:,5].numpy(),2)+np.power(errors[:,4].numpy(),2)))
eucstd = np.std(np.sqrt(np.power(errors[:,5].numpy(),2)+np.power(errors[:,4].numpy(),2)))
#Check algorithm on image at a point which is a multiple of 5mm
#sensor_data_str = loadmat('/home/kfj5051/Documents/Sumit_Machine Learning Dataset/sensor_data_needed_alt_location.mat')
#sensor_data_image = sensor_data_str['sensor_data_needed']
#sensor_data_str2 = loadmat('/home/kfj5051/Documents/Sumit_Machine Learning Dataset/sensor_data_needed_alt_location2.mat')
#sensor_data_image2 = sensor_data_str['sensor_data_needed']


print('\n Mean lateral error: %f' % latmean)
print('\n STD of lateral error: %f' %latstd)
print('\n Mean axial error: %f' % axmean)
print('\n STD of axial error: %f' %axstd)
print('\n Mean euclidian error: %f' % eucmean)
print('\n STD of euclidian error: %f' %eucstd)
if load == False:
    torch.save(model.state_dict(), model_path) 
