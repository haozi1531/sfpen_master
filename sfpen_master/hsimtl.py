# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:25:06 2021

@author: 17783
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
import pandas as pd
import math
import sklearn.preprocessing as sk
from tensorboardX import SummaryWriter
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import random
import scipy.io as sio
from torch.utils.data.dataset import Dataset

global band_shuliang
band_shuliang=224

global class_shuliang
class_shuliang=16


patch_size=3

class TrainDS(Dataset):
    def __init__(self, Xtrain, ytrain):
        self.ytrain=ytrain
        self.x_data = torch.FloatTensor(Xtrain)
        self.x_data = self.x_data.unsqueeze(1)
        self.y_data = torch.LongTensor(ytrain)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        # 返回文件数据的数目
        return len(self.ytrain)



class TrainDAE(Dataset):
    def __init__(self, Xtrain, ytrain):
        self.ytrain=ytrain
        self.x_data = torch.FloatTensor(Xtrain)
        self.x_data = self.x_data.unsqueeze(1)
        self.y_data = torch.FloatTensor(ytrain)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        # 返回文件数据的数目
        return len(self.ytrain)




""" Testing dataset"""
class TestDS(Dataset):
    def __init__(self, Xtest, ytest):
        self.len = len(ytest)
        self.x_data = torch.FloatTensor(Xtest)
        self.x_data = self.x_data.unsqueeze(1)
        self.y_data = torch.LongTensor(ytest)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        # 返回文件数据的数目
        return self.len


def layerlist_code(multi_layerlist):
    core_list=[3]*len(multi_layerlist)
    num_list=multi_layerlist
    
    for i in range(len(multi_layerlist)):
        if multi_layerlist[i]<20:
            multi_layerlist[i]=20
        if multi_layerlist[i]>260:
            multi_layerlist[i]=260
        else:
            if multi_layerlist[i]>=20 and multi_layerlist[i]<100:
                num_list[i]=multi_layerlist[i]
                core_list[i]=3
            if multi_layerlist[i]>=100 and multi_layerlist[i]<180:
                num_list[i]=multi_layerlist[i]-80
                core_list[i]=5
            if multi_layerlist[i]>=180 and multi_layerlist[i]<=260:
                num_list[i]=multi_layerlist[i]-160
                core_list[i]=7

    return num_list,core_list 


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX



def createImageCubes(X, y, windowSize=3, removeZeroLabels = True):
    
    hsi_number=16#高光谱数据地物的种类
    
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            #import matplotlib.pyplot as plt
            #plt.imshow(patch[:, :, 100])
            #plt.show()
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    
    cla_Data = np.zeros((patchesLabels.shape[0], windowSize, windowSize, X.shape[2]))
    cla_Labels = np.zeros((patchesLabels.shape[0]))
    label_num=0
    for i in range(hsi_number):
        for k in range(patchesLabels.shape[0]):
            if patchesLabels[k]==i:
                
                cla_Data[label_num, :, :, :] = patchesData[k, :, :, :]
                cla_Labels[label_num] = patchesLabels[k]
                label_num+=1

    cla_flag=[0]*hsi_number#高光谱地物分类的类数          
    for i in range(hsi_number):
        for k in range(patchesLabels.shape[0]):
            if patchesLabels[k]==i:
                cla_flag[i]+=1     #每类分类目标的数量
    #print(cla_flag)
    return cla_Data, cla_Labels.astype("int"),cla_flag
'''
def data_augmentation(cla_Data,cla_Labels,cla_flag):
    windowSize=3
    '''
'''
    cla_Data------切割后的三维输入高光谱数据块，去掉背景
    cla_Labels----标签
    cla_flag---------每个类别的数量
    '''
'''
    DataAug_num=int(sum(cla_flag)/len(cla_flag))
    #DataAug_num=max(cla_flag)*5
    all_AugData= np.zeros((DataAug_num*len(cla_flag), windowSize, windowSize, band_shuliang))
    all_Auglabels = np.zeros((DataAug_num*len(cla_flag)))

    lll=0
       
    for i in range(len(cla_flag)):
        #cla_Data[label_num, :, :, :] = patchesData[k, :, :, :]
        #cla_Labels[label_num] = patchesLabels[k]
        #label_num+=1

        Aug_patchesData = np.zeros((DataAug_num, windowSize, windowSize, band_shuliang))
        Aug_patchesLabels = np.zeros((DataAug_num))
        
        AugData = cla_Data[cla_Labels==i,:,:,:]
        AugLabels = cla_Labels[cla_Labels==i]
        for k in range(DataAug_num):#扩充数据的数量，每类数据的均值
            x=random.sample(range(0,len(AugLabels)),2)
            a=random.randint(1,3)
            b=[0,1,2]
            random.shuffle(b)
            c=random.randint(1,3)
            d=[0,1,2]
            random.shuffle(d)
            #Aug_patchesLabels = AugLabels
            Aug_patchesData[k,:,:,:] = AugData[x[0],:,:,:]
            
            for m in range(a):
                for n in range(c):
                    Aug_patchesData[k,int(b[m]),int(d[n]),:] = AugData[x[1],int(b[m]),int(d[n]),:]
            #AugLabels[k]=i
            all_AugData[lll,:,:,:]=Aug_patchesData[k,:,:,:]
            all_Auglabels[lll]=AugLabels[1]
            lll+=1
    return all_AugData,all_Auglabels
'''
def AugData_split(cla_Data, cla_Labels, cla_flag):
    
    #aug_data=int((sum(cla_flag)/len(cla_flag))*0.3)
    aug_data=200
    
    windowSize=3
    
    split_AugData = np.zeros((aug_data*len(cla_flag), windowSize, windowSize, band_shuliang))
    split_Auglabels = np.zeros((aug_data*len(cla_flag)))

    
    mmm=0
    #a_x=[]
    for i in range(len(cla_flag)):
        if cla_flag[i]>=aug_data:
            samp_Data = cla_Data[cla_Labels==i,:,:,:]
            samp_Labels = cla_Labels[cla_Labels==i]
            a=[]
            for xx in range(cla_flag[i]):
                a.append(xx)
            random.shuffle(a)
    
            for k in range(aug_data):
                split_AugData[mmm,:,:,:] = samp_Data[a[k],:,:,:]
                split_Auglabels[mmm] = samp_Labels[a[k]]
                mmm+=1

        if cla_flag[i]<aug_data:
            
            #samp_Data1 = all_AugData[all_Auglabels==i,:,:,:]
            #samp_Labels1 = all_Auglabels[all_Auglabels==i]
            
            samp_Data = cla_Data[cla_Labels==i,:,:,:]
            samp_Labels = cla_Labels[cla_Labels==i]
            
            aug_time=aug_data//cla_flag[i]
            
            for aug in range(aug_time):
                for k in range(len(samp_Labels)):
                    split_AugData[mmm,:,:,:] = samp_Data[k,:,:,:]
                    split_Auglabels[mmm] = samp_Labels[k]
                    mmm+=1
            
            for k in range(aug_data-aug_time*cla_flag[i]):
                split_AugData[mmm,:,:,:] = samp_Data[k,:,:,:]
                split_Auglabels[mmm] = samp_Labels[k]
                mmm+=1
                
    return split_AugData,split_Auglabels


class MTLnet(nn.Module):
    def __init__(self):
        super(MTLnet, self).__init__()
 
        self.sharedlayer = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=[3,3,3], stride=1,padding=(0,0,1)),  #输出55*55 96
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(16)
        )
        self.tower1 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=20, kernel_size=[1,1,3], stride=1,padding=(0,0,1)),  #输出55*55 96
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(20),
            # nn.Dropout(0.5),
            # nn.Linear(4480, 16)
        )
        self.tower2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=24, kernel_size=[1,1,3], stride=1,padding=(0,0,1)),  #输出55*55 96
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(24),
        )     
        self.fc2 = nn.Sequential(
            #nn.Linear((hsi_len-2*len(layerList)) * layerList[1], 64),
            #nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(224*24, 16)
            #nn.Softmax(dim = 1),
            )        
        
        self.fc1 = nn.Sequential(
            #nn.Linear((hsi_len-2*len(layerList)) * layerList[1], 64),
            #nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(224*20, 16)
            #nn.Softmax(dim = 1),
            )        
    def forward(self, x):
        h_shared = self.sharedlayer(x)
        out1 = self.tower1(h_shared)
        out2 = self.tower2(h_shared)
        out1 = out1.view(out1.size(0), -1)

        out2 = out2.view(out2.size(0), -1)
       
        s1=self.fc1(out1)
        s2=self.fc2(out2)
        return s1, s2

root = os.getcwd()
    
#data_path_indian = root+'//denoiseHSI//PaviaU.mat'

#label_path_indian = root+'//denoiseHSI//PaviaU_gt.mat'


#data_path_indian = root+'//denoiseHSI//Indian_pines.mat'
#label_path_indian = root+'//denoiseHSI//Indian_pines_gt.mat'


data_path_indian = root+'//denoiseHSI//Salinas.mat'
label_path_indian = root+'//denoiseHSI//Salinas_gt.mat'

#label_path_indian = root+'//denoiseHSI//Indian_pines_gt.mat'



#data_path_indian = root+'//denoiseHSI//WHU_Hi_HongHu.mat'
#label_path_indian = root+'//denoiseHSI//WHU_Hi_HongHu_gt.mat'





data_path = data_path_indian
label_path = label_path_indian


X = sio.loadmat(data_path)


X=X['d_salinas']

    
#X=X['d_paviau']

#X=X['WHU_Hi_HongHu']


#X=X['d_indian_pines']
    
shapeor = X.shape
data = X.reshape(-1, X.shape[-1])
#data = StandardScaler().fit_transform(data)
X = data.reshape(shapeor)
    
    
y_label=sio.loadmat(label_path_indian)

#y_label = y_label['indian_pines_gt']

y_label = y_label['salinas_gt']

#y_label = y_label['paviaU_gt']

#X= = preprocessing.normalize(X, norm='l2')

#y_label = y_label['WHU_Hi_HongHu_gt']


    
    #patchesData, patchesLabels = createImageCubes(X, y_label, windowSize=patch_size)
    
cla_Data,cla_Labels,cla_flag = createImageCubes(X, y_label,windowSize=patch_size)

#aaa_data,aaa_label=data_augmentation(cla_Data,cla_Labels,cla_flag)

auged_data,auged_labels=AugData_split(cla_Data,cla_Labels,cla_flag)



Xtrain = auged_data

ytrain  = auged_labels


#Xtrain = Xtrain.transpose(0, 3, 1, 2)

#Xtest  = Xtest.transpose(0, 3, 1, 2)
print('after transpose: Xtrain shape: ', Xtrain.shape)

#print('after transpose: Xtest  shape: ', Xtest.shape)

trainset = TrainDS(Xtrain, ytrain)


train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=100, shuffle=True)



device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


model =MTLnet()


model = model.to(device)

print(model)



Loss_list = []
Accuracy_list = []

data=[]

Epoch =30
min_loss=10
criterion = nn.CrossEntropyLoss().to(device)


    


LR = 0.001
epoch = 50
  
    
 
#MTL = MTLnet()
optimer = torch.optim.Adam(model.parameters(), lr=LR)
#loss_func = nn.MSELoss()
cost1tr = []
cost2tr = []
cost1D = []
cost2D = []
cost1ts = []
cost2ts = []
costtr = []
costD = []
costts = [] 
 
for it in range(epoch):
    epoch_cost = 0
    epoch_cost1 = 0
    epoch_cost2 = 0
    num_minibatches = 100
    for indx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
                
        optimer.zero_grad()
                    
        out1,out2 = model(image)
                   
        l1 = criterion(out1, label)
        l2 = criterion(out2, label) 
        
        
        
        
        lx1=l1.tolist()
        costtr.append(lx1)
        

        loss =  (l1 + l2)/2
        
        optimer.zero_grad()
        loss.backward()
        #l2.backward()
        optimer.step()
        
        
        
        
        print(lx1)
        #epoch_cost = epoch_cost + (loss / num_minibatches)
        #epoch_cost1 = epoch_cost1 + (l1 / num_minibatches)
        #epoch_cost2 = epoch_cost2 + (l2 / num_minibatches)
    #costtr.append(loss)
    cost1tr.append(l1)
    cost2tr.append(l2)
    
    
    test=costtr[len(costtr)-10:len(costtr)]
    
    
    ave=np.mean(test)
    

    # with torch.no_grad():
    #     Yhat1D, Yhat2D = MTL(X_valid)
    #     l1D = loss_func(Yhat1D, Y1_valid.view(-1,1))
    #     l2D = loss_func(Yhat2D, Y2_valid.view(-1,1))
    #     cost1D.append(l1D)
    #     cost2D.append(l2D)
    #     costD.append((l1D+l2D)/2)
    #     print('Iter-{}; Total loss: {:.4}'.format(it, loss.item()))
    #     x=MTL

    
plt.plot(np.squeeze(costtr), '-r',np.squeeze(costD), '-b')
plt.plot(np.squeeze(cost1tr), '-b', np.squeeze(cost1D), '-b')
plt.plot(np.squeeze(cost2tr), '-y', np.squeeze(cost1D), '-b')
plt.ylabel('total cost')
plt.xlabel('iterations (per tens)')
plt.show() 
 
plt.plot(np.squeeze(cost1tr), '-r', np.squeeze(cost1D), '-b')
plt.ylabel('task 1 cost')
plt.xlabel('iterations (per tens)')
plt.show() 
 
plt.plot(np.squeeze(cost2tr),'-r', np.squeeze(cost2D),'-b')
plt.ylabel('task 2 cost')
plt.xlabel('iterations (per tens)')
plt.show()