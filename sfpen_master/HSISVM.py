# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 15:01:48 2021

@author: 17783
"""

from sklearn.svm import SVC
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from torch import optim
from torchvision.datasets import MNIST
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import os
from torchvision import transforms as T
#import cv2



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
#from tensorboardX import SummaryWriter
#import seaborn as sns
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
import scipy.io as sio
from torch.utils.data.dataset import Dataset

global band_shuliang
band_shuliang=224

global class_shuliang
class_shuliang=16


patch_size=1

class TrainDS(Dataset):
    def __init__(self, Xtrain, ytrain):
        self.ytrain=ytrain
        self.x_data = torch.FloatTensor(Xtrain)
        #self.y_data = self.x_data
        
        self.x_data = self.x_data.unsqueeze(1)
        
        self.y_data =  torch.FloatTensor(Xtrain)
        
        self.y_data=self.y_data.unsqueeze(1)
        #self.y = torch.FloatTensor(ytrain)
        #self.y_data = self.y_data.unsqueeze(1)
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


def createImageCubes(X, y, windowSize=1, removeZeroLabels = True):
    
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
    

    return cla_Data,cla_Labels.astype("int"),cla_flag


def AugData_split(cla_Data, cla_Labels, cla_flag):
    
    #aug_data=int((sum(cla_flag)/len(cla_flag))*0.3)
    aug_data=200
    
    windowSize=1
    
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
data = StandardScaler().fit_transform(data)
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

X=auged_data[:,0,0,:]
Y=auged_labels
y_train=cla_Data[:,0,0,:]
y_test=cla_Labels


clf = SVC(decision_function_shape='ovo')
clf.fit(X, Y) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
#dec = clf.decision_function(X[1,:])
#dec.shape[1] # 4 classes: 4*3/2 = 6
ans=clf.predict(y_train)


print(clf.predict(y_train))

label_x=[[0] for row in range(len(cla_Labels))]

for i in range(len(cla_Labels)):
    label_x[i]=ans[i]


wh_x=label_x

salinas_color=[[40,42,117],[45,51,130],[45,72,155],[71,88,166],[71,88,166],[77,199,244],
               [129,191,231],[139,208,181],[137,202,135],[166,206,56],[245,235,4],
               [254,204,19],[247,154,31],[238,61,35],[236,38,38],[204,34,39],[153,27,56]]

salinas_color=np.uint8(salinas_color)

root = os.getcwd()
label_path_indian = root+'//denoiseHSI//Salinas_gt.mat'
label_path = label_path_indian
  
y_label=sio.loadmat(label_path_indian)


salinas_gt = y_label['salinas_gt']


d=[[0] for j in range(111104)]

for i in range(56975):
    d[i]=0
for i in range(0,54129):
    d[i+56975]=wh_x[i]+1

list_result = np.empty((512,217,3))



m=0

for l in range(17):
    for i in range(512):
        for j in range(217):
            if salinas_gt[i][j] == l:
                list_result[i][j][0]=salinas_color[int(d[m])][0]    
                list_result[i][j][1]=salinas_color[int(d[m])][1]
                list_result[i][j][2]=salinas_color[int(d[m])][2]
                m=m+1

xxxx = list_result.astype(np.uint8)


dataNew= root+'//svmsalinas.mat'
import scipy.io

scipy.io.savemat(dataNew, mdict={'my_data': xxxx})



