# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:31:43 2021

@author: 17783
"""

import torch
import os
import torch.nn as nn
import numpy as np
import scipy.io as sio
#from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
#from operator import truediv
#from torch.utils.data.dataset import Dataset
#import torch.nn as nn
import Model
import random








class TrainDS(torch.utils.data.Dataset):
    def __init__(self, Xtrain, ytrain):
        self.ytrain=ytrain
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        # 返回文件数据的数目
        return len(self.ytrain)

""" Testing dataset"""
class TestDS(torch.utils.data.Dataset):
    def __init__(self, Xtest, ytest):
        self.len = len(ytest)
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        # 返回文件数据的数目
        return self.len


#def split_data_fix(pixels, labels, n_samples, rand_state=None):
#    pixels_number = np.unique(labels, return_counts=1)[1]
#    train_set_size = [n_samples] * len(np.unique(labels))
#    tr_size = int(sum(train_set_size))
#    te_size = int(sum(pixels_number)) - int(sum(train_set_size))
#    sizetr = np.array([tr_size]+list(pixels.shape)[1:])
#    sizete = np.array([te_size]+list(pixels.shape)[1:])
#    train_x = np.empty((sizetr)); train_y = np.empty((tr_size)); test_x = np.empty((sizete)); test_y = np.empty((te_size))
#    trcont = 0; tecont = 0;
#    for cl in np.unique(labels):
#        pixels_cl = pixels[labels==cl]
#        labels_cl = labels[labels==cl]
#        pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
#        for cont, (a,b) in enumerate(zip(pixels_cl, labels_cl)):
#            if cont < train_set_size[cl]:
#                train_x[trcont,:,:,:] = a
#                train_y[trcont] = b
#                trcont += 1
#            else:
#                test_x[tecont,:,:,:] = a
#                test_y[tecont] = b
#                tecont += 1
#    train_x, train_y = random_unison(train_x, train_y, rstate=rand_state)
#    return train_x, test_x, train_y, test_y
#
#def split_data(pixels, labels, percent, splitdset="custom", rand_state=69):
#    splitdset = "sklearn"
#    if splitdset == "sklearn":
#        return train_test_split(pixels, labels, test_size=(1-percent), stratify=labels, random_state=rand_state)
#    elif splitdset == "custom":
#        pixels_number = np.unique(labels, return_counts=1)[1]
#        train_set_size = [int(np.ceil(a*percent)) for a in pixels_number]
#        tr_size = int(sum(train_set_size))
#        te_size = int(sum(pixels_number)) - int(sum(train_set_size))
#        sizetr = np.array([tr_size]+list(pixels.shape)[1:])
#        sizete = np.array([te_size]+list(pixels.shape)[1:])
#        train_x = np.empty((sizetr)); train_y = np.empty((tr_size)); test_x = np.empty((sizete)); test_y = np.empty((te_size))
#        trcont = 0; tecont = 0;
#        for cl in np.unique(labels):
#            pixels_cl = pixels[labels==cl]
#            labels_cl = labels[labels==cl]
#            pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
#            for cont, (a,b) in enumerate(zip(pixels_cl, labels_cl)):
#                if cont < train_set_size[cl]:
#                    train_x[trcont,:,:,:] = a
#                    train_y[trcont] = b
#                    trcont += 1
#                else:
#                    test_x[tecont,:,:,:] = a
#                    test_y[tecont] = b
#                    tecont += 1
#        train_x, train_y = random_unison(train_x, train_y, rstate=rand_state)
#        return train_x, test_x, train_y, test_y


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=3, removeZeroLabels = True):
    
    hsi_number=16  #高光谱数据地物的种类
    
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

def data_augmentation(cla_Data,cla_Labels,cla_flag):
    windowSize=3
    '''
    cla_Data------切割后的三维输入高光谱数据块，去掉背景
    cla_Labels----标签
    cla_flag---------每个类别的数量
    '''
    DataAug_num=int(sum(cla_flag)/len(cla_flag))
    
    all_AugData= np.zeros((DataAug_num*len(cla_flag), 3, 3, 224))
    all_Auglabels = np.zeros((DataAug_num*len(cla_flag)))
    
    lll=0
    for i in range(len(cla_flag)):
        #cla_Data[label_num, :, :, :] = patchesData[k, :, :, :]
        #cla_Labels[label_num] = patchesLabels[k]
        #label_num+=1
        '''     根据不同的数据集进行修改           '''
        Aug_patchesData = np.zeros((DataAug_num, 3, 3, 224))
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

def AugData_split(all_AugData,all_Auglabels,cla_Data, cla_Labels, cla_flag):
    # x1 = len(cla_Labels)
    # x2 = len(all_Auglabels)
    
    aug_data=int((sum(cla_flag)/len(cla_flag))*0.1)
    
    split_AugData = np.zeros((aug_data*len(cla_flag), 3, 3, 224))
    split_Auglabels = np.zeros((aug_data*len(cla_flag)))
    
    # for i in range(x1):
    #     split_AugData[i,:,:,:] = cla_Data[i,:,:,:]
    #     split_Auglabels[i] = cla_Labels[i]
    # for j in range(x2):
    #     split_AugData[x1+j,:,:,:] = all_AugData[j,:,:,:]
    #     split_Auglabels[x1+j] = all_Auglabels[j]
    
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
            
            #a_x.append(a[0:aug_data])

            # file=open('C://Users//17783//Desktop//data.txt','w')
            # file.write(str(a_x));
            # file.close()

            
            for k in range(aug_data):
                split_AugData[mmm,:,:,:] = samp_Data[a[k],:,:,:]
                split_Auglabels[mmm] = samp_Labels[a[k]]
                mmm+=1

        if cla_flag[i]<aug_data:
            samp_Data = cla_Data[cla_Labels==i,:,:,:]
            samp_Labels = cla_Labels[cla_Labels==i]
            for k in range(len(samp_Labels)):
                split_AugData[mmm,:,:,:] = samp_Data[a[k],:,:,:]
                split_Auglabels[mmm] = samp_Labels[a[k]]
                mmm+=1
            for l in range(aug_data-len(samp_Labels)):
                samp_Data = all_AugData[cla_Labels==i,:,:,:]
                samp_Labels = all_Auglabels[cla_Labels==i]
                a=[]
                for xx in range(cla_flag[i]):
                    a.append(xx)
                random.shuffle(a)
                
                split_AugData[mmm,:,:,:] = samp_Data[a[l],:,:,:]
                split_Auglabels[mmm] = samp_Labels[a[l]]
                mmm+=1
    #np.save(r"C://Users//17783//Desktop//data.npy", a_x)
    return split_AugData,split_Auglabels





data_path_indian = 'D://Hyperspectral_image//HSI//Salinas.mat'
label_path_indian = 'D://Hyperspectral_image//HSI//Salinas_gt.mat'
    #data_path_pavia = 'dataset/Pavia_corrected.mat'
    #label_path_pavia = 'dataset/Pavia_gt.mat'
    #data_path_salinas = 'dataset/Salinas_corrected.mat'
    #label_path_salinas = 'dataset/Salinas_gt.mat'
    
data_path = data_path_indian
label_path = label_path_indian
    
    
X = sio.loadmat(data_path)
    
X=X['salinas']
    
shapeor = X.shape
data = X.reshape(-1, X.shape[-1])
data = StandardScaler().fit_transform(data)
X = data.reshape(shapeor)
    
    
y_label=sio.loadmat(label_path_indian)
y_label = y_label['salinas_gt']
patch_size=3
    
    #patchesData, patchesLabels = createImageCubes(X, y_label, windowSize=patch_size)
    
cla_Data,cla_Labels,cla_flag = createImageCubes(X, y_label,windowSize=patch_size)

aaa_data,aaa_label=data_augmentation(cla_Data,cla_Labels,cla_flag)

auged_data,auged_labels=AugData_split(aaa_data,aaa_label,cla_Data, cla_Labels, cla_flag)




Xtrain = auged_data

ytrain  = auged_labels


test_x=torch.FloatTensor(Xtrain)


#Xtrain = Xtrain.transpose(0, 3, 1, 2)
#Xtest  = Xtest.transpose(0, 3, 1, 2)
print('after transpose: Xtrain shape: ', Xtrain.shape)
#print('after transpose: Xtest  shape: ', Xtest.shape)

trainset = TrainDS(Xtrain, ytrain)
testset = TestDS(Xtrain, ytrain)
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=100, shuffle=True)


# max_sent_len=35, batch_size=50, embedding_size=300
#conv1 = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=3)

conv1 = nn.Conv3d(1, 20, (3, 3, 3), stride=(1, 1, 1))

test_x=test_x.unsqueeze(1)
#input = torch.randn(50, 35, 300)
# batch_size x max_sent_len x embedding_size -> batch_size x embedding_size x max_sent_len
#input = input.permute(0, 2, 1)
print("input:", test_x.size())
output = conv1(test_x)
print("output:", output.size())

conv2 = nn.Conv3d(in_channels=20, out_channels=12, kernel_size=[1,1,3], stride=1)

print("input2:", output.size())
output = conv2(output)
print("output2:", output.size())


# conv3 = nn.Conv1d(in_channels=12, out_channels=24, kernel_size=[1,3], stride=1)

# print("input2:", output.size())
# output = conv3(output)
# print("output2:", output.size())

# conv4 = nn.Conv1d(in_channels=24, out_channels=8, kernel_size=[1,3], stride=1)

# print("input2:", output.size())
# output = conv4(output)
# print("output2:", output.size())







# 最大池化
'''
pool1d = nn.MaxPool1d(kernel_size=35-3+1)
pool1d_value = pool1d(output)
print("最大池化输出：", pool1d_value.size())
# 全连接
fc = nn.Linear(in_features=100, out_features=2)
fc_inp = pool1d_value.view(-1, pool1d_value.size(1))
print("全连接输入：", fc_inp.size())
fc_outp = fc(fc_inp)
print("全连接输出：", fc_outp.size())
# softmax
m = nn.Softmax()
out = m(fc_outp)
print("输出结果值：", out)
'''