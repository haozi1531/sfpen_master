# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:00:05 2021

@author: 17783
"""


import torch
import os
import torch.nn as nn
import numpy as np
import scipy.io as sio
#from sklearn import processing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
#from operator import truediv
from torch.utils.data.dataset import Dataset
#import torch.nn as nn
import new_Model as Model
import random
from torch.nn import init



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


def padWithZeros(X, margin=1):
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
    #DataAug_num=max(cla_flag)*5
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
    
    #aug_data=int((sum(cla_flag)/len(cla_flag))*0.3)
    aug_data=200
    split_AugData = np.zeros((aug_data*len(cla_flag), 3, 3, 224))
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
    
#data_path_indian = root+'//HSI//Indian_pines.mat'

#data_path_indian = 'D://Hyperspectral_image//HSI//denoise_salinas.mat'
#label_path_indian = root+'//HSI//Indian_pines_gt.mat'
data_path_pavia = root + '//HSI//Salinas.mat'
label_path_pavia = root + '//HSI//Salinas_gt.mat'
    #data_path_salinas = 'dataset/Salinas_corrected.mat'
    #label_path_salinas = 'dataset/Salinas_gt.mat'
    
data_path = data_path_pavia
label_path = label_path_pavia
    
    
X = sio.loadmat(data_path)
    
X=X['salinas']
    
shapeor = X.shape
data = X.reshape(-1, X.shape[-1])
data = StandardScaler().fit_transform(data)
X = data.reshape(shapeor)
    
    
y_label=sio.loadmat(label_path)

y_label = y_label['salinas_gt']


#X= = preprocessing.normalize(X, norm='l2')


patch_size=3
    
#patchesData, patchesLabels = createImageCubes(X, y_label, windowSize=patch_size)
    
cla_Data,cla_Labels,cla_flag = createImageCubes(X, y_label,windowSize=patch_size)

aaa_data,aaa_label=data_augmentation(cla_Data,cla_Labels,cla_flag)

auged_data,auged_labels=AugData_split(aaa_data,aaa_label,cla_Data,cla_Labels,cla_flag)

Xtrain = auged_data

ytrain  = auged_labels


''' 
creat_txt-------训练集的文件目录  
creat_txt1--------所有不包含训练集的数据集
'''
        
train_batch_size = 32

Epoch = 200



''' 删除上一代增加的伪标签 '''

''' Xtrain,ytrain '''

trainset = TrainDS(Xtrain, ytrain)
    
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=train_batch_size, shuffle=True)           




testset = TestDS(cla_Data, cla_Labels)
    
#train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=train_batch_size, shuffle=True)     
    

        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
'''   labels-----预测的数据集长度    '''
label_x=[[0]*2 for row in range(len(auged_labels))]

popList=[[33,96],[78,96]]


 
for i in range(2):
    layerList=popList[i]
    model = Model.Net(layerList).to(device)
    print(model)
        
    creterion = nn.CrossEntropyLoss()
        
    optimer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
    for epcho in range(0,Epoch):
        trained_data = 0
        for indx, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            out = model(image)
            loss = creterion(out, label)
            optimer.zero_grad()
            loss.backward()
            optimer.step()
            trained_data = len(image) * (indx + 1)
            if (indx + 1) % 10 == 0:
                print('{}/{}               第{}轮第{}批的loss = {}'.format(trained_data, len(train_loader.dataset), epcho,
                         indx,loss))

    ttt=0
    test_loader  = torch.utils.data.DataLoader(dataset=testset,batch_size=1,shuffle=False)
    '''  labels--------将要预测的无标签数据总数    '''
        
    for k in range(len(cla_Labels)):
            
        testset.__getitem__(k)

        test_loader  = torch.utils.data.DataLoader(dataset=testset,  batch_size=1, shuffle=False)

        image=test_loader.dataset.__getitem__(k)[0]

        #image = image.unsqueeze(1)
                
        image = image.to(device)
    
        with torch.no_grad(): # 在测试的时候必须加上这行和下一行代码，否则预测会出问题，这里是防止还有梯度更新这些，而且如果不加这个，后面又没有进行梯度更新的话，可能会报显存不够用的错误，我怀疑是数据没有被清理
            model.eval() # 使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval，eval（）时，框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大！！！！！！         
            out = model(image)

            pred = torch.argmax(out, dim = 1)
    
            xxx = pred.cpu().numpy()
    
            xxx.tolist()
    
            label_x[k]=xxx[0]                

            if xxx[0]==test_loader.dataset.__getitem__(k)[1]:
                ttt+=1

            print('{}  {}'.format(xxx[0],test_loader.dataset.__getitem__(k)[1]))
        aacc1=ttt/54129

acce_num=0
        
pseulabel_xtrain = np.zeros((len(cla_Labels), 3, 3, 224))
        
pseulabel_labels = np.zeros((len(cla_Labels)))
        
pseulabel_num = 0 #增加的的伪标签数量

my_iter=1        
with open(root + '/accept_txt'+ str(my_iter) + '.txt','w') as f:         
    '''  labels---------       '''
    for i in range(len(cla_Labels)):
        #if label_x[i][0]==label_x[i][1] and label_x[i][0]==label_x[i][2] and label_x[i][1]==label_x[i][2] and random.random()<0.3:
        if label_x[i][0]==label_x[i][1]:
            acce_num=acce_num+1
            f.writelines(str(label_x[i][0]))
            f.writelines(' ' + cla_Labels[i] + '\n')
            
            pseulabel_xtrain[pseulabel_num,:,:,:] = cla_Labels[i,:,:,:]
            
            pseulabel_labels[pseulabel_num] = cla_Labels[i]
            
            pseulabel_num += 1
            print('接受的标签{},图片名称{}'.format(label_x[i][0],cla_Labels[i]))
    f.writelines('接受的伪标签总数' + str(acce_num) + '\n')
    f.writelines('伪标签预测准确率' + str(aacc1))
f.close()                
print('扩充的标签总数{}'.format(acce_num))
        


add_psenum=400*16


add_pseudata = np.zeros((len(cla_Labels) + add_psenum, 3, 3, 224))
        
add_pseulabel = np.zeros((len(cla_Labels) + add_psenum))        
        

a=[]
for xx in range(pseulabel_num ):
    a.append(xx)
random.shuffle(a)

        
        
for i in range(add_psenum):

    add_pseudata[i,:,:,:] = pseulabel_xtrain[a[i],:,:,:]
                        
    add_pseulabel[i] = pseulabel_labels[a[i]]
        
for i in range(len(auged_data)):

    add_pseudata[i+add_psenum,:,:,:] = auged_data[i,:,:,:]
                        
    add_pseulabel[i+add_psenum] = auged_data[i]
        
        
'''----------------------------------------------------------------------'''        


trainset = TrainDS(add_pseudata, add_pseulabel)

#testset = TestDS(Xtrain, ytrain)

train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=100, shuffle=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model.Net([78,96]).to(device)



print(model)

Epoch =200
min_loss=10
criterion = nn.CrossEntropyLoss().to(device)
optimer = torch.optim.Adam(model.parameters(), lr=0.00001)
for epcho in range(0,Epoch):
    #trained_data = 0
    for indx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
                
        optimer.zero_grad()
                    
        out = model(image)
                   
        loss = criterion(out, label)
                    #optimer.zero_grad()
        loss.backward()
        optimer.step()
                    
        loss_m=loss
                    
        loss_x=loss_m.tolist()
        # if min_loss>loss_x:
        #     min_loss=loss_x
                    #        optimer.zero_grad()
                    #        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        trained_data = len(image) * (indx + 1)
        
        print('{}/{}               第{}轮第{}批的loss = {}'.format(trained_data, len(train_loader.dataset), epcho,
                                                                      indx,loss))



test_data = TestDS(cla_Data,cla_Labels)


label_x=[[0] for row in range(len(cla_Labels))]

ttt=0
for k in range(len(cla_Labels)):
    # print(train_data.__getitem__(1))
    # train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False,num_workers=1)
    # print('num_of_trainData:', len(train_data))
    # print(train_data.imgs[1])
    # print(train_loader.dataset.__getitem__(1))
    
    test_data.__getitem__(k)
    
    test_loader  = torch.utils.data.DataLoader(dataset=test_data,  batch_size=1, shuffle=False)
    
    # print('num_of_trainData:', len(train_data))
    
    #test_data.imgs[k]
    
    image=test_loader.dataset.__getitem__(k)[0]
    
    
    image = image.unsqueeze(1)
    image = image.to(device)
    
    #label = label.to(device)
    
    out = model(image)
    
    #loss = creterion(out, label)
    
    pred = torch.argmax(out, dim = 1)
    
    xxx = pred.cpu().numpy()
    
    xxx.tolist()
    
    label_x[k]=xxx[0]                
    
    #print(xxx)
    
    if xxx[0]==test_loader.dataset.__getitem__(k)[1]:
        ttt+=1

    print('{}  {}'.format(xxx[0],test_loader.dataset.__getitem__(k)[1]))
#aacc=ttt/10249
aacc=ttt/54129
print(aacc)

wh_x=label_x

import os
from PIL import Image
import numpy as np
import scipy.io as sio
# 读取本地文件，文件格式为txt,将文件中的数据转存在一个list列表中




salinas_color=[[0,0,0],[255,255,143],[0,0,254],[255,51,0],[0,255,153],
               [255,0,254],[0,51,205],[51,153,251],[127,128,0],
               [0,255,1],[153,0,153],[0,153,203],[101,103,152],
               [147,209,76],[102,50,0],[1,255,205],[254,254,3]]




root = os.getcwd()
label_path_indian = root+'//HSI//Indian_pines_gt.mat'
    #data_path_pavia = 'dataset/Pavia_corrected.mat'
    #label_path_pavia = 'dataset/Pavia_gt.mat'
    #data_path_salinas = 'dataset/Salinas_corrected.mat'
    #label_path_salinas = 'dataset/Salinas_gt.mat'
    

label_path = label_path_indian
   
y_label=sio.loadmat(label_path_indian)

salinas_gt = y_label['indian_pines_gt']


'''     salinas           '''
# d=[[0] for j in range(207400)]

# for i in range(164624):
#     d[i]=0
# for i in range(0,54129):
#     d[i+56975]=wh_x[i]+1

# list_result = np.empty((512,217,3))



# m=0

# for l in range(17):
#     for i in range(512):
#         for j in range(217):
#             if salinas_gt[i][j] == l:
#                 list_result[i][j][0]=salinas_color[d[m]][0]    
#                 list_result[i][j][1]=salinas_color[d[m]][1]
#                 list_result[i][j][2]=salinas_color[d[m]][2]
#                 m=m+1

# xxxx = list_result.astype(np.uint8)


'''  paviau '''


d=[[0] for j in range(21025)]


for i in range(10776):
    d[i]=0
for i in range(0,10249):
    d[i+10776]=wh_x[i]+1

list_result = np.empty((145,145,3))




m=0

for l in range(17):
    for i in range(145):
        for j in range(145):
            if salinas_gt[i][j] == l:
                list_result[i][j][0]=salinas_color[d[m]][0]    
                list_result[i][j][1]=salinas_color[d[m]][1]
                list_result[i][j][2]=salinas_color[d[m]][2]
                m=m+1

xxxx = list_result.astype(np.uint8)



new_map = Image.fromarray(xxxx,"RGB")

# 显示图像

#new_map.show()

filename=root+'//solution.jpg' 

new_map.save(filename)