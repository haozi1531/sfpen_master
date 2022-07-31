# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:55:03 2021

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
import Modelss as new_Model
import random
#from torch.nn import init
#import matplotlib.pyplot as plt
#from torchstat import stat



global band_shuliang
band_shuliang=220

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



root = os.getcwd()
    
#data_path_indian = root+'//denoiseHSI//PaviaU.mat'
#label_path_indian = root+'//denoiseHSI//PaviaU_gt.mat'

#data_path_indian = root+'//denoiseHSI//Salinas.mat'
#label_path_indian = root+'//denoiseHSI//Salinas_gt.mat'

data_path_indian = root+'//denoiseHSI//Indian_pines.mat'
label_path_indian = root+'//denoiseHSI//Indian_pines_gt.mat'


#data_path_indian = root+'//HSI//Indian_pines.mat'
#label_path_indian = root+'//HSI//Indian_pines_gt.mat'

#label_path_indian = root+'//denoiseHSI//Indian_pines_gt.mat'



#data_path_indian = root+'//denoiseHSI//WHU_Hi_HongHu.mat'
#label_path_indian = root+'//denoiseHSI//WHU_Hi_HongHu_gt.mat'





data_path = data_path_indian
label_path = label_path_indian


X = sio.loadmat(data_path)


#X=X['d_salinas']

    
X=X['d_indian_pines']

#X=X['WHU_Hi_HongHu']


#X=X['d_indian_pines']
    
shapeor = X.shape
data = X.reshape(-1, X.shape[-1])
#data = StandardScaler().fit_transform(data)
X = data.reshape(shapeor)
    
    
y_label=sio.loadmat(label_path_indian)

#y_label = y_label['indian_pines_gt']

#y_label = y_label['salinas_gt']

y_label = y_label['indian_pines_gt']

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

#testset = TestDS(Xtrain, ytrain)

train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=100, shuffle=True)

#test_loader  = torch.utils.data.DataLoader(dataset=testset,  batch_size=100, shuffle=False)


# train_loader=train_loader.unsqueeze(1)

# test_loader=test_loader.unsqueeze(1)



#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


layerList=[44,122,170,146]

#layerList=[28,77,258-180,20,139-100]


num_list,core_list=layerlist_code(layerList)

model = new_Model.Net(num_list,core_list)
#model = new_Model.LiEtAl()


#model = nn.DataParallel(model)
model = model.to(device)

#model = new_Model.HuEtAl().to(device)
print(model)

# stat(model,(1,3,3,103))

Loss_list = []
Accuracy_list = []

data=[]

Epoch = 500
min_loss=10
criterion = nn.CrossEntropyLoss().to(device)
optimer = torch.optim.Adam(model.parameters(), lr=0.00001)

#optimer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

#optimer = torch.optim.Rprop(model.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))



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
        data.append(loss_x)
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

        with torch.no_grad(): # 在测试的时候必须加上这行和下一行代码，否则预测会出问题，这里是防止还有梯度更新这些，而且如果不加这个，后面又没有进行梯度更新的话，可能会报显存不够用的错误，我怀疑是数据没有被清理
            model.eval() # 使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval，eval（）时，框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大！！！！！！      
    #label = label.to(device)
    
            out = model(image)
    
    #loss = creterion(out, label)
    
            pred = torch.argmax(out, dim = 1)
    
            xxx = pred.cpu().numpy()
    
            xxx.tolist()
    
            label_x[k]=xxx[0]                
        
            if xxx[0]==test_loader.dataset.__getitem__(k)[1]:
                ttt+=1
            print('{}  {}'.format(xxx[0],test_loader.dataset.__getitem__(k)[1]))
aacc=ttt/10249
#aacc=ttt/54129
#aacc=ttt/42776

#aacc=ttt/386693


print("OA:")
print(aacc)

    #定义两个数组

wh_x=label_x


hsi_number=16 #高光谱数据地物的种类

d_m=[[0] for j in range(hsi_number)]
for i in range(hsi_number):
    d_m[i]=0


mmm=0

for i in range(hsi_number):
    for j in range(cla_flag[i]):
        if cla_Labels[mmm]==label_x[mmm]:
            d_m[i]+=1
        mmm+=1


oa_d_m=[[0] for j in range(hsi_number)]


for i in range(hsi_number):
    oa_d_m[i]=d_m[i]/cla_flag[i]

oa=sum(oa_d_m)/hsi_number




print("AA:")

print(oa_d_m)

print("AA_ave")

print(oa)


fenlei=[[0] for j in range(hsi_number)]
for i in range(hsi_number):
    fenlei[i]=0

for i in range(hsi_number):
    for j in range(len(cla_Labels)):
        if label_x[j]==i:
            fenlei[i]=fenlei[i]+1


kapa_x=0
for i in range(hsi_number):
    kapa_x+=fenlei[i]*cla_flag[i]

kapal_x=kapa_x/(10249 * 10249)

kappa_x=(aacc-kapal_x)/(1-kapal_x)

print("kappa:")
print(kappa_x)


import os
from PIL import Image
#import numpy as np
import scipy.io as sio
# 读取本地文件，文件格式为txt,将文件中的数据转存在一个list列表中

''' salinas '''
'''
salinas_color=[[0,0,0],[255,255,143],[0,0,254],[255,51,0],[0,255,153],
               [255,0,254],[0,51,205],[51,153,251],[127,128,0],
               [0,255,1],[153,0,153],[0,153,203],[101,103,152],
               [147,209,76],[102,50,0],[1,255,205],[254,254,3]]
'''

'''
salinas_color=[[40,42,117],[45,51,130],[45,72,155],[71,88,166],[71,88,166],[77,199,244],
               [129,191,231],[139,208,181],[137,202,135],[166,206,56],[245,235,4],
               [254,204,19],[247,154,31],[238,61,35],[236,38,38],[204,34,39],[153,27,56]]



salinas_color=np.uint8(salinas_color)
'''
''' indian_pines'''



'''
salinas_color=[[0,0,0],[255,255,102],[0,48,205],[255,102,0],[0,255,154],
 [255,48,205],[102,0,255],[0,154,255],[0,255,0],
 [129,129,0],[129,0,129],[47,205,205],[0,102,102],
 [47,205,48],[102,48,0],[102,255,255],[255,255,0]]

'''
'''
salinas_color=[[40,42,117],[45,51,130],[45,72,155],[71,88,166],[71,88,166],[77,199,244],
               [129,191,231],[139,208,181],[137,202,135],[166,206,56],[245,235,4],
               [254,204,19],[247,154,31],[238,61,35],[236,38,38],[204,34,39],[153,27,56]]
'''

''' paviau  '''

'''
salinas_color=[[0,0,0],[179,179,180],[0,255,0],[102,255,255],
               [0,129,0],[255,48,205],[154,102,48],
               [154,0,154],[255,0,0],[255,255,0]]
'''

salinas_color=[[0,0,0],[252,226,223],[81,184,70],[136,204,234],
               [38,139,67],[159,93,166],[161,82,44],[128,70,156],
               [238,33,35],[246,236,20]]
'''

salinas_color=[[0,0,0],[243,16,6],[255,255,255],[168,50,97],
               [254,255,0],[245,128,79],[87,254,0],[68,204,0],
               [43,138,0],[147,254,211],[155,37,242],[214,191,217],
               [30,10,255],[11,3,140],[211,113,216],[154,83,44],
               [92,254,255],[247,165,0],[145,254,0],[138,139,0],
               [46,139,139],[203,181,206],[230,154,0]]
'''

root = os.getcwd()
#label_path_indian = root+'//denoiseHSI//Salinas_gt.mat'
#label_path_indian = root+'//denoiseHSI//Indian_pines_gt.mat'

#label_path_indian = root+'//denoiseHSI//WHU_Hi_HongHu_gt.mat'

label_path_indian = root+'//denoiseHSI//Salinas_gt.mat'
    #data_path_pavia = 'dataset/Pavia_corrected.mat'
    #label_path_pavia = 'dataset/Pavia_gt.mat'
    #data_path_salinas = 'dataset/Salinas_corrected.mat'
    #label_path_salinas = 'dataset/Salinas_gt.mat'
label_path = label_path_indian
  
y_label=sio.loadmat(label_path_indian)

#salinas_gt = y_label['indian_pines_gt']
salinas_gt = y_label['salinas_gt']
#salinas_gt = y_label['salinas_gt']
#salinas_gt = y_label['WHU_Hi_HongHu_gt']

''' salinas  '''
'''
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
                list_result[i][j][0]=salinas_color[d[m]][0]    
                list_result[i][j][1]=salinas_color[d[m]][1]
                list_result[i][j][2]=salinas_color[d[m]][2]
                m=m+1

xxxx = list_result.astype(np.uint8)
'''

'''  indian '''

'''
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
'''
'''
'''
'''paviau'''

d=[[0] for j in range(207400)]

for i in range(164624):
    d[i]=0
for i in range(0,42776):
    d[i+164624]=wh_x[i]+1

list_result = np.empty((610,340,3))

m=0

for l in range(10):
    for i in range(610):
        for j in range(340):
            if salinas_gt[i][j] == l:
                list_result[i][j][0]=salinas_color[d[m]][0]    
                list_result[i][j][1]=salinas_color[d[m]][1]
                list_result[i][j][2]=salinas_color[d[m]][2]
                m=m+1

xxxx = list_result.astype(np.uint8)

#dataNew= root+'//paviau.mat'
#import scipy.io

#scipy.io.savemat(dataNew, mdict={'my_data': xxxx})

'''
d=[[0] for j in range(446500)]

for i in range(59807):
    d[i]=0
for i in range(0,386693):
    d[i+59807]=wh_x[i]+1

list_result = np.empty((940,475,3))

m=0

for l in range(23):
    for i in range(940):
        for j in range(475):
            if salinas_gt[i][j] == l:
                list_result[i][j][0]=salinas_color[d[m]][0]    
                list_result[i][j][1]=salinas_color[d[m]][1]
                list_result[i][j][2]=salinas_color[d[m]][2]
                m=m+1

xxxx = list_result.astype(np.uint8)


'''

# 再利用numpy将列表包装为数组

# array1 = np.array(list_result)

# # 进一步将array包装成矩阵

# data = array1

# # 重新reshape一个矩阵为一个方阵

#data = np.reshape(list_result,"RGB")

# 调用Image的formarray方法将矩阵数据转换为图像PIL类型的数据

new_map = Image.fromarray(xxxx,"RGB")

# 显示图像

#new_map.show()
'''
#filename=root+'//solution.jpg' 

#new_map.save(filename)
'''

print(model)

'''
#定义两个数组
Loss_list = []
Accuracy_list = []

Loss_list.append(train_loss / (len(train_dataset)))
Accuracy_list.append(100 * train_acc / (len(train_dataset)))



#我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
x1 = range(0, 200)
x2 = range(0, 200)
y1 = Accuracy_list
y2 = Loss_list
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.show()
plt.savefig("accuracy_loss.jpg")

'''



data1=[]

for i in range(10000):
        data1.append(data[16384+i])
        



import csv

filename= root + 'lossdata.csv'
filename1= root + 'lossdata1.csv'
def storFile(data,fileName):
    with open(fileName,'w',newline ='') as f:
        mywrite = csv.writer(f)
        mywrite.writerow(data)
#data = [1,0,1,1,0,1,0]
storFile(data,filename)
storFile(data1,filename1)

#storFile(data1,filename1)