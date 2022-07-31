# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:02:24 2021

@author: 17783
"""
import tensorflow as tf
from tensorflow.keras import optimizers,layers
import os
import numpy as np
import scipy.io as sio
#from sklearn import processing
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
#from operator import truediv
#import torch.nn as nn
import random



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
    all_AugData= np.zeros((DataAug_num*len(cla_flag), 3, 3, 220))
    all_Auglabels = np.zeros((DataAug_num*len(cla_flag)))

    lll=0
       
    for i in range(len(cla_flag)):
        #cla_Data[label_num, :, :, :] = patchesData[k, :, :, :]
        #cla_Labels[label_num] = patchesLabels[k]
        #label_num+=1
        '''     根据不同的数据集进行修改           '''
        Aug_patchesData = np.zeros((DataAug_num, 3, 3, 220))
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
    
    aug_data=int((sum(cla_flag)/len(cla_flag))*0.3)
    #aug_data=200
    split_AugData = np.zeros((aug_data*len(cla_flag), 3, 3, 220))
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
            
            # a=[]
            # for xx in range(cla_flag[i]):
            #     a.append(xx)
            # random.shuffle(a)
            
            samp_Data1 = all_AugData[all_Auglabels==i,:,:,:]
            samp_Labels1 = all_Auglabels[all_Auglabels==i]
            
            samp_Data = cla_Data[cla_Labels==i,:,:,:]
            samp_Labels = cla_Labels[cla_Labels==i]
            for k in range(len(samp_Labels)):
                split_AugData[mmm,:,:,:] = samp_Data[k,:,:,:]
                split_Auglabels[mmm] = samp_Labels[k]
                mmm+=1

            a=[]
            for xx in range(len(samp_Labels1)):
                a.append(xx)
            random.shuffle(a)                
            
            for l in range(aug_data-cla_flag[i]):
                
                split_AugData[mmm,:,:,:] = samp_Data1[a[l],:,:,:]
                split_Auglabels[mmm] = samp_Labels1[a[l]]
                mmm+=1
    #np.save(r"C://Users//17783//Desktop//data.npy", a_x)
    return split_AugData,split_Auglabels



root = os.getcwd()
    
data_path_indian = root+'//HSI//Salinas.mat'

#data_path_indian = 'D://Hyperspectral_image//HSI//denoise_salinas.mat'
label_path_indian = root+'//HSI//Salinas_gt.mat'
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



#X= = preprocessing.normalize(X, norm='l2')


patch_size=3
    
    #patchesData, patchesLabels = createImageCubes(X, y_label, windowSize=patch_size)
    
cla_Data,cla_Labels,cla_flag = createImageCubes(X, y_label,windowSize=patch_size)

aaa_data,aaa_label=data_augmentation(cla_Data,cla_Labels,cla_flag)

auged_data,auged_labels=AugData_split(aaa_data,aaa_label,cla_Data,cla_Labels,cla_flag)




Xtrain = auged_data

ytrain  = auged_labels


#Xtrain = Xtrain.transpose(0, 3, 1, 2)

#Xtest  = Xtest.transpose(0, 3, 1, 2)
print('after transpose: Xtrain shape: ', Xtrain.shape)

'''
def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32) / 255                   #   将特征数据转化为float32类型，并缩放到0到1之间
    y = tf.cast(y,dtype=tf.int32)                           #   将标记数据转化为int32类型
    y = tf.one_hot(y,depth= 10)                             #   将标记数据转为one_hot编码
    return x,y
'''

def get_data(Xtrain,ytrain):
    # 加载手写数字数据
    #mnist = tf.keras.datasets.mnist
    #(train_x, train_y), (test_x, test_y) = mnist.load_data()
    #   开始预处理数据
        #   训练数据
    db = tf.data.Dataset.from_tensor_slices((Xtrain,ytrain))          #   将数据特征与标记组合
    #db = db.map(preprocess)                                             #   根据预处理函数对组合数据进行处理
    db = db.shuffle(3200).batch(32)                                   #   将数据按10000行为单位打乱，并以100行为一个整体进行随机梯度下降
        #   测试数据
    db_test = tf.data.Dataset.from_tensor_slices((Xtrain,ytrain))
    #db_test = db_test.map(preprocess)
    db_test = db_test.shuffle(3200).batch(32)
    return db,db_test




db,db_test = get_data(Xtrain,ytrain)             #   获取训练和测试数据



iter = 10
learn_rate = 0.01
#   定义模型和优化器
model = tf.keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),           #   全连接
    layers.Dense(10)
])




model = tf.keras.Sequential([
  tf.keras.layers.Conv3D(64, (3,3,3), activation='relu', input_shape=(3, 3, 224)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv3D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])











optimizer = optimizers.SGD(learning_rate=learn_rate)            #   优化器

#   迭代代码
for i in range(iter):
    for step,(x,y) in enumerate(db):                            #   对每个batch样本做梯度计算
        # print('x.shape:{},y.shape:{}'.format(x.shape,y.shape))
        with tf.GradientTape() as tape:
            x = tf.reshape(x,(-1,28*28))               #   将28*28展开为784
            out = model(x)
            loss = tf.reduce_mean(tf.square(out-y))
        grads = tape.gradient(loss,model.trainable_variables)               #   求梯度
        grads,_ = tf.clip_by_global_norm(grads,15)                          #   梯度参数进行限幅，防止偏导的nan和无穷大
        optimizer.apply_gradients(zip(grads,model.trainable_variables))     #   优化器进行参数优化
        if step % 100 == 0:
            print('i:{} ,step:{} ,loss:{}'.format(i, step,loss.numpy()))
            #   求准确率
            acc = tf.equal(tf.argmax(out,axis=1),tf.argmax(y,axis=1))
            acc = tf.cast(acc,tf.int8)
            acc = tf.reduce_mean(tf.cast(acc,tf.float32))
            print('acc:',acc.numpy())











