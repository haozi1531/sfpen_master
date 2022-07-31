# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 20:51:05 2021

@author: Whao
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
import new_Model as Model
import random
from torch.utils.data import Dataset, DataLoader



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
        self.x_data=self.x_data.unsqueeze(1)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        # 返回文件数据的数目
        return self.len


# def layerlist_code(multi_layerlist):
#     core_list=multi_layerlist
#     num_list=multi_layerlist
    
#     for i in range(len(multi_layerlist)):
#         if multi_layerlist[i]<20:
#             multi_layerlist[i]=20
#         if multi_layerlist[i]>260:
#             multi_layerlist[i]=260

#     for i in range(len(multi_layerlist)):

#         if multi_layerlist[i]>=20 and multi_layerlist[i]<100:
#             num_list[i]=multi_layerlist[i]
#             core_list[i]=3
#         if multi_layerlist[i]>=100 and multi_layerlist[i]<180:
#             num_list[i]=multi_layerlist[i]
#             core_list[i]=5
#         if multi_layerlist[i]>=180 and multi_layerlist[i]<=260:
#             num_list[i]=multi_layerlist[i]
#             core_list[i]=7

#     return num_list,core_list 


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
    # x1 = len(cla_Labels)
    # x2 = len(all_Auglabels)
    
    #aug_data=int((sum(cla_flag)/len(cla_flag))*0.1)
    
    aug_data=200
    
    
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

class Evocnn:
        def __init__(self,
                 layerllist=None,
                 auged_data=None,
                 auged_label=None,
                 batchSize=32,
                 trainEpochs=4):
            self.layerllist=layerllist
            self.auged_data=auged_data
            self.auged_label=auged_label
            self.trainEpochs = trainEpochs
            self.batchSize = batchSize
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#           self.device = torch.device("cpu")
        def Evocnn_Fitness(self):
         
            train_batch_size = self.batchSize
            #test_batch_size = self.batchSize
            Epoch = self.trainEpochs
            layerllist = self.layerllist
            device = self.device
            auged_data=self.auged_data
            auged_label=self.auged_label
            
            print("evocnn")
            print(layerllist)
            #min_loss=10
            loss_list_x=[]
            
            
            
            Xtrain = auged_data
            ytrain  = auged_label
                      
        
            trainset = TrainDS(Xtrain, ytrain)
            #testset = TestDS(Xtrain, ytrain)
            train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size= train_batch_size, shuffle=True,num_workers=1)
#            test_loader  = torch.utils.data.DataLoader(dataset=testset,  batch_size=100, shuffle=False)

            #train_loader = tud.DataLoader(dataset = train_data, batch_size = train_batch_size, shuffle = True)
            #test_loader = tud.DataLoader(dataset = test_dataset, batch_size = test_batch_size, shuffle = True)

            #layerList=[16,32]
            #model = Model.AlexNet().to(device)

            #layerList=[16,32]
            #model = Model.AlexNet().to(device)
            
            #num_list,core_list=layerlist_code(layerllist)

            
            cor_list=[0]*len(layerllist)
            num_list=[0]*len(layerllist)
            
            multi_layerlist=layerllist
            
            for i in range(len(multi_layerlist)):
                if multi_layerlist[i]<20:
                    multi_layerlist[i]=20
                if multi_layerlist[i]>260:
                    multi_layerlist[i]=260

            for i in range(len(multi_layerlist)):

                if multi_layerlist[i]>=20 and multi_layerlist[i]<100:
                    num_list[i]=multi_layerlist[i]
                    cor_list[i]=3
                if multi_layerlist[i]>=100 and multi_layerlist[i]<180:
                    num_list[i]=multi_layerlist[i]-80
                    cor_list[i]=5
                if multi_layerlist[i]>=180 and multi_layerlist[i]<=260:
                    num_list[i]=multi_layerlist[i]-160
                    cor_list[i]=7


            print("evocnn1111")
            print(layerllist)

            
            model = Model.Net(num_list,cor_list).to(device)
            print(model)
            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #
            #model = Model.AlexNet().to(device)
            
            criterion = nn.CrossEntropyLoss().to(device)
            #optimer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
            optimer = torch.optim.Adam(model.parameters(), lr=0.00001)

            #min_loss=10
#           Epoch = self.trainEpochs
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
                    
                    loss_list_x.append(loss_x)
                    #if min_loss>loss_x:
                    #    min_loss=loss_x
                    #        optimer.zero_grad()
                    #        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
                    trained_data = len(image) * (indx + 1)
                    #if (indx + 1) % 10 == 0:
                    #    print('{}/{}               第{}轮第{}批的loss = {}'.format(trained_data, len(train_loader.dataset), epcho,
                    #                                                  indx,loss))
                    loss_len=len(loss_list_x)
                    
                min_loss=(loss_list_x[loss_len-1]+loss_list_x[loss_len-2]+loss_list_x[loss_len-3]+loss_list_x[loss_len-4]+loss_list_x[loss_len-5])/5
                min_loss1=(loss_list_x[loss_len-6]+loss_list_x[loss_len-7]+loss_list_x[loss_len-8]+loss_list_x[loss_len-9]+loss_list_x[loss_len-10])/5
                min_loss=2*min_loss-min_loss1
            print("evocnn")
            print(layerllist)
            
            return min_loss


        def Evocnn_acc_Fitness(self):
         
            train_batch_size = self.batchSize
            #test_batch_size = self.batchSize
            Epoch = 6
            layerllist = self.layerllist
            device = self.device
            #auged_data=self.auged_data
            #auged_label=self.auged_label
            loss_list_x=[]

           # Xtrain = auged_data
           # ytrain  = auged_label
            
            data_path_indian = root+'//HSI//Indian_pines.mat'
            label_path_indian = root+'//HSI//Indian_pines_gt.mat'
            data_path = data_path_indian
            #label_path = label_path_indian
            X = sio.loadmat(data_path)
    
            X=X['indian_pines']
    
            shapeor = X.shape
            data = X.reshape(-1, X.shape[-1])
            data = StandardScaler().fit_transform(data)
            X = data.reshape(shapeor)
    
    
            y_label=sio.loadmat(label_path_indian)
            y_label = y_label['indian_pines_gt']
            patch_size=3
            
            cla_Data,cla_Labels,cla_flag = createImageCubes(X, y_label,windowSize=patch_size)
    
            aaa_data,aaa_label=data_augmentation(cla_Data,cla_Labels,cla_flag)
    
            XX,YY=AugData_split(aaa_data,aaa_label,cla_Data, cla_Labels, cla_flag)
    
#            auged_data=self.auged_data#训练数据集
#            auged_label=self.auged_label        
#        
#            all_data_list = np.array(all_data,dtype=float)
#            all_label_list = np.array(all_label)
#        
#            X=all_data_list[0]
#            Y=all_label_list[0]
#        
#            hsiclass_num=16
#            hsiband_num=220
        
            
        
            trainset = TrainDS(XX, YY)
            #testset = TestDS(Xtrain, ytrain)
            train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size= train_batch_size, shuffle=True,num_workers=1)

            cor_list=[0]*len(layerllist)
            num_list=[0]*len(layerllist)
            
            multi_layerlist=layerllist
            
            for i in range(len(multi_layerlist)):
                if multi_layerlist[i]<20:
                    multi_layerlist[i]=20
                if multi_layerlist[i]>260:
                    multi_layerlist[i]=260

            for i in range(len(multi_layerlist)):

                if multi_layerlist[i]>=20 and multi_layerlist[i]<100:
                    num_list[i]=multi_layerlist[i]
                    cor_list[i]=3
                if multi_layerlist[i]>=100 and multi_layerlist[i]<180:
                    num_list[i]=multi_layerlist[i]-80
                    cor_list[i]=5
                if multi_layerlist[i]>=180 and multi_layerlist[i]<=260:
                    num_list[i]=multi_layerlist[i]-160
                    cor_list[i]=7
            
            print("acc")
            print(layerllist)
            
            
            model = Model.Net(num_list,cor_list).to(device)
            print(model)
            
            criterion = nn.CrossEntropyLoss().to(device)
            #optimer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
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
                    
                    loss_list_x.append(loss_x)

                    trained_data = len(image) * (indx + 1)
            #        if (indx + 1) % 10 == 0:
            #            print('{}/{}               第{}轮第{}批的loss = {}'.format(trained_data, len(train_loader.dataset), epcho,
            #                                                          indx,loss))

            label_x=[[0]*2 for row in range(len(YY))]
        
            test_data = TestDS(XX,YY) #测试数据集，即所有的待分类数据，对这些数据进行标签预测
        
            ttt=0
            
            for k in range(len(YY)):

                test_data.__getitem__(k)
    
                test_loader  = torch.utils.data.DataLoader(dataset=test_data,  batch_size=1, shuffle=False)

                image=test_loader.dataset.__getitem__(k)[0]
    
                #image = image.unsqueeze(0)
                image = image.unsqueeze(0)
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
            #    print('{}  {}'.format(xxx[0],test_loader.dataset.__getitem__(k)[1]))

            aacc=ttt/10249
            
            return aacc



class Pop_evo:
    def __init__(self,
              auged_data=None,
              auged_label=None,
              #popList=None,
              #vote_data=None,#加入伪标签的训练数据
              #vote_label=None,#加入伪标签的训练标签
              all_data=None,#所有无标签数据
              all_label=None,#所有无标签数据的真实标签   
              pop_num=50,
              convnum_max=260,
              convnum_min=20,
              elistsmFrac=0.3,
              pc=0.3,#交叉概率
              pm=0.3,#变异概率
              popSize=50,
              batchSize=32,
              layer_num=10):
        self.auged_data=auged_data
        self.auged_label=auged_label
        
        #self.vote_data=vote_data
        #self.vote_label=vote_label
        
        self.all_data=all_data,#所有无标签数据
        self.all_label=all_label,#所有无标签数据的真实标签 
        
        
        self.pop_num=pop_num
        self.convnum_max=convnum_max
        self.convnum_min=convnum_min
        self.elistsmFrac=elistsmFrac
        self.pc=pc
        self.pm=pm
        self.layer_num=layer_num
        self.popSize=popSize
        self.frac=elistsmFrac
        self.batchSize=batchSize
#        self.trainErrors=None
        self.best = None,
        self.trainError = None, #训练误差
        self.offspringList = None,
        self.popList = None
    def pop_init(self):
#       pop_num=self.pop_num
        convnum_max=self.convnum_max
        convnum_min=self.convnum_min
        layer_num=self.layer_num
        popList = []      #种群列表
        for i in range(self.popSize):
            currentLayernum = random.randint(1, layer_num) #当前网络层大小layerList，当前网络层的数量最大为10
            currentLayerconvnum = []#当前卷积层卷积核的个数
            for j in range(currentLayernum): #为每一层网络初始化神经元个数
                neuron = random.randint(convnum_min, convnum_max)
                currentLayerconvnum.append(neuron)
            popList.append(currentLayerconvnum)
        print(popList)
        self.popList=popList            
        self.myFitness(popList) #训练误差，当前种群对应的网络训练误差，myFitness可以批量计算多个个体的适应度
        #self.trainErrors=t_errors
        #self.popList=popList

        return self.trainError,popList
        #调用适应度函数，计算该网络对应的训练误差
        # self.popList = popList
        # self.trainErrors = t_errors
        # self.calculateThreshold(t_errors)  #调用calculateThreshold

    def myFitness(self,pop):
        """计算并返回每一个个体经过相同条件的训练之后的在训练集的拟合误差和在测试集的拟合误差"""
        auged_data1=self.auged_data
        auged_label1=self.auged_label
        #pop=self.popList
        print("__________________")
        print(pop)
        t_errors = [] 
        for i in range(len(pop)):
            my_cnn=Evocnn(layerllist=pop[i],auged_data=auged_data1,auged_label=auged_label1)
            fitness_cnn=my_cnn.Evocnn_Fitness()
            t_errors.append(fitness_cnn)      #训练误差列表，记录整个种群的训练loss，loss为最小的batch——loss    

        print("afterevocnn")
        print(pop)

#            NetStruct = list(pop[i])
#            NetStruct.insert(0, self.inputSize)
#            NetStruct.append(self.outSize)
#            net = Net(NetStruct)
#            net.to(self.device)
#            t_error = self.evaluate(net)
#            t_errors.append(t_error)
#        x=np.sort(t_errors)
        
        aacc_sum=0
        xx=np.argsort(t_errors)
        lll=[]
        for i in range(30):
            lll.append(i)
        random.shuffle(lll)
        #lll=[48,47,45,43,42,40,39]
        for i in range(6):
            my_cnn=Evocnn(layerllist=pop[xx[lll[i]]],auged_data=auged_data1,auged_label=auged_label1)
            acc_fitness_cnn=my_cnn.Evocnn_acc_Fitness()
            t_errors[xx[lll[i]]] = t_errors[xx[lll[i]]] - acc_fitness_cnn
            aacc_sum+=acc_fitness_cnn
        
        aacc_ave=aacc_sum/6

        for i in range(24):
            t_errors[xx[lll[i+6]]] = t_errors[xx[lll[i]]] - aacc_ave
            
        for i in range(20):
            t_errors[xx[i+30]] = t_errors[xx[lll[i]]] - aacc_ave
            
#        order_errors=[]
#        order_pop=[]
#        for i in range(len(t_errors)):
#            b = t_errors[xx[i]]
#            c = pop[xx[x]]
#            order_errors.append(b)
#            order_pop.append(c)
#            
#        self.popList=order_pop
        self.trainError=t_errors
        return t_errors
    
    def binaryTournamentSelect(self, pop, t_errors):
        """二元锦标赛选择更优个体"""
        idx = random.sample(list(range(len(pop))), 2)
        individuals = []
        te = []
        for i in idx:
            individuals.append(pop[i])
            te.append(t_errors[i])
        
        
        
        cor_list0=[0]*len(individuals[0])
        num_list0=[0]*len(individuals[0])
            
        multi_layerlist=individuals[0]
            
        for i in range(len(multi_layerlist)):
            if multi_layerlist[i]<20:
                multi_layerlist[i]=20
            if multi_layerlist[i]>260:
                multi_layerlist[i]=260

        for i in range(len(multi_layerlist)):

            if multi_layerlist[i]>=20 and multi_layerlist[i]<100:
                num_list0[i]=multi_layerlist[i]
                cor_list0[i]=3
            if multi_layerlist[i]>=100 and multi_layerlist[i]<180:
                num_list0[i]=multi_layerlist[i]-80
                cor_list0[i]=5
            if multi_layerlist[i]>=180 and multi_layerlist[i]<=260:
                num_list0[i]=multi_layerlist[i]-160
                cor_list0[i]=7        
        
        
        
        
        cor_list1=[0]*len(individuals[1])
        num_list1=[0]*len(individuals[1])
            
        multi_layerlist=individuals[1]
            
        for i in range(len(multi_layerlist)):
            if multi_layerlist[i]<20:
                multi_layerlist[i]=20
            if multi_layerlist[i]>260:
                multi_layerlist[i]=260

        for i in range(len(multi_layerlist)):

            if multi_layerlist[i]>=20 and multi_layerlist[i]<100:
                num_list1[i]=multi_layerlist[i]
                cor_list1[i]=3
            if multi_layerlist[i]>=100 and multi_layerlist[i]<180:
                num_list1[i]=multi_layerlist[i]-80
                cor_list1[i]=5
            if multi_layerlist[i]>=180 and multi_layerlist[i]<=260:
                num_list1[i]=multi_layerlist[i]-160
                cor_list1[i]=7            
                
        lss0,s0=self.my_minloss(te[0],te[1],individuals[0],individuals[1])
        return s0,lss0
#        if abs(te[0]-te[1])>=0.1:#如果loss之前的差距大于0.01，则直接返回loss较小的net
#            lss0,s0=self.my_minloss(te[0],te[1],individuals[0],individuals[1])
#
#        else:
#            if len(individuals[0])>len(individuals[1]):
#                lss0=te[0]
#                s0=individuals[0]
#            if len(individuals[0])<len(individuals[1]):
#                lss0=te[1]
#                s0=individuals[1]
#            if len(individuals[0])==len(individuals[1]):
#                if sum(num_list0)<=sum(num_list1):
#                    lss0=te[0]
#                    s0=individuals[0]
#                else:
#                    lss0=te[1]
#                    s0=individuals[1]

    
    
    def my_minloss(self, lss_a, lss_b, idi_a,idi_b):
        if lss_a<=lss_b:
            lss_c = lss_a
            idi=idi_a
        else:
            lss_c = lss_b
            idi=idi_b
        return lss_c,idi      


    def my_min(self, a, b):
        c = 0
        if a<=b:
            c = a
        else:
            c = b
        return c       

    def mutate(self, g1):
        """
        变异操作
        1: Addition;
        2: Del;
        3: Change neuron num. 
        """
        g2 = []
        r1 = random.random()
        if r1 <= self.pm:
            L = len(g1)#变异个体g1
            r2 = np.random.rand(L)#随机生成1-L的一个整数，随机选择交叉位置
            p = []
            for i, prob in enumerate(r2, 0):
                if prob>=0.5:
                    p.append(i)
#            a = self.maxLayerSize - L
            a=self.layer_num-L
            lp = len(p)
            A = self.my_min(a, lp) # the number of addition，还能增加的层数
            B = self.my_min(lp, L-1) # the number of deletion，还能被删除的层数
            C = lp # the number of change neuron number，改变神经网络神经元的总层数
            pool = [1]*A + [2]*B + [3]*C
#            [A,B,B,C,C,C]
            random.shuffle(pool) #随机排列pool
            for i, h in enumerate(g1, 0):
                if i in p:
                    j = random.sample(pool, 1)
                    j = j[0]
                    if j==1:
                        n = random.randint(1, self.convnum_max)
                        g2.append(h)
                        g2.append(n)
                    elif j==3:
                        n = random.randint(1, self.convnum_max)
                        g2.append(n)
                    p.remove(i)
                    dd = set(pool)
                    for d in dd:
                        pool.remove(d)
                else:
                    g2.append(h)
        else:
            g2 = g1
        return g2

    def crossover(self, parent1, parent2):
        o1 = []
        o2 = []
        r = random.random()
        if r<= self.pc:
            L1 = len(parent1) #L1---长度更长的个体对应的长度 
            L2 = len(parent2)
            if L1<L2:
                L1 = len(parent2)
                L2 = len(parent1)
                p1 = parent2
                p2 = parent1
            else:
                p1 = parent1 #卷积层更多的个体
                p2 = parent2
            d1 = L1 - L2
            d2 = random.randint(0, d1)#生成一个[0,d1]的整数
            o1 += p1[:d2] #扩展d2长度的p1卷积层数
            a = np.random.rand(L2) #   
            for idx, prob in enumerate(a):
                if prob >= 0.5:
                    o1.append(p2[idx])
                    o2.append(p1[d2+idx])
                else:
                    o1.append(p1[d2+idx])
                    o2.append(p2[idx])
            if d2+L2<L1:
                o1 += p1[d2+L2:]
        else:
            o1 += parent1
            o2 += parent2
        return o1, o2






    def offspringGenerate(self):
        """生成并返回子代个体"""
        popList=self.popList
        
        print("offspringpoplist")
        print(popList)
        matingPool = []
        offspringList = []
#        self.calculateThreshold(self.trainErrors)
        for i in range(self.pop_num):
            parent, parent_loss = self.binaryTournamentSelect(popList, self.trainError)
#          二元锦标赛选择优秀个体
            matingPool.append(list(parent))
        while(len(matingPool) != 0):
            poolSize = len(matingPool)
            idx = random.sample(list(range(poolSize)), 2)
            p1 = matingPool[idx[0]]
            p2 = matingPool[idx[1]]
            # Crossover
            offspring1, offspring2 = self.crossover(p1, p2)
            # mutation
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)
            offspringList.append(offspring1)
            offspringList.append(offspring2)
            if idx[0] > idx[1]:
                matingPool.pop(idx[0])#弹出对应的个体
                matingPool.pop(idx[1])
            else:
                matingPool.pop(idx[1])
                matingPool.pop(idx[0])
        self.offspringList=offspringList
        print("---offspringlist----cross--- mute---")
        print(offspringList)
        return offspringList

    def new_offspringGenerate(self):
        """生成并返回子代个体"""
        popList=self.offspringList
        matingPool = []
        offspringList = []
#        self.calculateThreshold(self.trainErrors)
        for i in range(self.pop_num):
            parent, _ = self.binaryTournamentSelect(popList, self.trainErrors)
#          二元锦标赛选择优秀个体
            matingPool.append(list(parent))
        while(len(matingPool) != 0):
            poolSize = len(matingPool)
            idx = random.sample(list(range(poolSize)), 2)
            p1 = matingPool[idx[0]]
            p2 = matingPool[idx[1]]
            # Crossover
            offspring1, offspring2 = self.crossover(p1, p2)
            # mutation
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)
            offspringList.append(offspring1)
            offspringList.append(offspring2)
            if idx[0] > idx[1]:
                matingPool.pop(idx[0])#弹出对应的个体
                matingPool.pop(idx[1])
            else:
                matingPool.pop(idx[1])
                matingPool.pop(idx[0])
        self.offspringList=offspringList
        return offspringList

    def envSelect(self):
        offspringList=self.offspringList
        """在父代和子代组成的大种群中筛选出规定数量的优秀个体"""
        #popSize=self.popSize
        newPopList = []
        newTrainErrors = []
#        newValErrors = []
#        a = math.floor((self.popSize + len(offspringList)) * self.frac)
        a = self.popSize
        offspring_trainErrors = self.myFitness(offspringList)
        trainErrors = self.trainError
        popList = self.popList
        popList = popList + offspringList
        print("--------906------")

        
        trainErrors = trainErrors + offspring_trainErrors
        while(len(newPopList)<a):
            best_idx = trainErrors.index(min(trainErrors))
            if len(newPopList)==0:
                self.best = popList[best_idx]
            newPopList.append(popList[best_idx])
            newTrainErrors.append(trainErrors[best_idx])
            popList.pop(best_idx)
            trainErrors.pop(best_idx)

        while(len(newPopList)<self.popSize):
            ns, ns_tError = self.binaryTournamentSelect(newPopList, newTrainErrors)
            newPopList.append(ns)
            newTrainErrors.append(ns_tError)
        self.popList = newPopList
        self.trainError = newTrainErrors
        
        print(self.popList)
        print(self.trainError)
        
        
        return newPopList[0],newTrainErrors[0]

if __name__=="__main__":
    #读取并创建数据集
    # 定义各类参数
    print("indian")
    root = os.getcwd()
    
    #data_path_indian = root+'//HSI//PaviaU.mat'
    #label_path_indian = root+'//HSI//PaviaU_gt.mat'
    #data_path_pavia = 'dataset/Pavia_corrected.mat'
    #label_path_pavia = 'dataset/Pavia_gt.mat'
    data_path_indian = root+'//HSI//Indian_pines'
    label_path_indian = root+'//HSI//Indian_pines_gt.mat'
    
    data_path = data_path_indian
    label_path = label_path_indian
    
    
    X = sio.loadmat(data_path)
    
    X=X['indian_pines']
    
    shapeor = X.shape
    data = X.reshape(-1, X.shape[-1])
    data = StandardScaler().fit_transform(data)
    X = data.reshape(shapeor)
    
    
    y_label=sio.loadmat(label_path_indian)
    y_label = y_label['indian_pines_gt']
    patch_size=3
    
    #patchesData, patchesLabels = createImageCubes(X, y_label, windowSize=patch_size)
    
    cla_Data,cla_Labels,cla_flag = createImageCubes(X, y_label,windowSize=patch_size)
    
    aaa_data,aaa_label=data_augmentation(cla_Data,cla_Labels,cla_flag)
    
    auged_data,auged_labels=AugData_split(aaa_data,aaa_label,cla_Data, cla_Labels, cla_flag)
    

    
    '''
    
              auged_data=None,
              auged_label=None,
              pse_auged_data=None,#加入伪标签的训练数据
              pse_auged_label=None,#加入伪标签的训练标签
              all_data=None,#所有无标签数据
              all_label=None,#所有无标签数据的真实标签   

    '''
        
    all_data = cla_Data
    
    all_label = cla_Labels
    
    my_pop=[]
    
    pop_acc=[]
    #x=Pop_evo(auged_data,auged_labels)
    
    x=Pop_evo(auged_data=auged_data,auged_label=auged_labels,all_data=cla_Data,all_label=cla_Labels)
    #x=Pop_evo(auged_data=auged_data1,auged_label=auged_labels1)
    y_errors,y_pop=x.pop_init()
    print("fanhuizhi---------------")
    print(y_pop)
    #_ = x.offspringGenerate()
#    x.envSelect()    
    for i in range(50):
        _ = x.offspringGenerate()
        my_pop1,pop_acc1=x.envSelect()
#        print(cla_Data)
#        print(cla_Labels)
        #x.elit_netvote(my_iter=i)
        my_pop.append(my_pop1)
        pop_acc.append(pop_acc1)
        #pse_auged_data,pse_auged_label = x.elit_netvote(i)
        # x=Pop_evo(auged_data=auged_data1,auged_label=auged_labels1,pse_auged_data=pse_auged_data,
        #       pse_auged_label=pse_auged_label,all_data=all_data,all_label=all_label)
        #x=Pop_evo(auged_data=auged_data1,auged_label=auged_labels1)
        data_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
        data_root += '/hsi_train'      
        
        filename = data_root+'/y_pop.txt' 
        output = open(filename,'w+')
        for i in range(len(my_pop)):
            for j in range(len(my_pop[i])):
                output.write(str(my_pop[i][j]))
                output.write(' ')   
            output.write('\n')      
        output.close()    

        filename = data_root+'/y_acc.txt' 

        np.savetxt(filename,pop_acc,fmt='%f',delimiter=',') 
