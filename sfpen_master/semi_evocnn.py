# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:20:32 2020

@author: Whao
"""

import torch
import torch.nn as nn
#import Data_processing as Data
import Model
import torch.optim
import numpy as np
#import torch.optim as optim 
#import torchvision
#import torchvision.datasets as datasets
#import torchvision.transforms as tfs
import torch.utils.data as tud
import os
#import numpy as np 
import random 
#import math
from PIL import Image
#import matplotlib.pyplot as plt 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import shutil
import time

def default_loader(path):
    return Image.open(path).convert('L')

class MyDataset(Dataset): #创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
	def __init__(self,txt, transform=None,target_transform=None, loader=default_loader): #初始化一些需要传入的参数
		super(MyDataset,self).__init__()#对继承自父类的属性进行初始化
		fh = open(txt, 'r')#按照传入的路径和txt文本参数，打开这个文本，并读取内容
		imgs = []
		for line in fh: #迭代该列表#按行循环txt文本中的内
			line = line.strip('\n')
			line = line.rstrip('\n')# 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
			words = line.split() #用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
			imgs.append((words[0],int(words[1]))) #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定 
                                                 # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable       
		self.imgs = imgs
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader        
        
	def __getitem__(self, index):#这个方法是必须要有的，用于按照索引读取每个元素的具体内容
		fn, label = self.imgs[index] #fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
		img = self.loader(fn) # 按照路径读取图片
		if self.transform is not None:
			img = self.transform(img)#数据标签转换为Tensor
		return img,torch.tensor(label)#return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
	def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
		return len(self.imgs)


class Evocnn:
        def __init__(self,
                 layerList=None,
                 batchSize=32,
                 trainEpochs=1):
            self.layerList=layerList
            self.trainEpochs = trainEpochs
            self.batchSize = batchSize
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#           self.device = torch.device("cpu")
        def Evocnn_Fitness(self):
            train_batch_size = self.batchSize
            #test_batch_size = self.batchSize
            Epoch = self.trainEpochs
            layerList = self.layerList
            device = self.device
            min_loss=10
            #数据预处理
#            data_tfs = {
#                'train': tfs.Compose([tfs.Grayscale(1),
#                                          tfs.ToTensor(),
#                                          tfs.transforms.Normalize([0.5], [0.5])])
#                'test': tfs.Compose([tfs.Grayscale(1),
#                              #            tfs.RandomResizedCrop(227),
#                              tfs.ToTensor(),
#                              tfs.transforms.Normalize([0.5], [0.5])])
#                        }

            #读取数据集
#            data_root = os.getcwd()
            #data_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
            #data_root += '/hsi_train'

            #train_dataset = datasets.ImageFolder(root = data_root + '/train', transform = data_tfs['train'])
            #test_dataset = datasets.ImageFolder(root = data_root + '/val', transform = data_tfs['test'])
            #test_dataset = datasets.ImageFolder(root = data_root + '/train', transform = data_tfs['test'])
            root = os.getcwd()
            train_data=MyDataset(txt=root+'/creat_txt.txt',
            transform=transforms.Compose([
                transforms.Grayscale(1),
                #transforms.RandomResizedCrop(15),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
                                        ]))


            train_loader = tud.DataLoader(dataset = train_data, batch_size = train_batch_size, shuffle = True)
            #test_loader = tud.DataLoader(dataset = test_dataset, batch_size = test_batch_size, shuffle = True)

            #layerList=[16,32]
            #model = Model.AlexNet().to(device)
            model = Model.Net(layerList).to(device)
            print(model)
            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #
            #model = Model.AlexNet().to(device)
            
            criterion = nn.CrossEntropyLoss().to(device)
            #optimer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
            optimer = torch.optim.Adam(model.parameters(), lr=0.01)

            # min_loss=10
#           Epoch = self.trainEpochs
            for epcho in range(0,Epoch):
                trained_data = 0
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
                    
                    if min_loss>loss_x:
                        min_loss=loss_x
                    #        optimer.zero_grad()
                    #        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
                    trained_data = len(image) * (indx + 1)
                    if (indx + 1) % 10 == 0:
                        print('{}/{}               第{}轮第{}批的loss = {}'.format(trained_data, len(train_loader.dataset), epcho,
                                                                      indx,loss))
            return min_loss


class Pop_evo:
    def __init__(self,
              pop_num=16,
              convnum_max=60,
              convnum_min=10,
              elistsmFrac=0.3,
              pc=0.3,#交叉概率
              pm=0.3,#变异概率
              popSize=16,
              batchSize=32,
              layer_num=8):
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
        self.best = None
        self.trainErrors = None #训练误差
        self.offspringList = None
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
        t_errors = self.myFitness(popList) #训练误差，当前种群对应的网络训练误差，myFitness可以批量计算多个个体的适应度
        self.trainErrors=t_errors
        self.popList=popList
        return t_errors,popList
        #调用适应度函数，计算该网络对应的训练误差
        # self.popList = popList
        # self.trainErrors = t_errors
        # self.calculateThreshold(t_errors)  #调用calculateThreshold

    def myFitness(self, pop):
        """计算并返回每一个个体经过相同条件的训练之后的在训练集的拟合误差和在测试集的拟合误差"""
        t_errors = [] 
        for i in range(len(pop)):
            my_cnn=Evocnn(layerList=pop[i])
            fitness_cnn=my_cnn.Evocnn_Fitness()
            t_errors.append(fitness_cnn)      #训练误差列表，记录整个种群的训练loss，loss为最小的batch——loss    
#            NetStruct = list(pop[i])
#            NetStruct.insert(0, self.inputSize)
#            NetStruct.append(self.outSize)
#            net = Net(NetStruct)
#            net.to(self.device)
#            t_error = self.evaluate(net)
#            t_errors.append(t_error)
        self.trainErrors=t_errors
        return t_errors
    def binaryTournamentSelect(self, pop, t_errors):
        """二元锦标赛选择更优个体"""
        idx = random.sample(list(range(len(pop))), 2)
        individuals = []
        te = []
        for i in idx:
            individuals.append(pop[i])
            te.append(t_errors[i])
        if abs(te[0]-te[1])>=0.01:#如果loss之前的差距大于0.01，则直接返回loss较小的net
            lss0,s0=self.my_minloss(te[0],te[1],individuals[0],individuals[1])
#            acc0----训练准确率高的个体对应的准确率，s0--训练准确率较高对应的个体
#            s0=individuals[0]
#            t0=te[0]
        else:
            if len(pop[0])<len(pop[1]):
                lss0=te[0]
                s0=individuals[0]
            if len(pop[0])>len(pop[1]):
                lss0=te[1]
                s0=individuals[1]
            if len(pop[0])==len(pop[1]):
                if sum(pop[0])<=sum(pop[1]):
                    lss0=te[0]
                    s0=individuals[0]
                else:
                    lss0=te[1]
                    s0=individuals[1]              
        return s0,lss0

#    def my_max(self, acc_a, acc_b, idi_a,idi_b):
#        if acc_a>=acc_b:
#            acc_c = acc_a
#            idi=idi_a
#        else:
#            acc_c = acc_b
#            idi=idi_b
#        return acc_c,idi  
   
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
        #popList=self.popList
        matingPool = []
        offspringList = []
#        self.calculateThreshold(self.trainErrors)
        for i in range(self.pop_num):
            parent, _ = self.binaryTournamentSelect(self.popList, self.trainErrors)
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
        trainErrors = self.trainErrors
        popList = self.popList
        popList = popList + offspringList
        trainErrors = trainErrors + offspring_trainErrors
        while(len(newPopList)<a):
            best_idx = trainErrors.index(min(trainErrors))
            if len(newPopList)==0:
                self.best = popList[best_idx]
            newPopList.append(popList[best_idx])
            newTrainErrors.append(trainErrors[best_idx])
            popList.pop(best_idx)
            trainErrors.pop(best_idx)

#        self.calculateThreshold(trainErrors)
        while(len(newPopList)<self.popSize):
            ns, ns_tError = self.binaryTournamentSelect(newPopList, newTrainErrors)
            newPopList.append(ns)
            newTrainErrors.append(ns_tError)
        self.popList = newPopList
        self.trainErrors = newTrainErrors   
        return newPopList,trainErrors

    def elit_netvote(self,my_iter=None):
        
        ''' 
            creat_txt-------训练集的文件目录  
            creat_txt1--------所有不包含训练集的数据集
        '''
        train_batch_size = self.batchSize
        #test_batch_size = 1
        Epoch = 6
        #trainErrors=self.trainErrors
        popList=self.popList
        # my_iter=self.my_iter      
        ''' 删除上一代增加的伪标签 '''
#        if my_iter>0:
#            ''' 更改图片写入的路径 '''      
#            root_3 = os.getcwd()
#            '''-------------------------------------------------'''
#            txt_3=root_3 + '/accept_txt'+ str(my_iter) + '.txt'
#            fh = open(txt_3, 'r')#按照传入的路径和txt文本参数，打开这个文本，并读取内容
#            imgs = []
#            for line in fh: #迭代该列表#按行循环txt文本中的内
#                line = line.strip('\n')
#                line = line.rstrip('\n')# 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
#                words = line.split() #用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
#                #imgs.append((words[0],int(words[1]))) #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
#                if int(words[0])<=17:
#                    imgs.append((int(words[0]),words[1]))#把txt里的内容读入imgs
#                    
#            new_img=[]
#            for img in imgs:
#                split_img = img[1].split('/')
#                #for i in range(1,8):
#                new_img.append('/'+split_img[1]+'/'+split_img[2]+ \
#                               '/'+split_img[3]+'/'+split_img[4]+ \
#                               '/'+split_img[5]+'/'+str(img[0])+ \
#                               '/'+split_img[7])
##                my_remove_img='/'+split_img[1]+'/'+split_img[2]+ \
##                              '/'+split_img[3]+'/'+split_img[4]+ \
##                              '/'+split_img[5]+'/'+str(img[0])+
##                              '/'+split_img[7]
##                os.remove(my_remove_img)
#            for img in imgs:
#                os.remove(img)

        root = os.getcwd()
        txt=root+'/creat_txt1.txt'
        fh = open(txt, 'r')#按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
        labels = []
        for line in fh: #迭代该列表#按行循环txt文本中的内
            line = line.strip('\n')
            line = line.rstrip('\n')# 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split() #用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            imgs.append(words[0]) #图片路径 
        #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定 
        #很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable     
            labels.append(words[1])  #标签
        
        root_2 = os.getcwd()
        txt_2=root_2+'/creat_txt1.txt'
        fh_2 = open(txt_2, 'r')#按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs_2 = []
        labels_2 = []
        for line_2 in fh_2: #迭代该列表#按行循环txt文本中的内
            line_2 = line_2.strip('\n')
            line_2 = line_2.rstrip('\n')# 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words_2 = line_2.split() #用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            imgs_2.append(words_2[0]) #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定 
#           很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable     
            labels_2.append(words_2[1])  
            
            #数据预处理
#        data_tfs = {
#                'train': tfs.Compose([tfs.Grayscale(1),
#                                          #            tfs.RandomResizedCrop(227),
#                                          #                         tfs.RandomHorizontalFlip(p=0.5),
#                                          tfs.ToTensor(),
#                                          tfs.transforms.Normalize([0.5], [0.5])])
#                # 'test': tfs.Compose([tfs.Grayscale(1),
#                #               #            tfs.RandomResizedCrop(227),
#                #               tfs.ToTensor(),
#                #               tfs.transforms.Normalize([0.5], [0.5])])
#                        }

            #读取数据集
#        data_root = os.getcwd()
        #data_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
        #data_root += '/hsi_train'

        root = os.getcwd()
        train_data=MyDataset(txt=root+'/creat_txt.txt',
            transform=transforms.Compose([
                transforms.Grayscale(1),
                #transforms.RandomResizedCrop(15),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
                                        ]))

        #train_dataset = datasets.ImageFolder(root = data_root + '/train', transform = data_tfs['train'])
            #test_dataset = datasets.ImageFolder(root = data_root + '/val', transform = data_tfs['test'])
        # test_dataset = datasets.ImageFolder(root = data_root + '/train', transform = data_tfs['test'])

        train_loader = tud.DataLoader(dataset = train_data, batch_size = train_batch_size, shuffle = True)
        
        # test_loader = tud.DataLoader(dataset = test_dataset, batch_size = test_batch_size, shuffle = False)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        label_x=[[0]*2 for row in range(len(labels))]
        
        for i in range(2):
            layerList=popList[i]
            model = Model.Net(layerList).to(device)
            print(model)
            
            creterion = nn.CrossEntropyLoss()
            #optimer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
            
            optimer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            for epcho in range(0,Epoch):
                trained_data = 0
                for indx, (image, label) in enumerate(train_loader):
                    image = image.to(device)
                    label = label.to(device)
                    out = model(image)
                    loss = creterion(out, label)
                    # optimer.zero_grad()
                    optimer.zero_grad()
                    loss.backward()
                    optimer.step()
                    #  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
                    trained_data = len(image) * (indx + 1)
                    if (indx + 1) % 10 == 0:
                        print('{}/{}               第{}轮第{}批的loss = {}'.format(trained_data, len(train_loader.dataset), epcho,
                                                                              indx,loss))

            root = os.getcwd()

            test_data=MyDataset(txt=root+'/creat_txt1.txt',
            transform=transforms.Compose([
                transforms.Grayscale(1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
                                        ]))
            
            for k in range(len(imgs)):
            # print(train_data.__getitem__(1))
            # train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False,num_workers=1)
            # print('num_of_trainData:', len(train_data))
            # print(train_data.imgs[1])
            # print(train_loader.dataset.__getitem__(1))
                test_data.__getitem__(k)
                test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False,num_workers=1)
                # print('num_of_trainData:', len(train_data))
                test_data.imgs[k]
                image=test_loader.dataset.__getitem__(k)[0]
                image=image.unsqueeze(0)
                image = image.to(device)
                #label = label.to(device)
                out = model(image)
                #loss = creterion(out, label)
                pred = torch.argmax(out, dim = 1)
                xxx = pred.cpu().numpy()
                xxx.tolist()
                label_x[k][i]=xxx[0]                
                #print(xxx)
                print('{}  {}'.format(xxx[0],test_loader.dataset.__getitem__(k)[1]))
            # for indx, (image, label) in enumerate(test_loader):
            #         image = image.to(device)
            #         label = label.to(device)
            #         out = model(image)
            #         #loss = creterion(out, label)
            #         pred = torch.argmax(out, dim = 1)
            #         xxx = pred.cpu().numpy()
            #         xxx.tolist()
            #         label_x[indx][i]=xxx
        
        if my_iter>0:
            ''' 删除上一代增加的伪标签 '''      
            root_3 = os.getcwd()
            '''-------------------------------------------------'''
            txt_3=root_3 + '/accept_txt'+ str(my_iter-1) + '.txt'
            fh = open(txt_3, 'r')#按照传入的路径和txt文本参数，打开这个文本，并读取内容
            iter_imgs = []
            for line in fh: #迭代该列表#按行循环txt文本中的内
                line = line.strip('\n')
                line = line.rstrip('\n')# 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
                words = line.split() #用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
                #imgs.append((words[0],int(words[1]))) #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
                if int(words[0])<=17:
                    iter_imgs.append((int(words[0]),words[1]))#把txt里的内容读入imgs
                    
            iter_new_img=[]
            for img in iter_imgs:
                split_img = img[1].split('/')
                #for i in range(1,8):
                iter_new_img.append('/'+split_img[1]+'/'+split_img[2]+ \
                               '/'+split_img[3]+'/'+split_img[4]+ \
                               '/'+ 'train' +'/'+str(img[0])+ \
                               '/'+split_img[7])
#                my_remove_img='/'+split_img[1]+'/'+split_img[2]+ \
#                              '/'+split_img[3]+'/'+split_img[4]+ \
#                              '/'+split_img[5]+'/'+str(img[0])+
#                              '/'+split_img[7]
#                os.remove(my_remove_img)
            for img in iter_new_img:
                os.remove(img)
        print('安全时间，可以终止程序...........................')
        time.sleep(10)
        print('危险时间，请勿终止程序！！！！！！！！！！！！！！！！！')
#            ''' 更改图片写入的路径 '''      
#        root_3 = os.getcwd()
#        '''-------------------------------------------------'''
#        txt_3=root_3 + '/accept_txt'+ str(my_iter) + '.txt'
#        fh = open(txt_3, 'r')#按照传入的路径和txt文本参数，打开这个文本，并读取内容
#        imgs = []
#        for line in fh: #迭代该列表#按行循环txt文本中的内
#            line = line.strip('\n')
#            line = line.rstrip('\n')# 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
#            words = line.split() #用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
#            #imgs.append((words[0],int(words[1]))) #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
#            if int(words[0])<=17:
#                imgs.append((int(words[0]),words[1]))#把txt里的内容读入imgs
#
#        new_img=[]
#        for img in imgs:
#            split_img = img[1].split('/')
#            #for i in range(1,8):
#            new_img.append('/'+split_img[1]+'/'+split_img[2]+ \
#            '/'+split_img[3]+'/'+split_img[4]+ \
#            '/'+split_img[5]+'/'+str(img[0])+ \
#            '/'+split_img[7])

        acce_num=0
        
        with open(root + '/accept_txt'+ str(my_iter) + '.txt','w') as f:         
            for i in range(len(imgs)):
                #if label_x[i][0]==label_x[i][1] and label_x[i][0]==label_x[i][2] and label_x[i][1]==label_x[i][2] and random.random()<0.3:
                    if label_x[i][0]==label_x[i][1] and random.random()<0.3:
                        acce_num=acce_num+1
                        f.writelines(str(label_x[i][0]))
                        f.writelines(' ' + imgs[i] + '\n')
#                        old_img = imgs[i][1]
#                        new_img = new_img[i]
#                        shutil.copyfile(old_img,new_img)
                        print('接受的标签{},图片名称{}'.format(label_x[i][0],imgs[i]))
            f.writelines(str(acce_num))         
        f.close()                
        print('扩充的标签总数{}'.format(acce_num))            


        print('危险时间，不可以中断程序!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


        ''' 更改图片写入的路径 '''      
        root_3 = os.getcwd()
        '''-------------------------------------------------'''
        txt_3=root_3 + '/accept_txt'+ str(my_iter) + '.txt'
        fh = open(txt_3, 'r')#按照传入的路径和txt文本参数，打开这个文本，并读取内容
        iter_imgs = []
        for line in fh: #迭代该列表#按行循环txt文本中的内
            line = line.strip('\n')
            line = line.rstrip('\n')# 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split() #用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            #imgs.append((words[0],int(words[1]))) #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            if int(words[0])<=17:
                iter_imgs.append((int(words[0]),words[1]))#把txt里的内容读入imgs

        iter_new_img=[]
        for img in iter_imgs:
            split_img = img[1].split('/')
            #for i in range(1,8):
            iter_new_img.append('/'+split_img[1]+'/'+split_img[2]+ \
            '/'+split_img[3]+'/'+split_img[4]+ \
            '/'+ 'train' +'/'+str(img[0])+ \
            '/'+split_img[7])

        ''' 增加标签,将投票结果一致的图片复制到训练集中 ''' 
#        with open(root + '/accept_txt'+ str(my_iter) + '.txt','w') as f:         
        for i in range(acce_num):
                #if label_x[i][0]==label_x[i][1] and label_x[i][0]==label_x[i][2] and label_x[i][1]==label_x[i][2] and random.random()<0.3:
                #    if label_x[i][0]==label_x[i][1] and random.random()<0.3:
#                        acce_num=acce_num+1
#                        f.writelines(str(label_x[i][0]))
#                        f.writelines(' ' + imgs[i] + '\n')
            old_img1 = iter_imgs[i][1]
            new_img1 = iter_new_img[i]
            shutil.copyfile(old_img1,new_img1)
#           print('接受的标签{},图片名称{}'.format(label_x[i][0],imgs[i]))

        '''   重新生成训练集目录   '''
        root = os.getcwd()
        files = os.listdir(root+'/train')

        with open(root + '/creat_txt.txt','w') as f:
            for filename in files:
                images = os.listdir(root + '/train/' +filename)
                for image in images:
                    #f.writelines(os.path.join(root,'/train',filename) + '/' + image)
                    f.writelines(root + '/train/' + filename + '/' + image)
                    f.writelines(' ' + filename + '\n')
        f.close()

##               img = Image.open('E://evo_dataset/train/0/'+ str(a[i])+'.jpg')
#                img=Image.open(imgs[i])
##                     plt.figure("Image") # 图像窗口名称
##                     plt.imshow(img)
##                     plt.axis('off') # 关掉坐标轴为 off
##                     plt.title('image') # 图像题目
##                     plt.show()
##                     print(os.path.join(str(a[i]) + '.png'))
#                #root = data_root + '/train'
##                     img.save(os.path.join(str(a[i]) + '.png'))
#                ''' 有问题  '''
#                img.save(imgs_2[i])
#                #img.save(root + '/' + labels[i] +'.jpg')

if __name__=="__main__":

    my_pop=[]
    pop_acc=[]
    x=Pop_evo()
    y_errors,y_pop=x.pop_init()
    _ = x.offspringGenerate()
#    x.envSelect()    
    for i in range(30):
        _ = x.offspringGenerate()
        my_pop1,pop_acc1=x.envSelect()  
        my_pop.append(my_pop1[0])
        pop_acc.append(pop_acc1[0])
#        x.elit_netvote(i)

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
