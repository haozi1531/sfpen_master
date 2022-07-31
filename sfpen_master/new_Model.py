# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 12:10:02 2021

@author: 17783
"""

import torch.nn as nn
from torch.nn import init


class Net(nn.Module):
    
        def weight_init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
                init.kaiming_uniform(m.weight)
                init.zeros_(m.bias)        
        def __init__(self,layerList,core_list):
            self.layerList = layerList
            self.core_list = core_list
            super(Net, self).__init__()
            
            
            
            hsi_len=220
            cla_num=16
            if len(layerList)==1:
                self.conv1 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=layerList[0], kernel_size=[3,3,3], stride=1,padding=(0,0,1)),  #输出55*55 96
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm3d(layerList[0]))    #输出27*27 96
                self.fc = nn.Sequential(
                                #nn.Linear((hsi_len-2*len(layerList)) * layerList[0], 64),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear((hsi_len) * layerList[0], cla_num)
                                #nn.Softmax(dim = 1)
                                )
            if len(layerList)==2:
                self.conv1 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=layerList[0], kernel_size=[3,3,3], stride=1,padding=(0,0,1)),  #输出55*55 96
                                    nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[0])) #输出27*27 96
                self.conv2 = nn.Sequential(nn.Conv3d(in_channels=layerList[0], out_channels=layerList[1], kernel_size=[1,1,core_list[1]], stride=1,padding=(0,0,core_list[1]//2)),  #输出27*27 256
                                    nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[1]))
                self.fc = nn.Sequential(
                                #nn.Linear((hsi_len-2*len(layerList)) * layerList[1], 64),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear((hsi_len) * layerList[1], cla_num)
                                #nn.Softmax(dim = 1),
                                )
                #self.dropout = nn.Dropout(p=0.6)
            if len(layerList)==3:
                self.conv1 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=layerList[0], kernel_size=[3,3,3], stride=1,padding=(0,0,1)),  #输出55*55 96
                                    nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[0]))    #输出27*27 96
                self.conv2 = nn.Sequential(nn.Conv3d(in_channels=layerList[0], out_channels=layerList[1], kernel_size=[1,1,core_list[1]], stride=1,padding=(0,0,core_list[1]//2)),  #输出27*27 256
                                    nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[1]))
                self.conv3 = nn.Sequential(nn.Conv3d(in_channels=layerList[1], out_channels=layerList[2], kernel_size=[1,1,core_list[2]], stride=1,padding=(0,0,core_list[2]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[2]))
                self.fc = nn.Sequential(
                                #nn.Linear((hsi_len-2*len(layerList)) * layerList[2], 64),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear((hsi_len) * layerList[2], cla_num)
                                #nn.Softmax(dim = 1)
                                )
                #self.dropout = nn.Dropout(p=0.6)
            if len(layerList)==4:
                self.conv1 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=layerList[0], kernel_size=[3,3,3], stride=1,padding=(0,0,1)),  #输出55*55 96
                                    nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[0]))    #输出27*27 96
                self.conv2 = nn.Sequential(nn.Conv3d(in_channels=layerList[0], out_channels=layerList[1], kernel_size=[1,1,core_list[1]],stride=1,padding=(0,0,core_list[1]//2)),  #输出27*27 256
                                    nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[1]))
                self.conv3 = nn.Sequential(nn.Conv3d(in_channels=layerList[1], out_channels=layerList[2], kernel_size=[1,1,core_list[2]], stride=1,padding=(0,0,core_list[2]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[2]))
                self.conv4 = nn.Sequential(nn.Conv3d(in_channels=layerList[2], out_channels=layerList[3], kernel_size=[1,1,core_list[3]], stride=1,padding=(0,0,core_list[3]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[3]))                
                self.fc = nn.Sequential(
                                #nn.Linear((hsi_len-2*len(layerList))  * layerList[3], 64),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear((hsi_len) * layerList[3], cla_num)
                                #nn.Softmax(dim = 1)
                                )
            if len(layerList)==5:
                self.conv1 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=layerList[0], kernel_size=[3,3,3], stride=1,padding=(0,0,1)),  #输出55*55 96
                                    nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[0]))    #输出27*27 96
                self.conv2 = nn.Sequential(nn.Conv3d(in_channels=layerList[0], out_channels=layerList[1], kernel_size=[1,1,core_list[1]],stride=1,padding=(0,0,core_list[1]//2)),  #输出27*27 256
                                    nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[1]))
                self.conv3 = nn.Sequential(nn.Conv3d(in_channels=layerList[1], out_channels=layerList[2], kernel_size=[1,1,core_list[2]], stride=1,padding=(0,0,core_list[2]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[2]))
                self.conv4 = nn.Sequential(nn.Conv3d(in_channels=layerList[2], out_channels=layerList[3], kernel_size=[1,1,core_list[3]], stride=1,padding=(0,0,core_list[3]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[3]))                
                self.conv5 = nn.Sequential(nn.Conv3d(in_channels=layerList[3], out_channels=layerList[4], kernel_size=[1,1,core_list[4]], stride=1,padding=(0,0,core_list[4]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[4]))    
                self.fc = nn.Sequential(
                                #nn.Linear((hsi_len-2*len(layerList))  *  layerList[4], 64),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear((hsi_len ) * layerList[4], cla_num)
                                #nn.Softmax(dim = 1)
                                )
            if len(layerList)==6:
                self.conv1 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=layerList[0], kernel_size=[3,3,3], stride=1,padding=(0,0,1)),  #输出55*55 96
                                    nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[0]))    #输出27*27 96
                self.conv2 = nn.Sequential(nn.Conv3d(in_channels=layerList[0], out_channels=layerList[1], kernel_size=[1,1,core_list[1]],stride=1,padding=(0,0,core_list[1]//2)),  #输出27*27 256
                                    nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[1]))
                self.conv3 = nn.Sequential(nn.Conv3d(in_channels=layerList[1], out_channels=layerList[2], kernel_size=[1,1,core_list[2]], stride=1,padding=(0,0,core_list[2]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[2]))
                self.conv4 = nn.Sequential(nn.Conv3d(in_channels=layerList[2], out_channels=layerList[3], kernel_size=[1,1,core_list[3]], stride=1,padding=(0,0,core_list[3]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[3]))                
                self.conv5 = nn.Sequential(nn.Conv3d(in_channels=layerList[3], out_channels=layerList[4], kernel_size=[1,1,core_list[4]], stride=1,padding=(0,0,core_list[4]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[4])) 
                self.conv6 = nn.Sequential(nn.Conv3d(in_channels=layerList[4], out_channels=layerList[5], kernel_size=[1,1,core_list[5]], stride=1,padding=(0,0,core_list[5]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[5]))           
                self.fc = nn.Sequential(
                                #nn.Linear((hsi_len-6*len(layerList))  *  layerList[5], 64),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear((hsi_len) * layerList[5], cla_num)
                                #nn.Softmax(dim = 1)
                                )               
            if len(layerList)==7:
                self.conv1 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=layerList[0], kernel_size=[3,3,3], stride=1,padding=(0,0,1)),  #输出55*55 96
                                    nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[0]))    #输出27*27 96
                self.conv2 = nn.Sequential(nn.Conv3d(in_channels=layerList[0], out_channels=layerList[1], kernel_size=[1,1,core_list[1]],stride=1,padding=(0,0,core_list[1]//2)),  #输出27*27 256
                                    nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[1]))
                self.conv3 = nn.Sequential(nn.Conv3d(in_channels=layerList[1], out_channels=layerList[2], kernel_size=[1,1,core_list[2]], stride=1,padding=(0,0,core_list[2]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[2]))
                self.conv4 = nn.Sequential(nn.Conv3d(in_channels=layerList[2], out_channels=layerList[3], kernel_size=[1,1,core_list[3]], stride=1,padding=(0,0,core_list[3]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[3]))                
                self.conv5 = nn.Sequential(nn.Conv3d(in_channels=layerList[3], out_channels=layerList[4], kernel_size=[1,1,core_list[4]], stride=1,padding=(0,0,core_list[4]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[4])) 
                self.conv6 = nn.Sequential(nn.Conv3d(in_channels=layerList[4], out_channels=layerList[5], kernel_size=[1,1,core_list[5]], stride=1,padding=(0,0,core_list[5]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[5]))  
                self.conv7 = nn.Sequential(nn.Conv3d(in_channels=layerList[5], out_channels=layerList[6], kernel_size=[1,1,core_list[6]], stride=1,padding=(0,0,core_list[6]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[6]))            
                self.fc = nn.Sequential(
                                #nn.Linear((hsi_len-2*len(layerList))  *  layerList[6], 64),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear((hsi_len) * layerList[6], cla_num)
                                #nn.Softmax(dim = 1)
                                )
            if len(layerList)==8:
                self.conv1 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=layerList[0], kernel_size=[3,3,3], stride=1,padding=(0,0,1)),  #输出55*55 96
                                    nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[0]))    #输出27*27 96
                self.conv2 = nn.Sequential(nn.Conv3d(in_channels=layerList[0], out_channels=layerList[1], kernel_size=[1,1,core_list[1]],stride=1,padding=(0,0,core_list[1]//2)),  #输出27*27 256
                                    nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[1]))
                self.conv3 = nn.Sequential(nn.Conv3d(in_channels=layerList[1], out_channels=layerList[2], kernel_size=[1,1,core_list[2]], stride=1,padding=(0,0,core_list[2]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[2]))
                self.conv4 = nn.Sequential(nn.Conv3d(in_channels=layerList[2], out_channels=layerList[3], kernel_size=[1,1,core_list[3]], stride=1,padding=(0,0,core_list[3]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[3]))                
                self.conv5 = nn.Sequential(nn.Conv3d(in_channels=layerList[3], out_channels=layerList[4], kernel_size=[1,1,core_list[4]], stride=1,padding=(0,0,core_list[4]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[4])) 
                self.conv6 = nn.Sequential(nn.Conv3d(in_channels=layerList[4], out_channels=layerList[5], kernel_size=[1,1,core_list[5]], stride=1,padding=(0,0,core_list[5]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[5]))  
                self.conv7 = nn.Sequential(nn.Conv3d(in_channels=layerList[5], out_channels=layerList[6], kernel_size=[1,1,core_list[6]], stride=1,padding=(0,0,core_list[6]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[6]))   
                self.conv8 = nn.Sequential(nn.Conv3d(in_channels=layerList[6], out_channels=layerList[7], kernel_size=[1,1,core_list[7]], stride=1,padding=(0,0,core_list[7]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[7]))
                self.fc = nn.Sequential(
                                #nn.Linear((hsi_len-2*len(layerList)) *  layerList[7], 64),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear((hsi_len) * layerList[7], cla_num)
                                #nn.Softmax(dim = 1)
                                )
                
            if len(layerList)==9:
                self.conv1 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=layerList[0], kernel_size=[3,3,3], stride=1,padding=(0,0,1)),  #输出55*55 96
                                    nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[0]))    #输出27*27 96
                self.conv2 = nn.Sequential(nn.Conv3d(in_channels=layerList[0], out_channels=layerList[1], kernel_size=[1,1,core_list[1]],stride=1,padding=(0,0,core_list[1]//2)),  #输出27*27 256
                                    nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[1]))
                self.conv3 = nn.Sequential(nn.Conv3d(in_channels=layerList[1], out_channels=layerList[2], kernel_size=[1,1,core_list[2]], stride=1,padding=(0,0,core_list[2]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[2]))
                self.conv4 = nn.Sequential(nn.Conv3d(in_channels=layerList[2], out_channels=layerList[3], kernel_size=[1,1,core_list[3]], stride=1,padding=(0,0,core_list[3]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[3]))                
                self.conv5 = nn.Sequential(nn.Conv3d(in_channels=layerList[3], out_channels=layerList[4], kernel_size=[1,1,core_list[4]], stride=1,padding=(0,0,core_list[4]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[4])) 
                self.conv6 = nn.Sequential(nn.Conv3d(in_channels=layerList[4], out_channels=layerList[5], kernel_size=[1,1,core_list[5]], stride=1,padding=(0,0,core_list[5]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[5]))  
                self.conv7 = nn.Sequential(nn.Conv3d(in_channels=layerList[5], out_channels=layerList[6], kernel_size=[1,1,core_list[6]], stride=1,padding=(0,0,core_list[6]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[6]))   
                self.conv8 = nn.Sequential(nn.Conv3d(in_channels=layerList[6], out_channels=layerList[7], kernel_size=[1,1,core_list[7]], stride=1,padding=(0,0,core_list[7]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[7]))
                self.conv9 = nn.Sequential(nn.Conv3d(in_channels=layerList[7], out_channels=layerList[8], kernel_size=[1,1,core_list[8]], stride=1,padding=(0,0,core_list[8]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[8]))                
                self.fc = nn.Sequential(
                                #nn.Linear((hsi_len-2*len(layerList)) *  layerList[8], 64),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear((hsi_len) * layerList[8], cla_num)
                                #nn.Softmax(dim = 1)
                                )
            if len(layerList)==10:
                self.conv1 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=layerList[0], kernel_size=[3,3,3], stride=1,padding=(0,0,1)),  #输出55*55 96
                                    nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[0]))    #输出27*27 96
                self.conv2 = nn.Sequential(nn.Conv3d(in_channels=layerList[0], out_channels=layerList[1], kernel_size=[1,1,core_list[1]],stride=1,padding=(0,0,core_list[1]//2)),  #输出27*27 256
                                    nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[1]))
                self.conv3 = nn.Sequential(nn.Conv3d(in_channels=layerList[1], out_channels=layerList[2], kernel_size=[1,1,core_list[2]], stride=1,padding=(0,0,core_list[2]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[2]))
                self.conv4 = nn.Sequential(nn.Conv3d(in_channels=layerList[2], out_channels=layerList[3], kernel_size=[1,1,core_list[3]], stride=1,padding=(0,0,core_list[3]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[3]))                
                self.conv5 = nn.Sequential(nn.Conv3d(in_channels=layerList[3], out_channels=layerList[4], kernel_size=[1,1,core_list[4]], stride=1,padding=(0,0,core_list[4]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[4])) 
                self.conv6 = nn.Sequential(nn.Conv3d(in_channels=layerList[4], out_channels=layerList[5], kernel_size=[1,1,core_list[5]], stride=1,padding=(0,0,core_list[5]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[5]))  
                self.conv7 = nn.Sequential(nn.Conv3d(in_channels=layerList[5], out_channels=layerList[6], kernel_size=[1,1,core_list[6]], stride=1,padding=(0,0,core_list[6]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[6]))   
                self.conv8 = nn.Sequential(nn.Conv3d(in_channels=layerList[6], out_channels=layerList[7], kernel_size=[1,1,core_list[7]], stride=1,padding=(0,0,core_list[7]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[7]))
                self.conv9 = nn.Sequential(nn.Conv3d(in_channels=layerList[7], out_channels=layerList[8], kernel_size=[1,1,core_list[8]], stride=1,padding=(0,0,core_list[8]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[8])) 
                self.conv10 = nn.Sequential(nn.Conv3d(in_channels=layerList[8], out_channels=layerList[9], kernel_size=[1,1,core_list[9]], stride=1,padding=(0,0,core_list[9]//2)),
                                  nn.ReLU(inplace=True),nn.BatchNorm3d(layerList[9]))               
                self.fc = nn.Sequential(
                                #nn.Linear((hsi_len-2*len(layerList)) *  layerList[9], 64),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear((hsi_len) * layerList[9], cla_num)
                                #nn.Softmax(dim = 1)
                                )
                 
        def forward(self, input):
            layerList=self.layerList
            if len(layerList)==1:
                input = self.conv1(input)
            if len(layerList)==2:
                input = self.conv1(input)
                input = self.conv2(input)
            if len(layerList)==3:
                input = self.conv1(input)
                input = self.conv2(input)
                input = self.conv3(input)
            if len(layerList)==4:
                input = self.conv1(input)
                input = self.conv2(input)
                input = self.conv3(input)
                input = self.conv4(input)                
            if len(layerList)==5:
                input = self.conv1(input)
                input = self.conv2(input)
                input = self.conv3(input)
                input = self.conv4(input)  
                input = self.conv5(input)
            if len(layerList)==6:
                input = self.conv1(input)
                input = self.conv2(input)
                input = self.conv3(input)
                input = self.conv4(input)  
                input = self.conv5(input)
                input = self.conv6(input)
            if len(layerList)==7:
                input = self.conv1(input)
                input = self.conv2(input)
                input = self.conv3(input)
                input = self.conv4(input)  
                input = self.conv5(input)
                input = self.conv6(input)
                input = self.conv7(input)
            if len(layerList)==8:
                input = self.conv1(input)
                input = self.conv2(input)
                input = self.conv3(input)
                input = self.conv4(input)  
                input = self.conv5(input)
                input = self.conv6(input)
                input = self.conv7(input)
                input = self.conv8(input)
            if len(layerList)==9:
                input = self.conv1(input)
                input = self.conv2(input)
                input = self.conv3(input)
                input = self.conv4(input)  
                input = self.conv5(input)
                input = self.conv6(input)
                input = self.conv7(input)
                input = self.conv8(input)
                input = self.conv9(input)
            if len(layerList)==10:
                input = self.conv1(input)
                input = self.conv2(input)
                input = self.conv3(input)
                input = self.conv4(input)  
                input = self.conv5(input)
                input = self.conv6(input)
                input = self.conv7(input)
                input = self.conv8(input)
                input = self.conv9(input)
                input = self.conv10(input)
            #input = self.dropout(input)    
            input = input.view(input.size(0), -1)
            input = self.fc(input)
            return input