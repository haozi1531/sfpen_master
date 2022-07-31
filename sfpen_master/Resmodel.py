# -*- coding: utf-8 -*-
"""
Created on Wed May 12 21:46:56 2021

@author: 17783
"""
import torch.nn as nn
class Net(nn.Module):
        def __init__(self,layerList):
            self.layerList = layerList
            super(Net, self).__init__()
            hsi_len=220
            
            cla_num=16
            
            
            if len(layerList)==1:
                self.conv1 = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=layerList[0], kernel_size=[3,3], stride=1,padding=1),  #输出55*55 96
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm2d(layerList[0]))    #输出27*27 96
                self.fc = nn.Sequential(
                                #nn.Linear((hsi_len-2*len(layerList)) *  layerList[9], 64),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear((hsi_len) * layerList[0], cla_num)
                                #nn.Softmax(dim = 1)
                                )
            if len(layerList)==2:
                self.conv1 = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=layerList[0], kernel_size=[3,3], stride=1,padding=1),  #输出55*55 96
                                    nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[0]))    #输出27*27 96
                self.conv2 = nn.Sequential(nn.Conv1d(in_channels=layerList[0], out_channels=layerList[1], kernel_size=[3,1], stride=1, padding=1),  #输出27*27 256
                                    nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[1]))
                self.fc = nn.Sequential(
                                #nn.Linear((hsi_len-2*len(layerList)) *  layerList[9], 64),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(15984, cla_num)
                                #nn.Softmax(dim = 1)
                                )
            if len(layerList)==3:
                self.conv1 = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=layerList[0], kernel_size=[3,3], stride=1,padding=1),  #输出55*55 96
                                    nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[0]))    #输出27*27 96
                self.conv2 = nn.Sequential(nn.Conv1d(in_channels=layerList[0], out_channels=layerList[1], kernel_size=[3,1], stride=1,padding=1),  #输出27*27 256
                                    nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[1]))
                self.conv3 = nn.Sequential(nn.Conv1d(in_channels=layerList[1], out_channels=layerList[2], kernel_size=[3,1], stride=1,padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[2]))
                self.fc = nn.Sequential(
                                #nn.Linear((hsi_len-2*len(layerList)) *  layerList[9], 64),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear((hsi_len) * layerList[2], cla_num)
                                #nn.Softmax(dim = 1)
                                )
            if len(layerList)==4:
                self.conv1 = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=layerList[0], kernel_size=[3,3], stride=1,padding=1),  #输出55*55 96
                                    nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[0]))    #输出27*27 96
                self.conv2 = nn.Sequential(nn.Conv1d(in_channels=layerList[0], out_channels=layerList[1], kernel_size=[3,1],stride=1, padding=1),  #输出27*27 256
                                    nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[1]))
                self.conv3 = nn.Sequential(nn.Conv1d(in_channels=layerList[1], out_channels=layerList[2], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[2]))
                self.conv4 = nn.Sequential(nn.Conv1d(in_channels=layerList[2], out_channels=layerList[3], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[3]))                
                self.fc = nn.Sequential(
                                #nn.Linear((hsi_len-2*len(layerList)) *  layerList[9], 64),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(17250, cla_num)
                                #nn.Softmax(dim = 1)
                                )
            if len(layerList)==5:
                self.conv1 = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=layerList[0], kernel_size=[3,3], stride=1,padding=1),  #输出55*55 96
                                    nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[0]))    #输出27*27 96
                self.conv2 = nn.Sequential(nn.Conv1d(in_channels=layerList[0], out_channels=layerList[1], kernel_size=[3,1], stride=1, padding=1),  #输出27*27 256
                                    nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[1]))
                self.conv3 = nn.Sequential(nn.Conv1d(in_channels=layerList[1], out_channels=layerList[2], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[2]))
                self.conv4 = nn.Sequential(nn.Conv1d(in_channels=layerList[2], out_channels=layerList[3], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[3]))
                self.conv5 = nn.Sequential(nn.Conv1d(in_channels=layerList[3], out_channels=layerList[4], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[4]))    
                self.fc = nn.Sequential(
                                #nn.Linear((hsi_len-2*len(layerList)) *  layerList[9], 64),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(22272, cla_num)
                                #nn.Softmax(dim = 1)
                                )
            if len(layerList)==6:
                self.conv1 = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=layerList[0], kernel_size=[3,3], stride=1,padding=1),  #输出55*55 96
                                    nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[0]))    #输出27*27 96
                self.conv2 = nn.Sequential(nn.Conv1d(in_channels=layerList[0], out_channels=layerList[1], kernel_size=[3,1], stride=1, padding=1),  #输出27*27 256
                                    nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[1]))
                self.conv3 = nn.Sequential(nn.Conv1d(in_channels=layerList[1], out_channels=layerList[2], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[2]))
                self.conv4 = nn.Sequential(nn.Conv1d(in_channels=layerList[2], out_channels=layerList[3], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[3]))
                self.conv5 = nn.Sequential(nn.Conv1d(in_channels=layerList[3], out_channels=layerList[4], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[4]))
                self.conv6 = nn.Sequential(nn.Conv1d(in_channels=layerList[4], out_channels=layerList[5], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[5]))           
                self.fc = nn.Sequential(
                                #nn.Linear((hsi_len-2*len(layerList)) *  layerList[9], 64),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(16848, cla_num)
                                #nn.Softmax(dim = 1)
                                )              
            if len(layerList)==7:
                self.conv1 = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=layerList[0], kernel_size=[3,3], stride=1,padding=1),  #输出55*55 96
                                    nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[0]))    #输出27*27 96
                self.conv2 = nn.Sequential(nn.Conv1d(in_channels=layerList[0], out_channels=layerList[1], kernel_size=[3,1], stride=1, padding=1),  #输出27*27 256
                                    nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[1]))
                self.conv3 = nn.Sequential(nn.Conv1d(in_channels=layerList[1], out_channels=layerList[2], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[2]))
                self.conv4 = nn.Sequential(nn.Conv1d(in_channels=layerList[2], out_channels=layerList[3], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[3]))
                self.conv5 = nn.Sequential(nn.Conv1d(in_channels=layerList[3], out_channels=layerList[4], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[4]))
                self.conv6 = nn.Sequential(nn.Conv1d(in_channels=layerList[4], out_channels=layerList[5], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[5]))
                self.conv7 = nn.Sequential(nn.Conv1d(in_channels=layerList[5], out_channels=layerList[6], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[6]))            
                self.fc = nn.Sequential(
                                #nn.Linear((hsi_len-2*len(layerList)) *  layerList[9], 64),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear((hsi_len) * layerList[6], cla_num)
                                #nn.Softmax(dim = 1)
                                )
                
            if len(layerList)==8:
                self.conv1 = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=layerList[0], kernel_size=[3,3], stride=1,padding=1),  #输出55*55 96
                                    nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[0]))    #输出27*27 96
                self.conv2 = nn.Sequential(nn.Conv1d(in_channels=layerList[0], out_channels=layerList[1], kernel_size=[3,1], stride=1, padding=1),  #输出27*27 256
                                    nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[1]))
                self.conv3 = nn.Sequential(nn.Conv1d(in_channels=layerList[1], out_channels=layerList[2], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[2]))
                self.conv4 = nn.Sequential(nn.Conv1d(in_channels=layerList[2], out_channels=layerList[3], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[3]))
                self.conv5 = nn.Sequential(nn.Conv1d(in_channels=layerList[3], out_channels=layerList[4], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[4]))
                self.conv6 = nn.Sequential(nn.Conv1d(in_channels=layerList[4], out_channels=layerList[5], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[5]))
                self.conv7 = nn.Sequential(nn.Conv1d(in_channels=layerList[5], out_channels=layerList[6], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[6]))
                self.conv8 = nn.Sequential(nn.Conv1d(in_channels=layerList[6], out_channels=layerList[7], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[7]))
                self.fc = nn.Sequential(
                                #nn.Linear((hsi_len-2*len(layerList)) *  layerList[9], 64),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear((hsi_len) * layerList[7], cla_num)
                                #nn.Softmax(dim = 1)
                                )
                
            if len(layerList)==9:
                self.conv1 = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=layerList[0], kernel_size=[3,3], stride=1,padding=1),  #输出55*55 96
                                    nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[0]))    #输出27*27 96
                self.conv2 = nn.Sequential(nn.Conv1d(in_channels=layerList[0], out_channels=layerList[1], kernel_size=[3,1], stride=1, padding=1),  #输出27*27 256
                                    nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[1]))
                self.conv3 = nn.Sequential(nn.Conv1d(in_channels=layerList[1], out_channels=layerList[2], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[2]))
                self.conv4 = nn.Sequential(nn.Conv1d(in_channels=layerList[2], out_channels=layerList[3], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[3]))
                self.conv5 = nn.Sequential(nn.Conv1d(in_channels=layerList[3], out_channels=layerList[4], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[4]))
                self.conv6 = nn.Sequential(nn.Conv1d(in_channels=layerList[4], out_channels=layerList[5], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[5]))
                self.conv7 = nn.Sequential(nn.Conv1d(in_channels=layerList[5], out_channels=layerList[6], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[6]))
                self.conv8 = nn.Sequential(nn.Conv1d(in_channels=layerList[6], out_channels=layerList[7], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[7]))
                self.conv9 = nn.Sequential(nn.Conv1d(in_channels=layerList[7], out_channels=layerList[8], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[8]))                
                self.fc = nn.Sequential(
                                #nn.Linear((hsi_len-2*len(layerList)) *  layerList[9], 64),
                                #nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear((hsi_len) * layerList[8], cla_num)
                                #nn.Softmax(dim = 1)
                                )
            if len(layerList)==10:
                self.conv1 = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=layerList[0], kernel_size=[3,3], stride=1,padding=1),  #输出55*55 96
                                    nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[0]))    #输出27*27 96
                self.conv2 = nn.Sequential(nn.Conv1d(in_channels=layerList[0], out_channels=layerList[1], kernel_size=[3,1], stride=1, padding=1),  #输出27*27 256
                                    nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[1]))
                self.conv3 = nn.Sequential(nn.Conv1d(in_channels=layerList[1], out_channels=layerList[2], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[2]))
                self.conv4 = nn.Sequential(nn.Conv1d(in_channels=layerList[2], out_channels=layerList[3], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[3]))
                self.conv5 = nn.Sequential(nn.Conv1d(in_channels=layerList[3], out_channels=layerList[4], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[4]))
                self.conv6 = nn.Sequential(nn.Conv1d(in_channels=layerList[4], out_channels=layerList[5], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[5]))
                self.conv7 = nn.Sequential(nn.Conv1d(in_channels=layerList[5], out_channels=layerList[6], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[6]))
                self.conv8 = nn.Sequential(nn.Conv1d(in_channels=layerList[6], out_channels=layerList[7], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[7]))
                self.conv9 = nn.Sequential(nn.Conv1d(in_channels=layerList[7], out_channels=layerList[8], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[8]))
                self.conv10 = nn.Sequential(nn.Conv1d(in_channels=layerList[8], out_channels=layerList[9], kernel_size=[3,1], stride=1, padding=1),
                                  nn.ReLU(inplace=True),nn.BatchNorm2d(layerList[9]))               
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
            input = input.view(input.size(0), -1)
            input = self.fc(input)
            return input

