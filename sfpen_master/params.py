# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:39:50 2021

@author: 17783
"""
import torch
from torchvision.models import resnet50
from thop import profile
import Modelss as new_Model
import models
from torchsummaryX import summary
      

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



device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


layerList=[256,256,128,128,128]

num_list,core_list=layerlist_code(layerList)

model = new_Model.Net(num_list,core_list).to(device)




hamida =  models.HamidaEtAl().to(device)

he =  models.HeEtAl().to(device)

li=  models.LiEtAl().to(device)

luo =  models.LuoEtAl().to(device)


input = torch.randn(1,1, 3, 3, 103)

#a=summary(model, input)

macs1, params1 = profile(model, inputs=(input, ))

#print(params)


input = torch.randn(1,1,3, 3, 103)
macs, hamida1 = profile(hamida, inputs=(input, ))


macs, he1 = profile(he, inputs=(input, ))



macs,li1  = profile(li, inputs=(input, ))

macs, luo1 = profile(luo, inputs=(input, ))




#summary(modelss, input)