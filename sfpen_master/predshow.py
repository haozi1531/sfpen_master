# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 12:16:04 2021

@author: 17783
"""

# coding=gbk
# 实现读取一个TXT文件，将文件中的数据存放在一个列表中，
# 再将列表逐渐转换为数组和矩阵
# 最后利用矩阵中的数据，将其以图像的形式呈现出来
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
label_path_indian = root+'//HSI//Salinas_gt.mat'
    #data_path_pavia = 'dataset/Pavia_corrected.mat'
    #label_path_pavia = 'dataset/Pavia_gt.mat'
    #data_path_salinas = 'dataset/Salinas_corrected.mat'
    #label_path_salinas = 'dataset/Salinas_gt.mat'
    

label_path = label_path_indian
    
    


    
    
y_label=sio.loadmat(label_path_indian)
salinas_gt = y_label['salinas_gt']









list_result = dp=[[[0]*3 for j in range(512)]for i in range(217)]



for i in range(56975):
        list_result[i] = 0

for l in range(16):
    for i in range(512):
        for j in range(217):
            if salinas_gt[i][j]== l:
                list_result[i][j][0]=salinas_color[l][0]    
                list_result[i][j][1]=salinas_color[l][1]
                list_result[i][j][2]=salinas_color[l][2]




    # 再利用numpy将列表包装为数组
array1 = np.array(list_result)
# 进一步将array包装成矩阵
data = np.matrix(array1)
# 重新reshape一个矩阵为一个方阵
data = np.reshape(data,"RGB")
# 调用Image的formarray方法将矩阵数据转换为图像PIL类型的数据
new_map = Image.fromarray(data)
# 显示图像
new_map.show()




