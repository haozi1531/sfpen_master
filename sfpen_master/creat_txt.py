import os

#root = os.getcwd()
#files = os.listdir(root+'/train_file')
#
#with open(root + '/creat_txt1.txt','w') as f:
#    for filename in files:
#        images = os.listdir(root + '/train_file/' +filename)
#        for image in images:
#            #f.writelines(os.path.join(root,'/train',filename) + '/' + image)
#            f.writelines(root + '/train_file/' + filename + '/' + image)
#            f.writelines(' ' + filename + '\n')
#f.close()


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

# import os
# root = os.getcwd()
# files = os.listdir(root+'/train')
# with open(root + '/creat_txt1.txt','w') as f:
#     for filename in range(17):
#         images = os.listdir(root + '/train/' + str(filename))
#         for image in images:
#             f.writelines(os.path.join(root,'train',str(filename)) + '\\' + image)
#             f.writelines(' ' + str(filename) + '\n')
# f.close()
