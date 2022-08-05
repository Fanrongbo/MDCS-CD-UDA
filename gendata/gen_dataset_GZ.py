# -*- coding: utf-8 -*-

# from PIL import Image
import os
import cv2
import sys
import numpy as np


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        dirflag = True
    else:
        dirflag = False
    return dirflag


def read_GZ(root_path, in_name):
    img1 = cv2.imread(root_path + 'T1/' + in_name)
    img2 = cv2.imread(root_path + 'T2/' + in_name)
    label = cv2.imread(root_path + '/labels_change/' + in_name)
    return img1, img2, label


# imageDir="./example/images/"  #./Original/Images/Labels     #原大图像数据
# saveDir="./example/" + str(crop_w) + "x" + str(crop_h) + "/image/"    ##裁剪小图像数据
def gen_data(train, names, root_path, saveDir, crop_w,stride):

    num = 0
    crop_h = crop_w
    for in_name in names:
        I1, I2, cm = read_GZ(root_path, in_name)
        I1 = I1.astype(np.uint8)
        I2 = I2.astype(np.uint8)
        cm = cm.astype(np.uint8)
        h, w, _ = I1.shape
        padding_h = (h // stride + 1) * stride
        padding_w = (w // stride + 1) * stride
        padding_img_T1 = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
        padding_img_T2 = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
        padding_img_cm = np.zeros((padding_h, padding_w), dtype=np.uint8)
        padding_img_T1[0:h, 0:w, :] = I1[:, :, :]
        padding_img_T2[0:h, 0:w, :] = I2[:, :, :]
        padding_img_cm[0:h, 0:w] = cm[:, :, 0]

        h_leave = h % crop_h
        w_leave = w % crop_w
        mask_whole = np.zeros((padding_h, padding_w), dtype=np.uint8)
        for i in range(padding_h // stride):
            for j in range(padding_w // stride):
                crop_T1 = padding_img_T1[i * stride:i * stride + crop_h, j * stride:j * stride + crop_w, :3]
                crop_T2 = padding_img_T2[i * stride:i * stride + crop_h, j * stride:j * stride + crop_w, :3]
                crop_cm = padding_img_cm[i * stride:i * stride + crop_h, j * stride:j * stride + crop_w]
                _, ch, cw = crop_T1.shape
                if (len(crop_cm[crop_cm != 0])) / (crop_w * crop_h) > 0.1 or not train:
                    if train:
                        saveName = in_name.split('.')[0] + '_' + str(i) + '_' + str(j) + ".png"  # 小图像名称，内含小图像的顺序
                        cv2.imwrite(root_path + saveDir[0] + saveName, crop_T1)
                        cv2.imwrite(root_path + saveDir[1] + saveName, crop_T2)
                        cv2.imwrite(root_path + saveDir[2] + saveName, crop_cm)
                        # print(saveDir[2] + saveName)
                        num = num + 1
                        print('train num generated: ', num)
                        # break
                    else:
                        saveName = in_name.split('.')[0] + '_' + str(i) + '_' + str(j) + ".png"  # 小图像名称，内含小图像的顺序
                        cv2.imwrite(root_path + saveDir[3] + saveName, crop_T1)
                        cv2.imwrite(root_path + saveDir[4] + saveName, crop_T2)
                        cv2.imwrite(root_path + saveDir[5] + saveName, crop_cm)
                        num = num + 1
                        print('val num generated: ', num)


crop_w = 256  # 裁剪图像宽度
crop_h = 256  # 裁剪图像高度
root_path = '/data/frb/python_project_frb/datasets/CD_Data_GZ/data/'
# root_path = './'

dataset_name = 'GZ_aug'
saveDir = ['%s/train/T1/' % dataset_name, '%s/train/T2/' % dataset_name, '%s/train/label/' % dataset_name,
           '%s/test/T1/' % dataset_name, '%s/test/T2/' % dataset_name, '%s/test/label/' % dataset_name]
for dirr in saveDir:
    dirflag = mkdir(root_path + dirr)
    if not dirflag:
        break

fnametrain = 'trainnnew.txt'
fnametest = 'testtnew.txt'

with open(root_path + fnametrain, "r") as f:  # 打开文件
    datatrain = f.read()  # 读取文件
datatrain = datatrain.split('\n')
namestrain = datatrain
print('train names list:', namestrain)

with open(root_path + fnametest, "r") as f:  # 打开文件
    datatest = f.read()  # 读取文件
datatest = datatest.split('\n')
namestest = datatest
print('test names list:', namestest)

if not dirflag:
    print("The Floder Have Existed!")
    sys.exit(0)
print('##############train dataset##################')
train = True
stride = 100
gen_data(train, namestrain, root_path, saveDir, crop_w,stride)
print('##############test dataset##################')
train = False
stride = crop_w
gen_data(train, namestest, root_path, saveDir, crop_w,stride)
