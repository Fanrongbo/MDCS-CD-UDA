# -*- coding: utf-8 -*-

# from PIL import Image
import os
import cv2
import sys
import numpy as np
from skimage import io


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

def readmultichannel(path):
    im_name = os.listdir(path)[0][:-7]
    # print(path + im_name + "B04.tif")
    # r = cv2.imread(path + im_name + "B04.tif")
    # g = cv2.imread(path + im_name + "B03.tif")
    # b = cv2.imread(path + im_name + "B02.tif")
    r = io.imread(path + im_name + "B04.tif")
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    img = np.empty((r.shape[0], r.shape[1], 3), dtype="uint8")  # 512x512
    # img[:, :, 0] = b/b.max()*255
    # img[:, :, 1] = g/g.max()*255
    # img[:, :, 2] = r/r.max()*255
    img[:, :, 0] = (b-b.min())/(b.max()-b.min())*255
    img[:, :, 1] = (g-g.min())/(g.max()-g.min())*255
    img[:, :, 2] = (r-r.min())/(r.max()-r.min())*255
    # img = (img - img.mean()) / img.std()
    img = img.astype(np.uint8)
    return img
def read_OSCD(root_path, in_name):
    path=root_path+in_name
    # img1=readmultichannel(path+ '/imgs_1_rect/')
    # img2=readmultichannel(path+ '/imgs_2_rect/')
    # label = cv2.imread(path + '/cm/cm.png')
    img1 = cv2.imread(path + '/pair/img1.png')
    img2 = cv2.imread(path + '/pair/img2.png')
    label = cv2.imread(path + '/cm/cm.png')
    # label=(label/label.max())*255
    label = label.astype(np.uint8)

    return img1, img2, label

# imageDir="./example/images/"  #./Original/Images/Labels     #原大图像数据
# saveDir="./example/" + str(crop_w) + "x" + str(crop_h) + "/image/"    ##裁剪小图像数据
def gen_data(train, names, root_path, saveDir, crop_w,stride):

    num = 0
    crop_h = crop_w
    for in_name in names:
        I1, I2, cm = read_OSCD(root_path, in_name)#read_OSCD  read_GZ
        h_ori, w_ori, _ = I1.shape
        ex=4
        I1 = cv2.resize(I1, (h_ori*ex, w_ori*ex), interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024))
        I2 = cv2.resize(I2, (h_ori*ex, w_ori*ex), interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024))
        cm = cv2.resize(cm, (h_ori*ex, w_ori*ex), interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024))
        I1_ori = I1.astype(np.uint8)
        I2_ori = I2.astype(np.uint8)
        cm_ori = cm.astype(np.uint8)
        # I1_ori,I2_ori,cm_ori=I1, I2, cm
        h_ori, w_ori, _ = I1_ori.shape
        print(in_name)
        # if in_name=="paris" or in_name=="bordeaux" or in_name=="beihai":
        #     continue
            # cv2.imwrite('I1_ori.png', I1_ori)
            # cv2.imwrite('I2_ori.png', I2_ori)
            # cv2.imwrite('cm_ori.png', cm_ori)
            # sys.exit(0)
        for ratio in range(1,999):
            if w_ori / ratio < 260 or h_ori / ratio < 260:
                break
            if ratio!=1:
                print(in_name,'resize:',ratio)
                w_new=int(w_ori/ratio)
                h_new=int(h_ori/ratio)
                I1 = cv2.resize(I1_ori, (h_new,w_new),interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024))
                I2 = cv2.resize(I2_ori, (h_new,w_new),interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024))
                cm = cv2.resize(cm_ori, (h_new,w_new),interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024))
                I1 = I1.astype(np.uint8)
                I2 = I2.astype(np.uint8)
                cm = cm.astype(np.uint8)
            else:
                I1 = I1_ori
                I2 = I2_ori
                cm = cm_ori
            h, w, _ = I1.shape
            padding_h = (h // stride + 1) * stride
            padding_w = (w // stride + 1) * stride
            # nw = ceil((s[0] - self.patch_side + 1) / self.stride)  #
            # nh = ceil((s[1] - self.patch_side + 1) / self.stride)
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
                    ch, cw,_ = crop_T1.shape
                    if (len(crop_cm[crop_cm != 0])) / (crop_w * crop_h) > (0.2/ratio) and ch==crop_h and cw==crop_w  or not train:
                    # if ch==crop_h and cw==crop_w:
                        if train:
                            saveName = in_name.split('.')[0] + '_' + str(i) + '_' + str(j) + ".png"  # 小图像名称，内含小图像的顺序
                            cv2.imwrite(root_path + saveDir[0] + saveName, crop_T1)
                            cv2.imwrite(root_path + saveDir[1] + saveName, crop_T2)
                            cv2.imwrite(root_path + saveDir[2] + saveName, crop_cm)
                            # print(saveDir[2] + saveName)
                            num = num + 1
                            print('train num generated: ', num,crop_T1.shape)

                            # if i >0:
                            #     break
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
# root_path = '/data/frb/python_project_frb/datasets/CD_Data_GZ/data/'
root_path= '/data/frb/python_project_frb/datasets/OSCD/image/'
# root_path = './'

# dataset_name = '../OSCD_aug3'
dataset_name = '../OSCD_augTotal'
saveDir = ['%s/train/T1/' % dataset_name, '%s/train/T2/' % dataset_name, '%s/train/label/' % dataset_name,
           '%s/test/T1/' % dataset_name, '%s/test/T2/' % dataset_name, '%s/test/label/' % dataset_name]
for dirr in saveDir:
    dirflag = mkdir(root_path + dirr)
    if not dirflag:
        break

fnametrain = 'total.txt'#train  total
fnametest = 'test.txt'#test

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
# train = False
# stride = crop_w
# gen_data(train, namestest, root_path, saveDir, crop_w,stride)
