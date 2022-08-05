
import torch
import torch.utils.data
from torch.autograd import Variable

import numpy as np
import cv2
import os
def train_path_loader(data_dir):
    print('train_path_loader:',data_dir)
    train_data = [i for i in os.listdir(data_dir + 'train/T1/') if not
    i.startswith('.')]
    train_data.sort()

    train_label_paths = []
    for img in train_data:
        train_label_paths.append(data_dir + 'train/label/' + img)
    train_data_path = []
    for img in train_data:
        train_data_path.append([data_dir + 'train/', img])
    train_dataset = {}
    for cp in range(len(train_data)):
        train_dataset[cp] = {'img_path': train_data_path[cp],
                         'label_img_path': train_label_paths[cp]}
    return train_dataset
def val_path_loader(data_dir):
    print('val_path_loader:',data_dir)

    valid_data = [i for i in os.listdir(data_dir + 'test/T1/') if not
    i.startswith('.')]
    valid_data.sort()
    val_label_paths = []

    for img in valid_data:
        val_label_paths.append(data_dir + 'test/label/' + img)
    val_data_path = []

    for img in valid_data:
        val_data_path.append([data_dir + 'test/', img])
    val_dataset = {}
    for cp in range(len(valid_data)):
        val_dataset[cp] = {'img_path': val_data_path[cp],
                         'label_img_path': val_label_paths[cp]}

    return  val_dataset
class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, data_dir,DAFlag=False):
        train_dataset_path=train_path_loader(data_dir)

        self.new_img_h = 256
        self.new_img_w = 256
        self.DAFlag=DAFlag
        self.examples = train_dataset_path
        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        dir = img_path[0]
        name = img_path[1]

        img1 = cv2.imread(dir + 'T1/' + name) # (shape: (1024, 2048, 3))
        img2 = cv2.imread(dir + 'T2/' + name) # (shape: (1024, 2048, 3))
        # print(dir + 'T2/' + name,dir + 'T1/' + name)

        ########## resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        # img1 = cv2.resize(img1, (self.new_img_w, self.new_img_h),
        #                  interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024, 3))
        # img2 = cv2.resize(img2, (self.new_img_w, self.new_img_h),
        #                  interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024, 3))
        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path) # (shape: (1024, 2048))
        # resize label_img without interpolation (want the resulting image to
        # still only contain pixel values corresponding to an object class):
        # label_img = cv2.resize(label_img, (self.new_img_w, self.new_img_h),
        #                        interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024))
        # nameout=name.split('.')[0]
        # cv2.imwrite(nameout+"test12.png", img1)
        # cv2.imwrite(nameout+"test22.png", img2)
        # cv2.imwrite(nameout+"label_img2.png", label_img)

        # flip the img and the label with 0.5 probability:
        flip = np.random.randint(low=0, high=2)
        # print('flip',flip)
        if flip == 1:
            img1 = cv2.flip(img1, 1)
            img2 = cv2.flip(img2, 1)
            label_img = cv2.flip(label_img, 1)

        ########################################################################
        # randomly scale the img and the label:
        ########################################################################
        scale = np.random.uniform(low=0.7, high=2.0)
        new_img_h = int(scale*self.new_img_h)
        new_img_w = int(scale*self.new_img_w)

        # resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        img1 = cv2.resize(img1, (new_img_w, new_img_h),
                         interpolation=cv2.INTER_NEAREST) # (shape: (new_img_h, new_img_w, 3))
        img2 = cv2.resize(img2, (new_img_w, new_img_h),
                         interpolation=cv2.INTER_NEAREST)  # (shape: (new_img_h, new_img_w, 3))
        # resize label_img without interpolation (want the resulting image to
        # still only contain pixel values corresponding to an object class):
        label_img = cv2.resize(label_img, (new_img_w, new_img_h),
                               interpolation=cv2.INTER_NEAREST) # (shape: (new_img_h, new_img_w))

        img1 = cv2.resize(img1, (self.new_img_w, self.new_img_h),
                          interpolation=cv2.INTER_NEAREST)  # (shape: (new_img_h, new_img_w, 3))
        img2 = cv2.resize(img2, (self.new_img_w, self.new_img_h),
                          interpolation=cv2.INTER_NEAREST)  # (shape: (new_img_h, new_img_w, 3))
        # resize label_img without interpolation (want the resulting image to
        # still only contain pixel values corresponding to an object class):
        label_img = cv2.resize(label_img, (self.new_img_w, self.new_img_h),
                               interpolation=cv2.INTER_NEAREST)  # (shape: (new_img_h, new_img_w))
        ########################################################################

        # # # # # # # # debug visualization START
        # print (scale)
        # print (new_img_h)
        # print (new_img_w)
        #
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("test", label_img)
        # cv2.waitKey(0)
        # # # # # # # # debug visualization END

        ########################################################################
        # select a 256x256 random crop from the img and label:
        ########################################################################
        # start_x = np.random.randint(low=0, high=(new_img_w - 256))
        # end_x = start_x + 256
        # start_y = np.random.randint(low=0, high=(new_img_h - 256))
        # end_y = start_y + 256
        #
        # img1 = img1[start_y:end_y, start_x:end_x] # (shape: (256, 256, 3))
        # img2 = img2[start_y:end_y, start_x:end_x] # (shape: (256, 256, 3))
        #
        # label_img = label_img[start_y:end_y, start_x:end_x] # (shape: (256, 256))
        ########################################################################

        # # # # # # # # debug visualization START

        # cv2.imwrite(nameout+"test1.png", img1)
        # cv2.imwrite(nameout+"test2.png", img2)
        # cv2.imwrite(nameout+"label_img.png", label_img)

        # # # # # # # # debug visualization END
        # nameout=name.split('.')[0]
        # cv2.imwrite(nameout+"test12.png", img1)
        # cv2.imwrite(nameout+"test22.png", img2)
        # cv2.imwrite(nameout+"label_img2.png", label_img)

        # normalize the img (with the mean and std for the pretrained ResNet):
        img1 = img1/255.0
        img2 = img2/255.0
        if self.DAFlag:
            img1 = img1 - np.array([0.485, 0.456, 0.406])
            img2 = img2 - np.array([0.485, 0.456, 0.406])

            img1 = img1/np.array([0.229, 0.224, 0.225]) # (shape: (256, 256, 3))
            img2 = img2/np.array([0.229, 0.224, 0.225]) # (shape: (256, 256, 3))

        img1 = np.transpose(img1, (2, 0, 1)) # (shape: (3, 256, 256))
        img2 = np.transpose(img2, (2, 0, 1)) # (shape: (3, 256, 256))

        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        # convert numpy -> torch:
        img1 = torch.from_numpy(img1) # (shape: (3, 256, 256))
        img2 = torch.from_numpy(img2) # (shape: (3, 256, 256))
        label_img=label_img[:,:,0]/255.0
        label_img = torch.from_numpy(label_img) # (shape: (256, 256))
        # print(img1.shape,label_img.shape)
        flip = np.random.randint(low=0, high=2)
        # print('flip',flip)
        if flip == 1:
            img1_out = img1
            img2_out = img2
        else:
            img1_out = img2
            img2_out = img1

        return (img1_out,img2_out,label_img)

    def __len__(self):
        return self.num_examples


class DatasetVal(torch.utils.data.Dataset):
    def __init__(self, data_dir,DAFlag=False):
        val_dataset_path = val_path_loader(data_dir)

        self.new_img_h = 256
        self.new_img_w = 256
        self.DAFlag=DAFlag
        self.examples = val_dataset_path
        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        dir = img_path[0]
        name = img_path[1]

        img1 = cv2.imread(dir + 'T1/' + name)  # (shape: (1024, 2048, 3))
        img2 = cv2.imread(dir + 'T2/' + name)  # (shape: (1024, 2048, 3))
        ########## resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        # img1 = cv2.resize(img1, (self.new_img_w, self.new_img_h),
        #                  interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024, 3))
        # img2 = cv2.resize(img2, (self.new_img_w, self.new_img_h),
        #                  interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024, 3))
        label_img_path = example["label_img_path"]
        # print(dir + 'T1/' + name, -1)
        # print(label_img_path)
        label_img = cv2.imread(label_img_path)  # (shape: (1024, 2048))
        # resize label_img without interpolation (want the resulting image to
        # still only contain pixel values corresponding to an object class):
        # label_img = cv2.resize(label_img, (self.new_img_w, self.new_img_h),
        #                        interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024))
        # nameout=name.split('.')[0]
        # cv2.imwrite(nameout+"test12.png", img1)
        # cv2.imwrite(nameout+"test22.png", img2)
        # cv2.imwrite(nameout+"label_img2.png", label_img)
        # # # # # # # debug visualization START
        # print (scale)
        # print (new_img_h)
        # print (new_img_w)
        #
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("test", label_img)
        # cv2.waitKey(0)
        # # # # # # # # debug visualization END

        # normalize the img (with the mean and std for the pretrained ResNet):
        img1 = img1 / 255.0
        img2 = img2 / 255.0
        if self.DAFlag:
            img1 = img1 - np.array([0.485, 0.456, 0.406])
            img2 = img2 - np.array([0.485, 0.456, 0.406])

            img1 = img1 / np.array([0.229, 0.224, 0.225])  # (shape: (256, 256, 3))
            img2 = img2 / np.array([0.229, 0.224, 0.225])  # (shape: (256, 256, 3))

        img1 = np.transpose(img1, (2, 0, 1))  # (shape: (3, 256, 256))
        img2 = np.transpose(img2, (2, 0, 1))  # (shape: (3, 256, 256))

        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        # convert numpy -> torch:
        img1 = torch.from_numpy(img1)  # (shape: (3, 256, 256))
        img2 = torch.from_numpy(img2)  # (shape: (3, 256, 256))
        label_img=label_img[:,:,0]/255.0

        label_img = torch.from_numpy(label_img)  # (shape: (256, 256))

        # print(img1.shape, label_img.shape)
        flip = np.random.randint(low=0, high=2)
        if flip == 1:
            img1_out = img1
            img2_out = img2
        else:
            img1_out = img2
            img2_out = img1
        return (img1_out, img2_out, label_img)

    def __len__(self):
        return self.num_examples
def getloader(path,batch_size):
    # train_dataset = DatasetTrain("/data/frb/python_project_frb/datasets/CD_Data_GZ/GZ_patch_256_select/")
    # val_dataset= val_path_loader("/data/frb/python_project_frb/datasets/CD_Data_GZ/GZ_patch_256_select/")
    train_dataset = DatasetTrain(path)
    val_dataset= DatasetVal(path)
    # print('val_dataset',val_dataset)
    source_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=1)
    target_train_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=1)

    target_test_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=1)
    return  source_loader,target_train_loader,target_test_loader

def getloaderDA(source_path,target_path,batch_size):
    # train_dataset = DatasetTrain("/data/frb/python_project_frb/datasets/CD_Data_GZ/GZ_patch_256_select/")
    # val_dataset= val_path_loader("/data/frb/python_project_frb/datasets/CD_Data_GZ/GZ_patch_256_select/")
    source_dataset = DatasetTrain(source_path,DAFlag=False)
    target_dataset= DatasetTrain(target_path,DAFlag=False)

    source_loader = torch.utils.data.DataLoader(dataset=source_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=1)
    target_train_loader = torch.utils.data.DataLoader(dataset=target_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=1)

    target_test_loader = torch.utils.data.DataLoader(dataset=target_dataset,
                                             batch_size=1, shuffle=False,
                                             num_workers=1)
    return  source_loader,target_train_loader,target_test_loader
# source_loader, _, target_test_loader = getloader("/data/frb/python_project_frb/datasets/CD_Data_GZ/GZ_patch_256_select/",16)

# for step, (img1,img2, label_imgs) in enumerate(target_test_loader):
#     break
#     # img1 = Variable(img1).cuda(0)  # (shape: (batch_size, 3, img_h, img_w))
#     # label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda(0)  # (shape: (batch_size, img_h, img_w))
#     # if step>1:
#     #     break