import os
import cv2
import sys
import numpy as np
import torch
from torch.autograd import Variable
import models.Models

def confuseMatrix(pred,label_source):
    pred = pred.cpu().numpy()
    label_source = label_source.cpu().numpy()
    pr = (pred > 0)
    gt = (label_source > 0)
    # print(pr)
    tp_e = np.logical_and(pr, gt).sum()
    tn_e = np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
    fp_e = np.logical_and(pr, np.logical_not(gt)).sum()
    fn_e = np.logical_and(np.logical_not(pr), gt).sum()

    return tp_e,tn_e,fp_e,fn_e
def read_GZ(root_path, in_name):
    img1 = cv2.imread(root_path + 'T1/' + in_name)
    img2 = cv2.imread(root_path + 'T2/' + in_name)
    label = cv2.imread(root_path + '/labels_change/' + in_name)
    return img1, img2, label
def predict(network,DEVICE, names, root_path, crop_w,stride):
    # load the trained convolutional neural network
    # softmax = nn.Softmax(dim=1)
    # softmax_0 = nn.LogSoftmax(dim=1)
    num = 0
    # stride = crop_w // 2
    crop_h = crop_w
    pre_list = []
    label_list = []
    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # for n in range(len(TEST_SET)):

    a=0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for in_name in names:
        I1, I2, cm = read_GZ(root_path, in_name)
        # I1 = I1.astype(np.uint8)
        # I2 = I2.astype(np.uint8)
        # cm = cm.astype(np.uint8)
        h, w, _ = I1.shape
        padding_h = (h // stride + 1) * stride
        padding_w = (w // stride + 1) * stride
        padding_img_T1 = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
        padding_img_T2 = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
        padding_img_cm = np.zeros((padding_h, padding_w), dtype=np.uint8)
        # print(padding_img_T1.shape,h,w)
        padding_img_T1[0:h, 0:w, :] = I1[:, :, :]
        padding_img_T2[0:h, 0:w, :] = I2[:, :, :]
        padding_img_cm[0:h, 0:w] = cm[:, :, 0]
        # cv2.imwrite('label_' + '_' + str(image_size) + '.png', cm*255)
        h_leave = h % crop_h
        w_leave = w % crop_w
        mask_whole = np.zeros((padding_h, padding_w), dtype=np.uint8)
        # if a>2:
        #     break
        # a=a+1.

        for i in range(padding_h // stride):
            for j in range(padding_w // stride):
                crop_T1 = padding_img_T1[i * stride:i * stride + crop_h, j * stride:j * stride + crop_w, :3]
                crop_T2 = padding_img_T2[i * stride:i * stride + crop_h, j * stride:j * stride + crop_w, :3]
                crop_cm = padding_img_cm[i * stride:i * stride + crop_h, j * stride:j * stride + crop_w]
                crop_T1 = crop_T1 / 255.0
                crop_T2 = crop_T2 / 255.0
                crop_cm=crop_cm/255.
                # crop_T1 = crop_T1 - np.array([0.485, 0.456, 0.406])
                # crop_T2 = crop_T2 - np.array([0.485, 0.456, 0.406])
                # crop_T1 = crop_T1 / np.array([0.229, 0.224, 0.225])  # (shape: (256, 256, 3))
                # crop_T2 = crop_T2 / np.array([0.229, 0.224, 0.225])  # (shape: (256, 256, 3))
                crop_T1 = np.transpose(crop_T1, (2, 0, 1))  # (shape: (3, 256, 256))
                crop_T2 = np.transpose(crop_T2, (2, 0, 1))  # (shape: (3, 256, 256))
                crop_T1 = crop_T1.astype(np.float32)
                crop_T2 = crop_T2.astype(np.float32)
                # convert numpy -> torch:
                crop_T1 = torch.from_numpy(crop_T1)  # (shape: (3, 256, 256))
                crop_T2 = torch.from_numpy(crop_T2)  # (shape: (3, 256, 256))
                crop_cm = torch.from_numpy(crop_cm)  # (shape: (3, 256, 256))
                # print(i,j,crop_T1.shape,crop_T2.shape,crop_cm.shape)
                _, ch, cw = crop_T1.shape
                network.eval()
                # network.freeze_bn_dr()
                with torch.no_grad():
                    data_T1_target = Variable(crop_T1.unsqueeze(0)).to(DEVICE)
                    data_T2_target = Variable(crop_T2.unsqueeze(0)).to(DEVICE)
                    label_target = Variable(
                        crop_cm.type(torch.LongTensor)).cuda()  # (shape: (batch_size, img_h, img_w))

                    s_output,_ = network.forward(data_T1_target, data_T2_target)
                    _, cd_preds = torch.max(s_output, 1)

                    tp_e, tn_e, fp_e, fn_e = confuseMatrix(cd_preds, label_target)  ###notice name of variance
                    tp += tp_e
                    tn += tn_e
                    fp += fp_e
                    fn += fn_e


                    pr = cd_preds.cpu().numpy()
                    pr = (pr > 0)
                    pr = pr.reshape((crop_h, crop_w)).astype(np.uint8)
                    mask_whole[i * stride:i * stride + crop_h, j * stride:j * stride + crop_w] = pr[:, :]
                    # if i == 0 and j == 0:
                    #     mask_whole[i * stride:i * stride + crop_h, j * stride:j * stride + crop_w] = pr[:, :]
                    #
                    # if i != 0 and j == 0:
                    #     mask_whole[i * stride + stride // 2:i * stride + crop_h - stride // 2,
                    #     j * stride:j * stride + crop_w] = pr[stride // 2: crop_h - stride // 2, :]
                    #
                    # if i == 0 and j != 0:
                    #     mask_whole[i * stride:i * stride + crop_h,
                    #     j * stride + stride // 2:j * stride + crop_w - stride // 2] = pr[:,
                    #                                                                       stride // 2: crop_w - stride // 2]
                    # if i != 0 and j != 0:
                    #     mask_whole[i * stride + stride // 2:i * stride + crop_h - stride // 2,
                    #     j * stride + stride // 2:j * stride + crop_w - stride // 2] = pr[stride // 2: crop_h - stride // 2,
                    #                                                                       stride // 2: crop_w - stride // 2]

        pre_list.append(mask_whole[:h, :w] * 255)
        label_list.append(cm )
        print(cm.shape,mask_whole[:h, :w].shape)

    nochange_acc = tn / (tn + fp)
    change_acc = tp / (tp + fn)
    total_acc = (tn + tp) / (tn + fp + tp + fn)
    print(
        '\n Test: num_img:{:d}, Acc:{:.4f},nochg_acc:{:.4f},chg_acc:{:.4f},tn:{:d},tp:{:d},fn:{:d},fp:{:d}'
            .format(len(names), total_acc, nochange_acc, change_acc, tn, tp,
                    fn, fp))
    return pre_list, label_list, names

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    root_path = '/data/frb/python_project_frb/datasets/CD_Data_GZ/data/'
    crop_w = 256  # 裁剪图像宽度
    crop_h = 256  # 裁剪图像高度
    fnametest = 'testtnew.txt'#'trainnnew.txt' testtnew test2
    base_net = 'FCSiamDiff2bottleBN'#DeepLabori


    with open(root_path + fnametest, "r") as f:  # 打开文件
        datatest = f.read()  # 读取文件
    datatest = datatest.split('\n')
    namestest = datatest
    print('test names list:', namestest)
    train = False
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = models.Models.network_dict[base_net](3, 2)
    network = network.to(DEVICE)
    model_path = 'log/FCSiamDiff/20220602-11_07_CE_UCDNet_FCSiamDiff2_bottleBN/model/FCSiamDiff_95_acc-0.9426_chgAcc-0.8976_unchgAcc-0.9489.pth.tar'
    network.load_state_dict(torch.load(model_path))

    # model_path = './log/FCSiamDiffRes_GZ/20220525-20_47_SEPlus1_pow_resnet34/model/FCSiamDiffRes_GZ_15_acc-0.8989_chgAcc-0.6242_unchgAcc-0.9230.pth.tar'
    # model.load_state_dict(torch.load(model_path))
    ori_model_key = network.state_dict().keys()
    load_models = torch.load(model_path)
    part_sd = {k: v for k, v in load_models.items() if k in ori_model_key}
    # print('part_sd', len(part_sd.keys()), len(ori_model_key), len(load_models.keys()), part_sd.keys())
    network.state_dict().update(part_sd)


    stride = crop_w
    pre_list,label_list,names_list = predict(network, DEVICE, namestest, root_path, crop_w, stride)
    i = 0
    save_path = './log/testdisplay/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = open(save_path + 'test_wholePic_result.txt', 'w')
    filename.write(model_path + '\n')
    tp_e = 0
    tn_e = 0
    fp_e = 0
    fn_e = 0
    for name in names_list:

        if len(pre_list[i].shape) == 3:
            pr = pre_list[i][:, :, 0]
            # pr=np.expand_dims(pr,axis=2)
        else:
            pr = pre_list[i]
            # pr = np.expand_dims(pr, axis=2)
        if len(label_list[i].shape) == 3:
            gt = label_list[i][:, :, 0]
            # gt = np.expand_dims(gt, axis=2)
        else:
            gt = label_list[i]
            # gt = np.expand_dims(gt, axis=2)
            # gt=label_list[0]
        # print('pr',pr.shape,gt.shape)
        pr = pr.astype(np.uint8)
        gt = gt.astype(np.uint8)
        # cm = cm.astype(np.uint8)
        cv2.imwrite(save_path + 'pre_label' + '_' + str(crop_w) + '_' + name + '.png', np.concatenate([pr,gt],axis=1))
        # cv2.imwrite(save_path + 'pre' + '_' + str(crop_w) + '_' + name + '.png', pr)
        # cv2.imwrite(save_path + 'label' + '_' + str(crop_w) + '_' + name + '.png', gt)

        pr = (pr > 0)
        gt = (gt > 0)
        tp = np.logical_and(pr, gt).sum()
        tn = np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
        fp = np.logical_and(pr, np.logical_not(gt)).sum()
        fn = np.logical_and(np.logical_not(pr), gt).sum()
        tp_e += tp
        tn_e += tn
        fp_e += fp
        fn_e += fn
        accTotal=(tn+tp)/(tn+tp+fn+fp)
        nochange_acc = tn / (tn + fp)
        change_acc = tp / (tp + fn)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        dice = 2 * prec * rec / (prec + rec)
        f_meas = 2 * prec * rec / (prec + rec)
        filename.write(
            'Name:{} Acc:{:.4f},nochange_acc:{:.4f},change_acc:{:.4f},Prec:{:.4f},recall:{:.4f},Fmeasure:{:.4f},dice:{:.4f},tp:{:d},tn:{:d},fp:{:d},fn:{:d}'
            .format(name, accTotal, nochange_acc, change_acc, prec, rec, f_meas, dice, tp, tn, fp, fn) + '\n')
        i = i + 1
        print(
            'Name:{} Acc:{:.4f},nochange_acc:{:.4f},change_acc:{:.4f},Prec:{:.4f},recall:{:.4f},Fmeasure:{:.4f},dice:{:.4f},tp:{:d},tn:{:d},fp:{:d},fn:{:d}'
            .format(name, accTotal, nochange_acc, change_acc, prec, rec, f_meas, dice, tp, tn, fp, fn))
    tp=tp_e
    tn=tn_e
    fp=fp_e
    fn=fn_e
    nochange_acc = tn / (tn + fp)
    change_acc = tp / (tp + fn)
    total_acc = (tn + tp) / (tn + fp + tp + fn)
    prec = tp / (tp + fp + 1)
    rec = tp / (tp + fn + 1)
    f_meas = 2 * prec * rec / (prec + rec)
    print(
        '\n Total Result: total_num:{:d}, Acc:{:.4f},nochg_acc:{:.4f},chg_acc:{:.4f},rec:{:.4f},Fm:{:.4f},tn:{:d},tp:{:d},fn:{:d},fp:{:d}'
            .format(len(names_list), total_acc, nochange_acc, change_acc, rec, f_meas, tn, tp,
                    fn, fp))