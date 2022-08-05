from utils.dataloaderlow import getloader,getloaderDA


import torch
import torch.utils.data
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

import numpy as np
import os
import argparse as ag
import json
# from utils.helpers import get_loaders
from torch.autograd import Variable
# from models.transfer_models import Transfer_Net_DA
from tqdm import tqdm
# from IPython import display
import random
import openpyxl as op
# import warnings
# warnings.filterwarnings('ignore')
import shutil
import time
import models.Models
from IPython import display
import pickle

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()
def save_pickle(data, file_name):
    f = open(file_name, "wb")
    pickle.dump(data, f)
    f.close()


def load_pickle(file_name):
    f = open(file_name, "rb+")
    data = pickle.load(f)
    f.close()
    return data
class setFigure():
    # def __init__(self):
    def initialize_figure(self):
        self.metrics = {
            'nochange_acc': [],
            'change_acc': [],
            'prec': [],
            'rec': [],
            'f_meas': [],
            'clflossAvg': [],
            'transfer_lossAvg': [],
            'total_acc': [],
            'var': [],
            'marginLoss':[]
        }
        return self.metrics
    def set_figure(self,metric_dict, nochange_acc, change_acc, prec, rec, f_meas, clflossAvg, transfer_lossAvg, total_acc,var,marginLoss):
        metric_dict['nochange_acc'].append(nochange_acc)
        metric_dict['change_acc'].append(change_acc)
        metric_dict['prec'].append(prec)
        metric_dict['rec'].append(rec)
        metric_dict['f_meas'].append(f_meas)
        metric_dict['clflossAvg'].append(clflossAvg)
        metric_dict['transfer_lossAvg'].append(transfer_lossAvg)
        metric_dict['total_acc'].append(total_acc)
        metric_dict['var'].append(var)
        metric_dict['marginLoss'].append(marginLoss)

        return metric_dict

def add_weight_decay(net, l2_value, skip_list=()):
    # https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/

    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]

def get_parser_with_args(metadata_json='./utils/metadata_GZ.json'):
    parser = ag.ArgumentParser(description='Training change detection network')

    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)
        parser.set_defaults(**metadata)
        return parser, metadata
    return None
def plotFigure(figure_train_metrics,figure_test_metrics,num_epochs,name, time_now):
    t = np.linspace(1, num_epochs, num_epochs)
    e=num_epochs
    epoch_train_nochange_accuracy = figure_train_metrics['nochange_acc']
    epoch_train_change_accuracy = figure_train_metrics['change_acc']
    epoch_train_precision = figure_train_metrics['prec']
    epoch_train_recall = figure_train_metrics['rec']
    epoch_train_Fmeasure = figure_train_metrics['f_meas']
    epoch_train_clf_loss = figure_train_metrics['clflossAvg']
    epoch_train_trans_loss = figure_train_metrics['transfer_lossAvg']
    epoch_train_accuracy = figure_train_metrics['total_acc']
    epoch_train_var = figure_train_metrics['var']
    epoch_train_marginLoss = figure_train_metrics['marginLoss']

    epoch_test_nochange_accuracy = figure_test_metrics['nochange_acc']
    epoch_test_change_accuracy = figure_test_metrics['change_acc']
    epoch_test_accuracy = figure_test_metrics['total_acc']
    epoch_test_recall = figure_test_metrics['rec']
    epoch_test_Fmeasure = figure_test_metrics['f_meas']
    epoch_test_loss = figure_test_metrics['clflossAvg']
    epoch_test_precision = figure_test_metrics['prec']
    epoch_test_var = figure_test_metrics['var']
    epoch_test_marginLoss = figure_test_metrics['marginLoss']

    plt.figure(num=1)
    plt.clf()
    train_loss=[]
    test_loss=[]
    for i in range(len(epoch_train_clf_loss[:e + 1])):
        train_loss.append(epoch_train_clf_loss[:e + 1][i]+epoch_train_marginLoss[:e + 1][i])
        test_loss.append(epoch_test_loss[:e + 1][i]+epoch_test_marginLoss[:e + 1][i])

    l1_1, = plt.plot(t[:e + 1], epoch_train_clf_loss[:e + 1], label='Train class loss')
    l1_2, = plt.plot(t[:e + 1], epoch_test_loss[:e + 1], label='Test class loss')
    l1_3, = plt.plot(t[:e + 1], epoch_train_marginLoss[:e + 1], label='Train Margin loss')
    l1_4, = plt.plot(t[:e + 1], epoch_test_marginLoss[:e + 1], label='Test Margin loss')
    l1_5, = plt.plot(t[:e + 1], train_loss, label='Train Total loss')
    l1_6, = plt.plot(t[:e + 1], test_loss, label='Test Total loss')
    plt.legend(handles=[l1_1, l1_2, l1_3, l1_4, l1_5, l1_6])
    plt.grid()
    #         plt.gcf().gca().set_ylim(bottom = 0)
    plt.gcf().gca().set_xlim(left=0)
    plt.title('Loss')
    display.clear_output(wait=True)
    display.display(plt.gcf())

    plt.figure(num=2)
    plt.clf()
    l2_1, = plt.plot(t[:e + 1], epoch_train_accuracy[:e + 1], label='Train accuracy')
    l2_2, = plt.plot(t[:e + 1], epoch_test_accuracy[:e + 1], label='Test accuracy')
    plt.legend(handles=[l2_1, l2_2])
    plt.grid()
    plt.gcf().gca().set_ylim(0, 1)
    plt.title('Accuracy')
    display.clear_output(wait=True)
    display.display(plt.gcf())

    plt.figure(num=3)
    plt.clf()
    l3_1, = plt.plot(t[:e + 1], epoch_train_nochange_accuracy[:e + 1], label='Train accuracy: no change')
    l3_2, = plt.plot(t[:e + 1], epoch_train_change_accuracy[:e + 1], label='Train accuracy: change')
    l3_3, = plt.plot(t[:e + 1], epoch_test_nochange_accuracy[:e + 1], label='Test accuracy: no change')
    l3_4, = plt.plot(t[:e + 1], epoch_test_change_accuracy[:e + 1], label='Test accuracy: change')
    plt.legend(loc='best', handles=[l3_1, l3_2, l3_3, l3_4])
    plt.grid()
    plt.gcf().gca().set_ylim(0, 1)
    plt.title('Accuracy per class')
    display.clear_output(wait=True)
    display.display(plt.gcf())

    plt.figure(num=4)
    plt.clf()
    l4_1, = plt.plot(t[:e + 1], epoch_train_precision[:e + 1], label='Train precision')
    l4_2, = plt.plot(t[:e + 1], epoch_train_recall[:e + 1], label='Train recall')
    l4_3, = plt.plot(t[:e + 1], epoch_train_Fmeasure[:e + 1], label='Train Dice/F1')
    l4_4, = plt.plot(t[:e + 1], epoch_test_precision[:e + 1], label='Test precision')
    l4_5, = plt.plot(t[:e + 1], epoch_test_recall[:e + 1], label='Test recall')
    l4_6, = plt.plot(t[:e + 1], epoch_test_Fmeasure[:e + 1], label='Test Dice/F1')
    plt.legend(loc='best', handles=[l4_1, l4_2, l4_3, l4_4, l4_5, l4_6])
    plt.grid()
    plt.gcf().gca().set_ylim(0, 1)
    #         plt.gcf().gca().set_ylim(bottom = 0)
    #         plt.gcf().gca().set_xlim(left = 0)
    plt.title('Precision, Recall and F-measure')
    display.clear_output(wait=True)
    display.display(plt.gcf())

    plt.figure(num=5)
    plt.clf()
    l5_1, = plt.plot(t[:e + 1], epoch_train_var[:e + 1], label='Train Variance')
    l5_2, = plt.plot(t[:e + 1], epoch_test_var[:e + 1], label='Test Variance')
    plt.legend(loc='best', handles=[l5_1, l5_2])
    plt.grid()
    plt.gcf().gca().set_xlim(left=0)
    plt.title('Variance Curve')
    display.clear_output(wait=True)
    display.display(plt.gcf())

    plt.figure(num=6)
    plt.clf()
    l6_1, = plt.plot(t[:e + 1], epoch_train_trans_loss[:e + 1], label='transferLoss')
    plt.legend(loc='best', handles=[l6_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left=0)
    plt.title('transferLoss Curve')
    display.clear_output(wait=True)
    display.display(plt.gcf())


    save = True
    if save:
        plt.figure(num=1)
        plt.savefig('./log/%s/%s/%s-01-loss.png' % (name, time_now, name))

        plt.figure(num=2)
        plt.savefig('./log/%s/%s/%s-02-accuracy.png' % (name, time_now, name))

        plt.figure(num=3)
        plt.savefig('./log/%s/%s/%s-03-accuracy_per_class.png' % (name, time_now, name))

        plt.figure(num=4)
        plt.savefig('./log/%s/%s/%s-04-prec_rec_fmeas.png' % (name, time_now, name))

        plt.figure(num=5)
        plt.savefig('./log/%s/%s/%s-05-var.png' % (name, time_now, name))

        plt.figure(num=6)
        plt.savefig('./log/%s/%s/%s-06-trans.png' % (name, time_now, name))
def MakeRecordFloder(name, time_now,opt):
    os.makedirs('./log/{}/{}'.format(name, time_now))
    os.makedirs('./log/{}/{}/model'.format(name, time_now))
    filename = open('./log/{}/{}/'.format(name, time_now) + '_result.txt', 'w')
    print('path:', './log/{}/{}'.format(name, time_now))
    fout = open('./log/{}/{}/'.format(name, time_now) + '_result.json', 'w')
    json.dump('PATH_TO_DATASET: ' + opt.dataset_dir + '\\n', fout)
    filename.write('PATH_TO_DATASET: ' + opt.dataset_dir + '\n')

    shutil.copytree('./models/', './log/{}/{}/models'.format(name, time_now))
    shutil.copytree('./utils/', './log/{}/{}/utils'.format(name, time_now))
    # copy source file

    src_file = './utils/metrics.py'
    newtargetpath = './log/{}/{}/'.format(name, time_now) + 'metrics.py'
    shutil.copyfile(src_file, newtargetpath)
    src_file = json_file
    newtargetpath = './log/{}/{}/'.format(name, time_now) + 'json_file.json'
    shutil.copyfile(src_file, newtargetpath)
    src_file = './models/Models.py'
    newtargetpath = './log/{}/{}/'.format(name, time_now) + 'Models.py'
    shutil.copyfile(src_file, newtargetpath)
    filename_main = os.path.basename(__file__)
    src_file = './' + filename_main
    newtargetpath = './log/{}/{}/'.format(name, time_now) + filename_main
    shutil.copyfile(src_file, newtargetpath)
    print('\n#########make record floder successfully!###################')
    return filename
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

from utils.metrics import jaccard_loss, dice_loss,CD_MarginRankingLoss,CD_MarginRankingLossmultic,CD_Variancemultic\
    ,CD_MarginRankingVarLossvar,CD_MarginRankingVarLossvarsource,CD_MarginRankingVarLossvartarget

from utils.metrics_DA import MMD_loss,CORAL,MMD_lossclass,MMD_lossclass2,MMD_lossclass3


import math
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    base_net = 'backbone2BN_conATSA4'#DeepLabori
    num_epochs = 3
    json_file = './utils/metadata_SYSU_OSCD.json'
    lr = 0.0001
    weight=torch.tensor([1.0,1.0]).cuda()
    # # 如果 label = 0, 则权重为 1.0, label = 1, 则权重为 2.0。
    # criterion = torch.nn.CrossEntropyLoss(weight=weight)
    loss_fn = nn.CrossEntropyLoss()
    criterion_margine = CD_MarginRankingVarLossvar
    criterion = [loss_fn, CD_MarginRankingVarLossvartarget,CD_MarginRankingVarLossvarsource]
    transfer_fn=MMD_lossclass3()
    margin_value = .3
    margin_ratio = 10

    name = 'backbone2BN_conATSA4'
    time_strat=time.strftime("%Y%m%d-%H_%M", time.localtime())
    time_now = time.strftime("%Y%m%d-%H_%M_SYSU_OSCD_trans_marginsstt_stttssVra_max0all_FCSiamDiff", time.localtime())


    parser, metadata = get_parser_with_args(json_file)
    opt = parser.parse_args()
    print('\n##########Load Model#################')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # network = Transfer_Net_DA(num_class=2, transfer_loss=None, base_net=base_net).to(DEVICE)
    # network = DeepLab(3, 2)
    network = models.Models.network_dict[base_net](3, 2)
    # network=Transfer_Net_DAlow(base_net)
    network = network.to(DEVICE)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    print('\n########## Basic Info#################')
    print('torch:', torch.__version__)
    print('cuda:', torch.version.cuda)
    if False:
        for mouble_name, parameters in network.named_parameters():
            print(mouble_name, ':', parameters.size())

    print('\n########## Load Dataset#################')
    # source_loader, _, target_test_loader = get_loaders(opt)
    # source_loader, target_train_loader, target_test_loader = getloader(opt.dataset_dir,opt.batch_size)
    source_loader, target_train_loader, target_test_loader = getloaderDA(opt.dataset_dir, opt.target_dir,
                                                                         opt.batch_size)
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    len_target_test = len(target_test_loader)
    print('len_source_loader:', len_source_loader, 'len_target_loader:', len_target_loader,'len_target_test:',len_target_test)

    print('\n ########## Recording File Initialization#################')
    filename = MakeRecordFloder(name, time_now, opt)
    wb = op.Workbook()
    ws = wb['Sheet']
    ws.append(
        ['type', 'n_epochs', 'clfLoss', 'trans_loss', 'Acc', 'nochange_acc', 'change_acc', 'recall'
            , 'Fmeasure', 'TN', 'TP', 'FN', 'FP','val','marginLoss'])
    train_metrics = setFigure()
    test_metrics = setFigure()
    figure_train_metrics = train_metrics.initialize_figure()
    figure_test_metrics = test_metrics.initialize_figure()

    for epoch in range(num_epochs):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        loss_tr=0
        iter_source = iter(source_loader)
        iter_target = iter(target_train_loader)
        if len_source_loader>len_target_loader:
            tbar = tqdm(range(len_source_loader))
        else:
            tbar = tqdm(range(len_target_loader))

        network.train()
        margin_loss_1_total = 0
        var_total = 0
        var_loss_total = 0
        transfer_lossAvg=0
        margin_loss_1 = torch.tensor(0).cuda()

        dist = torch.tensor([0, 0]).cuda()
        var= torch.tensor(0).cuda()
        lambd = 4 / (1 + math.exp(-10 * (epoch+1) / num_epochs)) - 1  # epoch越大lambd越大
        # network.unfreeze_bn()
        for i in tbar:
            if (i+1) % len_target_loader== 0 and i !=0:
                iter_target = iter(target_train_loader)
            if (i+1) % len_source_loader== 0 and i !=0:
                iter_source = iter(source_loader)
            data_T1_source, data_T2_source, label_source = iter_source.next()
            data_T1_target, data_T2_target, label_target = iter_target.next()

            data_T1_source=Variable(data_T1_source).to(DEVICE)
            data_T2_source=Variable(data_T2_source).to(DEVICE)
            label_source = Variable(label_source.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))
            data_T1_target=Variable(data_T1_target).to(DEVICE)
            data_T2_target=Variable(data_T2_target).to(DEVICE)
            # label_target = Variable(label_target.type(torch.LongTensor)).cuda()  # (shape: (batch_size, img_h, img_w))


            if data_T1_source.shape[0]<2:
                break
            if  data_T1_source.shape[0] != data_T1_target.shape[0]:
                continue
            data_source = [data_T1_source, data_T2_source]
            data_target = [data_T1_target, data_T2_target]

            cd_preds, feature_source = network(data_source[0], data_source[1])
            loss_class = criterion[0](cd_preds[0], label_source)

            cd_predsTarget, feature_target = network(data_target[0], data_target[1])
            _, pred_target = torch.max(cd_predsTarget[0], 1)

            source_feature_select, target_feature_select, s_label_select, t_label_select = transfer_fn.select_feature_unban(
                feature_source, feature_target,
                label_source, pred_target,
                torch.nn.functional.softmax(cd_predsTarget[0],
                                            dim=1))

            transfer_loss2_c,var = transfer_fn.conditional(source_feature_select, target_feature_select,
                                                       s_label_select, t_label_select, DEVICE)
            transfer_loss = transfer_loss2_c
            loss2=lambd * transfer_loss / 10
            loss = loss_class + lambd * transfer_loss / 10

            # optimization step:
            optimizer.zero_grad()  # (reset gradients)
            loss.backward()  # (compute gradients)
            optimizer.step()  # (perform optimization step)

            # clf_loss = loss.data().cpu().numpy()
            clf_loss = loss_class.item()
            loss_tr = loss_tr + clf_loss
            transfer_loss_value = transfer_loss.item()
            transfer_lossAvg = transfer_lossAvg + transfer_loss_value
            margin_loss_1_total = margin_loss_1.item() / margin_ratio + margin_loss_1_total
            var_total = var_total + var.item()
            var_loss_total = var_loss_total + dist[1].item()

            _, pred = torch.max(cd_preds[0], 1)
            tp_e, tn_e, fp_e, fn_e = confuseMatrix(pred, label_source)
            tp += tp_e
            tn += tn_e
            fp += fp_e
            fn += fn_e
            nochange_acc = tn_e / (tn_e + fp_e + 1)
            change_acc = tp_e / (tp_e + fn_e + 1)
            acc = (tn_e + tp_e) / (tn_e + fp_e + tp_e + fn_e)
            tbar.set_description(
                "Train: epoch {:d} info var:{:.4f} | marginLoss:{:.4f} |clfLoss:{:.4f} | trLoss:{:.4f} | Acc:{:.4f} | NoChgacc:{:.4f} | Chgacc:{:.4f} | ".format(
                    epoch,var.item(),margin_loss_1.item()/margin_ratio,
                    clf_loss, transfer_loss_value,
                    acc,
                    nochange_acc,
                    change_acc))
            # sys.exit(0)
            if i>3:
                break
            # del data_T1_source, data_T2_source, label_source
        nochange_acc = tn / (tn + fp)
        change_acc = tp / (tp + fn)
        total_acc=(tn+tp)/(tn + fp+tp + fn)
        clflossAvg=loss_tr/i
        prec = tp / (tp + fp + 1)
        rec = tp / (tp + fn + 1)
        f_meas = 2 * prec * rec / (prec + rec)
        transfer_lossAvg=transfer_lossAvg/i

        print(
            '\n Train: n_epochs:{:d}, clfLoss:{:.4f},var:{:.4f},var_loss:{:.4f},marin_loss:{:.4f},trans_loss:{:.4f}, Acc:{:.4f},nochg_acc:{:.4f},chg_acc:{:.4f},rec:{:.4f},Fm:{:.4f},tn:{:d},tp:{:d},fn:{:d},fp:{:d}'
                .format(epoch, clflossAvg,var_total/i,var_loss_total/i,margin_loss_1_total / i, transfer_lossAvg, total_acc, nochange_acc, change_acc, rec, f_meas, tn, tp,
                        fn, fp))
        filename.write(
            'Train: n_epochs:{:d}, clfLoss:{:.4f},var:{:.4f},var_loss:{:.4f},marin_loss:{:.4f},trans_loss:{:.4f}, Acc:{:.4f},nochange_acc:{:.4f},change_acc:{:.4f},'
            'recall:{:.4f},Fmeasure:{:.4f},tn:{:d},tp:{:d},fn:{:d},fp:{:d}'
            .format(epoch, clflossAvg, var_total/i,var_loss_total/i,margin_loss_1_total / i, transfer_lossAvg, total_acc, nochange_acc, change_acc, rec ,f_meas, tn, tp, fn,
                    fp) + '\n')
        exel_out = 'train', epoch, clflossAvg, transfer_lossAvg, total_acc, nochange_acc, change_acc, rec, f_meas \
            , str(tn), str(tp), str(fn), str(fp),var_total/i,margin_loss_1_total / i
        ws.append(exel_out)
        figure_train_metrics = train_metrics.set_figure(figure_train_metrics, nochange_acc, change_acc, prec, rec,
                                                        f_meas, clflossAvg, transfer_lossAvg, total_acc,
                                                        var_total/i,margin_loss_1_total / i)
        scheduler.step()

        ############################################################################
        # val:
        ############################################################################
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        loss_val = 0
        margin_lossT = 0
        var_total = 0
        var_loss_total = 0
        tbar = tqdm(range(len_target_test))
        iter_test = iter(target_test_loader)
        network.eval()
        # margin_loss = torch.tensor(0).cuda()
        # dist = torch.tensor([0, 0]).cuda()
        # network.freeze_bn_dr()
        for i in tbar:
            with torch.no_grad():
                data_T1_target, data_T2_target, label_target = iter_test.next()
                # data_T1_target, data_T2_target, target_label = data_T1_target.to(DEVICE), data_T2_target.to(DEVICE), label_target.to(
                #     DEVICE).long()

                data_T1_target = Variable(data_T1_target).to(DEVICE)
                data_T2_target = Variable(data_T2_target).to(DEVICE)
                label_target = Variable(label_target.type(torch.LongTensor)).cuda()  # (shape: (batch_size, img_h, img_w))
                # s_output= network.predict([data_T1_target, data_T2_target])

                s_output,_ = network.forward(data_T1_target, data_T2_target)
                # network.eval()  # (set in evaluation mode, this affects BatchNorm and dropout)
                # loss = loss_fn(s_output, label_target)

                lossclass = criterion[0](s_output[0], label_target)
                margin_loss, dist = criterion[1](s_output[1], label_target,
                                                 torch.nn.functional.softmax(s_output[0],dim=1),
                                                 DEVICE, margin=margin_value)

                # loss_value = loss.data().cpu().numpy()
                loss_value=lossclass.item()
                loss_val=loss_val+loss_value
                transfer_loss=0

                var_total = var_total + dist[0].item()
                var_loss_total = var_loss_total + dist[1].item()
                margin_lossT = margin_lossT + margin_loss.item()/margin_ratio

                _, pred = torch.max(s_output[0], 1)



                tp_e, tn_e, fp_e, fn_e = confuseMatrix(pred, label_target)###notice name of variance
                tp += tp_e
                tn += tn_e
                fp += fp_e
                fn += fn_e
                nochange_acc = tn_e / (tn_e + fp_e + 1)
                change_acc = tp_e / (tp_e + fn_e + 1)
                acc = (tn_e + tp_e) / (tn_e + fp_e + tp_e + fn_e)
                tbar.set_description(
                    "Test: epoch {:d} info var:{:.4f} | margin_loss:{:.4f} | clfLoss:{:.4f} | transferLoss:{:.4f} | Acc:{:.4f} | NoChgacc:{:.4f} | Chgacc:{:.4f} | ".format(
                        epoch,dist[0].item(), margin_loss.item()/margin_ratio,
                        loss_value, transfer_loss,
                        acc,
                        nochange_acc,
                        change_acc))
                # del data_T1_target, data_T2_target,label_target
                # if i > 3:
                #     break
        nochange_acc = tn / (tn + fp)
        change_acc = tp / (tp + fn)
        total_acc = (tn + tp) / (tn + fp + tp + fn)
        clflossAvg = loss_val / i
        prec = tp / (tp + fp + 1)
        rec = tp / (tp + fn + 1)
        f_meas = 2 * prec * rec / (prec + rec)
        transfer_lossAvg = 0
        print(
            '\n Test: n_epochs:{:d},Var:{:.4f},margin_loss:{:.4f}, clfLoss:{:.4f},trans_loss:{:.4f}, Acc:{:.4f},nochg_acc:{:.4f},chg_acc:{:.4f},tn:{:d},tp:{:d},fn:{:d},fp:{:d}'
                .format(epoch, var_total/i,margin_lossT/i, clflossAvg, transfer_lossAvg, total_acc, nochange_acc, change_acc, tn, tp,
                        fn, fp))
        filename.write(
            'Test: n_epochs:{:d},Var:{:.4f},margin_loss:{:.4f},Loss:{:.4f} Acc:{:.4f},nochange_acc:{:.4f},change_acc:{:.4f},recall:{:.4f},Fmeasure:{:.4f},'
            'tn:{:d},tp:{:d},fn:{:d},fp:{:d}'
            .format(epoch, var_total/i,margin_lossT/i,clflossAvg, total_acc, nochange_acc, change_acc, rec, f_meas, tn, tp, fn,
                    fp) + '\n')
        exel_out = 'test', epoch, clflossAvg, transfer_lossAvg, total_acc, nochange_acc, change_acc, rec, f_meas \
            , str(tn), str(tp), str(fn), str(fp),var_total/i,margin_lossT/i
        ws.append(exel_out)
        figure_test_metrics = test_metrics.set_figure(figure_test_metrics, nochange_acc, change_acc, prec, rec,
                                                      f_meas, clflossAvg, transfer_lossAvg, total_acc,var_total/i,margin_lossT/i)
        save_str = './log/%s/%s/model/%s_%d_acc-%.4f_chgAcc-%.4f_unchgAcc-%.4f.pth.tar' \
                   % (name, time_now, name, epoch + 1, total_acc, change_acc, nochange_acc)
        torch.save(network.state_dict(), save_str)
        save_pickle(figure_train_metrics, "./log/%s/%s/fig_train.pkl" % (name, time_now))
        save_pickle(figure_test_metrics, "./log/%s/%s/fig_test.pkl" % (name, time_now))
    wb.save('./log/%s/%s/%s-log.xlsx' % (name, time_now, name))

    plotFigure(figure_train_metrics, figure_test_metrics, num_epochs, name, time_now)
    print('##########Training Completed!#################')
    time_end=time.strftime("%Y%m%d-%H_%M", time.localtime())
    print('Training Start Time:',time_strat,'  Training Completion Time:',time_end,'Total Epoch Num:',epoch)
    if scheduler:
        print('Training Start lr:', lr,'  Training Completion lr:',scheduler.get_last_lr())
        # scheduler.get_lr()
    print('saved path:', './log/{}/{}'.format(name, time_now))
