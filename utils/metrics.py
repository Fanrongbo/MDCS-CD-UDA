import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
class FocalLoss2(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True , reduce=True):
        super(FocalLoss2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)

            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)

            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))


        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def CD_MarginRankingLoss(dist, target, margin=.1):
    # loss1=0
    # loss2=0
    # # for x in zip(dist):
    # for i in range(dist.shape[1]):
    #     loss1=loss1+(1-target)*torch.pow(dist[:,i,:,:],2)
    #     zeros = torch.zeros_like(dist[:,i,:,:])
    #     margin_out=torch.pow(dist[:,i,:,:]-margin,2)
    #     margin_out = torch.where(margin_out > 0, margin_out, zeros)
    #     loss2=loss2+target*margin_out
    target=target.float()
    dist = torch.abs(dist)
    Max=True
    if Max and len(dist.shape)==4 and dist.shape[1]!=1:
        # print('dist', dist.shape)
        dist_max,_=torch.max(dist,dim=1,keepdim=True)
        dist_max=dist_max.float()
        dist_min, _ = torch.min(dist, dim=1, keepdim=True)
        dist_min = dist_min.float()
    # print('dist',dist.shape)
    dist=dist_max
    dist=dist.mean()
    zeros = torch.zeros_like(dist)
    margin_out = dist - margin
    margin_out = torch.where(margin_out > 0, margin_out, zeros)
    margin_out=torch.abs(margin_out).float()
    loss1 = (1 - target) * torch.exp(margin_out)
    loss1 = loss1.mean()

    dist=dist_min
    loss2 = (target * torch.exp(-10*dist)).mean()
    loss = loss1 + loss2
    # loss=loss2

    # loss1 = (1 - target) * torch.abs(dist)
    #
    # # loss1 = (1 - target) * (torch.exp(dist)-1)
    # # print('loss1',loss1.shape)
    # loss1=loss1.mean()
    # zeros = torch.zeros_like(dist)
    # margin_out=dist - margin
    # margin_out = torch.where(margin_out > 0, margin_out, zeros)
    # margin_out = torch.abs(margin_out)
    # # loss2 = (target * torch.log(margin_out+1)).mean()
    # loss2=(target * torch.exp(-margin_out)).mean()
    # loss = loss1 - loss2
    # # loss = loss.mean()

    return loss,[dist.mean(),dist.max(),dist.min()]
import torch
import torch.nn as nn
import torch.nn.functional as F
class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction
    def forward(self, logits, target):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)   # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))    # [NHW, C]
        target = target.view(-1, 1)    # [NHW，1]

        logits = F.log_softmax(logits, 1)
        logits = logits.gather(1, target)   # [NHW, 1]
        loss = -1 * logits

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class CCE(nn.Module):
    def __init__(self, device, balancing_factor=1):
        super(CCE, self).__init__()
        self.nll_loss = nn.NLLLoss()
        self.device = device # {'cpu', 'cuda:0', 'cuda:1', ...}
        self.balancing_factor = balancing_factor
        self.ce=CrossEntropyLoss()
    def forward(self, yHat, y):
        # Note: yHat.shape[1] <=> number of classes

        batch_size = len(y)
        # cross entropy
        cross_entropy = self.nll_loss(F.log_softmax(yHat, dim=1), y)
        # y_s=F.log_softmax(yHat, dim=1)
        # y_s2 = F.log_softmax((1-yHat), dim=1)
        # cross_entropy2=(y*(y_s)+(1-y)*(y_s2)).mean()
        # cross_entropy2=self.ce(yHat,y)
        # print('cross_entropy',cross_entropy,cross_entropy2)
        # complement entropy
        # print(yHat.shape,y.shape)
        # yHat=torch.flatten(yHat,start_dim=2,end_dim=3)
        # y=torch.flatten(y,start_dim=1,end_dim=2)
        yHat = F.softmax(yHat, dim=1)
        Yg = yHat.gather(dim=1, index=torch.unsqueeze(y, 1))
        # print(Yg.shape, y.shape,yHat.shape)

        Px = yHat / (1 - Yg+1e-7) + 1e-7
        Px_log = torch.log(Px + 1e-10)
        # y_zerohot = torch.ones(batch_size, yHat.shape[1]).scatter_(
        #     1, y.view(batch_size, 1).data.cpu(), 0)
        y_zerohot = torch.ones_like(yHat).scatter_(1, y.unsqueeze(1), 0)
        output = Px * Px_log * y_zerohot.to(device=self.device)
        complement_entropy = torch.sum(output) / (float(batch_size) * float(yHat.shape[1]))

        return cross_entropy - self.balancing_factor * complement_entropy
def CD_MarginRankingLossmultic(dist, target, margin=.3):
    dist=dist.mean([1])
    # print('target',target.shape,dist.shape)
    dist=dist.unsqueeze(1)
    target=target.unsqueeze(1)
    target=(target.repeat([1,dist.shape[1],1,1])).float()
    # print('target',target.shape,dist.shape)
    dist=torch.pow(dist,2)

    zeros = torch.zeros_like(dist)
    margin_out = dist - margin
    margin_out = torch.where(margin_out > 0, margin_out, zeros).float()
    loss1 = (1 - target) * torch.abs(margin_out)
    loss1 = loss1.mean()

    # zeros = torch.zeros_like(dist)
    # margin_out = torch.exp(-1*dist)
    margin_out=1/(dist+1).float()
    # margin_out = torch.where(dist > 0.5, zeros, margin_out).float()
    # loss2 = (target * (margin_out-1/0.5)).mean()
    loss2 = (target * (margin_out)).mean()


    loss = loss1 + loss2
    # loss=loss2

    return loss,[dist.mean(),dist.max(),dist.min()]
def CD_MarginRankingVarLossmultic(dist,distag, target, margin=.3):
    margin=0.1
    dist=dist.mean([1])
    # print('target',target.shape,dist.shape)
    dist=dist.unsqueeze(1)
    target=target.unsqueeze(1)
    target=(target.repeat([1,dist.shape[1],1,1])).float()
    # print('target',target.shape,dist.shape)


    zeros = torch.zeros_like(dist)
    margin_out = dist - margin
    margin_out = torch.where(margin_out > 0, margin_out, zeros).float()
    loss1 = (1 - target) * torch.abs(margin_out)
    loss1 = loss1.mean()

    dist = torch.pow(dist, 2)
    # zeros = torch.zeros_like(dist)
    # margin_out = torch.exp(-1*dist)
    margin_out=1/(dist+1).float()
    # margin_out = torch.where(dist > 0.5, zeros, margin_out).float()
    # loss2 = (target * (margin_out-1/0.5)).mean()

    var=distag.var()
    loss2 = (((target+0.1) * (margin_out)).mean())/(var.mean())

    loss = 0.01*loss1 + loss2
    # loss=loss2

    return loss,[dist.mean(),dist.max(),dist.min()]

def CD_MarginRankingVarLossvar(dist, target,device, margin=.3):
    # margin=0.1
    target_ori=target

    H, W = target.size(1), target.size(2)
    # dist=dist.mean([1])
    # print('target',target.shape,dist.shape)
    # dist=dist.unsqueeze(1)
    c=dist.size(1)
    #1
    avg_out = torch.mean(dist[:,:c//2,:,:], dim=1, keepdim=True)
    max_out, _ = torch.max(dist[:,:c//2,:,:], dim=1, keepdim=True)
    # min_out, _ = torch.min(dist[:,:c//2,:,:], dim=1, keepdim=True)
    #2
    avg_out1 = torch.mean(dist[:,c//2:,:,:], dim=1, keepdim=True)
    max_out1, _ = torch.max(dist[:,c//2:,:,:], dim=1, keepdim=True)
    # min_out1, _ = torch.min(dist[:,c//2:,:,:], dim=1, keepdim=True)
    # #3
    # avg_out2 = torch.mean(dist[:, c // 2:3*c//4, :, :], dim=1, keepdim=True)
    # max_out2, _ = torch.max(dist[:, c // 2:3*c//4, :, :], dim=1, keepdim=True)
    # min_out2, _ = torch.min(dist[:, c // 2:3*c//4, :, :], dim=1, keepdim=True)
    # #4
    # avg_out3 = torch.mean(dist[:, 3*c//4:, :, :], dim=1, keepdim=True)
    # max_out3, _ = torch.max(dist[:, 3*c//4:, :, :], dim=1, keepdim=True)
    # min_out3, _ = torch.min(dist[:, 3*c//4:, :, :], dim=1, keepdim=True)
    # 4
    avg_out4 = torch.mean(dist, dim=1, keepdim=True)
    max_out4, _ = torch.max(dist, dim=1, keepdim=True)
    # min_out4, _ = torch.min(dist, dim=1, keepdim=True)
    #
    # # dist=torch.cat([avg_out,max_out,min_out],1)
    # dist=torch.cat([avg_out,max_out,min_out,avg_out1,max_out1,min_out1],1)
    # dist = torch.cat([avg_out, max_out, min_out, avg_out1, max_out1, min_out1,
    #                   avg_out2, max_out2, min_out2,avg_out3, max_out3, min_out3,
    #                   avg_out4, max_out4, min_out4], 1)
    dist = torch.cat([avg_out, max_out, avg_out1, max_out1, avg_out4, max_out4], 1)
    dist=torch.pow(dist,2)
    # dist=dist/(dist.mean([2,3]).view(dist.shape[0],dist.shape[1],1,1)+0.00001)
    max_dis,_=torch.max(dist,dim=1)
    # print('max_dis',max_dis.shape)
    dist=dist/(max_dis.unsqueeze(1)+0.00001)

    # print(dist.mean([1]).shape)
    # print(dist)

    # var = dist.var([2, 3])  # b,2
    var=torch.var(dist,dim=[2,3])
    # print(var.shape,var)
    dist = F.interpolate(dist, size=(H, W), mode='bilinear', align_corners=True)
    target=target.unsqueeze(1)
    target=(target.repeat([1,dist.shape[1],1,1])).float()
    zeros = torch.zeros_like(dist)
    ones = torch.ones_like(dist)
    margin_out_sim_out = dist - margin
    margin_out_sim = torch.where(margin_out_sim_out > 0, margin_out_sim_out, zeros).float()

    var_mean=torch.mean(torch.log(var+1),dim=1,keepdim=True).view(var.shape[0],1,1,1)
    margin_out_sim_flag = torch.where(margin_out_sim_out > 0, ones, zeros).float()
    unchgnum = (margin_out_sim_flag * (1 - target)).sum()+1
    # loss1 = (1 - target) * (torch.exp(margin_out_sim)-1)/(var_mean+0.001)#unchg
    loss1 = (1 - target) * (torch.exp(margin_out_sim) - 1)
    # unchgnum = len(torch.nonzero(loss1))+ 1
    # unchgnum = (1 - target).sum() + 1
    loss1 = loss1.sum()/unchgnum


    lossdist = ((target) * (torch.exp(-dist)))
    chgnum=target.sum()+1
    lossdist=lossdist.sum()/ chgnum

    # loss = unchgnum/(chgnum+unchgnum)*loss1 + chgnum/(chgnum+unchgnum)*lossdist#+(1/(var_mean+0.001)).mean()
    loss = loss1+lossdist -var_mean#+(1/(var_mean+0.001)).mean()

    # loss=lossvar
    # loss=loss2
    var_mean = var.mean()

    return loss,[var_mean,loss1.mean()]
def CD_MarginRankingVarLossvartarget(dist, target,targetfeat,device, margin=.3):
    # margin=0.1
    target_ori=target

    H, W = target.size(1), target.size(2)
    # dist=dist.mean([1])
    # print('target',target.shape,dist.shape)
    # dist=dist.unsqueeze(1)
    c=dist.size(1)
    #1
    avg_out = torch.mean(dist[:,:c//2,:,:], dim=1, keepdim=True)
    max_out, _ = torch.max(dist[:,:c//2,:,:], dim=1, keepdim=True)
    # min_out, _ = torch.min(dist[:,:c//2,:,:], dim=1, keepdim=True)
    #2
    avg_out1 = torch.mean(dist[:,c//2:,:,:], dim=1, keepdim=True)
    max_out1, _ = torch.max(dist[:,c//2:,:,:], dim=1, keepdim=True)
    # min_out1, _ = torch.min(dist[:,c//2:,:,:], dim=1, keepdim=True)
    # 4
    avg_out4 = torch.mean(dist, dim=1, keepdim=True)
    max_out4, _ = torch.max(dist, dim=1, keepdim=True)

    dist = torch.cat([avg_out, max_out, avg_out1, max_out1, avg_out4, max_out4], 1)
    dist = torch.pow(dist,2)
    # dist=dist/(dist.mean([2,3]).view(dist.shape[0],dist.shape[1],1,1)+0.00001)
    max_dis,_ = torch.max(dist,dim=1)
    dist = dist/(max_dis.unsqueeze(1)+0.00001)

    var=torch.var(dist,dim=[2,3])
    # print(var.shape,var)
    dist = F.interpolate(dist, size=(H, W), mode='bilinear', align_corners=True)
    target = target.unsqueeze(1)
    target = (target.repeat([1,dist.shape[1],1,1])).float()
    zeros = torch.zeros_like(dist)
    ones = torch.ones_like(dist)
    margin_out_sim_out = dist - margin
    margin_out_sim = torch.where(margin_out_sim_out > 0, margin_out_sim_out, zeros).float()

    var_mean = torch.mean(torch.log(var+1),dim=1,keepdim=True).view(var.shape[0],1,1,1)
    margin_out_sim_flag = torch.where(margin_out_sim_out > 0, ones, zeros).float()
    unchgnum = (margin_out_sim_flag * (1 - target)).sum()+1
    # # loss1 = (1 - target) * (torch.exp(margin_out_sim)-1)/(var_mean+0.001)#unchg
    loss1 = targetfeat[:,0,:,:]*(1 - target) * (torch.exp(margin_out_sim) - 1)
    loss1 = 0.3*loss1.sum()/unchgnum


    lossdist = (targetfeat[:,1,:,:]*(target) * (torch.exp(-dist)))
    chgnum = target.sum()+1
    unchgnum_l = (1 - target).sum()
    lossdist = unchgnum_l/(unchgnum_l+chgnum)*lossdist.sum()/chgnum

    # loss = unchgnum/(chgnum+unchgnum)*loss1 + chgnum/(chgnum+unchgnum)*lossdist#+(1/(var_mean+0.001)).mean()
    # loss = loss1-lossdist-var_mean.mean() #+(1/(var_mean+0.001)).mean()
    loss = loss1+lossdist
    # loss=lossvar
    # loss=loss2
    var_mean = var.mean()

    return loss,[var_mean,var_mean]
def CD_MarginRankingVarLossvarsource(dist, target,device, margin=.3):
    # margin=0.1
    target_ori=target

    H, W = target.size(1), target.size(2)
    # dist=dist.mean([1])
    # print('target',target.shape,dist.shape)
    # dist=dist.unsqueeze(1)
    c=dist.size(1)
    #1
    avg_out = torch.mean(dist[:,:c//2,:,:], dim=1, keepdim=True)
    max_out, _ = torch.max(dist[:,:c//2,:,:], dim=1, keepdim=True)
    # min_out, _ = torch.min(dist[:,:c//2,:,:], dim=1, keepdim=True)
    #2
    avg_out1 = torch.mean(dist[:,c//2:,:,:], dim=1, keepdim=True)
    max_out1, _ = torch.max(dist[:,c//2:,:,:], dim=1, keepdim=True)

    avg_out4 = torch.mean(dist, dim=1, keepdim=True)
    max_out4, _ = torch.max(dist, dim=1, keepdim=True)

    dist = torch.cat([avg_out, max_out, avg_out1, max_out1, avg_out4, max_out4], 1)

    # dist=dist/(dist.mean([2,3]).view(dist.shape[0],dist.shape[1],1,1)+0.00001)
    max_dis,_=torch.max(dist,dim=1)
    # print('max_dis',max_dis.shape)
    dist=dist/(max_dis.unsqueeze(1)+0.00001)
    dist = torch.pow(dist, 2)

    var=torch.var(dist,dim=[2,3])
    # print(var.shape,var)
    dist = F.interpolate(dist, size=(H, W), mode='bilinear', align_corners=True)
    target=target.unsqueeze(1)
    target=(target.repeat([1,dist.shape[1],1,1])).float()
    #uchg
    zeros = torch.zeros_like(dist)
    ones = torch.ones_like(dist)
    margin_out_sim_out = dist - margin
    margin_out_sim = torch.where(margin_out_sim_out > 0, margin_out_sim_out, zeros).float()

    var_mean=torch.mean(torch.log(var+1),dim=1,keepdim=True).view(var.shape[0],1,1,1)
    margin_out_sim_flag = torch.where(margin_out_sim_out > 0, ones, zeros).float()
    unchgnum = (margin_out_sim_flag * (1 - target)).sum()+1
    # loss1 = (1 - target) * (torch.exp(margin_out_sim)-1)/(var_mean+0.001)#unchg
    loss1 = (1 - target) * (torch.exp(margin_out_sim) - 1)
    loss1 = loss1.sum()/unchgnum
#chg
    lossdist = ((target) * (torch.exp(-dist)))
    chgnum=target.sum()+1
    lossdist=lossdist.sum()/ chgnum
    loss = loss1+lossdist #+(1/(var_mean+0.001)).mean()

    # var_mean = var.mean()

    return loss,[lossdist.mean(),lossdist.mean()]

def CD_Marginvar(dist, target,device, margin=.3):
    # margin=0.1

    H, W = target.size(1), target.size(2)
    c=dist.size(1)
    #1
    avg_out = torch.mean(dist[:,:c//2,:,:], dim=1, keepdim=True)
    max_out, _ = torch.max(dist[:,:c//2,:,:], dim=1, keepdim=True)
    #2
    avg_out1 = torch.mean(dist[:,c//2:,:,:], dim=1, keepdim=True)
    max_out1, _ = torch.max(dist[:,c//2:,:,:], dim=1, keepdim=True)

    avg_out4 = torch.mean(dist, dim=1, keepdim=True)
    max_out4, _ = torch.max(dist, dim=1, keepdim=True)

    dist = torch.cat([avg_out, max_out, avg_out1, max_out1, avg_out4, max_out4], 1)
    dist=torch.pow(dist,2)
    # dist=dist/(dist.mean([2,3]).view(dist.shape[0],dist.shape[1],1,1)+0.00001)
    dist=dist/(dist.max([1]).unsqueeze(1)+0.00001)

    var=torch.var(dist,dim=[2,3])

    # target=target.unsqueeze(1)

    var_mean=torch.mean(torch.log(var+1),dim=1,keepdim=True).view(var.shape[0],1,1,1)
    # unchgnum = (1 - target).sum()+1

    # loss1= (1/(var_mean+0.001))*(unchgnum/(target.size(0)*target.size(1)*target.size(2)))
    loss1 = (1 / (var_mean + 0.001))
    loss1 = loss1.sum()


    # lossdist = ((target) * (torch.exp(-dist)))
    # # lossdist = ((target) * (1/(dist+0.1)))
    #
    # # chgnum = len(torch.nonzero(lossdist)) + 1
    # chgnum=target.sum()+1
    # lossdist=lossdist.sum()/ chgnum

    # loss = unchgnum/(chgnum+unchgnum)*loss1 + chgnum/(chgnum+unchgnum)*lossdist#+(1/(var_mean+0.001)).mean()
    loss = loss1 #+(1/(var_mean+0.001)).mean()

    # loss=lossvar
    # loss=loss2
    var_mean = var.mean()

    return loss,[var_mean,loss1.mean()]
def CD_Variancemultic(dist, target, margin=.1):
    target = target.float()
    dist = torch.abs(dist)
    Max = True
    if Max and len(dist.shape) == 4 and dist.shape[1] != 1:
        # print('dist', dist.shape)
        dist_max, _ = torch.max(dist, dim=1, keepdim=True)
        dist_max = dist_max.float()
        # dist_min, _ = torch.min(dist, dim=1, keepdim=True)
        # dist_min = dist_min.float()
    # print('dist',dist.shape)
    dist = dist_max
    zeros = torch.zeros_like(dist)
    margin_out = dist - margin



    # margin_out = torch.where(margin_out > 0, margin_out, zeros)
    # margin_out = torch.abs(margin_out).float()
    # loss1 = (1 - target) * torch.exp(margin_out)
    # loss1 = loss1.mean()
    #
    # dist = dist_min
    # loss2 = (target * torch.exp(-10 * dist)).mean()
    # loss = loss1 + loss2


    # print('target',target.shape)
    # target=target.unsqueeze(1)
    # target=(target.repeat([1,dist.shape[1],1,1])).float()
    # # print('target',target.shape,dist.shape)
    # dist=torch.abs(dist)
    #
    # zeros = torch.zeros_like(dist)
    # margin_out = dist - margin
    # margin_out = torch.where(margin_out > 0, margin_out, zeros).float()
    # loss1 = (1 - target) * torch.abs(margin_out)
    # loss1 = loss1.mean()
    #
    # zeros = torch.zeros_like(dist)
    # margin_out = torch.exp(-1*dist)
    # margin_out = torch.where(dist > 1, zeros, margin_out).float()
    # loss2 = (target * margin_out).mean()

    loss = 0
    # loss=loss2

    return loss,[dist.mean(),dist.max(),dist.min()]

def dice_loss(logits, true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        # print('true_1_hot',true_1_hot,true_1_hot.shape)
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def jaccard_loss(logits, true, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, eps=1e-7, size_average=True):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.size_average = size_average
        self.eps = eps

    def forward(self, logits, true):
        """Computes the Tversky loss [1].
        Args:
            true: a tensor of shape [B, H, W] or [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            alpha: controls the penalty for false positives.
            beta: controls the penalty for false negatives.
            eps: added to the denominator for numerical stability.
        Returns:
            tversky_loss: the Tversky loss.
        Notes:
            alpha = beta = 0.5 => dice coeff
            alpha = beta = 1 => tanimoto coeff
            alpha + beta = 1 => F beta coeff
        References:
            [1]: https://arxiv.org/abs/1706.05721
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)

        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        fps = torch.sum(probas * (1 - true_1_hot), dims)
        fns = torch.sum((1 - probas) * true_1_hot, dims)
        num = intersection
        denom = intersection + (self.alpha * fps) + (self.beta * fns)
        tversky_loss = (num / (denom + self.eps)).mean()
        return (1 - tversky_loss)
