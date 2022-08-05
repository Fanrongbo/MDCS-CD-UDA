
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import math
import os
import logging
import numpy as np
from torchvision import models
from models.appendix import ASSP,BaseModel,initialize_weights
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        #return summary(self, input_shape=(2, 3, 224, 224))

class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet34', pretrained=True,AG_flag=False):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 16:
            s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8:
            s3, s4, d3, d4 = (1, 1, 2, 4)

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)
        self.diff_ag2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff_ag3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff_ag4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff_se2 = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64 // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64 // 2, 64, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff_se3 = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(128, 128 // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128 // 2, 128, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff_se4 = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, 256 // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256 // 2, 256, kernel_size=1),
            nn.Sigmoid()
        )
        self.AG_flag=AG_flag
    ####18:
    # layer0  torch.Size([2, 64, 64, 64])
    # layer1  torch.Size([2, 64, 64, 64])
    # layer2  torch.Size([2, 128, 32, 32])
    # layer3  torch.Size([2, 256, 16, 16])  Total params: 2,782,784
    # layer4  torch.Size([2, 512, 16, 16])  Total params: 11,176,512
    ######34:
    # layer0   torch.Size([2, 64, 64, 64])
    # layer1   torch.Size([2, 64, 64, 64])
    # layer2    torch.Size([2, 128, 32, 32])
    # layer3   torch.Size([2, 256, 16, 16])  Total params: 8,170,304
    # layer4    torch.Size([2, 512, 16, 16]) Total params: 21,284,672
    def forward(self, x1, x2):
        x11 = self.layer0(x1)
        x12 = self.layer0(x2)
        ########stage 2
        x21 = self.layer1(x11)
        x22 = self.layer1(x12)
        diff2 = torch.abs(x21 - x22)


        if self.AG_flag:
            # diff2_AG = self.diff_ag2(diff2)
            # x21 = x21 * diff2_AG+ diff2
            # x22 = x22 * diff2_AG+ diff2
            diff2_SE= self.diff_se2(diff2)
            x21 = x21 + diff2_SE*(diff2)
            x22 = x22 + diff2_SE*(diff2)
        else:
            x21 = x21 + diff2
            x22 = x22 + diff2

        ########stage 3
        x31 = self.layer2(x21)
        x32 = self.layer2(x22)
        diff3 = torch.abs(x31 - x32)

        if self.AG_flag:
            # diff3_AG = self.diff_ag3(diff3)
            # x31 = x31 * diff3_AG+ diff3
            # x32 = x32 * diff3_AG+ diff3
            diff3_SE = self.diff_se3(diff3)
            x31 = x31 + diff3_SE * (diff3)
            x32 = x32 + diff3_SE * (diff3)
        else:
            x31 = x31 + diff3
            x32 = x32 + diff3
        ########stage 4
        x41 = self.layer3(x31)
        x42 = self.layer3(x32)
        diff4 = torch.abs(x41 - x42)

        if self.AG_flag:
            # diff4_AG = self.diff_ag4(diff4)
            # x41 = x41 * diff4_AG+ diff4
            # x42 = x42 * diff4_AG+ diff4
            diff4_SE = self.diff_se4(diff4)
            x41 = x41 + diff4_SE * (diff4)
            x42 = x42 + diff4_SE * (diff4)
        else:
            x41 = x41 + diff4
            x42 = x42 + diff4

        if self.AG_flag:
            return [x11, x21, x31, x41], [x12, x22, x32, x42],[diff2,diff3,diff4]
        else:
            return [x11, x21, x31, x41], [x12, x22, x32, x42]
#


class DecoderDA(nn.Module):
    def __init__(self, low_level_channels, num_classes,asspflag=True):
        super(DecoderDA, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        if asspflag:
            # Table 2, best performance with two 3x3 convs
            self.output = nn.Sequential(
                nn.Conv2d(48+256, 256, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(256, num_classes, 1, stride=1),
            )
        else:
            self.output = nn.Sequential(
                nn.Conv2d(48 + low_level_channels, 256, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(256, num_classes, 1, stride=1),
            )
        initialize_weights(self)

        self.low_ag = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(48, 24, kernel_size=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(24, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_ag = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256+48, (256+48)//2, kernel_size=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d((256+48)//2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.assp_ag = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256 // 2, kernel_size=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(256 // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x, low_level_features):
        # x_ag=self.assp_ag(x)
        # x=x*x_ag
        low_level_features = self.conv1(low_level_features)
        low_ag = self.low_ag(low_level_features)
        # low_level_features=low_level_features*low_ag
        # low_level_features = self.relu(self.bn1(low_level_features))


        H, W = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        cat_feature=torch.cat((low_level_features, x), dim=1)
        cat_feature_ag=self.out_ag(cat_feature)
        cat_feature=cat_feature*cat_feature_ag
        x = self.output(cat_feature)
        return x
class Decoder2(nn.Module):
    def __init__(self, low_level_channels, num_classes,asspflag=False):
        super(Decoder2, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        # Table 2, best performance with two 3x3 convs
        self.output = nn.Sequential(
            nn.Conv2d(128+512, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1, stride=1),
        )

        initialize_weights(self)

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        H, W = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x
class DeepLabAGDA(BaseModel):
    def __init__(self, in_channels=3,num_classes=2,  backbone='resnet', pretrained=True,
                 output_stride=16, freeze_bn=False, **_):

        super(DeepLabAGDA, self).__init__()
        assert ('xception' or 'resnet' in backbone)
        if 'resnet' in backbone:
            self.backbone = ResNet(in_channels=in_channels, output_stride=output_stride, pretrained=pretrained,AG_flag=True)
            # low_level_channels = 64
            # low_level_channels2=128
            low_level_channels = 512
            low_level_channels2 = 64
        # else:
        #     self.backbone = Xception(output_stride=output_stride, pretrained=pretrained)
        #     low_level_channels = 128
        twodecode=False
        if twodecode:
            self.decoder2 = Decoder2(128, 256, asspflag=False)
            self.ASSP = ASSP(in_channels=256, output_stride=output_stride)
            self.decoder = DecoderDA(64, num_classes,asspflag=True)
        else:
            self.ASSP = ASSP(in_channels=512, output_stride=output_stride)
            self.decoder = DecoderDA(64, num_classes, asspflag=True)


        if freeze_bn: self.freeze_bn()

    def forward(self, x1,x2):
        # x2=x1
        feature1,feature2,diff=self.backbone(x1,x2)
        H, W = x1.size(2), x1.size(3)
        # x1, low_level_features1 = self.backbone(x1)#[2, 256, 16, 16]
        # x2, low_level_features2 = self.backbone(x2)  # [2, 256, 16, 16]
        out_diff=torch.abs(feature1[-1]-feature2[-1])
        x=torch.cat([feature1[-1],feature2[-1]],dim=1)
        low_level_diff=torch.abs(feature1[1]-feature2[1])
        low_level_diff2=torch.abs(feature1[2]-feature2[2])
        # low_level_features=torch.cat([feature1[1],feature2[1]],dim=1)

        # x= self.decoder2(x,low_level_diff2)
        x = self.ASSP(x)
        x = self.decoder(x, low_level_diff)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        # print('out',x.shape)
        return x,x



#
# conv=DeepLab(2)
# conv = conv.to(DEVICE)
# print(summary(conv, (3, 256, 256), batch_size=1))

