import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DeepWise_PointWise_Conv(nn.Module):
    def __init__(self, in_ch, out_ch,s):
        super(DeepWise_PointWise_Conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=s,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

#输入的feature_map是经过主干网卷积后的结果�?
class ASSP(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ASSP,self).__init__()
        dilations=[1,6,12,18]

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilations[0],#cannot set kernalsize to 3,otherwise cannot converge
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[1], dilation=dilations[1],
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[2], dilation=dilations[2],
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[3], dilation=dilations[3],
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU6()
        )
    def forward(self,x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x)
        x5 = self.layer5(x)
        # print(x5.shape)
        x5=F.upsample(x5,size=x1.size()[2:],mode='bilinear',align_corners=True)
        x=torch.cat((x1,x2,x3,x4,x5),dim=1)
        # print('x',x1.shape,x2.shape,x3.shape,x4.shape,x5.shape,)
        x=self.layer6(x)

        return x
class ASSPnoBN(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ASSPnoBN,self).__init__()
        dilations=[1,6,12,18]

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilations[0],#cannot set kernalsize to 3,otherwise cannot converge
                      bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[1], dilation=dilations[1],
                      bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[2], dilation=dilations[2],
                      bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[3], dilation=dilations[3],
                      bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=3, stride=1,padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU6()
        )
    def forward(self,x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x)
        x5 = self.layer5(x)
        # print(x5.shape)
        x5=F.upsample(x5,size=x1.size()[2:],mode='bilinear',align_corners=True)
        x=torch.cat((x1,x2,x3,x4,x5),dim=1)
        # print('x',x1.shape,x2.shape,x3.shape,x4.shape,x5.shape,)
        x=self.layer6(x)

        return x
class NSSP(nn.Module):
    def __init__(self, in_channels, out_channels,size):
        super(NSSP, self).__init__()
        h = size
        self.pooling_B2_down = DeepWise_PointWise_Conv(in_ch=in_channels, out_ch=in_channels // 4, s=2)
        self.pooling_B2_avg = nn.AvgPool2d(kernel_size=2)
        self.pooling_B2_conv = nn.Conv2d(in_channels=in_channels , out_channels=in_channels // 4, kernel_size=1,
                                         padding=0)

        self.pooling_B4_down = DeepWise_PointWise_Conv(in_ch=in_channels, out_ch=in_channels // 4, s=4)
        self.pooling_B4_avg = nn.AvgPool2d(kernel_size=4)
        self.pooling_B4_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1,
                                         padding=0)

        self.pooling_B8_down = DeepWise_PointWise_Conv(in_ch=in_channels, out_ch=in_channels // 4, s=8)
        self.pooling_B8_avg = nn.AvgPool2d(kernel_size=8)
        self.pooling_B8_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1,
                                         padding=0)

        self.pooling_B16_down = DeepWise_PointWise_Conv(in_ch=in_channels, out_ch=in_channels // 4, s=16)
        self.pooling_B16_avg = nn.AvgPool2d(kernel_size=16)
        self.pooling_B16_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1,
                                          padding=0)

        self.global_pooling_B2_conv = nn.Conv2d(in_channels=in_channels // 4, out_channels=in_channels // 4,
                                                kernel_size=1,
                                                padding=0)
        self.global_pooling_B2_up = nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=in_channels // 4,
                                                       kernel_size=3,
                                                       padding=0, stride=h, output_padding=h-3)

        self.global_pooling_B4_conv = nn.Conv2d(in_channels=in_channels // 4, out_channels=in_channels // 4,
                                                kernel_size=1,
                                                padding=0)
        self.global_pooling_B4_up = nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=in_channels // 4,
                                                       kernel_size=3,
                                                       padding=0, stride=h, output_padding=h-3)

        self.global_pooling_B8_conv = nn.Conv2d(in_channels=in_channels // 4, out_channels=in_channels // 4,
                                                kernel_size=1,
                                                padding=0)
        self.global_pooling_B8_up = nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=in_channels // 4,
                                                       kernel_size=3,
                                                       padding=0, stride=h, output_padding=h-3)
        self.global_pooling_B16_conv = nn.Conv2d(in_channels=in_channels // 4, out_channels=in_channels // 4,
                                                 kernel_size=1,
                                                 padding=0)
        self.global_pooling_B16_up = nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=in_channels // 4,
                                                       kernel_size=3,
                                                       padding=0, stride=h, output_padding=h-3)
        self.conv=nn.Conv2d(in_channels=in_channels*2,out_channels=out_channels ,kernel_size=1,padding=0)
    def forward(self, x):
        PB1_1 = self.pooling_B2_down(x)
        PB1_2 = self.pooling_B2_avg(x)
        PB1_2 = self.pooling_B2_conv(PB1_2)
        PB1 = PB1_1 + PB1_2
        GP1_1=PB1.mean(dim=[2,3])
        GP1_1=GP1_1.reshape(GP1_1.shape[0],GP1_1.shape[1],1,1)
        GP1_2=self.global_pooling_B2_conv(GP1_1)
        GP1_3=self.global_pooling_B2_up(GP1_2)

        PB2_1 = self.pooling_B4_down(x)
        PB2_2 = self.pooling_B4_avg(x)
        PB2_2 = self.pooling_B4_conv(PB2_2)
        PB2 = PB2_1 + PB2_2
        GP2_1 = PB2.mean(dim=[2, 3])
        GP2_1 = GP2_1.reshape(GP2_1.shape[0], GP2_1.shape[1], 1, 1)
        GP2_2 = self.global_pooling_B4_conv(GP2_1)
        GP2_3 = self.global_pooling_B4_up(GP2_2)

        PB3_1 = self.pooling_B8_down(x)
        PB3_2 = self.pooling_B8_avg(x)
        PB3_2 = self.pooling_B8_conv(PB3_2)
        PB3 = PB3_1 + PB3_2
        GP3_1 = PB3.mean(dim=[2, 3])
        GP3_1 = GP3_1.reshape(GP3_1.shape[0], GP3_1.shape[1], 1, 1)
        GP3_2 = self.global_pooling_B8_conv(GP3_1)
        GP3_3 = self.global_pooling_B8_up(GP3_2)

        PB4_1 = self.pooling_B16_down(x)
        PB4_2 = self.pooling_B16_avg(x)
        PB4_2 = self.pooling_B16_conv(PB4_2)
        PB4 = PB4_1 + PB4_2
        GP4_1 = PB4.mean(dim=[2, 3])
        GP4_1 = GP4_1.reshape(GP4_1.shape[0], GP4_1.shape[1], 1, 1)
        GP4_2 = self.global_pooling_B16_conv(GP4_1)
        GP4_3 = self.global_pooling_B16_up(GP4_2)

        GP=torch.cat([GP1_3,GP2_3,GP3_3,GP4_3,x],1)
        out=self.conv(GP)

        return out

class ConvBlock(nn.Module):
    def __init__(self,dim_in,dim_feature):
        super(ConvBlock, self).__init__()
        self.conv=nn.Conv2d(dim_in, dim_feature, kernel_size=7, padding=3,stride=1, bias=False)
        self.bn = nn.BatchNorm2d(dim_feature)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x=self.conv(x)
        # x=self.bn(x)
        x=self.relu(x)
        # x=self.max_pool(x)
        return x

class UCDNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UCDNet, self).__init__()
        hideen_num = [16, 32, 64, 128]
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            # nn.ReLU()
        )
        self.conv_1_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            # nn.ReLU()
        )
        self.conv_2_2 = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            # nn.ReLU()
        )
        self.conv_3_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4_1 = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU()
        )
        self.conv_4_2 = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            # nn.ReLU()
        )

        self.conv_diff_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        # self.diff1_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(16, 16, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 16, kernel_size=1),
        #     nn.Sigmoid()
        # )


        self.conv_diff_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # self.diff2_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(32, 32 , kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32 , 32, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.conv_diff_3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # self.diff3_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.conv_diff_4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # self.diff4_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(128, 128, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.conv_diff_5 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # self.diff5_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.Sigmoid()
        # )
        # self.NSSP = NSSP(192, 192, 32)
        self.ASSP = ASSP(192, 192)
        self.ASSPconv = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu=nn.ReLU()
    def forward(self, pre_data,post_data):
        #####################
        # decoder
        #####################
        # post_data=pre_data
        feature_T1_11 = self.conv_1_1(pre_data)#conv
        feature_T1_12 = self.conv_1_2(feature_T1_11)#relu+conv+relu
        feature_T2_11 = self.conv_1_1(post_data)
        feature_T2_12 = self.conv_1_2(feature_T2_11)

        diff1 = torch.abs(feature_T1_11 - feature_T2_11)
        diff1 = self.conv_diff_1(diff1)
        # diff1_weight=diff1*self.diff1_se(diff1)
        # diff1_weight=self.relu(diff1)
        feature_T1_13 = torch.cat([feature_T1_12, diff1], 1)
        feature_T1_14 = self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12, diff1], 1)
        feature_T2_14 = self.max_pool_1(feature_T2_13)

        #Stage 2

        feature_T1_21 = self.conv_2_1(feature_T1_14)#conv
        feature_T1_22 = self.conv_2_2(feature_T1_21)#relu+conv+relu
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        feature_T2_22 = self.conv_2_2(feature_T2_21)

        diff2 = torch.abs(feature_T1_21 - feature_T2_21)
        diff2 = self.conv_diff_2(diff2)
        # diff2_weight = diff2 * self.diff2_se(diff2)
        # diff2_weight = self.relu(diff2)
        feature_T1_23 = torch.cat([feature_T1_22, diff2], 1)
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22, diff2], 1)
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        #Stage 3

        feature_T1_31 = self.conv_3_1(feature_T1_24)#conv
        feature_T1_32 = self.conv_3_2(feature_T1_31)#relu+conv+relu
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        feature_T2_32 = self.conv_3_2(feature_T2_31)

        diff3 = torch.abs(feature_T1_31 - feature_T2_31)
        diff3 = self.conv_diff_3(diff3)
        # diff3_weight = diff3 * self.diff3_se(diff3)
        diff3_weight = self.relu(diff3)

        feature_T1_33 = torch.cat([feature_T1_32, diff3_weight], 1)
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32, diff3_weight], 1)  # 128
        feature_T2_34 = self.max_pool_3(feature_T2_33)
        #stage 4
        feature_T1_41 = self.conv_4_1(feature_T1_34)#conv
        feature_T1_42 = self.conv_4_2(feature_T1_41)#relu+conv+relu
        feature_T2_41 = self.conv_4_1(feature_T2_34)
        feature_T2_42 = self.conv_4_2(feature_T2_41)

        diff4 = torch.abs(feature_T1_41 - feature_T2_41)
        diff4 = self.conv_diff_4(diff4)
        # diff4_weight = diff4 * self.diff4_se(diff4)
        diff4_weight = self.relu(diff4)

        feature_T1_43 = torch.cat([feature_T1_42, diff4_weight], 1)
        feature_T1_44 = self.conv_4_3(feature_T1_43)#conv
        # feature_T1_45 = self.ASSP(feature_T1_44)
        # feature_T1_45 = self.ASSPconv(feature_T1_44)
        # feature_T1_45=torch.cat([feature_T1_44,feature_T1_45],1)
        feature_T2_43 = torch.cat([feature_T2_42, diff4_weight], 1)
        feature_T2_44 = self.conv_4_3(feature_T2_43)
        # feature_T2_45 = self.ASSP(feature_T2_44)
        # feature_T2_45 = self.ASSPconv(feature_T2_44)

        # feature_T2_45 = torch.cat([feature_T2_44, feature_T2_45], 1)

        diff5 = torch.abs(feature_T1_44 - feature_T2_44)  # 64
        # diff5 = self.conv_diff_5(diff5)
        # diff5_weight = diff5 * self.diff5_se(diff5)
        diff5_weight=diff5
        feature_Bottleneck = torch.cat([feature_T1_44, feature_T2_44, diff5_weight], 1)
        feature_Bottleneck=self.relu(feature_Bottleneck)
        # print('feature_Bottleneck', feature_Bottleneck.shape)
        feature_BottleneckASSP=self.ASSP(feature_Bottleneck)
        ASSPconv_bottle = self.ASSPconv(feature_Bottleneck)
        feature_Bottleneckout=torch.cat([feature_BottleneckASSP,ASSPconv_bottle],1)
        decode_1 = self.up_sample_1(feature_Bottleneckout)
        decode_1 = torch.cat([feature_T1_33, feature_T2_33, decode_1], 1)  # 320

        decode_2 = self.deconv_1(decode_1)
        decode_2 = torch.cat([feature_T1_23, feature_T2_23, decode_2], 1)

        DA = self.deconv_2(decode_2)
        decode_3 = torch.cat([feature_T1_13, feature_T2_13, DA], 1)

        outfeature = self.deconv_3(decode_3)
        # outfeature=self.sigmoid(outfeature)
        # outfeature = self.softmax(outfeature)
        # print('outfeature',outfeature.shape)
        return [outfeature,DA], [DA,DA]
    def freeze_bn_dr(self):
        # if a==True:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            if isinstance(module, nn.Dropout):
                module.eval()
class UCDNetnoBN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UCDNetnoBN, self).__init__()
        hideen_num = [16, 32, 64, 128]
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            # nn.ReLU()
        )
        self.conv_1_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU()
        )
        self.conv_2_2 = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU()
        )
        self.conv_3_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4_1 = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU()
        )
        self.conv_4_2 = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU()
        )

        # self.conv_diff_1 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
        #     # nn.BatchNorm2d(16),
        #     nn.ReLU()
        # )
        # self.diff1_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(16, 16, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 16, kernel_size=1),
        #     nn.Sigmoid()
        # )


        # self.conv_diff_2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
        #     # nn.BatchNorm2d(32),
        #     nn.ReLU()
        # )
        # self.diff2_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(32, 32 , kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32 , 32, kernel_size=1),
        #     nn.Sigmoid()
        # )
        # self.conv_diff_3 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
        #     # nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        # self.diff3_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.Sigmoid()
        # )
        # self.conv_diff_4 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
        #     # nn.BatchNorm2d(128),
        #     nn.ReLU()
        # )
        # self.diff4_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(128, 128, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, kernel_size=1),
        #     nn.Sigmoid()
        # )
        # self.conv_diff_5 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
        #     # nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        # self.diff5_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.Sigmoid()
        # )
        # self.NSSP = NSSP(192, 192, 32)
        self.ASSP = ASSP(192, 192)
        # self.ASSP = ASSPnoBN(192, 192)

        self.ASSPconv = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu=nn.ReLU()
    def forward(self, pre_data,post_data):
        #####################
        # decoder
        #####################
        # post_data=pre_data
        feature_T1_11 = self.conv_1_1(pre_data)#conv
        feature_T1_12 = self.conv_1_2(feature_T1_11)#relu+conv+relu
        feature_T2_11 = self.conv_1_1(post_data)
        feature_T2_12 = self.conv_1_2(feature_T2_11)

        diff1 = torch.abs(feature_T1_11 - feature_T2_11)
        # diff1 = self.conv_diff_1(diff1)
        # diff1_weight=diff1*self.diff1_se(diff1)
        # diff1_weight=self.relu(diff1)
        feature_T1_13 = torch.cat([feature_T1_12, diff1], 1)
        feature_T1_14 = self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12, diff1], 1)
        feature_T2_14 = self.max_pool_1(feature_T2_13)

        #Stage 2

        feature_T1_21 = self.conv_2_1(feature_T1_14)#conv
        feature_T1_22 = self.conv_2_2(feature_T1_21)#relu+conv+relu
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        feature_T2_22 = self.conv_2_2(feature_T2_21)

        diff2 = torch.abs(feature_T1_21 - feature_T2_21)
        # diff2 = self.conv_diff_2(diff2)
        # diff2_weight = diff2 * self.diff2_se(diff2)
        # diff2_weight = self.relu(diff2)
        feature_T1_23 = torch.cat([feature_T1_22, diff2], 1)
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22, diff2], 1)
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        #Stage 3

        feature_T1_31 = self.conv_3_1(feature_T1_24)#conv
        feature_T1_32 = self.conv_3_2(feature_T1_31)#relu+conv+relu
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        feature_T2_32 = self.conv_3_2(feature_T2_31)

        diff3 = torch.abs(feature_T1_31 - feature_T2_31)
        # diff3 = self.conv_diff_3(diff3)
        # diff3_weight = diff3 * self.diff3_se(diff3)
        # diff3_weight = self.relu(diff3)

        feature_T1_33 = torch.cat([feature_T1_32, diff3], 1)
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32, diff3], 1)  # 128
        feature_T2_34 = self.max_pool_3(feature_T2_33)
        #stage 4
        feature_T1_41 = self.conv_4_1(feature_T1_34)#conv
        feature_T1_42 = self.conv_4_2(feature_T1_41)#relu+conv+relu
        feature_T2_41 = self.conv_4_1(feature_T2_34)
        feature_T2_42 = self.conv_4_2(feature_T2_41)

        diff4 = torch.abs(feature_T1_41 - feature_T2_41)
        # diff4 = self.conv_diff_4(diff4)
        # diff4_weight = diff4 * self.diff4_se(diff4)
        # diff4_weight = self.relu(diff4)

        feature_T1_43 = torch.cat([feature_T1_42, diff4], 1)
        feature_T1_44 = self.conv_4_3(feature_T1_43)#conv
        # feature_T1_45 = self.ASSP(feature_T1_44)
        # feature_T1_45 = self.ASSPconv(feature_T1_44)
        # feature_T1_45=torch.cat([feature_T1_44,feature_T1_45],1)
        feature_T2_43 = torch.cat([feature_T2_42, diff4], 1)
        feature_T2_44 = self.conv_4_3(feature_T2_43)
        # feature_T2_45 = self.ASSP(feature_T2_44)
        # feature_T2_45 = self.ASSPconv(feature_T2_44)

        # feature_T2_45 = torch.cat([feature_T2_44, feature_T2_45], 1)

        diff5 = torch.abs(feature_T1_44 - feature_T2_44)  # 64
        # diff5 = self.conv_diff_5(diff5)
        # diff5_weight = diff5 * self.diff5_se(diff5)
        diff5_weight=diff5
        feature_Bottleneck = torch.cat([feature_T1_44, feature_T2_44, diff5_weight], 1)
        feature_Bottleneck=self.relu(feature_Bottleneck)
        # print('feature_Bottleneck', feature_Bottleneck.shape)
        feature_BottleneckASSP=self.ASSP(feature_Bottleneck)
        ASSPconv_bottle = self.ASSPconv(feature_Bottleneck)
        feature_Bottleneckout=torch.cat([feature_BottleneckASSP,ASSPconv_bottle],1)
        decode_1 = self.up_sample_1(feature_Bottleneckout)
        decode_1 = torch.cat([feature_T1_33, feature_T2_33, decode_1], 1)  # 320

        decode_2 = self.deconv_1(decode_1)
        decode_2 = torch.cat([feature_T1_23, feature_T2_23, decode_2], 1)

        decode_3 = self.deconv_2(decode_2)
        decode_3 = torch.cat([feature_T1_13, feature_T2_13, decode_3], 1)

        outfeature = self.deconv_3(decode_3)
        # outfeature=self.sigmoid(outfeature)
        # outfeature = self.softmax(outfeature)
        # print('outfeature',outfeature.shape)
        return outfeature, outfeature
    def freeze_bn_dr(self):
        # if a==True:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            if isinstance(module, nn.Dropout):
                module.eval()
class UCDNet_ori(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(UCDNet_ori, self).__init__()
        hideen_num = [16, 32, 64, 128]
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.conv_diff_1= nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.NSSP=NSSP(192,192,32)
        self.ASSP=ASSP(192,192)
        self.ASSPconv = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.up_sample_1 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )
        self.sigmoid=nn.Sigmoid()
        self.softmax=nn.Softmax()
    def forward(self, pre_data,post_data):
        #####################
        # decoder
        #####################
        # post_data=pre_data
        feature_T1_11 = self.conv_1_1(pre_data)
        feature_T1_12 = self.conv_1_2(feature_T1_11)
        feature_T2_11 = self.conv_1_1(post_data)
        feature_T2_12 = self.conv_1_2(feature_T2_11)

        diff1=torch.abs(feature_T1_11-feature_T2_11)
        diff1=self.conv_diff_1(diff1)

        feature_T1_13=torch.cat([feature_T1_12,diff1],1)
        feature_T1_14=self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12, diff1], 1)
        feature_T2_14 = self.max_pool_1(feature_T2_13)

        feature_T1_21=self.conv_2_1(feature_T1_14)
        feature_T1_22=self.conv_2_2(feature_T1_21)
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        feature_T2_22 = self.conv_2_2(feature_T2_21)

        diff2 = torch.abs(feature_T1_21 - feature_T2_21)
        diff2 = self.conv_diff_2(diff2)

        feature_T1_23 = torch.cat([feature_T1_22, diff2], 1)
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22, diff2], 1)
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        feature_T1_31 = self.conv_3_1(feature_T1_24)
        feature_T1_32 = self.conv_3_2(feature_T1_31)
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        feature_T2_32 = self.conv_3_2(feature_T2_31)

        diff3 = torch.abs(feature_T1_31 - feature_T2_31)
        diff3 = self.conv_diff_3(diff3)

        feature_T1_33 = torch.cat([feature_T1_32, diff3], 1)
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32, diff3], 1)#128
        feature_T2_34 = self.max_pool_3(feature_T2_33)

        feature_T1_41 = self.conv_4_1(feature_T1_34)
        feature_T1_42 = self.conv_4_2(feature_T1_41)
        feature_T2_41 = self.conv_4_1(feature_T2_34)
        feature_T2_42 = self.conv_4_2(feature_T2_41)

        diff4 = torch.abs(feature_T1_41 - feature_T2_41)
        diff4 = self.conv_diff_4(diff4)

        feature_T1_43 = torch.cat([feature_T1_42, diff4], 1)
        feature_T1_44=self.conv_4_3(feature_T1_43)
        feature_T2_43 = torch.cat([feature_T2_42, diff4], 1)
        feature_T2_44 = self.conv_4_3(feature_T2_43)

        diff5=torch.abs(feature_T1_44-feature_T2_44)
        feature_Bottleneck=torch.cat([feature_T1_44,feature_T2_44,diff5],1)
        # print('feature_Bottleneck', feature_Bottleneck.shape)
        # feature_Bottleneck=self.ASSP(feature_Bottleneck)
        feature_Bottleneck = self.ASSPconv(feature_Bottleneck)

        decode_1=self.up_sample_1(feature_Bottleneck)
        decode_1=torch.cat([feature_T1_33,feature_T2_33,decode_1],1)#320

        decode_2=self.deconv_1(decode_1)
        decode_2=torch.cat([feature_T1_23,feature_T2_23,decode_2],1)

        decode_3=self.deconv_2(decode_2)
        decode_3 = torch.cat([feature_T1_13,feature_T2_13,decode_3],1)

        outfeature=self.deconv_3(decode_3)
        # outfeature=self.sigmoid(outfeature)
        # outfeature = self.softmax(outfeature)
        # print('outfeature',outfeature.shape)
        return outfeature, outfeature

class UCDNet_sig(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(UCDNet_sig, self).__init__()
        hideen_num = [16, 32, 64, 128]
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.conv_diff_1= nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            nn.Softsign()
        )
        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.Softsign()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.Softsign()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.Softsign()
        )
        # self.NSSP=NSSP(192,192,32)
        self.ASSP=ASSP(192,192)
        self.ASSPconv = nn.Sequential(
            nn.Conv2d(192, 48, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.up_sample_1 = nn.Sequential(
            nn.ConvTranspose2d(192+48, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )
        self.sigmoid=nn.Sigmoid()
        self.softmax=nn.Softmax()
    def forward(self, pre_data,post_data):
        #####################
        # decoder
        #####################
        # post_data=pre_data
        feature_T1_11 = self.conv_1_1(pre_data)
        feature_T1_12 = self.conv_1_2(feature_T1_11)
        feature_T2_11 = self.conv_1_1(post_data)
        feature_T2_12 = self.conv_1_2(feature_T2_11)

        diff1=torch.abs(feature_T1_11-feature_T2_11)
        # diff1=feature_T1_11-feature_T2_11

        diff1=self.conv_diff_1(diff1)

        feature_T1_13=torch.cat([feature_T1_12,diff1],1)
        feature_T1_14=self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12, diff1], 1)
        feature_T2_14 = self.max_pool_1(feature_T2_13)

        feature_T1_21=self.conv_2_1(feature_T1_14)
        feature_T1_22=self.conv_2_2(feature_T1_21)
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        feature_T2_22 = self.conv_2_2(feature_T2_21)

        diff2 = torch.abs(feature_T1_21 - feature_T2_21)
        # diff2 = feature_T1_21 - feature_T2_21
        diff2 = self.conv_diff_2(diff2)

        feature_T1_23 = torch.cat([feature_T1_22, diff2], 1)
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22, diff2], 1)
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        feature_T1_31 = self.conv_3_1(feature_T1_24)
        feature_T1_32 = self.conv_3_2(feature_T1_31)
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        feature_T2_32 = self.conv_3_2(feature_T2_31)

        diff3 = torch.abs(feature_T1_31 - feature_T2_31)
        # diff3 = feature_T1_31 - feature_T2_31
        diff3 = self.conv_diff_3(diff3)

        feature_T1_33 = torch.cat([feature_T1_32, diff3], 1)
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32, diff3], 1)#128
        feature_T2_34 = self.max_pool_3(feature_T2_33)

        feature_T1_41 = self.conv_4_1(feature_T1_34)
        feature_T1_42 = self.conv_4_2(feature_T1_41)
        feature_T2_41 = self.conv_4_1(feature_T2_34)
        feature_T2_42 = self.conv_4_2(feature_T2_41)

        diff4 = torch.abs(feature_T1_41 - feature_T2_41)
        # diff4 = feature_T1_41 - feature_T2_41
        diff4 = self.conv_diff_4(diff4)

        feature_T1_43 = torch.cat([feature_T1_42, diff4], 1)
        feature_T1_44=self.conv_4_3(feature_T1_43)
        feature_T2_43 = torch.cat([feature_T2_42, diff4], 1)
        feature_T2_44 = self.conv_4_3(feature_T2_43)

        diff5=torch.abs(feature_T1_44-feature_T2_44)
        feature_Bottleneck=torch.cat([feature_T1_44,feature_T2_44,diff5],1)
        # print('feature_Bottleneck', feature_Bottleneck.shape)
        feature_Bottleneckassp=self.ASSP(feature_Bottleneck)
        feature_BottleneckConv = self.ASSPconv(feature_Bottleneck)
        feature_Bottleneck=torch.cat([feature_Bottleneckassp,feature_BottleneckConv],1)

        decode_1=self.up_sample_1(feature_Bottleneck)
        decode_1=torch.cat([feature_T1_33,feature_T2_33,decode_1],1)#320

        decode_2=self.deconv_1(decode_1)
        decode_2=torch.cat([feature_T1_23,feature_T2_23,decode_2],1)

        decode_3=self.deconv_2(decode_2)
        decode_3 = torch.cat([feature_T1_13,feature_T2_13,decode_3],1)

        outfeature=self.deconv_3(decode_3)
        # outfeature=self.sigmoid(outfeature)
        # outfeature = self.softmax(outfeature)
        # print('outfeature',outfeature.shape)
        return outfeature, outfeature
    def freeze_bn_dr(self):
        # if a==True:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            if isinstance(module, nn.Dropout):
                module.eval()

class UCDNet_bn(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(UCDNet_bn, self).__init__()
        hideen_num = [16, 32, 64, 128]
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            # nn.ReLU()
            nn.BatchNorm2d(16)
        )
        self.conv_1_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.ReLU()
            nn.BatchNorm2d(32)

        )
        self.conv_2_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.ReLU()
            nn.BatchNorm2d(64)

        )
        self.conv_3_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # nn.ReLU()
            nn.BatchNorm2d(128)

        )
        self.conv_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            # nn.ReLU()
            nn.BatchNorm2d(64)

        )

        self.conv_diff_1= nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.conv_diff_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.NSSP=NSSP(192,192,32)
        self.ASSP=ASSP(192,192)
        self.ASSPconv = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.up_sample_1 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )
        self.sigmoid=nn.Sigmoid()
        self.softmax=nn.Softmax()
        self.relu=nn.ReLU()
    def forward(self, pre_data,post_data):
        #####################
        # decoder
        #####################
        # post_data=pre_data
        feature_T1_11 = self.conv_1_1(pre_data)#conv
        feature_T1_12 = self.conv_1_2(feature_T1_11)#relu-conv-relu
        feature_T2_11 = self.conv_1_1(post_data)
        feature_T2_12 = self.conv_1_2(feature_T2_11)

        diff1=torch.pow(feature_T1_11-feature_T2_11,2)
        diff1=self.conv_diff_1(diff1)#relu-conv-relu

        feature_T1_13=torch.cat([feature_T1_12,diff1],1)
        feature_T1_14=self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12, diff1], 1)
        feature_T2_14 = self.max_pool_1(feature_T2_13)

        feature_T1_21=self.conv_2_1(feature_T1_14)#conv
        feature_T1_22=self.conv_2_2(feature_T1_21)#relu-conv-relu
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        feature_T2_22 = self.conv_2_2(feature_T2_21)

        diff2 = torch.pow(feature_T1_21 - feature_T2_21,2)
        diff2 = self.conv_diff_2(diff2)#relu-conv-relu

        feature_T1_23 = torch.cat([feature_T1_22, diff2], 1)
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22, diff2], 1)
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        feature_T1_31 = self.conv_3_1(feature_T1_24)#conv
        feature_T1_32 = self.conv_3_2(feature_T1_31)#relu-conv-relu
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        feature_T2_32 = self.conv_3_2(feature_T2_31)

        diff3 = torch.pow(feature_T1_31 - feature_T2_31,2)
        diff3 = self.conv_diff_3(diff3)#relu-conv-relu

        feature_T1_33 = torch.cat([feature_T1_32, diff3], 1)
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32, diff3], 1)#128
        feature_T2_34 = self.max_pool_3(feature_T2_33)

        feature_T1_41 = self.conv_4_1(feature_T1_34)#conv
        feature_T1_42 = self.conv_4_2(feature_T1_41)#relu-conv-relu
        feature_T2_41 = self.conv_4_1(feature_T2_34)
        feature_T2_42 = self.conv_4_2(feature_T2_41)

        diff4 = torch.pow(feature_T1_41 - feature_T2_41,2)
        diff4 = self.conv_diff_4(diff4)#relu-conv-relu

        feature_T1_43 = torch.cat([feature_T1_42, diff4], 1)
        feature_T1_44=self.conv_4_3(feature_T1_43)#conv
        feature_T2_43 = torch.cat([feature_T2_42, diff4], 1)
        feature_T2_44 = self.conv_4_3(feature_T2_43)

        diff5=torch.pow(feature_T1_44-feature_T2_44,2)

        feature_Bottleneck=torch.cat([feature_T1_44,feature_T2_44,diff5],1)
        feature_Bottleneck = self.relu(feature_Bottleneck)
        # print('feature_Bottleneck', feature_Bottleneck.shape)
        # feature_Bottleneck=self.ASSP(feature_Bottleneck)
        feature_Bottleneck = self.ASSPconv(feature_Bottleneck)

        decode_1=self.up_sample_1(feature_Bottleneck)
        decode_1=torch.cat([feature_T1_33,feature_T2_33,decode_1],1)#320

        decode_2=self.deconv_1(decode_1)
        decode_2=torch.cat([feature_T1_23,feature_T2_23,decode_2],1)

        decode_3=self.deconv_2(decode_2)
        decode_3 = torch.cat([feature_T1_13,feature_T2_13,decode_3],1)

        outfeature=self.deconv_3(decode_3)
        # outfeature=self.sigmoid(outfeature)
        # outfeature = self.softmax(outfeature)
        # print('outfeature',outfeature.shape)
        return outfeature, outfeature

class UCDNet_ASSP(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(UCDNet_ASSP, self).__init__()
        hideen_num = [16, 32, 64, 128]
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_1_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_2_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_3_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            # nn.ReLU()
        )

        self.conv_diff_1= nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.conv_diff_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0),
            nn.ReLU()
        )
        # self.NSSP=NSSP(192,192,32)
        self.ASSP=ASSP(128,64)
        self.ASSPconv = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.up_sample_1 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )
        self.sigmoid=nn.Sigmoid()
        self.softmax=nn.Softmax()
        self.relu=nn.ReLU()
    def forward(self, pre_data,post_data):
        #####################
        # decoder
        #####################
        # post_data=pre_data
        feature_T1_11 = self.conv_1_1(pre_data)#conv
        feature_T1_12 = self.conv_1_2(feature_T1_11)#relu-conv-relu
        feature_T2_11 = self.conv_1_1(post_data)
        feature_T2_12 = self.conv_1_2(feature_T2_11)

        diff1=torch.pow(feature_T1_11-feature_T2_11,2)
        diff1=self.conv_diff_1(diff1)#relu-conv-relu

        feature_T1_13=torch.cat([feature_T1_12,diff1],1)
        feature_T1_14=self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12, diff1], 1)
        feature_T2_14 = self.max_pool_1(feature_T2_13)
        ######stage2
        feature_T1_21=self.conv_2_1(feature_T1_14)#conv
        feature_T1_22=self.conv_2_2(feature_T1_21)#relu-conv-relu
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        feature_T2_22 = self.conv_2_2(feature_T2_21)

        diff2 = torch.pow(feature_T1_21 - feature_T2_21,2)
        diff2 = self.conv_diff_2(diff2)#relu-conv-relu

        feature_T1_23 = torch.cat([feature_T1_22, diff2], 1)
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22, diff2], 1)
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        ######stage3

        feature_T1_31 = self.conv_3_1(feature_T1_24)#conv
        feature_T1_32 = self.conv_3_2(feature_T1_31)#relu-conv-relu
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        feature_T2_32 = self.conv_3_2(feature_T2_31)

        diff3 = torch.pow(feature_T1_31 - feature_T2_31,2)
        diff3 = self.conv_diff_3(diff3)#relu-conv-relu

        feature_T1_33 = torch.cat([feature_T1_32, diff3], 1)
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32, diff3], 1)#128
        feature_T2_34 = self.max_pool_3(feature_T2_33)

        feature_T1_41 = self.conv_4_1(feature_T1_34)#conv
        feature_T1_42 = self.conv_4_2(feature_T1_41)#relu-conv-relu
        feature_T2_41 = self.conv_4_1(feature_T2_34)
        feature_T2_42 = self.conv_4_2(feature_T2_41)

        diff4 = torch.pow(feature_T1_41 - feature_T2_41,2)
        diff4 = self.conv_diff_4(diff4)#relu-conv-relu

        feature_T1_43 = torch.cat([feature_T1_42, diff4], 1)#128
        # feature_T1_44=self.conv_4_3(feature_T1_43)#conv
        feature_T1_44 = self.ASSP(feature_T1_43)  # conv
        feature_T2_43 = torch.cat([feature_T2_42, diff4], 1)
        # feature_T2_44 = self.conv_4_3(feature_T2_43)
        feature_T2_44 = self.ASSP(feature_T2_43)  # conv

        diff5=torch.pow(feature_T1_44-feature_T2_44,2)

        feature_Bottleneck=torch.cat([feature_T1_44,feature_T2_44,diff5],1)
        feature_Bottleneck = self.relu(feature_Bottleneck)
        # print('feature_Bottleneck', feature_Bottleneck.shape)
        # feature_Bottleneck=self.ASSP(feature_Bottleneck)
        feature_Bottleneck = self.ASSPconv(feature_Bottleneck)

        decode_1=self.up_sample_1(feature_Bottleneck)
        decode_1=torch.cat([feature_T1_33,feature_T2_33,decode_1],1)#320

        decode_2=self.deconv_1(decode_1)
        decode_2=torch.cat([feature_T1_23,feature_T2_23,decode_2],1)

        decode_3=self.deconv_2(decode_2)
        decode_3 = torch.cat([feature_T1_13,feature_T2_13,decode_3],1)

        outfeature=self.deconv_3(decode_3)
        # outfeature=self.sigmoid(outfeature)
        # outfeature = self.softmax(outfeature)
        # print('outfeature',outfeature.shape)
        return outfeature, outfeature

class UCDNet_ASSPMulti(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UCDNet_ASSPMulti, self).__init__()
        hideen_num = [16, 32, 64, 128]
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_1_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_2_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_3_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_4_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU()
        )

        self.conv_diff_1 = nn.Sequential(

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            # nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.diff1_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.Sigmoid()
        )


        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            # nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.diff2_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 32 , kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32 , 32, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.diff3_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            # nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.diff4_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.diff5_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        self.NSSP = NSSP(192, 192, 32)
        self.ASSP = ASSP(192, 192)
        self.ASSPconv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            # nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu=nn.ReLU()
    def forward(self, pre_data,post_data):
        #####################
        # decoder
        #####################
        # post_data=pre_data
        feature_T1_11 = self.conv_1_1(pre_data)#conv
        feature_T1_12 = self.conv_1_2(feature_T1_11)#relu+conv+relu
        feature_T2_11 = self.conv_1_1(post_data)
        feature_T2_12 = self.conv_1_2(feature_T2_11)

        diff1 = torch.pow(feature_T1_11 - feature_T2_11,2)
        # diff1 = self.conv_diff_1(diff1)
        diff1_weight=diff1*self.diff1_se(diff1)
        diff1_weight=self.relu(diff1_weight)
        feature_T1_13 = torch.cat([feature_T1_12, diff1_weight], 1)
        feature_T1_14 = self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12, diff1_weight], 1)
        feature_T2_14 = self.max_pool_1(feature_T2_13)

        #Stage 2

        feature_T1_21 = self.conv_2_1(feature_T1_14)#relu+conv
        feature_T1_22 = self.conv_2_2(feature_T1_21)#relu+conv+relu
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        feature_T2_22 = self.conv_2_2(feature_T2_21)

        diff2 = torch.pow(feature_T1_21 - feature_T2_21,2)
        diff2 = self.conv_diff_2(diff2)
        diff2_weight = diff2 * self.diff2_se(diff2)
        diff2_weight = self.relu(diff2_weight)
        feature_T1_23 = torch.cat([feature_T1_22, diff2_weight], 1)
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22, diff2_weight], 1)
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        #Stage 3

        feature_T1_31 = self.conv_3_1(feature_T1_24)#relu+conv
        feature_T1_32 = self.conv_3_2(feature_T1_31)#relu+conv+relu
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        feature_T2_32 = self.conv_3_2(feature_T2_31)

        diff3 = torch.pow(feature_T1_31 - feature_T2_31,2)
        diff3 = self.conv_diff_3(diff3)
        diff3_weight = diff3 * self.diff3_se(diff3)
        diff3_weight = self.relu(diff3_weight)

        feature_T1_33 = torch.cat([feature_T1_32, diff3_weight], 1)
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32, diff3_weight], 1)  # 128
        feature_T2_34 = self.max_pool_3(feature_T2_33)
        #stage 4
        feature_T1_41 = self.conv_4_1(feature_T1_34)#relu+conv
        feature_T1_42 = self.conv_4_2(feature_T1_41)#relu+conv+relu
        feature_T2_41 = self.conv_4_1(feature_T2_34)
        feature_T2_42 = self.conv_4_2(feature_T2_41)

        diff4 = torch.pow(feature_T1_41 - feature_T2_41,2)
        # diff4 = self.conv_diff_4(diff4)
        diff4_weight = diff4 * self.diff4_se(diff4)
        diff4_weight = self.relu(diff4_weight)

        feature_T1_43 = torch.cat([feature_T1_42, diff4_weight], 1)
        feature_T1_44 = self.conv_4_3(feature_T1_43)#conv
        # feature_T1_45 = self.ASSP(feature_T1_44)
        # feature_T1_45 = self.ASSPconv(feature_T1_44)
        # feature_T1_45=torch.cat([feature_T1_44,feature_T1_45],1)
        feature_T2_43 = torch.cat([feature_T2_42, diff4_weight], 1)
        feature_T2_44 = self.conv_4_3(feature_T2_43)
        # feature_T2_45 = self.ASSP(feature_T2_44)
        # feature_T2_45 = self.ASSPconv(feature_T2_44)

        # feature_T2_45 = torch.cat([feature_T2_44, feature_T2_45], 1)

        diff5 = torch.pow(feature_T1_44 - feature_T2_44,2)  # 64
        # diff5 = self.conv_diff_5(diff5)
        diff5_weight = diff5 * self.diff5_se(diff5)

        feature_Bottleneck = torch.cat([feature_T1_44, feature_T2_44, diff5_weight], 1)
        feature_Bottleneck=self.relu(feature_Bottleneck)
        # print('feature_Bottleneck', feature_Bottleneck.shape)
        feature_Bottleneck=self.ASSP(feature_Bottleneck)

        decode_1 = self.up_sample_1(feature_Bottleneck)
        decode_1 = torch.cat([feature_T1_33, feature_T2_33, decode_1], 1)  # 320

        decode_2 = self.deconv_1(decode_1)
        decode_2 = torch.cat([feature_T1_23, feature_T2_23, decode_2], 1)

        decode_3 = self.deconv_2(decode_2)
        decode_3 = torch.cat([feature_T1_13, feature_T2_13, decode_3], 1)

        outfeature = self.deconv_3(decode_3)
        # outfeature=self.sigmoid(outfeature)
        # outfeature = self.softmax(outfeature)
        # print('outfeature',outfeature.shape)
        return outfeature, outfeature

class UCDNet_ASSPMultiout(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UCDNet_ASSPMultiout, self).__init__()
        hideen_num = [16, 32, 64, 128]
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv_diff_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            # nn.BatchNorm2d(16),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff1_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU()
        )


        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff2_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 32 , kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32 , 32, kernel_size=1),
            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0)
            # nn.BatchNorm2d(64),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff3_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0)
            # nn.BatchNorm2d(128),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff4_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            # nn.BatchNorm2d(64),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff5_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU()
        )
        self.NSSP = NSSP(192, 192, 32)
        self.ASSP = ASSP(64, 64)
        self.ASSPconv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            # nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )
        self.updiff3=nn.Upsample(scale_factor=4, mode='bilinear')
        self.updiff4=nn.Upsample(scale_factor=8, mode='bilinear')
        self.updiff5=nn.Upsample(scale_factor=8, mode='bilinear')

        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU()
        self.softmax = nn.Softmax()
        self.softsign=nn.Softsign()
        self.tanh=nn.Tanh()

    def forward(self, pre_data,post_data):
        #####################
        # decoder
        #####################
        # post_data=pre_data
        feature_T1_11 = self.conv_1_1(pre_data)
        feature_T1_12 = self.conv_1_2(feature_T1_11)
        feature_T2_11 = self.conv_1_1(post_data)
        feature_T2_12 = self.conv_1_2(feature_T2_11)

        diff1 = torch.abs(feature_T1_11 - feature_T2_11)
        diff1 = self.conv_diff_1(diff1)
        diff1=diff1*self.diff1_se(diff1)
        diff1_weight = self.relu(torch.abs(diff1))

        # diff1_weight = diff1
        feature_T1_13 = torch.cat([feature_T1_12, diff1_weight], 1)
        feature_T1_14 = self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12, diff1_weight], 1)
        feature_T2_14 = self.max_pool_1(feature_T2_13)

        feature_T1_21 = self.conv_2_1(feature_T1_14)
        feature_T1_22 = self.conv_2_2(feature_T1_21)
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        feature_T2_22 = self.conv_2_2(feature_T2_21)

        diff2 = torch.abs(feature_T1_21 - feature_T2_21)
        diff2 = self.conv_diff_2(diff2)
        diff2 = diff2 * self.diff2_se(diff2)
        diff2_weight=self.relu(torch.abs(diff2))
        # diff2_weight=diff2
        feature_T1_23 = torch.cat([feature_T1_22, diff2_weight], 1)
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22, diff2_weight], 1)
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        feature_T1_31 = self.conv_3_1(feature_T1_24)
        feature_T1_32 = self.conv_3_2(feature_T1_31)
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        feature_T2_32 = self.conv_3_2(feature_T2_31)

        diff3 = torch.abs(feature_T1_31 - feature_T2_31)
        diff3 = self.conv_diff_3(diff3)
        diff3 = diff3 * self.diff3_se(diff3)
        diff3_weight = self.relu(torch.abs(diff3))
        # diff3_weight=diff3
        feature_T1_33 = torch.cat([feature_T1_32, diff3_weight], 1)
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32, diff3_weight], 1)  # 128
        feature_T2_34 = self.max_pool_3(feature_T2_33)

        feature_T1_41 = self.conv_4_1(feature_T1_34)
        feature_T1_42 = self.conv_4_2(feature_T1_41)
        feature_T2_41 = self.conv_4_1(feature_T2_34)
        feature_T2_42 = self.conv_4_2(feature_T2_41)

        diff4 = torch.abs(feature_T1_41 - feature_T2_41)
        diff4 = self.conv_diff_4(diff4)
        diff4 = diff4 * self.diff4_se(diff4)
        # diff4_weight=self.tanh(diff4)
        diff4_weight = self.relu(torch.abs(diff4))

        # diff4_weight=diff4
        feature_T1_43 = torch.cat([feature_T1_42, diff4_weight], 1)
        feature_T1_44 = self.conv_4_3(feature_T1_43)
        feature_T1_45 = self.ASSP(feature_T1_44)
        # feature_T1_45 = self.ASSPconv(feature_T1_44)
        # feature_T1_45=torch.cat([feature_T1_44,feature_T1_45],1)
        feature_T2_43 = torch.cat([feature_T2_42, diff4_weight], 1)
        feature_T2_44 = self.conv_4_3(feature_T2_43)
        feature_T2_45 = self.ASSP(feature_T2_44)
        # feature_T2_45 = self.ASSPconv(feature_T2_44)

        # feature_T2_45 = torch.cat([feature_T2_44, feature_T2_45], 1)

        diff5 = torch.abs(feature_T1_45 - feature_T2_45)  # 64
        diff5 = self.conv_diff_5(diff5)

        diff5 = diff5 * self.diff5_se(diff5)
        # diff5_weight=self.relu(diff5)
        diff5_weight = self.relu(torch.abs(diff5))
        # diff5_weight=diff5
        # print('diff5_weight',diff5_weight.shape)
        feature_Bottleneck = torch.cat([feature_T1_44, feature_T2_44, diff5_weight], 1)

        # print('feature_Bottleneck', feature_Bottleneck.shape)
        # feature_Bottleneck=self.ASSP(feature_Bottleneck)

        decode_1 = self.up_sample_1(feature_Bottleneck)
        decode_1 = torch.cat([feature_T1_33, feature_T2_33, decode_1], 1)  # 320

        decode_2 = self.deconv_1(decode_1)
        decode_2 = torch.cat([feature_T1_23, feature_T2_23, decode_2], 1)

        decode_3 = self.deconv_2(decode_2)#150
        decode_3 = torch.cat([feature_T1_13, feature_T2_13, decode_3], 1)#80

        outfeature = self.deconv_3(decode_3)
        diff3_weight=self.updiff3(diff3_weight)
        diff4_weight = self.updiff4(diff4_weight)
        diff5_weight = self.updiff5(diff5_weight)
        # outfeature=self.sigmoid(outfeature)
        # outfeature = self.softmax(outfeature)
        # print('outfeature',outfeature.shape)
        # return [outfeature,diff4_weight,diff5_weight], decode_3

        return [outfeature, diff4_weight, diff5_weight], decode_3

class UCDNet_ASSPMultiout_new(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UCDNet_ASSPMultiout_new, self).__init__()
        hideen_num = [16, 32, 64, 128]
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU6()
        )
        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU6()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU6()
        )
        self.conv_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU6()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU6()
        )
        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),

            nn.ReLU6()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU6()
        )
        self.conv_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU6(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU6()
        )
        self.conv_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU6()
        )

        self.conv_diff_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            # nn.BatchNorm2d(16),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff1_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU()
        )


        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff2_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 32 , kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32 , 32, kernel_size=1),
            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0)
            # nn.BatchNorm2d(64),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff3_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0)
            # nn.BatchNorm2d(128),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff4_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU()
        )
        self.conv_diff_5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            # nn.BatchNorm2d(64),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff5_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU()
        )
        self.NSSP = NSSP(192, 192, 32)
        self.ASSP = ASSP(64, 64)
        self.ASSPconv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            # nn.BatchNorm2d(192),
            nn.ReLU6(),

            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU6(),
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU6(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU6(),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU6(),

            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            # nn.BatchNorm2d(32),
            nn.ReLU6()
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU6(),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU6(),

            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            # nn.BatchNorm2d(16),
            nn.ReLU6()
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU6(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )
        self.updiff3=nn.Upsample(scale_factor=4, mode='bilinear')
        self.updiff4=nn.Upsample(scale_factor=8, mode='bilinear')
        self.updiff5=nn.Upsample(scale_factor=8, mode='bilinear')

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.relu6=nn.ReLU6()
        self.softmax = nn.Softmax()
        self.softsign=nn.Softsign()
        self.tanh=nn.Tanh()

    def forward(self, pre_data,post_data):
        #####################
        # decoder
        #####################
        # post_data=pre_data
        feature_T1_11 = self.conv_1_1(pre_data)
        feature_T1_12 = self.conv_1_2(feature_T1_11)
        feature_T2_11 = self.conv_1_1(post_data)
        feature_T2_12 = self.conv_1_2(feature_T2_11)

        diff1 = torch.abs(feature_T1_11 - feature_T2_11)
        diff1 = self.conv_diff_1(diff1)
        diff1=diff1*self.diff1_se(diff1)
        # diff1_weight = self.relu(torch.abs(diff1))
        diff1_weight = self.relu(diff1)
        # diff1_weight = diff1
        feature_T1_13 = torch.cat([feature_T1_12, diff1_weight], 1)
        feature_T1_14 = self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12, diff1_weight], 1)
        feature_T2_14 = self.max_pool_1(feature_T2_13)

        feature_T1_21 = self.conv_2_1(feature_T1_14)
        feature_T1_22 = self.conv_2_2(feature_T1_21)
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        feature_T2_22 = self.conv_2_2(feature_T2_21)

        diff2 = torch.abs(feature_T1_21 - feature_T2_21)
        diff2 = self.conv_diff_2(diff2)
        diff2 = diff2 * self.diff2_se(diff2)
        # diff2_weight=self.relu(torch.abs(diff2))
        diff2_weight = self.relu(diff2)
        # diff2_weight=diff2
        feature_T1_23 = torch.cat([feature_T1_22, diff2_weight], 1)
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22, diff2_weight], 1)
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        feature_T1_31 = self.conv_3_1(feature_T1_24)
        feature_T1_32 = self.conv_3_2(feature_T1_31)
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        feature_T2_32 = self.conv_3_2(feature_T2_31)

        diff3 = torch.abs(feature_T1_31 - feature_T2_31)
        diff3 = self.conv_diff_3(diff3)
        diff3 = diff3 * self.diff3_se(diff3)
        # diff3_weight = self.relu(torch.abs(diff3))
        diff3_weight = self.relu(diff3)
        # diff3_weight=diff3
        feature_T1_33 = torch.cat([feature_T1_32, diff3_weight], 1)
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32, diff3_weight], 1)  # 128
        feature_T2_34 = self.max_pool_3(feature_T2_33)

        feature_T1_41 = self.conv_4_1(feature_T1_34)
        feature_T1_42 = self.conv_4_2(feature_T1_41)
        feature_T2_41 = self.conv_4_1(feature_T2_34)
        feature_T2_42 = self.conv_4_2(feature_T2_41)

        diff4 = torch.abs(feature_T1_41 - feature_T2_41)
        diff4 = self.conv_diff_4(diff4)
        diff4 = diff4 * self.diff4_se(diff4)
        # diff4_weight=self.tanh(diff4)
        # diff4_weight = self.relu(torch.abs(diff4))
        diff4_weight = self.relu(diff4)

        # diff4_weight=diff4
        feature_T1_43 = torch.cat([feature_T1_42, diff4_weight], 1)
        feature_T1_44 = self.conv_4_3(feature_T1_43)
        feature_T1_45 = self.ASSP(feature_T1_44)
        # feature_T1_45 = self.ASSPconv(feature_T1_44)
        # feature_T1_45=torch.cat([feature_T1_44,feature_T1_45],1)
        feature_T2_43 = torch.cat([feature_T2_42, diff4_weight], 1)
        feature_T2_44 = self.conv_4_3(feature_T2_43)
        feature_T2_45 = self.ASSP(feature_T2_44)
        # feature_T2_45 = self.ASSPconv(feature_T2_44)

        # feature_T2_45 = torch.cat([feature_T2_44, feature_T2_45], 1)

        diff5 = torch.abs(feature_T1_45 - feature_T2_45)  # 64
        diff5 = self.conv_diff_5(diff5)

        diff5 = diff5 * self.diff5_se(diff5)
        # diff5_weight=self.relu(diff5)
        # diff5_weight = self.relu(torch.abs(diff5))
        diff5_weight = self.relu(diff5)
        # diff5_weight=diff5
        # print('diff5_weight',diff5_weight.shape)
        feature_Bottleneck = torch.cat([feature_T1_44, feature_T2_44, diff5_weight], 1)

        # print('feature_Bottleneck', feature_Bottleneck.shape)
        # feature_Bottleneck=self.ASSP(feature_Bottleneck)

        decode_1 = self.up_sample_1(feature_Bottleneck)
        decode_1 = torch.cat([feature_T1_33, feature_T2_33, decode_1], 1)  # 320

        decode_2 = self.deconv_1(decode_1)
        decode_2 = torch.cat([feature_T1_23, feature_T2_23, decode_2], 1)

        decode_3 = self.deconv_2(decode_2)#150
        decode_3 = torch.cat([feature_T1_13, feature_T2_13, decode_3], 1)#80

        outfeature = self.deconv_3(decode_3)
        diff3_weight=self.updiff3(diff3_weight)
        diff4_weight = self.updiff4(diff4_weight)
        diff5_weight = self.updiff5(diff5_weight)
        # outfeature=self.sigmoid(outfeature)
        # outfeature = self.softmax(outfeature)
        # print('outfeature',outfeature.shape)
        # return [outfeature,diff4_weight,diff5_weight], decode_3

        return [outfeature, diff4_weight, diff5_weight], decode_3

class UCDNet_ASSP_SEAG(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UCDNet_ASSP_SEAG, self).__init__()
        hideen_num = [16, 32, 64, 128]
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU6()
        )
        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            # nn.ReLU6()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU6()
        )
        self.conv_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU6()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU6()
        )
        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),

            # nn.ReLU6()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU6()
        )
        self.conv_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU6(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # nn.ReLU6()
        )
        self.conv_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU6()
        )

        self.conv_diff_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            # nn.BatchNorm2d(16),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff1_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff1_AG= nn.Sequential(
            nn.ReLU(),
            # nn.Conv2d(16, 16//2, kernel_size=1),
            # nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff2_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 32 , kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32 , 32, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff2_AG = nn.Sequential(
            nn.ReLU(),
            # nn.Conv2d(32, 32 // 2, kernel_size=1),
            # nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0)
            # nn.BatchNorm2d(64),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff3_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff3_AG = nn.Sequential(
            nn.ReLU(),
            # nn.Conv2d(64, 64 // 2, kernel_size=1),
            # nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0)
            # nn.BatchNorm2d(128),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff4_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff4_AG = nn.Sequential(
            nn.ReLU(),
            # nn.Conv2d(128, 128 // 2, kernel_size=1),
            # nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            # nn.BatchNorm2d(64),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff5_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff5_AG = nn.Sequential(
            nn.ReLU(),
            # nn.Conv2d(128, 128 // 2, kernel_size=1),
            # nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.NSSP = NSSP(192, 192, 32)
        self.ASSP = ASSP(64, 64)
        self.ASSPconv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            # # nn.BatchNorm2d(64),
            # nn.ReLU()
        )

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU6(),

            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU6(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU6(),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(),

            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            # nn.BatchNorm2d(32),
            # nn.ReLU6()
        )
        self.deconv_2 = nn.Sequential(
            nn.ReLU6(),
            nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU6(),

            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            # nn.BatchNorm2d(16),
            # nn.ReLU6()
        )
        self.deconv_3 = nn.Sequential(
            nn.ReLU6(),
            nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU6(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )
        self.updiff3=nn.Upsample(scale_factor=4, mode='bilinear',align_corners=True)
        self.updiff4=nn.Upsample(scale_factor=8, mode='bilinear',align_corners=True)
        self.updiff5=nn.Upsample(scale_factor=8, mode='bilinear',align_corners=True)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.relu6=nn.ReLU6()
        self.softmax = nn.Softmax()
        self.softsign=nn.Softsign()
        self.tanh=nn.Tanh()

    def forward(self, pre_data,post_data):
        #####################
        # decoder
        #####################
        # post_data=pre_data
        ####Stage 1
        feature_T1_11 = self.conv_1_1(pre_data)#conv+relu
        feature_T1_12 = self.conv_1_2(feature_T1_11)#conv
        feature_T2_11 = self.conv_1_1(post_data)
        feature_T2_12 = self.conv_1_2(feature_T2_11)

        diff1 = torch.abs(feature_T1_11 - feature_T2_11)
        diff1 = self.conv_diff_1(diff1)#conv
        diff1_SE = self.diff1_se(diff1)
        diff1_AG = self.diff1_AG(diff1)
        diff1_weight = self.relu6(diff1 * diff1_SE * diff1_AG)

        feature_T1_12 = feature_T1_12 * diff1_AG
        feature_T2_12 = feature_T2_12 * diff1_AG
        feature_T1_12 = self.relu6(feature_T1_12)
        feature_T2_12 = self.relu6(feature_T2_12)

        feature_T1_13 = torch.cat([feature_T1_12, diff1_weight], 1)
        feature_T1_14 = self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12, diff1_weight], 1)
        feature_T2_14 = self.max_pool_1(feature_T2_13)


        ####Stage 2
        feature_T1_21 = self.conv_2_1(feature_T1_14)
        feature_T1_22 = self.conv_2_2(feature_T1_21)
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        feature_T2_22 = self.conv_2_2(feature_T2_21)

        diff2 = torch.abs(feature_T1_21 - feature_T2_21)
        diff2 = self.conv_diff_2(diff2)
        diff2_SE = self.diff2_se(diff2)
        diff2_AG = self.diff2_AG(diff2)
        diff2_weight = self.relu6(diff2 * diff2_SE * diff2_AG)

        feature_T1_22 = feature_T1_22 * diff2_AG
        feature_T2_22 = feature_T2_22 * diff2_AG
        feature_T1_22 = self.relu6(feature_T1_22)
        feature_T2_22 = self.relu6(feature_T2_22)

        feature_T1_23 = torch.cat([feature_T1_22, diff2_weight], 1)
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22, diff2_weight], 1)
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        ######Stage3
        feature_T1_31 = self.conv_3_1(feature_T1_24)
        feature_T1_32 = self.conv_3_2(feature_T1_31)
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        feature_T2_32 = self.conv_3_2(feature_T2_31)

        diff3 = torch.abs(feature_T1_31 - feature_T2_31)
        diff3 = self.conv_diff_3(diff3)
        diff3_SE = self.diff3_se(diff3)
        diff3_AG = self.diff3_AG(diff3)
        diff3_weight = self.relu6(diff3 * diff3_SE * diff3_AG)

        feature_T1_32 = feature_T1_32 * diff3_AG
        feature_T2_32 = feature_T2_32 * diff3_AG
        feature_T1_32 = self.relu6(feature_T1_32)
        feature_T2_32 = self.relu6(feature_T2_32)

        feature_T1_33 = torch.cat([feature_T1_32, diff3_weight], 1)
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32, diff3_weight], 1)  # 128
        feature_T2_34 = self.max_pool_3(feature_T2_33)
        ##############Stage 4
        feature_T1_41 = self.conv_4_1(feature_T1_34)
        feature_T1_42 = self.conv_4_2(feature_T1_41)
        feature_T2_41 = self.conv_4_1(feature_T2_34)
        feature_T2_42 = self.conv_4_2(feature_T2_41)

        diff4 = torch.abs(feature_T1_41 - feature_T2_41)
        diff4 = self.conv_diff_4(diff4)
        diff4_SE = self.diff4_se(diff4)
        diff4_AG = self.diff4_AG(diff4)
        diff4_weight = self.relu6(diff4 * diff4_SE * diff4_AG)

        feature_T1_42 = feature_T1_42 * diff4_AG
        feature_T2_42 = feature_T2_42 * diff4_AG
        feature_T1_42 = self.relu6(feature_T1_42)
        feature_T2_42 = self.relu6(feature_T2_42)

        feature_T1_43 = torch.cat([feature_T1_42, diff4_weight], 1)
        feature_T1_44 = self.conv_4_3(feature_T1_43)
        feature_T2_43 = torch.cat([feature_T2_42, diff4_weight], 1)
        feature_T2_44 = self.conv_4_3(feature_T2_43)

        ##############Stage ASSP
        feature_T1_45 = self.ASSP(feature_T1_44)#64,32,32
        feature_T1_46 = self.ASSPconv(feature_T1_45)
        feature_T2_45 = self.ASSP(feature_T2_44)
        feature_T2_46 = self.ASSPconv(feature_T2_45)

        diff5 = torch.abs(feature_T1_45 - feature_T2_45)  # 64
        diff5 = self.conv_diff_5(diff5)#64
        diff5_SE = self.diff5_se(diff5)
        diff5_AG = self.diff5_AG(diff5)
        diff5_weight = self.relu6(diff5 * diff5_SE * diff5_AG)

        feature_T1_46 = feature_T1_46 * diff5_AG
        feature_T2_46 = feature_T2_46 * diff5_AG
        feature_T1_46 = self.relu6(feature_T1_46)
        feature_T2_46 = self.relu6(feature_T2_46)
        feature_Bottleneck = torch.cat([feature_T1_46, feature_T2_46, diff5_weight], 1)#192


        decode_1 = self.up_sample_1(feature_Bottleneck)
        decode_1 = torch.cat([feature_T1_33, feature_T2_33, decode_1], 1)  # 320

        decode_2 = self.deconv_1(decode_1)
        decode_2 = torch.cat([feature_T1_23, feature_T2_23, decode_2], 1)

        decode_3 = self.deconv_2(decode_2)#150
        decode_3 = torch.cat([feature_T1_13, feature_T2_13, decode_3], 1)#80

        outfeature = self.deconv_3(decode_3)
        diff3_weight=self.updiff3(diff3_weight)
        diff4_weight = self.updiff4(diff4_weight)
        diff5_weight = self.updiff5(diff5_weight)
        # outfeature=self.sigmoid(outfeature)
        # outfeature = self.softmax(outfeature)
        # print('outfeature',outfeature.shape)
        # return [outfeature,diff4_weight,diff5_weight], decode_3

        return outfeature, decode_3
class UCDNetNew(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UCDNetNew, self).__init__()
        hideen_num = [16, 32, 64, 128]
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_1_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_2_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_3_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_4_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU()
        )

        self.conv_diff_1 = nn.Sequential(

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            # nn.BatchNorm2d(16),
            # nn.ReLU()
        )
        self.diff1_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU()
        )
        self.diff2_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            # nn.BatchNorm2d(64),
            # nn.ReLU()
        )
        self.diff3_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            # nn.BatchNorm2d(128),
            # nn.ReLU()
        )
        self.diff4_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            # nn.BatchNorm2d(64),
            # nn.ReLU()
        )
        self.diff5_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        self.NSSP = NSSP(192, 192, 32)
        self.ASSP = ASSP(192, 192)
        self.ASSPconv = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=1, padding=0),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            # # nn.BatchNorm2d(64),
            # nn.ReLU()
        )

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(),

            nn.Conv2d(in_channels=160, out_channels=80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(),

            nn.Conv2d(in_channels=80, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        # post_data=pre_data
        ###########Stage 1
        feature_T1_11 = self.conv_1_1(pre_data)  # conv
        feature_T1_12 = self.conv_1_2(feature_T1_11)  # relu+conv+relu
        feature_T2_11 = self.conv_1_1(post_data)
        feature_T2_12 = self.conv_1_2(feature_T2_11)

        # diff1 = torch.abs(feature_T1_11 - feature_T2_11)
        diff1 = torch.pow(feature_T1_11 - feature_T2_11,2)
        diff1 = self.conv_diff_1(diff1)
        # diff1_weight=diff1*self.diff1_se(diff1)
        diff1_weight = self.relu(diff1)
        feature_T1_13 = torch.cat([feature_T1_12, diff1_weight], 1)
        feature_T1_14 = self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12, diff1_weight], 1)
        feature_T2_14 = self.max_pool_1(feature_T2_13)

        ###########Stage 2

        feature_T1_21 = self.conv_2_1(feature_T1_14)  # relu+conv  32
        feature_T1_22 = self.conv_2_2(feature_T1_21)  # relu+conv+relu
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        feature_T2_22 = self.conv_2_2(feature_T2_21)

        # diff2 = torch.abs(feature_T1_21 - feature_T2_21)
        diff2 = torch.pow(feature_T1_21 - feature_T2_21,2)
        diff2 = self.conv_diff_2(diff2)
        # diff2_weight = diff2 * self.diff2_se(diff2)
        diff2_weight = self.relu(diff2)

        feature_T1_23 = torch.cat([feature_T1_22, diff2_weight], 1)
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22, diff2_weight], 1)
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        ##########Stage 3
        feature_T1_31 = self.conv_3_1(feature_T1_24)  # relu+conv
        feature_T1_32 = self.conv_3_2(feature_T1_31)  # relu+conv+relu
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        feature_T2_32 = self.conv_3_2(feature_T2_31)  # 64

        # diff3 = torch.abs(feature_T1_31 - feature_T2_31)
        diff3 = torch.pow(feature_T1_31 - feature_T2_31,2)
        diff3 = self.conv_diff_3(diff3)
        # diff3_weight = diff3 * self.diff3_se(diff3)
        diff3_weight = self.relu(diff3)

        feature_T1_33 = torch.cat([feature_T1_32, diff3_weight], 1)
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32, diff3_weight], 1)  # 128
        feature_T2_34 = self.max_pool_3(feature_T2_33)

        ##########stage 4
        feature_T1_41 = self.conv_4_1(feature_T1_34)  # relu+conv
        feature_T1_42 = self.conv_4_2(feature_T1_41)  # relu+conv+relu
        feature_T2_41 = self.conv_4_1(feature_T2_34)
        feature_T2_42 = self.conv_4_2(feature_T2_41)

        # diff4 = torch.abs(feature_T1_41 - feature_T2_41)
        diff4 = torch.pow(feature_T1_41 - feature_T2_41,2)
        diff4 = self.conv_diff_4(diff4)
        # diff4_weight = diff4 * self.diff4_se(diff4)
        diff4_weight = self.relu(diff4)

        feature_T1_43 = torch.cat([feature_T1_42, diff4_weight], 1)
        feature_T1_44 = self.conv_4_3(feature_T1_43)  # conv
        feature_T2_43 = torch.cat([feature_T2_42, diff4_weight], 1)
        feature_T2_44 = self.conv_4_3(feature_T2_43)

        #########stage Bottleneck
        diff5 = torch.abs(feature_T1_44 - feature_T2_44)  # 64
        diff5 = self.conv_diff_5(diff5)
        # diff5_weight = diff5 * self.diff5_se(diff5)
        diff5_weight = diff5
        feature_Bottleneck = torch.cat([feature_T1_44, feature_T2_44, diff5_weight], 1)
        feature_Bottleneck = self.relu(feature_Bottleneck)
        feature_Bottleneck_conv = self.ASSPconv(feature_Bottleneck)#conv  192
        feature_Bottleneck_assp = self.ASSP(feature_Bottleneck) #no relu nobn
        # feature_Bottleneck_plus=torch.cat([feature_Bottleneck,feature_Bottleneck_assp],1)
        feature_Bottleneck_plus = feature_Bottleneck_assp+feature_Bottleneck_conv
        feature_Bottleneck_plus = self.relu(feature_Bottleneck_plus)

        # stage decode_1
        decode_1 = self.up_sample_1(feature_Bottleneck_plus)
        # decode_1 = torch.cat([feature_T1_33, feature_T2_33, decode_1], 1)  # 320
        decode_1 = torch.cat([diff3_weight, decode_1], 1)  # 96+64
        # stage decode_2
        decode_2 = self.deconv_1(decode_1)  # 32
        # decode_2 = torch.cat([feature_T1_23, feature_T2_23, decode_2], 1)
        decode_2 = torch.cat([diff2_weight, decode_2], 1)  # 32+32
        # stage decode_3
        decode_3 = self.deconv_2(decode_2)  # 16
        decode_3 = torch.cat([diff1_weight, decode_3], 1)  # 16+16

        outfeature = self.deconv_3(decode_3)

        return outfeature, outfeature
class UCDNetNew_SE(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UCDNetNew_SE, self).__init__()
        hideen_num = [16, 32, 64, 128]
        reluAC=nn.ReLU()
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_1_2 = nn.Sequential(
            reluAC,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            reluAC
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_2_2 = nn.Sequential(
            reluAC,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            reluAC
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3_1 = nn.Sequential(
            reluAC,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_3_2 = nn.Sequential(
            reluAC,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            reluAC,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            reluAC
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4_1 = nn.Sequential(
            reluAC,
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            reluAC,
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_4_2 = nn.Sequential(
            reluAC,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            reluAC,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            reluAC
        )
        # self.conv_4_3 = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(64),
        #     # nn.ReLU()
        # )

        self.conv_diff_1 = nn.Sequential(

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            # nn.BatchNorm2d(16),
            # nn.ReLU()
        )
        self.diff1_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU()
        )
        self.diff2_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            # nn.BatchNorm2d(64),
            # nn.ReLU()
        )
        self.diff3_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            # nn.BatchNorm2d(128),
            # nn.ReLU()
        )
        self.diff4_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        # self.conv_diff_5 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
        #     # nn.BatchNorm2d(64),
        #     # nn.ReLU()
        # )
        # self.diff5_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.Sigmoid()
        # )
        # self.NSSP = NSSP(192, 192, 32)
        self.ASSP = ASSP(192, 192)
        self.ASSPconv = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=192+64, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(),

            nn.Conv2d(in_channels=160, out_channels=80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(),

            nn.Conv2d(in_channels=80, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU6()

    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        # post_data=pre_data
        ###########Stage 1
        feature_T1_11 = self.conv_1_1(pre_data)  # conv
        feature_T1_12 = self.conv_1_2(feature_T1_11)  # relu+conv+relu
        feature_T2_11 = self.conv_1_1(post_data)
        feature_T2_12 = self.conv_1_2(feature_T2_11)

        diff1 = torch.abs(feature_T1_11 - feature_T2_11)
        # diff1 = torch.pow(feature_T1_11 - feature_T2_11,2)
        diff1 = self.conv_diff_1(diff1)#conv
        diff1=diff1*self.diff1_se(diff1)
        diff1_weight = self.relu(diff1)
        feature_T1_13 = torch.cat([feature_T1_12, diff1_weight], 1)
        feature_T1_14 = self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12, diff1_weight], 1)
        feature_T2_14 = self.max_pool_1(feature_T2_13)

        ###########Stage 2

        feature_T1_21 = self.conv_2_1(feature_T1_14)  # relu+conv  32
        feature_T1_22 = self.conv_2_2(feature_T1_21)  # relu+conv+relu
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        feature_T2_22 = self.conv_2_2(feature_T2_21)

        diff2 = torch.abs(feature_T1_21 - feature_T2_21)
        # diff2 = torch.pow(feature_T1_21 - feature_T2_21,2)
        diff2 = self.conv_diff_2(diff2)#conv
        diff2 = diff2 * self.diff2_se(diff2)
        diff2_weight = self.relu(diff2)

        feature_T1_23 = torch.cat([feature_T1_22, diff2_weight], 1)
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22, diff2_weight], 1)
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        ##########Stage 3
        feature_T1_31 = self.conv_3_1(feature_T1_24)  # relu+conv
        feature_T1_32 = self.conv_3_2(feature_T1_31)  # relu+conv+relu
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        feature_T2_32 = self.conv_3_2(feature_T2_31)  # 64

        diff3 = torch.abs(feature_T1_31 - feature_T2_31)
        # diff3 = torch.pow(feature_T1_31 - feature_T2_31,2)
        diff3 = self.conv_diff_3(diff3)#conv
        diff3 = diff3 * self.diff3_se(diff3)
        diff3_weight = self.relu(diff3)

        feature_T1_33 = torch.cat([feature_T1_32, diff3_weight], 1)
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32, diff3_weight], 1)  # 128
        feature_T2_34 = self.max_pool_3(feature_T2_33)

        ##########stage 4
        feature_T1_41 = self.conv_4_1(feature_T1_34)  # relu+conv
        feature_T1_42 = self.conv_4_2(feature_T1_41)  # relu+conv+relu
        feature_T2_41 = self.conv_4_1(feature_T2_34)#128
        feature_T2_42 = self.conv_4_2(feature_T2_41)

        diff4 = torch.abs(feature_T1_41 - feature_T2_41)
        # diff4 = torch.pow(feature_T1_41 - feature_T2_41,2)
        diff4 = self.conv_diff_4(diff4)#conv 128
        diff4 = diff4 * self.diff4_se(diff4)
        diff4_weight = self.relu(diff4)

        # feature_T1_43 = torch.cat([feature_T1_42, diff4_weight], 1)
        # feature_T1_44 = self.conv_4_3(feature_T1_43)  # conv
        # feature_T2_43 = torch.cat([feature_T2_42, diff4_weight], 1)
        # feature_T2_44 = self.conv_4_3(feature_T2_43)
        #
        # #########stage Bottleneck
        # # diff5 = torch.abs(feature_T1_44 - feature_T2_44)  # 64
        # diff5 = torch.pow(feature_T1_44 - feature_T2_44,2)  # 64
        # diff5 = self.conv_diff_5(diff5)
        # diff5 = diff5 * self.diff5_se(diff5)
        # diff5_weight = diff5
        feature_Bottleneck = torch.cat([feature_T1_42, feature_T2_42, diff4_weight], 1)
        feature_Bottleneck_conv = self.ASSPconv(feature_Bottleneck)#conv  192
        feature_Bottleneck_assp = self.ASSP(feature_Bottleneck) #no relu nobn
        feature_Bottleneck_plus=torch.cat([feature_Bottleneck_conv,feature_Bottleneck_assp],1)
        # feature_Bottleneck_plus = feature_Bottleneck_assp+feature_Bottleneck_conv
        feature_Bottleneck_plus = self.relu(feature_Bottleneck_plus)

        # stage decode_1
        decode_1 = self.up_sample_1(feature_Bottleneck_plus)
        # decode_1 = torch.cat([feature_T1_33, feature_T2_33, decode_1], 1)  # 320
        decode_1 = torch.cat([diff3_weight, decode_1], 1)  # 96+64
        # stage decode_2
        decode_2 = self.deconv_1(decode_1)  # 32
        # decode_2 = torch.cat([feature_T1_23, feature_T2_23, decode_2], 1)
        decode_2 = torch.cat([diff2_weight, decode_2], 1)  # 32+32
        # stage decode_3
        decode_3 = self.deconv_2(decode_2)  # 16
        decode_3 = torch.cat([diff1_weight, decode_3], 1)  # 16+16

        outfeature = self.deconv_3(decode_3)

        return outfeature, outfeature

class UCDNetNew_AG(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UCDNetNew_AG, self).__init__()
        hideen_num = [16, 32, 64, 128]
        reluAC=nn.ReLU()
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_1_2 = nn.Sequential(
            reluAC,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            reluAC
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_2_2 = nn.Sequential(
            reluAC,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            reluAC
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3_1 = nn.Sequential(
            reluAC,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_3_2 = nn.Sequential(
            reluAC,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            reluAC,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            reluAC
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4_1 = nn.Sequential(
            reluAC,
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_4_2 = nn.Sequential(
            reluAC,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            reluAC,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            reluAC
        )
        # self.conv_4_3 = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
        #     reluAC,
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(64),
        #     # nn.ReLU()
        # )

        self.conv_diff_1 = nn.Sequential(

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            # nn.BatchNorm2d(16),
            # nn.ReLU()
        )
        self.diff1_AG = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(16, 16 // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16 // 2, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        # self.diff1_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(16, 16, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 16, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU()
        )
        # self.diff2_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(32, 32, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.diff2_AG = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(32, 32 // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32 // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            # nn.BatchNorm2d(64),
            # nn.ReLU()
        )
        # self.diff3_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.diff3_AG = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(64, 64 // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64 // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            # nn.BatchNorm2d(128),
            # nn.ReLU()
        )
        # self.diff4_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.diff4_AG = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(64, 64 // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64 // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # self.conv_diff_5 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
        #     # nn.BatchNorm2d(64),
        #     # nn.ReLU()
        # )
        # self.diff5_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.Sigmoid()
        # )
        # self.diff5_AG = nn.Sequential(
        #     # nn.ReLU(),
        #     nn.Conv2d(64, 64 // 2, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64 // 2, 1, kernel_size=1),
        #     nn.Sigmoid()
        # )
        #self.NSSP = NSSP(192, 192, 32)
        self.ASSP = ASSP(192, 192)
        self.ASSPconv = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Conv2d(96, 64, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.BatchNorm2d(64),
        )

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=192+64, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(),

            nn.Conv2d(in_channels=160, out_channels=80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(),

            nn.Conv2d(in_channels=80, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1)
        )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.relu6 = nn.ReLU6()

    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        # post_data=pre_data
        ###########Stage 1
        feature_T1_11 = self.conv_1_1(pre_data)  # conv
        feature_T1_12 = self.conv_1_2(feature_T1_11)  # relu+conv+relu
        feature_T2_11 = self.conv_1_1(post_data)
        feature_T2_12 = self.conv_1_2(feature_T2_11)

        diff1 = torch.abs(feature_T1_11 - feature_T2_11)
        # diff1 = torch.pow(feature_T1_11 - feature_T2_11,2)
        diff1 = self.conv_diff_1(diff1)#conv
        diff1_AG = self.diff1_AG(diff1)
        diff1_weight = self.relu(diff1 * diff1_AG)
        # diff1 = diff1 * self.diff1_se(diff1)
        # diff1_weight = self.relu(diff1)
        feature_T1_13 = torch.cat([feature_T1_12*diff1_AG, diff1_weight], 1)
        # feature_T1_13=feature_T1_12+diff1_weight
        feature_T1_14 = self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12*diff1_AG, diff1_weight], 1)
        # feature_T2_13=feature_T2_12+diff1_weight
        feature_T2_14 = self.max_pool_1(feature_T2_13)

        ###########Stage 2

        feature_T1_21 = self.conv_2_1(feature_T1_14)  # relu+conv  32
        feature_T1_22 = self.conv_2_2(feature_T1_21)  # relu+conv+relu
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        feature_T2_22 = self.conv_2_2(feature_T2_21)

        # diff2 = torch.abs(feature_T1_21 - feature_T2_21)
        diff2 = torch.pow(feature_T1_21 - feature_T2_21,2)
        diff2 = self.conv_diff_2(diff2)  # conv
        diff2_AG = self.diff2_AG(diff2)
        diff2_weight = self.relu(diff2 * diff2_AG)

        feature_T1_23 = torch.cat([feature_T1_22* diff2_AG, diff2_weight], 1)
        # feature_T1_23 = feature_T1_22 + diff2_weight
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22* diff2_AG, diff2_weight], 1)
        # feature_T2_23 = feature_T2_22 + diff2_weight
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        ##########Stage 3
        feature_T1_31 = self.conv_3_1(feature_T1_24)  # relu+conv
        feature_T1_32 = self.conv_3_2(feature_T1_31)  # relu+conv+relu
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        feature_T2_32 = self.conv_3_2(feature_T2_31)  # 64

        # diff3 = torch.abs(feature_T1_31 - feature_T2_31)
        diff3 = torch.pow(feature_T1_31 - feature_T2_31,2)
        diff3 = self.conv_diff_3(diff3)  # conv
        diff3_AG = self.diff3_AG(diff3)
        diff3_weight = self.relu(diff3 * diff3_AG)

        feature_T1_33 = torch.cat([feature_T1_32 * diff3_AG, diff3_weight], 1)
        # feature_T1_33=feature_T1_32 + diff3_weight
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32 * diff3_AG, diff3_weight], 1)  # 128
        # feature_T2_33=feature_T2_32 + diff3_weight
        feature_T2_34 = self.max_pool_3(feature_T2_33)

        ##########stage 4
        feature_T1_41 = self.conv_4_1(feature_T1_34)  # relu+conv
        feature_T1_42 = self.conv_4_2(feature_T1_41)  # relu+conv+relu64
        feature_T2_41 = self.conv_4_1(feature_T2_34)
        feature_T2_42 = self.conv_4_2(feature_T2_41)

        diff4 = torch.abs(feature_T1_41 - feature_T2_41)
        # diff4 = torch.pow(feature_T1_41 - feature_T2_41,2)
        diff4 = self.conv_diff_4(diff4)  # conv
        diff4_AG = self.diff4_AG(diff4)
        diff4_weight = self.relu(diff4 * diff4_AG)

        # feature_T1_43 = torch.cat([feature_T1_42 * diff4_AG, diff4_weight], 1)
        # feature_T1_43=feature_T1_42 + diff4_weight
        # feature_T1_44 = self.conv_4_3(feature_T1_43)  # conv-relu-conv
        # feature_T2_43 = torch.cat([feature_T2_42 * diff4_AG, diff4_weight], 1)
        # feature_T2_43=feature_T2_42 + diff4_weight
        # feature_T2_44 = self.conv_4_3(feature_T2_43)

        #########stage Bottleneck
        # diff5 = torch.abs(feature_T1_44 - feature_T2_44)  # 64
        # diff5 = torch.pow(feature_T1_44 - feature_T2_44,2)  # 64
        # diff5 = self.conv_diff_5(diff5)  # conv
        # diff5_AG = self.diff5_AG(diff5)
        # diff5_weight = self.relu(diff5 * diff5_AG)

        # feature_Bottleneck = torch.cat([feature_T1_44 * diff5_AG, feature_T2_44 * diff5_AG, diff5_weight], 1)
        feature_Bottleneck = torch.cat([feature_T1_42, feature_T2_42, diff4_weight], 1)
        # feature_Bottleneck = self.relu(feature_Bottleneck)

        feature_Bottleneck_conv = self.ASSPconv(feature_Bottleneck)#conv  192
        feature_Bottleneck_assp = self.ASSP(feature_Bottleneck) #no relu nobn
        feature_Bottleneck_plus=torch.cat([feature_Bottleneck_conv,feature_Bottleneck_assp],1)
        # feature_Bottleneck_plus = feature_Bottleneck_assp+feature_Bottleneck_conv
        # feature_Bottleneck_plus = self.relu(feature_Bottleneck_plus)

        # stage decode_1
        decode_1 = self.up_sample_1(feature_Bottleneck_plus)
        # decode_1 = torch.cat([feature_T1_33, feature_T2_33, decode_1], 1)  # 320
        decode_1 = torch.cat([diff3_weight, decode_1], 1)  # 96+64   128+64
        # stage decode_2
        decode_2 = self.deconv_1(decode_1)  # 32
        # decode_2 = torch.cat([feature_T1_23, feature_T2_23, decode_2], 1)
        decode_2 = torch.cat([diff2_weight, decode_2], 1)  # 32+32   64+32
        # stage decode_3
        decode_3 = self.deconv_2(decode_2)  # 16
        decode_3 = torch.cat([diff1_weight, decode_3], 1)  # 16+16  32+16

        outfeature = self.deconv_3(decode_3)

        return outfeature, outfeature

class UCDNetNew_AG2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UCDNetNew_AG2, self).__init__()
        hideen_num = [16, 32, 64, 128]
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_1_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_2_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_3_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.conv_4_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU()
        )

        self.conv_diff_1 = nn.Sequential(

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            # nn.BatchNorm2d(16),
            # nn.ReLU()
        )
        self.diff1_AG = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(16, 16 // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16 // 2, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.diff1_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU()
        )
        self.diff2_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff2_AG = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, 32 // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32//2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            # nn.BatchNorm2d(64),
            # nn.ReLU()
        )
        self.diff3_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff3_AG = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 64 // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64//2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            # nn.BatchNorm2d(128),
            # nn.ReLU()
        )
        self.diff4_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff4_AG = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(128, 128 // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128//2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            # nn.BatchNorm2d(64),
            # nn.ReLU()
        )
        self.diff5_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff5_AG = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 64 // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64//2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.NSSP = NSSP(192, 192, 32)
        self.ASSP = ASSP(192, 192)
        self.ASSPconv = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=1, padding=0),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            # # nn.BatchNorm2d(64),
            # nn.ReLU()
        )

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(),

            nn.Conv2d(in_channels=160, out_channels=80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(),

            nn.Conv2d(in_channels=80, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        # post_data=pre_data
        ###########Stage 1
        feature_T1_11 = self.conv_1_1(pre_data)  # conv
        feature_T1_12 = self.conv_1_2(feature_T1_11)  # relu+conv+relu
        feature_T2_11 = self.conv_1_1(post_data)
        feature_T2_12 = self.conv_1_2(feature_T2_11)

        # diff1 = torch.abs(feature_T1_11 - feature_T2_11)
        diff1 = torch.pow(feature_T1_11 - feature_T2_11,2)
        # diff1 = self.conv_diff_1(diff1)
        diff1=diff1*self.diff1_se(diff1)
        diff1_weight = self.relu(diff1)
        feature_T1_13 = torch.cat([feature_T1_12, diff1_weight], 1)
        feature_T1_14 = self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12, diff1_weight], 1)
        feature_T2_14 = self.max_pool_1(feature_T2_13)

        ###########Stage 2

        feature_T1_21 = self.conv_2_1(feature_T1_14)  # relu+conv  32
        feature_T1_22 = self.conv_2_2(feature_T1_21)  # relu+conv+relu
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        feature_T2_22 = self.conv_2_2(feature_T2_21)

        # diff2 = torch.abs(feature_T1_21 - feature_T2_21)
        diff2 = torch.pow(feature_T1_21 - feature_T2_21,2)
        # diff2 = self.conv_diff_2(diff2)
        diff2 = diff2 * self.diff2_se(diff2)
        diff2_weight = self.relu(diff2)

        feature_T1_23 = torch.cat([feature_T1_22, diff2_weight], 1)
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22, diff2_weight], 1)
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        ##########Stage 3
        feature_T1_31 = self.conv_3_1(feature_T1_24)  # relu+conv
        feature_T1_32 = self.conv_3_2(feature_T1_31)  # relu+conv+relu
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        feature_T2_32 = self.conv_3_2(feature_T2_31)  # 64

        # diff3 = torch.abs(feature_T1_31 - feature_T2_31)
        diff3 = torch.pow(feature_T1_31 - feature_T2_31,2)
        # diff3 = self.conv_diff_3(diff3)
        diff3 = diff3 * self.diff3_se(diff3)
        diff3_weight = self.relu(diff3)

        feature_T1_33 = torch.cat([feature_T1_32, diff3_weight], 1)
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32, diff3_weight], 1)  # 128
        feature_T2_34 = self.max_pool_3(feature_T2_33)

        ##########stage 4
        feature_T1_41 = self.conv_4_1(feature_T1_34)  # relu+conv
        feature_T1_42 = self.conv_4_2(feature_T1_41)  # relu+conv+relu
        feature_T2_41 = self.conv_4_1(feature_T2_34)
        feature_T2_42 = self.conv_4_2(feature_T2_41)

        # diff4 = torch.abs(feature_T1_41 - feature_T2_41)
        diff4 = torch.pow(feature_T1_41 - feature_T2_41,2)
        # diff4 = self.conv_diff_4(diff4)
        diff4 = diff4 * self.diff4_se(diff4)
        diff4_weight = self.relu(diff4)

        feature_T1_43 = torch.cat([feature_T1_42, diff4_weight], 1)
        feature_T1_44 = self.conv_4_3(feature_T1_43)  # conv
        feature_T2_43 = torch.cat([feature_T2_42, diff4_weight], 1)
        feature_T2_44 = self.conv_4_3(feature_T2_43)

        #########stage Bottleneck
        # diff5 = torch.abs(feature_T1_44 - feature_T2_44)  # 64
        diff5 = torch.pow(feature_T1_44 - feature_T2_44,2)  # 64

        # diff5 = self.conv_diff_5(diff5)
        diff5 = diff5 * self.diff5_se(diff5)
        diff5_weight = diff5
        feature_Bottleneck = torch.cat([feature_T1_44, feature_T2_44, diff5_weight], 1)
        feature_Bottleneck = self.relu(feature_Bottleneck)
        feature_Bottleneck_conv = self.ASSPconv(feature_Bottleneck)#conv  192
        feature_Bottleneck_assp = self.ASSP(feature_Bottleneck) #no relu nobn
        # feature_Bottleneck_plus=torch.cat([feature_Bottleneck,feature_Bottleneck_assp],1)
        feature_Bottleneck_plus = feature_Bottleneck_assp+feature_Bottleneck_conv
        feature_Bottleneck_plus = self.relu(feature_Bottleneck_plus)

        # stage decode_1
        decode_1 = self.up_sample_1(feature_Bottleneck_plus)
        # decode_1 = torch.cat([feature_T1_33, feature_T2_33, decode_1], 1)  # 320
        decode_1 = torch.cat([diff3_weight, decode_1], 1)  # 96+64
        # stage decode_2
        decode_2 = self.deconv_1(decode_1)  # 32
        # decode_2 = torch.cat([feature_T1_23, feature_T2_23, decode_2], 1)
        decode_2 = torch.cat([diff2_weight, decode_2], 1)  # 32+32
        # stage decode_3
        decode_3 = self.deconv_2(decode_2)  # 16
        decode_3 = torch.cat([diff1_weight, decode_3], 1)  # 16+16

        outfeature = self.deconv_3(decode_3)

        return outfeature, outfeature

class UCDNet_ASSP_AG(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UCDNet_ASSP_AG, self).__init__()
        hideen_num = [16, 32, 64, 128]
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU6()
        )
        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            # nn.ReLU6()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU6()
        )
        self.conv_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU6()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU6()
        )
        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),

            # nn.ReLU6()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU6()
        )
        self.conv_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU6(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # nn.ReLU6()
        )
        self.conv_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU6()
        )

        self.conv_diff_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            # nn.BatchNorm2d(16),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff1_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff1_AG= nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(16, 16//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16//2, 1, kernel_size=3, padding=1,bias=True),
            nn.Sigmoid()
        )

        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff2_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 32 , kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32 , 32, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff2_AG = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, 32 // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32 // 2, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0)
            # nn.BatchNorm2d(64),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff3_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff3_AG = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 64 // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64 // 2, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0)
            # nn.BatchNorm2d(128),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff4_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff4_AG = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(128, 128 // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128 // 2, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.conv_diff_5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            # nn.BatchNorm2d(64),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff5_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff5_AG = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 64 // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64 // 2, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.NSSP = NSSP(192, 192, 32)
        self.ASSP = ASSP(192, 192)
        self.ASSPconv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            # # nn.BatchNorm2d(64),
            # nn.ReLU()
        )

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            # nn.BatchNorm2d(192),
            nn.ReLU6(),

            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU6(),
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU6(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU6(),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU6(),

            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            # nn.BatchNorm2d(32),
            # nn.ReLU6()
        )
        self.deconv_2 = nn.Sequential(
            nn.ReLU6(),
            nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU6(),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU6(),

            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            # nn.BatchNorm2d(16),
            # nn.ReLU6()
        )
        self.deconv_3 = nn.Sequential(
            nn.ReLU6(),
            nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU6(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )
        self.updiff3=nn.Upsample(scale_factor=4, mode='bilinear',align_corners=True)
        self.updiff4=nn.Upsample(scale_factor=8, mode='bilinear',align_corners=True)
        self.updiff5=nn.Upsample(scale_factor=8, mode='bilinear',align_corners=True)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.relu6=nn.ReLU6()
        self.softmax = nn.Softmax()
        self.softsign=nn.Softsign()
        self.tanh=nn.Tanh()

    def forward(self, pre_data,post_data):
        #####################
        # decoder
        #####################
        # post_data=pre_data
        ####Stage 1
        feature_T1_11 = self.conv_1_1(pre_data)#conv+relu
        feature_T1_12 = self.conv_1_2(feature_T1_11)#conv
        feature_T2_11 = self.conv_1_1(post_data)
        feature_T2_12 = self.conv_1_2(feature_T2_11)

        # diff1 = torch.abs(feature_T1_11 - feature_T2_11)
        diff1 = torch.pow(feature_T1_11 - feature_T2_11,2)
        # diff1 = self.conv_diff_1(diff1)#conv
        # diff1_SE = self.diff1_se(diff1)
        diff1_AG = self.diff1_AG(diff1)#relu+conv+relu+conv+sigmoid
        # diff1_weight = self.relu6(diff1 * diff1_SE * diff1_AG)
        diff1_weight = self.relu6(diff1 * diff1_AG)

        feature_T1_12 = self.relu6(feature_T1_12 * diff1_AG)
        feature_T2_12 = self.relu6(feature_T2_12 * diff1_AG)

        feature_T1_13 = torch.cat([feature_T1_12, diff1_weight], 1)
        feature_T1_14 = self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12, diff1_weight], 1)
        feature_T2_14 = self.max_pool_1(feature_T2_13)


        ####Stage 2
        feature_T1_21 = self.conv_2_1(feature_T1_14)
        feature_T1_22 = self.conv_2_2(feature_T1_21)
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        feature_T2_22 = self.conv_2_2(feature_T2_21)

        # diff2 = torch.abs(feature_T1_21 - feature_T2_21)
        diff2 = torch.pow(feature_T1_21 - feature_T2_21,2)

        # diff2 = self.conv_diff_2(diff2)
        # diff2_SE = self.diff2_se(diff2)
        diff2_AG = self.diff2_AG(diff2)#relu+conv+relu+conv+sigmoid
        # diff2_weight = self.relu6(diff2 * diff2_SE * diff2_AG)
        diff2_weight = self.relu6(diff2 * diff2_AG)

        feature_T1_22 = self.relu6(feature_T1_22 * diff2_AG)
        feature_T2_22 = self.relu6(feature_T2_22 * diff2_AG)

        feature_T1_23 = torch.cat([feature_T1_22, diff2_weight], 1)
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22, diff2_weight], 1)
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        ######Stage3
        feature_T1_31 = self.conv_3_1(feature_T1_24)
        feature_T1_32 = self.conv_3_2(feature_T1_31)
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        feature_T2_32 = self.conv_3_2(feature_T2_31)

        # diff3 = torch.abs(feature_T1_31 - feature_T2_31)
        diff3 = torch.pow(feature_T1_31 - feature_T2_31,2)

        # diff3 = self.conv_diff_3(diff3)
        # diff3_SE = self.diff3_se(diff3)
        diff3_AG = self.diff3_AG(diff3)#relu+conv+relu+conv+sigmoid
        # diff3_weight = self.relu6(diff3 * diff3_SE * diff3_AG)
        diff3_weight = self.relu6(diff3 * diff3_AG)

        feature_T1_32 = self.relu6(feature_T1_32 * diff3_AG)
        feature_T2_32 = self.relu6(feature_T2_32 * diff3_AG)

        feature_T1_33 = torch.cat([feature_T1_32, diff3_weight], 1)
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32, diff3_weight], 1)  # 128
        feature_T2_34 = self.max_pool_3(feature_T2_33)
        ##############Stage 4
        feature_T1_41 = self.conv_4_1(feature_T1_34)
        feature_T1_42 = self.conv_4_2(feature_T1_41)
        feature_T2_41 = self.conv_4_1(feature_T2_34)
        feature_T2_42 = self.conv_4_2(feature_T2_41)

        # diff4 = torch.abs(feature_T1_41 - feature_T2_41)
        diff4 = torch.pow(feature_T1_41 - feature_T2_41,2)
        # diff4 = self.conv_diff_4(diff4)
        # diff4_SE = self.diff4_se(diff4)
        diff4_AG = self.diff4_AG(diff4)#relu+conv+relu+conv+sigmoid
        # diff4_weight = self.relu6(diff4 * diff4_SE * diff4_AG)
        diff4_weight = self.relu6(diff4 * diff4_AG)

        feature_T1_42 = self.relu6(feature_T1_42 * diff4_AG)
        feature_T2_42 = self.relu6(feature_T2_42 * diff4_AG)

        feature_T1_43 = torch.cat([feature_T1_42, diff4_weight], 1)
        feature_T1_44 = self.conv_4_3(feature_T1_43)#conv 64
        feature_T2_43 = torch.cat([feature_T2_42, diff4_weight], 1)
        feature_T2_44 = self.conv_4_3(feature_T2_43)

        ##############Stage ASSP

        # feature_T1_45 = self.ASSP(feature_T1_44)#64,32,32
        feature_T1_45 = self.ASSPconv(feature_T1_44)#conv
        # feature_T2_45 = self.ASSP(feature_T2_44)
        feature_T2_45 = self.ASSPconv(feature_T2_44)

        # diff5 = torch.abs(feature_T1_45 - feature_T2_45)  # 64
        diff5 = torch.pow(feature_T1_45 - feature_T2_45,2)

        # diff5 = self.conv_diff_5(diff5)#64
        # diff5_SE = self.diff5_se(diff5)
        diff5_AG = self.diff5_AG(diff5)
        # diff5_weight = self.relu6(diff5 * diff5_SE * diff5_AG)
        diff5_weight = self.relu6(diff5 * diff5_AG)

        feature_T1_46 = self.relu6(feature_T1_45 * diff5_AG)
        feature_T2_46 = self.relu6(feature_T2_45 * diff5_AG)
        feature_Bottleneck = torch.cat([feature_T1_46, feature_T2_46, diff5_weight], 1)#192

        feature_Bottleneck=self.ASSP(feature_Bottleneck)
        decode_1 = self.up_sample_1(feature_Bottleneck)
        decode_1 = torch.cat([feature_T1_33, feature_T2_33, decode_1], 1)  # 320

        decode_2 = self.deconv_1(decode_1)
        decode_2 = torch.cat([feature_T1_23, feature_T2_23, decode_2], 1)

        decode_3 = self.deconv_2(decode_2)#150
        decode_3 = torch.cat([feature_T1_13, feature_T2_13, decode_3], 1)#80

        outfeature = self.deconv_3(decode_3)
        diff3_weight=self.updiff3(diff3_weight)
        diff4_weight = self.updiff4(diff4_weight)
        diff5_weight = self.updiff5(diff5_weight)
        # outfeature=self.sigmoid(outfeature)
        # outfeature = self.softmax(outfeature)
        # print('outfeature',outfeature.shape)
        # return [outfeature,diff4_weight,diff5_weight], decode_3

        # return [outfeature, diff4_weight, diff5_weight], decode_3

        return outfeature, decode_3


# conv = UCDNet(3,2)
# conv = conv.to(DEVICE)
# print(summary(conv, (3, 256, 256), batch_size=1))

# self.BasicBlock2 = BasicBlock(hideen_num[0], hideen_num[1])  #64,64,64

class UCDNet_ASSPDR(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UCDNet_ASSPDR, self).__init__()
        hideen_num = [16, 32, 64, 128]
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Dropout2d(p=0.2)
        )
        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=0.2)
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout2d(p=0.2)
        )
        self.conv_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.2)
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout2d(p=0.2)
        )
        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.2)
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout2d(p=0.2)
        )
        self.conv_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.2)
        )
        self.conv_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.2)
        )

        self.conv_diff_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Dropout2d(p=0.2)
        )
        self.diff1_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            # nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.diff2_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 32 , kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32 , 32, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            # nn.ReLU()
            nn.Sigmoid()
        )
        self.diff3_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            # nn.ReLU(),
            nn.Sigmoid()
        )
        self.diff4_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.diff5_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        self.NSSP = NSSP(192, 192, 32)
        self.ASSP = ASSP(64, 64)
        self.ASSPconv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.2)
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout2d(p=0.2)
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=0.2)
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.updiff3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.updiff4 = nn.Upsample(scale_factor=8, mode='bilinear')
    def forward(self, pre_data,post_data):
        #####################
        # decoder
        #####################
        # post_data=pre_data
        feature_T1_11 = self.conv_1_1(pre_data)
        feature_T1_12 = self.conv_1_2(feature_T1_11)
        feature_T2_11 = self.conv_1_1(post_data)
        feature_T2_12 = self.conv_1_2(feature_T2_11)

        diff1 = torch.abs(feature_T1_11 - feature_T2_11)
        diff1 = self.conv_diff_1(diff1)
        diff1_weight=diff1*self.diff1_se(diff1)

        feature_T1_13 = torch.cat([feature_T1_12, diff1_weight], 1)
        feature_T1_14 = self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12, diff1_weight], 1)
        feature_T2_14 = self.max_pool_1(feature_T2_13)

        feature_T1_21 = self.conv_2_1(feature_T1_14)
        feature_T1_22 = self.conv_2_2(feature_T1_21)
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        feature_T2_22 = self.conv_2_2(feature_T2_21)

        diff2 = torch.abs(feature_T1_21 - feature_T2_21)
        diff2 = self.conv_diff_2(diff2)
        diff2_weight = diff2 * self.diff2_se(diff2)

        feature_T1_23 = torch.cat([feature_T1_22, diff2_weight], 1)
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22, diff2_weight], 1)
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        feature_T1_31 = self.conv_3_1(feature_T1_24)
        feature_T1_32 = self.conv_3_2(feature_T1_31)
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        feature_T2_32 = self.conv_3_2(feature_T2_31)

        diff3 = torch.abs(feature_T1_31 - feature_T2_31)
        diff3 = self.conv_diff_3(diff3)
        diff3_weight = diff3 * self.diff3_se(diff3)

        feature_T1_33 = torch.cat([feature_T1_32, diff3_weight], 1)
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32, diff3_weight], 1)  # 128
        feature_T2_34 = self.max_pool_3(feature_T2_33)

        feature_T1_41 = self.conv_4_1(feature_T1_34)
        feature_T1_42 = self.conv_4_2(feature_T1_41)
        feature_T2_41 = self.conv_4_1(feature_T2_34)
        feature_T2_42 = self.conv_4_2(feature_T2_41)

        diff4 = torch.abs(feature_T1_41 - feature_T2_41)
        diff4 = self.conv_diff_4(diff4)
        diff4_weight = diff4 * self.diff4_se(diff4)

        feature_T1_43 = torch.cat([feature_T1_42, diff4_weight], 1)
        feature_T1_44 = self.conv_4_3(feature_T1_43)
        feature_T1_45 = self.ASSP(feature_T1_44)
        # feature_T1_45 = self.ASSPconv(feature_T1_44)
        # feature_T1_45=torch.cat([feature_T1_44,feature_T1_45],1)
        feature_T2_43 = torch.cat([feature_T2_42, diff4_weight], 1)
        feature_T2_44 = self.conv_4_3(feature_T2_43)
        feature_T2_45 = self.ASSP(feature_T2_44)
        # feature_T2_45 = self.ASSPconv(feature_T2_44)

        # feature_T2_45 = torch.cat([feature_T2_44, feature_T2_45], 1)

        diff5 = torch.abs(feature_T1_45 - feature_T2_45)  # 64
        diff5 = self.conv_diff_5(diff5)
        diff5_weight = diff5 * self.diff5_se(diff5)

        feature_Bottleneck = torch.cat([feature_T1_44, feature_T2_44, diff5_weight], 1)

        # print('feature_Bottleneck', feature_Bottleneck.shape)
        # feature_Bottleneck=self.ASSP(feature_Bottleneck)

        decode_1 = self.up_sample_1(feature_Bottleneck)
        decode_1 = torch.cat([feature_T1_33, feature_T2_33, decode_1], 1)  # 320

        decode_2 = self.deconv_1(decode_1)
        decode_2 = torch.cat([feature_T1_23, feature_T2_23, decode_2], 1)

        decode_3 = self.deconv_2(decode_2)
        decode_3 = torch.cat([feature_T1_13, feature_T2_13, decode_3], 1)

        outfeature = self.deconv_3(decode_3)
        # outfeature=self.sigmoid(outfeature)
        # outfeature = self.softmax(outfeature)
        # print('outfeature',outfeature.shape)
        # return outfeature, outfeature

        diff3_weight = self.updiff3(diff3_weight)
        diff4_weight = self.updiff4(diff4_weight)
        # outfeature=self.sigmoid(outfeature)
        # outfeature = self.softmax(outfeature)
        # print('outfeature',outfeature.shape)
        return [outfeature, diff3_weight, diff4_weight], outfeature



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dilation=1, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.convplus = nn.Conv2d(inplanes, planes, stride=stride,kernel_size=1, bias=False)

    def forward(self, x):
        # residual = self.relu(x)
        residual = self.convplus(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class UCDNetRes(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(UCDNetRes, self).__init__()
        # hideen_num = [16, 32, 64, 128]
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),

            nn.ReLU()
        )
        # self.conv_1_2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #
        #     nn.ReLU()
        # )
        self.resconv_1_2=BasicBlock(16,16)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        #stage2
        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),

            nn.ReLU()
        )
        # self.conv_2_2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #
        #     nn.ReLU()
        # )
        self.resconv_2_2 = BasicBlock(32, 32)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
        #stage3
        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),

            nn.ReLU()
        )
        # self.conv_3_2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        self.resconv_3_2 = BasicBlock(64, 64)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)
        #stage4
        self.conv_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),

            nn.ReLU()
        )
        # self.conv_4_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #
        #     nn.ReLU()
        # )
        self.resconv_4_2 = BasicBlock(128, 128)
        self.conv_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.conv_diff_1= nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),

            nn.ReLU()
        )
        self.conv_diff_2 = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),

            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),

            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),

            nn.ReLU()
        )
        # self.NSSP=NSSP(192,192,32)
        # self.ASSP=ASSP(192,192)
        # self.ASSPconv = nn.Sequential(
        #     nn.Conv2d(192, 192, kernel_size=1, stride=1, bias=False),
        #     # nn.BatchNorm2d(192),
        #     nn.ReLU()
        # )
        # self.resconvASSP = BasicBlock(192, 192)#add relu!!!!!!!!
        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),

            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )
        self.sigmoid=nn.Sigmoid()
        self.softmax=nn.Softmax()
        self.relu=nn.ReLU()
    def forward(self, pre_data,post_data):
        #####################
        # decoder
        #####################
        # post_data=pre_data
        feature_T1_11 = self.conv_1_1(pre_data)#conv
        # feature_T1_12 = self.conv_1_2(feature_T1_11)#relu-conv-relu
        feature_T1_12=self.resconv_1_2(feature_T1_11)
        feature_T2_11 = self.conv_1_1(post_data)
        # feature_T2_12 = self.conv_1_2(feature_T2_11)
        feature_T2_12 = self.resconv_1_2(feature_T2_11)

        diff1=torch.abs(feature_T1_11-feature_T2_11)
        diff1=self.conv_diff_1(diff1)#relu-conv-relu

        feature_T1_13=torch.cat([feature_T1_12,diff1],1)
        feature_T1_14=self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12, diff1], 1)
        feature_T2_14 = self.max_pool_1(feature_T2_13)

        feature_T1_21=self.conv_2_1(feature_T1_14)#conv
        # feature_T1_22=self.conv_2_2(feature_T1_21)#relu-conv-relu
        feature_T1_22 = self.resconv_2_2(feature_T1_21)
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        # feature_T2_22 = self.conv_2_2(feature_T2_21)
        feature_T2_22 = self.resconv_2_2(feature_T2_21)

        diff2 = torch.abs(feature_T1_21 - feature_T2_21)
        diff2 = self.conv_diff_2(diff2)

        feature_T1_23 = torch.cat([feature_T1_22, diff2], 1)
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22, diff2], 1)
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        feature_T1_31 = self.conv_3_1(feature_T1_24)#conv
        # feature_T1_32 = self.conv_3_2(feature_T1_31)#relu-conv-relu
        feature_T1_32 = self.resconv_3_2(feature_T1_31)
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        # feature_T2_32 = self.conv_3_2(feature_T2_31)
        feature_T2_32 = self.resconv_3_2(feature_T2_31)#relu-conv-relu



        diff3 = torch.abs(feature_T1_31 - feature_T2_31)
        diff3 = self.conv_diff_3(diff3)

        feature_T1_33 = torch.cat([feature_T1_32, diff3], 1)
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32, diff3], 1)#128
        feature_T2_34 = self.max_pool_3(feature_T2_33)

        feature_T1_41 = self.conv_4_1(feature_T1_34)#conv
        # feature_T1_42 = self.conv_4_2(feature_T1_41)#relu-conv-relu
        feature_T1_42 = self.resconv_4_2(feature_T1_41)  # relu-conv-relu
        feature_T2_41 = self.conv_4_1(feature_T2_34)
        # feature_T2_42 = self.conv_4_2(feature_T2_41)
        feature_T2_42 = self.resconv_4_2(feature_T2_41)  # relu-conv-relu

        diff4 = torch.abs(feature_T1_41 - feature_T2_41)
        diff4 = self.conv_diff_4(diff4)

        feature_T1_43 = torch.cat([feature_T1_42, diff4], 1)
        feature_T1_44=self.conv_4_3(feature_T1_43)#conv
        feature_T2_43 = torch.cat([feature_T2_42, diff4], 1)
        feature_T2_44 = self.conv_4_3(feature_T2_43)

        diff5=torch.abs(feature_T1_44-feature_T2_44)

        feature_Bottleneck=torch.cat([feature_T1_44,feature_T2_44,diff5],1)
        # feature_Bottleneck = self.relu(feature_Bottleneck)
        # print('feature_Bottleneck', feature_Bottleneck.shape)
        # feature_Bottleneck=self.ASSP(feature_Bottleneck)
        # feature_Bottleneck = self.resconvASSP(feature_Bottleneck)

        decode_1=self.up_sample_1(feature_Bottleneck)
        decode_1=torch.cat([feature_T1_33,feature_T2_33,decode_1],1)#320

        decode_2=self.deconv_1(decode_1)
        decode_2=torch.cat([feature_T1_23,feature_T2_23,decode_2],1)

        decode_3=self.deconv_2(decode_2)
        decode_3 = torch.cat([feature_T1_13,feature_T2_13,decode_3],1)

        outfeature=self.deconv_3(decode_3)
        # outfeature=self.sigmoid(outfeature)
        # outfeature = self.softmax(outfeature)
        # print('outfeature',outfeature.shape)
        return [outfeature], outfeature
