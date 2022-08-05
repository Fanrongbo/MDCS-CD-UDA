import torch
import torch.nn as nn
import torch.nn.functional as F

class _PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, pool_size, norm_layer)
                                     for pool_size in pool_sizes])
        out_channelsn=in_channels // 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channelsn,
                      kernel_size=3, padding=1, bias=False),
            # norm_layer(out_channelsn),
            # nn.ReLU(),
            # nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU()
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]

        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output
def assp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channles),
            nn.ReLU())
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
class ASSP(nn.Module):
    def __init__(self, in_channels,out_channels, output_stride=16):
        super(ASSP, self).__init__()

        assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]

        self.aspp1 = assp_branch(in_channels, out_channels, 1, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, out_channels, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, out_channels, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, out_channels, 3, dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.1)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        # # x = self.dropout(self.relu(x))
        x=self.relu(x)
        return x
class SpatialAttention(nn.Module):
    def __init__(self, in_dim,kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(in_dim, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class similarity(nn.Module):
    def __init__(self, in_dim,kernel_size=3):
        super(similarity, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.convin = nn.Conv2d(in_dim, in_dim, kernel_size, padding=padding, bias=False)
        self.conv1 = nn.Conv2d(in_dim, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1,x2):
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x = torch.cat([avg_out, max_out], dim=1)
        x=self.convin(x)

        x = self.sigmoid(self.conv1(x))

        return self.sigmoid(x)
class backbone(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(backbone, self).__init__()

        self.conv_block_1conv1=nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1conv2=nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1bn1=nn.BatchNorm2d(16)
        # self.conv_block_1bn2=nn.BatchNorm2d(16)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)


        self.conv_block_2conv1=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn11=nn.BatchNorm2d(32)
        # self.conv_block_2bn12=nn.BatchNorm2d(32)
        self.conv_block_2conv2=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn21=nn.BatchNorm2d(32)
        # self.conv_block_2bn22=nn.BatchNorm2d(32)

        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3conv1=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn11=nn.BatchNorm2d(64)
        # self.conv_block_3bn12=nn.BatchNorm2d(64)
        self.conv_block_3conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn21=nn.BatchNorm2d(64)
        # self.conv_block_3bn22=nn.BatchNorm2d(64)


        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4conv1= nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn11=nn.BatchNorm2d(128)
        # self.conv_block_4bn12=nn.BatchNorm2d(128)
        self.conv_block_4conv2=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn21 = nn.BatchNorm2d(128)
        # self.conv_block_4bn22=nn.BatchNorm2d(128)

        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=(128*3) , out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
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

        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        c=0
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32+c, out_channels=(32+c)//2, kernel_size=3, padding=1),
            nn.BatchNorm2d((32+c)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=(32+c)//2, out_channels=out_dim, kernel_size=1, padding=0)
        )

    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        H, W = pre_data.size(2), pre_data.size(3)
        feature_11=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(pre_data)))))
        feature_12=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(post_data)))))

        out11 = self.max_pool_1(feature_11)
        out12 = self.max_pool_1(feature_12)

        #stage2
        feature_21 = F.relu(self.conv_block_2bn21(self.conv_block_2conv2(F.relu(self.conv_block_2bn11(self.conv_block_2conv1(out11))))))
        feature_22 = F.relu(self.conv_block_2bn21(self.conv_block_2conv2(F.relu(self.conv_block_2bn11(self.conv_block_2conv1(out12))))))

        out21 = self.max_pool_2(feature_21)
        out22 = self.max_pool_2(feature_22)

        #stage3
        feature_31 = F.relu(self.conv_block_3bn21(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21))))))
        feature_32 = F.relu(self.conv_block_3bn21(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out22))))))


        out31 = self.max_pool_3(feature_31)
        out32 = self.max_pool_3(feature_32)
        #stage4

        feature_41 = F.relu(self.conv_block_4bn21(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31))))))
        feature_42 = F.relu(self.conv_block_4bn21(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out32))))))


        diff4_ori = torch.abs(feature_41 - feature_42)
        diff4 = self.conv_diff_4(diff4_ori)

        out = torch.cat([feature_41 , feature_42 , diff4], 1)  # 128*3

        out=self.max_pool_4(out)

        up_feature_5 = self.up_sample_1(out)#128
        concat_feature_5 = torch.cat([up_feature_5, torch.pow(feature_41 - feature_42,2)], dim=1)
        # concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)#256
        # concat_feature_5 = torch.cat([up_feature_5, diff3_conv], dim=1)#128+64
        feature_5 = self.conv_block_5(concat_feature_5)#64

        up_feature_6 = self.up_sample_2(feature_5)#64
        concat_feature_6 = torch.cat([up_feature_6, torch.pow(feature_31 - feature_32,2)], dim=1)
        # concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)#128
        # concat_feature_6 = torch.cat([up_feature_6, diff2_conv], dim=1)#64+32
        feature_6 = self.conv_block_6(concat_feature_6)#32


        up_feature_7 = self.up_sample_3(feature_6)
        concat_feature_7 = torch.cat([up_feature_7, torch.pow(feature_21 - feature_22,2)], dim=1)
        # concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)#64
        # concat_feature_7 = torch.cat([up_feature_7, diff1_conv], dim=1)#32+16

        feature_7 = self.conv_block_7(concat_feature_7)
        up_feature_8 = self.up_sample_4(feature_7)
        concat_feature_8 = torch.cat([up_feature_8, torch.pow(feature_11 - feature_12,2)], dim=1)
        # concat_feature_8 = torch.cat([up_feature_8, torch.abs(feature_11 - feature_12)], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)

        return [output_feature, diff4_ori], feature_7

class backbone2BN(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(backbone2BN, self).__init__()

        self.conv_block_1conv1=nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1conv2=nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1bn1=nn.BatchNorm2d(16)
        self.conv_block_1bn2=nn.BatchNorm2d(16)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)


        self.conv_block_2conv1=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn11=nn.BatchNorm2d(32)
        self.conv_block_2bn12=nn.BatchNorm2d(32)
        self.conv_block_2conv2=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn21=nn.BatchNorm2d(32)
        self.conv_block_2bn22=nn.BatchNorm2d(32)

        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3conv1=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn11=nn.BatchNorm2d(64)
        self.conv_block_3bn12=nn.BatchNorm2d(64)
        self.conv_block_3conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn21=nn.BatchNorm2d(64)
        self.conv_block_3bn22=nn.BatchNorm2d(64)


        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4conv1= nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn11=nn.BatchNorm2d(128)
        self.conv_block_4bn12=nn.BatchNorm2d(128)
        self.conv_block_4conv2=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn21 = nn.BatchNorm2d(128)
        self.conv_block_4bn22=nn.BatchNorm2d(128)

        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=(128*3) , out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
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

        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        c=0
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32+c, out_channels=(32+c)//2, kernel_size=3, padding=1),
            nn.BatchNorm2d((32+c)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=(32+c)//2, out_channels=out_dim, kernel_size=1, padding=0)
        )

    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        H, W = pre_data.size(2), pre_data.size(3)
        feature_11=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(pre_data)))))
        feature_12=F.relu(self.conv_block_1bn2(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(post_data)))))

        out11 = self.max_pool_1(feature_11)
        out12 = self.max_pool_1(feature_12)

        #stage2
        feature_21 = F.relu(self.conv_block_2bn21(self.conv_block_2conv2(F.relu(self.conv_block_2bn11(self.conv_block_2conv1(out11))))))
        feature_22 = F.relu(self.conv_block_2bn22(self.conv_block_2conv2(F.relu(self.conv_block_2bn12(self.conv_block_2conv1(out12))))))

        out21 = self.max_pool_2(feature_21)
        out22 = self.max_pool_2(feature_22)

        #stage3
        feature_31 = F.relu(self.conv_block_3bn21(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21))))))
        feature_32 = F.relu(self.conv_block_3bn22(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn12(self.conv_block_3conv1(out22))))))


        out31 = self.max_pool_3(feature_31)
        out32 = self.max_pool_3(feature_32)
        #stage4

        feature_41 = F.relu(self.conv_block_4bn21(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31))))))
        feature_42 = F.relu(self.conv_block_4bn22(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn12(self.conv_block_4conv1(out32))))))


        diff4_ori = torch.abs(feature_41 - feature_42)
        diff4 = self.conv_diff_4(diff4_ori)

        out = torch.cat([feature_41 , feature_42 , diff4], 1)  # 128*3

        out=self.max_pool_4(out)

        up_feature_5 = self.up_sample_1(out)#128
        # concat_feature_5 = torch.cat([up_feature_5, torch.pow(feature_41 - feature_42,2)], dim=1)
        concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)#256
        # concat_feature_5 = torch.cat([up_feature_5, diff3_conv], dim=1)#128+64
        feature_5 = self.conv_block_5(concat_feature_5)#64

        up_feature_6 = self.up_sample_2(feature_5)#64
        # concat_feature_6 = torch.cat([up_feature_6, torch.pow(feature_31 - feature_32,2)], dim=1)
        concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)#128
        # concat_feature_6 = torch.cat([up_feature_6, diff2_conv], dim=1)#64+32
        feature_6 = self.conv_block_6(concat_feature_6)#32


        up_feature_7 = self.up_sample_3(feature_6)
        # concat_feature_7 = torch.cat([up_feature_7, torch.pow(feature_21 - feature_22,2)], dim=1)
        concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)#64
        # concat_feature_7 = torch.cat([up_feature_7, diff1_conv], dim=1)#32+16

        feature_7 = self.conv_block_7(concat_feature_7)
        up_feature_8 = self.up_sample_4(feature_7)
        # concat_feature_8 = torch.cat([up_feature_8, torch.pow(feature_11 - feature_12,2)], dim=1)
        concat_feature_8 = torch.cat([up_feature_8, torch.abs(feature_11 - feature_12)], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)

        return [output_feature, diff4_ori], feature_7

class backbone2BN_con2(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(backbone2BN_con2, self).__init__()

        self.conv_block_1conv1=nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1conv2=nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1bn1=nn.BatchNorm2d(16)
        # self.conv_block_1bn2=nn.BatchNorm2d(16)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)


        self.conv_block_2conv1=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn11=nn.BatchNorm2d(32)
        self.conv_block_2bn12=nn.BatchNorm2d(32)
        self.conv_block_2conv2=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn21=nn.BatchNorm2d(32)
        self.conv_block_2bn22=nn.BatchNorm2d(32)

        # self.diff_ag2=SpatialAttention()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3conv1=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn11=nn.BatchNorm2d(64)
        self.conv_block_3bn12=nn.BatchNorm2d(64)
        self.conv_block_3conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn21=nn.BatchNorm2d(64)
        self.conv_block_3bn22=nn.BatchNorm2d(64)

        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4conv1= nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn11=nn.BatchNorm2d(128)
        self.conv_block_4bn12=nn.BatchNorm2d(128)
        self.conv_block_4conv2=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn21 = nn.BatchNorm2d(128)
        self.conv_block_4bn22=nn.BatchNorm2d(128)

        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=(128*3) , out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_diff_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv_diff_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        c=0
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32+c, out_channels=(32+c)//2, kernel_size=3, padding=1),
            nn.BatchNorm2d((32+c)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=(32+c)//2, out_channels=out_dim, kernel_size=1, padding=0)
        )
        self.conv3x3_last = nn.Conv2d(128 * 3, 128 * 3, kernel_size=3, padding=1)
        self.conv3x3_catBN_last = nn.BatchNorm2d(128 * 3)
    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        H, W = pre_data.size(2), pre_data.size(3)
        feature_11=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(pre_data)))))
        feature_12=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(post_data)))))
        # diff1 = torch.abs(feature_11 - feature_12)
        # diff1 = self.conv_diff_1(diff1)
        diff1_ori = torch.abs(feature_11 - feature_12)

        out11 = self.max_pool_1(feature_11)
        out12 = self.max_pool_1(feature_12)

        #stage2
        feature_21 = F.relu(self.conv_block_2bn21(self.conv_block_2conv2(F.relu(self.conv_block_2bn11(self.conv_block_2conv1(out11))))))
        feature_22 = F.relu(self.conv_block_2bn22(self.conv_block_2conv2(F.relu(self.conv_block_2bn12(self.conv_block_2conv1(out12))))))

        diff2_ori = torch.abs(feature_21 - feature_22)
        diff2=self.conv_diff_2(diff2_ori)

        out21 = torch.cat([feature_21, diff2], 1)
        out22 = torch.cat([feature_22, diff2], 1)

        out21 = self.max_pool_2(out21)
        out22 = self.max_pool_2(out22)

        #stage3
        feature_31 = F.relu(self.conv_block_3bn21(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21))))))
        feature_32 = F.relu(self.conv_block_3bn22(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn12(self.conv_block_3conv1(out22))))))

        diff3_ori = torch.abs(feature_31 - feature_32)
        diff3 = self.conv_diff_3(diff3_ori)

        out31 = torch.cat([feature_31, diff3], 1)
        out32 = torch.cat([feature_32, diff3], 1)

        out31 = self.max_pool_3(out31)
        out32 = self.max_pool_3(out32)
        #stage4
        feature_41 = F.relu(self.conv_block_4bn21(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31))))))
        feature_42 = F.relu(self.conv_block_4bn22(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn12(self.conv_block_4conv1(out32))))))

        diff4_ori = torch.abs(feature_41 - feature_42)
        diff4 = self.conv_diff_4(diff4_ori)
        out = torch.cat([feature_41 , feature_42 , diff4], 1)  # 128*3
        out=self.max_pool_4(out)
        out = F.relu(self.conv3x3_catBN_last(self.conv3x3_last(out)))

        up_feature_5 = self.up_sample_1(out)#128
        diff4_con = self.conv_diff_4_2(diff4_ori)
        # concat_feature_5 = torch.cat([up_feature_5, torch.pow(feature_41 - feature_42,2)], dim=1)
        # concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)#256
        concat_feature_5 = torch.cat([up_feature_5, diff4_con], dim=1)#256
        # concat_feature_5 = torch.cat([up_feature_5, diff3_conv], dim=1)#128+64
        feature_5 = self.conv_block_5(concat_feature_5)#64
        up_feature_6 = self.up_sample_2(feature_5)#64
        # concat_feature_6 = torch.cat([up_feature_6, torch.pow(feature_31 - feature_32,2)], dim=1)

        diff3_con = self.conv_diff_3_2(diff3_ori)
        # concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)#128
        # concat_feature_6 = torch.cat([up_feature_6, diff2_conv], dim=1)#64+32
        concat_feature_6 = torch.cat([up_feature_6, diff3_con], dim=1)#128
        feature_6 = self.conv_block_6(concat_feature_6)#32

        up_feature_7 = self.up_sample_3(feature_6)
        # concat_feature_7 = torch.cat([up_feature_7, torch.pow(feature_21 - feature_22,2)], dim=1)
        diff2_con = self.conv_diff_2_2(diff2_ori)
        # concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)#64
        # concat_feature_7 = torch.cat([up_feature_7, diff1_conv], dim=1)#32+16
        concat_feature_7 = torch.cat([up_feature_7, diff2_con], dim=1)  # 64
        feature_7 = self.conv_block_7(concat_feature_7)

        up_feature_8 = self.up_sample_4(feature_7)
        # concat_feature_8 = torch.cat([up_feature_8, torch.pow(feature_11 - feature_12,2)], dim=1)
        diff1_con=self.conv_diff_1(diff1_ori)
        # concat_feature_8 = torch.cat([up_feature_8, torch.abs(feature_11 - feature_12)], dim=1)
        concat_feature_8 = torch.cat([up_feature_8, diff1_con], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)

        return [output_feature, diff4_ori], feature_7


class backbone2BN_con(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(backbone2BN_con, self).__init__()

        self.conv_block_1conv1=nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1conv2=nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1bn1=nn.BatchNorm2d(16)
        # self.conv_block_1bn2=nn.BatchNorm2d(16)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)


        self.conv_block_2conv1=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn11=nn.BatchNorm2d(32)
        self.conv_block_2bn12=nn.BatchNorm2d(32)
        self.conv_block_2conv2=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn21=nn.BatchNorm2d(32)
        self.conv_block_2bn22=nn.BatchNorm2d(32)

        # self.diff_ag2=SpatialAttention()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3conv1=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn11=nn.BatchNorm2d(64)
        self.conv_block_3bn12=nn.BatchNorm2d(64)
        self.conv_block_3conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn21=nn.BatchNorm2d(64)
        self.conv_block_3bn22=nn.BatchNorm2d(64)

        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4conv1= nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn11=nn.BatchNorm2d(128)
        self.conv_block_4bn12=nn.BatchNorm2d(128)
        self.conv_block_4conv2=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn21 = nn.BatchNorm2d(128)
        self.conv_block_4bn22=nn.BatchNorm2d(128)

        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=(128*3) , out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        c=0
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32+c, out_channels=(32+c)//2, kernel_size=3, padding=1),
            nn.BatchNorm2d((32+c)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=(32+c)//2, out_channels=out_dim, kernel_size=1, padding=0)
        )
    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        H, W = pre_data.size(2), pre_data.size(3)
        feature_11=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(pre_data)))))
        feature_12=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(post_data)))))
        # diff1 = torch.abs(feature_11 - feature_12)
        # diff1 = self.conv_diff_1(diff1)

        out11 = self.max_pool_1(feature_11)
        out12 = self.max_pool_1(feature_12)

        #stage2
        feature_21 = F.relu(self.conv_block_2bn21(self.conv_block_2conv2(F.relu(self.conv_block_2bn11(self.conv_block_2conv1(out11))))))
        feature_22 = F.relu(self.conv_block_2bn22(self.conv_block_2conv2(F.relu(self.conv_block_2bn12(self.conv_block_2conv1(out12))))))

        diff2 = torch.abs(feature_21 - feature_22)
        diff2=self.conv_diff_2(diff2)

        out21 = torch.cat([feature_21, diff2], 1)
        out22 = torch.cat([feature_22, diff2], 1)

        out21 = self.max_pool_2(out21)
        out22 = self.max_pool_2(out22)

        #stage3
        feature_31 = F.relu(self.conv_block_3bn21(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21))))))
        feature_32 = F.relu(self.conv_block_3bn22(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn12(self.conv_block_3conv1(out22))))))

        diff3 = torch.abs(feature_31 - feature_32)
        diff3 = self.conv_diff_3(diff3)

        out31 = torch.cat([feature_31, diff3], 1)
        out32 = torch.cat([feature_32, diff3], 1)

        out31 = self.max_pool_3(out31)
        out32 = self.max_pool_3(out32)
        #stage4
        feature_41 = F.relu(self.conv_block_4bn21(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31))))))
        feature_42 = F.relu(self.conv_block_4bn22(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn12(self.conv_block_4conv1(out32))))))

        diff4_ori = torch.abs(feature_41 - feature_42)
        diff4 = self.conv_diff_4(diff4_ori)
        out = torch.cat([feature_41 , feature_42 , diff4], 1)  # 128*3
        out=self.max_pool_4(out)

        up_feature_5 = self.up_sample_1(out)#128
        # concat_feature_5 = torch.cat([up_feature_5, torch.pow(feature_41 - feature_42,2)], dim=1)
        concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)#256
        # concat_feature_5 = torch.cat([up_feature_5, diff3_conv], dim=1)#128+64
        feature_5 = self.conv_block_5(concat_feature_5)#64

        up_feature_6 = self.up_sample_2(feature_5)#64
        # concat_feature_6 = torch.cat([up_feature_6, torch.pow(feature_31 - feature_32,2)], dim=1)
        concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)#128
        # concat_feature_6 = torch.cat([up_feature_6, diff2_conv], dim=1)#64+32
        feature_6 = self.conv_block_6(concat_feature_6)#32

        up_feature_7 = self.up_sample_3(feature_6)
        # concat_feature_7 = torch.cat([up_feature_7, torch.pow(feature_21 - feature_22,2)], dim=1)
        concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)#64
        # concat_feature_7 = torch.cat([up_feature_7, diff1_conv], dim=1)#32+16

        feature_7 = self.conv_block_7(concat_feature_7)
        up_feature_8 = self.up_sample_4(feature_7)
        # concat_feature_8 = torch.cat([up_feature_8, torch.pow(feature_11 - feature_12,2)], dim=1)
        concat_feature_8 = torch.cat([up_feature_8, torch.abs(feature_11 - feature_12)], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)

        return [output_feature, diff4_ori], feature_7

class backbone2BN_con3(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(backbone2BN_con3, self).__init__()

        self.conv_block_1conv1=nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1conv2=nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1bn1=nn.BatchNorm2d(16)
        # self.conv_block_1bn2=nn.BatchNorm2d(16)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)


        self.conv_block_2conv1=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn11=nn.BatchNorm2d(32)
        self.conv_block_2bn12=nn.BatchNorm2d(32)
        self.conv_block_2conv2=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn21=nn.BatchNorm2d(32)
        self.conv_block_2bn22=nn.BatchNorm2d(32)

        # self.diff_ag2=SpatialAttention()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3conv1=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn11=nn.BatchNorm2d(64)
        self.conv_block_3bn12=nn.BatchNorm2d(64)
        self.conv_block_3conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn21=nn.BatchNorm2d(64)
        self.conv_block_3bn22=nn.BatchNorm2d(64)

        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4conv1= nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn11=nn.BatchNorm2d(128)
        self.conv_block_4bn12=nn.BatchNorm2d(128)
        self.conv_block_4conv2=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn21 = nn.BatchNorm2d(128)
        self.conv_block_4bn22=nn.BatchNorm2d(128)

        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=(128*3) , out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=128*3, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=64*3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=32*3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        c=16
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32+c, out_channels=(32+c)//2, kernel_size=3, padding=1),
            nn.BatchNorm2d((32+c)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=(32+c)//2, out_channels=out_dim, kernel_size=1, padding=0)
        )
    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        H, W = pre_data.size(2), pre_data.size(3)
        feature_11=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(pre_data)))))
        feature_12=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(post_data)))))
        # diff1 = torch.abs(feature_11 - feature_12)
        # diff1 = self.conv_diff_1(diff1)

        out11 = self.max_pool_1(feature_11)
        out12 = self.max_pool_1(feature_12)

        #stage2
        feature_21 = F.relu(self.conv_block_2bn21(self.conv_block_2conv2(F.relu(self.conv_block_2bn11(self.conv_block_2conv1(out11))))))
        feature_22 = F.relu(self.conv_block_2bn22(self.conv_block_2conv2(F.relu(self.conv_block_2bn12(self.conv_block_2conv1(out12))))))

        diff2 = torch.abs(feature_21 - feature_22)
        diff2=self.conv_diff_2(diff2)

        out21 = torch.cat([feature_21, diff2], 1)
        out22 = torch.cat([feature_22, diff2], 1)

        out21 = self.max_pool_2(out21)
        out22 = self.max_pool_2(out22)

        #stage3
        feature_31 = F.relu(self.conv_block_3bn21(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21))))))
        feature_32 = F.relu(self.conv_block_3bn22(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn12(self.conv_block_3conv1(out22))))))

        diff3 = torch.abs(feature_31 - feature_32)
        diff3 = self.conv_diff_3(diff3)

        out31 = torch.cat([feature_31, diff3], 1)
        out32 = torch.cat([feature_32, diff3], 1)

        out31 = self.max_pool_3(out31)
        out32 = self.max_pool_3(out32)
        #stage4
        feature_41 = F.relu(self.conv_block_4bn21(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31))))))
        feature_42 = F.relu(self.conv_block_4bn22(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn12(self.conv_block_4conv1(out32))))))

        diff4_ori = torch.abs(feature_41 - feature_42)
        diff4 = self.conv_diff_4(diff4_ori)
        out = torch.cat([feature_41 , feature_42 , diff4], 1)  # 128*3
        out=self.max_pool_4(out)

        up_feature_5 = self.up_sample_1(out)#128
        # concat_feature_5 = torch.cat([up_feature_5, torch.pow(feature_41 - feature_42,2)], dim=1)
        # concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 + feature_42)], dim=1)#256
        concat_feature_5 = torch.cat([up_feature_5, feature_41 , feature_42], dim=1)#256

        # concat_feature_5 = torch.cat([up_feature_5, diff3_conv], dim=1)#128+64
        feature_5 = self.conv_block_5(concat_feature_5)#64

        up_feature_6 = self.up_sample_2(feature_5)#64
        # concat_feature_6 = torch.cat([up_feature_6, torch.pow(feature_31 - feature_32,2)], dim=1)
        # concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 + feature_32)], dim=1)#128
        concat_feature_6 = torch.cat([up_feature_6, feature_31 , feature_32], dim=1)#128

        # concat_feature_6 = torch.cat([up_feature_6, diff2_conv], dim=1)#64+32
        feature_6 = self.conv_block_6(concat_feature_6)#32

        up_feature_7 = self.up_sample_3(feature_6)
        # concat_feature_7 = torch.cat([up_feature_7, torch.pow(feature_21 - feature_22,2)], dim=1)
        # concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 + feature_22)], dim=1)#64
        concat_feature_7 = torch.cat([up_feature_7, feature_21 , feature_22], dim=1)#64

        # concat_feature_7 = torch.cat([up_feature_7, diff1_conv], dim=1)#32+16

        feature_7 = self.conv_block_7(concat_feature_7)
        up_feature_8 = self.up_sample_4(feature_7)
        # concat_feature_8 = torch.cat([up_feature_8, torch.pow(feature_11 - feature_12,2)], dim=1)
        # concat_feature_8 = torch.cat([up_feature_8, torch.abs(feature_11 + feature_12)], dim=1)
        concat_feature_8 = torch.cat([up_feature_8, feature_11 , feature_12], dim=1)

        output_feature = self.conv_block_8(concat_feature_8)

        return [output_feature, diff4_ori], feature_7

class backbone2BN_conSA(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(backbone2BN_conSA, self).__init__()

        self.conv_block_1conv1=nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1conv2=nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1bn1=nn.BatchNorm2d(16)
        self.conv_block_1bn2=nn.BatchNorm2d(16)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)


        self.conv_block_2conv1=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn11=nn.BatchNorm2d(32)
        self.conv_block_2bn12=nn.BatchNorm2d(32)
        self.conv_block_2conv2=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn21=nn.BatchNorm2d(32)
        self.conv_block_2bn22=nn.BatchNorm2d(32)

        self.diff_ag2=SpatialAttention(32)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3conv1=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn11=nn.BatchNorm2d(64)
        self.conv_block_3bn12=nn.BatchNorm2d(64)
        self.conv_block_3conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn21=nn.BatchNorm2d(64)
        self.conv_block_3bn22=nn.BatchNorm2d(64)

        self.diff_ag3=SpatialAttention(64)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4conv1= nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn11=nn.BatchNorm2d(128)
        self.conv_block_4bn12=nn.BatchNorm2d(128)
        self.conv_block_4conv2=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn21 = nn.BatchNorm2d(128)
        self.conv_block_4bn22=nn.BatchNorm2d(128)
        self.diff_ag4=SpatialAttention(128)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=(128*3) , out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        c=0
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32+c, out_channels=(32+c)//2, kernel_size=3, padding=1),
            nn.BatchNorm2d((32+c)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=(32+c)//2, out_channels=out_dim, kernel_size=1, padding=0)
        )
    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        H, W = pre_data.size(2), pre_data.size(3)
        feature_11=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(pre_data)))))
        feature_12=F.relu(self.conv_block_1bn2(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(post_data)))))
        # diff1 = torch.abs(feature_11 - feature_12)
        # diff1 = self.conv_diff_1(diff1)

        out11 = self.max_pool_1(feature_11)
        out12 = self.max_pool_1(feature_12)

        #stage2
        feature_21 = F.relu(self.conv_block_2bn21(self.conv_block_2conv2(F.relu(self.conv_block_2bn11(self.conv_block_2conv1(out11))))))
        feature_22 = F.relu(self.conv_block_2bn22(self.conv_block_2conv2(F.relu(self.conv_block_2bn12(self.conv_block_2conv1(out12))))))

        diff2_ori = torch.abs(feature_21 - feature_22)
        diff2=self.conv_diff_2(diff2_ori)
        weight2=self.diff_ag2(diff2_ori)

        out21 = torch.cat([feature_21*weight2, diff2], 1)
        out22 = torch.cat([feature_22*weight2, diff2], 1)

        out21 = self.max_pool_2(out21)
        out22 = self.max_pool_2(out22)

        #stage3
        feature_31 = F.relu(self.conv_block_3bn21(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21))))))
        feature_32 = F.relu(self.conv_block_3bn22(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn12(self.conv_block_3conv1(out22))))))

        diff3_ori = torch.abs(feature_31 - feature_32)
        diff3 = self.conv_diff_3(diff3_ori)
        weight3 = self.diff_ag3(diff3_ori)
        out31 = torch.cat([feature_31*weight3, diff3], 1)
        out32 = torch.cat([feature_32*weight3, diff3], 1)

        out31 = self.max_pool_3(out31)
        out32 = self.max_pool_3(out32)
        #stage4
        feature_41 = F.relu(self.conv_block_4bn21(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31))))))
        feature_42 = F.relu(self.conv_block_4bn22(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn12(self.conv_block_4conv1(out32))))))

        diff4_ori = torch.abs(feature_41 - feature_42)
        diff4 = self.conv_diff_4(diff4_ori)
        weight4 = self.diff_ag4(diff4_ori)

        out = torch.cat([feature_41*weight4 , feature_42*weight4 , diff4], 1)  # 128*3
        out=self.max_pool_4(out)

        up_feature_5 = self.up_sample_1(out)#128
        # concat_feature_5 = torch.cat([up_feature_5, torch.pow(feature_41 - feature_42,2)], dim=1)
        concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)#256
        # concat_feature_5 = torch.cat([up_feature_5, diff3_conv], dim=1)#128+64
        feature_5 = self.conv_block_5(concat_feature_5)#64

        up_feature_6 = self.up_sample_2(feature_5)#64
        # concat_feature_6 = torch.cat([up_feature_6, torch.pow(feature_31 - feature_32,2)], dim=1)
        concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)#128
        # concat_feature_6 = torch.cat([up_feature_6, diff2_conv], dim=1)#64+32
        feature_6 = self.conv_block_6(concat_feature_6)#32

        up_feature_7 = self.up_sample_3(feature_6)
        # concat_feature_7 = torch.cat([up_feature_7, torch.pow(feature_21 - feature_22,2)], dim=1)
        concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)#64
        # concat_feature_7 = torch.cat([up_feature_7, diff1_conv], dim=1)#32+16

        feature_7 = self.conv_block_7(concat_feature_7)
        up_feature_8 = self.up_sample_4(feature_7)
        # concat_feature_8 = torch.cat([up_feature_8, torch.pow(feature_11 - feature_12,2)], dim=1)
        concat_feature_8 = torch.cat([up_feature_8, torch.abs(feature_11 - feature_12)], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)

        return [output_feature, diff4_ori], feature_7


class backbone2BN_con_ASSP(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(backbone2BN_con_ASSP, self).__init__()

        self.conv_block_1conv1=nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1conv2=nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1bn1=nn.BatchNorm2d(16)
        self.conv_block_1bn2=nn.BatchNorm2d(16)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)


        self.conv_block_2conv1=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn11=nn.BatchNorm2d(32)
        self.conv_block_2bn12=nn.BatchNorm2d(32)
        self.conv_block_2conv2=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn21=nn.BatchNorm2d(32)
        self.conv_block_2bn22=nn.BatchNorm2d(32)

        # self.diff_ag2=SpatialAttention()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3conv1=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn11=nn.BatchNorm2d(64)
        self.conv_block_3bn12=nn.BatchNorm2d(64)
        self.conv_block_3conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn21=nn.BatchNorm2d(64)
        self.conv_block_3bn22=nn.BatchNorm2d(64)
        # self.diff_ag3 = SpatialAttention()
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4conv1= nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn11=nn.BatchNorm2d(128)
        self.conv_block_4bn12=nn.BatchNorm2d(128)
        self.conv_block_4conv2=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn21 = nn.BatchNorm2d(128)
        self.conv_block_4bn22=nn.BatchNorm2d(128)

        # self.diff_ag4 = SpatialAttention()
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=(128*3) , out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        c=0
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32+c, out_channels=(32+c)//2, kernel_size=3, padding=1),
            nn.BatchNorm2d((32+c)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=(32+c)//2, out_channels=out_dim, kernel_size=1, padding=0)
        )
        self.ASSP = ASSP(128*3, (128*3)//2)
        self.ASSPconv=nn.Conv2d(128*3, (128*3)//2, kernel_size=1, padding=0)
        # self.ASSPconv = nn.Sequential(
        #     nn.Conv2d(256, 48, kernel_size=1, padding=0),
        #     # nn.BatchNorm2d(48, track_running_stats=False),
        #     nn.ReLU()
        # )

        self.conv_block_asspbn21 = nn.BatchNorm2d(128*3)
        # self.conv_block_asspbn22 = nn.BatchNorm2d(128)

        # self.conv_block_asspbn11 = nn.BatchNorm2d(64)
        # self.conv_block_asspbn12 = nn.BatchNorm2d(64)
    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        H, W = pre_data.size(2), pre_data.size(3)
        feature_11=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(pre_data)))))
        feature_12=F.relu(self.conv_block_1bn2(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(post_data)))))
        # diff1 = torch.abs(feature_11 - feature_12)
        # diff1 = self.conv_diff_1(diff1)

        out11 = self.max_pool_1(feature_11)
        out12 = self.max_pool_1(feature_12)

        #stage2
        feature_21 = F.relu(self.conv_block_2bn21(self.conv_block_2conv2(F.relu(self.conv_block_2bn11(self.conv_block_2conv1(out11))))))
        feature_22 = F.relu(self.conv_block_2bn22(self.conv_block_2conv2(F.relu(self.conv_block_2bn12(self.conv_block_2conv1(out12))))))

        diff2 = torch.abs(feature_21 - feature_22)
        diff2=self.conv_diff_2(diff2)

        out21 = torch.cat([feature_21, diff2], 1)
        out22 = torch.cat([feature_22, diff2], 1)

        out21 = self.max_pool_2(out21)
        out22 = self.max_pool_2(out22)

        #stage3
        feature_31 = F.relu(self.conv_block_3bn21(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21))))))
        feature_32 = F.relu(self.conv_block_3bn22(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn12(self.conv_block_3conv1(out22))))))

        diff3 = torch.abs(feature_31 - feature_32)
        diff3 = self.conv_diff_3(diff3)

        out31 = torch.cat([feature_31, diff3], 1)
        out32 = torch.cat([feature_32, diff3], 1)

        out31 = self.max_pool_3(out31)
        out32 = self.max_pool_3(out32)
        #stage4
        feature_41 = F.relu(self.conv_block_4bn21(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31))))))
        feature_42 = F.relu(self.conv_block_4bn22(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn12(self.conv_block_4conv1(out32))))))

        # out41_ASSP = self.ASSP(feature_41)
        # out41_ASSPconv=self.ASSPconv(feature_41)
        # feature_41 = torch.cat([out41_ASSP, out41_ASSPconv], 1)  # 128*2
        # feature_41 = F.relu(self.conv_block_asspbn21(feature_41))

        # out41_ASSP = F.relu(self.conv_block_asspbn21(out41_ASSP))
        # out41_ASSPconv= F.relu(self.conv_block_asspbn11(self.ASSPconv(feature_41)))

        # out42_ASSP = self.ASSP(feature_42)
        # out42_ASSPconv=self.ASSPconv(feature_42)
        # feature_42 = torch.cat([out42_ASSP, out42_ASSPconv], 1)  # 128*2
        # feature_42 = F.relu(self.conv_block_asspbn22(feature_42))

        # out42_ASSP = F.relu(self.conv_block_asspbn22(out42_ASSP))
        # out42_ASSPconv= F.relu(self.conv_block_asspbn12(self.ASSPconv(feature_42)))

        # out42_conv = self.ASSPconv(feature_42)
        # feature_41 = torch.cat([out41_ASSP, out41_ASSPconv], 1)  # 128*2
        # feature_42 = torch.cat([out42_ASSP, out42_ASSPconv], 1)  # 128*2

        diff4_ori = torch.abs(feature_41 - feature_42)
        diff4 = self.conv_diff_4(diff4_ori)
        out = torch.cat([feature_41 , feature_42 , diff4], 1)  # 128*3
        outASSP = self.ASSP(out)
        outConv=self.ASSPconv(out)
        out = torch.cat([outASSP, outConv], 1)  # 128*3
        out= F.relu(self.conv_block_asspbn21(out))
        out=self.max_pool_4(out)

        up_feature_5 = self.up_sample_1(out)#128
        # concat_feature_5 = torch.cat([up_feature_5, torch.pow(feature_41 - feature_42,2)], dim=1)
        concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)#256
        # concat_feature_5 = torch.cat([up_feature_5, diff3_conv], dim=1)#128+64
        feature_5 = self.conv_block_5(concat_feature_5)#64

        up_feature_6 = self.up_sample_2(feature_5)#64
        # concat_feature_6 = torch.cat([up_feature_6, torch.pow(feature_31 - feature_32,2)], dim=1)
        concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)#128
        # concat_feature_6 = torch.cat([up_feature_6, diff2_conv], dim=1)#64+32
        feature_6 = self.conv_block_6(concat_feature_6)#32

        up_feature_7 = self.up_sample_3(feature_6)
        # concat_feature_7 = torch.cat([up_feature_7, torch.pow(feature_21 - feature_22,2)], dim=1)
        concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)#64
        # concat_feature_7 = torch.cat([up_feature_7, diff1_conv], dim=1)#32+16

        feature_7 = self.conv_block_7(concat_feature_7)
        up_feature_8 = self.up_sample_4(feature_7)
        # concat_feature_8 = torch.cat([up_feature_8, torch.pow(feature_11 - feature_12,2)], dim=1)
        concat_feature_8 = torch.cat([up_feature_8, torch.abs(feature_11 - feature_12)], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)

        return [output_feature, diff4_ori], feature_7

class backbone2BN_con_PSP(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(backbone2BN_con_PSP, self).__init__()

        self.conv_block_1conv1=nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1conv2=nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1bn1=nn.BatchNorm2d(16)
        self.conv_block_1bn2=nn.BatchNorm2d(16)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)


        self.conv_block_2conv1=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn11=nn.BatchNorm2d(32)
        self.conv_block_2bn12=nn.BatchNorm2d(32)
        self.conv_block_2conv2=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn21=nn.BatchNorm2d(32)
        self.conv_block_2bn22=nn.BatchNorm2d(32)

        # self.diff_ag2=SpatialAttention()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3conv1=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn11=nn.BatchNorm2d(64)
        self.conv_block_3bn12=nn.BatchNorm2d(64)
        self.conv_block_3conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn21=nn.BatchNorm2d(64)
        self.conv_block_3bn22=nn.BatchNorm2d(64)
        # self.diff_ag3 = SpatialAttention()
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4conv1= nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn11=nn.BatchNorm2d(128)
        self.conv_block_4bn12=nn.BatchNorm2d(128)
        self.conv_block_4conv2=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn21 = nn.BatchNorm2d(128)
        self.conv_block_4bn22=nn.BatchNorm2d(128)

        # self.diff_ag4 = SpatialAttention()
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=(128*3) , out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        c=0
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32+c, out_channels=(32+c)//2, kernel_size=3, padding=1),
            nn.BatchNorm2d((32+c)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=(32+c)//2, out_channels=out_dim, kernel_size=1, padding=0)
        )
        # self.ASSP = ASSP(128, 64)
        self.ASSPconv=nn.Conv2d(128, 64, kernel_size=1, padding=0)
        norm_layer = nn.BatchNorm2d
        self.psp = _PSPModule(128, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer)

        self.conv_block_asspbn21 = nn.BatchNorm2d(128)
        self.conv_block_asspbn22 = nn.BatchNorm2d(128)

        # self.conv_block_asspbn11 = nn.BatchNorm2d(64)
        # self.conv_block_asspbn12 = nn.BatchNorm2d(64)
    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        H, W = pre_data.size(2), pre_data.size(3)
        feature_11=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(pre_data)))))
        feature_12=F.relu(self.conv_block_1bn2(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(post_data)))))
        # diff1 = torch.abs(feature_11 - feature_12)
        # diff1 = self.conv_diff_1(diff1)

        out11 = self.max_pool_1(feature_11)
        out12 = self.max_pool_1(feature_12)

        #stage2
        feature_21 = F.relu(self.conv_block_2bn21(self.conv_block_2conv2(F.relu(self.conv_block_2bn11(self.conv_block_2conv1(out11))))))
        feature_22 = F.relu(self.conv_block_2bn22(self.conv_block_2conv2(F.relu(self.conv_block_2bn12(self.conv_block_2conv1(out12))))))

        diff2 = torch.abs(feature_21 - feature_22)
        diff2=self.conv_diff_2(diff2)

        out21 = torch.cat([feature_21, diff2], 1)
        out22 = torch.cat([feature_22, diff2], 1)

        out21 = self.max_pool_2(out21)
        out22 = self.max_pool_2(out22)

        #stage3
        feature_31 = F.relu(self.conv_block_3bn21(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21))))))
        feature_32 = F.relu(self.conv_block_3bn22(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn12(self.conv_block_3conv1(out22))))))

        diff3 = torch.abs(feature_31 - feature_32)
        diff3 = self.conv_diff_3(diff3)

        out31 = torch.cat([feature_31, diff3], 1)
        out32 = torch.cat([feature_32, diff3], 1)

        out31 = self.max_pool_3(out31)
        out32 = self.max_pool_3(out32)
        #stage4
        feature_41 = F.relu(self.conv_block_4bn21(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31))))))
        feature_42 = F.relu(self.conv_block_4bn22(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn12(self.conv_block_4conv1(out32))))))

        # out41_ASSP = self.psp(feature_41)
        # out41_ASSP = F.relu(self.conv_block_asspbn21(out41_ASSP))
        # out41_ASSPconv= F.relu(self.conv_block_asspbn11(self.ASSPconv(feature_41)))
        #
        # out42_ASSP = self.psp(feature_42)
        # out42_ASSP = F.relu(self.conv_block_asspbn22(out42_ASSP))
        # out42_ASSPconv= F.relu(self.conv_block_asspbn12(self.ASSPconv(feature_42)))

        out41_ASSP = self.psp(feature_41)
        out41_ASSPconv = self.ASSPconv(feature_41)
        feature_41 = torch.cat([out41_ASSP, out41_ASSPconv], 1)  # 128*2
        feature_41 = F.relu(self.conv_block_asspbn21(feature_41))

        out42_ASSP = self.psp(feature_42)
        out42_ASSPconv = self.ASSPconv(feature_42)
        feature_42 = torch.cat([out42_ASSP, out42_ASSPconv], 1)  # 128*2
        feature_42 = F.relu(self.conv_block_asspbn22(feature_42))


        # feature_41 = torch.cat([out41_ASSP, out41_ASSPconv], 1)  # 128*2
        # feature_42 = torch.cat([out42_ASSP, out42_ASSPconv], 1)  # 128*2

        diff4_ori = torch.abs(feature_41 - feature_42)
        diff4 = self.conv_diff_4(diff4_ori)
        out = torch.cat([feature_41 , feature_42 , diff4], 1)  # 128*3
        out=self.max_pool_4(out)

        up_feature_5 = self.up_sample_1(out)#128
        # concat_feature_5 = torch.cat([up_feature_5, torch.pow(feature_41 - feature_42,2)], dim=1)
        concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)#256
        # concat_feature_5 = torch.cat([up_feature_5, diff3_conv], dim=1)#128+64
        feature_5 = self.conv_block_5(concat_feature_5)#64

        up_feature_6 = self.up_sample_2(feature_5)#64
        # concat_feature_6 = torch.cat([up_feature_6, torch.pow(feature_31 - feature_32,2)], dim=1)
        concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)#128
        # concat_feature_6 = torch.cat([up_feature_6, diff2_conv], dim=1)#64+32
        feature_6 = self.conv_block_6(concat_feature_6)#32

        up_feature_7 = self.up_sample_3(feature_6)
        # concat_feature_7 = torch.cat([up_feature_7, torch.pow(feature_21 - feature_22,2)], dim=1)
        concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)#64
        # concat_feature_7 = torch.cat([up_feature_7, diff1_conv], dim=1)#32+16

        feature_7 = self.conv_block_7(concat_feature_7)
        up_feature_8 = self.up_sample_4(feature_7)
        # concat_feature_8 = torch.cat([up_feature_8, torch.pow(feature_11 - feature_12,2)], dim=1)
        concat_feature_8 = torch.cat([up_feature_8, torch.abs(feature_11 - feature_12)], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)

        return [output_feature, diff4_ori], feature_7

from models.appendix import SPM,FPM,ChannelAttention,PAM
class backbone2BN_con_AT(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(backbone2BN_con_AT, self).__init__()

        self.conv_block_1conv1=nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1conv2=nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1bn1=nn.BatchNorm2d(16)
        self.conv_block_1bn2=nn.BatchNorm2d(16)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)


        self.conv_block_2conv1=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn11=nn.BatchNorm2d(32)
        self.conv_block_2bn12=nn.BatchNorm2d(32)
        self.conv_block_2conv2=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn21=nn.BatchNorm2d(32)
        self.conv_block_2bn22=nn.BatchNorm2d(32)

        # self.diff_ag2=SpatialAttention()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3conv1=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn11=nn.BatchNorm2d(64)
        self.conv_block_3bn12=nn.BatchNorm2d(64)
        self.conv_block_3conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn21=nn.BatchNorm2d(64)
        self.conv_block_3bn22=nn.BatchNorm2d(64)
        # self.diff_ag3 = SpatialAttention()
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4conv1= nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn11=nn.BatchNorm2d(128)
        self.conv_block_4bn12=nn.BatchNorm2d(128)
        self.conv_block_4conv2=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn21 = nn.BatchNorm2d(128)
        self.conv_block_4bn22=nn.BatchNorm2d(128)

        # self.diff_ag4 = SpatialAttention()
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=(128*3) , out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        c=0
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32+c, out_channels=(32+c)//2, kernel_size=3, padding=1),
            nn.BatchNorm2d((32+c)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=(32+c)//2, out_channels=out_dim, kernel_size=1, padding=0)
        )
        # self.ASSP = ASSP(128*3, (128*3)//2)
        # self.ASSPconv=nn.Conv2d(128*3, (128*3)//2, kernel_size=1, padding=0)
        # self.conv_block_asspbn21 = nn.BatchNorm2d(128*3)
        # # self.conv_block_asspbn22 = nn.BatchNorm2d(128)
        # # self.conv_block_asspbn11 = nn.BatchNorm2d(64)
        # # self.conv_block_asspbn12 = nn.BatchNorm2d(64)
        # self.pam = PAM(128)
        self.psp = SPM(128, 128, sizes=(1, 2, 3, 6))
        self.fpa = FPM(128)
        # self.drop = nn.Dropout2d(p=0.2)
        # self.ca = ChannelAttention(in_channels=128*3)
        # self.ca2 = ChannelAttention(in_channels=128 * 3)
        self.conv1x1 = nn.Conv2d(128, (128)//2, kernel_size=1, stride=1, bias=False)
        self.conv1x1_2 = nn.Conv2d(128, (128)//2, kernel_size=1, stride=1, bias=False)

        # self.conv1x1_catBN1 = nn.BatchNorm2d(128*3)
        # self.conv1x1_catBN2 = nn.BatchNorm2d(128*3)
        self.conv3x3_last = nn.Conv2d(128*3, 128*3, kernel_size=3, padding=1)
        self.conv3x3_catBN_last = nn.BatchNorm2d(128 * 3)

    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        H, W = pre_data.size(2), pre_data.size(3)
        feature_11=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(pre_data)))))
        feature_12=F.relu(self.conv_block_1bn2(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(post_data)))))
        # diff1 = torch.abs(feature_11 - feature_12)
        # diff1 = self.conv_diff_1(diff1)

        out11 = self.max_pool_1(feature_11)
        out12 = self.max_pool_1(feature_12)

        #stage2
        feature_21 = F.relu(self.conv_block_2bn21(self.conv_block_2conv2(F.relu(self.conv_block_2bn11(self.conv_block_2conv1(out11))))))
        feature_22 = F.relu(self.conv_block_2bn22(self.conv_block_2conv2(F.relu(self.conv_block_2bn12(self.conv_block_2conv1(out12))))))

        diff2 = torch.abs(feature_21 - feature_22)
        diff2=self.conv_diff_2(diff2)

        out21 = torch.cat([feature_21, diff2], 1)
        out22 = torch.cat([feature_22, diff2], 1)

        out21 = self.max_pool_2(out21)
        out22 = self.max_pool_2(out22)

        #stage3
        feature_31 = F.relu(self.conv_block_3bn21(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21))))))
        feature_32 = F.relu(self.conv_block_3bn22(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn12(self.conv_block_3conv1(out22))))))

        diff3 = torch.abs(feature_31 - feature_32)
        diff3 = self.conv_diff_3(diff3)

        out31 = torch.cat([feature_31, diff3], 1)
        out32 = torch.cat([feature_32, diff3], 1)

        out31 = self.max_pool_3(out31)
        out32 = self.max_pool_3(out32)
        #stage4
        feature_41 = F.relu(self.conv_block_4bn21(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31))))))
        feature_42 = F.relu(self.conv_block_4bn22(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn12(self.conv_block_4conv1(out32))))))

        # out41_ASSP = self.ASSP(feature_41)
        # out41_ASSPconv=self.ASSPconv(feature_41)
        # feature_41 = torch.cat([out41_ASSP, out41_ASSPconv], 1)  # 128*2
        # feature_41 = F.relu(self.conv_block_asspbn21(feature_41))

        # out41_ASSP = F.relu(self.conv_block_asspbn21(out41_ASSP))
        # out41_ASSPconv= F.relu(self.conv_block_asspbn11(self.ASSPconv(feature_41)))

        # out42_ASSP = self.ASSP(feature_42)
        # out42_ASSPconv=self.ASSPconv(feature_42)
        # feature_42 = torch.cat([out42_ASSP, out42_ASSPconv], 1)  # 128*2
        # feature_42 = F.relu(self.conv_block_asspbn22(feature_42))

        # out42_ASSP = F.relu(self.conv_block_asspbn22(out42_ASSP))
        # out42_ASSPconv= F.relu(self.conv_block_asspbn12(self.ASSPconv(feature_42)))

        # out42_conv = self.ASSPconv(feature_42)
        # feature_41 = torch.cat([out41_ASSP, out41_ASSPconv], 1)  # 128*2
        # feature_42 = torch.cat([out42_ASSP, out42_ASSPconv], 1)  # 128*2
        psp1 = F.relu(self.conv1x1(self.psp(feature_41)))
        fpa1 = F.relu(self.conv1x1_2(self.fpa(feature_41)))
        feature_41=torch.cat([psp1, fpa1], dim=1)
        psp2 = F.relu(self.conv1x1(self.psp(feature_42)))
        fpa2 = F.relu(self.conv1x1_2(self.fpa(feature_42)))
        feature_42 = torch.cat([psp2, fpa2], dim=1)
        # feature_41=F.relu(self.conv1x1_catBN(self.conv1x1(torch.cat([psp1,fpa1],dim=1))))


        diff4_ori = torch.abs(feature_41 - feature_42)
        diff4 = self.conv_diff_4(diff4_ori)
        out = torch.cat([feature_41 , feature_42 , diff4], 1)  # 128*3


        # outASSP = self.ASSP(out)
        # outConv=self.ASSPconv(out)
        # out = torch.cat([outASSP, outConv], 1)  # 128*3
        # out= F.relu(self.conv_block_asspbn21(out))
        # out = self.pam(out)
        # # Spatial Pyramid Module
        # psp = self.psp(out)
        # # pspdrop = self.drop(psp)
        # capsp = self.ca(psp)
        # capsp = self.conv1x1(capsp)
        # # Feature Pyramid Attention Module
        # fpa = self.fpa(out)
        # # fpadrop = self.drop(fpa)
        # cafpa = self.ca2(fpa)
        # cafpa = self.conv1x1_2(cafpa)
        # ca_psp_fpa = torch.cat([capsp, cafpa], dim=1)
        # ca_psp_fpa=F.relu(self.conv1x1_catBN(ca_psp_fpa))
        out=F.relu(self.conv3x3_catBN_last(self.conv3x3_last(out)))
        out=self.max_pool_4(out)

        up_feature_5 = self.up_sample_1(out)#128
        # concat_feature_5 = torch.cat([up_feature_5, torch.pow(feature_41 - feature_42,2)], dim=1)
        concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)#256
        # concat_feature_5 = torch.cat([up_feature_5, diff3_conv], dim=1)#128+64
        feature_5 = self.conv_block_5(concat_feature_5)#64

        up_feature_6 = self.up_sample_2(feature_5)#64
        # concat_feature_6 = torch.cat([up_feature_6, torch.pow(feature_31 - feature_32,2)], dim=1)
        concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)#128
        # concat_feature_6 = torch.cat([up_feature_6, diff2_conv], dim=1)#64+32
        feature_6 = self.conv_block_6(concat_feature_6)#32

        up_feature_7 = self.up_sample_3(feature_6)
        # concat_feature_7 = torch.cat([up_feature_7, torch.pow(feature_21 - feature_22,2)], dim=1)
        concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)#64
        # concat_feature_7 = torch.cat([up_feature_7, diff1_conv], dim=1)#32+16

        feature_7 = self.conv_block_7(concat_feature_7)
        up_feature_8 = self.up_sample_4(feature_7)
        # concat_feature_8 = torch.cat([up_feature_8, torch.pow(feature_11 - feature_12,2)], dim=1)
        concat_feature_8 = torch.cat([up_feature_8, torch.abs(feature_11 - feature_12)], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)

        return [output_feature, diff4_ori], feature_7

class backbone2BN_con_AT(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(backbone2BN_con_AT, self).__init__()

        self.conv_block_1conv1=nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1conv2=nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1bn1=nn.BatchNorm2d(16)
        self.conv_block_1bn2=nn.BatchNorm2d(16)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)


        self.conv_block_2conv1=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn11=nn.BatchNorm2d(32)
        self.conv_block_2bn12=nn.BatchNorm2d(32)
        self.conv_block_2conv2=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn21=nn.BatchNorm2d(32)
        self.conv_block_2bn22=nn.BatchNorm2d(32)

        # self.diff_ag2=SpatialAttention()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3conv1=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn11=nn.BatchNorm2d(64)
        self.conv_block_3bn12=nn.BatchNorm2d(64)
        self.conv_block_3conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn21=nn.BatchNorm2d(64)
        self.conv_block_3bn22=nn.BatchNorm2d(64)
        # self.diff_ag3 = SpatialAttention()
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4conv1= nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn11=nn.BatchNorm2d(128)
        self.conv_block_4bn12=nn.BatchNorm2d(128)
        self.conv_block_4conv2=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn21 = nn.BatchNorm2d(128)
        self.conv_block_4bn22=nn.BatchNorm2d(128)

        # self.diff_ag4 = SpatialAttention()
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=(128*3) , out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        c=0
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32+c, out_channels=(32+c)//2, kernel_size=3, padding=1),
            nn.BatchNorm2d((32+c)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=(32+c)//2, out_channels=out_dim, kernel_size=1, padding=0)
        )
        # self.ASSP = ASSP(128*3, (128*3)//2)
        # self.ASSPconv=nn.Conv2d(128*3, (128*3)//2, kernel_size=1, padding=0)
        # self.conv_block_asspbn21 = nn.BatchNorm2d(128*3)
        # # self.conv_block_asspbn22 = nn.BatchNorm2d(128)
        # # self.conv_block_asspbn11 = nn.BatchNorm2d(64)
        # # self.conv_block_asspbn12 = nn.BatchNorm2d(64)
        # self.pam = PAM(128)
        self.psp = SPM(128, 128, sizes=(1, 2, 3, 6))
        self.fpa = FPM(128)
        # self.drop = nn.Dropout2d(p=0.2)
        # self.ca = ChannelAttention(in_channels=128*3)
        # self.ca2 = ChannelAttention(in_channels=128 * 3)
        self.conv1x1 = nn.Conv2d(128, (128)//2, kernel_size=1, stride=1, bias=False)
        self.conv1x1_2 = nn.Conv2d(128, (128)//2, kernel_size=1, stride=1, bias=False)

        self.conv1x1_catBN1 = nn.BatchNorm2d(128)
        self.conv1x1_catBN2 = nn.BatchNorm2d(128)
        self.conv3x3_last = nn.Conv2d(128*3, 128*3, kernel_size=3, padding=1)
        self.conv3x3_catBN_last = nn.BatchNorm2d(128 * 3)

    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        H, W = pre_data.size(2), pre_data.size(3)
        feature_11=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(pre_data)))))
        feature_12=F.relu(self.conv_block_1bn2(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(post_data)))))
        # diff1 = torch.abs(feature_11 - feature_12)
        # diff1 = self.conv_diff_1(diff1)

        out11 = self.max_pool_1(feature_11)
        out12 = self.max_pool_1(feature_12)

        #stage2
        feature_21 = F.relu(self.conv_block_2bn21(self.conv_block_2conv2(F.relu(self.conv_block_2bn11(self.conv_block_2conv1(out11))))))
        feature_22 = F.relu(self.conv_block_2bn22(self.conv_block_2conv2(F.relu(self.conv_block_2bn12(self.conv_block_2conv1(out12))))))

        diff2 = torch.abs(feature_21 - feature_22)
        diff2=self.conv_diff_2(diff2)

        out21 = torch.cat([feature_21, diff2], 1)
        out22 = torch.cat([feature_22, diff2], 1)

        out21 = self.max_pool_2(out21)
        out22 = self.max_pool_2(out22)

        #stage3
        feature_31 = F.relu(self.conv_block_3bn21(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21))))))
        feature_32 = F.relu(self.conv_block_3bn22(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn12(self.conv_block_3conv1(out22))))))

        diff3 = torch.abs(feature_31 - feature_32)
        diff3 = self.conv_diff_3(diff3)

        out31 = torch.cat([feature_31, diff3], 1)
        out32 = torch.cat([feature_32, diff3], 1)

        out31 = self.max_pool_3(out31)
        out32 = self.max_pool_3(out32)
        #stage4
        feature_41 = F.relu(self.conv_block_4bn21(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31))))))
        feature_42 = F.relu(self.conv_block_4bn22(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn12(self.conv_block_4conv1(out32))))))

        # out41_ASSP = self.ASSP(feature_41)
        # out41_ASSPconv=self.ASSPconv(feature_41)
        # feature_41 = torch.cat([out41_ASSP, out41_ASSPconv], 1)  # 128*2
        # feature_41 = F.relu(self.conv_block_asspbn21(feature_41))

        # out41_ASSP = F.relu(self.conv_block_asspbn21(out41_ASSP))
        # out41_ASSPconv= F.relu(self.conv_block_asspbn11(self.ASSPconv(feature_41)))

        # out42_ASSP = self.ASSP(feature_42)
        # out42_ASSPconv=self.ASSPconv(feature_42)
        # feature_42 = torch.cat([out42_ASSP, out42_ASSPconv], 1)  # 128*2
        # feature_42 = F.relu(self.conv_block_asspbn22(feature_42))

        # out42_ASSP = F.relu(self.conv_block_asspbn22(out42_ASSP))
        # out42_ASSPconv= F.relu(self.conv_block_asspbn12(self.ASSPconv(feature_42)))

        # out42_conv = self.ASSPconv(feature_42)
        # feature_41 = torch.cat([out41_ASSP, out41_ASSPconv], 1)  # 128*2
        # feature_42 = torch.cat([out42_ASSP, out42_ASSPconv], 1)  # 128*2
        psp1 = F.relu(self.conv1x1(self.psp(feature_41)))
        fpa1 = F.relu(self.conv1x1_2(self.fpa(feature_41)))
        feature_41=F.relu(self.conv1x1_catBN1(torch.cat([psp1, fpa1], dim=1)))

        psp2 = F.relu(self.conv1x1(self.psp(feature_42)))
        fpa2 = F.relu(self.conv1x1_2(self.fpa(feature_42)))
        feature_42 = F.relu(self.conv1x1_catBN2(torch.cat([psp2, fpa2], dim=1)))
        # feature_41=F.relu(self.conv1x1_catBN(self.conv1x1(torch.cat([psp1,fpa1],dim=1))))


        diff4_ori = torch.abs(feature_41 - feature_42)
        diff4 = self.conv_diff_4(diff4_ori)
        out = torch.cat([feature_41 , feature_42 , diff4], 1)  # 128*3


        # outASSP = self.ASSP(out)
        # outConv=self.ASSPconv(out)
        # out = torch.cat([outASSP, outConv], 1)  # 128*3
        # out= F.relu(self.conv_block_asspbn21(out))
        # out = self.pam(out)
        # # Spatial Pyramid Module
        # psp = self.psp(out)
        # # pspdrop = self.drop(psp)
        # capsp = self.ca(psp)
        # capsp = self.conv1x1(capsp)
        # # Feature Pyramid Attention Module
        # fpa = self.fpa(out)
        # # fpadrop = self.drop(fpa)
        # cafpa = self.ca2(fpa)
        # cafpa = self.conv1x1_2(cafpa)
        # ca_psp_fpa = torch.cat([capsp, cafpa], dim=1)
        # ca_psp_fpa=F.relu(self.conv1x1_catBN(ca_psp_fpa))
        out=F.relu(self.conv3x3_catBN_last(self.conv3x3_last(out)))
        out=self.max_pool_4(out)

        up_feature_5 = self.up_sample_1(out)#128
        # concat_feature_5 = torch.cat([up_feature_5, torch.pow(feature_41 - feature_42,2)], dim=1)
        concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)#256
        # concat_feature_5 = torch.cat([up_feature_5, diff3_conv], dim=1)#128+64
        feature_5 = self.conv_block_5(concat_feature_5)#64

        up_feature_6 = self.up_sample_2(feature_5)#64
        # concat_feature_6 = torch.cat([up_feature_6, torch.pow(feature_31 - feature_32,2)], dim=1)
        concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)#128
        # concat_feature_6 = torch.cat([up_feature_6, diff2_conv], dim=1)#64+32
        feature_6 = self.conv_block_6(concat_feature_6)#32

        up_feature_7 = self.up_sample_3(feature_6)
        # concat_feature_7 = torch.cat([up_feature_7, torch.pow(feature_21 - feature_22,2)], dim=1)
        concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)#64
        # concat_feature_7 = torch.cat([up_feature_7, diff1_conv], dim=1)#32+16

        feature_7 = self.conv_block_7(concat_feature_7)
        up_feature_8 = self.up_sample_4(feature_7)
        # concat_feature_8 = torch.cat([up_feature_8, torch.pow(feature_11 - feature_12,2)], dim=1)
        concat_feature_8 = torch.cat([up_feature_8, torch.abs(feature_11 - feature_12)], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)

        return [output_feature, diff4_ori], feature_7

class backbone2BN_conATSA(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(backbone2BN_conATSA, self).__init__()

        self.conv_block_1conv1=nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1conv2=nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1bn1=nn.BatchNorm2d(16)
        self.conv_block_1bn2=nn.BatchNorm2d(16)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)


        self.conv_block_2conv1=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn11=nn.BatchNorm2d(32)
        self.conv_block_2bn12=nn.BatchNorm2d(32)
        self.conv_block_2conv2=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn21=nn.BatchNorm2d(32)
        self.conv_block_2bn22=nn.BatchNorm2d(32)

        self.diff_ag2=SpatialAttention(32)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3conv1=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn11=nn.BatchNorm2d(64)
        self.conv_block_3bn12=nn.BatchNorm2d(64)
        self.conv_block_3conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn21=nn.BatchNorm2d(64)
        self.conv_block_3bn22=nn.BatchNorm2d(64)

        self.diff_ag3=SpatialAttention(64)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4conv1= nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn11=nn.BatchNorm2d(128)
        self.conv_block_4bn12=nn.BatchNorm2d(128)
        self.conv_block_4conv2=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn21 = nn.BatchNorm2d(128)
        self.conv_block_4bn22=nn.BatchNorm2d(128)
        self.diff_ag4=SpatialAttention(128)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=(128*3) , out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        c=0
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32+c, out_channels=(32+c)//2, kernel_size=3, padding=1),
            nn.BatchNorm2d((32+c)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=(32+c)//2, out_channels=out_dim, kernel_size=1, padding=0)
        )
        self.psp = SPM(128, 128, sizes=(1, 2, 3, 6))
        self.fpa = FPM(128)
        self.conv1x1 = nn.Conv2d(128, (128) // 2, kernel_size=1, stride=1, bias=False)
        self.conv1x1_2 = nn.Conv2d(128, (128) // 2, kernel_size=1, stride=1, bias=False)

        self.conv1x1_catBN1 = nn.BatchNorm2d(128)
        self.conv1x1_catBN2 = nn.BatchNorm2d(128)
        self.conv3x3_last = nn.Conv2d(128 * 3, 128 * 3, kernel_size=3, padding=1)
        self.conv3x3_catBN_last = nn.BatchNorm2d(128 * 3)
    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        H, W = pre_data.size(2), pre_data.size(3)
        feature_11=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(pre_data)))))
        feature_12=F.relu(self.conv_block_1bn2(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(post_data)))))
        # diff1 = torch.abs(feature_11 - feature_12)
        # diff1 = self.conv_diff_1(diff1)

        out11 = self.max_pool_1(feature_11)
        out12 = self.max_pool_1(feature_12)

        #stage2
        feature_21 = F.relu(self.conv_block_2bn21(self.conv_block_2conv2(F.relu(self.conv_block_2bn11(self.conv_block_2conv1(out11))))))
        feature_22 = F.relu(self.conv_block_2bn22(self.conv_block_2conv2(F.relu(self.conv_block_2bn12(self.conv_block_2conv1(out12))))))

        diff2_ori = torch.abs(feature_21 - feature_22)
        diff2=self.conv_diff_2(diff2_ori)
        weight2=self.diff_ag2(diff2_ori)

        out21 = torch.cat([feature_21*weight2, diff2], 1)
        out22 = torch.cat([feature_22*weight2, diff2], 1)

        out21 = self.max_pool_2(out21)
        out22 = self.max_pool_2(out22)

        #stage3
        feature_31 = F.relu(self.conv_block_3bn21(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21))))))
        feature_32 = F.relu(self.conv_block_3bn22(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn12(self.conv_block_3conv1(out22))))))

        diff3_ori = torch.abs(feature_31 - feature_32)
        diff3 = self.conv_diff_3(diff3_ori)
        weight3 = self.diff_ag3(diff3_ori)
        out31 = torch.cat([feature_31*weight3, diff3], 1)
        out32 = torch.cat([feature_32*weight3, diff3], 1)

        out31 = self.max_pool_3(out31)
        out32 = self.max_pool_3(out32)
        #stage4
        feature_41 = F.relu(self.conv_block_4bn21(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31))))))
        feature_42 = F.relu(self.conv_block_4bn22(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn12(self.conv_block_4conv1(out32))))))

        psp1 = F.relu(self.conv1x1(self.psp(feature_41)))
        fpa1 = F.relu(self.conv1x1_2(self.fpa(feature_41)))
        feature_41 = F.relu(self.conv1x1_catBN1(torch.cat([psp1, fpa1], dim=1)))

        psp2 = F.relu(self.conv1x1(self.psp(feature_42)))
        fpa2 = F.relu(self.conv1x1_2(self.fpa(feature_42)))
        feature_42 = F.relu(self.conv1x1_catBN2(torch.cat([psp2, fpa2], dim=1)))


        diff4_ori = torch.abs(feature_41 - feature_42)
        diff4 = self.conv_diff_4(diff4_ori)
        weight4 = self.diff_ag4(diff4_ori)

        out = torch.cat([feature_41*weight4 , feature_42*weight4 , diff4], 1)  # 128*3
        out = F.relu(self.conv3x3_catBN_last(self.conv3x3_last(out)))
        out=self.max_pool_4(out)

        up_feature_5 = self.up_sample_1(out)#128
        # concat_feature_5 = torch.cat([up_feature_5, torch.pow(feature_41 - feature_42,2)], dim=1)
        concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)#256
        # concat_feature_5 = torch.cat([up_feature_5, diff3_conv], dim=1)#128+64
        feature_5 = self.conv_block_5(concat_feature_5)#64

        up_feature_6 = self.up_sample_2(feature_5)#64
        # concat_feature_6 = torch.cat([up_feature_6, torch.pow(feature_31 - feature_32,2)], dim=1)
        concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)#128
        # concat_feature_6 = torch.cat([up_feature_6, diff2_conv], dim=1)#64+32
        feature_6 = self.conv_block_6(concat_feature_6)#32

        up_feature_7 = self.up_sample_3(feature_6)
        # concat_feature_7 = torch.cat([up_feature_7, torch.pow(feature_21 - feature_22,2)], dim=1)
        concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)#64
        # concat_feature_7 = torch.cat([up_feature_7, diff1_conv], dim=1)#32+16

        feature_7 = self.conv_block_7(concat_feature_7)
        up_feature_8 = self.up_sample_4(feature_7)
        # concat_feature_8 = torch.cat([up_feature_8, torch.pow(feature_11 - feature_12,2)], dim=1)
        concat_feature_8 = torch.cat([up_feature_8, torch.abs(feature_11 - feature_12)], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)

        return [output_feature, diff4_ori], [feature_7,feature_7]


class backbone2BN_conATSA2(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(backbone2BN_conATSA2, self).__init__()

        self.conv_block_1conv1=nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1conv2=nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1bn1=nn.BatchNorm2d(16)
        self.conv_block_1bn2=nn.BatchNorm2d(16)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)


        self.conv_block_2conv1=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn11=nn.BatchNorm2d(32)
        self.conv_block_2bn12=nn.BatchNorm2d(32)
        self.conv_block_2conv2=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn21=nn.BatchNorm2d(32)
        self.conv_block_2bn22=nn.BatchNorm2d(32)

        self.diff_ag2=SpatialAttention(32)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3conv1=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn11=nn.BatchNorm2d(64)
        self.conv_block_3bn12=nn.BatchNorm2d(64)
        self.conv_block_3conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn21=nn.BatchNorm2d(64)
        self.conv_block_3bn22=nn.BatchNorm2d(64)

        self.diff_ag3=SpatialAttention(64)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4conv1= nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn11=nn.BatchNorm2d(128)
        self.conv_block_4bn12=nn.BatchNorm2d(128)
        self.conv_block_4conv2=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn21 = nn.BatchNorm2d(128)
        self.conv_block_4bn22=nn.BatchNorm2d(128)
        self.diff_ag4=SpatialAttention(128)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=(128*3) , out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        c=0
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32+c, out_channels=(32+c)//2, kernel_size=3, padding=1),
            nn.BatchNorm2d((32+c)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=(32+c)//2, out_channels=out_dim, kernel_size=1, padding=0)
        )
        self.psp = SPM(32, 32, sizes=(1, 2, 3, 6))
        self.fpa = FPM(32)
        self.conv1x1 = nn.Conv2d(32, (32) // 2, kernel_size=1, stride=1, bias=False)
        self.conv1x1_2 = nn.Conv2d(32, (32) // 2, kernel_size=1, stride=1, bias=False)

        self.conv1x1_catBN1 = nn.BatchNorm2d(32)
        self.conv1x1_catBN2 = nn.BatchNorm2d(32)
        self.conv3x3_last = nn.Conv2d(128 * 3, 128 * 3, kernel_size=3, padding=1)
        self.conv3x3_catBN_last = nn.BatchNorm2d(128 * 3)
    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        H, W = pre_data.size(2), pre_data.size(3)
        feature_11=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(pre_data)))))
        feature_12=F.relu(self.conv_block_1bn2(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(post_data)))))
        # diff1 = torch.abs(feature_11 - feature_12)
        # diff1 = self.conv_diff_1(diff1)

        out11 = self.max_pool_1(feature_11)
        out12 = self.max_pool_1(feature_12)

        #stage2
        feature_21 = F.relu(self.conv_block_2bn21(self.conv_block_2conv2(F.relu(self.conv_block_2bn11(self.conv_block_2conv1(out11))))))
        feature_22 = F.relu(self.conv_block_2bn22(self.conv_block_2conv2(F.relu(self.conv_block_2bn12(self.conv_block_2conv1(out12))))))

        psp1 = F.relu(self.conv1x1(self.psp(feature_21)))
        fpa1 = F.relu(self.conv1x1_2(self.fpa(feature_21)))
        feature_41 = F.relu(self.conv1x1_catBN1(torch.cat([psp1, fpa1], dim=1)))

        psp2 = F.relu(self.conv1x1(self.psp(feature_22)))
        fpa2 = F.relu(self.conv1x1_2(self.fpa(feature_22)))
        feature_42 = F.relu(self.conv1x1_catBN2(torch.cat([psp2, fpa2], dim=1)))


        diff2_ori = torch.abs(feature_21 - feature_22)
        diff2=self.conv_diff_2(diff2_ori)
        weight2=self.diff_ag2(diff2_ori)

        out21 = torch.cat([feature_21*weight2, diff2], 1)
        out22 = torch.cat([feature_22*weight2, diff2], 1)

        out21 = self.max_pool_2(out21)
        out22 = self.max_pool_2(out22)

        #stage3
        feature_31 = F.relu(self.conv_block_3bn21(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21))))))
        feature_32 = F.relu(self.conv_block_3bn22(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn12(self.conv_block_3conv1(out22))))))

        diff3_ori = torch.abs(feature_31 - feature_32)
        diff3 = self.conv_diff_3(diff3_ori)
        weight3 = self.diff_ag3(diff3_ori)
        out31 = torch.cat([feature_31*weight3, diff3], 1)
        out32 = torch.cat([feature_32*weight3, diff3], 1)

        out31 = self.max_pool_3(out31)
        out32 = self.max_pool_3(out32)
        #stage4
        feature_41 = F.relu(self.conv_block_4bn21(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31))))))
        feature_42 = F.relu(self.conv_block_4bn22(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn12(self.conv_block_4conv1(out32))))))




        diff4_ori = torch.abs(feature_41 - feature_42)
        diff4 = self.conv_diff_4(diff4_ori)
        weight4 = self.diff_ag4(diff4_ori)

        out = torch.cat([feature_41*weight4 , feature_42*weight4 , diff4], 1)  # 128*3
        out = F.relu(self.conv3x3_catBN_last(self.conv3x3_last(out)))
        out=self.max_pool_4(out)

        up_feature_5 = self.up_sample_1(out)#128
        # concat_feature_5 = torch.cat([up_feature_5, torch.pow(feature_41 - feature_42,2)], dim=1)
        concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)#256
        # concat_feature_5 = torch.cat([up_feature_5, diff3_conv], dim=1)#128+64
        feature_5 = self.conv_block_5(concat_feature_5)#64

        up_feature_6 = self.up_sample_2(feature_5)#64
        # concat_feature_6 = torch.cat([up_feature_6, torch.pow(feature_31 - feature_32,2)], dim=1)
        concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)#128
        # concat_feature_6 = torch.cat([up_feature_6, diff2_conv], dim=1)#64+32
        feature_6 = self.conv_block_6(concat_feature_6)#32

        up_feature_7 = self.up_sample_3(feature_6)
        # concat_feature_7 = torch.cat([up_feature_7, torch.pow(feature_21 - feature_22,2)], dim=1)
        concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)#64
        # concat_feature_7 = torch.cat([up_feature_7, diff1_conv], dim=1)#32+16

        feature_7 = self.conv_block_7(concat_feature_7)
        up_feature_8 = self.up_sample_4(feature_7)
        # concat_feature_8 = torch.cat([up_feature_8, torch.pow(feature_11 - feature_12,2)], dim=1)
        concat_feature_8 = torch.cat([up_feature_8, torch.abs(feature_11 - feature_12)], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)

        return [output_feature, diff4_ori], feature_7


class backbone2BN_conATSA3(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(backbone2BN_conATSA3, self).__init__()

        self.conv_block_1conv1=nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1conv2=nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1bn1=nn.BatchNorm2d(16)
        self.conv_block_1bn2=nn.BatchNorm2d(16)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_2conv1=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn11=nn.BatchNorm2d(32)
        self.conv_block_2bn12=nn.BatchNorm2d(32)
        self.conv_block_2conv2=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn21=nn.BatchNorm2d(32)
        self.conv_block_2bn22=nn.BatchNorm2d(32)

        self.diff_ag2=SpatialAttention(32)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3conv1=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn11=nn.BatchNorm2d(64)
        self.conv_block_3bn12=nn.BatchNorm2d(64)
        self.conv_block_3conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn21=nn.BatchNorm2d(64)
        self.conv_block_3bn22=nn.BatchNorm2d(64)

        self.diff_ag3=SpatialAttention(64)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4conv1= nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn11=nn.BatchNorm2d(128)
        self.conv_block_4bn12=nn.BatchNorm2d(128)
        self.conv_block_4conv2=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn21 = nn.BatchNorm2d(128)
        self.conv_block_4bn22=nn.BatchNorm2d(128)
        self.diff_ag4=SpatialAttention(128)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=(128*3) , out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_block_7 = nn.Sequential(
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        c=0
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32+c, out_channels=(32+c)//2, kernel_size=3, padding=1),
            nn.BatchNorm2d((32+c)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=(32+c)//2, out_channels=out_dim, kernel_size=1, padding=0)
        )
        cpsp=64
        # self.psp = SPM(cpsp, cpsp, sizes=(1, 2, 3, 6))
        # self.fpa = FPM(cpsp)
        # self.conv1x1 = nn.Conv2d(cpsp, (cpsp) // 2, kernel_size=1, stride=1, bias=False)
        # self.conv1x1_2 = nn.Conv2d(cpsp, (cpsp) // 2, kernel_size=1, stride=1, bias=False)
        #
        # self.conv1x1_catBN1 = nn.BatchNorm2d(cpsp)
        # # self.conv1x1_catBN2 = nn.BatchNorm2d(32)
        self.conv3x3_last = nn.Conv2d(128 * 3, 128 * 3, kernel_size=3, padding=1)
        self.conv3x3_catBN_last = nn.BatchNorm2d(128 * 3)

        self.ASSP = ASSP(cpsp, (cpsp)//2)
        self.ASSPconv = nn.Sequential(
            nn.Conv2d(cpsp, (cpsp) // 2, kernel_size=1, padding=0),
            nn.BatchNorm2d((cpsp) // 2),
            nn.ReLU()
        )
        # self.conv_block_asspbn21 = nn.BatchNorm2d(cpsp)

    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        H, W = pre_data.size(2), pre_data.size(3)
        feature_11=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(pre_data)))))
        feature_12=F.relu(self.conv_block_1bn2(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(post_data)))))
        # diff1 = torch.abs(feature_11 - feature_12)
        # diff1 = self.conv_diff_1(diff1)

        out11 = self.max_pool_1(feature_11)
        out12 = self.max_pool_1(feature_12)

        #stage2
        feature_21 = F.relu(self.conv_block_2bn21(self.conv_block_2conv2(F.relu(self.conv_block_2bn11(self.conv_block_2conv1(out11))))))
        feature_22 = F.relu(self.conv_block_2bn22(self.conv_block_2conv2(F.relu(self.conv_block_2bn12(self.conv_block_2conv1(out12))))))

        diff2_ori = torch.abs(feature_21 - feature_22)
        diff2=self.conv_diff_2(diff2_ori)
        weight2=self.diff_ag2(diff2_ori)

        out21 = torch.cat([feature_21*weight2, diff2], 1)
        out22 = torch.cat([feature_22*weight2, diff2], 1)

        out21 = self.max_pool_2(out21)
        out22 = self.max_pool_2(out22)

        #stage3
        feature_31 = F.relu(self.conv_block_3bn21(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21))))))
        feature_32 = F.relu(self.conv_block_3bn22(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn12(self.conv_block_3conv1(out22))))))

        diff3_ori = torch.abs(feature_31 - feature_32)
        diff3 = self.conv_diff_3(diff3_ori)
        weight3 = self.diff_ag3(diff3_ori)
        out31 = torch.cat([feature_31*weight3, diff3], 1)
        out32 = torch.cat([feature_32*weight3, diff3], 1)

        out31 = self.max_pool_3(out31)
        out32 = self.max_pool_3(out32)
        #stage4
        feature_41 = F.relu(self.conv_block_4bn21(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31))))))
        feature_42 = F.relu(self.conv_block_4bn22(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn12(self.conv_block_4conv1(out32))))))

        diff4_ori = torch.abs(feature_41 - feature_42)
        diff4 = self.conv_diff_4(diff4_ori)
        weight4 = self.diff_ag4(diff4_ori)

        out = torch.cat([feature_41*weight4 , feature_42*weight4 , diff4], 1)  # 128*3
        out = F.relu(self.conv3x3_catBN_last(self.conv3x3_last(out)))
        out=self.max_pool_4(out)

        up_feature_5 = self.up_sample_1(out)#128
        # concat_feature_5 = torch.cat([up_feature_5, torch.pow(feature_41 - feature_42,2)], dim=1)
        concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)#256
        # concat_feature_5 = torch.cat([up_feature_5, diff3_conv], dim=1)#128+64
        feature_5 = self.conv_block_5(concat_feature_5)#64

        up_feature_6 = self.up_sample_2(feature_5)#64
        # concat_feature_6 = torch.cat([up_feature_6, torch.pow(feature_31 - feature_32,2)], dim=1)
        concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)#128
        # concat_feature_6 = torch.cat([up_feature_6, diff2_conv], dim=1)#64+32
        feature_6 = self.conv_block_6(concat_feature_6)#32

        up_feature_7 = self.up_sample_3(feature_6)
        # concat_feature_7 = torch.cat([up_feature_7, torch.pow(feature_21 - feature_22,2)], dim=1)
        concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)#64
        # concat_feature_7 = torch.cat([up_feature_7, diff1_conv], dim=1)#32+16

        # psp1 = F.relu(self.conv1x1(self.psp(concat_feature_7)))
        # fpa1 = F.relu(self.conv1x1_2(self.fpa(concat_feature_7)))
        # catpsp = F.relu(self.conv1x1_catBN1(torch.cat([psp1, fpa1], dim=1)))
        #
        assp=self.ASSP(concat_feature_7)
        asspConv=self.ASSPconv(concat_feature_7)
        feature_7=torch.cat([assp, asspConv], dim=1)
        feature_7 = self.conv_block_7(feature_7)
        up_feature_8 = self.up_sample_4(feature_7)
        # concat_feature_8 = torch.cat([up_feature_8, torch.pow(feature_11 - feature_12,2)], dim=1)
        concat_feature_8 = torch.cat([up_feature_8, torch.abs(feature_11 - feature_12)], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)

        return [output_feature, diff4_ori], feature_7


class backbone2BN_conATSA4(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(backbone2BN_conATSA4, self).__init__()

        self.conv_block_1conv1=nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1conv2=nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv_block_1bn1=nn.BatchNorm2d(16)
        self.conv_block_1bn2=nn.BatchNorm2d(16)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_2conv1=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn11=nn.BatchNorm2d(32)
        self.conv_block_2bn12=nn.BatchNorm2d(32)
        self.conv_block_2conv2=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv_block_2bn21=nn.BatchNorm2d(32)
        self.conv_block_2bn22=nn.BatchNorm2d(32)

        self.diff_ag2=SpatialAttention(32)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3conv1=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn11=nn.BatchNorm2d(64)
        self.conv_block_3bn12=nn.BatchNorm2d(64)
        self.conv_block_3conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn21=nn.BatchNorm2d(64)
        self.conv_block_3bn22=nn.BatchNorm2d(64)

        self.diff_ag3=SpatialAttention(64)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4conv1= nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn11=nn.BatchNorm2d(128)
        self.conv_block_4bn12=nn.BatchNorm2d(128)
        self.conv_block_4conv2=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn21 = nn.BatchNorm2d(128)
        self.conv_block_4bn22=nn.BatchNorm2d(128)
        self.diff_ag4=SpatialAttention(128)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=(128*3) , out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_block_7 = nn.Sequential(
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        c=0
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32+c, out_channels=(32+c)//2, kernel_size=3, padding=1),
            nn.BatchNorm2d((32+c)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=(32+c)//2, out_channels=out_dim, kernel_size=1, padding=0)
        )
        cpsp=64
        self.psp = SPM(cpsp, cpsp, sizes=(1, 2, 3, 6))
        self.fpa = FPM(cpsp)
        self.conv1x1 = nn.Conv2d(cpsp, (cpsp) // 2, kernel_size=1, stride=1, bias=False)
        self.conv1x1_2 = nn.Conv2d(cpsp, (cpsp) // 2, kernel_size=1, stride=1, bias=False)
        #
        self.conv1x1_catBN1 = nn.BatchNorm2d(cpsp)
        # self.conv1x1_catBN2 = nn.BatchNorm2d(32)
        self.conv3x3_last = nn.Conv2d(128 * 3, 128 * 3, kernel_size=3, padding=1)
        self.conv3x3_catBN_last = nn.BatchNorm2d(128 * 3)
        #
        # self.ASSP = ASSP(cpsp, (cpsp)//2)
        # self.ASSPconv = nn.Sequential(
        #     nn.Conv2d(cpsp, (cpsp) // 2, kernel_size=1, padding=0),
        #     nn.BatchNorm2d((cpsp) // 2),
        #     nn.ReLU()
        # )
        # self.conv_block_asspbn21 = nn.BatchNorm2d(cpsp)

    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        H, W = pre_data.size(2), pre_data.size(3)
        feature_11=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(pre_data)))))
        feature_12=F.relu(self.conv_block_1bn2(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(post_data)))))
        # diff1 = torch.abs(feature_11 - feature_12)
        # diff1 = self.conv_diff_1(diff1)

        out11 = self.max_pool_1(feature_11)
        out12 = self.max_pool_1(feature_12)

        #stage2
        feature_21 = F.relu(self.conv_block_2bn21(self.conv_block_2conv2(F.relu(self.conv_block_2bn11(self.conv_block_2conv1(out11))))))
        feature_22 = F.relu(self.conv_block_2bn22(self.conv_block_2conv2(F.relu(self.conv_block_2bn12(self.conv_block_2conv1(out12))))))

        diff2_ori = torch.abs(feature_21 - feature_22)
        diff2=self.conv_diff_2(diff2_ori)
        weight2=self.diff_ag2(diff2_ori)

        out21 = torch.cat([feature_21*weight2, diff2], 1)
        out22 = torch.cat([feature_22*weight2, diff2], 1)

        out21 = self.max_pool_2(out21)
        out22 = self.max_pool_2(out22)

        #stage3
        feature_31 = F.relu(self.conv_block_3bn21(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21))))))
        feature_32 = F.relu(self.conv_block_3bn22(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn12(self.conv_block_3conv1(out22))))))

        diff3_ori = torch.abs(feature_31 - feature_32)
        diff3 = self.conv_diff_3(diff3_ori)
        weight3 = self.diff_ag3(diff3_ori)
        out31 = torch.cat([feature_31*weight3, diff3], 1)
        out32 = torch.cat([feature_32*weight3, diff3], 1)

        out31 = self.max_pool_3(out31)
        out32 = self.max_pool_3(out32)
        #stage4
        feature_41 = F.relu(self.conv_block_4bn21(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31))))))
        feature_42 = F.relu(self.conv_block_4bn22(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn12(self.conv_block_4conv1(out32))))))

        diff4_ori = torch.abs(feature_41 - feature_42)
        diff4 = self.conv_diff_4(diff4_ori)
        weight4 = self.diff_ag4(diff4_ori)

        out = torch.cat([feature_41*weight4 , feature_42*weight4 , diff4], 1)  # 128*3
        out = F.relu(self.conv3x3_catBN_last(self.conv3x3_last(out)))
        out=self.max_pool_4(out)

        up_feature_5 = self.up_sample_1(out)#128
        # concat_feature_5 = torch.cat([up_feature_5, torch.pow(feature_41 - feature_42,2)], dim=1)
        concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)#256
        # concat_feature_5 = torch.cat([up_feature_5, diff3_conv], dim=1)#128+64
        feature_5 = self.conv_block_5(concat_feature_5)#64

        up_feature_6 = self.up_sample_2(feature_5)#64
        # concat_feature_6 = torch.cat([up_feature_6, torch.pow(feature_31 - feature_32,2)], dim=1)
        concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)#128
        # concat_feature_6 = torch.cat([up_feature_6, diff2_conv], dim=1)#64+32
        feature_6 = self.conv_block_6(concat_feature_6)#32

        up_feature_7 = self.up_sample_3(feature_6)
        # concat_feature_7 = torch.cat([up_feature_7, torch.pow(feature_21 - feature_22,2)], dim=1)
        concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)#64
        # concat_feature_7 = torch.cat([up_feature_7, diff1_conv], dim=1)#32+16

        psp1 = F.relu(self.conv1x1(self.psp(concat_feature_7)))
        fpa1 = F.relu(self.conv1x1_2(self.fpa(concat_feature_7)))
        catpsp = F.relu(self.conv1x1_catBN1(torch.cat([psp1, fpa1], dim=1)))

        # assp=self.ASSP(concat_feature_7)
        # asspConv=self.ASSPconv(concat_feature_7)
        # feature_7=torch.cat([assp, asspConv], dim=1)
        feature_7 = self.conv_block_7(catpsp)
        up_feature_8 = self.up_sample_4(feature_7)
        # concat_feature_8 = torch.cat([up_feature_8, torch.pow(feature_11 - feature_12,2)], dim=1)
        concat_feature_8 = torch.cat([up_feature_8, torch.abs(feature_11 - feature_12)], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)

        return [output_feature, diff4_ori], feature_7