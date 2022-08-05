import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d

class _PSPModulenobn(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer):
        super(_PSPModulenobn, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, pool_size, norm_layer)
                                     for pool_size in pool_sizes])
        out_channelsn=in_channels // 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channelsn,
                      kernel_size=3, padding=1, bias=False),
            # norm_layer(out_channelsn),
            nn.ReLU(),
            # nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # bn = norm_layer(out_channels)
        relu = nn.ReLU()
        return nn.Sequential(prior, conv, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]

        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output,pyramids
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
            norm_layer(out_channelsn),
            nn.ReLU(),
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
        return output,pyramids


def assp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(inplace=True))
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
        # x = self.dropout(self.relu(x))
        x=self.relu(x)
        return x

class ASSPtest(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASSPtest, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels,track_running_stats=False)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=6,
                               dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels,track_running_stats=False)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=12,
                               dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels,track_running_stats=False)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=18,
                               dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels,track_running_stats=False)
        self.conv5 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                               dilation=1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels,track_running_stats=False)
        self.convf = nn.Conv2d(in_channels=out_channels * 5, out_channels=out_channels, kernel_size=1, stride=1,
                               padding=0, dilation=1, bias=False)
        self.bnf = nn.BatchNorm2d(out_channels,track_running_stats=False)
        self.adapool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)
        x5 = self.adapool(x)
        x5 = self.conv5(x5)
        x5 = self.bn5(x5)
        x5 = self.relu(x5)
        x5 = F.interpolate(x5, size=tuple(x4.shape[-2:]), mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # channels first
        x = self.convf(x)
        x = self.bnf(x)
        x = self.relu(x)
        return x
class ASSPnobn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASSPnobn, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=6,
                               dilation=6, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=12,
                               dilation=12, bias=False)
        # self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=18,
                               dilation=18, bias=False)
        # self.bn4 = nn.BatchNorm2d(out_channels)
        self.conv5 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                               dilation=1, bias=False)
        # self.bn5 = nn.BatchNorm2d(out_channels)
        self.convf = nn.Conv2d(in_channels=out_channels * 5, out_channels=out_channels, kernel_size=1, stride=1,
                               padding=0, dilation=1, bias=False)
        # self.bnf = nn.BatchNorm2d(out_channels)
        self.adapool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1 = self.conv1(x)
        # x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = self.conv2(x)
        # x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x3 = self.conv3(x)
        # x3 = self.bn3(x3)
        x3 = self.relu(x3)
        x4 = self.conv4(x)
        # x4 = self.bn4(x4)
        x4 = self.relu(x4)
        x5 = self.adapool(x)
        x5 = self.conv5(x5)
        # x5 = self.bn5(x5)
        x5 = self.relu(x5)
        x5 = F.interpolate(x5, size=tuple(x4.shape[-2:]), mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # channels first
        x = self.convf(x)
        # x = self.bnf(x)
        x = self.relu(x)
        return x
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class FCSiamDiffBNnewMultiPSPDA(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(FCSiamDiffBNnewMultiPSPDA, self).__init__()

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

        # self.diff_se2 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(32, 32 // 2, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32 // 2, 32, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.diff_ag2=SpatialAttention()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)


        self.conv_block_3conv1=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn11=nn.BatchNorm2d(64)
        self.conv_block_3bn12=nn.BatchNorm2d(64)
        self.conv_block_3conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn21=nn.BatchNorm2d(64)
        self.conv_block_3bn22=nn.BatchNorm2d(64)
        # self.conv_block_3conv3= nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.conv_block_3bn31=nn.BatchNorm2d(64)
        # self.conv_block_3bn32=nn.BatchNorm2d(64)
        # self.diff_se3 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(64, 64 // 4, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64 // 4, 64, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.diff_ag3 = SpatialAttention()

        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4conv1= nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn11=nn.BatchNorm2d(128)
        self.conv_block_4bn12=nn.BatchNorm2d(128)
        self.conv_block_4conv2=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn21 = nn.BatchNorm2d(128)
        self.conv_block_4bn22 = nn.BatchNorm2d(128)
        # self.conv_block_4conv3=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.conv_block_4bn31 = nn.BatchNorm2d(128)
        # self.conv_block_4bn32 = nn.BatchNorm2d(128)
        # self.diff_se4 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(128, 128 // 4, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128 // 4, 128, kernel_size=1),
        #     nn.Sigmoid()
        # )

        # self.diff_se4 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(128, 128 // 4, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128 // 4, 128, kernel_size=1),
        #     nn.Sigmoid()
        # )

        self.diff_ag4 = SpatialAttention()
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
        norm_layer = nn.BatchNorm2d
        # self.psp=_PSPModule(32, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer)
        self.pspnobn = _PSPModulenobn(128, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer)

        # self.master_branch = nn.Sequential(
        #     _PSPModule(32, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
        #     # nn.Conv2d(256 // 4, num_classes, kernel_size=1)
        # )

        self.aux_branch = True

        if self.aux_branch:
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                # norm_layer(16),
                nn.ReLU(),
                # nn.Dropout2d(0.1),
                # nn.Conv2d(out_channel // 8, num_classes, kernel_size=1)
            )

        for p in self.parameters():
            p.requires_grad=False

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

        # self.conv_diff_1 = nn.Sequential(
        #     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
        #     nn.ReLU()
        # )

        # self.ASSP = ASSP(256, 256)
        # self.ASSPconv = nn.Sequential(
        #     nn.Conv2d(256, 48, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(48, track_running_stats=False),
        #     nn.ReLU()
        # )
        # self.ASSP = ASSP(32, 32)
        # self.ASSPconv = nn.Sequential(
        #     nn.Conv2d(32, 16, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(16, track_running_stats=False),
        #     nn.ReLU()
        # )


    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        H, W = pre_data.size(2), pre_data.size(3)
        feature_11=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(pre_data)))))
        feature_12=F.relu(self.conv_block_1bn2(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(post_data)))))

        # feature_11 = self.conv_block_1(pre_data)
        # feature_12 = self.conv_block_1(post_data)
        # diff1=torch.abs(feature_11-feature_12)
        # out11=torch.cat([feature_11,diff1],1)
        out11 = self.max_pool_1(feature_11)
        out12 = self.max_pool_1(feature_12)

        #stage2
        feature_21 = F.relu(self.conv_block_2bn21(self.conv_block_2conv2(F.relu(self.conv_block_2bn11(self.conv_block_2conv1(out11))))))
        feature_22 = F.relu(self.conv_block_2bn22(self.conv_block_2conv2(F.relu(self.conv_block_2bn12(self.conv_block_2conv1(out12))))))
        # feature_21 = self.conv_block_2(out11)
        # feature_22 = self.conv_block_2(out12)
        diff2 = torch.abs(feature_21 - feature_22)
        diff2=self.conv_diff_2(diff2)
        # diff2_weightse = self.diff_se2(diff2)
        diff2_weight = self.diff_ag2(diff2)
        # feature_21=feature_21*diff2_weight
        # feature_22=feature_22*diff2_weight
        out21 = torch.cat([feature_21*diff2_weight, diff2], 1)
        out22 = torch.cat([feature_22*diff2_weight, diff2], 1)

        out21 = self.max_pool_2(out21)
        out22 = self.max_pool_2(out22)

        #stage3
        # feature_31 = F.relu(self.conv_block_3bn31(self.conv_block_3conv3(F.relu(self.conv_block_3bn21(self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21)))))))))
        # feature_32 = F.relu(self.conv_block_3bn32(self.conv_block_3conv3(F.relu(self.conv_block_3bn22(self.conv_block_3conv2(F.relu(self.conv_block_3bn12(self.conv_block_3conv1(out22)))))))))
        feature_31 = F.relu(self.conv_block_3bn21(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21))))))
        feature_32 = F.relu(self.conv_block_3bn22(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn12(self.conv_block_3conv1(out22))))))

        diff3 = torch.abs(feature_31 - feature_32)
        diff3 = self.conv_diff_3(diff3)
        # diff3_weightse = self.diff_se3(diff3)
        diff3_weight = self.diff_ag3(diff3)
        # feature_31 = self.conv_block_3(out21)
        # feature_32 = self.conv_block_3(out22)
        # feature_31 = feature_31 * diff3_weight
        # feature_32 = feature_32 * diff3_weight
        out31 = torch.cat([feature_31* diff3_weight, diff3], 1)
        out32 = torch.cat([feature_32* diff3_weight, diff3], 1)

        out31 = self.max_pool_3(out31)
        out32 = self.max_pool_3(out32)
        #stage4

        # feature_41 = F.relu(self.conv_block_4bn31(self.conv_block_4conv3(F.relu(self.conv_block_4bn21(self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31)))))))))
        # feature_42 = F.relu(self.conv_block_4bn32(self.conv_block_4conv3(F.relu(self.conv_block_4bn22(self.conv_block_4conv2(F.relu(self.conv_block_4bn12(self.conv_block_4conv1(out32)))))))))
        feature_41 = F.relu(self.conv_block_4bn21(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31))))))
        feature_42 = F.relu(self.conv_block_4bn22(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn12(self.conv_block_4conv1(out32))))))
        feature_41_psp,_=self.pspnobn(feature_41)
        feature_41_conv=self.auxiliary_branch(feature_41)
        feature_41=torch.cat([feature_41_psp,feature_41_conv],1)
        feature_42_psp,_ = self.pspnobn(feature_42)
        feature_42_conv = self.auxiliary_branch(feature_42)
        feature_42=torch.cat([feature_42_psp,feature_42_conv],1)

        feature_DA=torch.cat([feature_41,feature_42],1)
        # feature_41 = self.conv_block_4(out31)#128
        # feature_42 = self.conv_block_4(out32)
        diff4_ori = torch.abs(feature_41 - feature_42)
        diff4 = self.conv_diff_4(diff4_ori)
        # diff4_weightse = self.diff_se4(diff4)
        diff4_weight = self.diff_ag4(diff4)
        # feature_41 = feature_41 * diff4_weight
        # feature_42 = feature_42 * diff4_weight
        # out=torch.cat([feature_41,feature_42],1)#128*3
        out = torch.cat([feature_41 * diff4_weight, feature_42 * diff4_weight, diff4], 1)  # 128*3

        # out=torch.cat([feature_41,feature_42,diff4],1)#128*3
        out=self.max_pool_4(out)
        # out_PSP = self.master_branch(out)
        # out_conv = self.auxiliary_branch(out)
        # out=torch.cat([out_PSP, out_conv], 1)#128*2
        # out=out_ASSP+out_conv#128*2

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

        # out_PSP = self.master_branch(concat_feature_8)
        # out_PSP,multifeature=self.psp(concat_feature_8)

        # out_conv = self.auxiliary_branch(concat_feature_8)
        # out=torch.cat([out_PSP, out_conv], 1)#32
        output_feature = self.conv_block_8(concat_feature_8)

        # out=self.outDropout(out)

        # output_feature=self.convout(out)

        # diff4 = F.interpolate(diff4, size=(H, W), mode='bilinear', align_corners=True)#nouse
        DAfeature=[feature_DA,feature_7]
        # output = F.softmax(output_feature, dim=1)
        # return output_feature, output_feature
        # flag=False
        # for module in self.modules():
        #     if isinstance(module, nn.BatchNorm2d):
        #         for name in module.state_dict():
        #             if name=='bias':
        #                 print('bias',module.state_dict()[name])
        #                 flag = True
        #     if flag:
        #         break
        return [output_feature, diff4_ori], DAfeature
    def unfreeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()
            if isinstance(module, nn.Dropout):
                module.train()
    def freeze_bn_dr(self):
        # if a==True:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            if isinstance(module, nn.Dropout):
                module.eval()

class FCSiamDiffBNnewMultiPSPDAnobn(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(FCSiamDiffBNnewMultiPSPDAnobn, self).__init__()

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

        # self.diff_se2 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(32, 32 // 2, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32 // 2, 32, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.diff_ag2=SpatialAttention()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)


        self.conv_block_3conv1=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn11=nn.BatchNorm2d(64)
        self.conv_block_3bn12=nn.BatchNorm2d(64)
        self.conv_block_3conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn21=nn.BatchNorm2d(64)
        self.conv_block_3bn22=nn.BatchNorm2d(64)
        # self.conv_block_3conv3= nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.conv_block_3bn31=nn.BatchNorm2d(64)
        # self.conv_block_3bn32=nn.BatchNorm2d(64)
        # self.diff_se3 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(64, 64 // 4, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64 // 4, 64, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.diff_ag3 = SpatialAttention()

        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4conv1= nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn11=nn.BatchNorm2d(128)
        self.conv_block_4bn12=nn.BatchNorm2d(128)
        self.conv_block_4conv2=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn21 = nn.BatchNorm2d(128)
        self.conv_block_4bn22 = nn.BatchNorm2d(128)
        # self.conv_block_4conv3=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.conv_block_4bn31 = nn.BatchNorm2d(128)
        # self.conv_block_4bn32 = nn.BatchNorm2d(128)
        # self.diff_se4 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(128, 128 // 4, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128 // 4, 128, kernel_size=1),
        #     nn.Sigmoid()
        # )

        # self.diff_se4 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(128, 128 // 4, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128 // 4, 128, kernel_size=1),
        #     nn.Sigmoid()
        # )

        self.diff_ag4 = SpatialAttention()
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
        norm_layer = nn.BatchNorm2d
        # self.psp=_PSPModule(32, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer)
        self.pspnobn = _PSPModulenobn(128, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer)

        # self.master_branch = nn.Sequential(
        #     _PSPModule(32, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
        #     # nn.Conv2d(256 // 4, num_classes, kernel_size=1)
        # )

        self.aux_branch = True

        if self.aux_branch:
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                # norm_layer(16),
                nn.ReLU(),
                # nn.Dropout2d(0.1),
                # nn.Conv2d(out_channel // 8, num_classes, kernel_size=1)
            )

        for p in self.parameters():
            p.requires_grad=False

        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU()
        )
        c=0
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32+c, out_channels=(32+c)//2, kernel_size=3, padding=1),
            # nn.BatchNorm2d((32+c)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=(32+c)//2, out_channels=out_dim, kernel_size=1, padding=0)
        )

        # self.conv_diff_1 = nn.Sequential(
        #     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
        #     nn.ReLU()
        # )

        # self.ASSP = ASSP(256, 256)
        # self.ASSPconv = nn.Sequential(
        #     nn.Conv2d(256, 48, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(48, track_running_stats=False),
        #     nn.ReLU()
        # )
        # self.ASSP = ASSP(32, 32)
        # self.ASSPconv = nn.Sequential(
        #     nn.Conv2d(32, 16, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(16, track_running_stats=False),
        #     nn.ReLU()
        # )


    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        H, W = pre_data.size(2), pre_data.size(3)
        feature_11=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(pre_data)))))
        feature_12=F.relu(self.conv_block_1bn2(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(post_data)))))

        # feature_11 = self.conv_block_1(pre_data)
        # feature_12 = self.conv_block_1(post_data)
        # diff1=torch.abs(feature_11-feature_12)
        # out11=torch.cat([feature_11,diff1],1)
        out11 = self.max_pool_1(feature_11)
        out12 = self.max_pool_1(feature_12)

        #stage2
        feature_21 = F.relu(self.conv_block_2bn21(self.conv_block_2conv2(F.relu(self.conv_block_2bn11(self.conv_block_2conv1(out11))))))
        feature_22 = F.relu(self.conv_block_2bn22(self.conv_block_2conv2(F.relu(self.conv_block_2bn12(self.conv_block_2conv1(out12))))))
        # feature_21 = self.conv_block_2(out11)
        # feature_22 = self.conv_block_2(out12)
        diff2 = torch.abs(feature_21 - feature_22)
        diff2=self.conv_diff_2(diff2)
        # diff2_weightse = self.diff_se2(diff2)
        diff2_weight = self.diff_ag2(diff2)
        # feature_21=feature_21*diff2_weight
        # feature_22=feature_22*diff2_weight
        out21 = torch.cat([feature_21*diff2_weight, diff2], 1)
        out22 = torch.cat([feature_22*diff2_weight, diff2], 1)

        out21 = self.max_pool_2(out21)
        out22 = self.max_pool_2(out22)

        #stage3
        # feature_31 = F.relu(self.conv_block_3bn31(self.conv_block_3conv3(F.relu(self.conv_block_3bn21(self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21)))))))))
        # feature_32 = F.relu(self.conv_block_3bn32(self.conv_block_3conv3(F.relu(self.conv_block_3bn22(self.conv_block_3conv2(F.relu(self.conv_block_3bn12(self.conv_block_3conv1(out22)))))))))
        feature_31 = F.relu(self.conv_block_3bn21(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21))))))
        feature_32 = F.relu(self.conv_block_3bn22(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn12(self.conv_block_3conv1(out22))))))

        diff3 = torch.abs(feature_31 - feature_32)
        diff3 = self.conv_diff_3(diff3)
        # diff3_weightse = self.diff_se3(diff3)
        diff3_weight = self.diff_ag3(diff3)
        # feature_31 = self.conv_block_3(out21)
        # feature_32 = self.conv_block_3(out22)
        # feature_31 = feature_31 * diff3_weight
        # feature_32 = feature_32 * diff3_weight
        out31 = torch.cat([feature_31* diff3_weight, diff3], 1)
        out32 = torch.cat([feature_32* diff3_weight, diff3], 1)

        out31 = self.max_pool_3(out31)
        out32 = self.max_pool_3(out32)
        #stage4

        # feature_41 = F.relu(self.conv_block_4bn31(self.conv_block_4conv3(F.relu(self.conv_block_4bn21(self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31)))))))))
        # feature_42 = F.relu(self.conv_block_4bn32(self.conv_block_4conv3(F.relu(self.conv_block_4bn22(self.conv_block_4conv2(F.relu(self.conv_block_4bn12(self.conv_block_4conv1(out32)))))))))
        feature_41 = F.relu(self.conv_block_4bn21(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31))))))
        feature_42 = F.relu(self.conv_block_4bn22(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn12(self.conv_block_4conv1(out32))))))
        feature_41_psp,_=self.pspnobn(feature_41)
        feature_41_conv=self.auxiliary_branch(feature_41)
        feature_41=torch.cat([feature_41_psp,feature_41_conv],1)
        feature_42_psp,_ = self.pspnobn(feature_42)
        feature_42_conv = self.auxiliary_branch(feature_42)
        feature_42=torch.cat([feature_42_psp,feature_42_conv],1)

        feature_DA=torch.cat([feature_41,feature_42],1)
        # feature_41 = self.conv_block_4(out31)#128
        # feature_42 = self.conv_block_4(out32)
        diff4_ori = torch.abs(feature_41 - feature_42)
        diff4 = self.conv_diff_4(diff4_ori)
        # diff4_weightse = self.diff_se4(diff4)
        diff4_weight = self.diff_ag4(diff4)
        # feature_41 = feature_41 * diff4_weight
        # feature_42 = feature_42 * diff4_weight
        # out=torch.cat([feature_41,feature_42],1)#128*3
        out = torch.cat([feature_41 * diff4_weight, feature_42 * diff4_weight, diff4], 1)  # 128*3

        # out=torch.cat([feature_41,feature_42,diff4],1)#128*3
        out=self.max_pool_4(out)
        # out_PSP = self.master_branch(out)
        # out_conv = self.auxiliary_branch(out)
        # out=torch.cat([out_PSP, out_conv], 1)#128*2
        # out=out_ASSP+out_conv#128*2

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

        # out_PSP = self.master_branch(concat_feature_8)
        # out_PSP,multifeature=self.psp(concat_feature_8)

        # out_conv = self.auxiliary_branch(concat_feature_8)
        # out=torch.cat([out_PSP, out_conv], 1)#32
        output_feature = self.conv_block_8(concat_feature_8)

        # out=self.outDropout(out)

        # output_feature=self.convout(out)

        # diff4 = F.interpolate(diff4, size=(H, W), mode='bilinear', align_corners=True)#nouse
        DAfeature=[feature_DA,feature_7]
        # output = F.softmax(output_feature, dim=1)
        # return output_feature, output_feature
        # flag=False
        # for module in self.modules():
        #     if isinstance(module, nn.BatchNorm2d):
        #         for name in module.state_dict():
        #             if name=='bias':
        #                 print('bias',module.state_dict()[name])
        #                 flag = True
        #     if flag:
        #         break
        return [output_feature, diff4_ori], DAfeature


class FCSiamDiffBNnewMultiPSPDABN(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(FCSiamDiffBNnewMultiPSPDABN, self).__init__()

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

        # self.diff_se2 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(32, 32 // 2, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32 // 2, 32, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.diff_ag2=SpatialAttention()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)


        self.conv_block_3conv1=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn11=nn.BatchNorm2d(64)
        self.conv_block_3bn12=nn.BatchNorm2d(64)
        self.conv_block_3conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_block_3bn21=nn.BatchNorm2d(64)
        self.conv_block_3bn22=nn.BatchNorm2d(64)
        # self.conv_block_3conv3= nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.conv_block_3bn31=nn.BatchNorm2d(64)
        # self.conv_block_3bn32=nn.BatchNorm2d(64)
        # self.diff_se3 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(64, 64 // 4, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64 // 4, 64, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.diff_ag3 = SpatialAttention()

        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4conv1= nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn11=nn.BatchNorm2d(128)
        self.conv_block_4bn12=nn.BatchNorm2d(128)
        self.conv_block_4conv2=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_block_4bn21 = nn.BatchNorm2d(128)
        self.conv_block_4bn22 = nn.BatchNorm2d(128)
        # self.conv_block_4conv3=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.conv_block_4bn31 = nn.BatchNorm2d(128)
        # self.conv_block_4bn32 = nn.BatchNorm2d(128)
        # self.diff_se4 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(128, 128 // 4, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128 // 4, 128, kernel_size=1),
        #     nn.Sigmoid()
        # )

        # self.diff_se4 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(128, 128 // 4, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128 // 4, 128, kernel_size=1),
        #     nn.Sigmoid()
        # )

        self.diff_ag4 = SpatialAttention()
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
        # self.up_sample_3 = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU()
        # )
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
        norm_layer = nn.BatchNorm2d
        # self.psp=_PSPModule(32, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer)
        self.pspnobn = _PSPModulenobn(128, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer)

        # self.master_branch = nn.Sequential(
        #     _PSPModule(32, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
        #     # nn.Conv2d(256 // 4, num_classes, kernel_size=1)
        # )

        self.aux_branch = True

        if self.aux_branch:
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                # norm_layer(16),
                nn.ReLU(),
                # nn.Dropout2d(0.1),
                # nn.Conv2d(out_channel // 8, num_classes, kernel_size=1)
            )
        #
        # for p in self.parameters():
        #     print("p",p)
        #     p.requires_grad=False

        # self.conv_block_7 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU()
        # )
        # self.up_sample_4 = nn.Sequential(
        #     nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU()
        # )
        # c=0
        # self.conv_block_8 = nn.Sequential(
        #     nn.Conv2d(in_channels=32+c, out_channels=(32+c)//2, kernel_size=3, padding=1),
        #     nn.BatchNorm2d((32+c)//2),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=(32+c)//2, out_channels=out_dim, kernel_size=1, padding=0)
        # )

    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        H, W = pre_data.size(2), pre_data.size(3)
        feature_11=F.relu(self.conv_block_1bn1(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(pre_data)))))
        feature_12=F.relu(self.conv_block_1bn2(self.conv_block_1conv2(F.relu(self.conv_block_1conv1(post_data)))))

        # feature_11 = self.conv_block_1(pre_data)
        # feature_12 = self.conv_block_1(post_data)
        # diff1=torch.abs(feature_11-feature_12)
        # out11=torch.cat([feature_11,diff1],1)
        out11 = self.max_pool_1(feature_11)
        out12 = self.max_pool_1(feature_12)

        #stage2
        feature_21 = F.relu(self.conv_block_2bn21(self.conv_block_2conv2(F.relu(self.conv_block_2bn11(self.conv_block_2conv1(out11))))))
        feature_22 = F.relu(self.conv_block_2bn22(self.conv_block_2conv2(F.relu(self.conv_block_2bn12(self.conv_block_2conv1(out12))))))
        # feature_21 = self.conv_block_2(out11)
        # feature_22 = self.conv_block_2(out12)
        diff2 = torch.abs(feature_21 - feature_22)
        diff2=self.conv_diff_2(diff2)
        # diff2_weightse = self.diff_se2(diff2)
        diff2_weight = self.diff_ag2(diff2)
        # feature_21=feature_21*diff2_weight
        # feature_22=feature_22*diff2_weight
        out21 = torch.cat([feature_21*diff2_weight, diff2], 1)
        out22 = torch.cat([feature_22*diff2_weight, diff2], 1)

        out21 = self.max_pool_2(out21)
        out22 = self.max_pool_2(out22)

        #stage3
        # feature_31 = F.relu(self.conv_block_3bn31(self.conv_block_3conv3(F.relu(self.conv_block_3bn21(self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21)))))))))
        # feature_32 = F.relu(self.conv_block_3bn32(self.conv_block_3conv3(F.relu(self.conv_block_3bn22(self.conv_block_3conv2(F.relu(self.conv_block_3bn12(self.conv_block_3conv1(out22)))))))))
        feature_31 = F.relu(self.conv_block_3bn21(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn11(self.conv_block_3conv1(out21))))))
        feature_32 = F.relu(self.conv_block_3bn22(
            self.conv_block_3conv2(F.relu(self.conv_block_3bn12(self.conv_block_3conv1(out22))))))

        diff3 = torch.abs(feature_31 - feature_32)
        diff3 = self.conv_diff_3(diff3)
        # diff3_weightse = self.diff_se3(diff3)
        diff3_weight = self.diff_ag3(diff3)
        # feature_31 = self.conv_block_3(out21)
        # feature_32 = self.conv_block_3(out22)
        # feature_31 = feature_31 * diff3_weight
        # feature_32 = feature_32 * diff3_weight
        out31 = torch.cat([feature_31* diff3_weight, diff3], 1)
        out32 = torch.cat([feature_32* diff3_weight, diff3], 1)

        out31 = self.max_pool_3(out31)
        out32 = self.max_pool_3(out32)
        #stage4

        # feature_41 = F.relu(self.conv_block_4bn31(self.conv_block_4conv3(F.relu(self.conv_block_4bn21(self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31)))))))))
        # feature_42 = F.relu(self.conv_block_4bn32(self.conv_block_4conv3(F.relu(self.conv_block_4bn22(self.conv_block_4conv2(F.relu(self.conv_block_4bn12(self.conv_block_4conv1(out32)))))))))
        feature_41 = F.relu(self.conv_block_4bn21(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn11(self.conv_block_4conv1(out31))))))
        feature_42 = F.relu(self.conv_block_4bn22(
            self.conv_block_4conv2(F.relu(self.conv_block_4bn12(self.conv_block_4conv1(out32))))))
        feature_41_psp,_=self.pspnobn(feature_41)
        feature_41_conv=self.auxiliary_branch(feature_41)
        feature_41=torch.cat([feature_41_psp,feature_41_conv],1)
        feature_42_psp,_ = self.pspnobn(feature_42)
        feature_42_conv = self.auxiliary_branch(feature_42)
        feature_42=torch.cat([feature_42_psp,feature_42_conv],1)

        feature_DA=torch.cat([feature_41,feature_42],1)
        # feature_41 = self.conv_block_4(out31)#128
        # feature_42 = self.conv_block_4(out32)
        diff4_ori = torch.abs(feature_41 - feature_42)
        diff4 = self.conv_diff_4(diff4_ori)
        # diff4_weightse = self.diff_se4(diff4)
        diff4_weight = self.diff_ag4(diff4)
        # feature_41 = feature_41 * diff4_weight
        # feature_42 = feature_42 * diff4_weight
        # out=torch.cat([feature_41,feature_42],1)#128*3
        out = torch.cat([feature_41 * diff4_weight, feature_42 * diff4_weight, diff4], 1)  # 128*3

        # out=torch.cat([feature_41,feature_42,diff4],1)#128*3
        out=self.max_pool_4(out)
        # out_PSP = self.master_branch(out)
        # out_conv = self.auxiliary_branch(out)
        # out=torch.cat([out_PSP, out_conv], 1)#128*2
        # out=out_ASSP+out_conv#128*2

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
        flag = False
        # for module in self.modules():
        #     if isinstance(module, nn.BatchNorm2d):
        #         for name in module.state_dict():
        #             if name=='weight':
        #                 print('weight',module.state_dict()[name])
        #                 flag = True
        #     if flag:
        #         break
        return feature_DA,feature_6,feature_21,feature_22,feature_11,feature_12,diff4_ori
        # up_feature_7 = self.up_sample_3(feature_6)
        # # concat_feature_7 = torch.cat([up_feature_7, torch.pow(feature_21 - feature_22,2)], dim=1)
        # concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)#64
        # # concat_feature_7 = torch.cat([up_feature_7, diff1_conv], dim=1)#32+16
        #
        #
        # feature_7 = self.conv_block_7(concat_feature_7)
        # up_feature_8 = self.up_sample_4(feature_7)
        # # concat_feature_8 = torch.cat([up_feature_8, torch.pow(feature_11 - feature_12,2)], dim=1)
        # concat_feature_8 = torch.cat([up_feature_8, torch.abs(feature_11 - feature_12)], dim=1)
        #
        # # out_PSP = self.master_branch(concat_feature_8)
        # # out_PSP,multifeature=self.psp(concat_feature_8)
        #
        # # out_conv = self.auxiliary_branch(concat_feature_8)
        # # out=torch.cat([out_PSP, out_conv], 1)#32
        # output_feature = self.conv_block_8(concat_feature_8)
        #
        # # out=self.outDropout(out)
        #
        # # output_feature=self.convout(out)
        #
        # # diff4 = F.interpolate(diff4, size=(H, W), mode='bilinear', align_corners=True)#nouse
        # DAfeature=[feature_DA,feature_7]
        # # output = F.softmax(output_feature, dim=1)
        # # return output_feature, output_feature
        # flag=False
        # for module in self.modules():
        #     if isinstance(module, nn.BatchNorm2d):
        #         for name in module.state_dict():
        #             if name=='weight':
        #                 print('weight',module.state_dict()[name])
        #                 flag = True
        #     if flag:
        #         break
        # return [output_feature, diff4_ori], DAfeature
    def unfreeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()
            if isinstance(module, nn.Dropout):
                module.train()
    def freeze_bn_dr(self):
        # if a==True:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            if isinstance(module, nn.Dropout):
                module.eval()

class FCSiamDiffoneBNlow(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(FCSiamDiffoneBNlow, self).__init__()

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


        self.diff_ag2=SpatialAttention()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.ReLU()
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

        diff2 = torch.abs(feature_21 - feature_22)
        diff2=self.conv_diff_2(diff2)
        diff2_weight = self.diff_ag2(diff2)

        out21 = torch.cat([feature_21*diff2_weight, diff2], 1)
        out22 = torch.cat([feature_22*diff2_weight, diff2], 1)

        out21 = self.max_pool_2(out21)
        out22 = self.max_pool_2(out22)
        return out21,out22,feature_21, feature_22,feature_11, feature_12

    def unfreeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()
            if isinstance(module, nn.Dropout):
                module.train()
    def freeze_bn_dr(self):
        # if a==True:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            if isinstance(module, nn.Dropout):
                module.eval()

class FCSiamDiffBNlow(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(FCSiamDiffBNlow, self).__init__()

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


        self.diff_ag2=SpatialAttention()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.ReLU()
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

        diff2 = torch.abs(feature_21 - feature_22)
        diff2=self.conv_diff_2(diff2)
        diff2_weight = self.diff_ag2(diff2)

        out21 = torch.cat([feature_21*diff2_weight, diff2], 1)
        out22 = torch.cat([feature_22*diff2_weight, diff2], 1)

        out21 = self.max_pool_2(out21)
        out22 = self.max_pool_2(out22)
        return out21,out22,feature_21, feature_22,feature_11, feature_12

    def unfreeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()
            if isinstance(module, nn.Dropout):
                module.train()
    def freeze_bn_dr(self):
        # if a==True:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            if isinstance(module, nn.Dropout):
                module.eval()
class FCSiamConc(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(FCSiamConc, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up_sample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up_sample_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up_sample_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=out_dim, kernel_size=1, padding=0),
        )

    def encoder(self, input_data):
        #####################
        # encoder
        #####################
        feature_1 = self.conv_block_1(input_data)
        down_feature_1 = self.max_pool_1(feature_1)

        feature_2 = self.conv_block_2(down_feature_1)
        down_feature_2 = self.max_pool_2(feature_2)

        feature_3 = self.conv_block_3(down_feature_2)
        down_feature_3 = self.max_pool_3(feature_3)

        feature_4 = self.conv_block_4(down_feature_3)
        down_feature_4 = self.max_pool_4(feature_4)

        return down_feature_4, feature_4, feature_3, feature_2, feature_1

    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        down_feature_41, feature_41, feature_31, feature_21, feature_11 = self.encoder(pre_data)
        down_feature_42, feature_42, feature_32, feature_22, feature_12 = self.encoder(post_data)

        up_feature_5 = self.up_sample_1(down_feature_41)
        concat_feature_5 = torch.cat([up_feature_5, feature_41, feature_42], dim=1)
        feature_5 = self.conv_block_5(concat_feature_5)

        up_feature_6 = self.up_sample_2(feature_5)
        concat_feature_6 = torch.cat([up_feature_6, feature_31, feature_32], dim=1)
        feature_6 = self.conv_block_6(concat_feature_6)

        up_feature_7 = self.up_sample_3(feature_6)
        concat_feature_7 = torch.cat([up_feature_7, feature_21, feature_22], dim=1)
        feature_7 = self.conv_block_7(concat_feature_7)

        up_feature_8 = self.up_sample_4(feature_7)
        concat_feature_8 = torch.cat([up_feature_8, feature_11, feature_12], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)
        output = F.softmax(output_feature, dim=1)
        return output_feature, output