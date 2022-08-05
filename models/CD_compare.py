import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
from models.appendix import ASSP,BaseModel,initialize_weights
from torchvision import models
class ConvBlock(nn.Module):
    def __init__(self,dim_in,dim_feature):
        super(ConvBlock, self).__init__()
        self.conv=nn.Conv2d(dim_in, dim_feature, kernel_size=7, padding=3,stride=1, bias=False)
        self.bn = nn.BatchNorm2d(dim_feature)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        x=self.max_pool(x)
        return x

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
        residual = x
        residual = self.convplus(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()
        self.conv0 = nn.Conv2d(low_level_channels, 48, 1,  bias=False)
        self.bn0 = nn.BatchNorm2d(48)
        # self.conv1 = nn.Conv2d(256, 128, 1,  bias=False)
        # self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        # self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        # self.bn1 = nn.BatchNorm2d(48)
        # self.relu = nn.ReLU()

        # Table 2, best performance with two 3x3 convs
        self.output = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=False),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Conv2d(128, 128, 1, stride=1),
        )
        initialize_weights(self)

    def forward(self, x, low_level_features):
        low_level_features = self.conv0(low_level_features)
        low_level_features = self.relu(self.bn0(low_level_features))
        # low_level_features = self.conv1(low_level_features)
        # low_level_features = self.relu(self.bn1(low_level_features))
        H, W = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x

class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet34', pretrained=True):
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
        initialize_weights(self.layer3)
        initialize_weights(self.layer4)

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

        x21 = self.layer1(x11)
        x22 = self.layer1(x12)
        diff2=torch.abs(x21-x22)
        x31 = self.layer2(x21)
        x32 = self.layer2(x22)

        x41 = self.layer3(x31)
        x42 = self.layer3(x32)


        return [x11, x21, x31, x41], [x12, x22, x32, x42]

class DeepLabori(BaseModel):
    def __init__(self, in_channels=3,num_classes=2,  backbone='resnet', pretrained=True,
                 output_stride=16, freeze_bn=False, **_):

        super(DeepLabori, self).__init__()
        assert ('xception' or 'resnet' in backbone)
        if 'resnet' in backbone:
            self.backbone = ResNet(in_channels=in_channels, output_stride=output_stride, pretrained=pretrained)
            low_level_channels = 256


        self.ASSP = ASSP(in_channels=256*3, output_stride=output_stride)
        self.decoder = Decoder(low_level_channels, num_classes)
        self.upsample=nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(32, 2, 1, stride=1),
        )
        # self.sa_diff=SpatialAttention()
        # self.diff_se=SeModule(512)
        # self.diff_se=ChannelAttention(512)
        if freeze_bn: self.freeze_bn()

    def forward(self, x1,x2):
        feature1,feature2=self.backbone(x1,x2)
        x = torch.cat([feature1[-1], feature2[-1],torch.abs(feature1[-1] - feature2[-1])], dim=1)
        low_level_features=torch.cat([feature1[2],feature2[2]],1)
        x = self.ASSP(x)
        x_DA = self.decoder(x, low_level_features)
        x=self.upsample(x_DA)
        return [x,x_DA], [x_DA,x_DA]

    def unfreeze_bn(self):
        # if a==True:
        self.ASSP.train()
        self.decoder.train()
        self.backbone.train()
    def freeze_bn_dr(self):
        # if a==True:
        self.ASSP.eval()
        self.decoder.eval()
        self.backbone.eval()
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

class FCSiamDiff(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(FCSiamDiff, self).__init__()

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
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=128*2, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up_sample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=out_dim, kernel_size=1, padding=0)
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
        out=torch.cat([down_feature_41,down_feature_42],1)
        up_feature_5 = self.up_sample_1(out)
        # print('up_feature_5',up_feature_5.shape,torch.abs(feature_41 - feature_42).shape)
        concat_feature_5 = torch.cat([up_feature_5, torch.pow(feature_41 - feature_42,2)], dim=1)
        feature_5 = self.conv_block_5(concat_feature_5)

        up_feature_6 = self.up_sample_2(feature_5)
        concat_feature_6 = torch.cat([up_feature_6, torch.pow(feature_31 - feature_32,2)], dim=1)
        feature_6 = self.conv_block_6(concat_feature_6)

        up_feature_7 = self.up_sample_3(feature_6)
        concat_feature_7 = torch.cat([up_feature_7, torch.pow(feature_21 - feature_22,2)], dim=1)
        feature_7 = self.conv_block_7(concat_feature_7)

        DA = self.up_sample_4(feature_7)
        concat_feature_8 = torch.cat([DA, torch.pow(feature_11 - feature_12,2)], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)
        output = F.softmax(output_feature, dim=1)
        return [output_feature,DA], [DA,DA]

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

        self.conv_diff_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

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



