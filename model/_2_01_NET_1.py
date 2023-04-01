
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from net.Res2Net import res2net50_v1b_26w_4s
from net.backbone.get_backbone import get_backbone
"""
    把swin 的层数调到了2048 效果反而还没有1024层好
"""
class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class EAM(nn.Module):
    def __init__(self,inplanes4=1024,inplanes1=256):
        super(EAM, self).__init__()
        self.reduce1 = Conv1x1(inplanes1, 64)
        self.reduce4 = Conv1x1(inplanes4, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x4, x1):
        """

        :param x4: S4(B*1024*12*12)
        :param x1: R1(B*256*96*96)
        :return: EDGE b*1*96*96
        """
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)
        out = torch.sigmoid(out)
        return out
class CAM(nn.Module):
    def __init__(self, hchannel, channel):
        super(CAM, self).__init__()
        self.conv1_1 = Conv1x1(hchannel + channel, channel)
        self.conv3_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=2)
        self.dconv7_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=3)
        self.dconv9_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=4)
        self.conv1_2 = Conv1x1(channel, channel)
        self.conv3_3 = ConvBNR(channel, channel, 3)

    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((lf, hf), dim=1)
        x = self.conv1_1(x)
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.dconv5_1(xc[1] + x0 + xc[2])
        x2 = self.dconv7_1(xc[2] + x1 + xc[3])
        x3 = self.dconv9_1(xc[3] + x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.conv3_3(x + xx)

        return x
class EFM(nn.Module):
    def __init__(self, channel):
        super(EFM, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)# channel wise global attention
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c
        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei
        return x
class FDM(nn.Module):
    def __init__(self,inplanes,pre_planes,planes):
        super(FDM, self).__init__()
        self.efm = EFM(inplanes)
        self.reduce = Conv1x1(inplanes,planes)
        self.cam = CAM(pre_planes,planes)
    def forward(self,feature,edge,flow=None):
        efm_out = self.efm(feature,edge)
        efm_out = self.reduce(efm_out)
        if flow == None:
            return efm_out
        cam_out = self.cam(efm_out,flow)
        return cam_out

class BoostBlock(nn.Module):
    num = 4
    def __init__(self, inplanes_s=256,inplanes_r = 512,pre_planes=256,planes=128):
        super(BoostBlock, self).__init__()
        self.fdm_s = FDM(inplanes_s,pre_planes,planes)
        self.fdm_r = FDM(inplanes_r,pre_planes,planes)
        self.sft = ConvBNR(planes,planes)
        self.sft_1 = ConvBNR(planes,planes)
        self.sft_2 = ConvBNR(planes,planes)

    def forward(self, s,r,edge,flow = None):
        """
        :param s: swin feature
        :param r: res feature
        :param flow: pre feature
        :return: next flow
        """

        fdm_s_out = self.fdm_s(s,edge,flow)
        fdm_r_out = self.fdm_r(r,edge,flow)
        r_s = self.sft(fdm_r_out + fdm_s_out)
        r_s = self.sft_1(r_s + fdm_s_out)
        out = self.sft_2(r_s  + fdm_r_out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.swin=get_backbone()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.eam = EAM(inplanes4 = 1024,inplanes1=256)
        self.boost1 = BoostBlock(inplanes_s=1024,inplanes_r=2048,pre_planes=256,planes=256)
        self.boost2 = BoostBlock(inplanes_s=512,inplanes_r=1024,pre_planes=256,planes=256)
        self.boost3 = BoostBlock(inplanes_s=256,inplanes_r=512,pre_planes=256,planes=128)
        self.boost4 = BoostBlock(inplanes_s=128,inplanes_r=256,pre_planes=128,planes=64)
        self.predict1 = nn.Conv2d(256,1,1)
        self.predict2 = nn.Conv2d(128,1,1)
        self.predict3 = nn.Conv2d(64,1,1)


    def forward(self,x):
        r1, r2, r3, r4 = self.resnet(x)
        s1, s2, s3, s4 = self.swin(x)
        # r1, r2, r3, r4 = [
        #     torch.randn(2, 256, 96, 96),
        #     torch.randn(2, 512, 48, 48),
        #     torch.randn(2, 1024, 24, 24),
        #     torch.randn(2, 2048, 12, 12)
        # ]
        # s1, s2, s3, s4 = [
        #     torch.randn(2, 128, 96, 96),
        #     torch.randn(2, 256, 48, 48),
        #     torch.randn(2, 512, 24, 24),
        #     torch.randn(2, 1024, 12, 12)
        # ]
        edge = self.eam(s4,r1)  # 1*96*96

        flow1 = self.boost1(s4,r4,edge)
        flow2 = self.boost2(s3,r3, edge,flow1)
        flow3 = self.boost3(s2,r2,edge,flow2)
        pred = self.boost4(s1,r1,edge,flow3)
        flow2 = F.interpolate(self.predict1(flow2), scale_factor=16, mode='bilinear', align_corners=False)
        flow3 = F.interpolate(self.predict2(flow3), scale_factor=8, mode='bilinear', align_corners=False)
        pred = self.predict3(pred)
        pred = F.interpolate(pred, scale_factor=4, mode='bilinear', align_corners=False)
        edge = F.interpolate(edge, scale_factor=4, mode='bilinear', align_corners=False)
        return flow2,flow3,pred,edge


if __name__ == '__main__':
    # boost = BoostBlock(inplanes_s=512,inplanes_r=1024,index=1)
    # feature_resnet = torch.randn(1,1024,24,24)
    # feature_swim = torch.randn(1,512,24,24)
    # flow = torch.randn(1,64,12,12)
    # edge = torch.randn(1,1,96,96)
    # out = boost(feature_swim,feature_resnet,edge,flow)
    # print('boost test:',out[0].shape)
    net =Net()
    image = torch.randn(2,3,384,384)
    out = net(image)
    print('net test ',out[1].shape)



