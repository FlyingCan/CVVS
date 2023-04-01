import torch
import torch.nn as nn
import torch.nn.functional as F
 
from math import log
 
from net.backbone.get_backbone import get_backbone

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
    def __init__(self,inplanes4=1024,inplanes1=128):
        super(EAM, self).__init__()
        self.reduce1 = Conv1x1(inplanes1, 64)
        self.reduce4 = Conv1x1(inplanes4, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out


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

class BoostBlock(nn.Module):

    def __init__(self,inplanes=1024,axu_inplanes =128, index=0,mode = 'add',num=5):
        super(BoostBlock, self).__init__()
        self.mode = mode
        self.num = num
        self.index = index
        self.eam = EAM(inplanes,axu_inplanes)
        self.efm = EFM(inplanes)
        self.cam = CAM(inplanes,inplanes)
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.sigmoid = nn.Sigmoid()
    def forward(self,x4,x1):
        """

        :param x4: main flow
        :param x1: axu flow
        :return: B*(2*inplaens) *H(x4)*W if mode =cat and 0<index<num
                 B*(inplanes)*H(x4)*W if mode = add and 0<index<num
                 EDGE:B*1*H(x1)*W,mainflow:B*inplanes*H*W if index = 0
                 B*(inplanes)*H*W,B*(inplanes)*H*W if index = num-1
        """
        edge = self.eam(x4,x1)     #B*1*H/4*W/4
        edge_att = torch.sigmoid(edge)
        out1 = self.efm(x4,edge_att)   #B*inplanes*(x1.H)*(x1.W)
        out2 = self.cam(x4,out1)
        if x4.size() != edge_att.size():
            att = F.interpolate(edge_att, out2.size()[2:], mode='bilinear', align_corners=False)
        out = out2 * att + out2
        out = self.sigmoid(out)
        if self.index == 0:
            return edge_att,self.beta * out1 + (1 - self.beta) * out
        elif self.index == self.num-1:
            return self.beta * out1 + (1 - self.beta) * out,out1,out
        else:
            if self.mode == 'cat':
                out = torch.cat((out1,out),dim=1)
                return out
            if self.mode == 'add':
                out = self.beta * out1 + (1 - self.beta) * out
                return out
class Net(nn.Module):
    def __init__(self,path=None):
        super(Net, self).__init__()
        self.backbone=get_backbone()
        self.boost_1 = BoostBlock(inplanes=1024,axu_inplanes=128,index=0)
        self.boost_2 = BoostBlock(inplanes=1024,axu_inplanes=1024,index=1,mode='add')
        self.boost_3 = BoostBlock(inplanes=512,axu_inplanes=1024,index=2,mode='add')
        self.reduce3 = Conv1x1(1024,512)
        self.reduce2 = Conv1x1(512,256)
        self.boost_4 = BoostBlock(inplanes=256,axu_inplanes=512,index=3,mode='add')
        self.boost_5 = BoostBlock(inplanes=128,axu_inplanes=256,index=4)
        self.conv2d_1 = ConvBNR(512,512,3)
        self.conv2d_2 = ConvBNR(256,256,3)
        self.predict1 = Conv1x1(128,1)
        self.predict2 = Conv1x1(128,1)
        self.predict3 = Conv1x1(128,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        features=self.backbone(x)
        x1, x2, x3, x4 = features
        edge,mainflow = self.boost_1(x4,x1)#mainflow: B*1024*12*12
        mainflow_4 = self.boost_2(mainflow,x4) #B*1024*12*12
        mainflow_3 = self.boost_3(x3,mainflow_4) #B*512*24*24
        mainflow_4 = self.reduce3(mainflow_4)
        # att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        mainflow_4 = F.interpolate(mainflow_4, mainflow_3.size()[2:], mode='bilinear', align_corners=False)#upsample
        mainflow_3 = mainflow_3 +  mainflow_4
        mainflow_3 = self.conv2d_1(mainflow_3)
        mainflow_3 = self.sigmoid(mainflow_3)
        mainflow_2 = self.boost_4(x2,mainflow_3 ) #B*256*48*48
        mainflow_3 = self.reduce2(mainflow_3)
        mainflow_3 = F.interpolate(mainflow_3, mainflow_2.size()[2:], mode='bilinear', align_corners=False)  # upsampl
        mainflow_2 = mainflow_2 + mainflow_3
        mainflow_2 = self.conv2d_2(mainflow_2)
        mainflow_2 = self.sigmoid(mainflow_2)
        pred3, pred1, pred2 = self.boost_5(x1,mainflow_2)
        pred1 = self.predict1(pred1)
        pred1 = F.interpolate(pred1, scale_factor=4, mode='bilinear', align_corners=False)
        pred2 = self.predict2(pred2)
        pred2 = F.interpolate(pred2, scale_factor=4, mode='bilinear', align_corners=False)
        pred3 = self.predict3(pred3)
        pred3 = F.interpolate(pred3, scale_factor=4, mode='bilinear', align_corners=False)
        edge = F.interpolate(edge, scale_factor=4, mode='bilinear', align_corners=False)
        return pred1,pred2,pred3,edge




if __name__ == '__main__':
    image = torch.randn(1,3,384,384)
    net = Net()
    out = net(image)
    print('NET test:',out[0].shape)
    # boost = BoostBlock(inplanes=512,axu_inplanes= 2048,index=2,mode='cat',num=4)
    # x4 = torch.randn((2,2048,12,12))
    # x1 = torch.randn((2,512,24,24))
    # out = boost(x1,x4)
    # print('BoostBlock test:',out[0].shape)
