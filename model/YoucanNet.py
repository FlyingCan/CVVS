###1
"""
 @Time    : 2023/1/10 21:08
 @Author  : Youcan Xv
 @E-mail  : 2682615572@mail.dlut.edu.cn

 @Project : YoucanCVPR
 @File    : EDM.py
 @Function: Edge detection module
"""
from math import log
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


from net.backbone.get_backbone import get_backbone
from net.utils.LocalAttention import LocalAtten
from net.utils.ConvBlock import ConvBNR


class EDM(nn.Module):
    """
    in: B*inplanes*H*W
    out: B*1*H*W
    """
    def __init__(self,inplanes):
        super(EDM, self).__init__()
        planes = inplanes//4
        self.reduce1 = ConvBNR(inplanes, planes,1)#12,3
        self.reduce4 = ConvBNR(inplanes, planes,1)#12,3
        self.conv_3_1 = ConvBNR(2*planes, inplanes,kernel_size=(3,1))
        self.conv_1_3 = ConvBNR(2*planes, inplanes, kernel_size=(1, 3))
        self.block = nn.Sequential(
            ConvBNR(inplanes, 256, 3),
            nn.Conv2d(256, 1, 1)
        )

    def forward(self, x1, x2):
        """

        :param x1:
        :param x2:
        :return: out : x1.shape
        """
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x2 = self.reduce4(x2)
        x2 = F.interpolate(x2, size, mode='bilinear', align_corners=False)
        merge = torch.cat((x2, x1), dim=1)
        edge1 = self.conv_3_1(merge)
        edge2 = self.conv_1_3(merge)
        out = edge1 + edge2
        out = self.block(out)

        return out

class LDM(nn.Module):
    """
    in: B*channel*H*W
    out: B*channel*H*W
    """
    def __init__(self, channel,trainSize=384,blockNum=0):
        super(LDM, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.localAtten = LocalAtten(inplanes=channel,img_size=trainSize,num_index=blockNum)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)# channel wise  attention
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.beta = nn.Parameter(torch.ones(1))
    def forward(self, x1, edge):
        if x1.size() != edge.size():
            edge = F.interpolate(edge, x1.size()[2:], mode='bilinear', align_corners=False)
        x = x1  + edge
        x = self.localAtten(x)
        out = self.avg_pool(x)
        out = self.conv1d(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        x = self.beta*out + x

        return x


class GDM(nn.Module):
    """
    in: B*channel*H*W
    out: B*channel*H*W
    """
    def __init__(self,inplanes,chunk_channel=4):
        super(GDM, self).__init__()
        planes = 128
        assert inplanes % chunk_channel == 0, "channel size wrong!"
        self.chunkNum = 2*planes // chunk_channel
        self.reduce1 = ConvBNR(inplanes, planes, 1)  #
        self.reduce2 = ConvBNR(inplanes, planes, 1)  # 12,3
        self.conv_chunk = ConvBNR(256,256,3)
        self.weight = nn.Parameter(torch.randn(self.chunkNum,self.chunkNum))
        self.conv = ConvBNR(256,inplanes,3)
        self.conv_1 = ConvBNR(256,256,1)
    def forward(self,x1,x2):
        x1 = self.reduce1(x1)
        x2 = self.reduce2(x2)
        if x1.size()[2:] != x2.size()[2:]:
            x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x1, x2), dim=1)
        x_chunk = list( x.chunk(self.chunkNum,1) )
        for beta_i in range(self.chunkNum):
            for gama_j in range(self.chunkNum):
                x_chunk[beta_i] = x_chunk[beta_i] + x_chunk[gama_j] *self.weight[beta_i][gama_j]
        x1 = torch.cat(x_chunk[:],dim=1)
        x1 = self.conv_chunk(x1)
        x = x1 + x
        out = self.conv_1(x)
        out = self.conv(out)

        return out
class BoostBlock(nn.Module):
    """
    WiseMerge:Branchformer inference
    in: B*inplanes*H*W
    out: B*planes*H*W
    """

    def __init__(self,inplanes=3,planes=128,blockNum=0,merge_method='learned_ave',
                 trainSize=96,chunk_channel =4, attn_branch_drop_rate=0.2,lastBlock=False):
        super(BoostBlock, self).__init__()
        assert trainSize == 96,"trainSize wrong"
        self.training = True
        self.attn_branch_drop_rate = attn_branch_drop_rate
        self.merge_method = merge_method
        self.lastBlock = lastBlock
        # self.channel_wise1 = ConvBNR(inplanes,planes,3)
        # self.channel_wise2 = nn.Identity() if blockNum<4 else ConvBNR(inplanes,planes)
        self.conv_edge = ConvBNR(1,planes,1)
        self.edm = EDM(planes)
        self.ldm = LDM(planes,trainSize,blockNum)
        self.gdm = GDM(planes,chunk_channel=chunk_channel)
        #have three Branch
        if lastBlock:
            self.conv_to1 = ConvBNR(planes,1,1)
        else:
            if merge_method == "concat":
                self.conv_concat = ConvBNR(planes*3,planes)
            elif merge_method == "learned_ave":
                # attention-based pooling for three branches
                self.pooling_proj1 = nn.Sequential(ConvBNR(planes,1),
                                                   nn.Flatten(-2,-1)) #out B*1*(H*W)
                self.pooling_proj2 = nn.Sequential(ConvBNR(planes, 1),
                                                   nn.Flatten(-2, -1))
                self.pooling_proj3 = nn.Sequential(ConvBNR(planes, 1),
                                                   nn.Flatten(-2, -1))
                # linear projections for calculating merging weights
                self.weight_proj1 = torch.nn.Linear(planes , 1)
                self.weight_proj2 = torch.nn.Linear(planes , 1)
                self.weight_proj3 = torch.nn.Linear(planes , 1)
                self.conv_learnedave = ConvBNR(1,1,1)

    def forward(self,x1,x2):
        """
        :param x1:BoostBlock feature
        :param x2:BackBone feature
        :return:next BoostBlock feature
        """

        # x1 = self.channel_wise1(x1)
        # x2 = self.channel_wise2(x2)
        edge = self.edm(x1,x2) #B*1*H*W
        edge_attn = self.conv_edge(edge) #B*planes*H*W

        x_local = self.ldm(x1,edge_attn)
        x_global = self.gdm(x1,x_local)
        if self.lastBlock:
            predict1 = self.conv_to1(x_local)
            predict2 = self.conv_to1(x_global)
            return edge, predict1, predict2
        if self.merge_method =='concat':
            out = torch.cat((edge_attn,x_local,x_global),dim=1)
            out = self.conv_concat(out)
            return out
        elif self.merge_method == "learned_ave":
            score1, score2, score3 = self.learned_ave(edge_attn,x_local,x_global)
            out = score1*edge_attn + score2*x_local + score3*x_global
            return out



    def learned_ave(self,x1,x2,x3):
        """

        :param x1: feature1 B*C*H*W
        :param x2: feature2 B*C*H*W
        :param x2: feature3 B*C*H*W
        :return:score1,score2,score3
        """
        B, C, H, W = x1.shape
        x1_flatten = x1.view(B, C, -1)
        x2_flatten = x2.view(B, C, -1)
        x3_flatten = x3.view(B, C, -1)
        if (
                self.training
                and self.attn_branch_drop_rate > 0
                and torch.rand(1).item() < self.attn_branch_drop_rate
        ):
            # Drop the other two branch randomly
            a = [0.0,0.0,1.0]
            w1 = a.pop(random.randint(0,len(a)-1))
            w2 = a.pop(random.randint(0,len(a)-1))
            w3 = a.pop(random.randint(0,len(a)-1))
        else:
            # branch1
            score1 = (
                    self.pooling_proj1(x1) / x1_flatten.shape[-1] ** 0.5
            )  # (batch, 1, time)
            score1 = torch.softmax(score1, dim=-1)
            pooled1 = torch.matmul(score1, x1_flatten.transpose(1,2)).squeeze(1)  # (batch, size)
            weight1 = self.weight_proj1(pooled1)  # (batch, 1)

            # branch2
            score2 = (
                    self.pooling_proj2(x2) / x2_flatten.shape[-1] ** 0.5
            )  # (batch, 1, time)

            score2 = torch.softmax(score2, dim=-1)
            pooled2 = torch.matmul(score2, x2_flatten.transpose(1, 2)).squeeze(1)  # (batch, size)
            weight2 = self.weight_proj2(pooled2)  # (batch, 1)

            # branch3
            score3 = (
                    self.pooling_proj2(x3) / x3_flatten.shape[-1] ** 0.5
            )  # (batch, 1, time)

            score3 = torch.softmax(score3, dim=-1)
            pooled3 = torch.matmul(score3, x3_flatten.transpose(1, 2)).squeeze(1)  # (batch, size)
            weight3 = self.weight_proj3(pooled3)  # (batch, 1)
            # normalize weights of three branches
            merge_weights = torch.softmax(
                torch.cat([weight1, weight2, weight3], dim=-1), dim=-1
            )  # (batch, 3)
            merge_weights = merge_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1
            )  # (batch, 3, 1, 1, 1)
            w1, w2, w3 = merge_weights[:, 0], merge_weights[:, 1], merge_weights[:, 2]  # (batch, 1, 1, 1)
        return w1, w2, w3

class Net(nn.Module):
    channel_list = [128,256,512,1024]
    def __init__(self,inplanes=3,trainSize=384,Blocklayers=7,
                 chunk_channel =4, attn_branch_drop_rate=0.2,
                 merge_method='learned_ave',training =True,backbonepath =None):
        super(Net, self).__init__()
        self.training =training
        self.blockLayers = Blocklayers
        self.backbone = get_backbone(backbonepath)
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Sequential(
            ConvBNR(inplanes,128,3),
            nn.AdaptiveAvgPool2d(96)
        ))
        for i_layer in range(1, self.blockLayers):
            self.conv_layers.append(ConvBNR(
                inplanes=self.channel_list[(i_layer - 1) % 4],
                planes=self.channel_list[i_layer % 4],
                kernel_size=3
            ))


        # build layers
        self.layers = nn.ModuleList()  #
        self.layers.append(
            BoostBlock(inplanes=128,planes=128,trainSize = trainSize//4,blockNum=0)    #
        )
        for i_layer in range(1,self.blockLayers):
            layer = BoostBlock(
                inplanes = self.channel_list[i_layer % 4],
                planes = self.channel_list[i_layer%4],
                blockNum = i_layer,
                merge_method = merge_method,
                trainSize = trainSize//4,
                chunk_channel = chunk_channel,
                attn_branch_drop_rate  = attn_branch_drop_rate,
                lastBlock = (i_layer == Blocklayers-1)
            )
            self.layers.append(layer)

    def forward(self,x):
        features = self.backbone(x) #[a1,a2,a3,a4]
        for i in range(0,self.blockLayers-1):
            layer = self.layers[i]
            conv_layer = self.conv_layers[i] #output i_layer channel
            x = conv_layer(x)
            if i <4:
                x = x + layer(x,features[i])
            else:
                x = x + layer(x,x)

        x = self.conv_layers[self.blockLayers-1](x)
        edge, predict1, predict2 = self.layers[self.blockLayers-1](x,x)
        edge = F.interpolate(edge, scale_factor=4, mode='bilinear', align_corners=False)
        predict1 = F.interpolate(predict1, scale_factor=4, mode='bilinear', align_corners=False)
        predict2 = F.interpolate(predict2, scale_factor=4, mode='bilinear', align_corners=False)
        return edge, predict1,predict2

if __name__ =='__main__':
    image = torch.randn((2,3,96,96))
    net = Net(inplanes=3,trainSize=96,Blocklayers=6)
    edge, predict1, predict2 = net(image)
    print(edge.shape,predict1.shape,predict2.shape)












