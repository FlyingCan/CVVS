"""
    time:3/18
    author:youcan
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from math import log
from sklearn.cluster import DBSCAN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from net.Res2Net import res2net50_v1b_26w_4s
from net.backbone.get_backbone import get_backbone
class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1,  dilation=1, bias=False):
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
class MEEM(nn.Module):
    def __init__(self):
        """多尺度边缘提取"""
        super(MEEM, self).__init__()
        self.reduce1 = Conv1x1(128, 64)
        self.reduce2 = Conv1x1(256, 64)
        self.reduce3 = Conv1x1(512, 256)
        self.reduce4 = Conv1x1(1024, 256)
        self.pixAtten = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256+64, 8, kernel_size=1, padding=0),
            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))
    def forward(self, s1,s2,s3,s4):
        size1 = s1.size()[2:]
        size2 = s2.size()[2:]
        s1 = self.reduce1(s1)
        s2 = self.reduce2(s2)
        s3 = self.reduce3(s3)
        s4 = self.reduce4(s4)
        s4 = F.interpolate(s4, size1, mode='bilinear', align_corners=True)
        s3 = F.interpolate(s3, size2, mode='bilinear', align_corners=True)
        flow1 = torch.cat((s4, s1), dim=1)
        flow2 = torch.cat((s3, s2), dim=1)
        flow2 = self.pixAtten(flow2)
        flow2 = F.interpolate(flow2, size1, mode='bilinear', align_corners=True)
        out = flow1 *flow2 + flow1
        out = self.block(out)
        return torch.sigmoid(out)
class Catten(nn.Module):
    def __init__(self,channel):
        """通道注意力机制"""
        super(Catten, self).__init__()
        self.conv = ConvBNR(channel,channel)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)# channel wise global attention
        self.linear = nn.Linear(channel,channel)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.conv(x)
        channel_weight = self.avg_pool(x)
        channel_weight = self.linear(channel_weight.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x = x * self.sigmoid(channel_weight)
        return x
class MST(nn.Module):
    def __init__(self,channel):
        """ Multiple -scale transformation 多尺度变换"""
        super(MST, self).__init__()
        self.conv0 = ConvBNR(channel//4, channel//4, 3, 1, 1)
        self.conv1 = ConvBNR(channel//4, channel//4, 3, 1, 3)
        self.conv2 = ConvBNR(channel//4, channel//4, 3 , 1, 5)
        self.conv3 = ConvBNR(channel//4, channel//4, 3 , 1, 7)
        self.conv_1 = Conv1x1(channel, channel)
    def forward(self,x,edge = None):
        """
        :param x: B*C*H*W
        :return: B*C*H*W
        """
        k = 4
        factor = 3
        shape = x.shape[2:]
        xc = torch.chunk(x, k, dim=1)
        xc = self.zoomOut(xc,factor,k)
        if edge != None:
            edge =F.interpolate(edge, size=shape, mode='bilinear', align_corners=True)
            edge = self.zoomOut(edge,factor,k)
            xc = [xc[i] * edge[i] for i in range(k) ]
        xc[0] = self.conv0(xc[0])
        xc[1] = self.conv1(xc[1])
        xc[2] = self.conv2(xc[2])
        xc[3] = self.conv3(xc[3])
        xc = [F.interpolate(xc[i], size=shape, mode='bilinear', align_corners=True) for i in
                  range(k)]
        out = torch.cat(xc,dim=1)
        out = x + self.conv_1(out)
        return out




    def zoomOut(self,xc, scale=2.0, k=4):
        """
        :param xc: feature list
        :param scale: max scale factor
        :return: different feature
        """
        if scale ==1:
            return xc
        step = (scale - 1) / k
        assert step in [0.25, 0.5, 0.75, 1], 'scale factor step should in [0.25,0.5,0.75,1]'
        if isinstance(xc, tuple):
            assert k == len(xc), 'k should == len(xc)'
            xc = [F.interpolate(xc[i], scale_factor=1.25 + i * step, mode='bilinear', align_corners=True) for i in
                  range(k)]
        else:
            xc = [F.interpolate(xc, scale_factor=1.25 + i * step, mode='bilinear', align_corners=True) for i in
                  range(k)]
        return xc


class MSAM(nn.Module):
    def __init__(self, inplanes,outplanes,preplanes):
        super(MSAM, self).__init__()
        self.conv2d1 =ConvBNR(inplanes,inplanes)
        self.mst1 = MST(inplanes)
        self.catten = Catten(inplanes)
        self.reduce = Conv1x1(inplanes,outplanes)
        self.conv1_1 = Conv1x1(outplanes + preplanes, outplanes)
        self.mst2 = MST(outplanes)
        self.conv2d2 =ConvBNR(outplanes,outplanes,3)
    def forward(self, s, rx,edge):
        """
        :param s: swin backbone feature
        :param rx: last boostblock feature
        :param edge: edge
        :return: out
        """
        if s.size() != edge.size():
            edge = F.interpolate(edge, s.size()[2:], mode='bilinear', align_corners=True)
        x = self.conv2d1(s)
        x = self.mst1(x, edge)
        x = self.catten(x)
        x = self.reduce(x)
        if x.size()[2:] != rx.size()[2:]:
            rx = F.interpolate(rx, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x,rx),dim=1)
        x = self.conv1_1(x)
        out = self.mst2(x)
        out = self.conv2d2(out)
        return out

class FEGM(nn.Module):
    def __init__(self, dim=128, num_heads=4, qkv_bias=True, qk_scale=None,patch_height=12,
                 patch_width=12):
        super().__init__()
        t = int(abs((log(dim, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Conv1d(64, 64, kernel_size=k, padding=(k - 1) // 2, bias=False),#16*64*(18432)
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.patch_embedding_re =  nn.Sequential(
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                      p1=patch_height, p2=patch_width, h=8, w=8),
            nn.AdaptiveAvgPool2d(32)
        )
        self.pool = nn.AdaptiveAvgPool2d(32)
        self.x_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.y_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.conv_1d = nn.Conv1d(128, 128, kernel_size=k, padding=(k - 1) // 2, bias=False)  # 16*64*(18432)
        self.lnx = nn.LayerNorm(128)
        self.lny = nn.LayerNorm(128)
    def forward(self, s1, o, index:list):
        """
        Args:
            s1:patch_pred  16*128*16*16
            o:patch in backbone  16*128*16*16

        Returns:
            patch: B*C*16*16
            atten: B*C*256*256
        """
        short = self.mask(s1,index) #32 32
        batch_size = s1.shape[0]
        shape = s1.shape[2:]
        chanel = s1.shape[1]
        o = self.pool(o)  # 32 32
        o = o.view(batch_size, chanel, -1).permute(0, 2, 1)
        o = self.lnx(o)
        s1 = short.view(batch_size, chanel, -1).permute(0, 2, 1)  # 16*1024*128
        ori = s1
        s1 = self.lny(s1)


        B, N, C = s1.shape
        x_qkv = self.x_qkv(o).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        y_v = self.y_v(s1).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_q, x_k, x_v = x_qkv[0], x_qkv[1], x_qkv[2]
        y_v = y_v[0]
        attn_x = (x_q @ x_k.transpose(-2, -1)) * self.scale
        attn_x = attn_x.softmax(dim=-1)
        y_v = attn_x @ y_v
        x_v = x_v + y_v
        x_v = x_v.transpose(1, 2).reshape(B, N, C)
        x_v = self.proj(x_v)
        x_v = x_v + ori
        x_v = x_v.permute(0, 2, 1)
        x_v = x_v + self.conv_1d(x_v)
        x_v = x_v.view(batch_size, chanel, *short.size()[2:])

        x_v = F.interpolate(x_v, size=shape, mode='bilinear', align_corners=True)
        return x_v
    def mask(self, s1, center_index):
        o_flatten = self.to_patch_embedding(s1)   ##16*64*(18432)
        mask = o_flatten.clone()
        for x,y in center_index:
            index = x//12 * 8 + y//12
            mask[:,index,:] = 1
        o_flatten = o_flatten * mask
        return self.patch_embedding_re(o_flatten)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.swin=get_backbone()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.eam = MEEM()
        self.mst = MST(1024)
        self.high_atten = Catten(1024)
        self.reduce = Conv1x1(1024,256)
        self.boost1 = MSAM(inplanes=512, outplanes=256, preplanes=256)
        self.boost2 = MSAM(inplanes=256, outplanes=128, preplanes=256)
        self.boost3 = MSAM(inplanes=128, outplanes=128, preplanes=128)
        self.boost4 = MSAM(inplanes=256, outplanes=64, preplanes=128)
        self.fegm = FEGM()
        self.predict3 = nn.Conv2d(256, 1, 1)
        self.predict2 = nn.Conv2d(128, 1, 1)
        self.predict1 = nn.Conv2d(128, 1, 1)
        self.predict = nn.Conv2d(128, 1, 1)
        self.predict0 = nn.Conv2d(64, 1, 1)
        self.conv = ConvBNR(256,128,3)
    def forward(self,x):

        s1, s2, s3, s4 = self.swin(x)
        x1,_,_,_ = self.resnet(x)
        edge = self.eam(s1,s2,s3,s4)
        feature = self.mst(s4,edge)
        feature = self.high_atten(feature)
        feature = self.reduce(feature)
        o3 = self.boost1(s3, feature,edge)
        o2 = self.boost2(s2,o3,edge)
        o1 = self.boost3(s1,o2,edge)
        _, _, Batch_patch_index = find_uncertainy(edge, o1, s1)
        short = self.fegm(s1,o1,Batch_patch_index)
        short = torch.cat((short,o1),dim=1)
        o1 = self.conv(short) + o1
        o0 = self.boost4(x1,o1,edge)
        o3 = F.interpolate(self.predict3(o3), scale_factor=16, mode='bilinear', align_corners=True)
        o2 = F.interpolate(self.predict2(o2), scale_factor=8, mode='bilinear', align_corners=True)
        o1 = F.interpolate(self.predict1(o1), scale_factor=4, mode='bilinear', align_corners=True)
        o0 = F.interpolate(self.predict0(o0), scale_factor=4, mode='bilinear', align_corners=True)
        edge = F.interpolate(edge, scale_factor=4, mode='bilinear', align_corners=True)
        return o3,o2,o1,o0,edge

def find_uncertainy(edge,gt,backbone):
    """
    :param edge: 16*C*H*W
    :param gt: 16*C*H*W
    :return: index
    """
    assert gt.shape == backbone.shape
    radium_ =12
    Batch_patch_index = []
    k = 1
    r = k * radium_
    # Batch_uncertain_patch = torch.zeros((gt.shape[0], 3, gt.shape[1], 2 * r, 2 * r))
    # backbone_uncertain_patch = torch.zeros((backbone.shape[0], 3, backbone.shape[1], 2 * r, 2 * r))
    Batch_uncertain_patch = torch.zeros((gt.shape[0],  gt.shape[1], 2 * r, 2 * r)).to(device)
    backbone_uncertain_patch = torch.zeros((backbone.shape[0],  backbone.shape[1], 2 * r, 2 * r)).to(device)
    for i in range(edge.shape[0]):
        indexs = dbscan_find_uncetainty(edge[i], radium_=radium_)
        Batch_patch_index.append(indexs[0])  # 找到每个Batch的不确定位置
        for j in range(len(indexs)):#实际只循环一次 不愿改了
            index = indexs[j]
            Batch_uncertain_patch[i,:, :, :] = gt[i, :, index[0] - r: index[0] + r,
                                                   index[1] - r: index[1] + r]
            backbone_uncertain_patch[i, :, :, :] = backbone[i, :, index[0] - r: index[0] + r,
                                                   index[1] - r: index[1] + r]
    #print(Batch_uncertain_patch.shape)
    # print(Batch_uncertain_patch.shape)
    # print(backbone_uncertain_patch.shape)
    # print(Batch_patch_index)
    return  Batch_uncertain_patch,backbone_uncertain_patch,Batch_patch_index
def dbscan_find_uncetainty(res,center_radium=6,radium_=6):
    """
    :param res: image_tensor
    :return:
    """
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    res = (res * 255).astype(np.uint8)  #转化为np类型
    uncertainty_threshold = 180  # 超参数 不确定点
    index_ = np.where(res > uncertainty_threshold)  #找到确定点的坐标
    X = list(zip(index_[0], index_[1]))  #转化为二维坐标
    uncertain_Points_num = 15  #超参数 不确定点个数
    db = DBSCAN(eps=center_radium, min_samples=uncertain_Points_num).fit(X)
    labels = db.labels_
    num_list = []
    for label in set(labels):
        if label != -1:
            num = (labels == label).sum()
            num_list.append((label,num)) #保存对应的
    num_list.sort(key= lambda x:x[1])
    patch_center_index = []
    X = np.array([list(e) for e in X])
    for label,_ in num_list:
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        patch_center_index.append( [int(e) for e in np.array(X[(labels==label) & core_samples_mask]).mean(0)] )
    shape = res.shape
    patch_center_index = [confim(index,shape,radium_) for index in patch_center_index]
    # center= confim(np.array(patch_center_index[0:3]).mean(0),shape,radium_*3)
    # print(patch_center_index[0:3])
    if(len(patch_center_index) ==1):
        patch_center_index.append(patch_center_index[0])
        patch_center_index.append(patch_center_index[0])
    if(len(patch_center_index) == 2):
        patch_center_index.append(
            confim([patch_center_index[0][0]+np.random.randint(-radium_,radium_),
                    patch_center_index[0][1]+np.random.randint(-radium_,radium_)],shape,radium_)
        )
    if (len(patch_center_index) ==0):
        patch_center_index = [
            confim([np.random.randint(0,96),np.random.randint(0,96)]) for i in range(3)
        ]
    assert  len(patch_center_index) >=3,'at least 3 patch'
    return patch_center_index[0:3]
def confim(index,shape=(96,96),radium=12):
    """

    :param index: 判断位置合法性
    :param shape: 原始图片大小
    :param radium:  范围
    :return: 合法化的位置
    """
    assert len(index) ==2,'the dim of index should be 2'
    index = [int(index[0]),int(index[1])]
    assert index[0] < shape[0] or index[1] < shape[1],"index wrong"
    index[0] = radium if index[0] < radium else index[0]
    index[0] = shape[0]-radium if index[0] > shape[0] - radium else index[0]

    index[1] = radium if index[1] < radium else index[1]
    index[1] = shape[1]-radium if index[1] > shape[1] - radium else index[1]
    return index
def flat(mask):
    batch_size = mask.shape[0]
    h = 32
    mask = F.interpolate(mask, size=(int(h), int(h)), mode='bilinear')
    x = mask.view(batch_size, 1, -1).permute(0, 2, 1)
    # print(x.shape)  b 28*28 1
    g = x @ x.transpose(-2, -1)  # b 24*24 24*24
    g = g.unsqueeze(1)  # b 1 24*24 24*24
    return g
def att_loss(pred, mask, p4, p5 ):

    g = flat(mask)
    np4 = torch.sigmoid(p4.detach())
    np5 = torch.sigmoid(p5.detach())
    p4 = flat(np4)
    p5 = flat(np5)
    w1 = torch.abs(g - p4)
    w2 = torch.abs(g - p5)
    w = (w1 + w2) * 0.5 + 1
    attbce = F.binary_cross_entropy_with_logits(pred, g, weight=w * 1.0, reduction='mean')
    return attbce
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()
if __name__ == '__main__':
    image = torch.randn(2,1,384,384)
    target = torch.randn(2, 1, 384, 384)
    net = Net()

    out = net(image)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # for i in range(100):
    #     optimizer.zero_grad()
    #     output = net(image)
    #     loss = att_loss(output[-1],target,output[-2],output[-2])
    #     loss.backward()
    #     optimizer.step()
    #     print(loss.item())
    print(out[4].shape)
