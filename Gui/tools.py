import os

import torch
from PyQt5 import QtGui
from torchvision.transforms import transforms

import torch.nn.functional as F
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt, colors
from timm.models.layers import to_2tuple
np.random.seed(2023)
class To_tensor:
    """load """

    def __init__(self, testsize=384):
        self.testsize = testsize
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

    def load_data(self, path):
        image = self.rgb_loader(path)
        image = self.transform(image).unsqueeze(0)
        return image

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __call__(self, path):
        return self.load_data(path)


def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def to_Qt_img(tensor, shape):
    """

    :param tensor: tensor图片
    :param shape: 原始图片形状
    :return: QImage对象
    """
    res = F.upsample(tensor, size=shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    res = (res * 255).astype(np.uint8)
    img = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    img = QtGui.QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3, QtGui.QImage.Format_RGB888)
    return img

def tensor_to_ndarray(tensor:torch.Tensor,shape=(384,384)):
    assert tensor.ndim == 4 and tensor.shape[0]==1 and tensor.shape[1] ==1,'tensor must be (1*1,H,W)'
    res = F.upsample(tensor, size=shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    res = (res * 255).astype(np.uint8)
    return res
def rgb_to_Qtimg(rgb_img:np.ndarray):
    """

    :param rgb_img:rgb(h,w,3),np
    :param shape:
    :return:
    """
    assert isinstance(rgb_img,np.ndarray),'type wrong'
    img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    img = QtGui.QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3, QtGui.QImage.Format_RGB888)

    return img



def split_RGBpic(file,color='black',shape=(24,24),show=True):
    """
    split the RGB picture and get the similarity of the patch and note the max difference
    :param file:文件路劲
    :param color: 分割线颜色
    :param show: true 为显示
    :return: result(np rgb (h,w,3)) ,  patch vector list [v1,v2,...vn] v1:np.array dim=2
    """
    print(color,end=' ')
    color = np.array(get_stride_color_rgb(color))
    print(color)
    img = Image.open(file)
    img = img.resize((384, 384))
    img = np.array(img)
    patchSize = np.array(shape)
    patchNum = (int(img.shape[0] / patchSize[0]), int(img.shape[1] / patchSize[1]))  # 16 * 16 个patch
    stride_Num = (patchNum[0] - 1, patchNum[1] - 1)  # 384/16-1  (15,15) 条间隔
    patchStride = (patchSize / 8).astype(np.int32)

    patch_vector_list = []  # 每一个元素为 24*24*3 的特征向量
    result = np.zeros(
        (img.shape[0] + stride_Num[0] * patchStride[0], img.shape[0] + stride_Num[1] * patchStride[1], 3)).astype(np.uint8)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i,j] = 255*color
    for i in range(0, patchNum[0]):
        for j in range(0, patchNum[1]):
            patch = img[i * patchSize[0]:(i + 1) * patchSize[0], j * patchSize[1]:(j + 1) * patchSize[1]]
            result[i * (patchSize[0] + patchStride[0]):i * (patchSize[0] + patchStride[0]) + patchSize[0],
            j * (patchSize[1] + patchStride[1]):j * (patchSize[1] + patchStride[1]) + patchSize[1]] = patch
            patch_vector_list.append(patch.reshape((1, -1)))  # 256 个向量
    result = result.astype(np.uint8)
    ###处理相似度  相似度最大最小的分别标出来
    similarity,max_index,min_index=cla_cossine_similarity_mean(vector_list=patch_vector_list, )


    if show:
        plt.imshow(result)
        plt.xticks([])
        plt.yticks([])
        plt.show()
    return result,similarity




def get_stride_color_rgb(color):
    """
    :param color:str 'black
    :return: color rgb (0,0,0)
    """
    _stride_color=color
    color_rgb = colors.hex2color(colors.cnames[_stride_color])
    r,g,b=color_rgb
    color_rgb=(b,g,r)
    return color_rgb
def cla_cossine_similarity(vector_list):
    """
    :param vector_list: [array(v1,v2,...vn)]
    :return:similarity multiple of vector, max_different_index, min_different_index
    """
    cosine_list = []
    v_average = np.mean(vector_list, 0)
    for v in vector_list:
        if v.sum() == 0:
            v = np.random.uniform(0,1,v.shape)
        similarity = cosine_similarity(v, v_average)
        cosine_list.append(similarity.squeeze())
    similarity = 1
    cosine_list = np.array(cosine_list)
    for e in list(cosine_list):
        similarity = similarity * e
    max_index, min_index = cosine_list.argmax(), cosine_list.argmin()
    return similarity,max_index,min_index

def cla_cossine_similarity_mean(vector_list):
    """
    :param vector_list: array(v1,v2,...vn)
    :return:similarity multiple of vector, max_different_index, min_different_index
    """
    cosine_list = []
    v_average = np.mean(vector_list, 0)
    for v in vector_list:
        if v.sum() == 0:
            v = 100*np.random.randn(v.shape[0],v.shape[1])
        similarity = cosine_similarity(v, v_average)
        cosine_list.append(similarity.squeeze())

    cosine_list = np.array(cosine_list)
    similarity = cosine_list.mean()
    max_index, min_index = cosine_list.argmax(), cosine_list.argmin()
    return similarity,max_index,min_index
def local_mae(patch_list,patch_size,kernel_size,img_size):
    """

    :param patch_list: [32*32*patch]
    :param patch_size:
    :param kernel_size:
    :param img_size:
    :return:
    """
    kernel_size = to_2tuple(kernel_size)
    assert patch_size[0] == patch_size[1] and img_size[0] == img_size[1], 'size wrong!'
    stride_num_row = img_size[0] // patch_size[0] -kernel_size[0]+1#32-3+1
    stride_num_col = img_size[1] // patch_size[1] - kernel_size[1] + 1
    mae_list=[]
    for row in range(stride_num_row):#沿着列
        for col in range(stride_num_col):
            window = []
            for i in range(kernel_size[0]*kernel_size[1]):
                window.append(patch_list[row+i//kernel_size[0]][col+i%kernel_size[1]])
            window =np.array(window) #window patch  9*(patchsize*patchsize)
            window_ave = window.mean(0)
            mae_list.append(((window-window_ave)**2).mean())
    return sum(mae_list)**0.5

def local_cos(patch_list, patch_size, kernel_size, img_size):
    """

    :param patch_list: [32*32*patch]
    :param patch_size:
    :param kernel_size:
    :param img_size:
    :return:
    """
    assert patch_size[0] == patch_size[1] and img_size[0] == img_size[1],'size wrong!'
    zero_ = np.random.uniform(0, 0.01, (1,patch_size[1]*patch_size[0]*3))
    kernel_size = to_2tuple(kernel_size)
    stride_num_row = img_size[0] // patch_size[0] -kernel_size[0]+1#32-3+1
    stride_num_col = img_size[1] // patch_size[1] - kernel_size[1] + 1
    local_list = []
    for row in range(stride_num_row):  # 沿着列
        for col in range(stride_num_col):
            window = []
            cos_list = []
            for i in range(kernel_size[0]*kernel_size[1]):
                window.append(patch_list[row + i // kernel_size[0]][col + i % kernel_size[1]].reshape(1,-1))
            window = np.array(window)
            window_ave = window.mean(0) if window.sum() != 0 else zero_
            for v in window:
                if v.sum() == 0:
                    v = zero_
                similarity = cosine_similarity(v, window_ave)
                cos_list.append(similarity.squeeze())
            cos_list = np.array(cos_list)
            similarity = cos_list.mean()
            local_list.append(similarity)
    return sum(local_list)
def split_RGBpic_cal_local_loss(file,color='black',shape=(24,24),show=True):
    """
    split the RGB picture and get the similarity of the patch and note the max difference
    :param file:文件路劲
    :param color: 分割线颜色
    :param show: true 为显示
    :return: result(np rgb (h,w,3)) ,  patch vector list [v1,v2,...vn] v1:np.array dim=2
    """
    print(color,end=' ')
    color = np.array(get_stride_color_rgb(color))
    print(color)
    img = Image.open(file)
    img = img.resize((384, 384))
    img = np.array(img)
    patchSize = np.array(shape)
    patchNum = (int(img.shape[0] / patchSize[0]), int(img.shape[1] / patchSize[1]))  # 16 * 16 个patch
    stride_Num = (patchNum[0] - 1, patchNum[1] - 1)  # 384/16-1  (15,15) 条间隔
    patchStride = (patchSize / 8).astype(np.int32)


    result = np.zeros(
        (img.shape[0] + stride_Num[0] * patchStride[0], img.shape[0] + stride_Num[1] * patchStride[1], 3)).astype(np.uint8)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i,j] = 255*color
    patch_list = [[]]*patchNum[0]
    for i in range(0, patchNum[0]):
        for j in range(0, patchNum[1]):
            patch = img[i * patchSize[0]:(i + 1) * patchSize[0], j * patchSize[1]:(j + 1) * patchSize[1]]
            result[i * (patchSize[0] + patchStride[0]):i * (patchSize[0] + patchStride[0]) + patchSize[0],
            j * (patchSize[1] + patchStride[1]):j * (patchSize[1] + patchStride[1]) + patchSize[1]] = patch
            patch_list[i].append(patch)
    result = result.astype(np.uint8)
    ###处理局部相似度

    local_cos_value = local_cos(patch_list=patch_list,patch_size=patchSize,kernel_size=3,img_size=img.shape )
    local_mae_value = local_mae(patch_list=patch_list,patch_size=patchSize,kernel_size=(3,3),img_size=img.shape )

    if show:
        plt.imshow(result)
        plt.xticks([])
        plt.yticks([])
        plt.show()
    return result,local_cos_value,local_mae_value
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
def conv(img_path, edge_path, show=False):
    """
    根据边缘和 原图片进行rgb比较
    :param img_path: 图片地址
    :param edge_path: 边缘地址
    :param show: 是否展示
    :return:
    """
    image = cv2.imread(img_path, 1)
    image_edge = cv2.imread(edge_path, -1)
    Conv = np.array([-3, -2, -1, 0, 1, 2, 3])
    Conv1_7 = Conv[:, np.newaxis]
    padding = Conv1_7.shape[0] // 2
    assert image.ndim == 3, 'image.dim should be 3'
    image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)),
                   'constant', constant_values=0)  # 对图片进行填充 防止越界
    if image_edge.ndim == 3:
        image_edge = cv2.cvtColor(image_edge, cv2.COLOR_BGR2GRAY)
    image_edge = np.pad(image_edge, ((padding, padding), (padding, padding)),
                        'constant', constant_values=0)  # 对图片进行填充 防止越界
    X, Y = np.where(image_edge != 0)  # 得到原始边界坐标
    index = list(zip(X, Y))

    # image_pad = np.pad(image,((50,50),(50,50),(0,0)),'constant',constant_values=255)
    # tools.show_cv(image_pad)  #填充图片
    if show:
        cv2.imshow('image', image)
        cv2.waitKey(0)
    diff_x = []
    diff_y = []
    for x, y in index:
        left = y - 3
        right = y + 3
        top = x + 3
        down = x - 3
        diff_17 = image[x, left:right + 1] * Conv1_7
        diff_17 = np.abs(diff_17.sum(0)).sum()
        diff_x.append(diff_17)
        diff_71 = image[down:top + 1, y] * Conv1_7
        diff_71 = np.abs(diff_71.sum(0)).sum()
        diff_y.append(diff_71)
    value1 = np.mean(diff_x)
    value2 = np.mean(diff_y)
    return value1 / 100, value2 / 100
def merge(image_list,m,n,show = False):
    """
    还没有完善 对 m*1的会报错
    将二维列表中的元素合并
    :param image_list: np 图像列表 二维
    :param m: 合并m行 1....m
    :param n:  合并n列 1....n
    :return: 合并后的图
    """
    pading_W = 3
    pading_H = 4
    width = image_list[0][0].shape[1]
    width = n * width + pading_W * (n-1) + 1
    res_pic = 255 * np.ones((1,width,3),np.uint8)
    for i in range(m):
        image_line = image_list[i][0]
        shape = image_line.shape
        H, W = shape[0:-1]
        padding_height = 255 * np.ones((H, pading_W, 3), np.uint8)
        image_line = 255 * np.ones((H, 1, 3), np.uint8)
        for j in range(0,n-1):#最后一张图片左边不需要分割线
            image = image_list[i][j]
            assert shape[0] == image.shape[0],"merge wrong! all the pic should have the same height"
            image_line = np.concatenate([image_line,image,padding_height], axis=1)
        image_line = np.concatenate([image_line, image_list[i][n-1]], axis=1)
        W = image_line.shape[1]
        padding_width = 255 * np.ones((pading_H,W,3),np.uint8)
        if i ==0:
            res_pic = np.concatenate([res_pic,image_line], axis=0)
        else:
            res_pic = np.concatenate([res_pic,padding_width,image_line],axis=0)
    if show:
        cv2.imshow('image', res_pic)
        cv2.waitKey(0)
    return res_pic
def generate_colorMap(org_img,gray_img):
    """
    :param image: 图片
    :param mash: 掩码图片
    :return:
    """
    heat_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)  # 此处的三通道热力图是cv2使用GBR排列
    add_img = cv2.addWeighted(org_img, 0.3, heat_img, 0.7, 0)
    # 五个参数分别为 图像1 图像1透明度(权重) 图像2 图像2透明度(权重) 叠加后图像亮度
    return heat_img,add_img


