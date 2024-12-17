# -*- coding: utf-8 -*-
import torch
from skimage import measure

#AOLM(fm.detach(), conv5_b.detach())
def AOLM(fms, fm1):
    # 这段代码计算了特征图中哪些区域具有较高的像素值，即哪些区域被认为是感兴趣的。
    #计算输入的特征图 fms 的和，并取均值，然后生成一个二值掩码 M，其形状与输入特征图相同。
    A = torch.sum(fms, dim=1, keepdim=True) #dim=1，表示在通道维度上进行求和。这意味着对每个特征通道上的所有值进行求和，
                                            # 得到一个新的特征图，每个像素值表示原始特征图在对应通道上的值之和。
    a = torch.mean(A, dim=[2, 3], keepdim=True)
    M = (A > a).float()

    A1 = torch.sum(fm1, dim=1, keepdim=True)
    a1 = torch.mean(A1, dim=[2, 3], keepdim=True)
    M1 = (A1 > a1).float() #二进制掩码，其中大于平均值的像素被设置为 1，小于等于平均值的像素被设置为 0


    coordinates = []
    for i, m in enumerate(M):
        mask_np = m.cpu().numpy().reshape(8, 8)
        component_labels = measure.label(mask_np) #使用 measure.label 函数对二进制掩码进行标签化，
                                                # 即将相连的像素点分为一个区域并为每个区域分配唯一的标签。
                                                # 这通常用于标识掩码中的不同目标区域。

        properties = measure.regionprops(component_labels) #使用 measure.regionprops 函数获取标签化区域的属性。
                                                            # 这些属性包括区域的面积、边界框等。
        #存储每个区域的面积，并找到具有最大面积的区域的索引
        areas = []
        for prop in properties:
            areas.append(prop.area)
        max_idx = areas.index(max(areas))


        intersection = ((component_labels==(max_idx+1)).astype(int) + (M1[i][0].cpu().numpy()==1).astype(int)) ==2 #即两个掩码都为1，和才为0，即两个特征图都感兴趣的部分
        prop = measure.regionprops(intersection.astype(int))
        if len(prop) == 0:
            bbox = [0, 0, 8, 8]
            print('there is one img no intersection')
        else:
            bbox = prop[0].bbox


        x_lefttop = bbox[0] * 32 - 1
        y_lefttop = bbox[1] * 32 - 1
        x_rightlow = bbox[2] * 32 - 1
        y_rightlow = bbox[3] * 32 - 1
        # for image
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0
        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        coordinates.append(coordinate)
    return coordinates

