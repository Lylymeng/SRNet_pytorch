import torch
from skimage import measure
from MTL_MMAL_config import num_classes
import numpy as np
def cam(fc_weights,target_layer,logit):

    h_x = torch.nn.functional.softmax(logit, dim=1).data.squeeze()  # 每个类别对应概率 torch.Size([4, 2]) [[-0.3638,  0.2479],[-0.3971, -0.1304],[-0.4771,  0.1524],[-0.1847,  0.4594]]
    #h_x_transposed = torch.transpose(h_x, 0, 1)
    if h_x.dim() == 1:
        h_x=h_x.unsqueeze(0)
    probs, idx = h_x.sort(1, True)  # 输出概率升序排列 #hx: tensor([0.1640, 0.8360], device='cuda:0')

    #probs = probs.cpu().numpy()
    idx = idx.cpu().numpy() #[[1 0], [1 0], [1 0], [1 0]]
    best_idx = idx[:, 0]
    #print(best_idx)

    feature_conv = target_layer
    bs, nc, h, w = feature_conv.shape
    output_cam = []
    # 使用 unsqueeze 扩展维度
    #fc_weights = fc_weights.transpose(0, 1)
    for i in range(bs):
        feature_conv = feature_conv.reshape(bs,nc, h * w)
        #print(feature_conv.shape)
        cam = fc_weights[best_idx[i]].matmul(feature_conv[i])  # (2048, ) * (4,2048, 14*14) -> (4,14*14) torch.Size([4, 196])
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize
        cam_img = np.uint8(255 * cam_img.detach().to('cpu').numpy())
        output_cam.append(cam_img)
        #print(cam_img.shape)
    output_cam = np.array(output_cam).reshape(bs,h,w)
    output_cam_tensor=torch.tensor(output_cam).to(torch.float)


    #A = torch.sum(output_cam_tensor, dim=1, keepdim=True)  # dim=1，表示在通道维度上进行求和。这意味着对每个特征通道上的所有值进行求和，得到一个新的特征图，每个像素值表示原始特征图在对应通道上的值之和。
    a = torch.mean(output_cam_tensor, dim=[1, 2], keepdim=True)
    M = (output_cam_tensor > a).float()


    coordinates = []
    for i, m in enumerate(M):
        mask_np = m.cpu().numpy().reshape(8, 8)
        component_labels = measure.label(mask_np)  # 使用 measure.label 函数对二进制掩码进行标签化，
        # 即将相连的像素点分为一个区域并为每个区域分配唯一的标签。
        # 这通常用于标识掩码中的不同目标区域。

        properties = measure.regionprops(component_labels)  # 使用 measure.regionprops 函数获取标签化区域的属性。
        # 这些属性包括区域的面积、边界框等。
        # 存储每个区域的面积，并找到具有最大面积的区域的索引
        areas = []
        for prop in properties:
            areas.append(prop.area)
        max_idx = areas.index(max(areas))

        intersection = (component_labels == (max_idx + 1)).astype(int) == 1  # 即两个掩码都为1，和才为0，即两个特征图都感兴趣的部分
        #print('CAM激活区域:{}'.format(intersection))
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
