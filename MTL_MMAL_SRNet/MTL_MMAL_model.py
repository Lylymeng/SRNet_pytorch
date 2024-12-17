from threading import local
import torch
import os
import datetime
import torchvision
from torch import nn
import torch.nn.functional as F
# from networks import resnet
from MTL_MMAL_SRNet import SRNet
from MTL_MMAL_config import pretrain_path, coordinates_cat, iou_threshs, window_nums_sum, ratios, N_list
import numpy as np
from MTL_MMAL_AOLM import AOLM
# from utils.adjust_image import adjust_image
from MTL_MMAL_cam import cam
from torch.nn import Parameter
from MTL_MMAL_pad_image_to_square import pad_image_to_square
import torchvision.transforms as transforms

SRM_npy = np.load(os.path.join(os.path.dirname(__file__), 'SRM_Kernels.npy'))


class SRMConv2d(nn.Module):

    def __init__(self, stride=1, padding=0):
        super(SRMConv2d, self).__init__()
        self.in_channels = 1
        self.out_channels = 30
        self.kernel_size = (5, 5)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.dilation = (1, 1)
        self.transpose = False
        self.output_padding = (0,)
        self.groups = 1
        self.weight = Parameter(torch.Tensor(30, 1, 5, 5), requires_grad=True)
        self.bias = Parameter(torch.Tensor(30), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def nms(scores_np, proposalN, iou_threshs, coordinates):
    if not (type(scores_np).__module__ == 'numpy' and len(scores_np.shape) == 2 and scores_np.shape[1] == 1):
        raise TypeError('score_np is not right')

    windows_num = scores_np.shape[0]
    # print(windows_num)
    indices_coordinates = np.concatenate((scores_np, coordinates), 1)

    indices = np.argsort(indices_coordinates[:, 0])
    indices_coordinates = np.concatenate((indices_coordinates, np.arange(0, windows_num).reshape(windows_num, 1)), 1)[
        indices]  # [339,6]
    indices_results = []

    # print("indices_coordinates: ", indices_coordinates)

    res = indices_coordinates

    if res.shape[0] == 0:
        print("res.shape[0] == 0")

    while res.any():
        indice_coordinates = res[-1]
        indices_results.append(indice_coordinates[5])

        if len(indices_results) == proposalN:
            return np.array(indices_results).reshape(1, proposalN).astype(np.int64)
        res = res[:-1]

        # Exclude anchor boxes with selected anchor box whose iou is greater than the threshold
        start_max = np.maximum(res[:, 1:3], indice_coordinates[1:3])
        end_min = np.minimum(res[:, 3:5], indice_coordinates[3:5])
        lengths = end_min - start_max + 1
        intersec_map = lengths[:, 0] * lengths[:, 1]
        intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
        iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1] + 1) * (res[:, 4] - res[:, 2] + 1) +
                                      (indice_coordinates[3] - indice_coordinates[1] + 1) *
                                      (indice_coordinates[4] - indice_coordinates[2] + 1) - intersec_map)
        res = res[iou_map_cur <= iou_threshs]

    while len(indices_results) != proposalN:
        indices_results.append(indice_coordinates[5])

    return np.array(indices_results).reshape(1, -1).astype(np.int64)


class APPM(nn.Module):
    def __init__(self):
        super(APPM, self).__init__()
        self.avgpools = [nn.AvgPool2d(ratios[i], 1) for i in range(len(ratios))]  # 原代码

        # self.avgpools = [nn.AvgPool2d(kernel_size=1, stride=1) for _ in range(len(ratios))]

        # 如果ratios是一个二维列表，你可以这样修改你的代码
        # self.avgpools = [nn.AvgPool2d([r * 14 // 8 for r in ratios[i]], 1) for i in range(len(ratios))]

        # 如果ratios是一个一维列表，你可以这样修改你的代码
        # self.avgpools = [nn.AvgPool2d(ratios[i] // 2, 1) for i in range(len(ratios))]

    def forward(self, proposalN, x, ratios, window_nums_sum, N_list, iou_threshs, DEVICE='cuda'):
        batch, channels, _, _ = x.size()
        avgs = [self.avgpools[i](x) for i in range(len(ratios))]  # 原

        # feature map sum
        fm_sum = [torch.sum(avgs[i], dim=1) for i in range(len(ratios))]

        all_scores = torch.cat([fm_sum[i].view(batch, -1, 1) for i in range(len(ratios))], dim=1)
        windows_scores_np = all_scores.data.cpu().numpy()
        window_scores = torch.from_numpy(windows_scores_np).to(DEVICE).reshape(batch, -1)
        window_scores=window_scores.to(DEVICE)

        # nms
        proposalN_indices = []
        for i, scores in enumerate(windows_scores_np):
            indices_results = []
            for j in range(len(window_nums_sum) - 1):
                indices_results.append(
                    nms(scores[sum(window_nums_sum[:j + 1]):sum(window_nums_sum[:j + 2])], proposalN=N_list[j],
                        iou_threshs=iou_threshs[j],
                        coordinates=coordinates_cat[sum(window_nums_sum[:j + 1]):sum(window_nums_sum[:j + 2])]) + sum(
                        window_nums_sum[:j + 1]))
            # indices_results.reverse()
            proposalN_indices.append(np.concatenate(indices_results, 1))  # reverse

        proposalN_indices = np.array(proposalN_indices).reshape(batch, proposalN)
        proposalN_indices = torch.from_numpy(proposalN_indices).to(DEVICE)
        proposalN_windows_scores = torch.cat(
            [torch.index_select(all_score, dim=0, index=proposalN_indices[i]) for i, all_score in
             enumerate(all_scores)], 0).reshape(
            batch, proposalN)
        proposalN_windows_scores=proposalN_windows_scores.to(DEVICE)

        return proposalN_indices, proposalN_windows_scores, window_scores


class MainNet(nn.Module):
    def __init__(self, proposalN, num_classes, channels):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        self.proposalN = proposalN
        # self.pretrained_model = resnet.resnet50(pretrained=True, pth_path=pretrain_path)
        self.pretrained_model = SRNet(data_format='NCHW', init_weights=True)  # 修改SRNet，添加预训练参数路径
        self.rawcls_net = nn.Linear(channels, num_classes)
        self.APPM = APPM()
        self.srm_filter = SRMConv2d()  # 创建SRM滤波器的实例

    def forward(self, x, epoch, status='test', DEVICE='cuda'):
        # x = self.srm_filter(x)
        # 删除原始图像分支到目标图像的切割定位部分
        # outputs = self.pretrained_model(x)
        # print(len(outputs))
        # print(x.shape)
        # print(type(x))
        fm, embedding = self.pretrained_model(x)  # 特征图、嵌入向量、某个卷积层的输出
        # 特征图：输入图像中提取的高级特征表示，嵌入向量：用于图像相似性比较和检索
        # 卷积层：通常用于提取图像的低级特征，例如边缘和纹理
        # conc5_b是最后一层的前两个残差块的输出结果，fm是第三的也就是最后一个残差块的输出结果
        # print("Type of fm: ", type(fm))
        # print(fm.shape)
        batch_size, channel_size, side_size, _ = fm.shape
        # assert channel_size == 2048
        feature_temp = fm
        feature_temp=feature_temp.to(DEVICE)
        # raw branch
        raw_logits = self.rawcls_net(embedding).to(DEVICE)  # 第一层分支结果 embedding 隐写分析任务分类
        aux_logits = self.rawcls_net(embedding).to(DEVICE)  # 辅助任务分类
        # SCDA
        # coordinates = torch.tensor(AOLM(fm.detach(), conv5_b.detach())) #坐标

        # 获取全连接层的权重
        fc_weights = self.rawcls_net.weight  # torch.Size([2, 2048]) 2048是通道数，2是分类数

        # print("fc_weights: ", fc_weights)
        # print("fm: ", fm)
        # print("raw_logits: ", raw_logits)

        coordinates = torch.tensor(cam(fc_weights, fm, raw_logits)).to(DEVICE)

        local_imgs = torch.zeros([batch_size, 1, 256, 256]).to(DEVICE)  # 原为[N, 3, 448, 448]  11改隐写
        for i in range(batch_size):
            # print('x[i]',x[i])
            # print(x[i].shape)
            # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # save_path = './WOW_0.4'
            # save_path = os.path.join(save_path, "x_image_" + current_time + ".pgm")
            # x_tensor = x[i].float()  # 将 x[i] 转换为浮点型张量

            # 转换为 PIL 图像格式
            # to_pil = transforms.ToPILImage()(x_tensor)

            # 保存图像
            # to_pil.save(save_path)
            # torchvision.utils.save_image(x[i], save_path)
            [x0, y0, x1, y1] = coordinates[i]
            # print("coordinates[{}]:[{},{},{},{}]".format(i,x0, y0, x1, y1))
            # local_imgs[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1+1), y0:(y1+1)], size=(256, 256),
            #                                     mode='bilinear', align_corners=True)  # [N, 3, 224, 224] #11改隐写

            local_img = x[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)]

            h, w = local_img.shape[2:]
            local_imgs[i:i + 1, :, x0:(x0 + h), y0:(y0 + w)] = local_img
            # print('local_image',local_imgs[i])
            # save_path = './WOW_0.4'
            # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # save_path = os.path.join(save_path, "local_image_" + current_time + ".pgm")
            # torchvision.utils.save_image(local_imgs[i], save_path)

            # 11改隐写167-168
            # local_imgs[i:i+1] = adjust_image(local_imgs[i:i+1], [x0, y0, x1, y1], gamma=2.0)
            # print("Size of local_imgs[{}]: {}".format(i, local_imgs[i:i+1].size()))

        local_fm, local_embeddings = self.pretrained_model(local_imgs)  # [N, 2048]
        batch_size, channel_size, side_size, _ = local_fm.shape

        # # 输出local_fm的类型和内容
        # print("Type of local_fm: ", type(local_fm))
        # #print("Content of local_fm: ", local_fm)
        # print("Size of local_fm: ", local_fm.size())

        local_logits = self.rawcls_net(local_embeddings).to(DEVICE)  # [N, 2]
        # print(local_logits.shape)
        # aux_local_logits = self.rawcls_net(local_embeddings)
        proposalN_indices, proposalN_windows_scores, window_scores \
            = self.APPM(self.proposalN, local_fm.detach(), ratios, window_nums_sum, N_list, iou_threshs, DEVICE)

        if status == "train":
            # window_imgs cls0
            window_imgs = torch.zeros([batch_size, self.proposalN, 1, 256, 256]).to(
                DEVICE)  # 原为[N, 4, 3, 224, 224] 11改隐写
            for i in range(batch_size):
                for j in range(self.proposalN):
                    [x0, y0, x1, y1] = coordinates_cat[proposalN_indices[i, j]]
                    # local_imgs[i:i + 1, :] = pad_image_to_square(local_imgs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)],(200,200,200))
                    # window_imgs[i:i + 1, j] = F.interpolate(local_imgs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(256, 256),
                    #                                             mode='bilinear',
                    #                                             align_corners=True)  # [N, 4, 3, 224, 224]

                    window_img = local_imgs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)]  # 获取局部图像
                    h, w = window_img.shape[2:]  # 获取局部图像的大小
                    window_imgs[i:i + 1, j, :, x0:(x0 + h), y0:(y0 + w)] = window_img  # 将局部图像放置在全黑图像的相应位置

                    # save_path = './WOW_0.4'
                    # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    # save_path = os.path.join(save_path, "window_image_" + current_time + ".pgm")
                    # torchvision.utils.save_image(window_imgs[i], save_path)

                    # 11改隐写 191
                    # window_imgs[i:i + 1, j] = adjust_image(local_imgs[i:i + 1], [x0, y0, x1, y1], gamma=2.0)  # [N, 4, 3, 224, 224]

            window_imgs = window_imgs.reshape(batch_size * self.proposalN, 1, 256, 256)  # [N*4, 3, 224, 224]
            _, window_embeddings = self.pretrained_model(window_imgs.detach())  # [N*4, 2048]
            proposalN_windows_logits = self.rawcls_net(window_embeddings)  # [N* 4, 200]
        else:
            proposalN_windows_logits = torch.zeros([batch_size * self.proposalN, self.num_classes]).to(DEVICE)

        return proposalN_windows_scores, proposalN_windows_logits, proposalN_indices, \
            window_scores, coordinates, raw_logits, aux_logits, local_logits, local_imgs, local_fm, feature_temp
        '''
        return proposalN_windows_scores, proposalN_windows_logits, proposalN_indices, \
               window_scores, local_logits, x, local_fm, raw_logits'''