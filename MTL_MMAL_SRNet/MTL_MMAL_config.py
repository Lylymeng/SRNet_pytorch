from MTL_MMAL_indices2coordinates import indices2coordinates
from MTL_MMAL_compute_window_nums import compute_window_nums
import numpy as np

#CUDA_VISIBLE_DEVICES = 1,2  # The current version only supports one GPU training 选择GPU


set = 'yx'  # Different dataset with different  #LV为TTF  鸟集为CUB
model_name = 'cam'  #鉴伪点

batch_size = 8
vis_num = batch_size  # The number of visualized images in tensorboard
eval_trainset = False  # Whether or not evaluate trainset
save_interval = 10
max_checkpoint_num = 50
end_epoch = 180 #最大训练轮数
init_lr = 0.001
lr_milestones = [90, 120]
lr_decay_rate = 0.1
weight_decay = 1e-4
stride = 32
channels = 512
input_size = 256

# The pth path of pretrained model
pretrain_path = '/root/autodl-fs/MMAL_SRNet/checkpoint/MMAL_yx/yx_WOW_0.6/cam/epoch170.pth'


if set == 'CUB':
    model_path = './checkpoint/cub'  # pth save path
    root = './datasets/CUB_200_2011'  # dataset path
    num_classes = 200
    # windows info for CUB
    N_list = [2, 2, 3]
    proposalN = sum(N_list)  # proposal window num
    window_side = [128, 192, 256]
    iou_threshs = [0.25, 0.25, 0.25]
    ratios = [[4, 4], [3, 5], [5, 3],
              [6, 6], [5, 7], [7, 5],
              [8, 8], [6, 10], [10, 6], [7, 9], [9, 7], [7, 10], [10, 7]]
elif set == 'TTF':
    #model_path = './checkpoint/luxury_700/yuanbiao_gray'  # pth save path 模型参数保存路径
    #root = "./datasets/luxuryForMMAL/yuanbiao_gray/"  # dataset path 数据集加载路径 原代码
    #root = "./luxuryForMMAL/yuanbiao_gray/"  # dataset path 数据集加载路径
    model_path  = './checkpoint/gucci_augment/'
    root = "./datasets/GUCCI/sy/Gucci_label_augtri/"
    num_classes = 2
    # windows info for CUB
    N_list = [3, 2, 1]
    proposalN = sum(N_list)  # proposal window num
    window_side = [128, 192, 256]
    iou_threshs = [0.25, 0.25, 0.25]
    ratios = [[5, 5], [4, 6], [6, 4],
              [6, 6], [5, 7], [7, 5],
              [8, 8], [6, 10], [10, 6], [7, 9], [9, 7], [7, 10], [10, 7]]
elif set == 'yx':
    model_path = './checkpoint/MMAL_yx/yx_WOW_0.4'  # pth save path
    root = './datasets/nature/WOW_0.4'
    num_classes = 2
    N_list = [3, 2, 1]
    proposalN = sum(N_list)  # proposal window num
    window_side = [128, 192, 256]
    iou_threshs = [0.25, 0.25, 0.25]
    ratios = [[2, 2], [3, 5], [5, 3],
              [5, 5], [5, 6], [6, 5],
              [6, 6], [6, 8], [8, 6], [7, 7], [8, 7], [7, 8], [8, 8]]
else:
    # windows info for CAR and Aircraft
    #窗口信息
    N_list = [1, 2, 1]
    proposalN = sum(N_list)  # proposal window num
    iou_threshs = [0.25, 0.25, 0.25]

    ratios = [[7, 7], [8, 7], [7, 8],
              [8, 8], [6, 10], [10, 6], [7, 9], [9, 7],
              [10, 10], [9, 11], [11, 9], [8, 12], [12, 8]]
    window_side = [64, 128, 192]
    if set == 'CAR':
        model_path = './checkpoint/car'      # pth save path
        root = './datasets/Stanford_Cars'  # dataset path
        num_classes = 196
    elif set == 'Aircraft':
        model_path = './checkpoint/aircraft'      # pth save path
        root = './datasets/FGVC-aircraft'  # dataset path
        num_classes = 100
    elif set == 'Defect':
        model_path = './checkpoint/defect_detection'      # pth save path
        root = './datasets/defect_detection'  # dataset path
        num_classes = 13

'''indice2coordinates'''
window_nums = compute_window_nums(ratios, stride, input_size)
indices_ndarrays = [np.arange(0,window_num).reshape(-1,1) for window_num in window_nums]
coordinates = [indices2coordinates(indices_ndarray, stride, input_size, ratios[i]) for i, indices_ndarray in enumerate(indices_ndarrays)] # 每个window在image上的坐标
coordinates_cat = np.concatenate(coordinates, 0)
window_milestones = [sum(window_nums[:i+1]) for i in range(len(window_nums))]
if set == 'CUB':
    window_nums_sum = [0, sum(window_nums[:3]), sum(window_nums[3:6]), sum(window_nums[6:])]
else:
    window_nums_sum = [0, sum(window_nums[:3]), sum(window_nums[3:8]), sum(window_nums[8:])]
