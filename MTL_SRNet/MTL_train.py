import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from MTL_trainfunction1 import train

from MTL_dataloader1 import generate_train_data
from MTL_dataloader1 import generate_valid_data
from MTL_SRmodel import MultiTaskSRNet


# device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 实例化模型
model = MultiTaskSRNet(data_format='NCHW', init_weights=True)
print('model down successfully')

# 数据预处理
#训练集使用多任务的训练集（隐写，滤波，滤波隐写），验证和测试的时候只使用隐写数据集
data_path = {
    'train_cover':'../tc/',
    'train_stego': '../../../public/steganography_dataset/Bossbase-3-0.4bpp-HUGO-size256/ts/',
    'train_blurred':'../ave_blurred_3x3/tc_blurred/',
    'train_bs':'../blurres_stego/blurred3_stego0.4_HUGO/ts/',

    'valid_cover': '../../../public/steganography_dataset/Bossbase-1-0.4bpp-WOW-size256/vc/',
    'valid_stego': '../../../public/steganography_dataset/Bossbase-3-0.4bpp-HUGO-size256/vs/',
}


batch_size = {'train': 16, 'valid': 16}
train_loader = generate_train_data(data_path, batch_size)
valid_loader = generate_valid_data(data_path, batch_size)
print('data_loader down successfully')

# 训练参数设置
EPOCHS = 180
write_interval = 100
valid_interval = 500
save_interval = 1000
learning_rate = 1e-3

# 损失函数 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
#load_path = None
load_path = './Mo_3x3_0.6bpp_HUGO_best/bestModel_42000_0.9227.pth'
# 开始训练
print('start train')
train(model=model,
      train_loader=train_loader,
      valid_loader=valid_loader,
      EPOCHS=EPOCHS,
      optimizer=optimizer,
      criterion=criterion,
      device=device,
      valid_interval=valid_interval,
      save_interval=save_interval,
      write_interval=write_interval,
      load_path=load_path)

