import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from train_valid_function import train
from dataloader import generate_data
from SRNet import SRNet


# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 实例化模型
model = SRNet(data_format='NCHW', init_weights=True)
print('model down successfully')

# 数据预处理
data_path = {
    'train_cover':'../../../public/steganography_dataset/Bossbase-1-0.4bpp-WOW-size256/tc/',
    'train_stego': '../../../public/steganography_dataset/Bossbase-1-0.4bpp-WOW-size256/ts/',
    'valid_cover': '../../../public/steganography_dataset/Bossbase-1-0.4bpp-WOW-size256/vc/',
    'valid_stego': '../../../public/steganography_dataset/Bossbase-1-0.4bpp-WOW-size256/vs/'
}


batch_size = {'train': 16, 'valid': 16}

train_loader, valid_loader = generate_data(data_path, batch_size)
print('data_loader down successfully')

# 训练参数设置
EPOCHS = 180
write_interval = 375
valid_interval = 375
save_interval = 1000
learning_rate = 1e-3

# 损失函数 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(), lr=learning_rate)

load_path = None
#load_path = './Mo_onetask_WOW0.6_best/bestModel_36000_0.8785.pth'
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

