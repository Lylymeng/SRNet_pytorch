import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from onetask_looptrain1fun import train1
from dataloader import generate_data
from SRNet import SRNet

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 实例化模型
model = SRNet(data_format='NCHW', init_weights=True)
print('model down successfully')

# 数据预处理

bpp_datasets = {

    0.3: {
        'train_cover': '../MTL_MMAL_dataset/BOWS2+Bossbase-cover/tc/',
        'train_stego': './03bpp-SUNI/ts/',
        'valid_cover': '../MTL_MMAL_dataset/BOWS2+Bossbase-cover/vc/',
        'valid_stego': './03bpp-SUNI/vs/'
    },
    0.2: {
        'train_cover': '../MTL_MMAL_dataset/BOWS2+Bossbase-cover/tc/',
        'train_stego': './02bpp-SUNI/ts/',
        'valid_cover': '../MTL_MMAL_dataset/BOWS2+Bossbase-cover/vc/',
        'valid_stego': './02bpp-SUNI/vs/'
    }
}

batch_size = {'train': 16, 'valid': 16}

# 训练参数设置
EPOCHS = 180
write_interval = 875
valid_interval = 875
save_interval = 4375
learning_rate = 1e-3
# 损失函数 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
load_path = None

# 开始训练
print('start train')

train1(model=model,
       bpp_datasets=bpp_datasets,
       batch_size=batch_size,
       optimizer=optimizer,
       criterion=criterion,
       device=device,
       EPOCHS=EPOCHS,
       valid_interval=valid_interval,
       write_interval=write_interval,
       save_interval=save_interval,
       load_path=load_path)