import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from MTL_loop_train1fun import train
from MTL_SRmodel import MultiTaskSRNet


# device
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

#实例化模型
model = MultiTaskSRNet(data_format='NCHW', init_weights=True)
print('model down successfully')

#数据集
bpp_datasets = {

    0.3: {
        'train_cover': '../../MTL_MMAL_dataset/BOWS2+Bossbase-cover/tc/',
        'train_stego': '../03bpp-SUNI/ts/',
        #'train_blurred':'./dataset/rand_blurred/',
        'train_blurred':'../../MTL_MMAL_dataset/MTL_blurred/',
        'train_bs':'../../MTL_MMAL_dataset/MTL_bs_SUNI_0.3/',
        'valid_cover': '../../MTL_MMAL_dataset/BOWS2+Bossbase-cover/vc/',
        'valid_stego': '../03bpp-SUNI/vs/'
    },
    0.2: {
        'train_cover': '../../MTL_MMAL_dataset/BOWS2+Bossbase-cover/tc/',
        'train_stego': '../02bpp-SUNI/ts/',
        #'train_blurred':'./dataset/rand_blurred/',
        'train_blurred':'../../MTL_MMAL_dataset/MTL_blurred/',
        'train_bs':'../../MTL_MMAL_dataset/MTL_bs_SUNI_0.2/',
        'valid_cover': '../../MTL_MMAL_dataset/BOWS2+Bossbase-cover/vc/',
        'valid_stego': '../02bpp-SUNI/vs/'
    }
    
    
}

batch_size = {'train': 16, 'valid': 16}
print('data_loader down successfully')
EPOCHS = 230
write_interval = 875
valid_interval = 875
save_interval = 875
learning_rate = 1e-3
#损失函数 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
#预训练模型
load_path = None
#load_path = './Mo_3x3_0.6bpp_WOW_best/bestModel_44000_0.9025.pth'

print('start train')
train(model=model,
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

