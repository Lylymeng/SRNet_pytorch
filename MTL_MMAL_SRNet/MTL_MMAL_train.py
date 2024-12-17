# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from MTL_MMAL_trainfunction import train
from MTL_MMAL_model import MainNet
from MTL_MMAL_config import num_classes, model_name, model_path, lr_milestones, lr_decay_rate, input_size, \
    root, end_epoch, save_interval, init_lr, batch_size, weight_decay, \
    proposalN, set, channels  # ,CUDA_VISIBLE_DEVICES

# device
os.environ['CUDA_VISIBLE_DEVICES']='1,2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)


print('model down successfully')

bpp_datasets = {

    1.0: {
        'train_cover': '../MTL_MMAL_dataset/BOWS2+Bossbase-cover/tc/',
        'train_stego': '../MTL_MMAL_dataset/10bpp-WOW/ts/',
        # 'train_blurred':'./dataset/rand_blurred/',
        'train_blurred': '../MTL_MMAL_dataset/MTL_blurred/',
        'train_bs': '../MTL_MMAL_dataset/MTL_bs_WOW_1.0/',
        'valid_cover': '../MTL_MMAL_dataset/BOWS2+Bossbase-cover/vc/',
        'valid_stego': '../MTL_MMAL_dataset/10bpp-WOW/vs/'
    },
    0.8: {
        'train_cover': '../MTL_MMAL_dataset/BOWS2+Bossbase-cover/tc/',
        'train_stego': '../MTL_MMAL_dataset/08bpp-WOW/ts/',
        # 'train_blurred':'./dataset/rand_blurred/',
        'train_blurred': '../MTL_MMAL_dataset/MTL_blurred/',
        'train_bs': '../MTL_MMAL_dataset/MTL_bs_WOW_0.8/',
        'valid_cover': '../MTL_MMAL_dataset/BOWS2+Bossbase-cover/vc/',
        'valid_stego': '../MTL_MMAL_dataset/08bpp-WOW/vs/'
    },
    0.6: {
        'train_cover': '../MTL_MMAL_dataset/BOWS2+Bossbase-cover/tc/',
        'train_stego': '../MTL_MMAL_dataset/06bpp-WOW/ts/',
        # 'train_blurred':'./dataset/rand_blurred/',
        'train_blurred': '../MTL_MMAL_dataset/MTL_blurred/',
        'train_bs': '../MTL_MMAL_dataset/MTL_bs_WOW_0.6/',
        'valid_cover': '../MTL_MMAL_dataset/BOWS2+Bossbase-cover/vc/',
        'valid_stego': '../MTL_MMAL_dataset/06bpp-WOW/vs/'
    }

}

batch_size = {'train': 4, 'valid': 4}
print('data_loader down successfully')
EPOCHS = 180
write_interval = 3500
valid_interval = 3500
save_interval = 3500
learning_rate = 1e-3

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(), lr=learning_rate)

load_path = None
# load_path = './Mo_3x3_0.6bpp_WOW_best/bestModel_44000_0.9025.pth'

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

