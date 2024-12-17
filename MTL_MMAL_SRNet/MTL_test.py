import torch
from MTL_dataloader1 import generate_test_data
from MTL_MMAL_trainfunction import test
from MTL_MMAL_model import MainNet
import torch.nn as nn
from MTL_MMAL_config import num_classes, model_name, model_path, lr_milestones, lr_decay_rate, input_size, \
    root, end_epoch, save_interval, init_lr, batch_size, weight_decay, \
    proposalN, set, channels#,CUDA_VISIBLE_DEVICES
weight_path = './Model/MTL_MMAL_SUNI_best/bestModel_1.0_80epoch_0.9895.pth'

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)
print('model down successfully')


data_path = {
    'test_cover': '../../../../public/steganography_dataset/Bossbase-1-0.4bpp-WOW-size256/testc/',
    'test_stego': './MTL_MMAL_dataset/622-Boss-2-1.0bpp-S-UNIWARD-size256/tests/'
}
batch_size = 4
test_loader = generate_test_data(data_path, batch_size)
criterion = nn.CrossEntropyLoss()
print('data_loader down successfully')


test_accuracy = test(model=model, test_loader=test_loader, device=device, criterion=criterion,weight_path=weight_path)

print("Test Accuracy:", test_accuracy)