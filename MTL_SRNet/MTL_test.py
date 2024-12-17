import torch
from MTL_dataloader1 import generate_test_data
from MTL_loop_train1fun import test
from MTL_SRmodel import MultiTaskSRNet
import torch.nn as nn

weight_path = './Model/SUNI/best/bestModel_0.2_epoch230_0.6850.pth'

# device
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


model = MultiTaskSRNet(data_format='NCHW', init_weights=False)
print('model down successfully')


data_path = {
    'test_cover': '../../MTL_MMAL_dataset/BOWS2+Bossbase-cover/testc/',
    'test_stego': '../02bpp-SUNI/tests/'
}
batch_size = 16
test_loader = generate_test_data(data_path, batch_size)
criterion = nn.CrossEntropyLoss()
print('data_loader down successfully')


test_accuracy = test(model=model, test_loader=test_loader, device=device, criterion=criterion,weight_path=weight_path)

print("Test Accuracy:", test_accuracy)