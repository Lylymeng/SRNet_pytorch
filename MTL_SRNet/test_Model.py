import torch
from dataloader import generate_test_data
from onetask_looptrain1fun import test1
from SRNet import SRNet
import torch.nn as nn

weight_path = './mtlonetaskSUNI0302/Mo_onetask_SUNI_best/bestModel_0.2_157500_0.6462.pth'

# device
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# 实例化模型
model = SRNet(data_format='NCHW', init_weights=False)
print('model down successfully')

# 数据预处理
data_path = {
    'test_cover': '../MTL_MMAL_dataset/BOWS2+Bossbase-cover/testc/',
    #'test_stego': '../MTL_MMAL_dataset/04bpp-WOW/tests/'
    'test_stego': './02bpp-SUNI/tests/'
}
batch_size = 16
test_loader = generate_test_data(data_path, batch_size)
criterion = nn.CrossEntropyLoss()
print('data_loader down successfully')

# 执行测试
test_accuracy = test1(model=model, test_loader=test_loader, device=device, criterion=criterion,weight_path=weight_path)

print("Test Accuracy:", test_accuracy)