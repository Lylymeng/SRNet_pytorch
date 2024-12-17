import torch
import torch.nn as nn


#type1
class Block1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(),
        )

    def forward(self, inputs):
        ans = self.block(inputs)
        # print('ans shape: ', ans.shape)
        return ans

#type2
class Block2(nn.Module):
    def __init__(self):
        super(Block2, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
        )

    def forward(self, inputs):
        ans = torch.add(inputs, self.block(inputs))
        # print('ans shape: ', ans.shape)
        #return inputs + ans
        return ans


class Block3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=self.out_channels),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, inputs):
        ans = torch.add(self.branch1(inputs), self.branch2(inputs))
        # print('ans shape: ', ans.shape)
        return ans


class Block4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block4, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
        )

    def forward(self, inputs):
        temp = self.block(inputs)
        ans = torch.mean(temp, dim=(2, 3))
        # print('ans shape: ', ans.shape)
        return ans


class SRNet(nn.Module):
    def __init__(self, data_format='NCHW', init_weights=True, pth_path=None):
        super(SRNet, self).__init__()
        #self.conv1 = nn.Conv2d(444, 1, kernel_size=3, stride=1, padding=1)
        self.inputs = None
        self.outputs = None
        self.data_format = data_format

        # ��һ�ֽṹ����
        self.layer1 = Block1(1, 64)
        self.layer2 = Block1(64, 16)

        # �ڶ��ֽṹ����
        self.layer3 = Block2()
        self.layer4 = Block2()
        self.layer5 = Block2()
        self.layer6 = Block2()
        self.layer7 = Block2()

        # ����������
        self.layer8 = Block3(16, 16)
        self.layer9 = Block3(16, 64)
        self.layer10 = Block3(64, 128)
        self.layer11 = Block3(128, 256)
        self.layer12 = Block3(256, 512)

        # ����������
        self.layer13 = Block4(256, 512)#ԭ12��

        # ���һ�㣬ȫ���Ӳ�
        #self.layer13 = nn.Linear(512, 2)

        if init_weights:
            self._init_weights()
        if pth_path is not None:
            self.load_state_dict(torch.load(pth_path), strict=False)

    def forward(self, inputs):
        #inputs = inputs.permute(0, 3, 1, 2)  # NHWC -> NCHW

        # #11�ӣ�
        # inputs = self.conv1(inputs)

        self.inputs = inputs.float()
        # print('self.input.shape: ', self.inputs.shape)

        # ��һ�ֽṹ����
        x = self.layer1(self.inputs)
        x = self.layer2(x)

        # �ڶ��ֽṹ����
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        # ����������
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        #x = self.layer12(x)

        # ���������� ԭ
        #x = self.layer12(x)

        # ���һ��ȫ����
        #self.outputs = self.layer13(x)

        fm = self.layer12(x)
        embedding = self.layer13(x)
        #self.outputs = self.softmax(x)
        # print('self.outputs.shape: ', self.outputs.shape)
        return fm, embedding

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.2)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.001)


# # ��������ṹ�Ƿ���ȷ

# x = torch.rand(size=(3, 256, 256, 1))
# print(x.shape)

# net = SRNet(data_format='NCHW', init_weights=True)
# print(net)

# output_Y = net(x)
# print('output shape: ', output_Y.shape)
# print('output: ', output_Y)

