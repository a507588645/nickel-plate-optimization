import torch
import torch.nn as nn
from torch.nn import functional as F
 
 


 
class ResNet18BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResNet18BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        # 处理 shortcut 的维度不一致问题
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + shortcut)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个阶段，每个阶段包含两个 BasicBlock
        self.layer1 = self._make_layer(ResNet18BasicBlock, 64, 64, stride=1)
        self.layer2 = self._make_layer(ResNet18BasicBlock, 64, 128, stride=2)
        self.layer3 = self._make_layer(ResNet18BasicBlock, 128, 256, stride=2)
        self.layer4 = self._make_layer(ResNet18BasicBlock, 256, 512, stride=2)

        # 平均池化和全连接
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, in_channel, out_channel, stride):
        layers = []
        layers.append(block(in_channel, out_channel, stride))
        layers.append(block(out_channel, out_channel, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)  # 展平
        out = self.fc(out)
        return out
    

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    net = ResNet50()
    out = net(x)
    print('out.shape: ', out.shape)
    print(out)
 