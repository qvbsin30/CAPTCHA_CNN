# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CaptchaCNN(nn.Module):
    def __init__(self):
        super(CaptchaCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(512 * 5 * 2, 4 * 36)  # 輸入尺寸為 512 * 5 * 2，輸出為 4 * 36

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))      # 160x80 -> 80x40
        x = self.pool(F.relu(self.conv2(x)))      # 80x40 -> 40x20
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 40x20 -> 20x10
        x = self.pool(F.relu(self.conv4(x)))      # 20x10 -> 10x5
        x = self.pool(F.relu(self.bn5(self.conv5(x))))  # 10x5 -> 5x2
        x = x.view(-1, 512 * 5 * 2)  # 展平為 512 * 5 * 2
        x = self.fc(x)
        return x