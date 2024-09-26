from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_classes: int):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return self.softmax(x)


class CNN(nn.Module):
    def __init__(self, n_channels: int, n_classes: int):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 풀링
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # 64채널의 8x8 특징 맵 -> 512
        self.fc2 = nn.Linear(512, n_classes)  # 512 -> 10 (클래스 개수)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv1 + relu + pool
        x = self.pool(F.relu(self.conv2(x)))  # conv2 + relu + pool
        x = x.view(-1, 64 * 8 * 8)  # Flatten: 64채널, 8x8 크기의 특성 맵
        x = F.relu(self.fc1(x))  # Fully Connected Layer 1
        x = self.fc2(x)  # Fully Connected Layer 2 (출력)
        return x
    
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc(x)