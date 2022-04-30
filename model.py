# 搭建神经网络
import torch
from torch import nn


class cifar_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=2), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=2), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 100)
        )

    def forward(self, x):
        x = self.feature(x)
        output = self.classifier(x)
        return output


if __name__ == '__main__':
    test_cifar = cifar_model()
    Input = torch.ones((64, 3, 32, 32))
    output = test_cifar(Input)
    print(output.shape)  # torch.Size([64, 10]) # 检查模型的正确性

