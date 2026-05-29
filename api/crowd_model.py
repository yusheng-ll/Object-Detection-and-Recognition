import torch
import torch.nn as nn
import torch.nn.functional as F

class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
        )
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv2d(512, 256, 3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv2d(256, 128, 3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv2d(128, 64, 3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x