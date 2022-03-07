import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO: Copy paste your agent architectures that you want to use for evaluation here:
class MarioKartBCAgent(nn.Module):
    def __init__(self):
        super(MarioKartBCAgent, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.fc1 = nn.Linear(25600, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)      
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output    # return x for visualization