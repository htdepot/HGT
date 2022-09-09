import torch.nn as nn
from torch.nn import functional as F
import math

class CNN(nn.Module):
    def __init__(self, in_channel, hidden, out_channel):
        super(CNN, self).__init__()
        self.conv0 = nn.Conv2d(in_channel, hidden, kernel_size=1)
        self.conv1_1 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden * 2, kernel_size=1)
        self.conv3_1 = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden * 2, out_channel, kernel_size=1)

    def forward(self, data):
        out = F.relu(self.conv0(data))
        temp = out
        out = F.relu(self.conv1_1(out))
        out = F.relu(self.conv1_2(out))
        out = out + temp
        out = F.relu(self.conv2(out))
        temp = out
        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        out = out + temp
        out = self.conv4(out)
        return out