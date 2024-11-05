import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_p=0.2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x
        
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = ConvBlock(3, 16, stride=2)
        self.conv2 = ConvBlock(16, 32, stride=2)
        self.conv3 = ConvBlock(32, 32)
        self.conv4 = ConvBlock(32, 64, stride=2)
        self.conv5 = ConvBlock(64, 64)
        self.conv6 = ConvBlock(64, 64)
        self.conv7 = ConvBlock(64, 128, stride=2)
        self.conv8 = ConvBlock(128, 128)
        self.conv9 = ConvBlock(128, 256, stride=2)
        self.conv10 = ConvBlock(256, 256)
        # self.conv11 = ConvBlock(256, 256)
        # self.conv12 = ConvBlock(256, 256)
        
        self.fc1 = nn.Linear(256 * 7 * 7, 256)
        self.final_dropout = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 525)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        # x = self.conv11(x)
        # x = self.conv12(x)
        
        x = torch.flatten(x, 1)  # Flatten before the fully connected layers
        x = F.relu(self.bn(self.fc1(x))) 
        x = self.final_dropout(x)
        
        x = self.fc2(x)
        return x