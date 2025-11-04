from flow_matching.utils import ModelWrapper
import torch
import torch.nn as nn
import torch.nn.functional as F

FINAL_CHANNELS = 1  # velocity per pixel
KERNEL_SIZE_FINAL = 1
KERNEL_SIZE_CONV = 3
KERNEL_SIZE_RESAMPLE = 2
STRIDE_RESAMPLE = 2
PADDING = 1

class UnetVelocityField(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        # Encoder blocks
        self.enc1 = EncoderBlockGeneric(2, 16, dropout_rate)   # 2 input channels: image + t
        self.enc2 = EncoderBlockBottleneck(16, 32, dropout_rate)
        # Decoder blocks
        self.dec2 = DecoderBlockBottleneck(32, 16, dropout_rate)
        self.dec1 = DecoderBlockFinal(16, dropout_rate, apply_sigmoid=False)

    def forward(self, x, t):
        # x: [batch, 1, H, W]
        # t: [batch, 1] -> broadcast to spatial dimensions
        t_expanded = t.view(-1, 1, 1, 1).expand(x.shape[0], 1, x.shape[2], x.shape[3])
        x_in = torch.cat([x, t_expanded], dim=1)

        # Encoder
        x, skip1 = self.enc1(x_in)
        x = self.enc2(x)
        # Decoder
        x = self.dec2(x)
        x = self.dec1(x, skip1)
        return x

class EncoderBlockGeneric(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=KERNEL_SIZE_CONV, padding=PADDING)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=KERNEL_SIZE_CONV, padding=PADDING)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.pool = nn.MaxPool2d(kernel_size=KERNEL_SIZE_RESAMPLE, stride=STRIDE_RESAMPLE)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        skip = x
        x = self.pool(x)
        return x, skip

class EncoderBlockBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=KERNEL_SIZE_CONV, padding=PADDING)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        return x

class DecoderBlockBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=KERNEL_SIZE_CONV, padding=PADDING)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=KERNEL_SIZE_RESAMPLE, stride=STRIDE_RESAMPLE)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.upconv(x)
        return x

class DecoderBlockFinal(nn.Module):
    def __init__(self, in_channels, dropout_rate, apply_sigmoid):
        super().__init__()
        self.apply_sigmoid = apply_sigmoid
        self.conv1 = nn.Conv2d(2*in_channels, in_channels, kernel_size=KERNEL_SIZE_CONV, padding=PADDING)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=KERNEL_SIZE_CONV, padding=PADDING)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.final_conv = nn.Conv2d(in_channels, FINAL_CHANNELS, kernel_size=KERNEL_SIZE_FINAL)

    def forward(self, x, skip_connection):
        x = torch.cat([x, skip_connection], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.final_conv(x)
        if self.apply_sigmoid:
            x = torch.sigmoid(x)
        return x

class UnetVelocityModel(ModelWrapper):
    def __init__(self, dropout_rate):
        model = UnetVelocityField(dropout_rate)
        super().__init__(model)