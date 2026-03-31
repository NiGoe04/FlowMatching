from flow_matching.utils import ModelWrapper
import torch
import torch.nn as nn

from src.flow_matching.model.velocity_model_unet import (
    EncoderBlockGeneric,
    EncoderBlockBottleneck,
    DecoderBlockGeneric,
    DecoderBlockBottleneck,
    DecoderBlockFinal,
)

INPUT_CHANNELS = 3
TIME_CHANNELS = 1
FINAL_CHANNELS = 3  # rgb picture


class _ImageNetUnetBase(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _concat_time(x, t):
        t_expanded = t.view(-1, 1, 1, 1).expand(x.shape[0], TIME_CHANNELS, x.shape[2], x.shape[3])
        return torch.cat([x, t_expanded], dim=1)


class UnetVelocityFieldImageNet8(_ImageNetUnetBase):
    def __init__(self, dropout_rate):
        super().__init__()
        in_channels = INPUT_CHANNELS + TIME_CHANNELS

        self.enc_bottleneck = EncoderBlockBottleneck(in_channels, 16, dropout_rate)
        self.dec_final = DecoderBlockFinal(
            in_channels=16,
            dropout_rate=dropout_rate,
            apply_sigmoid=False,
            final_channels=FINAL_CHANNELS,
        )

    def forward(self, x, t):
        x = self._concat_time(x, t)
        x = self.enc_bottleneck(x)
        x = self.dec_final(x, x)
        return x


class UnetVelocityFieldImageNet16(_ImageNetUnetBase):
    def __init__(self, dropout_rate):
        super().__init__()
        in_channels = INPUT_CHANNELS + TIME_CHANNELS

        self.enc1 = EncoderBlockGeneric(in_channels, 16, dropout_rate)
        self.enc_bottleneck = EncoderBlockBottleneck(16, 32, dropout_rate)

        self.dec_bottleneck = DecoderBlockBottleneck(32, 16, dropout_rate)
        self.dec_final = DecoderBlockFinal(
            in_channels=16,
            dropout_rate=dropout_rate,
            apply_sigmoid=False,
            final_channels=3,
        )

    def forward(self, x, t):
        x = self._concat_time(x, t)

        x, skip1 = self.enc1(x)
        x = self.enc_bottleneck(x)

        x = self.dec_bottleneck(x)
        x = self.dec_final(x, skip1)
        return x


class UnetVelocityFieldImageNet32(_ImageNetUnetBase):
    def __init__(self, dropout_rate):
        super().__init__()
        in_channels = INPUT_CHANNELS + TIME_CHANNELS

        self.enc1 = EncoderBlockGeneric(in_channels, 16, dropout_rate)
        self.enc2 = EncoderBlockGeneric(16, 32, dropout_rate)
        self.enc_bottleneck = EncoderBlockBottleneck(32, 64, dropout_rate)

        self.dec_bottleneck = DecoderBlockBottleneck(64, 32, dropout_rate)
        self.dec2 = DecoderBlockGeneric(32, 16, dropout_rate)
        self.dec_final = DecoderBlockFinal(
            in_channels=16,
            dropout_rate=dropout_rate,
            apply_sigmoid=False,
            final_channels=FINAL_CHANNELS,
        )

    def forward(self, x, t):
        x = self._concat_time(x, t)

        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x = self.enc_bottleneck(x)

        x = self.dec_bottleneck(x)
        x = self.dec2(x, skip2)
        x = self.dec_final(x, skip1)
        return x


class UnetVelocityFieldImageNet64(_ImageNetUnetBase):
    def __init__(self, dropout_rate):
        super().__init__()
        in_channels = INPUT_CHANNELS + TIME_CHANNELS

        self.enc1 = EncoderBlockGeneric(in_channels, 16, dropout_rate)
        self.enc2 = EncoderBlockGeneric(16, 32, dropout_rate)
        self.enc3 = EncoderBlockGeneric(32, 64, dropout_rate)
        self.enc_bottleneck = EncoderBlockBottleneck(64, 128, dropout_rate)

        self.dec_bottleneck = DecoderBlockBottleneck(128, 64, dropout_rate)
        self.dec3 = DecoderBlockGeneric(64, 32, dropout_rate)
        self.dec2 = DecoderBlockGeneric(32, 16, dropout_rate)
        self.dec_final = DecoderBlockFinal(
            in_channels=16,
            dropout_rate=dropout_rate,
            apply_sigmoid=False,
            final_channels=FINAL_CHANNELS,
        )

    def forward(self, x, t):
        x = self._concat_time(x, t)

        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x = self.enc_bottleneck(x)

        x = self.dec_bottleneck(x)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec_final(x, skip1)
        return x


class UnetVelocityModelImageNet8(ModelWrapper):
    def __init__(self, dropout_rate, device):
        model = UnetVelocityFieldImageNet8(dropout_rate).to(device)
        super().__init__(model)


class UnetVelocityModelImageNet16(ModelWrapper):
    def __init__(self, dropout_rate, device):
        model = UnetVelocityFieldImageNet16(dropout_rate).to(device)
        super().__init__(model)


class UnetVelocityModelImageNet32(ModelWrapper):
    def __init__(self, dropout_rate, device):
        model = UnetVelocityFieldImageNet32(dropout_rate).to(device)
        super().__init__(model)


class UnetVelocityModelImageNet64(ModelWrapper):
    def __init__(self, dropout_rate, device):
        model = UnetVelocityFieldImageNet64(dropout_rate).to(device)
        super().__init__(model)
