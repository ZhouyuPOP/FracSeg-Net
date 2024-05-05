import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchsummary import summary


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid_flag, init_channel_number=32):
        super(Unet, self).__init__()

        self.encoders = nn.ModuleList([
            Encoder(in_channels, init_channel_number, max_pool_flag=False),
            Encoder(init_channel_number, 2 * init_channel_number),
            Encoder(2 * init_channel_number, 4 * init_channel_number),
            Encoder(4 * init_channel_number, 8 * init_channel_number)
        ])

        self.decoders = nn.ModuleList([
            Decoder((4+8) * init_channel_number, 4 * init_channel_number),
            Decoder((2+4) * init_channel_number, 2 * init_channel_number),
            Decoder((1+2) * init_channel_number, init_channel_number)
        ])

        # in the last layer a 1×1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(init_channel_number, out_channels, 1)
        if final_sigmoid_flag:
            if out_channels == 1:
                self.final_activation = nn.Sigmoid()
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        encoders_features = []

        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)
        # extremely important!! remove the last encoder's output from the list
        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders,  encoders_features):
            x = decoder(encoder_features, x)

        x = self.final_conv(x)
        if self.final_activation:
            x = self.final_activation(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3,
                 max_pool_flag=True, max_pool_kernel_size=(2, 2, 2)):
        super(Encoder, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size=max_pool_kernel_size, padding=0) if max_pool_flag else None
        self.double_conv = DoubleConv(in_channels, out_channels, kernel_size=conv_kernel_size)

    def forward(self, x):
        if self.max_pool is not None:
            x = self.max_pool(x)
        x = self.double_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2):
        super(Decoder, self).__init__()
        self.upsample = nn.ConvTranspose3d(2*out_channels, 2*out_channels, kernel_size, scale_factor, padding=1, output_padding=1)
        self.double_conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, encoder_features, x):
        x = self.upsample(x)
        x = torch.cat((encoder_features, x), dim=1)
        x = self.double_conv(x)
        return x


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DoubleConv, self).__init__()

        if in_channels < out_channels:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels // 2
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_conv(1, conv1_in_channels, conv1_out_channels, kernel_size)
        # conv2
        self.add_conv(2, conv2_in_channels, conv2_out_channels, kernel_size)


    def add_conv(self, pos, in_channels, out_channels, kernel_size):
        assert pos in [1, 2], 'pos must be either 1 or 2'

        self.add_module(f'conv{pos}', nn.Conv3d(in_channels, out_channels, kernel_size, padding=1))
        self.add_module(f'relu{pos}', nn.ReLU(inplace=True))
        self.add_module(f'norm{pos}', nn.BatchNorm3d(out_channels))


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = Unet(1, 5, final_sigmoid_flag=True, init_channel_number=64).cuda()


