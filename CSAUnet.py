import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchsummary import summary


class CsaUnet(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid_flag=False, init_channel_number=32):
        super(CsaUnet, self).__init__()

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

        self.attentions = nn.ModuleList([
            AttentionBlock(4 * init_channel_number, 8 * init_channel_number, 4 * init_channel_number),
            AttentionBlock(2 * init_channel_number, 4 * init_channel_number, 2 * init_channel_number),
            None
        ])
        # 1×1×1 convolution reduces the number of output channels to the number of class
        self.final_conv = nn.Conv3d(init_channel_number, out_channels, 1)

        if final_sigmoid_flag:
            self.final_activation = nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        encoders_features = []

        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)

        # extremely important!! remove the last encoder's output from the list
        first_layer_feature = encoders_features[-1]
        encoders_feature = encoders_features[1:]

        for decoder, attention, encoder_feature in zip(self.decoders, self.attentions, encoders_feature):
            if attention:
                features_after_att = attention(encoder_feature, x, first_layer_feature)
            else:    # no attention opr in first layer
                features_after_att = first_layer_feature
            x = decoder(features_after_att, x)

        x = self.final_conv(x)
        if hasattr(CsaUnet, 'final_activation'):
            x = self.final_activation(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3,
                 max_pool_flag=True, max_pool_kernel_size=(2, 2, 2)):
        super(Encoder, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size=max_pool_kernel_size, stride=2) if max_pool_flag else None
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


class AttentionBlock(nn.Module):
    def __init__(self, channel_l, channel_g, init_channel=64):
        super(AttentionBlock, self).__init__()
        self.W_x1 = nn.Conv3d(channel_l, channel_l, kernel_size=1)  # Encoder路径的第一层特征图
        self.W_x2 = nn.Conv3d(channel_l, channel_g, kernel_size=int(channel_g/channel_l))  # Encoder路径的任意层特征图
        self.W_g1 = nn.Conv3d(init_channel, channel_l, kernel_size=int(channel_l/init_channel))  # 第一次Attention的另一个输入
        self.W_g2 = nn.Conv3d(channel_g, channel_g, kernel_size=1)  # 第二次Attention的另一个输入
        self.relu = nn.ReLU()
        self.psi1 = nn.Conv3d(channel_l, out_channels=1, kernel_size=1)
        self.psi2 = nn.Conv3d(channel_g, out_channels=1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x_l, x_g, first_layer_f):
        # First Attention Operation
        first_layer_afterconv = self.W_g1(first_layer_f)
        xl_afterconv = self.W_x1(x_l)
        att_map_first = self.sig(self.psi(self.relu(first_layer_afterconv + xl_afterconv)))
        xl_after_first_att = x_l * att_map_first

        # Second Attention Operation
        xg_afterconv = self.W_g2(x_g)
        xl_after_first_att_and_conv = self.W_x2(xl_after_first_att)
        att_map_second = self.sig(self.psi(self.relu(xg_afterconv + xl_after_first_att_and_conv)))
        att_map_second_upsample = F.upsample(att_map_second, size=x_l.size()[2:], mode='trilinear')
        out = xl_after_first_att * att_map_second_upsample
        return out


if __name__ == '__main__':
    model = CsaUnet(1, 5, final_sigmoid_flag=True, init_channel_number=64).cuda()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    summary(model, (1, 96, 128, 128), batch_size=4)

