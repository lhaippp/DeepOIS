"""Defines the neural network, losss function and metrics"""

import torch
from torch import nn
import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, params, bilinear=True):
        super(UNet, self).__init__()
        self.input_channels = params.input_channels
        self.out_channels = params.out_channels
        self.model_size = params.model_size
        self.bilinear = bilinear

        self.inc = DoubleConv(self.input_channels, int(32 * self.model_size))
        self.down1 = Down(int(32 * self.model_size), int(32 * 2 * self.model_size))
        self.down2 = Down(int(32 * 2 * self.model_size), int(32 * 2 * 2 * self.model_size))
        self.down3 = Down(int(32 * 2 * 2 * self.model_size), int(32 * 2 * 2 * 2 * self.model_size))
        factor = 2 if bilinear else 1
        self.down4 = Down(int(32 * 2 * 2 * 2 * self.model_size), int(32 * 2 * 2 * 2 * 2 * self.model_size // factor))
        self.up1 = Up(int(32 * 2 * 2 * 2 * 2 * self.model_size), int(32 * 2 * 2 * 2 * self.model_size // factor), bilinear)
        self.up2 = Up(int(32 * 2 * 2 * 2 * self.model_size), int(32 * 2 * 2 * self.model_size // factor), bilinear)
        self.up3 = Up(int(32 * 2 * 2 * self.model_size), int(32 * 2 * self.model_size // factor), bilinear)
        self.up4 = Up(int(32 * 2 * self.model_size), int(32 * self.model_size), bilinear)
        self.outc = OutConv(int(32 * self.model_size), self.out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# class UNet(nn.Module):
#     def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False, batch_norm=False, up_mode='upconv'):
#         super(UNet, self).__init__()
#         assert up_mode in ('upconv', 'upsample')
#         self.padding = padding
#         self.depth = depth
#         prev_channels = in_channels
#         self.down_path = nn.ModuleList()
#         for i in range(depth):
#             self.down_path.append(
#                 UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
#             )
#             prev_channels = 2 ** (wf + i)
#
#         self.up_path = nn.ModuleList()
#         for i in reversed(range(depth - 1)):
#             self.up_path.append(
#                 UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
#             )
#             prev_channels = 2 ** (wf + i)
#
#         self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)
#
#     def forward(self, x):
#         blocks = []
#         for i, down in enumerate(self.down_path):
#             x = down(x)
#             if i != len(self.down_path) - 1:
#                 blocks.append(x)
#                 x = F.max_pool2d(x, 2)
#
#         for i, up in enumerate(self.up_path):
#             x = up(x, blocks[-i - 1])
#
#         return self.last(x)
#
#
# class UNetConvBlock(nn.Module):
#     def __init__(self, in_size, out_size, padding, batch_norm):
#         super(UNetConvBlock, self).__init__()
#         block = []
#
#         block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
#         block.append(nn.ReLU())
#         if batch_norm:
#             block.append(nn.BatchNorm2d(out_size))
#
#         block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
#         block.append(nn.ReLU())
#         if batch_norm:
#             block.append(nn.BatchNorm2d(out_size))
#
#         self.block = nn.Sequential(*block)
#
#     def forward(self, x):
#         out = self.block(x)
#         return out
#
#
# class UNetUpBlock(nn.Module):
#     def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
#         super(UNetUpBlock, self).__init__()
#         if up_mode == 'upconv':
#             self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
#         elif up_mode == 'upsample':
#             self.up = nn.Sequential(
#                 nn.Upsample(mode='bilinear', scale_factor=2),
#                 nn.Conv2d(in_size, out_size, kernel_size=1),
#             )
#
#         self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)
#
#     def center_crop(self, layer, target_size):
#         _, _, layer_height, layer_width = layer.size()
#         diff_y = (layer_height - target_size[0]) // 2
#         diff_x = (layer_width - target_size[1]) // 2
#         return layer[
#                :, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])
#                ]
#
#     def forward(self, x, bridge):
#         up = self.up(x)
#         crop1 = self.center_crop(bridge, up.shape[2:])
#         out = torch.cat([up, crop1], 1)
#         out = self.conv_block(out)
#
#         return out


def loss_fn_l1loss(output, label):
    return torch.nn.L1Loss()(output, label)


def loss_fn_l2loss(output, label):
    return torch.nn.MSELoss()(output, label)


def photo_loss_psnr(warp_imgs, match_imgs, data_range=255):
    eps = 1e-8

    warp_imgs = warp_imgs / data_range
    match_imgs = match_imgs / data_range

    # cut the black edge
    warp_imgs = warp_imgs[:, :, 10:-10, 10:-10]
    match_imgs = match_imgs[:, :, 10:-10, 10:-10]

    mse = torch.mean((warp_imgs - match_imgs)**2, dim=[1, 2, 3])
    score = -10 * torch.log10(mse + eps)

    psnr = 0 - torch.mean(score)

    return psnr


def photo_loss_l1(warp_imgs, match_imgs):
    return torch.nn.L1Loss()(warp_imgs[:, :, 10:-10, 10:-10], match_imgs[:, :, 10:-10, 10:-10])


def psnr(warp_imgs, match_imgs, data_range=255):
    """
    :param warp_imgs: gray image (torch tensor) NCHW
    :param match_imgs: gray image (torch tensor) NCHW
    :return: psnr (float): mean PSNR loss for all images in the batch
    """
    eps = 1e-8

    warp_imgs = warp_imgs / data_range
    match_imgs = match_imgs / data_range

    # cut the black edge
    warp_imgs = warp_imgs[:, :, 10:-10, 10:-10]
    match_imgs = match_imgs[:, :, 10:-10, 10:-10]

    mse = torch.mean((warp_imgs - match_imgs)**2, dim=[1, 2, 3])
    score = -10 * torch.log10(mse + eps)

    return float(score.mean().data)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'psnr': psnr,
    # could add more metrics such as accuracy for each token type
}
