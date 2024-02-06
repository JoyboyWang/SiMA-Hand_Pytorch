import torch.nn as nn
import torch
from model.h2onet.conv.spiralconv import SpiralConv
from model.mob_recon.models.transformer import *

# Init model weights
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Conv1d:
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.BatchNorm2d or type(m) == nn.BatchNorm1d:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class Reorg(nn.Module):
    dump_patches = True

    def __init__(self):
        """Reorg layer to re-organize spatial dim and channel dim
        """
        super(Reorg, self).__init__()

    def forward(self, x):
        ss = x.size()
        out = x.view(ss[0], ss[1], ss[2] // 2, 2, ss[3]).view(ss[0], ss[1], ss[2] // 2, 2, ss[3] // 2, 2). \
            permute(0, 1, 3, 5, 2, 4).contiguous().view(ss[0], -1, ss[2] // 2, ss[3] // 2)
        return out


def conv_layer(channel_in, channel_out, ks=1, stride=1, padding=0, dilation=1, bias=False, bn=True, relu=True, group=1):
    """Conv block

    Args:
        channel_in (int): input channel size
        channel_out (int): output channel size
        ks (int, optional): kernel size. Defaults to 1.
        stride (int, optional): Defaults to 1.
        padding (int, optional): Defaults to 0.
        dilation (int, optional): Defaults to 1.
        bias (bool, optional): Defaults to False.
        bn (bool, optional): Defaults to True.
        relu (bool, optional): Defaults to True.
        group (int, optional): group conv parameter. Defaults to 1.

    Returns:
        Sequential: a block with bn and relu
    """
    _conv = nn.Conv2d
    sequence = [_conv(channel_in, channel_out, kernel_size=ks, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=group)]
    if bn:
        sequence.append(nn.BatchNorm2d(channel_out))
    if relu:
        sequence.append(nn.ReLU())

    return nn.Sequential(*sequence)


def linear_layer(channel_in, channel_out, bias=False, bn=True, relu=True):
    """Fully connected block

    Args:
        channel_in (int): input channel size
        channel_out (_type_): output channel size
        bias (bool, optional): Defaults to False.
        bn (bool, optional): Defaults to True.
        relu (bool, optional): Defaults to True.

    Returns:
        Sequential: a block with bn and relu
    """
    _linear = nn.Linear
    sequence = [_linear(channel_in, channel_out, bias=bias)]

    if bn:
        sequence.append(nn.BatchNorm1d(channel_out))
    if relu:
        sequence.append(nn.Hardtanh(0, 4))

    return nn.Sequential(*sequence)


class mobile_unit(nn.Module):
    dump_patches = True

    def __init__(self, channel_in, channel_out, stride=1, has_half_out=False, num3x3=1):
        """Init a depth-wise sparable convolution

        Args:
            channel_in (int): input channel size
            channel_out (_type_): output channel size
            stride (int, optional): conv stride. Defaults to 1.
            has_half_out (bool, optional): whether output intermediate result. Defaults to False.
            num3x3 (int, optional): amount of 3x3 conv layer. Defaults to 1.
        """
        super(mobile_unit, self).__init__()
        self.stride = stride
        self.channel_in = channel_in
        self.channel_out = channel_out
        if num3x3 == 1:
            self.conv3x3 = nn.Sequential(conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in), )
        else:
            self.conv3x3 = nn.Sequential(
                conv_layer(channel_in, channel_in, ks=3, stride=1, padding=1, group=channel_in),
                conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in),
            )
        self.conv1x1 = conv_layer(channel_in, channel_out)
        self.has_half_out = has_half_out

    def forward(self, x):
        half_out = self.conv3x3(x)
        out = self.conv1x1(half_out)
        if self.stride == 1 and (self.channel_in == self.channel_out):
            out = out + x
        if self.has_half_out:
            return half_out, out
        else:
            return out


def Pool(x, trans, dim=1):
    """Upsample a mesh

    Args:
        x (tensor): input tensor, BxNxD
        trans (tuple): upsample indices and valus
        dim (int, optional): upsample axis. Defaults to 1.

    Returns:
        tensor: upsampled tensor, BxN"xD
    """
    row, col, value = trans[0].to(x.device), trans[1].to(x.device), trans[2].to(x.device)
    value = value.unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out2 = torch.zeros(x.size(0), row.size(0) // 3, x.size(-1)).to(x.device)
    idx = row.unsqueeze(0).unsqueeze(-1).expand_as(out)
    out2 = torch.scatter_add(out2, dim, idx, out)
    return out2


class SpiralDeblock(nn.Module):

    def __init__(self, in_channels, out_channels, indices, meshconv=SpiralConv):
        """Init a spiral conv block

        Args:
            in_channels (int): input feature dim
            out_channels (int): output feature dim
            indices (tensor): neighbourhood of each hand vertex
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(SpiralDeblock, self).__init__()
        self.conv = meshconv(in_channels, out_channels, indices)
        self.relu = nn.ReLU(inplace=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = self.relu(self.conv(out))
        return out


class SIMA_GlobRotReg(nn.Module):

    def __init__(self):
        super(SIMA_GlobRotReg, self).__init__()
        self.self_attn_1 = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )

        self.self_attn_2 = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )
        
        self.crs_attn = MultiHeadAttention(
            n_head=8, 
            d_model=1024, 
            d_k=512, 
            d_v=512, 
            dropout=0.1
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block_1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=True), 
            nn.Linear(32, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, 6))


    def forward(self, j_x, r_x):
        B, C = j_x.size(0), j_x.size(1)

        j_x = j_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])
        r_x = r_x.view(B, C, -1).permute(0, 2, 1)  # torch.Size([B, 16, 1024])

        j_x_slf, _ = self.self_attn_1(j_x, j_x, j_x)  # torch.Size([B, 16, 1024])
        r_x_slf, _ = self.self_attn_2(r_x, r_x, r_x)  # torch.Size([B, 16, 1024])

        x, _ = self.crs_attn(j_x_slf, r_x_slf, r_x_slf)  # torch.Size([B, 16, 1024])

        x = x.permute(0, 2, 1)  # torch.Size([B, 1024, 16])

        rot_reg_latent = x.view(B, C, 4, 4)  # torch.Size([B, 1024, 4, 4])

        x = self.conv_block(x)  # (B, C, HW)

        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)

        pred_rot = self.fc_block_1(x)

        return pred_rot