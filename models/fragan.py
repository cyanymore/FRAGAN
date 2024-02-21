import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchstat import stat


class ConvLReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kSize, stride, padding):
        super(ConvLReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kSize, stride, padding, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ConvINLReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kSize, stride, padding):
        super(ConvINLReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kSize, stride, padding, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ConvINLReLUAdd(nn.Module):
    def __init__(self, in_ch, out_ch, kSize, stride, padding):
        super(ConvINLReLUAdd, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kSize, stride, padding, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x) + x


class DConvINLReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kSize, stride, padding, dilation):
        super(DConvINLReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kSize, stride, padding, dilation, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# Residual Feature Refinement Module
class RFRM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RFRM, self).__init__()
        self.br1_0 = ConvINLReLU(in_ch, out_ch, 1, 1, 0)
        self.br1_1 = ConvINLReLU(out_ch, out_ch, 1, 1, 0)

        self.br2_0 = ConvINLReLU(in_ch, out_ch, 1, 1, 0)
        self.br2_1 = ConvINLReLUAdd(out_ch, out_ch, 3, 1, 1)

        self.br3_0 = ConvINLReLU(in_ch, out_ch, 1, 1, 0)
        self.br3_1 = ConvINLReLUAdd(out_ch, out_ch, 3, 1, 1)
        self.br3_2 = ConvINLReLUAdd(out_ch, out_ch, 3, 1, 1)

        self.br4_0 = ConvINLReLU(in_ch, out_ch, 1, 1, 0)
        self.br4_1 = ConvINLReLUAdd(out_ch, out_ch, 3, 1, 1)
        self.br4_2 = ConvINLReLUAdd(out_ch, out_ch, 3, 1, 1)
        self.br4_3 = ConvINLReLUAdd(out_ch, out_ch, 3, 1, 1)

        self.conv1x1 = nn.Conv2d(out_ch * 4, out_ch, 1, 1, 0, bias=False)

    def forward(self, x):
        br0 = x

        br1 = self.br1_0(x)
        br1 = self.br1_1(br1)

        br2 = self.br2_0(x)
        br2 = self.br2_1(br2)

        br3 = self.br3_0(x)
        br3 = self.br3_1(br3)
        br3 = self.br3_2(br3)

        br4 = self.br4_0(x)
        br4 = self.br4_1(br4)
        br4 = self.br4_2(br4)
        br4 = self.br4_3(br4)

        br = torch.cat([br1, br2, br3, br4], 1)
        out = self.conv1x1(br)

        return out


# Feature Adaptive Module
class FAM(nn.Module):
    """AtrousSpatialPyramidPooling module"""

    def __init__(self, in_ch, out_ch):
        super(FAM, self).__init__()
        mid_ch = int(in_ch / 4)
        self.conv1 = ConvLReLU(in_ch, mid_ch, 1, 1, 0)
        self.conv2 = ConvLReLU(in_ch, mid_ch, 1, 1, 0)
        self.conv3 = ConvLReLU(in_ch, mid_ch, 1, 1, 0)
        self.convout = ConvLReLU(in_ch, out_ch, 1, 1, 0)
        self.convrate3 = DConvINLReLU(in_ch, mid_ch, 3, 1, 3, dilation=3)
        self.convrate6 = DConvINLReLU(in_ch, mid_ch, 3, 1, 6, dilation=6)
        self.convrate9 = DConvINLReLU(in_ch, mid_ch, 3, 1, 9, dilation=9)

        self.out = ConvLReLU(mid_ch * 10, out_ch, 1, 1, 0)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 4)), size)

        atrous_block3 = self.convrate3(x)
        atrous_block6 = self.convrate6(x)
        atrous_block9 = self.convrate9(x)
        out = self.convout(self.pool(x, 1))

        x = torch.cat([x, feat1, feat2, feat3, atrous_block3, atrous_block6, atrous_block9], dim=1)
        x = self.out(x) + out

        return x


# Trinity Attention Module
class TAM(nn.Module):
    def __init__(self, in_ch, in_size):
        super(TAM, self).__init__()
        self.in_ch = in_ch
        self.mid_ch = self.in_ch // 2
        self.in_size = in_size
        self.mid_size = self.in_size // 2

        self.conv1x1_left1 = nn.Conv2d(self.in_ch, self.mid_ch, 1, 1, 0, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_left2 = nn.Conv2d(self.in_ch, self.mid_ch, 1, 1, 0, bias=False)
        self.softmax_left = nn.Softmax(dim=2)

        self.conv1x1_right1 = nn.Conv2d(self.in_size, self.mid_size, 1, 1, 0, bias=False)
        self.conv1x1_right2 = nn.Conv2d(self.in_size, self.mid_size, 1, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def spatial_att(self, x):
        x1 = self.conv1x1_left1(x)
        b1, c1, h1, w1 = x1.size()
        x2 = self.avg_pool(x1)
        b2, c2, h2, w2 = x2.size()
        x2 = x2.view(b1, c2, h2 * w2).permute(0, 2, 1)
        x3 = self.conv1x1_left2(x).view(b1, c1, h1 * w1)
        x4 = torch.matmul(x2, x3)
        x5 = self.softmax_left(x4)
        x5 = x5.view(b1, 1, h1, w1)
        x6 = self.sigmoid(x5)
        out = x * x6
        return out

    def channel_att(self, x):
        x1 = self.conv1x1_right1(x)
        b1, c1, h1, w1 = x1.size()
        x2 = self.avg_pool(x1)
        b2, c2, h2, w2 = x2.size()
        x2 = x2.view(b2, c2, h2 * w2).permute(0, 2, 1)

        x3 = self.conv1x1_right2(x).view(b1, c1, h1 * w1)
        x4 = torch.matmul(x2, x3)
        x5 = self.softmax_left(x4)
        x5 = x5.view(b1, 1, h1, w1)
        x6 = self.sigmoid(x5)
        out = x * x6
        return out

    def forward(self, x):
        sp = self.spatial_att(x)
        x1 = x.permute(0, 2, 1, 3).contiguous()
        ch_h = self.channel_att(x1)
        ch_h = ch_h.permute(0, 2, 1, 3).contiguous()

        x2 = x.permute(0, 3, 2, 1).contiguous()
        ch_w = self.channel_att(x2)
        ch_w = ch_w.permute(0, 3, 2, 1).contiguous()
        out = sp + ch_h + ch_w
        return out


# Concatenate Block1
class USM(nn.Module):
    def __init__(self, in_ch, out_ch, is_deconv):
        super(USM, self).__init__()
        if is_deconv:
            self.conv = ConvLReLU(in_ch, out_ch, 3, 1, 1)
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = ConvLReLU(in_ch, out_ch, 3, 1, 1)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, x0):
        x1 = self.up(x)
        x2 = torch.cat([x0, x1], 1)
        return self.conv(x2)


# Concatenate Block2
class USM_CRM(nn.Module):
    def __init__(self, in_ch, out_ch, in_size, is_deconv):
        super(USM_CRM, self).__init__()
        if is_deconv:
            self.conv = ConvLReLU(in_ch, out_ch, 3, 1, 1)
            self.att = TAM(out_ch, in_size)
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = ConvLReLU(in_ch, out_ch, 3, 1, 1)
            self.att = TAM(out_ch, in_size)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, x0):
        # print(x.shape, x0.shape)
        x1 = self.up(x)
        x2 = torch.cat([x0, x1], 1)
        x3 = self.conv(x2)
        return self.att(x3)


class FRAGAN(nn.Module):
    def __init__(self, in_ch, out_ch, is_deconv=True):
        super(FRAGAN, self).__init__()
        mid_ch = [32, 64, 128, 256, 512]
        mid_size = [32, 64, 128, 256]
        channel = 64

        # downsampling
        self.conv00 = self.make_layer(in_ch, mid_ch[0], 2)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = self.make_layer(mid_ch[0], mid_ch[1], 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = self.make_layer(mid_ch[1], mid_ch[2], 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = self.make_layer(mid_ch[2], mid_ch[3], 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = self.make_layer(mid_ch[3], mid_ch[4], 2)

        self.middle = FAM(mid_ch[4], mid_ch[4])

        # upsampling
        self.up_concat01 = USM(mid_ch[1], mid_ch[0], is_deconv)
        self.up_concat11 = USM(mid_ch[2], mid_ch[1], is_deconv)
        self.up_concat21 = USM(mid_ch[3], mid_ch[2], is_deconv)
        self.up_concat31 = USM_CRM(mid_ch[4], mid_ch[3], mid_size[0], is_deconv)

        self.up_concat02 = USM(mid_ch[1], mid_ch[0], is_deconv)
        self.up_concat12 = USM(mid_ch[2], mid_ch[1], is_deconv)
        self.up_concat22 = USM_CRM(mid_ch[3], mid_ch[2], mid_size[1], is_deconv)

        self.up_concat03 = USM(mid_ch[1], mid_ch[0], is_deconv)
        self.up_concat13 = USM_CRM(mid_ch[2], mid_ch[1], mid_size[2], is_deconv)

        self.up_concat04 = USM_CRM(mid_ch[1], mid_ch[0], mid_size[3], is_deconv)

        # decoder block1
        self.db1_pool1 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.db1_conv1 = ConvINLReLU(mid_ch[0], channel, 3, 1, 1)

        self.db1_pool2 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.db1_conv2 = ConvINLReLU(mid_ch[1], channel, 3, 1, 1)

        self.db1_pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.db1_conv3 = ConvINLReLU(mid_ch[2], channel, 3, 1, 1)

        self.db1_conv4 = ConvINLReLU(448, 256, 1, 1, 0)

        # decoder block2
        self.db2_pool1 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.db2_conv1 = ConvINLReLU(mid_ch[0], channel, 3, 1, 1)

        self.db2_pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.db2_conv2 = ConvINLReLU(mid_ch[1], channel, 3, 1, 1)

        self.db2_conv3 = ConvINLReLU(256, 128, 1, 1, 0)

        # decoder block3
        self.db3_pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.db3_conv1 = ConvINLReLU(mid_ch[0], channel, 3, 1, 1)

        self.db3_conv2 = ConvINLReLU(128, 64, 1, 1, 0)

        # final conv
        self.final_1 = nn.Conv2d(mid_ch[0], out_ch, 1, 1, 0)
        self.final_2 = nn.Conv2d(mid_ch[0], out_ch, 1, 1, 0)
        self.final_3 = nn.Conv2d(mid_ch[0], out_ch, 1, 1, 0)
        self.final_4 = nn.Conv2d(mid_ch[0], out_ch, 1, 1, 0)

    def make_layer(self, in_ch, out_ch, block_num):
        layers = []
        layers.append(RFRM(in_ch, out_ch))
        for i in range(1, block_num):
            layers.append(RFRM(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        # column : 0
        X_00 = self.conv00(x)
        maxpool0 = self.maxpool0(X_00)
        X_10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool1(X_10)
        X_20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool2(X_20)
        X_30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool3(X_30)
        X_40 = self.conv40(maxpool3)

        X_40 = self.middle(X_40)

        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)

        h1 = self.db1_conv1(self.db1_pool1(X_00))
        h2 = self.db1_conv2(self.db1_pool2(X_10))
        h3 = self.db1_conv3(self.db1_pool3(X_20))
        X_31 = torch.cat((X_31, h1, h2, h3), 1)
        X_31 = self.db1_conv4(X_31)

        # column : 2
        X_02 = self.up_concat02(X_11, X_01)
        X_12 = self.up_concat12(X_21, X_11)
        X_22 = self.up_concat22(X_31, X_21)

        h4 = self.db2_conv1(self.db2_pool1(X_00))
        h5 = self.db2_conv2(self.db2_pool2(X_10))
        X_22 = torch.cat((X_22, h4, h5), 1)
        X_22 = self.db2_conv3(X_22)

        # column : 3
        X_03 = self.up_concat03(X_12, X_02)
        X_13 = self.up_concat13(X_22, X_12)

        h6 = self.db3_conv1(self.db3_pool1(X_00))
        X_13 = torch.cat((X_13, h6), 1)
        X_13 = self.db3_conv2(X_13)

        # column : 4
        X_04 = self.up_concat04(X_13, X_03)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4
        return F.tanh(final)


if __name__ == "__main__":
    input = torch.Tensor(1, 3, 256, 256).cuda()
    model = FRAGAN(3, 3).cuda()
    model.eval()
    print(model)
    output = model(input)
    summary(model, (3, 256, 256))
    print(output.shape)
