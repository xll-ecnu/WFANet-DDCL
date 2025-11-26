import torch
import torch.nn as nn
from einops import rearrange

# Additional Scripts
from .WFA_encoder import WFANet_Encoder


class YellowBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.downsample = in_channels != out_channels #True or False

        self.conv_block = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding),
                                        normalization(out_channels),
                                        nn.LeakyReLU(negative_slope=.01, inplace=True),
                                        nn.Conv2d(out_channels, out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding),
                                        normalization(out_channels))

        if self.downsample:
            self.conv_block2 = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                       kernel_size=1, stride=1, padding=0),
                                             normalization(out_channels))

        self.leaky_relu = nn.LeakyReLU(negative_slope=.01, inplace=True)

    def forward(self, x):
        res = x

        conv_output = self.conv_block(x)

        if self.downsample:
            res = self.conv_block2(res)
            #print('res.shape af ds',res.shape)

        conv_output += res
        x = self.leaky_relu(conv_output)
        return x
class SingleBlueBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization):
        super().__init__()

        self.conv_block = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels,
                                                           kernel_size=2, stride=2, padding=0, bias=False),

                                        # Not exactly yellow block but it is
                                        YellowBlock(in_channels=out_channels,
                                                    out_channels=out_channels,
                                                    normalization=normalization))

    def forward(self, x):
        x = self.conv_block(x)
        return x


class BlueBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization, layer_num):
        super().__init__()
        print(in_channels, out_channels)
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels,
                                                 kernel_size=2, stride=2, padding=0, bias=False)

        layers = []
        for _ in range(layer_num):
            layers.append(SingleBlueBlock(in_channels=out_channels,
                                          out_channels=out_channels,
                                          normalization=normalization))

        self.blocks = nn.ModuleList(layers)

    def forward(self, x):

        #print('`````x.shape````',x.shape)

        x = self.transpose_conv(x)
        i = 0
        for block in self.blocks:
            i += 1
            x = block(x)
            #print('x.shape af {} blueblock'.format(i), x.shape)

        return x


class GreenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.deconv_block = nn.ConvTranspose2d(in_channels, out_channels,
                                               kernel_size=2, stride=2, padding=0, bias=False)

    def forward(self, x):
        x = self.deconv_block(x)
        return x


class WFANet(nn.Module):
    def __init__(self,  in_channels, base_filter = 16, class_num = 1,
                  embedding_dim = 256, block_num = 12,
                  mlp_dim = 1024, z_idx_list = [3, 6, 9, 12]):
        super().__init__()



        self.WFANet_Encoder = WFANet_Encoder(
                                         embedding_dim=embedding_dim,
                                         block_num=block_num,
                                         mlp_dim=mlp_dim,
                                         z_idx_list=z_idx_list)
        # self.z0_yellow_block = YellowBlock(in_channels=in_channels,
        #                                    out_channels=base_filter,
        #                                    normalization=nn.InstanceNorm2d)

        self.z3_blue_block = BlueBlock(in_channels=embedding_dim,
                                       out_channels=base_filter * 2,
                                       normalization=nn.InstanceNorm2d,
                                       layer_num=2)

        self.z6_blue_block = BlueBlock(in_channels=embedding_dim,
                                       out_channels=base_filter * 4,
                                       normalization=nn.InstanceNorm2d,
                                       layer_num=1)

        self.z9_blue_block = BlueBlock(in_channels=embedding_dim,
                                       out_channels=base_filter * 8,
                                       normalization=nn.InstanceNorm2d,
                                       layer_num=0)

        self.z3_green_block = GreenBlock(in_channels=base_filter * 2,
                                         out_channels=base_filter)

        self.z6_green_block = GreenBlock(in_channels=base_filter * 4,
                                         out_channels=base_filter * 2)

        self.z9_green_block = GreenBlock(in_channels=base_filter * 8,
                                         out_channels=base_filter * 4)

        self.z12_green_block = GreenBlock(in_channels=embedding_dim,
                                         out_channels=base_filter * 8)

        self.z3_yellow_block = YellowBlock(in_channels=base_filter * 2 * 2,
                                           out_channels=base_filter * 2,
                                           normalization=nn.InstanceNorm2d)

        self.z6_yellow_block = YellowBlock(in_channels=base_filter * 4 * 2,
                                           out_channels=base_filter * 4,
                                           normalization=nn.InstanceNorm2d)

        self.z9_yellow_block = YellowBlock(in_channels=base_filter * 8 ,
                                          out_channels=base_filter * 4,
                                          normalization=nn.InstanceNorm2d)

        self.output_block = nn.Sequential(YellowBlock(in_channels=base_filter ,
                                                      out_channels=base_filter,
                                                      normalization=nn.InstanceNorm2d),
                                          nn.Conv2d(base_filter, class_num, kernel_size=1, stride=1))

    def forward(self, x):
        #print('Decoding begin')
        res_x = x
        z_embedding = self.WFANet_Encoder(x)
        #print('self.patch_dim',self.patch_dim)

        z3, z6, z9, z12 = [z for z in z_embedding]
        z3 = self.z3_blue_block(z3)
        #print('z3.shape af z3_blue_block', z3.shape)
        z6 = self.z6_blue_block(z6)
        #print('z6.shape af z6_blue_block', z6.shape)
        z9 = self.z9_blue_block(z9)
        #print('z9.shape af z9_blue_block', z9.shape)

        # green and yellow blocks operations and their concatenations
        z12 = self.z12_green_block(z12)

        #print('z12.shape af z12_green_block', z12.shape)
        y = torch.cat([z12, z9], dim=1)
        y = self.z9_yellow_block(z9)
        #print('y.shape af z9_yellow_block', y.shape)
        y = self.z9_green_block(z9)
        #print('y.shape af z9_green_block', y.shape)
        y = torch.cat([y, z6], dim=1)
        #print('y.shape af cat([y, z6]', y.shape)
        y = self.z6_yellow_block(y)
        #print('y.shape af z6_yellow_block', y.shape)
        y = self.z6_green_block(y)
        #print('y.shape af z6_green_block', y.shape)
        y = torch.cat([y, z3], dim=1)
        #print('y.shape af cat([y, z3]', y.shape)
        y = self.z3_yellow_block(y)
        #print('y.shape af z3_yellow_block', y.shape)
        y = self.z3_green_block(y)
        #print('y.shape af z3_green_block', y.shape)
        #y = torch.cat([y, z0], dim=1)
        #print('y.shape af cat([y, z0]', y.shape)
        y = self.output_block(y)
        #print('y.shape af output_block', y.shape)
        y_out = torch.add(y , res_x)
        return y_out


if __name__ == '__main__':
    nn = WFANet(
               in_channels=1,
               base_filter=16,
               class_num=1,
               embedding_dim=256,
               block_num=12,
               mlp_dim=1024,
               z_idx_list=[3, 6, 9, 12])
    a = torch.rand(1, 1, 256, 256)
    r = nn(a)
    #print('r.shape',r.shape)
