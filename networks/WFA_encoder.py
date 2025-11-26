
import torch.nn as nn

from einops import rearrange, repeat
from .utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)
class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim, is_wave = False):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )
        self.is_wave = is_wave

    def forward(self, x):
        #print('x.shape bf mlp_layers',x.shape)
        x = self.mlp_layers(x)
        if self.is_wave:
            x = rearrange(x, 'b x y d -> b d x y'
                                    )
        #print('x.shape af mlp_layers', x.shape)
        return x

class WFA(nn.Module):
    def __init__(self, plans = 256, wavelet_type = 'haar',channel = 1024, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super().__init__()
        self.dwt = DWT_2D(wavename=wavelet_type)
        #self.channel_att = channel_attention(channel = channel)
        self.linear_1 = nn.Linear(channel, channel // 2).cuda()
        self.linear_2 = nn.Linear(channel // 2, channel).cuda()
        self.avg_pool = nn.functional.adaptive_avg_pool2d
        self.RELU = torch.nn.functional.relu
        self.SIGMOID = torch.nn.functional.sigmoid
        self.WDCC = WD_and_CC()
        self.idwt = IDWT_2D(wavename=wavelet_type)
        self.conv_1 = conv(plans, plans, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=stride).cuda()
        self.conv_2 = conv(plans, plans, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                           stride=stride).cuda()
        self.conv_3 = conv(plans, plans, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                           stride=stride).cuda()
        self.conv_4 = conv(plans, plans, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                           stride=stride).cuda()
    def forward(self, x):
        #print('WFA input shape', x.shape) # (B, D, feature_size，feature_size)
        x = x.to(device)
        #print('x.device', x.device)
        f_k1 = self.conv_1(x)
        f_k2 = self.conv_2(x)
        f_k3 = self.conv_3(x)
        f_k4 = self.conv_4(x)
        #print('f_k1 shape', f_k1.shape)
        #x = self.linear1(x)
        # x = rearrange(x, 'b x y d -> b d x y'
        #                         )
        wdccf_in = self.WDCC(x)
        wdccf_1 = self.WDCC(f_k1)
        wdccf_2 = self.WDCC(f_k2)
        wdccf_3 = self.WDCC(f_k3)
        wdccf_4 = self.WDCC(f_k4)
        #print('wdccf_in shape', wdccf_in.shape)  # (B, D, feature_size，feature_size)
        #print('wdccf_in.device', wdccf_in.device)
        
        v_in = self.avg_pool(wdccf_in, (1, 1))
        v_1 = self.avg_pool(wdccf_1, (1, 1))
        v_2 = self.avg_pool(wdccf_2, (1, 1))
        v_3 = self.avg_pool(wdccf_3, (1, 1))
        v_4 = self.avg_pool(wdccf_4, (1, 1))
        v_fuse = v_in + v_1 + v_2 + v_3 + v_4
        #print('v_fuse shape', v_fuse.shape)  # (B, D, feature_size，feature_size)
        #print('v_fuse.device', v_fuse.device)
        v_fuse = rearrange(v_fuse, 'b d x y -> b x y d'
                                )
        v_relu = self.RELU(self.linear_1(v_fuse))
        v_calibrated = self.SIGMOID(self.linear_2(v_relu))
        v_calibrated = rearrange(v_calibrated, 'b x y d -> b d x y'
                           )
        f_WFA = torch.multiply(wdccf_in,v_calibrated)
        #print('f_WFA shape', f_WFA.shape)  # (B, D, feature_size，feature_size)
        #print('f_WFA.device', f_WFA.device)

        LL, LH, HL, HH = tuple(rearrange(f_WFA, 'b (k d) x y -> k b d x y', k=4))

        f_out = self.idwt(LL, LH, HL, HH)
        f_out = (torch.tensor(f_out)).to(device)
        #print('f_out.shape', f_out.shape)
        #print('f_out.device', f_out.device)
        return f_out

class WD_and_CC(nn.Module):
    def __init__(self, wavename = 'haar'):
        super().__init__()
        self.dwt = DWT_2D(wavename = wavename)
    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        output = torch.cat((LL, LH, HL, HH), dim=1)
        return output
class WFAE(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.WFA = WFA()
        self.mlp = MLP(embedding_dim, mlp_dim,is_wave=True)

        self.layer_norm1 = nn.LayerNorm(embedding_dim).to(device)
        self.layer_norm2 = nn.LayerNorm(embedding_dim).to(device)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        #print('WFAE input shape', x.shape)
        x = x.to(device)
        #print('x.device', x.device)
        _x = self.WFA(x)

        #print('x.shape af WFA', _x.shape)
        _x = self.dropout(_x)
        x = x + _x

        x = rearrange(x, 'b d x y -> b x y d')
        x = self.layer_norm1(x)
        #x = rearrange(x, 'b x y d -> b d x y')
        _x = self.mlp(x)
        _x = rearrange(_x, 'b d x y -> b x y d')
        x = x + _x
        x = self.layer_norm2(x)
        x = rearrange(x, 'b x y d -> b d x y')
        #print('WaveletEncoder output shape', x.shape)
        return x

class WFAE_bottleneck(nn.Module):
    def __init__(self, embedding_dim,  mlp_dim, block_num, z_idx_list):
        super().__init__()
        self.z_idx_list = z_idx_list
        layers = []
        for _ in range(block_num):
            layers.append(WFAE(embedding_dim, mlp_dim))


        self.layer_blocks = nn.ModuleList(layers)

    def forward(self, x):

        z_outputs = []
        for idx, layer_block in enumerate(self.layer_blocks, start=1):
            x = layer_block(x)
            if idx in self.z_idx_list:
                #print('x.shape in {}'.format(idx),x.shape)
                z_outputs.append(x)

        return z_outputs

class WFANet_Encoder(nn.Module):
    def __init__(self,  embedding_dim=256, mlp_dim=1024,
                 block_num = 12,  z_idx_list= [3, 6, 9, 12]):
        super().__init__()
        ngf = 64
        use_bias = False
        norm_layer = nn.BatchNorm2d
        ############################################################################################
        # Layer1-Encoder1
        model = [#nn.ReflectionPad2d(3),
                 nn.Conv2d(1, ngf, kernel_size=3,  stride=2, padding=1,
                           bias=use_bias, ),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        setattr(self, 'encoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer2-Encoder2
        i = 0
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=5,
                           stride=2, padding=2, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer3-Encoder3
        i = 1
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=7,
                           stride=2, padding=3, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_3', nn.Sequential(*model))
        ############################################################################################
        # Layer4-Encoder4
        i = 1
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=9,
                           stride=2, padding=4, bias=use_bias, dilation=1),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_4', nn.Sequential(*model))
        self.WFANet_Encoder_encoder = WFAE_bottleneck(embedding_dim, mlp_dim, block_num, z_idx_list)

    def forward(self, x):
        #print('x.shape bf encoder_1', x.shape)
        x = self.encoder_1(x)
        #print('x.shape af encoder_1', x.shape)
        x = self.encoder_2(x)
        #print('x.shape af encoder_2', x.shape)
        x = self.encoder_3(x)
        #print('x.shape af encoder_3', x.shape)
        x = self.encoder_4(x)
        #print('x.shape af encoder_4', x.shape)
        z_outputs = self.WFANet_Encoder_encoder(x)
        #print('z_outputs.shape af encoder_3', z_outputs.size)
        return z_outputs

if __name__ == '__main__':

    trans = WFANet_Encoder(
                       embedding_dim=256,
                       mlp_dim=1024,
                       block_num=12,
                       z_idx_list=[3, 6, 9, 12]
                      )

    z3, z6, z9, z12 = trans(torch.rand(2, 1, 256, 256))
    print(z3.shape)
    print(z6.shape)
    print(z9.shape)
    print(z12.shape)