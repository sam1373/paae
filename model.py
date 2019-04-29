import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim

from modules import *

LEARNING_RATE = 0.0002


#generate latent codes
class Distr_Network(nn.Module):

    def __init__(self, p_dim=128, p0_dim=128, weight_decay=0.):

        super(Distr_Network, self).__init__()

        self.p_dim = p_dim
        self.weight_decay = weight_decay

        self.fc_1 = LinearUnit(p_dim, p0_dim)

        self.fc_2 = LinearUnit(p0_dim, p0_dim)

        self.fc_3 = LinearUnit(p0_dim, p0_dim)

        self.fc_4 = LinearUnit(p0_dim, p_dim, nonlinearity=None)
        

        self.cuda()

        self.optimizer = optim.Adam(self.parameters(), LEARNING_RATE, weight_decay=weight_decay)
        #weight decay for some regularization


    def forward(self, x0):
        #input_shape: bs, p_dim

        x = self.fc_1(x0)

        x = self.fc_2(x)

        x = self.fc_3(x)

        x = self.fc_4(x) + x0

        x = nn.Tanh()(x)
        #due to residual connection, the output should be close to input when weight decay is high

        return x

#encode real samples into latent codes
class Encoder_Network(nn.Module):

    def __init__(self, p_dim=128, p0_dim=128, conv_dim=128, in_channels=3, step=64, in_size=64):

        super(Encoder_Network, self).__init__()

        self.p_dim = p_dim
        self.conv_dim = conv_dim
        self.in_channels = in_channels
        self.in_size = in_size
        self.step = step

        self.final_conv_size = in_size // 2
        
        self.conv_z = nn.Sequential(
                ConvUnit(self.in_channels, step // 4, 3, 1, 1),
                ConvUnit(step // 4, step, 3, 2, 1)
                )
        self.conv_z_fc = nn.Sequential(LinearUnit(step * (self.final_conv_size ** 2), self.conv_dim * 2),
                LinearUnit(self.conv_dim * 2, self.conv_dim))

        self.fc_1 = LinearUnit(conv_dim, p0_dim)

        self.fc_2 = LinearUnit(p0_dim, p0_dim)

        self.fc_3 = LinearUnit(p0_dim, p_dim, nonlinearity=nn.Tanh())

        self.cuda()

        self.optimizer = optim.Adam(self.parameters(),LEARNING_RATE)


    def encode_conv(self, x):
        x = x.view(-1, self.in_channels, self.in_size, self.in_size)
        #x, _ = self.pix_attn_z(x)
        x = self.conv_z(x)
        #print(x.shape)
        x = x.view(-1, self.step * (self.final_conv_size ** 2))
        x = self.conv_z_fc(x)
        #x = x.view(-1, self.frames, self.conv_dim)
        return x

    def forward(self, x):

        x_conv = self.encode_conv(x)

        #z_0 = z_0.view(-1, self.z_dim_seq * self.frames)

        z = self.fc_1(x_conv)

        z = self.fc_2(z)

        z = self.fc_3(z)
        #z_logvar = self.fc_z_logvar(z_0)

        #z = self.reparameterize(z_mean, z_logvar)

        return z



#decode latent codes into sequences of images
class Decoder_Network(nn.Module):

    def __init__(self, p_dim=128, p0_dim=128, conv_dim=128, in_channels=3, step=64, in_size=64):

        super(Decoder_Network, self).__init__()

        self.p_dim = p_dim
        self.p0_dim = p0_dim
        self.conv_dim = conv_dim
        self.in_channels = in_channels
        self.in_size = in_size
        self.step = step

        self.final_conv_size = in_size // 2

        self.deconv_fc = nn.Sequential(LinearUnit(self.p0_dim, self.conv_dim * 2),
                LinearUnit(self.conv_dim * 2, step * (self.final_conv_size ** 2)))
        self.deconv = nn.Sequential(
                ConvUnitTranspose(step, step // 4, 3, 2, 1, 1),
                ConvUnitTranspose(step // 4, self.in_channels, 3, 1, 1, 0, nonlinearity=nn.Sigmoid()))

        self.fc_1 = LinearUnit(p_dim, p0_dim)

        self.fc_2 = LinearUnit(p0_dim, p0_dim)

        self.fc_3 = LinearUnit(p0_dim, p0_dim)

        self.cuda()

        self.optimizer = optim.Adam(self.parameters(),LEARNING_RATE)

    def decode_conv(self, zf):
        x = self.deconv_fc(zf)
        x = x.view(-1, self.step, self.final_conv_size, self.final_conv_size)
        x = self.deconv(x)
        #print(x.shape)
        return x

    def decode(self, z):

        z = self.fc_1(z)

        z = self.fc_2(z)

        z = self.fc_3(z)

        x_recon = self.decode_conv(z)

        return x_recon

    def forward(self, z):

        x_recon = self.decode(z)#.view(-1, self.frames, self.in_channels, self.in_size, self.in_size)

        return x_recon


#discriminate between encoded samples(0) and generated codes(1)
class Discr_Network(nn.Module):

    def __init__(self, p_dim=128, p0_dim=128, mha_stack_size=5):

        super(Discr_Network, self).__init__()

        self.p_dim = p_dim

        self.fc_1 = LinearUnit(p_dim, p0_dim)

        self.fc_2 = LinearUnit(p0_dim, p0_dim // 2)

        self.fc_3 = LinearUnit(p0_dim // 2, p0_dim // 4)

        self.fc_4 = LinearUnit(p0_dim // 4, 1, nonlinearity=nn.Tanh())

        self.cuda()

        self.optimizer = optim.Adam(self.parameters(),LEARNING_RATE)

    def forward(self, x):

        x = self.fc_1(x)

        x = self.fc_2(x)

        x = self.fc_3(x)

        x = self.fc_4(x)

        return x



class PAAE(nn.Module):

    def __init__(self, p_dim=128, in_size=64, in_channels=3, weight_decay=0):

        super(PAAE, self).__init__()

        self.p_dim = p_dim
        self.in_channels = in_channels
        self.in_size = in_size

        self.distr = Distr_Network(p_dim=p_dim, weight_decay=weight_decay)

        self.enc = Encoder_Network(p_dim=p_dim, in_size=in_size, in_channels=in_channels)

        self.dec = Decoder_Network(p_dim=p_dim, in_size=in_size, in_channels=in_channels)

        self.discr = Discr_Network(p_dim=p_dim)

    def forward(self, x, get_score=True):

        z = self.enc(x)

        x_recon = self.dec(z)

        if get_score == False:
            return x_recon

        D_score = self.discr(z)

        return x_recon, D_score

    def forward_score(self, x):

        z = self.enc(x)

        D_score = self.discr(z)

        return D_score

    def gen_codes(self, batch_size=32):

        inputs_r = torch.randn((batch_size, self.p_dim)).cuda()

        z = self.distr(inputs_r)

        D_score = self.discr(z)

        return z, D_score

    def gen_img(self, batch_size=32, get_score=False, mult=1.):


        inputs_r = torch.randn((batch_size, self.p_dim)).cuda() * mult

        z = self.distr(inputs_r)

        x_gen = self.dec(z)


        if get_score == False:

            return x_gen

        D_score = self.discr(z)

        return x_gen, D_score

    def zero_grad_all(self):

        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.25)

        self.distr.optimizer.zero_grad()
        self.discr.optimizer.zero_grad()
        self.enc.optimizer.zero_grad()
        self.dec.optimizer.zero_grad()

    def enc_dec_step(self):

        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.25)

        self.enc.optimizer.step()
        self.dec.optimizer.step()

    #def gen_codes_step(self):
    #    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.25)

    #    self.discr.optimizer.step()
    #    self.discr.optimizer.step()

    def encode(self, x):

        return self.enc(x)

    def decode(self, z):

        return self.dec(f, z)



import matplotlib.pyplot as plt

def plot(x):

    x0 = x.detach().cpu()

    plt.scatter(x0 [:, 0], x0 [:, 1])
    plt.show()


if __name__ == '__main__':

    batch_size = 32
    channels = 3

    p_dim = 2

    model = PAAE(p_dim = p_dim, in_channels=3)

    inputs_x = torch.randn((batch_size, channels, 64, 64)).cuda() + 0.5
    inputs_r = torch.randn((batch_size, p_dim)).cuda()

    #print(inputs_r.shape)

    out, sc = model.gen_seq(get_score=True)

    print(out.shape)
    print(sc.shape)

    out, sc = model(inputs_x)

    print(out.shape)
    print(sc.shape)

    enc = model.encode(inputs_x)
    print(enc.shape)

    plot(enc)

    z, sc = model.gen_codes()

    plot(z)

    plot(inputs_r)


    #plot(o1_f)

    """
    

    model = DisentangledVAE_Attention(frames=frames)

    inputs = torch.randn((batch_size * frames, channels, 64, 64)).to(model.device) * 0.2 - 0.05
    inputs = torch.clamp(inputs, 0., 1.)

    print(inputs.min(), inputs.max(), inputs.mean(), inputs.std())

    _, _, _, _, _, _, out = model.forward(inputs)

    print(out.shape)

    print(out.min(), out.max(), out.mean(), out.std())
    """

    """
    conv_x = model.encode_frames(inputs)

    print(conv_x.shape)

    mha_8 = multihead_attention(2048, 8, residual=False)

    z = mha_8(conv_x, conv_x, conv_x)

    print(z.shape)
    """


    #x = torch.randn((frames, 3, 64, 64))

    #m = pixel_attention(3, 64)

    #x = m(x)

