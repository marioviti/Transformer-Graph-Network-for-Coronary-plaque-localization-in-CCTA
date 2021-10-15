
import torch as th
import torch.nn.functional as F
import torch.nn as nn
from .Modules import Encoder, Decoder, QKV, MLP, ScaledDotProdAtt, MultiheadAttention
import math
import numpy as np
        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, pos):
        # len seq, batch_size, channels
        seq_size, batch_size = pos.shape
        pos_i = th.sin(pos.unsqueeze(-1) * self.div_term)
        pos_ip1 = th.cos(pos.unsqueeze(-1) * self.div_term)
        pe = th.cat([pos_i, pos_ip1], axis=-1)
        return pe

class AttentionEncoder(nn.Module):
    
    def __init__(self, input_channels, hidden_channels, num_heads, n_layers):
        super().__init__()
        self.attention_layers = nn.ModuleList([nn.TransformerEncoderLayer(input_channels, num_heads, dim_feedforward=hidden_channels)]*n_layers)
        self.positional_encoder = PositionalEncoding(input_channels)
        
    def forward(self, x, pos):
        batch_size, sequence_size, d_model = x.shape
        batch_size0, sequence_size0 = pos.shape
        assert batch_size0 == batch_size and sequence_size == sequence_size0, 'pe and x must have same batch and sequence size'
        x = x.transpose(0,1) # now it is sequence, batch_size, d_model
        pos = pos.transpose(0,1)
        x = x + self.positional_encoder(pos)
        for att in self.attention_layers:
            x = att(x) + x
        x = x.transpose(0,1) # now it is batch_size, sequence, d_model
        return x

class CNN(nn.Module):
    
    def __init__(self, encoder_hparams, mlp_hparams, input_channels, output_channels):
        super().__init__()
        encoder = Encoder(input_channels=input_channels, **encoder_hparams )
        embedd_channels = encoder.outputs_channels[-1]
        gap = nn.AdaptiveAvgPool3d(1)
        ln0 = nn.LayerNorm(embedd_channels)
        mlp = MLP(input_channels=embedd_channels, output_channels=output_channels, **mlp_hparams)
        
        self.encoder = encoder
        self.gap = gap
        self.ln0 = ln0
        self.mlp = mlp
                
    def project(self,x):
        a0 = self.gap(self.encoder(x)[-1])[...,0,0,0]
        return a0
    
    def predict(self,a0):
        y0 = self.mlp(a0)
        return y0
            
    def forward(self,x):
        batch_size, input_channels, depth, height, width = x.shape
        a0 = self.project(x)
        a0 = self.ln0(a0)
        y0 = self.predict(a0)
        return y0
    
class CNNGNN(nn.Module):
    
    def __init__(self, input_channels, output_channels, encoder_hparams, mlp_hparams):
        super().__init__()
        encoder = Encoder(input_channels=input_channels, **encoder_hparams )
        embedd_channels = encoder.outputs_channels[-1]
        gap = nn.AdaptiveAvgPool3d(1)
        g_mlp = nn.Sequential( nn.Linear(embedd_channels, embedd_channels), nn.LeakyReLU(inplace=True) )
        mlp = MLP(input_channels=embedd_channels, output_channels=output_channels, **mlp_hparams)
        
        self.encoder = encoder
        self.gap = gap
        self.g_mlp = g_mlp
        self.mlp = mlp
        
    def project(self,x):
        a0 = self.gap(self.encoder(x)[-1])[:,:]
        return a0        
            
    def forward(self,x,pos):
        batch_size, neighbourhood, input_channels, depth, height, width = x.shape
        x = x.view(batch_size*neighbourhood, input_channels, depth, height, width)
        a0 = self.project(x)
        a0 = a0.view(batch_size, neighbourhood, -1)
        v = a0[:,0,:]
        pos = pos[...,np.newaxis].clip(-30,30)/30
        N = (a0[:,1:,:]*(1/(pos[:,1:,:]+1e-19))).sum(axis=1)
        a1 = self.g_mlp(N) + v
        y = self.mlp(v)
        return y

class CNNATT(nn.Module):
    
    def __init__(self, input_channels, output_channels, encoder_hparams, att_encode_hparams, mlp_hparams):
        super().__init__()
        encoder = Encoder(input_channels=input_channels, **encoder_hparams )
        embedd_channels = encoder.outputs_channels[-1]
        gap = nn.AdaptiveAvgPool3d(1)
        att_enc = AttentionEncoder(input_channels=embedd_channels, **att_encode_hparams)
        mlp = MLP(input_channels=embedd_channels, output_channels=output_channels, **mlp_hparams)
        
        self.encoder = encoder
        self.gap = gap
        self.att_enc = att_enc
        self.mlp = mlp
        
    def project(self,x):
        a0 = self.gap(self.encoder(x)[-1])[:,:]
        return a0        
            
    def forward(self,x,pos):
        batch_size, neighbourhood, input_channels, depth, height, width = x.shape
        x = x.view(batch_size*neighbourhood, input_channels, depth, height, width)
        a0 = self.project(x)
        a0 = a0.view(batch_size, neighbourhood, -1)
        z0 = self.att_enc(a0,pos)
        z0 = z0.reshape(batch_size*neighbourhood, -1)
        y = self.mlp(z0)
        y = y.view(batch_size, neighbourhood, -1)
        return y
   
