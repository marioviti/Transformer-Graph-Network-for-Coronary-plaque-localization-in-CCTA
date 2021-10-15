import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ScaledDotProdAtt(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self,q,k,v, peM=None):
        d_k = q.size()[-1]
        temperature = math.sqrt(d_k)
        qkt = th.matmul(q,k.transpose(-2,-1))
        attn_logits = qkt/temperature
        attn = F.softmax(attn_logits, dim=-1)
        values = th.matmul(attn+peM,v) if peM is not None else th.matmul(attn,v)
        return values, attn
    
class QKV(nn.Module):
    
    def __init__(self, input_channels, embed_dim, num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scdpa = ScaledDotProdAtt()
        self.qkv_proj = nn.Linear(input_channels, 3*embed_dim)
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        
    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        return q, k, v
    
class MultiheadAttention(nn.Module):
    
    def __init__(self, input_channels, embed_dim, num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scdpa = ScaledDotProdAtt()
        self.qkv_proj = QKV(input_channels, embed_dim, num_heads=num_heads)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, peM=None, return_attention=True):
        batch_size, seq_length, embed_dim = x.size()
        q, k, v = self.qkv_proj(x)
        
        # Determine value outputs
        values, attention = self.scdpa(q, k, v, peM=peM)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, -1)
        o = self.o_proj(values)
        if return_attention:
            return o, attention
        return o

class ResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU(inplace=True), DIMS=3):
        super().__init__()
        self.act = activation
        conv_fun = { 2: nn.Conv2d, 3:nn.Conv3d }
        bn_fun = { 2:nn.BatchNorm2d, 3:nn.BatchNorm3d }
        self.conv_in = conv_fun[DIMS](in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_h = conv_fun[DIMS](out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_out = conv_fun[DIMS](out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn_out  = bn_fun[DIMS](out_channels)
        
    def forward(self,x):
        a0 = self.act(self.conv_in(x))
        a1 = self.act(self.conv_h(a0))
        a2 = self.bn_out(self.conv_out(a1))+a0
        return a2
    
class Encoder(th.nn.Module):
    
    def __init__(self, input_channels, root_channels=16, expansion_factor=2., n_layers=4, activation=nn.LeakyReLU(inplace=True), DIMS=3):
        super().__init__()
        self.conv_blocks = []
        self.outputs_channels = []
        pool_fun = {2:nn.MaxPool2d, 3:nn.MaxPool3d}
        hidden_channels = root_channels
        for i in range(n_layers):
            self.conv_blocks += [ ResBlock(input_channels, hidden_channels, activation=activation, DIMS=DIMS) ]
            self.outputs_channels += [hidden_channels]
            input_channels, hidden_channels = hidden_channels, int(hidden_channels*expansion_factor)
        self.conv_blocks = nn.ModuleList(self.conv_blocks)
        #self.pool = th.nn.MaxPool3d(2, padding=1) this is madness
        self.pool = pool_fun[DIMS](2)
        self.act = activation
        
    def forward(self,x):
        outputs = [self.conv_blocks[0](x)]    
        for conv in self.conv_blocks[1:]:
            outputs += [conv(self.act(self.pool(outputs[-1])))]
        return outputs
    
class MLP(th.nn.Module):
    
    def __init__(self, input_channels, output_channels, hidden_channels=256, n_hidden_layers=2, activation=nn.LeakyReLU(inplace=True)):
        super().__init__()
        input_channels, hidden_channels = input_channels, hidden_channels
        self.fcn_0 = th.nn.Linear(input_channels, hidden_channels)
        self.fcn_i = nn.ModuleList( [th.nn.Linear(hidden_channels, hidden_channels)]*n_hidden_layers )
        self.fcn_2 = th.nn.Linear(hidden_channels, output_channels)
        self.act = activation
        self.output_channels = output_channels
    
    def forward(self,x):
        x = self.act(self.fcn_0(x))
        for fcn in self.fcn_i:
            x = self.act(fcn(x) + x)
        x = self.fcn_2(x)
        return x
