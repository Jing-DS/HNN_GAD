import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GNN, Centralization

import manifolds
from manifolds.utils import acosh
from layers import LorentzGNN, LorentzCentralization, HyperbolicGNN, HyperbolicCentralization

class Encoder(nn.Module):
    def __init__(self, dropout, act, use_bias, layer_dims=None):
        super(Encoder, self).__init__()
        
        gc_layers = []
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            gc_layers.append(GNN(in_dim, out_dim, dropout, act, use_bias))
            gc_layers.append(Centralization())
        self.layers = nn.Sequential(*gc_layers)

    def forward(self, x, adj):
        input = (x, adj)
        x, _ = self.layers.forward(input)
        return x

class Attribute_Decoder(nn.Module):
    def __init__(self, dropout, act, use_bias, layer_dims=None):
        super(Attribute_Decoder, self).__init__()
        
        gc_layers = []
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            gc_layers.append(GNN(in_dim, out_dim, dropout, act, use_bias))
            gc_layers.append(Centralization())
        self.layers = nn.Sequential(*gc_layers)

    def forward(self, x, adj):
        input = (x, adj)
        x, _ = self.layers.forward(input)
        return x

class Structure_Decoder(nn.Module):
    def __init__(self, dropout, act, use_bias, layer_dims=None):
        super(Structure_Decoder, self).__init__()
        
        gc_layers = []
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            gc_layers.append(GNN(in_dim, out_dim, dropout, act, use_bias))
            gc_layers.append(Centralization())
        self.layers = nn.Sequential(*gc_layers)

    def forward(self, x, adj):
        input = (x, adj)
        x, _ = self.layers.forward(input)

        return x

class GNN_Euclidean(nn.Module):
    def __init__(self, dropout, act, use_bias, layer_dims=None):
        super(GNN_Euclidean, self).__init__()
        act = getattr(F, act)
        
        print('Model structure: encoder', layer_dims[0], ", contextual decoder", layer_dims[1], ", structural decoder", layer_dims[2])
        encoding_layer_dim = layer_dims[0]
        decoding_layer_dim = layer_dims[1]
        decodestruct_layer_dim = layer_dims[2]
        
        self.shared_encoder = Encoder(dropout, act, use_bias, layer_dims=encoding_layer_dim)
        self.attr_decoder = Attribute_Decoder(dropout, act, use_bias, layer_dims=decoding_layer_dim)
        self.struct_decoder = Structure_Decoder(dropout, act, use_bias, layer_dims=decodestruct_layer_dim)

    def forward(self, x, adj):
        x_emb = self.shared_encoder(x, adj)
        x_hat = self.attr_decoder(x_emb, adj)
        struc_decode = self.struct_decoder(x_emb, adj)
        
        return x_hat, struc_decode
    
class Encoder_LORENTZ(nn.Module):
    def __init__(self, manifold, use_bias, dropout, layer_dims=None):
        super(Encoder_LORENTZ, self).__init__()
        
        gc_layers = []
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            gc_layers.append(LorentzGNN(manifold, in_dim, out_dim, use_bias, dropout))
            gc_layers.append(LorentzCentralization(manifold))
        self.layers = nn.Sequential(*gc_layers)

    def forward(self, x, adj):
        input = (x, adj)
        x, _ = self.layers.forward(input)
        return x

class Attribute_Decoder_LORENTZ(nn.Module):
    def __init__(self, manifold, use_bias, dropout, layer_dims=None):
        super(Attribute_Decoder_LORENTZ, self).__init__()
        
        gc_layers = []
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            gc_layers.append(LorentzGNN(manifold, in_dim, out_dim, use_bias, dropout))
            gc_layers.append(LorentzCentralization(manifold))
        self.layers = nn.Sequential(*gc_layers)

    def forward(self, x, adj):
        input = (x, adj)
        x, _ = self.layers.forward(input)
        return x

class Structure_Decoder_LORENTZ(nn.Module):
    def __init__(self, manifold, use_bias, dropout, layer_dims=None):
        super(Structure_Decoder_LORENTZ, self).__init__()
        self.manifold = manifold

        gc_layers = []
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            gc_layers.append(LorentzGNN(manifold, in_dim, out_dim, use_bias, dropout))
            gc_layers.append(LorentzCentralization(manifold))
        self.layers = nn.Sequential(*gc_layers)

    def forward(self, x, adj):
        input = (x, adj)
        x, _ = self.layers.forward(input)
           
        return x

class GNN_LORENTZ(nn.Module):
    def __init__(self, manifold, use_bias, dropout, layer_dims=None):
        super(GNN_LORENTZ, self).__init__()
        self.manifold = getattr(manifolds, manifold)()
        
        print('Using model structure:',layer_dims)
        layer_dims[0][0] += 1
        layer_dims[1][-1] += 1
        encoding_layer_dim = layer_dims[0]
        decoding_layer_dim = layer_dims[1]
        decodestruct_layer_dim = layer_dims[2]
        
        self.shared_encoder = Encoder_LORENTZ(self.manifold, use_bias, dropout, layer_dims=encoding_layer_dim)
        self.attr_decoder = Attribute_Decoder_LORENTZ(self.manifold, use_bias, dropout, layer_dims=decoding_layer_dim)
        self.struct_decoder = Structure_Decoder_LORENTZ(self.manifold, use_bias, dropout, layer_dims=decodestruct_layer_dim)
    
    def forward(self, x, adj):
#         o = torch.zeros_like(x)
#         x = torch.cat([o[:, 0:1], x], dim=1)
#         x = self.manifold.expmap0(x)
        
        x_emb = self.shared_encoder(x, adj)
        x_hat = self.attr_decoder(x_emb, adj)
        struc_decode = self.struct_decoder(x_emb, adj)
        return x_hat, struc_decode

    
class Encoder_POINCARE(nn.Module):
    def __init__(self, manifold, c, dropout, act, use_bias, layer_dims=None):
        super(Encoder_POINCARE, self).__init__()
        
        gc_layers = []
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            gc_layers.append(HyperbolicGNN(manifold, in_dim, out_dim, c, c, dropout, act, use_bias, False, False))
            gc_layers.append(HyperbolicCentralization(manifold, c))
        self.layers = nn.Sequential(*gc_layers)

    def forward(self, x, adj):
        input = (x, adj)
        x, _ = self.layers.forward(input)
        return x

class Attribute_Decoder_POINCARE(nn.Module):
    def __init__(self, manifold, c, dropout, act, use_bias, layer_dims=None):
        super(Attribute_Decoder_POINCARE, self).__init__()
        
        gc_layers = []
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            gc_layers.append(HyperbolicGNN(manifold, in_dim, out_dim, c, c, dropout, act, use_bias, False, False))
            gc_layers.append(HyperbolicCentralization(manifold, c))
        self.layers = nn.Sequential(*gc_layers)

    def forward(self, x, adj):
        input = (x, adj)
        x, _ = self.layers.forward(input)
        return x

class Structure_Decoder_POINCARE(nn.Module):
    def __init__(self, manifold, c, dropout, act, use_bias, layer_dims=None):
        super(Structure_Decoder_POINCARE, self).__init__()
        self.manifold = manifold
        self.c = c
        
        gc_layers = []
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            gc_layers.append(HyperbolicGNN(manifold, in_dim, out_dim, c, c, dropout, act, use_bias, False, False))
            gc_layers.append(HyperbolicCentralization(manifold, c))
        self.layers = nn.Sequential(*gc_layers)

    def forward(self, x, adj):
        input = (x, adj)
        x, _ = self.layers.forward(input)

        return x

class GNN_POINCARE(nn.Module):
    def __init__(self, manifold, c, dropout, act, use_bias, layer_dims=None):
        super(GNN_POINCARE, self).__init__()
        self.manifold = getattr(manifolds, manifold)()
        self.c = c
        act = getattr(F, act)
            
        print('Using model structure:',layer_dims)
        encoding_layer_dim = layer_dims[0]
        decoding_layer_dim = layer_dims[1]
        decodestruct_layer_dim = layer_dims[2]
        
        self.shared_encoder = Encoder_POINCARE(self.manifold, self.c, dropout, act, use_bias, layer_dims=encoding_layer_dim)
        self.attr_decoder = Attribute_Decoder_POINCARE(self.manifold, self.c, dropout, act, use_bias, layer_dims=decoding_layer_dim)
        self.struct_decoder = Structure_Decoder_POINCARE(self.manifold, self.c, dropout, act, use_bias, layer_dims=decodestruct_layer_dim)

        
    def forward(self, x, adj):
#         x_tan = self.manifold.proj_tan0(x, self.c)
#         x_hyp = self.manifold.expmap0(x_tan, c=self.c)
#         x_hyp = self.manifold.proj(x_hyp, c=self.c)
        
        x_emb = self.shared_encoder(x, adj)
        x_hat = self.attr_decoder(x_emb, adj)
        struc_decode = self.struct_decoder(x_emb, adj)
        return x_hat, struc_decode
   