"""Hyperbolic layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt

from geoopt import ManifoldParameter

from manifolds.utils import acosh

def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to('cuda:'+str(args.cuda)) for curv in curvatures]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h

class LorentzMultiHeadedAttention(nn.Module):
    """
    Hyperbolic Multi-headed Attention
    """

    def __init__(self, head_count, dim, manifold, dropout = 0.0):
        super(LorentzMultiHeadedAttention, self).__init__()
        self.dim_per_head = dim // head_count
        self.dim = dim
        self.manifold = manifold
        self.head_count = head_count

        self.linear_key = LorentzLinear(manifold, dim, dim, dropout=dropout)
        self.linear_value = LorentzLinear(manifold, dim, dim, dropout=dropout)
        self.linear_query = LorentzLinear(manifold, dim, dim, dropout=dropout)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor([math.sqrt(dim)]))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, key, value, query, mask = None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        # key_len = key.size(1)
        # query_len = query.size(1)

        def shape(x):
            """Projection."""
            if len(x.size()) == 3:
                x = x.view(batch_size, -1, head_count, dim_per_head)
            return x.transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).view(batch_size, -1, head_count * dim_per_head)

        query = self.linear_query(query)
        key = self.linear_key(key)
        value =  self.linear_value(value)
        key = shape(key)
        value = shape(value)
        query = shape(query)
        # key_len = key.size(2)
        # query_len = query.size(2)

        attn = (2 + 2 * self.manifold.cinner(query, key)) / self.scale + self.bias
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            attn = attn.masked_fill(mask, -1e18)
        context = self.manifold.mid_point(value, attn)
        context = unshape(context)

        return context

class LorentzCentralization(torch.nn.Module):
    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold  

    def forward(self, input):
        x, adj = input

        mean = self.manifold.mid_point(x)
        out = self.manifold.logmap(mean, x)
        out = self.manifold.transp0back(mean, out)
        out_hyp = self.manifold.expmap0(out)
        return out_hyp, adj

    
class LorentzCentroidDistance(nn.Module):
    """
    Hyerbolic embeddings to Euclidean distance
    """

    def __init__(self, dim, n_classes, manifold, bias = True):
        super(LorentzCentroidDistance, self).__init__()
        self.manifold = manifold
        self.input_dim = dim
        self.output_dim = n_classes
        self.use_bias = bias
        self.cls = ManifoldParameter(
            self.manifold.random_normal((n_classes, dim), std=1./math.sqrt(dim)), 
            manifold=self.manifold)
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_classes))
        
    def forward(self, x):
        if self.use_bias:
            return (2 + 2 * self.manifold.cinner(x, self.cls)) + self.bias
        else:
            return 2 + 2 * self.manifold.cinner(x, self.cls)

class LorentzAct(Module):
    """
    Hyperboloid activation layer.
    """

    def __init__(self, manifold, act):
        super(LorentzAct, self).__init__()
        self.manifold = manifold
        self.act = act

    def forward(self, x):
        x = self.manifold.logmap0(x)
        x = self.act(x)
        x = self.manifold.expmap0(x)
        return x
    
class LorentzGNN(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, use_bias, dropout, use_att=False, local_agg=False, nonlin=None):
        super(LorentzGNN, self).__init__()
        self.linear = LorentzLinear(manifold, in_features, out_features, use_bias, dropout, nonlin=nonlin)
    def forward(self, input):
        x, adj = input
        h = self.linear(x)
        return h, adj
    
class LorentzGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, use_bias, dropout, use_att=False, local_agg=False, nonlin=None):
        super(LorentzGraphConvolution, self).__init__()
        self.linear = LorentzLinear(manifold, in_features, out_features, use_bias, dropout, nonlin=nonlin)
        self.agg = LorentzAgg(manifold, out_features, dropout, use_att, local_agg)
    def forward(self, input):
        x, adj = input
        h = self.linear(x)
        h = self.agg(h, adj)
        output = h, adj
        return output

class LorentzLinear(nn.Module):
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.1,
                 scale=10,
                 fixscale=False,
                 nonlin=None):
        super().__init__()
        #torch.manual_seed(1)
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)

    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1 + 1e-4
        if (x_narrow * x_narrow).sum()!=0:
            scale = (time * time - 1) / \
                (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)
            x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        else:
            time=time*0+1
            x = torch.cat([time, x_narrow], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


class LorentzAgg(Module):
    """
    Lorentz aggregation layer.
    """

    def __init__(self, manifold, in_features, dropout, use_att, local_agg):
        super(LorentzAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            # self.att = DenseAtt(in_features, dropout)
            self.key_linear = LorentzLinear(manifold, in_features, in_features)
            self.query_linear = LorentzLinear(manifold, in_features, in_features)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_features))

    def forward(self, x, adj):

        if self.use_att:
            if self.local_agg:
                query = self.query_linear(x)
                key = self.key_linear(x)
                att_adj = 2 + 2 * self.manifold.cinner(query, key)
                att_adj = att_adj / self.scale + self.bias
                att_adj = torch.sigmoid(att_adj)
                att_adj = torch.mul(adj.to_dense(), att_adj)
                support_t = torch.matmul(att_adj, x)
            else:
                adj_att = self.att(x, adj)
                support_t = torch.matmul(adj_att, x)
        else:
            support_t = torch.spmm(adj, x)
            
        denom = (-self.manifold.inner(None, support_t, keepdim=True))
        denom = denom.abs().clamp_min(1e-8).sqrt()
        output = support_t / denom
        return output

    def attention(self, x, adj):
        pass

class HyperbolicCentralization(torch.nn.Module):
    def __init__(self, manifold, c):
        super().__init__()

        self.manifold = manifold
        self.c = c

    def forward(self, input):
        x, adj = input
        
        out = self.manifold.logmap0(x, self.c)
        mean = torch.mean(out, dim=0)
        out = x - mean
        out_hyp = self.manifold.expmap0(out, self.c)
        out_hyp = self.manifold.proj(out_hyp, self.c)
        return out_hyp, adj

class HyperbolicGNN(nn.Module):

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGNN, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output
    
class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output
    
class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):

        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
