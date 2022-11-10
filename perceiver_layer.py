import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from timm.models.vision_transformer import _cfg, Mlp, Block
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from einops import rearrange, repeat
from einops.layers.torch import Reduce
import sys

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


"""
    GraphiT-GT
    
"""
def combine_h_p(h, p, operation='sum'):
    if operation == 'concat':
        h = torch.cat((h, p), dim=-1)
    elif operation == 'sum':
        h = h + p
    elif operation == 'product':
        h = h * p
    return h


class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, out_dim = 64, dropout = 0.):
        super().__init__()
        self.out_dim = out_dim
        self.inner_dim = out_dim * heads
        context_dim = default(context_dim, query_dim)

        self.scale = out_dim ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, self.inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(self.inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        # print(k.shape)

        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)


        sim = sim.clamp(-5, 5)
        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)
        

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

        """
        n_batch = q.size()[0]
        num_latents = q.size()[1]
        num_nodes = k.size()[1]

        q = q.reshape(n_batch, num_latents, h, self.out_dim)
        k = k.reshape(n_batch, num_nodes, h, self.out_dim)
        v = v.reshape(n_batch, num_nodes, h, self.out_dim).transpose(2, 1)
        # scores = torch.einsum('bihk,bjhk,bijhk->bhij', Q_h, K_h, E)
        #sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        sim = torch.einsum('b i h d, b j h d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)
        #1 sim = torch.exp(sim - sim.amax(dim=(-2, -1), keepdim=True))
        #1 softmax_denom = sim.sum(-1, keepdim=True).clamp(min=1e-6) # [n_batch, num_heads, num_nodes, 1]


        #1sim = self.dropout(sim)

        out = attn @ v # [n_batch, num_heads, num_nodes, out_dim]

        # Normalize sim
        #1 out = out / softmax_denom
        # Concatenate attention heads
        out = out.transpose(2, 1).reshape(-1, num_latents, self.heads * self.out_dim) # [n_batch, num_nodes, out_dim * num_heads]
        
        return self.to_out(out)
        """

        # if exists(mask):
        #     mask = rearrange(mask, 'b ... -> b (...)')
        #     max_neg_value = -torch.finfo(sim.dtype).max
        #     mask = repeat(mask, 'b j -> (b h) () j', h = h)
        #     sim.masked_fill_(~mask, max_neg_value)

        # # attention, what we cannot get enough of
        # attn = sim.softmax(dim = -1)
        # attn = self.dropout(attn)

        # out = torch.einsum('b i j, b j d -> b i d', attn, v)
        # out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        # return self.to_out(out)

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, in_dim_edges, out_dim, num_heads,
                 use_bias=False, use_attention_pe=True, use_edge_features=False):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_edge_features = use_edge_features
        self.use_attention_pe = use_attention_pe
        
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        if self.use_edge_features:
            self.E = nn.Linear(in_dim_edges, out_dim * num_heads, bias=use_bias)

        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        
    def forward(self, h, e, k_RW=None, mask=None, adj=None):
        
        # import ipdb; ipdb.set_trace()
        Q_h = self.Q(h) # [n_batch, num_nodes, out_dim * num_heads]
        K_h = self.K(h)
        V_h = self.V(h)

        n_batch = Q_h.size()[0]
        num_nodes = Q_h.size()[1]

        # Reshaping into [num_heads, num_nodes, feat_dim] to 
        # get projections for multi-head attention
        Q_h = Q_h.reshape(n_batch, num_nodes, self.num_heads, self.out_dim)
        K_h = K_h.reshape(n_batch, num_nodes, self.num_heads, self.out_dim)
        V_h = V_h.reshape(n_batch, num_nodes, self.num_heads, self.out_dim).transpose(2, 1) # [n_batch, num_heads, num_nodes, out_dim]

        # Normalize by sqrt(head dimension)
        scaling = float(self.out_dim) ** -0.5
        K_h = K_h * scaling

        if self.use_edge_features:

            E = self.E(e)   # [n_batch, num_nodes * num_nodes, out_dim * num_heads]
            E = E.reshape(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim)

            scores = torch.einsum('bihk,bjhk,bijhk->bhij', Q_h, K_h, E)
        else:
            # attention(i, j) = sum(Q_i * K_j)
            scores = torch.einsum('bihk,bjhk->bhij', Q_h, K_h)

        # Apply exponential and clamp for numerical stability
        # scores = torch.exp(scores.clamp(-5, 5)) # [n_batch, num_heads, num_nodes, num_nodes]
        scores = torch.exp(scores - scores.amax(dim=(-2, -1), keepdim=True))

        # Make sure attention scores for padding are 0
        if mask is not None:
            scores = scores * mask.view(-1, 1, num_nodes, 1) * mask.view(-1, 1, 1, num_nodes)

        if self.use_attention_pe:
            # Introduce new dimension for the different heads
            # k_RW = k_RW.unsqueeze(1)
            scores = scores * k_RW
        
        softmax_denom = scores.sum(-1, keepdim=True).clamp(min=1e-6) # [n_batch, num_heads, num_nodes, 1]

        h = scores @ V_h # [n_batch, num_heads, num_nodes, out_dim]
        # Normalize scores
        h = h / softmax_denom
        # Concatenate attention heads
        h = h.transpose(2, 1).reshape(-1, num_nodes, self.num_heads * self.out_dim) # [n_batch, num_nodes, out_dim * num_heads]

        return h
    

class GraphiT_Perceiver_Layer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, **layer_params):

        super().__init__()
        
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = layer_params['dropout']
        self.residual = layer_params['residual']
        self.layer_norm = layer_params['layer_norm']     
        self.batch_norm = layer_params['batch_norm']
        self.instance_norm = layer_params['instance_norm']
        self.feedforward = layer_params['feedforward']
        self.update_edge_features = layer_params['update_edge_features']
        self.use_node_pe = layer_params['use_node_pe']
        self.use_attention_pe = layer_params['use_attention_pe']
        self.multi_attention_pe = layer_params['multi_attention_pe']
        self.normalize_degree = layer_params['normalize_degree']
        self.update_pos_enc = layer_params['update_pos_enc']
        self.cross_dim = layer_params['cross_dim']
        self.num_latents = layer_params['num_latents']
        self.L_cross = layer_params['L_cross'] 


        attention_params = {
            param: layer_params[param] for param in ['use_bias', 'use_attention_pe', 'use_edge_features']
        }
        cross_attention_params = {
            #param: layer_params[param] for param in ['cross_attention_dim']
        }
        # in_dim*2 if positional embeddings are concatenated rather than summed
        in_dim_h = in_dim*2 if (self.use_node_pe == 'concat') else in_dim
        self.cross_attention_h = CrossAttentionLayer(query_dim = self.cross_dim, context_dim=in_dim, out_dim=out_dim//num_heads)
        self.attention_list = nn.ModuleList([])
        for i in range(self.L_cross):
            self.attention_list.append(CrossAttentionLayer(query_dim = self.cross_dim, context_dim=None, out_dim=out_dim//num_heads))
        # self.attention_queries = CrossAttentionLayer(query_dim = self.cross_dim, context_dim=None, out_dim=out_dim//num_heads)
        self.O_queries = nn.Linear(self.cross_dim, self.cross_dim, bias=False)

        
        if self.update_pos_enc:
            self.attention_p = MultiHeadAttentionLayer(in_dim, in_dim, out_dim//num_heads, num_heads, **attention_params)
            self.O_p = nn.Linear(out_dim, out_dim, bias=False)
        
        self.multi_attention_pe = layer_params['multi_attention_pe']
        self.learnable_attention_pe = (self.use_attention_pe and self.multi_attention_pe == 'aggregate')
        if self.learnable_attention_pe:
            attention_pe_dim = layer_params['attention_pe_dim']
            self.coef = nn.Parameter(torch.ones(attention_pe_dim) / attention_pe_dim)
    
        
        
        if self.layer_norm:
            #self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_queries = nn.LayerNorm(self.cross_dim)
            
        if self.batch_norm:
            # self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            #self.batch_norm1_h = MaskedBatchNorm1d(out_dim)
            self.batch_norm1_queries = nn.BatchNorm1d(self.cross_dim)
        
        if self.instance_norm:
            #self.instance_norm1_h = nn.InstanceNorm1d(out_dim)
            self.instance_norm1_queries = nn.InstanceNorm1d(self.cross_dim)
        
        # FFN for h
        if self.feedforward:
        #     self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2, bias=False)
        #     self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim, bias=False)
            self.FFN_queries_layer1 = nn.Linear(self.cross_dim, self.cross_dim*2, bias=False)
            self.FFN_queries_layer2 = nn.Linear(self.cross_dim*2, self.cross_dim, bias=False)
            if self.layer_norm:
                # self.layer_norm2_h = nn.LayerNorm(out_dim)
                self.layer_norm2_queries = nn.LayerNorm(self.cross_dim)
                
            if self.batch_norm:
                # self.batch_norm2_h = nn.BatchNorm1d(out_dim)
                # self.batch_norm2_h = MaskedBatchNorm1d(out_dim)
                self.batch_norm2_queries = nn.BatchNorm1d(self.cross_dim)

            if self.instance_norm:
                # self.instance_norm2_h = nn.InstanceNorm1d(out_dim)
                self.instance_norm2_queries = nn.InstanceNorm1d(self.cross_dim)


        if self.update_edge_features:
            self.B1 = nn.Linear(out_dim, out_dim, bias=False)
            self.B2 = nn.Linear(out_dim, out_dim, bias=False)
            self.E12 = nn.Linear(out_dim, out_dim, bias=False)
            if self.layer_norm:
                self.layer_norm_e = nn.LayerNorm(out_dim)
            if self.batch_norm and self.update_edge_features:
                self.batch_norm_e = nn.BatchNorm1d(out_dim)

    def forward_edges(self, h, e):
        '''
        Update edge features
        '''
        e_in = e
        B1_h = self.B1(h).unsqueeze(1)
        B2_h = self.B2(h).unsqueeze(2)
        E12 = self.E12(e)

        e = B1_h + B2_h + E12
        if self.batch_norm:
            n_batch, n_nodes, _, n_embedding = e.size()
            e = e.reshape(n_batch, n_nodes * n_nodes, n_embedding).transpose(1,2)
            e = self.batch_norm_e(e)
            e = e.transpose(1,2).reshape(n_batch, n_nodes, n_nodes, n_embedding)
        # e = self.batch_norm_e(e)
        e = e_in + F.relu(e)
        # e = e_in + torch.tanh(e)
        if self.layer_norm:
            e = self.layer_norm_e(e)
        # Batch norm not yet implemented, it would require a reshape
        return e

    def forward_p(self, p, e, k_RW=None, mask=None, adj=None):
        '''
        Update positional encoding p
        '''
        p_in1 = p # for residual connection
    
        p = self.attention_p(p, e, k_RW=k_RW, mask=mask, adj=adj)  
        p = F.dropout(p, self.dropout, training=self.training)
        p = self.O_p(p)
        p = torch.tanh(p)
        if self.residual:
            p = p_in1 + p # residual connection

        return p

    # def feed_forward_block(self, h, mask=None):
    #     ''' h = combine_h_p(h, p, operation=self.use_node_pe)
    #     # # FFN for h
    #     h_in2 = h # for second residual connection
    #     if self.layer_norm:
    #         h = self.layer_norm2_h(h)
    #     h = self.FFN_h_layer1(h)
    #     h = F.relu(h)
    #     h = F.dropout(h, self.dropout, training=self.training)
    #     h = self.FFN_h_layer2(h)

    #     if self.residual:
    #         h = h_in2 + h # residual connection       
    
    #     # if self.layer_norm:
    #     #     h = self.layer_norm2_h(h)

    #     if self.batch_norm:
    #         # h = self.batch_norm2_h(h.transpose(1,2)).transpose(1,2)
    #         h = self.batch_norm2_h(h.transpose(1,2), input_mask=mask.unsqueeze(1)).transpose(1,2)

    #     if self.instance_norm:
    #         # h = self.instance_norm2_h(h.transpose(1,2)).transpose(1,2)
    #         h = self.instance_norm1_h(h)
    #     return h

    def feed_forward_block_queries(self, queries, mask=None):
        '''
        Add dense layers to the self-attention
        '''
        # # FFN for h
        queries_in2 = queries # for second residual connection
        if self.layer_norm:
            queries = self.layer_norm2_queries(queries)
        queries = self.FFN_queries_layer1(queries)
        queries = F.relu(queries)
        queries = F.dropout(queries, self.dropout, training=self.training)
        queries = self.FFN_queries_layer2(queries)

        if self.residual:
            queries = queries_in2 + queries # residual connection       
    
        # if self.layer_norm:
        #     h = self.layer_norm2_h(h)

        if self.batch_norm:
            # h = self.batch_norm2_h(h.transpose(1,2)).transpose(1,2)
            queries = self.batch_norm2_queries(queries.transpose(1,2)).transpose(1,2)

        if self.instance_norm:
            # h = self.instance_norm2_h(h.transpose(1,2)).transpose(1,2)
            queries = self.instance_norm2_queries(queries)
        return queries

    def forward(self, latents, h, p, e, k_RW=None, mask=None, adj=None):

        
        queries = latents # for first residual connection

        # import ipdb; ipdb.set_trace()
        # [START] For calculation of h -----------------------------------------------------------------
        h = combine_h_p(h, p, operation=self.use_node_pe)

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.learnable_attention_pe:
            # Compute the weighted average of the relative positional encoding
            with torch.no_grad():
                coef = self.coef.data.clamp(min=0)
                coef /= coef.sum(dim=0, keepdim=True)
                self.coef.data.copy_(coef)
            k_RW = torch.tensordot(self.coef, k_RW, dims=[[0], [-1]])
        if self.use_attention_pe and self.multi_attention_pe != 'per_head':
            # Add dimension for the attention heads
            k_RW = k_RW.unsqueeze(1)
        elif self.use_attention_pe and self.multi_attention_pe == 'per_head':
            # One relative attention matrix per attention head
            k_RW = k_RW.transpose(1, -1)


        #cross attention out
        queries = self.cross_attention_h(x = queries, context = h, mask=mask)
        # multi-head attention out
        for i in range(self.L_cross-1):
            queries = self.attention_list[i](x = queries, context=None)

        # queries = self.attention_queries(x = queries, context=None)
        # if self.update_edge_features: 
        #     e = self.forward_edges(h_in1, e)
       
        queries = F.dropout(queries, self.dropout, training=self.training)

        queries = self.O_queries(queries)

        # Normalize by degree
        # The degree computation could be moved to the DataLoader
        # if self.normalize_degree:
        #     degrees = adj.sum(dim=-1, keepdim=True)
        #     degrees[degrees == 0] = 1
        #     h = h * degrees.pow(-0.5)


        
        if self.residual:
            queries = queries + latents # residual connection
          
        if self.layer_norm:
            queries = self.layer_norm1_queries(queries)

        # import ipdb; ipdb.set_trace()
        if self.batch_norm:
            # Apparently have to do this double transpose for 3D input
            # h = self.batch_norm1_h(h.transpose(1,2)).transpose(1,2)
            queries = self.batch_norm1_queries(queries.transpose(1,2)).transpose(1,2)
            # if mask is not None:
            #     h = mask.unsqueeze(-1) * h

        if self.instance_norm:
            # h = self.instance_norm1_h(h.transpose(1,2)).transpose(1,2)
            queries = self.instance_norm1_queries(queries)
        
        if self.feedforward:
            queries = self.feed_forward_block_queries(queries)
        """        
        # if self.use_node_pe and self.update_pos_enc:
        #     p = self.forward_p(p, e, k_RW=k_RW, mask=mask, adj=adj)

        # Set padding back to zero
        # if mask is not None:
        #     h = mask.unsqueeze(-1) * h
        """
        return queries #, p, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y