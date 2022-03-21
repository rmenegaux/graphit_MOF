import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from scipy import sparse as sp

"""
    GraphiT-GT and GraphiT-GT-LSPE
    
"""
from transformer_layer import GraphiT_GT_Layer, MLPReadout


def global_pooling(x, readout='mean'):
    if readout == 'mean':
        return x.mean(dim=1)
    elif readout == 'max':
        return x.max(dim=1)
    elif readout == 'sum':
        return x.sum(dim=1)


class GraphiTNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        
        gamma = net_params['gamma']
        self.adaptive_edge_PE = net_params['adaptive_edge_PE']
        
        GT_layers = net_params['L']
        GT_hidden_dim = net_params['hidden_dim']
        GT_out_dim = net_params['out_dim']
        GT_n_heads = net_params['n_heads']
        
        self.readout = net_params['readout']
        in_feat_dropout = net_params['in_feat_dropout']

        self.readout = net_params['readout']

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.pe_init = net_params['pe_init']
        
        self.pos_enc_dim = net_params['pos_enc_dim']
        self.use_edge_features = net_params['use_edge_features']

        layer_params = {}
        for param in [
            'double_attention',
            'dropout',
            'layer_norm',
            'batch_norm',
            'residual',
            'adaptive_edge_PE',
            'use_edge_features',
            'update_edge_features',
            'update_pos_enc',
            ]:
            layer_params[param] = net_params[param]
        
        if self.pe_init in ['rand_walk']:
            self.embedding_p = nn.Linear(self.pos_enc_dim, GT_hidden_dim)
        
        self.embedding_h = nn.Embedding(num_atom_type, GT_hidden_dim)
        if self.use_edge_features:
            self.embedding_e = nn.Embedding(num_bond_type + 1, GT_hidden_dim)
        
        self.layers = nn.ModuleList([
            GraphiT_GT_Layer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, **layer_params) for _ in range(GT_layers-1)
            ])
        layer_params['adaptive_edge_PE'] = False # Last layer with full vanilla attention
        self.layers.append(
            GraphiT_GT_Layer(gamma, GT_hidden_dim, GT_out_dim, GT_n_heads, **layer_params)
            )
        
        self.MLP_layer = MLPReadout(GT_out_dim, 1)   # 1 out dim since regression problem        
                
        
    def forward(self, h, p, e, k_RW=None, mask=None):
        h = h.squeeze()
        # input embedding
        h = self.embedding_h(h)

        # if self.use_edge_features:
        #     e = self.embedding_e(e)
        e = self.embedding_e(e)        

        h = self.in_feat_dropout(h)
        
        # if self.pe_init in ['rand_walk']:
        #     p = self.embedding_p(p)
        #     h = h + p
        p = self.embedding_p(p)
        h = h + p
        
        k_RW_0 = k_RW
        for conv in self.layers:
            h, p, e = conv(h, p, e, k_RW=k_RW)
            # This part should probably be moved to the DataLoader:
            k_RW = torch.matmul(k_RW, k_RW_0)
        
        # readout
        h = global_pooling(h, readout=self.readout)
        
        return self.MLP_layer(h)
        
    def loss(self, scores, targets):

        loss = nn.L1Loss()(scores, targets)
        
        return loss