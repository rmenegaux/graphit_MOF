import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from scipy import sparse as sp

"""
    GraphiT-GT and GraphiT-GT-LSPE
    
"""
from transformer_layer import GraphiT_GT_Layer, MLPReadout, combine_h_p


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, full_atom_feature_dims):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[-1]):
            x_embedding += self.atom_embedding_list[i](x[...,i])

        return x_embedding

class BondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim, full_bond_feature_dims):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[-1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[...,i])

        return bond_embedding

def global_pooling(x, readout='mean'):
    if readout == 'mean':
        return x.mean(dim=1)
    elif readout == 'max':
        return x.max(dim=1)
    elif readout == 'sum':
        return x.sum(dim=1)
    elif readout == 'first':
        return x[:, 0, :]


class GraphiTNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        
        self.use_node_pe = net_params['use_node_pe']
        if self.use_node_pe:
            self.pos_enc_dim = net_params['pos_enc_dim']
        self.update_pos_enc = net_params['update_pos_enc']
        self.progressive_attention = net_params['use_attention_pe'] and net_params['multi_attention_pe'] == 'per_layer'
        
        GT_layers = net_params['L']
        GT_hidden_dim = net_params['hidden_dim']
        GT_out_dim = net_params['out_dim']
        GT_n_heads = net_params['n_heads']
        
        self.readout = net_params['readout']

        self.n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layer_norm = net_params['layer_norm']
        if self.layer_norm:
            self.layer_norm_h = nn.LayerNorm(GT_out_dim, elementwise_affine=False)

        self.use_edge_features = net_params['use_edge_features']

        layer_params = {'use_bias': False}
        for param in [
            'double_attention',
            'dropout',
            'layer_norm',
            'batch_norm',
            'instance_norm',
            'residual',
            'use_node_pe',
            'use_attention_pe',
            'attention_pe_dim',
            'multi_attention_pe',
            'use_edge_features',
            'update_edge_features',
            'update_pos_enc',
            'normalize_degree',
            'feedforward',
            ]:
            layer_params[param] = net_params.get(param, None)
        
        if self.use_node_pe:
            self.embedding_p = nn.Linear(self.pos_enc_dim, GT_hidden_dim)
        

        if isinstance(num_atom_type, list):
            self.embedding_h = AtomEncoder(GT_hidden_dim, num_atom_type)
        else:
            self.embedding_h = nn.Embedding(num_atom_type, GT_hidden_dim)
        
        if self.use_edge_features:
            if isinstance(num_bond_type, list):
                self.embedding_e = BondEncoder(GT_hidden_dim, num_bond_type)
            else:
                self.embedding_e = nn.Embedding(num_bond_type + 1, GT_hidden_dim)
        

        self.layers = nn.ModuleList([
            GraphiT_GT_Layer(GT_hidden_dim, GT_hidden_dim, GT_n_heads, **layer_params) for _ in range(GT_layers-1)
            ])
        if net_params['last_layer_full_attention']:
            # Last layer with full vanilla attention (no kernel)
            layer_params['use_attention_pe'] = False 
        layer_params['update_edge_features'] = False
        self.layers.append(
            GraphiT_GT_Layer(GT_hidden_dim, GT_out_dim, GT_n_heads, **layer_params)
            )
        
        if self.use_node_pe:
            self.p_out = nn.Linear(GT_out_dim, self.pos_enc_dim)
            self.Whp = nn.Linear(GT_out_dim+self.pos_enc_dim, GT_out_dim)

        self.MLP_layer = MLPReadout(GT_out_dim, self.n_classes)   # 1 out dim when regression problem        
                
        
    def forward(self, h, p, e, k_RW=None, mask=None):
        h = h.squeeze()
        # Node embedding
        h = self.embedding_h(h)
        # Binary adjacency matrix (used for double attention)
        adj = (e > 0)
        # Edge embedding
        if self.use_edge_features:
            e = self.embedding_e(e)

        h = self.in_feat_dropout(h)
        
        if self.use_node_pe:
            p = self.embedding_p(p)

        for i, conv in enumerate(self.layers):
            # Concatenate/Add/Multiply h and p for first layer (or all layers)
            # if (i == 0) or self.update_pos_enc:
            # # if True:
            #     h = combine_h_p(h, p, operation=self.use_node_pe)
            k_RW_i = k_RW[:, :, :, i] if self.progressive_attention else k_RW
            h, p, e = conv(h, p, e, k_RW=k_RW_i, mask=mask, adj=adj)

        if self.use_node_pe:
            p = self.p_out(p)
            # Concat h and p before classification
            # FIXME: hp is not used for now
            hp = self.Whp(torch.cat((h, p), dim=-1))

        # readout
        h = global_pooling(h, readout=self.readout)
        if self.layer_norm:
            h = self.layer_norm_h(h)
        
        return self.MLP_layer(h)
    

    def loss(self, scores, targets):

        loss = 0 

        if self.n_classes == 1:
            loss = nn.L1Loss()(scores, targets)
        else:
            loss = torch.nn.BCEWithLogitsLoss()(scores, targets)
        
        return loss
        
    # def loss_l1(self, scores, targets):

    #     loss = nn.L1Loss()(scores, targets)
        
    #     return loss
    
    # def loss_bce(self, pred, labels):
        
    #     # Loss A: Task loss -------------------------------------------------------------
    #     loss_a = torch.nn.BCEWithLogitsLoss()(pred, labels)
        
    #     if self.use_lapeig_loss:
    #         raise NotImplementedError
    #     else:
    #         loss = loss_a
    #     return loss