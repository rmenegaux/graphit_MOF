import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

"""
    GraphiT-GT
    
"""

"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph, use_bias, adaptive_edge_PE, attention_for):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.gamma = gamma
        self.full_graph = full_graph
        self.attention_for = attention_for
        # self.use_edge_features = use_edge_features
        self.adaptive_edge_PE = adaptive_edge_PE
        
        if self.attention_for == "h": 
            if use_bias:
                self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                self.E = nn.Linear(in_dim, out_dim * num_heads, bias=True)

                # if self.full_graph:
                #     self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                #     self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                #     self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)

                self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)

            else:
                self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                self.E = nn.Linear(in_dim, out_dim * num_heads, bias=False)

                # if self.full_graph:
                #     self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                #     self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                #     self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)

                self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
    
    
    def forward(self, h, e, k_RW=None, mask=None):
        
        Q_h = self.Q(h) # [n_batch, num_nodes, out_dim * num_heads]
        K_h = self.K(h) # [n_batch, num_nodes, out_dim * num_heads]
        V_h = self.V(h)
        E = self.E(e)   # [n_batch, num_nodes * num_nodes, out_dim * num_heads]

        n_batch = Q_h.size()[0]
        num_nodes = Q_h.size()[1]  
        E = E.reshape(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim) # [num_nodes, num_nodes, out_dim * num_heads]
            

        # Reshaping into [num_heads, num_nodes, feat_dim] to 
        # get projections for multi-head attention
        Q_h = Q_h.reshape(n_batch, num_nodes, self.num_heads, self.out_dim)
        K_h = K_h.reshape(n_batch, num_nodes, self.num_heads, self.out_dim)
        V_h = V_h.reshape(n_batch, num_nodes, self.num_heads, self.out_dim).transpose(2, 1) # [n_batch, num_heads, num_nodes, out_dim]
        
        # Normalize by sqrt(head dimension)
        scaling = float(self.out_dim) ** -0.5
        K_h = K_h * scaling

        # attention(i, j) = sum(Q_i * K_j * E_ij)
        # scores = torch.einsum('bihk,bjhk->bhij', Q_h, K_h)
        scores = torch.einsum('bihk,bjhk,bijhk->bhij', Q_h, K_h, E)

        if mask is not None:
            scores = scores * mask * mask.unsqueeze(1)
        # Apply exponential and clamp for numerical stability
        scores = torch.exp(scores.clamp(-5, 5)) # [n_batch, num_heads, num_nodes, num_nodes]

        if self.adaptive_edge_PE:
            # Introduce new dimension for the different heads
            k_RW = k_RW.unsqueeze(1)
            scores = scores * k_RW
        
        softmax_denom = scores.sum(-1, keepdim=True).clamp(min=1e-6) # [n_batch, num_heads, num_nodes, 1]

        h = scores @ V_h # [n_batch, num_heads, num_nodes, out_dim]
        # Normalize scores
        h = h / softmax_denom
        # Concatenate attention heads
        h = h.transpose(2, 1).reshape(-1, num_nodes, self.num_heads * self.out_dim) # [n_batch, num_nodes, out_dim * num_heads]
        
        return h
    

class GraphiT_GT_Layer(nn.Module):
    """
        Param: 
    """
    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph, dropout=0.0,
                 layer_norm=False, batch_norm=True, residual=True, adaptive_edge_PE=False, use_bias=False):
        super().__init__()
        
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        
        self.attention_h = MultiHeadAttentionLayer(gamma, in_dim, out_dim//num_heads, num_heads,
                                                   full_graph, use_bias, adaptive_edge_PE, attention_for="h")
        
        self.O_h = nn.Linear(out_dim, out_dim)
        
        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
        
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            
        
    def forward(self, h, p, e, k_RW=None, mask=None):

        h_in1 = h # for first residual connection
        
        # [START] For calculation of h -----------------------------------------------------------------
        
        # multi-head attention out
        h = self.attention_h(h, e, k_RW=k_RW, mask=mask)
        
        # #Concat multi-head outputs
        # h = h_attn_out.view(-1, self.out_channels)
       
        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h # residual connection
            
        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            # Apparently have to do this double transpose for 3D input 
            h = self.batch_norm1_h(h.transpose(1,2)).transpose(1,2)

        h_in2 = h # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection       
    
        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h.transpose(1,2)).transpose(1,2)         
        
        # [END] For calculation of h -----------------------------------------------------------------
        

        return h, None
        
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