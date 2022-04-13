import torch

import torch_geometric.utils as utils


class RandomWalkNodePE(object):
    '''
    Returns a p_step-dimensional vector p for each node,
    with p_i = RW^i, the probability of landing back on that node after i steps in the graph
    '''
    def __init__(self, **parameters):
        self.p_steps = parameters.get('p_steps', 16)

    def get_embedding_dimension(self):
        return self.p_steps

    def __call__(self, graph):
        num_nodes = len(graph.x)
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        D = A.sum(dim=-1)
        RW = A / D
        RW_power = RW
        node_pe = torch.zeros((num_nodes, self.p_steps))
        node_pe[:, 0] = RW.diagonal()
        for power in range(self.p_steps-1):
            RW_power = RW @ RW_power
            node_pe[:, power + 1] = RW_power.diagonal()
        return node_pe

class RandomWalkAttentionPE(object):

    def __init__(self, **parameters):
        self.p_steps = parameters.get('p_steps', 16)
        self.beta = parameters.get('beta', 0.25)
        self.zero_diag = parameters.get('zero_diag', False)

    def __call__(self, graph):
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        k_RW_power = compute_RW_from_adjacency(A, beta=self.beta, p_steps=self.p_steps)
        if self.zero_diag:
            I = torch.eye(*A.size())
            k_RW_power = k_RW_power * (1 - I)
        return k_RW_power

    def get_dimension(self):
        return 1


def compute_RW_from_adjacency(A, beta=0.25, p_steps=1):
    '''
    Returns the random walk kernel matrix for an adjacency matrix A
    '''
    D = A.sum(dim=-1, keepdim=True)
    D[D == 0] = 1 # Prevent any division by 0 errors
    RW = A / D # A D^-1
    I = torch.eye(*RW.size(), out=torch.empty_like(RW))
    L = I - RW
    k_RW = I - beta * L
    k_RW_power = k_RW
    for power in range(p_steps-1):
        k_RW_power = k_RW_power @ k_RW
    return k_RW_power

class EdgeRWAttentionPE(object):
    '''
    Computes a separate random walk kernel for each edge type
    '''
    def __init__(self, **parameters):
        self.p_steps = parameters.get('p_steps', 16)
        self.beta = parameters.get('beta', 0.25)
        self.num_edge_type = parameters.get('num_edge_type', 3)

    def __call__(self, graph):
        k_RW_power = []
        for edge_type in range(self.num_edge_type):
            # Build adjacency matrix for each edge type
            edge_attr = (graph.edge_attr == edge_type + 1).long()
            A = utils.to_dense_adj(graph.edge_index, edge_attr=edge_attr).squeeze()
            k_RW_power.append(compute_RW_from_adjacency(A, beta=self.beta, p_steps=self.p_steps))
        return torch.stack(k_RW_power, dim=-1)

    def get_dimension(self):
        return self.num_edge_type

class PluralRWAttentionPE(object):
    '''
    Computes the random walk kernel for all number of steps from 1 to self.p_steps
    '''
    def __init__(self, **parameters):
        self.p_steps = parameters.get('p_steps', 16)
        self.beta = parameters.get('beta', 0.25)
    
    def __call__(self, graph):
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        k_RW = compute_RW_from_adjacency(A, beta=self.beta, p_steps=1)
        k_RW_all_powers = [k_RW]
        for i in range(self.p_steps-1):
            k_RW_all_powers.append(k_RW_all_powers[i] @ k_RW_all_powers[0])
        return torch.stack(k_RW_all_powers, dim=-1)

    def get_dimension(self):
        return self.p_steps

class AdjacencyAttentionPE(object):

    def __init__(self, **parameters):
        pass

    def __call__(self, graph):
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        return A / A.sum(dim=-1)

    def get_dimension(self):
        return 1

NodePositionalEmbeddings = {
    'rand_walk': RandomWalkNodePE
}

AttentionPositionalEmbeddings = {
    'rand_walk': RandomWalkAttentionPE,
    'progressive_RW': RandomWalkAttentionPE,
    'edge_RW': EdgeRWAttentionPE,
    'plural_RW': PluralRWAttentionPE,
    'adj': AdjacencyAttentionPE,
}