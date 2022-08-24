import torch

import torch_geometric.utils as utils

from scipy.linalg import expm


def compute_RW_from_adjacency(A):
    '''
    Returns the random walk transition matrix for an adjacency matrix A
    '''
    D = A.sum(dim=-1, keepdim=True)
    D[D == 0] = 1 # Prevent any division by 0 errors
    return A / D # A D^-1
    
def get_laplacian_from_adjacency(A):
    RW = compute_RW_from_adjacency(A)
    I = torch.eye(*RW.size(), out=torch.empty_like(RW))
    return I - RW

def RW_kernel_from_adjacency(A, beta=0.25, p_steps=1):
    '''
    Returns the random walk kernel matrix for an adjacency matrix A
    '''
    L = get_laplacian_from_adjacency(A)
    I = torch.eye(*L.size(), out=torch.empty_like(L))
    k_RW = I - beta * L
    k_RW_power = k_RW
    for power in range(p_steps-1):
        k_RW_power = k_RW_power @ k_RW
    return k_RW_power


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
        RW = compute_RW_from_adjacency(A)
        RW_power = RW
        node_pe = torch.zeros((num_nodes, self.p_steps))
        node_pe[:, 0] = RW.diagonal()
        for power in range(self.p_steps-1):
            RW_power = RW @ RW_power
            node_pe[:, power + 1] = RW_power.diagonal()
        return node_pe

class IterableNodePE(object):
    '''
    A disguised list, containing precomputed positional encodings. Careful indexing is required
    '''
    def __init__(self, pe_list, **parameters):
        self.current_index = 0
        self.pe_list = pe_list
        self.embedding_dimension = parameters.get('embedding_dimension', None)

    def get_embedding_dimension(self):
        return self.embedding_dimension

    def __call__(self, graph):
        node_pe = self.pe_list[self.current_index]
        self.current_index += 1
        return node_pe


class BaseAttentionPE(object):

    def __init__(self, **parameters):
        '''
        Parameters that are applicable to any Attention PE
        '''
        self.zero_diag = parameters.get('zero_diag', False)

    def __call__(self, graph):
        K = self.compute_attention_pe(graph)
        if self.zero_diag:
            I = torch.eye(*K.size()[:1])
            if K.ndim == 3:
                I = I.unsqueeze(-1)
            K = K * (1 - I)
        return K

    def compute_attention_pe(self):
        pass

    def get_dimension(self):
        '''
        Returns the size of K's last dimension
        '''
        return 1


class RandomWalkAttentionPE(BaseAttentionPE):

    def __init__(self, **parameters):
        self.p_steps = parameters.get('p_steps', 16)
        self.beta = parameters.get('beta', 0.25)
        super().__init__(**parameters)

    def compute_attention_pe(self, graph):
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        k_RW_power = RW_kernel_from_adjacency(A, beta=self.beta, p_steps=self.p_steps)
        
        return k_RW_power

    def get_dimension(self):
        return 1


class DiffusionAttentionPE(BaseAttentionPE):
    def __init__(self, **parameters):
        self.beta = parameters.get('beta', 0.5)
        super().__init__(**parameters)

    def compute_attention_pe(self, graph):
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        L = get_laplacian_from_adjacency(A)
        attention_pe = expm(-self.beta * L.numpy())
        return torch.from_numpy(attention_pe)
    
    def get_dimension(self):
        return 1


class EdgeRWAttentionPE(BaseAttentionPE):
    '''
    Computes a separate random walk kernel for each edge type
    '''
    def __init__(self, **parameters):
        self.p_steps = parameters.get('p_steps', 16)
        self.beta = parameters.get('beta', 0.25)
        self.num_edge_type = parameters.get('num_edge_type', 3)
        super().__init__(**parameters)

    def compute_attention_pe(self, graph):
        k_RW_power = []
        for edge_type in range(self.num_edge_type):
            # Build adjacency matrix for each edge type
            edge_attr = (graph.edge_attr == edge_type + 1).long()
            A = utils.to_dense_adj(graph.edge_index, edge_attr=edge_attr).squeeze()
            k_RW_power.append(RW_kernel_from_adjacency(A, beta=self.beta, p_steps=self.p_steps))
        return torch.stack(k_RW_power, dim=-1)

    def get_dimension(self):
        return self.num_edge_type


class MultiRWAttentionPE(BaseAttentionPE):
    '''
    Computes the random walk kernel for all number of steps from 1 to self.p_steps
    '''
    def __init__(self, **parameters):
        self.p_steps = parameters.get('p_steps', 16)
        self.stride = parameters.get('stride', 1)
        self.beta = parameters.get('beta', 0.25)
        super().__init__(**parameters)
    
    def compute_attention_pe(self, graph):
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        k_RW_0 = RW_kernel_from_adjacency(A, beta=self.beta, p_steps=1)
        k_RW_power = k_RW_0
        k_RW_all_powers = [k_RW_0]
        for i in range(self.p_steps-1):
            for _ in range(self.stride):
                k_RW_power = k_RW_power @ k_RW_0
            k_RW_all_powers.append(k_RW_power)
        return torch.stack(k_RW_all_powers, dim=-1)

    def get_dimension(self):
        return self.p_steps

class MultiDiffusionAttentionPE(BaseAttentionPE):
    '''
    Computes the diffusion kernel for all number of steps from 1 to self.p_steps
    '''
    def __init__(self, **parameters):
        self.p_steps = parameters.get('p_steps', 16)
        self.beta = parameters.get('beta', 0.25)
        super().__init__(**parameters)
    
    def compute_attention_pe(self, graph):
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        L = get_laplacian_from_adjacency(A)
        k_diff_0 = torch.from_numpy(expm(-self.beta * L.numpy()))
        k_diff_power = k_diff_0
        k_diff_all_powers = [k_diff_0]
        for i in range(self.p_steps-1):
            k_diff_power = k_diff_power @ k_diff_0
            k_diff_all_powers.append(k_diff_power)
        return torch.stack(k_diff_all_powers, dim=-1)

    def get_dimension(self):
        return self.p_steps


class AdjacencyAttentionPE(BaseAttentionPE):

    def __init__(self, **parameters):
        super().__init__(**parameters)

    def compute_attention_pe(self, graph):
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        return A / A.sum(dim=-1)

    def get_dimension(self):
        return 1


NodePositionalEmbeddings = {
    'rand_walk': RandomWalkNodePE,
    'gckn': IterableNodePE
}

AttentionPositionalEmbeddings = {
    'rand_walk': RandomWalkAttentionPE,
    'edge_RW': EdgeRWAttentionPE,
    'multi_RW': MultiRWAttentionPE,
    'multi_diffusion': MultiDiffusionAttentionPE,
    'adj': AdjacencyAttentionPE,
    'diffusion': DiffusionAttentionPE,
}