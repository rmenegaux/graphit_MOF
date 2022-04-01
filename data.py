# from torch.utils.data import Datasets
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import torch_geometric.utils as utils
from torch_geometric.transforms import ToDense #, VirtualNode


class GraphDataset(object):
    def __init__(self, dataset, pe='adj', degree=False):
        """a pytorch geometric dataset as input
        """
        self.dataset = list(dataset)
        self.node_pe_list = None
        self.attention_pe_list = None
        self.adj_matrix_list = None
        self.degree_list = None
        self.use_node_pe = False
        self.use_attention_pe = False
        if degree:
            self.compute_degree()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        if self.use_node_pe:
            data.node_pe = self.node_pe_list[index]
        if self.use_attention_pe:
            data.attention_pe = self.attention_pe_list[index]
        return data

    def add_virtual_nodes(self):
        AddVirtualNode = VirtualNode()
        for i, g in enumerate(self.dataset):
            self.dataset[i] = AddVirtualNode(g, x_fill=21, edge_attr_fill=4)

    def compute_degree(self):
        self.degree_list = []
        for g in self.dataset:
            deg = 1. / torch.sqrt(1. + utils.degree(g.edge_index[0], g.num_nodes))
            self.degree_list.append(deg)

    def compute_node_pe(self, node_pe):
        '''
        Takes as argument a function returning a nodewise positional embedding from a graph
        '''
        self.node_pe_list = []
        for i, g in enumerate(self.dataset):
            self.node_pe_list.append(node_pe(g))
        self.use_node_pe = True
        self.node_pe_dimension = node_pe.get_embedding_dimension()

    def compute_attention_pe(self, attention_pe):
        self.attention_pe_list = []
        for i, g in enumerate(self.dataset):
            self.attention_pe_list.append(attention_pe(g))
        self.use_attention_pe = True

    def collate_fn(self):
        def collate(batch):
            batch = list(batch)
            max_len = max(len(g.x) for g in batch)
            dense_transform = ToDense(max_len)

            padded_x = torch.zeros((len(batch), max_len), dtype=int)
            padded_adj = torch.zeros((len(batch), max_len, max_len), dtype=int)
            mask = torch.zeros((len(batch), max_len), dtype=bool)
            labels = []
            attention_pe = None
            padded_p = None
            if self.use_node_pe:
                padded_p = torch.zeros((len(batch), max_len, self.node_pe_dimension), dtype=float)
            if self.use_attention_pe:
                attention_pe = torch.zeros((len(batch), max_len, max_len))

            for i, g in enumerate(batch):
                labels.append(g.y.view(-1))
                num_nodes = len(g.x)
                g = dense_transform(g)
                padded_x[i] = g.x.squeeze()
                padded_adj[i] = g.adj.squeeze()
                mask[i] = g.mask
                if self.use_node_pe:
                    padded_p[i, :num_nodes] = g.node_pe
                if self.use_attention_pe:
                    attention_pe[i, :num_nodes, :num_nodes] = g.attention_pe
            return padded_x, padded_adj, padded_p, mask, attention_pe, default_collate(labels)
        return collate


def compute_pe(graph):
    graph.adj_matrix = utils.to_dense_adj(graph.edge_index, graph.edge_attr)
    #self.adj_matrix_list.append(adj_matrix)
    pe = (graph.adj_matrix > 0)
    # Normalize by degree
    pe = pe * (pe.sum(-1, keepdim=True) ** -0.5)
    #self.pe_list.append(pe)
    graph.pe = pe
    return graph

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
        num_nodes = len(graph.x)
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        RW = A / A.sum(dim=-1) # A D^-1
        I = torch.eye(num_nodes)
        L = I - RW
        k_RW = I - self.beta * L
        k_RW_power = k_RW
        for power in range(self.p_steps-1):
            k_RW_power = k_RW_power @ k_RW
        if self.zero_diag:
            k_RW_power = k_RW_power * (1 - I)
        return k_RW_power

class AdjacencyAttentionPE(object):

    def __init__(self, **parameters):
        pass

    def __call__(self, graph):
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        return A / A.sum(dim=-1)


class DiffusionAttentionPe(object):
    def __init__(self, **parameters):
        self.beta = parameters.get('beta', 0.5)

    def __call__(self, graph):
        num_nodes = len(graph.x)
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        D = A.sum(dim=-1)
        # rw norm!
        RW = A / D
        I = torch.eye(num_nodes)
        L = I - RW
        attention_pe = expm(-self.beta * L.numpy())
        return torch.from_numpy(attention_pe)

NodePositionalEmbeddings = {
    'rand_walk': RandomWalkNodePE
}

AttentionPositionalEmbeddings = {
    'diffusion': DiffusionAttentionPe,
    'rand_walk': RandomWalkAttentionPE,
    'adj': AdjacencyAttentionPE,
}

import torch
from torch import Tensor

from torch_geometric.data import Data
# from torch_geometric.data.datapipes import functional_transform
# from torch_geometric.transforms import BaseTransform

class BaseTransform:
    """
    Was getting an import error when importing this class, here it is gain
    """
    def __call__(self, data):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

# @functional_transform('virtual_node')
class VirtualNode(BaseTransform):
    r"""Appends a virtual node to the given homogeneous graph that is connected
    to all other nodes, as described in the `"Neural Message Passing for
    Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper
    
    Recoded to accomodate for old version of torch-geometric
    """
    def __call__(self, data: Data, edge_attr_fill=0, x_fill=0) -> Data:
        num_nodes, (row, col) = data.num_nodes, data.edge_index
        # "Data has no attribute get"
        #edge_type = data.__dict__.get('edge_type', torch.zeros_like(row))

        arange = torch.arange(num_nodes, device=row.device)
        full = row.new_full((num_nodes, ), num_nodes)
        row = torch.cat([row, arange, full], dim=0)
        col = torch.cat([col, full, arange], dim=0)
        edge_index = torch.stack([row, col], dim=0)

        # new_type = edge_type.new_full((num_nodes, ), int(edge_type.max()) + 1)
        # edge_type = torch.cat([edge_type, new_type, new_type + 1], dim=0)

        for key, value in data.__iter__():
        # for key, value in data.items():
            # if key == 'edge_index' or key == 'edge_type':
            if key == 'edge_index' or key == 'edge_type' or key == 'edge_attr':
                continue

            if isinstance(value, Tensor):
                dim = data.__cat_dim__(key, value)
                size = list(value.size())

                fill_value = None
                if key == 'edge_weight':
                    size[dim] = 2 * num_nodes
                    fill_value = 1.
                elif key == 'edge_attr':
                    size[dim] = 2 * num_nodes
                    # In PyG, fill_value is 0, we use a new value to distinguish it from 
                    # non-connected nodes after densification of the adjacency
                    # fill_value = 0.
                    fill_value = edge_attr_fill
                elif key == 'x':
                    size[dim] = 1
                    # fill_value = 0.
                    fill_value = x_fill

                if fill_value is not None:
                    new_value = value.new_full(size, fill_value)
                    # We change the order from the original pytorch geometric 
                    # to enable easy access to the virtual node (data.x[0])
                    data[key] = torch.cat([new_value, value], dim=dim)
                    # data[key] = torch.cat([value, new_value], dim=dim)

        #data.edge_index = edge_index
        full = row.new_full((2, 2), num_nodes)
        data.edge_index = torch.cat([data.edge_index, full], dim=1)
        full = row.new_full((2, ), 0.)
        data.edge_attr = torch.cat([data.edge_attr, full], dim=0)
        #data.edge_type = edge_type

        if 'num_nodes' in data:
            data.num_nodes = data.num_nodes + 1

        return data