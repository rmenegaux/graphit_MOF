# from torch.utils.data import Datasets
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import torch_geometric.utils as utils
from torch_geometric.transforms import ToDense
from scipy.linalg import expm


class GraphDataset(object):
    def __init__(self, dataset, pe='adj', degree=False):
        """a pytorch geometric dataset as input
        """
        self.dataset = dataset
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
            self.node_pe_list.append(node_pe(i, g))
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
            input_size = batch[0].x.shape[1]
            edge_input_size = 1 if batch[0].edge_attr.dim() == 1 else batch[0].edge_attr.shape[1]

            padded_x = torch.zeros((len(batch), max_len, input_size), dtype=int)
            padded_adj = torch.zeros((len(batch), max_len, max_len, edge_input_size), dtype=int).squeeze()
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
                # edge_index = utils.add_self_loops(batch[i].edge_index, None, num_nodes =  max_len)[0]
                g = dense_transform(g)
                # print("==================")
                # print("g.x.squeeze().shape")
                # print(g.x.squeeze().shape)
                # print("g.adj.shape")
                # print(g.adj.shape)
                # print("padded_x[i]")
                # print(padded_x[i].shape)
                # print("padded_adj[i]")
                # print(padded_adj[i].shape)
                padded_x[i] = g.x#.squeeze()
                 
                # adj = utils.to_dense_adj(edge_index).squeeze()
                padded_adj[i] = g.adj#.squeeze()
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

    def __call__(self, i, graph):
        num_nodes = len(graph.x)
        edge_index = utils.add_self_loops(graph.edge_index, None, num_nodes =  num_nodes)[0] 
        A = utils.to_dense_adj(edge_index).squeeze()
        D = A.sum(dim=-1)
        A.fill_diagonal_(0)
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
        return k_RW_power

class AdjacencyAttentionPE(object):

    def __init__(self, **parameters):
        pass

    def __call__(self, graph):
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        return A


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

