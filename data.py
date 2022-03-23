# from torch.utils.data import Datasets
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import torch_geometric.utils as utils
from torch_geometric.transforms import ToDense


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
        # if pe == 'adj':
        #     print('Computing positional encodings')
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
            self.node_pe_list.append(node_pe(g))
        self.use_node_pe = True

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
            #print([len(g.x) for g in batch])
            #print(max_len)

            padded_x = torch.zeros((len(batch), max_len), dtype=int)
            padded_adj = torch.zeros((len(batch), max_len, max_len), dtype=int)
            mask = torch.zeros((len(batch), max_len), dtype=bool)
            labels = []
            attention_pe = None
            padded_p = None
            if self.use_node_pe:
                padded_p = torch.zeros((len(batch), max_len, 16), dtype=float)
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

        self.embedding_dimension = self.p_steps

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

    def __call__(self, graph):
        num_nodes = len(graph.x)
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        RW = A / A.sum(dim=-1) # A D^-1
        I = torch.eye(num_nodes)
        L = I - RW
        k_RW = I - self.beta * L
        attention_pe = k_RW
        for power in range(self.p_steps-1):
            attention_pe = attention_pe @ k_RW
        return attention_pe

NodePositionalEmbeddings = {
    'rand_walk': RandomWalkNodePE
}

AttentionPositionalEmbeddings = {
    'rand_walk': RandomWalkAttentionPE
}