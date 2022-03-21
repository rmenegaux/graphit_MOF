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
        # if pe == 'adj':
        #     print('Computing positional encodings')
        self.compute_pe()
        if degree:
            self.compute_degree()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        if self.node_pe_list is not None and len(self.node_pe_list) == len(self.dataset):
            data.node_pe = self.node_pe_list[index]
            data.attention_pe = self.attention_pe_list[index]
            #print('pe: ', data.pe.size())
            #print('x: ', data.x.size())
        # if self.adj_matrix_list is not None and len(self.adj_matrix_list) == len(self.dataset):
        #     data.adj_matrix = self.adj_matrix_list[index]
        # if self.degree_list is not None and len(self.degree_list) == len(self.dataset):
        #     data.degree = self.degree_list[index]
        return data

    def compute_degree(self):
        self.degree_list = []
        for g in self.dataset:
            deg = 1. / torch.sqrt(1. + utils.degree(g.edge_index[0], g.num_nodes))
            self.degree_list.append(deg)

    def compute_pe(self):
        self.node_pe_list = []
        self.attention_pe_list = []
        for i, g in enumerate(self.dataset):
            node_pe, attention_pe = compute_pe_RW(g)
            self.node_pe_list.append(node_pe)
            self.attention_pe_list.append(attention_pe)
            # adj_matrix = utils.to_dense_adj(g.edge_index, g.edge_attr).squeeze()
            # # g.adj_matrix = adj_matrix
            # self.adj_matrix_list.append(adj_matrix)
            # pe = (adj_matrix > 0)
            # # Normalize by degree
            # pe = pe * (pe.sum(-1, keepdim=True) ** -0.5)
            # self.pe_list.append(pe)
            # # g.pe = pe
            # self.dataset[i] = g

    def collate_fn(self):
        def collate(batch):
            batch = list(batch)
            max_len = max(len(g.x) for g in batch)
            dense_transform = ToDense(max_len)
            #print([len(g.x) for g in batch])
            #print(max_len)

            padded_x = torch.zeros((len(batch), max_len), dtype=int)
            padded_p = torch.zeros((len(batch), max_len, 16), dtype=float)
            padded_adj = torch.zeros((len(batch), max_len, max_len), dtype=int)
            mask = torch.zeros((len(batch), max_len), dtype=bool)
            labels = []

            # TODO: check if position encoding matrix is sparse
            # if it's the case, use a huge sparse matrix
            # else use a dense tensor
            pos_enc = None
            use_pe = True # hasattr(batch[0], 'pe') and batch[0].pe is not None
            if use_pe:
                pos_enc = torch.zeros((len(batch), max_len, max_len))

            for i, g in enumerate(batch):
                labels.append(g.y.view(-1))
                num_nodes = len(g.x)
                g = dense_transform(g)
                padded_x[i] = g.x.squeeze()
                padded_adj[i] = g.adj.squeeze()
                padded_p[i, :num_nodes] = g.node_pe
                mask[i] = g.mask
                if use_pe:
                    pos_enc[i, :num_nodes, :num_nodes] = g.attention_pe
            return padded_x, padded_adj, padded_p.float(), mask, pos_enc, default_collate(labels)
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

def compute_pe_RW(graph, p=16):
    num_nodes = len(graph.x)
    A = utils.to_dense_adj(graph.edge_index).squeeze()
    D = A.sum(dim=-1)
    RW = A / D
    RW_power = RW
    node_pe = torch.zeros((num_nodes, p))
    node_pe[:, 0] = RW.diagonal()
    for power in range(p-1):
        RW_power = RW @ RW_power
        node_pe[:, power + 1] = RW_power.diagonal()
    I = torch.eye(num_nodes)
    L = I - RW
    attention_pe = I - 0.25 * L
    return node_pe, attention_pe