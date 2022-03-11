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
        self.n_features = dataset[0].x.shape[-1]
        self.pe_list = None
        self.adj_matrix_list = None
        self.lap_pe_list = None
        self.lap_pe_dim = None
        self.degree_list = None
        # if pe == 'adj':
        #     print('Computing positional encodings')
        #     self.compute_pe()
        if degree:
            self.compute_degree()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        if self.pe_list is not None and len(self.pe_list) == len(self.dataset):
            data.pe = self.pe_list[index]
        if self.adj_matrix_list is not None and len(self.adj_matrix_list) == len(self.dataset):
            data.adj_matrix = self.adj_matrix_list[index]
        if self.degree_list is not None and len(self.degree_list) == len(self.dataset):
            data.degree = self.degree_list[index]
        return data

    def compute_degree(self):
        self.degree_list = []
        for g in self.dataset:
            deg = 1. / torch.sqrt(1. + utils.degree(g.edge_index[0], g.num_nodes))
            self.degree_list.append(deg)

    def compute_pe(self):
        self.adj_matrix_list = []
        self.pe_list = []
        for i, g in enumerate(self.dataset):
            adj_matrix = utils.to_dense_adj(g.edge_index, g.edge_attr).squeeze()
            # g.adj_matrix = adj_matrix
            self.adj_matrix_list.append(adj_matrix)
            pe = (adj_matrix > 0)
            # Normalize by degree
            pe = pe * (pe.sum(-1, keepdim=True) ** -0.5)
            self.pe_list.append(pe)
            # g.pe = pe
            self.dataset[i] = g

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

            # TODO: check if position encoding matrix is sparse
            # if it's the case, use a huge sparse matrix
            # else use a dense tensor
            pos_enc = None
            use_pe = True#hasattr(batch[0], 'pe') and batch[0].pe is not None
            if use_pe:
                pos_enc = torch.zeros((len(batch), max_len, max_len))

            for i, g in enumerate(batch):
                labels.append(g.y.view(-1))
                g = dense_transform(g)
                g_len = len(g.x)

                padded_x[i] = g.x.squeeze()
                #print('g.adj_matrix: ', g.adj_matrix.size())
                #print('padded_adj[i]: ', padded_adj[i, :g_len, :g_len].size())
                padded_adj[i] = g.adj.squeeze()
                mask[i] = g.mask
                if use_pe:
                    pos_enc[i] = padded_adj[i]
            #print('len', len((padded_x, padded_adj, mask, pos_enc, default_collate(labels))))
            return padded_x, padded_adj, mask, pos_enc, default_collate(labels)
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
