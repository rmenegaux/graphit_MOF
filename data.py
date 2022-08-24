# from torch.utils.data import Datasets
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torch_geometric.transforms import ToDense
import torch.profiler as profiler

from virtual_node import VirtualNode


class GraphDataset(object):
    def __init__(self, dataset):
        """a pytorch geometric dataset as input
        """
        self.dataset = list(dataset)

        self.use_node_pe = False
        self.use_attention_pe = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def add_virtual_nodes(self, x_fill=21, edge_attr_fill=4):
        AddVirtualNode = VirtualNode()
        for i, g in enumerate(self.dataset):
            self.dataset[i] = AddVirtualNode(g, x_fill=x_fill, edge_attr_fill=edge_attr_fill)

    def compute_node_pe(self, node_pe):
        '''
        Takes as argument a function returning a nodewise positional embedding from a graph
        '''
        for g in self.dataset:
            g.node_pe = node_pe(g)
        self.use_node_pe = True
        self.node_pe_dimension = node_pe.get_embedding_dimension()

    def compute_attention_pe(self, attention_pe):
        '''
        Takes as argument a function returning a nodewise positional embedding from a graph
        '''
        for g in self.dataset:
            g.attention_pe = attention_pe(g)
        self.use_attention_pe = True
        self.attention_pe_dim = attention_pe.get_dimension()

    def collate_fn(self):
        def collate(batch):
            # with profiler.record_function("COLLATE FUNCTION"):
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
                attention_pe = torch.zeros((len(batch), max_len, max_len, self.attention_pe_dim)).squeeze()

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
