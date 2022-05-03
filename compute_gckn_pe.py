import os
import pickle
import torch
import torch.nn.functional as F
import torch_geometric.utils as utils
from gckn.data import PathLoader, S2VGraph
# from transformer.data import atom_one_hot
from gckn.models import PathSequential

class OneHotEdges(object):
    def __init__(self, num_edge_classes):
        self.num_edge_classes = num_edge_classes
        

    def __call__(self, data):
        data.edge_attr = F.one_hot(data.edge_attr.long()-1, self.num_edge_classes) 
        return data

def atom_one_hot(nodes, num_atom_types):	
    if isinstance(num_atom_types, int):	
        return F.one_hot(nodes.view(-1).long(), num_atom_types)	
    all_one_hot_feat = []	
    for col in range(len(num_atom_types)):	
        one_hot_feat = F.one_hot(nodes[:, col], num_atom_types[col])	
        all_one_hot_feat.append(one_hot_feat)	
    all_one_hot_feat = torch.cat(all_one_hot_feat, dim=1)	
    return all_one_hot_feat


def compute_gckn_pe(train_graphs, test_graphs=None, path_size=3, hidden_size=32,
                    batch_size=64, sigma=0.5, pooling='mean',
                    aggregation=True, normalize=False, use_cuda=False, encode_edges=False, dim_edges=0):
    data_loader = PathLoader(train_graphs, path_size, batch_size, aggregation=aggregation, encode_edges=encode_edges)
    input_size = data_loader.input_size
    input_edge_size = dim_edges
    data_loader.get_all_paths()

    model = PathSequential(input_size, [hidden_size], [path_size],
        kernel_args_list=[sigma], pooling=pooling, aggregation=aggregation,
        encode_edges=encode_edges, input_edge_size=input_edge_size)

    model.unsup_train(data_loader, n_sampling_paths=300000, use_cuda=use_cuda)

    scaler = None
    if normalize:
        from sklearn.preprocessing import StandardScaler
        train_pe, train_pe_edges = model.encode(data_loader, use_cuda=use_cuda)

        if train_pe_edges is not None:
            n_nodes = [pe.shape[0] for pe in train_pe]
            train_cat = torch.cat(train_pe, dim = 0)
            train_cat_edges = torch.cat(train_pe_edges, dim=0)
            train_pe = torch.cat((train_cat, train_cat_edges), dim=1)
            train_pe = torch.split(train_pe, n_nodes)

        scaler = StandardScaler()
        scaler.fit(torch.cat(train_pe, dim=0).numpy())
        train_pe = [torch.from_numpy(
            scaler.transform(pe.numpy())) for pe in train_pe]
    else:
        train_pe, train_pe_edges = model.encode(data_loader, use_cuda=use_cuda)
        if train_pe_edges is not None:
            train_pe = torch.cat((train_pe, train_pe_edges), dim=1)

    if test_graphs is not None:
        data_loader = PathLoader(test_graphs, path_size, batch_size, aggregation=aggregation, encode_edges = encode_edges)
        data_loader.get_all_paths()
        test_pe, test_pe_edges = model.encode(data_loader, use_cuda=use_cuda)

        if test_pe_edges is not None:
            n_nodes = [pe.shape[0] for pe in test_pe]
            test_cat = torch.cat(test_pe, dim = 0)
            test_cat_edges = torch.cat(test_pe_edges, dim=0)
            test_pe = torch.cat((test_cat, test_cat_edges), dim=1)
            test_pe = torch.split(test_pe, n_nodes)

        if scaler is not None:
            test_pe = [torch.from_numpy(
                scaler.transform(pe.numpy())) for pe in test_pe]
        return train_pe + test_pe
    return train_pe, None


def get_adj_list(g):
    neighbors = [[] for _ in range(g.num_nodes)]
    for k in range(g.edge_index.shape[-1]):
        i, j = g.edge_index[:, k]
        neighbors[i.item()].append(j.item())
    return neighbors

def convert_dataset(dataset, n_tags=None):
    """Convert a pytorch geometric dataset to gckn dataset
    """
    if dataset is None:
        return dataset
    graph_list = []
    for i, g in enumerate(dataset):
        new_g = S2VGraph(g, g.y)
        new_g.neighbors = get_adj_list(g)
        if n_tags is not None:
            # new_g.node_features = F.one_hot(g.x.view(-1).long(), n_tags).numpy()
             new_g.node_features = atom_one_hot(g.x, n_tags).numpy()
        else:
            new_g.node_features = g.x.numpy()
        degree_list = utils.degree(g.edge_index[0], g.num_nodes).numpy()
        new_g.max_neighbor = max(degree_list)
        new_g.mean_neighbor = (sum(degree_list) + len(degree_list) - 1) // len(degree_list)
        
        # testing to get edge_attr
        if g.edge_attr is not None:
            new_g.edge_features = g.edge_attr
            new_g.edge_index = g.edge_index

        graph_list.append(new_g)
    return graph_list


class GCKNEncoding(object):
    def __init__(self, savepath, dim, path_size, sigma=0.6, pooling='sum', aggregation=True,
                 normalize=True, encode_edges=False, dim_edges=0):
        self.savepath = savepath
        self.dim = dim
        self.path_size = path_size
        self.sigma = sigma
        self.pooling = pooling
        self.aggregation = aggregation
        self.normalize = normalize
        self.encode_edges = encode_edges
        self.dim_edges = dim_edges

        self.pos_enc_dim = dim
        if aggregation:
            self.pos_enc_dim = path_size * dim
        
        if self.encode_edges:
            if aggregation:
                self.pos_enc_dim += (path_size-1) * dim
            else:
                self.pos_enc_dim += dim

    def apply_to(self, train_dset, test_dset=None, batch_size=64, n_tags=None):
        """take pytorch geometric dataest as input
        """
        saved_pos_enc = self.load()
        if saved_pos_enc is not None:
            dset_len = len(train_dset) if test_dset is None else len(train_dset) + len(test_dset)
            if len(saved_pos_enc) != dset_len:
                raise ValueError("Incorrect save path!")
            return saved_pos_enc
        train_dset = convert_dataset(train_dset, n_tags)
        test_dset = convert_dataset(test_dset, n_tags)
        pos_enc = compute_gckn_pe(
            train_dset, test_dset, path_size=self.path_size, hidden_size=self.dim,
            batch_size=batch_size, sigma=self.sigma, pooling=self.pooling,
            aggregation=self.aggregation, normalize=self.normalize,
            encode_edges = self.encode_edges, dim_edges=self.dim_edges)
        
        self.save(pos_enc)
        return pos_enc

    def get_embedding_dimension(self):
        return self.pos_enc_dim

    def save(self, pos_enc):
        if self.savepath is None:
            return
        if not os.path.isfile(self.savepath):
            with open(self.savepath, 'wb') as handle:
                pickle.dump(pos_enc, handle)

    def load(self):
        if not os.path.isfile(self.savepath):
            return None
        with open(self.savepath, 'rb') as handle:
            pos_enc = pickle.load(handle)
        return pos_enc

		


if __name__ == "__main__":
    from torch_geometric import datasets
    data_path = '/scratch/curan/rmenegau/torch_datasets/ZINC'
    n_tags = 28
    dset = datasets.ZINC(data_path, subset=True, split='val')
    dset = convert_dataset(dset, 28)
    from gckn.graphs import get_paths
    graphs_pe = compute_gckn_pe(dset, batch_size=64, aggregation=True)
    print(len(graphs_pe))
    print(graphs_pe[0])
    print(graphs_pe[0].shape)

