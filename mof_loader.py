from glob import glob
import os
import json

import numpy as np
import pandas as pd
import tqdm
from scipy.stats import rankdata
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
from sklearn.preprocessing import LabelBinarizer
from ase import io


def get_split_set(dataset, train_size=0.6, val_size=0.2, cv_fold=None, seed=1234):
    '''
    Main loader that will build the splits over the dataset
    '''
    data_size = len(dataset)
    if cv_fold is not None:
        fold_size = data_size // cv_fold
        unused = data_size % cv_fold
        splits = [fold_size for _ in range(cv_fold)]
    else:
        splits = list((data_size * np.array(
                            [train_size, val_size, 1 - (train_size + val_size)])
                        ).astype(int))
        unused = data_size - np.sum(splits)
    splits.append(unused)
    
    datasets = torch.utils.data.random_split(dataset, splits,
                                    generator=torch.Generator().manual_seed(seed))

    return datasets[:-1]


class mof_dataset(Dataset):
    """
    After investigating dataset creation from
    https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html,
    we have developed the following.
    MOF_data represents 188 Mo as a whole but the data this function creates is about 6.4G.
    This function aims to create '.pt' files to be processed by a model.

    params:
    -------
     - data_path (str): path to the MOFs preprocessed as json
     - processed_path (str): name of the folder containing the processed .pt data
     - transform (callable, optional): A function/transform that transforms a Data object.
                Transformed before every access.
     - pre_transform (callable, optional): A function/transform that transforms a Data object.
                Transformed being saved to disk.
     - pre_filter (callable, optional): I don't get what it means, please refer to pytorch_geometric.
     - seed (int): fixed seed for reproducibility concerns
     - reprocess (bool): whether or not to reprocess
     - (others) processing_args (dict): additional arguments related to reprocessing the dataset
    """
    def __init__(self, data_path='/scratch2/clear/ejehanno/datasets/qmof_database', # MOF_data
                 save_path='/scratch/curan/rmenegau/torch_datasets/QMOF',
                 processed_path='processed', transform=None, pre_transform=None, pre_filter=None,
                 seed=1234, reprocess=False, **processing_args):
        self.data_path = data_path
        self.save_path = self.data_path if save_path is None else save_path
        self.processed_path = processed_path
        self.seed = seed

        if not os.path.exists(self.processed_dir) or reprocess:
            self.reprocess_mof(**processing_args)

        super().__init__(data_path, transform, pre_transform, pre_filter)

        #self.data, self.slices = torch.load(self.processed_paths[0])

        # NB: very smart to use the __qualname__ which calls the arborescent name of a method
        # (eg mof_dataset.process) instead of its __name__ value. This way if `process` or `download`
        # is not redefined, nothing is called.
        # Also, very smooth to use the @property decorator to attribute te return value of a function to itself

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def raw_dir(self):
        return os.path.join(self.data_path, 'MOF_data')

    @property
    def processed_dir(self):
        return os.path.join(self.save_path, self.processed_path)

    @property
    def processed_file_names(self):
        # hardcoded to a `data.pt` file as required by pytorch
        file_names = []
        for file_name in glob(self.processed_dir + '/data*.pt'):
            file_names.append(os.path.basename(file_name))
        return file_names

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(f'{self.processed_dir}/data_{idx}.pt')

    def reprocess_mof(self, targets_path='qmof.csv', max_dist=8, max_neigh=12,
                      label_of_interest='outputs.pbe.bandgap',
                      save_distances=False,
                      node_pe=None, attention_pe=None):
        '''
        It takes ~10 min to reprocess the whole dataset.
        
        Parameters
        ----------
        `node_pe`: function graph->Tensor(num_nodes, d)
            takes a torch geometric graph as input, and returns 
            node positional encodings of size `d` for each node in the graph
    
        `attention_pe`: function graph->Tensor(num_nodes, num_nodes, d)
            function that takes a torch geometric graph as input, and returns 
            edge positional encodings of size `d` for each edge (existing or not) in the graph

        Example usage
        -------------
        ```
        from mof_loader import mof_dataset
        from positional_encoding import RandomWalkNodePE
  
        NodePE = RandomWalkNodePE(p_steps=16, beta=0.25)
        processed_path='processed_graphit'
        if NodePE is not None:
            processed_path += '_{}'.format(str(NodePE))
        print('Saving graphs in {}'.format(processed_path))
        dataset = mof_dataset(
            processed_path=processed_path, node_pe=NodePE)
        ```
        '''
        dict_file = os.path.join(self.data_path, 'atoms_labels.json') # needs to be first created
        with open(dict_file) as f:
            atom_dictionary = json.load(f)
        targets = pd.read_csv(os.path.join(self.data_path, targets_path))
        distance_gaussian = GaussianSmearing()
        data_list, elements = [], []
        edge_mean, edge_std, feat_min, feat_max = 0. ,0., np.inf, -np.inf
        for qmof_id in tqdm.tqdm(targets['qmof_id'], 'Creating graphs from MOFs'): # 100% distinct
            data = Data()

            # Get the crystal through ase
            crystal = io.read(os.path.join(self.raw_dir, f'{qmof_id}.json'))
            elements.append(list(set(crystal.get_chemical_symbols())))

            # Get the distances
            distance_matrix = crystal.get_all_distances(mic=True)
            if save_distances:
                data.distances = distance_matrix

            # Filter the neighbors with some hyperparameters max_dist, max_neigh
            mask = distance_matrix > max_dist
            masked_distances = np.ma.array(distance_matrix, mask=mask) # values above the threshold are not considered anymore
            ranked_distances = rankdata(masked_distances, method='ordinal', axis=1) # 2 equal values will have successive rank (unfair)
            ranked_distances_filt = np.where(mask, max_neigh+2, ranked_distances)
            final_neigh_dist = np.where(ranked_distances_filt > max_neigh+1, 0, distance_matrix)

            # Transform to edges info
            torch_adj_matrix = dense_to_sparse(torch.Tensor(final_neigh_dist))
            edge_index, edge_weight = torch_adj_matrix[0], torch_adj_matrix[1]
            # Add self adjacency of nodes
            data.edge_index, data.edge_weight = add_self_loops(edge_index, edge_weight, 
                                                               num_nodes=len(crystal), fill_value=0)

            # Add Label to the data attributes
            y = [targets[targets['qmof_id']==qmof_id][label_of_interest].item()]
            data.y = torch.Tensor(y)

            # Add atom id & state features
            z = crystal.get_atomic_numbers()
            data.z = torch.LongTensor(z)
            data.u = torch.Tensor(np.zeros((1,3)))

            # Save only the atom ids as node features to gain space (and embed them later)
            data.x = data.z.reshape(-1, 1)
            # Uncomment these to use the predefined atom_dictionary:
            # atom_fea = np.vstack([atom_dictionary[str(atom_nb)] for atom_nb in z])
            # data.x = torch.Tensor(atom_fea.astype(float))

            # Add node degree to node features (appears to improve perf - wtff?)
            # deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
            # deg = F.one_hot(deg, num_classes=max_neigh+2).to(torch.float)
            # x = data.x.view(-1,1) if data.x.dim()==1 else data.x
            # data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)

            # TODO: use SOAP | use SM ?

            # TODO: edge features
            if len(edge_weight) > 0:
                edge_mean += edge_weight.mean()
                edge_std += edge_weight.std()
                if edge_weight.max() > feat_max:
                    feat_max = edge_weight.max()
                if edge_weight.min() < feat_min:
                    feat_min = edge_weight.min()

            # data.edge_attr = distance_gaussian(edge_weight)

            # And finally
            data_list.append(data)

        # Normalize distances between 0 and 1
        edge_mean /= len(targets)
        edge_std /= len(targets)

        for data in data_list:
            data.edge_weight = (data.edge_weight - feat_min) / (feat_max - feat_min)
            # Save the Gaussian smearing for the network, to save space
            data.edge_attr = data.edge_weight.reshape(-1, 1)

        # Compute positional encodings per node
        if node_pe is not None:
            for data in tqdm.tqdm(data_list, desc='Computing node positional encodings'):
                # Alternatively one could store the positional encodings directly in data.x
                data.node_pe = node_pe(data)

        # Compute positional encodings per edge (GraphiT kernel matrix K)
        if attention_pe is not None:
            for data in tqdm.tqdm(data_list, desc='Computing edge positional encodings'):
                data.attention_pe = attention_pe(data)
        
        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)

        for idx, data in tqdm.tqdm(enumerate(data_list), desc='Writing to disk'):
            torch.save(data,
                os.path.join(self.processed_dir, "data_{}.pt".format(idx)),
            )




def atoms_labelizer(elements, use_env='equal', one_hot=True):
    """
    Function to define our own labellisation of the atoms
    (wink wink the types of C for graphene and diamond)

    params
    ------
     - use_env (bool): whether the atom label depends on its environment or not
     - one_hot (bool): whether to consider some atoms as closer to each other or all equally different
    """
    all_atoms = list(set(sum(elements, []))) # concatenate and take all unique symbols
    all_atoms.sort()
    lb = LabelBinarizer()
    lb.fit(all_atoms)
    #data.x = torch.Tensor(lb.transform(crystal.get_chemical_symbols()))
    # Write to a `atoms_labels.json` file
    return lb


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0., stop=1., resolution=50, width=.2, **kwargs):
        super().__init__()
        offset = torch.linspace(start, stop, resolution)
        self.coeff = -0.5 / ((stop - start) * width) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset.view(1,-1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
