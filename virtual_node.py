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