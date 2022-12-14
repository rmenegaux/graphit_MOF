# -*- coding: utf-8 -*-
import torch
from torch import nn
from gckn.layers import PathLayer, NodePooling, Linear


class PathSequential(nn.Module):
    def __init__(self, input_size, hidden_sizes, path_sizes,
                 kernel_funcs=None, kernel_args_list=None,
                 pooling='mean', 
                 aggregation=False, encode_edges=False, input_edge_size=0, **kwargs):
        super(PathSequential, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.path_sizes = path_sizes
        self.n_layers = len(hidden_sizes)
        self.aggregation = aggregation
        self.encode_edges = encode_edges
        self.input_edge_size = input_edge_size


        layers = []
        output_size = hidden_sizes[-1]
        for i in range(self.n_layers):
            if kernel_funcs is None:
                kernel_func = "exp"
            else:
                kernel_func = kernel_funcs[i]
            if kernel_args_list is None:
                kernel_args = 0.5
            else:
                kernel_args = kernel_args_list[i]

            layer = PathLayer(input_size, hidden_sizes[i], path_sizes[i],
                              kernel_func, kernel_args, pooling, aggregation,
                              self.encode_edges, self.input_edge_size,
                              **kwargs)
            layers.append(layer)
            input_size = hidden_sizes[i]
            if aggregation:
                output_size *= path_sizes[i]
        self.output_size = output_size

        self.layers = nn.ModuleList(layers)

    def __getitem__(self, idx):
        return self.layers[idx]

    def __len__(self):
        return self.n_layers

    def __iter__(self):
        return iter(self.layers._modules.values())

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, features, paths_indices, other_info, edges_info=None):
        output = features
        for layer in self.layers:
            output, output_edges = layer(output, paths_indices, other_info, edges_info)
        return output, output_edges

    def representation(self, features, paths_indices, other_info, n=-1, edges_info=None):
        if n == -1:
            n = self.n_layers
        for i in range(n):
            features, features_edges = self.layers[i](features, paths_indices, other_info, edges_info)
        return features, features_edges

    def normalize_(self):
        for module in self.layers:
            module.normalize_()

    def unsup_train(self, data_loader, n_sampling_paths=100000,
                    init=None, use_cuda=False):
        self.train(False)
        for i, layer in enumerate(self.layers):
            print("Training layer {}".format(i + 1))
            n_sampled_paths = 0
            try:
                n_paths_per_batch = (
                    n_sampling_paths + len(data_loader) - 1
                    ) // len(data_loader)
            except Exception:
                n_paths_per_batch = 1000

            paths = torch.Tensor(
                n_sampling_paths, layer.path_size, layer.input_size)
            if use_cuda:
                paths = paths.cuda()

            if self.encode_edges:
                paths_edges = torch.Tensor(
                    n_sampling_paths, layer.path_size-1, self.input_edge_size)
                if use_cuda:
                    paths_edges = paths_edges.cuda()

            for data in data_loader.make_batch():
                if n_sampled_paths >= n_sampling_paths:
                    continue
                features = data['features']
                paths_indices = data['paths']
                n_paths = data['n_paths']
                n_nodes = data['n_nodes']
                if self.encode_edges:
                    edges = data['edges']
                    edge_features = data['edge_features']
                    paths_edges_indices = data['paths_edges']
                if use_cuda:
                    features = features.cuda()
                    edge_features = edge_features.cuda()
                    edges = edges.cuda()
                    if isinstance(n_paths, list):
                        paths_indices = [p.cuda() for p in paths_indices]
                        n_paths = [p.cuda() for p in n_paths]
                    else:
                        paths_indices = paths_indices.cuda()
                        n_paths = n_paths.cuda()
                    n_nodes = n_nodes.cuda()
                with torch.no_grad():
                    # edges_info=None
                    # if self.encode_edges:
                    #     edges_info={'edges': edges, 'edge_features': edge_features, 'paths_edges': paths_edges}
                    # features = self.representation(
                    #     features, paths_indices,
                    #     {'n_paths': n_paths, 'n_nodes': n_nodes}, n=i,
                    #     edges_info=edges_info)
                    paths_batch = layer.sample_paths(
                        features, paths_indices, n_paths_per_batch)
                    size = paths_batch.shape[0]
                    size = min(size, n_sampling_paths - n_sampled_paths)
                    paths[n_sampled_paths: n_sampled_paths + size
                          ] = paths_batch[:size]
                    if self.encode_edges: 
                        paths_edges_batch = layer.sample_paths_edges( 
                            edge_features, paths_edges_indices, n_paths_per_batch)
                        size_edge = paths_edges_batch.shape[0]
                        size_edge = min(size_edge, n_sampling_paths - n_sampled_paths)
                        paths_edges[n_sampled_paths: n_sampled_paths + size_edge
                              ] = paths_edges_batch[:size_edge]

                    n_sampled_paths += size

            print("total number of sampled paths: {}".format(n_sampled_paths))
            paths = paths[:n_sampled_paths]
            layer.unsup_train(paths, init=init)
            if self.encode_edges:
                paths_edges = paths_edges[:n_sampled_paths]
                layer.unsup_train_edges(paths_edges, init=init)

        return

    def encode(self, data_loader, use_cuda=False):
        if use_cuda:
            self.cuda()
        self.eval()
        output = []
        output_edges = None
        if self.encode_edges:
            output_edges = []

        for data in data_loader.make_batch(shuffle=False):
            features = data['features']
            paths_indices = data['paths']
            n_paths = data['n_paths']
            n_nodes = data['n_nodes']
            edges_info = None
            if self.encode_edges:
                edge_features = data['edge_features']
                paths_edges = data['paths_edges']
                edges = data['edges']
                edges_info={'edges': edges, 'edge_features': edge_features, 'paths_edges': paths_edges}
            size = len(n_nodes)
            if use_cuda:
                features = features.cuda()
                if isinstance(n_paths, list):
                    paths_indices = [p.cuda() for p in paths_indices]
                    n_paths = [p.cuda() for p in n_paths]
                else:
                    paths_indices = paths_indices.cuda()
                    n_paths = n_paths.cuda()
                n_nodes = n_nodes.cuda()
            with torch.no_grad():
                
                batch_out, batch_out_edges = self(features, paths_indices,
                                 {'n_paths': n_paths,
                                  'n_nodes': n_nodes},
                                  edges_info
                                 )
                batch_out = batch_out.cpu()
                batch_out = batch_out.reshape(features.shape[0], -1)
                batch_out = torch.split(batch_out, n_nodes.numpy().tolist())
                if batch_out_edges is not None:
                    batch_out_edges.cpu()
                    batch_out_edges = batch_out_edges.reshape(features.shape[0], -1)
                    batch_out_edges = torch.split(batch_out_edges, n_nodes.numpy().tolist())
                
            output.extend(batch_out)
            if self.encode_edges:
                output_edges.extend(batch_out_edges)
        return output, output_edges


class GCKNetFeature(nn.Module):
    def __init__(self, input_size, hidden_sizes, path_sizes,
                 kernel_funcs=None, kernel_args_list=None,
                 pooling='mean', global_pooling='sum',
                 heads=1, out_size=3, max_iter=100, eps=0.1,
                 aggregation=False, **kwargs):
        super().__init__()
        self.path_layers = PathSequential(
            input_size, hidden_sizes, path_sizes,
            kernel_funcs, kernel_args_list,
            pooling, aggregation, **kwargs)
        self.aggregation = aggregation
        self.global_pooling = global_pooling
        self.path_sizes = path_sizes
        self.hidden_sizes = hidden_sizes
        self.node_pooling = NodePooling(global_pooling)
        self.output_size = self.path_layers.output_size

    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()

    def forward(self, input, paths_indices, other_info):
        output = self.path_layers(input, paths_indices, other_info)
        return self.node_pooling(output, other_info)

    def unsup_train(self, data_loader, n_sampling_paths=100000,
                    n_nodes_max=100000, init=None, use_cuda=False):
        self.path_layers.unsup_train(data_loader, n_sampling_paths,
                                     init, use_cuda)

    def predict(self, data_loader, use_cuda=False):
        if use_cuda:
            self.cuda()
        self.eval()
        output = torch.Tensor(data_loader.n, self.output_size)

        batch_start = 0
        for data in data_loader.make_batch(shuffle=False):
            features = data['features']
            paths_indices = data['paths']
            n_paths = data['n_paths']
            n_nodes = data['n_nodes']
            size = len(n_nodes)
            # print(size, features.shape, len(n_paths))
            if use_cuda:
                features = features.cuda()
                if isinstance(n_paths, list):
                    paths_indices = [p.cuda() for p in paths_indices]
                    n_paths = [p.cuda() for p in n_paths]
                else:
                    paths_indices = paths_indices.cuda()
                    n_paths = n_paths.cuda()
                n_nodes = n_nodes.cuda()
            with torch.no_grad():
                batch_out, batch_out_edges = self(features, paths_indices,
                                 {'n_paths': n_paths,
                                  'n_nodes': n_nodes}
                                 ).cpu()
            output[batch_start: batch_start + size] = batch_out
            batch_start += size
        return output, data_loader.labels


class GCKNet(nn.Module):
    def __init__(self, nclass, input_size, hidden_sizes, path_sizes,
                 kernel_funcs=None, kernel_args_list=None,
                 pooling='mean', global_pooling='sum', heads=1, out_size=3,
                 max_iter=100, eps=0.1, aggregation=False, weight_decay=0.0,
                 batch_norm=False,
                 **kwargs):
        super().__init__()

        self.features = GCKNetFeature(
            input_size, hidden_sizes, path_sizes,
            kernel_funcs, kernel_args_list,
            pooling, global_pooling, heads, out_size,
            max_iter, eps, aggregation, **kwargs)
        self.output_size = self.features.output_size
        self.nclass = nclass

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(self.output_size)

        self.classifier = Linear(self.output_size, nclass, weight_decay)

    def reset_parameters(self):
        for layer in self.children():
            # if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    def representation(self, input, paths_indices, other_info):
        return self.features(input, paths_indices, other_info)

    def forward(self, input, paths_indices, other_info):
        features = self.representation(input, paths_indices, other_info)
        if self.batch_norm:
            features = self.bn_layer(features)
        return self.classifier(features)

    def unsup_train(self, data_loader, n_sampling_paths=100000,
                    init=None, use_cuda=False):
        self.features.unsup_train(data_loader=data_loader,
                                  n_sampling_paths=n_sampling_paths,
                                  init=init, use_cuda=use_cuda)

    def unsup_train_classifier(self, data_loader, criterion, use_cuda=False):
        encoded_data, labels = self.features.predict(data_loader, use_cuda)
        print(encoded_data.shape)
        self.classifier.fit(encoded_data, labels, criterion)
