




"""
    IMPORTING LIBS
"""
import numpy as np
import os
import time
import random
import glob
import argparse, json

import torch

import torch.optim as optim
from torch.utils.data import DataLoader
#import torch_geometric

from tqdm import tqdm

import wandb


"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from transformer_net import GraphiTNet
from perceiver_net import GraphiTPerceiverNet
from data import GraphDataset
from positional_encoding import NodePositionalEmbeddings, AttentionPositionalEmbeddings
# from compute_gckn_pe import OneHotEdges, GCKNEncoding



"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


#!/usr/bin/env python
# coding: utf-8

from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter

# save_run_tensorboard = True


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")

"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(model, MODEL_NAME):
    # model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param

"""
    GCKN CODE
"""

def precompute_gckn_embeddings_ZINC(node_pe_params, cache_dir):
    n_tags = 28
    if node_pe_params['encode_edges']:
        num_edge_features = 3
        train_dset, val_dset, test_dset = (
            ZINC(ZINC_PATH, subset=True, split=split, transform=OneHotEdges(num_edge_features)) for split in ['train', 'val', 'test']
            )
    else:
        num_edge_features = 0
        train_dset, val_dset, test_dset = (
            ZINC(ZINC_PATH, subset=True, split=split) for split in ['train', 'val', 'test']
            )
    gckn_pos_enc_path = '{}/zinc_gckn_{}_{}_{}_{}_{}_{}_{}.pkl'.format(
        cache_dir,
        node_pe_params['dim'], node_pe_params['path'], node_pe_params['sigma'],
        node_pe_params['pooling'], node_pe_params['aggregation'],
        node_pe_params['normalize'], node_pe_params['encode_edges'])
    gckn_pos_encoder = GCKNEncoding(
        gckn_pos_enc_path, node_pe_params['dim'], node_pe_params['path'],
        sigma=node_pe_params['sigma'], pooling=node_pe_params['pooling'], aggregation=node_pe_params['aggregation'],
        normalize=node_pe_params['normalize'], encode_edges=node_pe_params['encode_edges'], dim_edges=num_edge_features)
    print('GCKN Position encoding')
    gckn_pos_enc_values = gckn_pos_encoder.apply_to(
        train_dset, val_dset + test_dset, batch_size=64, n_tags=n_tags)
    NodePE = NodePositionalEmbeddings['gckn'](gckn_pos_enc_values, embedding_dimension=gckn_pos_encoder.pos_enc_dim)
    del gckn_pos_encoder
    return NodePE

"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, checkpoint_file):
    t0 = time.time()
    per_epoch_time = []
        
    DATASET_NAME = 'QMOF'

    timer = Timer()
    print('Initializing {} dataset'.format(DATASET_NAME))
    timer.start()
    trainset, valset, testset = dataset['train'], dataset['val'], dataset['test']
    timer.stop()

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file, cache_dir = dirs

    # Add virtual node connected to everyone
    if net_params['virtual_node'] == True:
        net_params['num_atom_type'] += 1
        net_params['num_bond_type'] += 1
        for dset in [trainset, valset, testset]:
            dset.add_virtual_nodes(x_fill=net_params['num_atom_type'], edge_attr_fill=net_params['num_bond_type'])

    device = net_params['device']
       
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    
    # if params["save_run_tensorboard"]:
    #     writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
        torch.cuda.manual_seed_all(params['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))

    # model = gnn_model(MODEL_NAME, net_params)
    if MODEL_NAME == 'GraphiT':
        model= GraphiTNet(net_params)
    elif MODEL_NAME == 'GraphiT_perceiver':  
        model = GraphiTPerceiverNet(net_params)
    net_params['total_param'] = view_model_param(model, MODEL_NAME)
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))

    model.load_state_dict(torch.load(checkpoint_file))
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #                                                  factor=params['lr_reduce_factor'],
    #                                                  patience=params['lr_schedule_patience'],
    #                                                  verbose=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50,
                                                     verbose=True)
    def lr_scheduler(s):
        if s < params['warmup']:
            lr = 1e-6 + s * (params['init_lr'] - 1e-6) / params['warmup']
        else:
            lr = params['init_lr']
        return lr

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_MAEs, epoch_val_MAEs = [], [] 
    
    # import train functions for all GNNs
    from train_epoch import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network
    
    train_loader = DataLoader(trainset, num_workers=4, batch_size=params['batch_size'], shuffle=True, collate_fn=trainset.collate_fn())
    val_loader = DataLoader(valset, num_workers=4, batch_size=params['batch_size'], shuffle=False, collate_fn=valset.collate_fn())
    test_loader = DataLoader(testset, num_workers=4, batch_size=params['batch_size'], shuffle=False, collate_fn=testset.collate_fn())
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()
                epoch_train_loss, epoch_train_mae, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, lr_scheduler, warmup=params['warmup'])
                epoch_val_loss, epoch_val_mae, __ = evaluate_network(model, device, val_loader, epoch)
                epoch_test_loss, epoch_test_mae, __ = evaluate_network(model, device, test_loader, epoch)
                del __
                
                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_MAEs.append(epoch_train_mae)
                epoch_val_MAEs.append(epoch_val_mae)

                #print("params['save_run_tensorboard'] : {}".format(params['save_run_tensorboard']))
                # if params['save_run_tensorboard']:
                #     writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                #     writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                #     writer.add_scalar('train/_mae', epoch_train_mae, epoch)
                #     writer.add_scalar('val/_mae', epoch_val_mae, epoch)
                #     writer.add_scalar('test/_mae', epoch_test_mae, epoch)
                #     writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                    
                wandb.log({'train/_loss': epoch_train_loss,
                          'val/_loss': epoch_val_loss,
                          'test/_mae': epoch_test_mae,
                          'learning_rate': optimizer.param_groups[0]['lr']})

                        
                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_MAE=epoch_train_mae, val_MAE=epoch_val_mae,
                              test_MAE=epoch_test_mae)

                per_epoch_time.append(time.time()-start)

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)

                # if params['warmup'] == False:
                # scheduler.step(epoch_val_loss)
                scheduler.step()
                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break

                    # prof.step()

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    test_loss_lapeig, test_mae, g_outs_test = evaluate_network(model, device, test_loader, epoch)
    train_loss_lapeig, train_mae, g_outs_train = evaluate_network(model, device, train_loader, epoch)
    
    print("Test MAE: {:.4f}".format(test_mae))
    print("Train MAE: {:.4f}".format(train_mae))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
    
    # if params["save_run_tensorboard"]:
    #     writer.close()
    wandb.finish()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST MAE: {:.4f}\nTRAIN MAE: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  test_mae, train_mae, epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))
        


def main():    
    """
        USER CONTROLS
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--checkpoint', help="Please give a value for out_dir")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--oarid', default=0, help="Please give a value for oarid")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")

    

    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']

    wandb.init(project="graphit-perceiver", entity="mselosse")

    wandb.config = config   
    if args.oarid != 0:
        wandb.run.name = args.oarid
    

    params = config['params']
    # Training parameters
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    
    # Network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    checkpoint_file = args.checkpoint

        # ---- Loading Data ----
    # dataset = LoadData(DATASET_NAME)
    # Initialize node positional embeddings
    NodePE = None
    if net_params['use_node_pe'] in ['concat', 'sum', 'product']:
        node_pe_params = net_params['node_pe_params']
        if (node_pe_params['node_pe'] not in NodePositionalEmbeddings.keys()):
            print('{} is not a recognized node positional embedding, defaulting to none')
            net_params['use_node_pe'] = False
        else:
            NodePE = NodePositionalEmbeddings[node_pe_params['node_pe']](**node_pe_params)
            net_params['pos_enc_dim'] = NodePE.get_embedding_dimension()
    else:
        net_params['use_node_pe'] = False
    # Pre-compute attention relative positional embeddings
    AttentionPE = None
    if net_params['use_attention_pe']:
        attention_pe_params = net_params['attention_pe_params']
        if (attention_pe_params['attention_pe'] not in AttentionPositionalEmbeddings.keys()):
            print('{} is not a recognized attention positional embedding, defaulting to none')
            net_params['use_attention_pe'] = False
            AttentionPE = None
        else:
            AttentionPE = AttentionPositionalEmbeddings[attention_pe_params['attention_pe']](**attention_pe_params)
            #for dset in [trainset, valset, testset]:
            #    dset.compute_attention_pe(AttentionPE)
            net_params['attention_pe_dim'] = AttentionPE.get_dimension()
            if net_params['attention_pe_dim'] > 1:
                net_params['multi_attention_pe'] = attention_pe_params['multi_attention_pe']
                # Multiple relative attention matrices, one must choose a way to deal with last dimension
                if net_params['multi_attention_pe'] not in ["per_layer", "per_head", "aggregate"]:
                    raise ValueError('''Attention PE is multi-dimensional, `multi_attention_pe` must be
                        one of ["per_layer", "per_head", "aggregate"]''')
                if net_params['multi_attention_pe'] == "per_layer" and net_params['L'] > net_params['attention_pe_dim']:
                    raise ValueError('''"multi_attention_pe" == "per_layer", so `attention_pe` last dimension ({})
                        must be at least equal to the number of layers ({})'''.format(net_params['attention_pe_dim'], net_params['L']))
                if net_params['multi_attention_pe'] == "per_head" and net_params['n_heads'] != net_params['attention_pe_dim']:
                    raise ValueError('''"multi_attention_pe" == "per_head", so `attention_pe` last dimension ({})
                        must match the number of attention heads ({})'''.format(net_params['attention_pe_dim'], net_params['n_heads']))
            else:
                net_params['multi_attention_pe'] = None

    from mof_loader import mof_dataset, get_split_set
    processed_path='processed_graphit'
    if NodePE is not None:
        processed_path += '_{}'.format(str(NodePE))
    if AttentionPE is not None:
        processed_path += '_{}'.format(str(AttentionPE))
    print('Saving graphs in {}'.format(processed_path))
    dataset = mof_dataset(
        # data_path='/scratch2/clear/ejehanno/datasets/qmof_database', # MOF_data
        data_path='/gpfsstore/rech/tbr/uho58uo/qmof_database', # MOF_data
        save_path=None,
        processed_path=processed_path, reprocess=False, node_pe=NodePE, attention_pe=AttentionPE)
    # net_params['use_node_pe'] = False
    # NodePE = None
    dataset = {
        split: GraphDataset(splitset, node_pe=NodePE, attention_pe=AttentionPE)
        for split, splitset in zip(['train', 'val', 'test'], get_split_set(dataset, train_size=0.8, val_size=0.05))
    }
    # ZINC
    # FIXME: move this to data.py
    # net_params['num_atom_type'] = dataset.num_atom_type
    # net_params['num_bond_type'] = dataset.num_bond_type
    net_params['num_atom_type'] = 100 # len(np.unique(dataset['train'][0].data.x))
    net_params['num_bond_type'] = 1 # len(np.unique(dataset['train'].data.edge_attr))
    net_params['n_classes'] = 1

    experiment_name = MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + \
    time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y') + "_" + str(args.oarid)
    print("Experiments dir : {}".format(experiment_name))


    root_log_dir = out_dir + 'logs/' + experiment_name
    root_ckpt_dir = out_dir + 'checkpoints/' + experiment_name
    write_file_name = out_dir + 'results/result_' + experiment_name
    write_config_file = out_dir + 'configs/config_' + experiment_name
    cache_dir = out_dir + 'cache/'
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file, cache_dir

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')
    
    if not os.path.exists(out_dir + 'cache'):
        os.makedirs(out_dir + 'cache')

    # net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, checkpoint_file)
   
main()