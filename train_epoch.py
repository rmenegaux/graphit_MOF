"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



# from train.metrics import MAE

def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    return MAE

def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    gpu_mem = 0
    for iter, batch_graphs in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_targets = batch_graphs.y.flatten().to(device)
        optimizer.zero_grad()
        batch_pos_enc = None
        
        # if model.pe_init == 'lap_pe':
        #     sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
        #     sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
        #     batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)

        batch_scores = model.forward(batch_graphs, batch_pos_enc).flatten()

        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
        # print(epoch_train_mae)
        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    
    return epoch_loss, epoch_train_mae, optimizer

def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0
    out_graphs_for_lapeig_viz = []
    with torch.no_grad():
        for iter, batch_graphs in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_targets = batch_graphs.y.flatten().to(device)

            # try:
            #     batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
            # except KeyError:
            #     batch_pos_enc = None
            batch_pos_enc = None
                
            batch_scores = model.forward(batch_graphs, batch_pos_enc).flatten()

            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
            
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        
    return epoch_test_loss, epoch_test_mae, None
