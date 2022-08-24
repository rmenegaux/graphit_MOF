"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.profiler as profiler


def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    return MAE

def train_epoch_sparse(model, optimizer, device, data_loader, epoch, lr_scheduler, warmup=0):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (padded_x, padded_adj, padded_p, batch_mask, attention_pe, batch_targets) in enumerate(data_loader):
        iteration = epoch * len(data_loader) + iter
        if iteration < warmup:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)

        # with profiler.record_function("TRANSFER TO GPU"):
        padded_x = padded_x.to(device)
        padded_adj = padded_adj.to(device)
        batch_mask = batch_mask.to(device)
        if attention_pe is not None:
            attention_pe = attention_pe.to(device)
        batch_targets = batch_targets.flatten().to(device)
        if padded_p is not None:
            padded_p = padded_p.float().to(device)

        optimizer.zero_grad()

        # with profiler.record_function("FORWARD PASS"):
        batch_scores = model.forward(padded_x, padded_p, padded_adj, k_RW=attention_pe, mask=batch_mask).flatten()

        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
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
        for iter, (padded_x, padded_adj, padded_p, batch_mask, attention_pe, batch_targets) in enumerate(data_loader):
            padded_x = padded_x.to(device)
            padded_adj = padded_adj.to(device)
            batch_mask = batch_mask.to(device)
            if attention_pe is not None:
                attention_pe = attention_pe.to(device)
            batch_targets = batch_targets.flatten().to(device)
            if padded_p is not None:
                padded_p = padded_p.float().to(device)

            batch_scores = model.forward(padded_x, padded_p, padded_adj, k_RW=attention_pe, mask=batch_mask).flatten()

            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
            
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        
    return epoch_test_loss, epoch_test_mae, None
