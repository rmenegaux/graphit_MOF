"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import numpy as np
import torch
import torch.nn as nn
import math
from tqdm import tqdm

def train_epoch_sparse(model, optimizer, device, data_loader, epoch, evaluator):
    model.train()
    
    epoch_loss = 0
    nb_data = 0
    
    y_true = []
    y_pred = []

    for iter, (padded_x, padded_adj, padded_p, batch_mask, attention_pe, batch_targets) in enumerate(data_loader):
        optimizer.zero_grad()

        padded_x = padded_x.to(device)
        padded_adj = padded_adj.to(device)
        batch_mask = batch_mask.to(device)
        if attention_pe is not None:
            attention_pe = attention_pe.to(device)
        batch_targets = batch_targets.to(device) #batch_targets.flatten().to(device)
        if padded_p is not None:
            padded_p = padded_p.float().to(device)

        optimizer.zero_grad()
        batch_scores = model.forward(padded_x, padded_p, padded_adj, k_RW=attention_pe, mask=batch_mask)
        # ignore nan labels (unlabeled) when computing training loss
        is_labeled = batch_targets == batch_targets
        loss = model.loss(batch_scores.to(torch.float32)[is_labeled], batch_targets.to(torch.float32)[is_labeled])
        
        loss.backward()
        optimizer.step()
        
        y_true.append(batch_targets.view(batch_scores.shape).detach().cpu())
        y_pred.append(batch_scores.detach().cpu())
        
        epoch_loss += loss.detach().item()
        nb_data += batch_targets.size(0)
    
    epoch_loss /= (iter + 1)
    
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    # compute performance metric using OGB evaluator
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    perf = evaluator.eval(input_dict)

        
    return_perf = perf['rocauc']
    
    return epoch_loss, return_perf, optimizer

def evaluate_network_sparse(model, device, data_loader, epoch, evaluator):
    model.eval()

    epoch_loss = 0
    nb_data = 0

    y_true = []
    y_pred = []
    
    out_graphs_for_lapeig_viz = []
    
    with torch.no_grad():
        # (batch_graphs, batch_labels, batch_snorm_n)
        for iter,  (padded_x, padded_adj, padded_p, batch_mask, attention_pe, batch_targets)  in enumerate(data_loader):
            padded_x = padded_x.to(device)
            padded_adj = padded_adj.to(device)
            batch_mask = batch_mask.to(device)
            if attention_pe is not None:
                attention_pe = attention_pe.to(device)
            batch_targets = batch_targets.to(device) #batch_targets.flatten().to(device)
            if padded_p is not None:
                padded_p = padded_p.float().to(device)

            batch_scores = model.forward(padded_x, padded_p, padded_adj, k_RW=attention_pe, mask=batch_mask)
            # ignore nan labels (unlabeled) when computing loss
            is_labeled = batch_targets == batch_targets
            loss = model.loss(batch_scores.to(torch.float32)[is_labeled], batch_targets.to(torch.float32)[is_labeled])
            
            y_true.append(batch_targets.view(batch_scores.shape).detach().cpu())
            y_pred.append(batch_scores.detach().cpu())

            epoch_loss += loss.detach().item()
            nb_data += batch_targets.size(0)
            
            
        epoch_loss /= (iter + 1)

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    # compute performance metric using OGB evaluator
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    perf = evaluator.eval(input_dict)
        
    return_perf = perf['rocauc']
        
    return epoch_loss, return_perf, None