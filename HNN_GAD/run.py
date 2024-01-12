#!/usr/bin/env python
from scipy.sparse import data
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import scipy.io
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from datetime import datetime
import argparse

from networks.main import build_network

from data import load_data
from data import normalize_adj
from optim.radam import RiemannianAdam
from torch_geometric.loader import NeighborLoader

import manifolds
from manifolds.utils import acosh
from geoopt import ManifoldParameter

from config import parser
args = parser.parse_args()

# contextual loss
def attribute_loss(manifold, attrs, X_hat):
    if args.manifold == "Euclidean":
        diff_attribute = torch.pow(X_hat - attrs, 2)
        attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1)+1e-10)
        attribute_cost = torch.mean(attribute_reconstruction_errors)
    if args.manifold == "Lorentz" or args.manifold == "PoincareBall":
        attribute_reconstruction_errors = manifold.dist(X_hat, attrs)
        attribute_cost = torch.mean(attribute_reconstruction_errors)
    return attribute_reconstruction_errors

# structural loss
def structure_loss(manifold, adj, dist_matrix, r, t):
    dist_matrix = (dist_matrix-r)/t
    err1 = torch.sum(torch.log(1+torch.exp(-dist_matrix))*(1-adj),1)
    err2 = torch.sum(dist_matrix*adj,1) + torch.sum(torch.log(1+torch.exp(-dist_matrix))*adj,1) 
    structure_reconstruction_errors = err1 / torch.sum((1-adj),1)  + err2 / torch.sum(adj,1)
    return structure_reconstruction_errors
  
# loss or outlier score
def loss_func(manifold, alpha, attrs, X_hat, adj, dist_matrix, r, t):
    attribute_reconstruction_errors = attribute_loss(manifold, attrs, X_hat)
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    structure_reconstruction_errors = structure_loss(manifold, adj, dist_matrix, r, t)
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost =  (1-alpha) * structure_reconstruction_errors + alpha * attribute_reconstruction_errors
    return cost, structure_cost, attribute_cost

# the squared pairwise distance between each node in X1 and X2, where each row of X1 and X2 is the embedding/feature for one node
def pairwise_dist2(X1, X2):        
    if args.manifold == "Euclidean":
        norm1 = torch.sum(torch.pow(X1, 2), 1)
        norm2 = torch.sum(torch.pow(X2, 2), 1)
        dist_matrix = torch.add(norm1.unsqueeze(1), norm2.unsqueeze(0)) - 2* (X1 @ X2.T)
        
    if args.manifold == "Lorentz":
        inner = -X1[:,0].unsqueeze(1) @ X2[:,0].unsqueeze(0) + X1[:,1:] @ X2[:,1:].T
        dist_matrix = acosh(-inner)
        dist_matrix = dist_matrix.pow(2)
        
    if args.manifold == "PoincareBall":
        norm1 = torch.sum(torch.pow(X1, 2), 1)
        norm2 = torch.sum(torch.pow(X2, 2), 1)
        numer = torch.add(norm1.unsqueeze(1), norm2.unsqueeze(0)) - 2* (X1 @ X2.T)
        numer = numer.clamp_min(1e-10)
        denom1 = (1-norm1).clamp_min(1e-10).unsqueeze(1)
        denom2 = (1-norm2).clamp_min(1e-10).unsqueeze(1)
        denom = denom1 @ denom2.T
        dist_matrix = acosh(1 + 2* numer/denom)
        dist_matrix = dist_matrix.pow(2)

    return dist_matrix


def Trainer(args):
    device = f"cuda:{args.cuda}" if args.cuda != -1 else "cpu"
    cpu = torch.device("cpu")
    
    # load data
    print("Loading data", args.dataset_name)
    print("Outlier type is", args.outlier_type)
    data, labels = load_data(args)
    num_nodes = data.num_nodes
    attrs = data.x
    edge_index = data.edge_index
    adj = torch.zeros((data.num_nodes, data.num_nodes)).to(cpu)
    adj[edge_index[0], edge_index[1]] = 1
    adj += torch.diag(torch.diag(adj)==0)
    if args.net_name=='GNN_mp':
        adj_mp = normalize_adj(adj.detach().numpy()).toarray()
        adj_mp = torch.FloatTensor(adj_mp).to(device)
    
    args.layer_dims[0][0] = attrs.shape[1]
    args.layer_dims[1][-1] = attrs.shape[1]
    
    if args.batch_size==-1:
        batch_size = attrs.shape[0]
    elif args.net_name=='GNN':
        batch_size = args.batch_size
    elif args.net_name=='GNN_mp':
        batch_size = attrs.shape[0]
        print("This training will be using all data in one batch.")
    print("Batch size is", batch_size)  
        
    num_edge = torch.sum(adj,1)
    num_noedge = num_nodes-num_edge

    # build model
    print("Building model", args.net_name)
    model = build_network(args)
    model = model.to(device)
      
    # optimizer
    if (args.optimizer == 'adam'):
        print("Using Adam optimizer")
        optimizer='adam'
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                           amsgrad=optimizer == 'amsgrad')
        
    elif (args.optimizer == 'radam'):
        print("Using RAdam optimizer")
        no_decay = ['bias', 'scale']
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(
                    nd in n
                    for nd in no_decay) and not isinstance(p, ManifoldParameter)
            ],
            'weight_decay':
            args.weight_decay
        }, {
            'params': [
                p for n, p in model.named_parameters() if p.requires_grad and any(
                    nd in n
                    for nd in no_decay) or isinstance(p, ManifoldParameter)
            ],
            'weight_decay':
            0.0
        }]
        optimizer = RiemannianAdam(params=optimizer_grouped_parameters,
                            lr=args.lr,
                            stabilize=10)
    
    # exponential map of feature matrix
    if args.manifold == "Euclidean":
        manifold = None
    if args.manifold == "Lorentz":
        manifold = getattr(manifolds, args.manifold)()
        o = torch.zeros_like(attrs)
        attrs = torch.cat([o[:, 0:1], attrs], dim=1)
        attrs = manifold.expmap0(attrs)
    if args.manifold == "PoincareBall":
        manifold = getattr(manifolds, args.manifold)()
        attrs = manifold.proj_tan0(attrs, args.c)
        attrs = manifold.expmap0(attrs, args.c)
        attrs = manifold.proj(attrs, args.c)
        
    # training
    print("Training model")
    model.train()
    
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        l = 0
        score = np.zeros(num_nodes)
        idx = np.random.permutation(num_nodes)  
        for i in range(-(-num_nodes//batch_size)):
            batch = idx[i*batch_size:(i+1)*batch_size]
            adj_batch = adj[batch][:,batch].to(device)
            attrs_batch = attrs[batch].to(device)
            if args.net_name=='GNN_mp':
                adj_mp_batch = adj_mp[batch][:,batch].to(device)
            else:
                adj_mp_batch = None
                
            X_hat, struc_decode = model(attrs_batch, adj_mp_batch)
            dist_matrix = pairwise_dist2(struc_decode, struc_decode)

            loss, struct_loss, feat_loss = loss_func(manifold, args.alpha, attrs_batch, X_hat, adj_batch, dist_matrix, args.r, args.t)
            l = torch.mean(loss)
            l.backward()
            optimizer.step()  
            score[batch] = loss.detach().cpu().numpy()

        if epoch%10==0:
            with torch.no_grad():
                print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "train/struct_loss=", "{:.5f}".format(struct_loss.item()),"train/feat_loss=", "{:.5f}".format(feat_loss.item()))

    # testing
    print("Testing model")
    model.eval()
                
    idx = np.arange(num_nodes)
    decode_dim = struc_decode.shape[1]
    hidden = torch.zeros((num_nodes, decode_dim)).to(device)
    score = np.zeros(num_nodes)
    
    for i in range(-(-num_nodes//batch_size)):
        batch = idx[i*batch_size:(i+1)*batch_size]
        attrs_batch = attrs[batch].to(device)
        adj_batch = adj[batch][:,batch].to(device)
        if args.net_name=='GNN_mp':
            adj_mp_batch = adj_mp[batch][:,batch].to(device)
        else:
            adj_mp_batch = None

        X_hat, struc_decode = model(attrs_batch, adj_mp_batch)
        hidden[batch] = struc_decode
        
        # test contextual loss
        cntxt_loss = args.alpha * attribute_loss(manifold, attrs_batch, X_hat)
        score[batch] += cntxt_loss.detach().cpu().numpy()
    
    # test structural loss (two batches are used to avoid out of memory)
    for i in range(-(-num_nodes//batch_size)):
        batch1 = idx[i*batch_size:(i+1)*batch_size]
        err1 = torch.zeros(batch1.shape[0]).to(cpu)
        err2 = torch.zeros(batch1.shape[0]).to(cpu)
        X1 = hidden[batch1]
        for j in range(-(-num_nodes//batch_size)):
            batch2 = idx[j*batch_size:(j+1)*batch_size]
            adj_batch = adj[batch1][:,batch2]
            X2 = hidden[batch2]
            
            dist_matrix = pairwise_dist2(X1, X2).to(cpu)
            dist_matrix = (dist_matrix-args.r)/args.t
            err1 += torch.sum(torch.log(1+torch.exp(-dist_matrix))*(1-adj_batch),1) 
            err2 += torch.sum(dist_matrix*adj_batch,1) + torch.sum(torch.log(1+torch.exp(-dist_matrix))*adj_batch,1)
        struc_loss = err1/num_noedge[batch1] + err2/num_edge[batch1]
        score[batch1] += ((1-args.alpha)*struc_loss).detach().numpy()
    
    print("AUC:", np.round(roc_auc_score(labels, score),3), "AP:", np.round(average_precision_score(labels, score),3))
    return
        
if __name__ == '__main__':
    Trainer(args)
    
