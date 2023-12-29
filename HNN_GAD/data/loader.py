"""Data utils functions for pre-processing and data loading."""
import os
import math
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy
import pickle as pkl
import sys
from sklearn import preprocessing
import json

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import coalesce
from torch_geometric.io import read_npz
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_undirected

from ogb.nodeproppred import PygNodePropPredDataset

from scipy.stats import ortho_group
import matplotlib.pyplot as plt

cpu = torch.device("cpu")


def preprocess_features(features, method="l2"):
    """Row-normalize feature matrix and convert to tuple representation"""
    if method=="l2":
        rowsum = np.array(np.sqrt((features**2).sum(1)))
    elif method=="l1":
        rowsum = np.array(features.sum(1))
    rowsum = (rowsum==0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()

def preset_parameters(args):
    """preset parameters for outlier node generation"""
    num_outlier_dict = {"squirrel":280, "chameleon":120, "actor":390, "cora":140, "citeseer":180, "pubmed":1000, "amazon":700, "flickr":4480, "ogbn-arxiv":8400}
    m_dict = {"squirrel":70, "chameleon":30, "actor":15, "cora":10, "citeseer":10, "pubmed":10, "amazon":70, "flickr":20, "ogbn-arxiv":30}
    
    outlier_num = num_outlier_dict[args.dataset_name]
    args.struc_clique_size = m_dict[args.dataset_name]
    args.struc_drop_prob = 0.2
    args.sample_size = m_dict[args.dataset_name]
    args.dice_ratio = 0.5
    return args, outlier_num


def gen_structural_outliers(data, m, n, p=0.2, y_outlier=None, random_state=None):
    
    if y_outlier is not None:
        node_set = set(list(np.where(y_outlier==0)[0]))
    else:
        y_outlier = np.zeros(data.num_nodes)
        node_set = set(range(data.num_nodes))   
        
    r = np.random.RandomState(random_state)
    outlier_idx = r.choice(list(node_set), size=m * n, replace=False)
    y_outlier[outlier_idx] = 1
    
    new_edges = []
    for i in range(0, n):
        for j in range(m * i, m * (i + 1)):
            for k in range(m * i, m * (i + 1)):
                if j != k:
                    node1, node2 = outlier_idx[j], outlier_idx[k]
                    new_edges.append(torch.tensor([[node1, node2]], dtype=torch.long))
    new_edges = torch.cat(new_edges)

    if p != 0:
        indices = torch.randperm(len(new_edges))[:int((1-p) * len(new_edges))]
        new_edges = new_edges[indices]

    data.edge_index = torch.cat([data.edge_index, new_edges.T], dim=1)

    return data, y_outlier

def gen_contextual_outliers(data, n, k, random_state=None):
    node_set = set(range(data.num_nodes))
    r = np.random.RandomState(random_state)
    outlier_idx = r.choice(list(node_set), size=n, replace=False)
    candidate_set = node_set.difference(set(outlier_idx))
    candidate_idx = r.choice(list(candidate_set), size=n * k)
    y_outlier = np.zeros(data.num_nodes)
    y_outlier[outlier_idx] = 1

    for i, idx in enumerate(outlier_idx):
        cur_candidates = candidate_idx[k * i: k * (i + 1)]
        euclidean_dist = torch.cdist(data.x[idx].unsqueeze(0), data.x[list(
            cur_candidates)])
        max_dist_idx = torch.argmax(euclidean_dist)
        max_dist_node = list(cur_candidates)[max_dist_idx]
        data.x[idx] = data.x[max_dist_node]

    return data, y_outlier

def gen_dice_outliers(data, n, r_perturb, y_outlier=None, random_state=None):
    r = np.random.RandomState(random_state)
    edge_index = data.edge_index
    labels = data.y.reshape(-1,)
    
    undirected_edge_index = to_undirected(edge_index)
    is_symmetric = torch.equal(undirected_edge_index, undirected_edge_index.t())
    
    if y_outlier is not None:
        node_set = set(list(np.where(y_outlier==0)[0]))
    else:
        y_outlier = np.zeros(data.num_nodes)
        node_set = set(range(data.num_nodes)) 
    outlier_idx = r.choice(list(node_set), size=n, replace=False)
    y_outlier[outlier_idx] = 1
    drop_list = []
    add_list = []
    for node1 in outlier_idx:
                
        node_drop_list = edge_index[:, edge_index[0]==node1][1]
        no_edge = (node_set - set(node_drop_list.tolist()))
        label_diff = torch.nonzero(labels!=labels[node1]).squeeze()
        node_add_list = torch.tensor(list(no_edge & set(label_diff.tolist())))

        drop_num = int(np.ceil( r_perturb * np.min((len(node_drop_list),data.num_nodes-len(node_drop_list))) ))
        
        edge_drop = r.choice(node_drop_list, size=drop_num, replace=False)
        for node2 in edge_drop:
            drop_list.append(torch.tensor([[node1, node2]], dtype=torch.long))
            if is_symmetric:
                drop_list.append(torch.tensor([[node2, node1]], dtype=torch.long))

        edge_add = r.choice(node_add_list, size=drop_num, replace=False)
        for node2 in edge_add:
            add_list.append(torch.tensor([[node1, node2]], dtype=torch.long))
            if is_symmetric:
                add_list.append(torch.tensor([[node2, node1]], dtype=torch.long))

    drop_list = torch.unique(torch.cat(drop_list), dim=0)
    add_list = torch.unique(torch.cat(add_list), dim=0)

    for ed in drop_list:
        edge_mask = (edge_index[0]!=ed[0]) | (edge_index[1]!=ed[1])
        edge_index = edge_index[:, edge_mask]
    edge_index = torch.cat([edge_index, add_list.T], dim=1)
    data.edge_index = edge_index

    return data, y_outlier

def gen_path_outliers(data, n, k, random_state=None):
    graph = to_networkx(data)
    graph = graph.to_undirected()
        
    y_outlier = np.zeros(data.num_nodes)
    r = np.random.RandomState(random_state)
    
    while sum(y_outlier) < n:
        node_set = set(list(np.where(y_outlier==0)[0]))
        outlier_idx = r.choice(list(node_set), size=n, replace=False)
        candidate_set = node_set.difference(set(outlier_idx))
        candidate_idx = r.choice(list(candidate_set), size=n * k)
    
        for i, idx in enumerate(outlier_idx):
            cur_candidates = candidate_idx[k * i: k * (i + 1)]
            path = []
            path_len = []
            for can in cur_candidates:
                try:
                    shortest_path = nx.shortest_path(graph, source=idx, target=can)
                    len_shortest_path = len(shortest_path)
                    path.append(shortest_path)
                    path_len.append(len_shortest_path)
                except:
                    path_len.append(-1)

            if np.sum(np.array(path_len)>0)>(k//2):
                max_path_idx = np.random.choice(np.where(path_len==np.max(path_len))[0])
                max_path_node = list(cur_candidates)[max_path_idx]
                data.x[idx] = data.x[max_path_node]
                y_outlier[idx] = 1

            if sum(y_outlier) >= n:
                break
    return data, y_outlier 

def gen_cont_struc_outliers(data, n, sample_size, struc_clique_size, clique_num, struc_drop_prob, random_state=None):
    data, y_outlier = gen_contextual_outliers(data, n//2, sample_size, random_state)
    data, y_outlier = gen_structural_outliers(data, struc_clique_size, clique_num, struc_drop_prob, y_outlier, random_state)
    return data, y_outlier

def gen_path_dice_outliers(data, n, sample_size, dice_ratio, random_state=None):
    data, y_outlier = gen_path_outliers(data, n//2, sample_size, random_state)
    data, y_outlier = gen_dice_outliers(data, n//2, dice_ratio, y_outlier, random_state)
    return data, y_outlier

def outlier_injection(args, data):  
    if args.outlier_preset == True:
        args, outlier_num = preset_parameters(args)
    else:
        outlier_num = int(data.num_nodes * args.outlier_ratio)
    
    if args.outlier_type=='structural':
        clique_num = math.ceil(outlier_num / args.struc_clique_size)
        data, y_outlier = gen_structural_outliers(data, args.struc_clique_size, clique_num, args.struc_drop_prob, None, args.outlier_seed)
        print('Generating structural outliers', int(np.sum(y_outlier)))
        
    elif args.outlier_type=='contextual':
        data, y_outlier = gen_contextual_outliers(data, outlier_num, args.sample_size, args.outlier_seed)
        print('Generating contextual outliers', int(np.sum(y_outlier)))
        
    elif args.outlier_type=='dice':
        data, y_outlier = gen_dice_outliers(data, outlier_num, args.dice_ratio, None, args.outlier_seed)
        print('Generating dice outliers', int(np.sum(y_outlier)))
  
    elif args.outlier_type=='path':
        data, y_outlier = gen_path_outliers(data, outlier_num, args.sample_size, args.outlier_seed)
        print('Generating path outliers', int(np.sum(y_outlier)))
        
    elif args.outlier_type=='cont_struc':
        clique_num = math.ceil(outlier_num /2 / args.struc_clique_size)
        data, y_outlier = gen_cont_struc_outliers(data, outlier_num, args.sample_size, args.struc_clique_size, clique_num, args.struc_drop_prob, args.outlier_seed)
        print('Generating contextual and structural outliers', int(np.sum(y_outlier)))  
        
    elif args.outlier_type=='path_dice':
        data, y_outlier = gen_path_dice_outliers(data, outlier_num, args.sample_size, args.dice_ratio, args.outlier_seed)
        print('Generating path and dice outliers', int(np.sum(y_outlier)))  
                
    return data, y_outlier
    

def load_new_data(args):
    dataset_str = args.dataset_name
    data_path = args.data_path+args.dataset_name
    
    graph_adjacency_list_file_path = os.path.join(data_path, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join(data_path, 'out1_node_feature_label.txt')

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}
    edge_index = []
    if dataset_str == 'actor':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])

    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                            label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                            label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))
            edge_index.append([int(line[0]), int(line[1])])
    
    features = torch.tensor([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])], dtype=torch.float)
    labels = torch.tensor([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    edge_index = torch.tensor(edge_index).t().contiguous()
    data = Data(x=features, edge_index=edge_index, y=labels)
    data.num_nodes = G.number_of_nodes()
    
    features = preprocess_features(data.x.numpy(),method="l1")
    data.x = torch.tensor(features, dtype=torch.float).to(cpu)
        
    data, y_outlier = outlier_injection(args, data)
    
    if args.featurenorm:
        features = preprocess_features(data.x.numpy(),method="l2")
        data.x = torch.tensor(features, dtype=torch.float).to(cpu)
        
    return data, y_outlier


def load_citation_data(args):
    dataset_str = args.dataset_name
    data_path = args.data_path+args.dataset_name
    
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    if dataset_str == 'citeseer':
        for i in range(allx.shape[0] + tx.shape[0], len(graph)):
            graph.pop(i)
        for i in range(len(graph)):
            graph[i] = [x for x in graph[i] if x < allx.shape[0] + tx.shape[0]]
            
    features = sp.vstack((allx, tx)).tolil()
    features = torch.tensor(features.todense(), dtype=torch.float)
    labels = np.vstack((ally, ty))
    labels = torch.from_numpy(np.argmax(labels, 1))
    edge_index = []
    for source, targets in graph.items():
        edge_index.extend([(source, target) for target in targets])
    edge_index = torch.tensor(edge_index).t().contiguous()
    data = Data(x=features, edge_index=edge_index, y=labels)
    data.num_nodes = features.shape[0]
    
    features = preprocess_features(data.x.numpy(),method="l1")
    data.x = torch.tensor(features, dtype=torch.float).to(cpu)
        
    data, y_outlier = outlier_injection(args, data)
    
    if args.featurenorm:
        features = preprocess_features(data.x.numpy(),method="l2")
        data.x = torch.tensor(features, dtype=torch.float).to(cpu)
        
    return data, y_outlier


def load_data_amazon(args):
    dataset_str = args.dataset_name
    data_path = args.data_path+args.dataset_name
    
    data = read_npz(os.path.join(data_path, f'amazon_electronics_computers.npz'))
    edge_index = data.edge_index
    
    features = preprocess_features(data.x.numpy(),method="l1")
    data.x = torch.tensor(features, dtype=torch.float).to(cpu)
        
    data, y_outlier = outlier_injection(args, data)
    
    if args.featurenorm:
        features = preprocess_features(data.x.numpy(),method="l2")
        data.x = torch.tensor(features, dtype=torch.float).to(cpu)
        
    return data, y_outlier


def load_data_flickr(args):
    dataset_str = args.dataset_name
    data_path = args.data_path+args.dataset_name
    
    f = np.load(os.path.join(data_path, 'adj_full.npz'))
    adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape']).todense()
    features = np.load(os.path.join(data_path, 'feats.npy'))
    labels = -np.ones(features.shape[0])
    with open(os.path.join(data_path, 'class_map.json')) as f:
        class_map = json.load(f)
        for key, item in class_map.items():
            labels[int(key)] = item
            
    features = torch.from_numpy(features).to(torch.float)
    labels = torch.from_numpy(labels)
    edge_index = torch.nonzero(torch.tensor(adj), as_tuple=True)
    edge_index = torch.stack(edge_index).contiguous()
    data = Data(x=features, edge_index=edge_index, y=labels)
    data.num_nodes = features.shape[0]
    
    features = preprocess_features(data.x.numpy(),method="l1")
    data.x = torch.tensor(features, dtype=torch.float).to(cpu)
        
    data, y_outlier = outlier_injection(args, data)
    
    if args.featurenorm:
        features = preprocess_features(data.x.numpy(),method="l2")
        data.x = torch.tensor(features, dtype=torch.float).to(cpu)
        
    return data, y_outlier


def load_data_ogbn(args):
    dataset_str = args.dataset_name
    data_path = args.data_path+'ogbn/'
    
    data = PygNodePropPredDataset(name=dataset_str, root=data_path)[0]
    edge_index = data.edge_index
    
    features = preprocess_features(data.x.numpy(),method="l1")
    data.x = torch.tensor(features, dtype=torch.float).to(cpu)
        
    data, y_outlier = outlier_injection(args, data)
    
    if args.featurenorm:
        features = preprocess_features(data.x.numpy(),method="l2")
        data.x = torch.tensor(features, dtype=torch.float).to(cpu)
        
    return data, y_outlier
