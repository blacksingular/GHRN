from dgl.data import FraudDataset
from dgl.data.utils import load_graphs
import dgl
import torch
import warnings
import pickle as pkl
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams['axes.unicode_minus']=False
import seaborn as sns
from dgl.nn.pytorch.conv import EdgeWeightNorm
import pickle as pkl
import dgl
import dgl.function as fn

warnings.filterwarnings("ignore")

class Dataset:
    def __init__(self, load_epoch, name='tfinance', del_ratio=0., homo=True, data_path='', adj_type='sym'):
        self.name = name
        graph = None
        prefix = data_path
        if name == 'tfinance':
            graph, label_dict = load_graphs(f'{prefix}/tfinance')
            graph = graph[0]
            graph.ndata['label'] = graph.ndata['label'].argmax(1)
            if del_ratio != 0.:
                graph = graph.add_self_loop()
                with open(f'probs_tfinance_BWGNN_{load_epoch}_{homo}.pkl', 'rb') as f:
                    pred_y = pkl.load(f)
                    graph.ndata['pred_y'] = pred_y
                graph = random_walk_update(graph, del_ratio, adj_type)
                graph = dgl.remove_self_loop(graph)

        elif name == 'tsocial':
            graph, label_dict = load_graphs(f'{prefix}/tsocial')
            graph = graph[0]
            if del_ratio != 0.:
                graph = graph.add_self_loop()
                with open(f'probs_tsocial_BWGNN_{load_epoch}_{homo}.pkl', 'rb') as f:
                    pred_y = pkl.load(f)
                    graph.ndata['pred_y'] = pred_y
                graph = random_walk_update(graph, del_ratio, adj_type)
                graph = dgl.remove_self_loop(graph)

        elif name == 'yelp':
            dataset = FraudDataset(name, train_size=0.4, val_size=0.2)
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
                if del_ratio != 0.:
                    with open(f'probs_yelp_BWGNN_{load_epoch}_{homo}.pkl', 'rb') as f:
                        graph.ndata['pred_y'] = pkl.load(f)
                    graph = random_walk_update(graph, del_ratio, adj_type)
                    graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
            else:
                if del_ratio != 0.:
                    with open(f'probs_yelp_BWGNN_{load_epoch}_{homo}.pkl', 'rb') as f:
                        pred_y = pkl.load(f)
                    data_dict = {}
                    flag = 1
                    for relation in graph.canonical_etypes:
                        graph_r = dgl.to_homogeneous(graph[relation], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                        graph_r = dgl.add_self_loop(graph_r)
                        graph_r.ndata['pred_y'] = pred_y
                        graph_r = random_walk_update(graph_r, del_ratio, adj_type)
                        graph_r = dgl.remove_self_loop(graph_r)
                        data_dict[('review', str(flag), 'review')] = graph_r.edges()
                        flag += 1
                    graph_new = dgl.heterograph(data_dict) 
                    graph_new.ndata['label'] = graph.ndata['label']
                    graph_new.ndata['feature'] = graph.ndata['feature']
                    graph_new.ndata['train_mask'] = graph.ndata['train_mask']
                    graph_new.ndata['val_mask'] = graph.ndata['val_mask']
                    graph_new.ndata['test_mask'] = graph.ndata['test_mask']
                    graph = graph_new

        elif name == 'amazon':
            dataset = FraudDataset(name, train_size=0.4, val_size=0.2)
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
                if del_ratio != 0.:
                    with open(f'probs_amazon_BWGNN_{load_epoch}_{homo}.pkl', 'rb') as f:
                        graph.ndata['pred_y'] = pkl.load(f)
                    graph = random_walk_update(graph, del_ratio, adj_type)
                    graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
            else:
                if del_ratio != 0.:
                    with open(f'probs_amazon_BWGNN_{load_epoch}_{homo}.pkl', 'rb') as f:
                        pred_y = pkl.load(f)
                    data_dict = {}
                    flag = 1
                    for relation in graph.canonical_etypes:
                        graph[relation].ndata['pred_y'] = pred_y
                        graph_r = dgl.add_self_loop(graph[relation])
                        graph_r = random_walk_update(graph_r, del_ratio, adj_type)
                        graph_r = dgl.remove_self_loop(graph_r)
                        data_dict[('review', str(flag), 'review')] = graph_r.edges()
                        flag += 1
                    graph_new = dgl.heterograph(data_dict) 
                    graph_new.ndata['label'] = graph.ndata['label']
                    graph_new.ndata['feature'] = graph.ndata['feature']
                    graph_new.ndata['train_mask'] = graph.ndata['train_mask']
                    graph_new.ndata['val_mask'] = graph.ndata['val_mask']
                    graph_new.ndata['test_mask'] = graph.ndata['test_mask']
                    graph = graph_new
        else:
            print('no such dataset')
            exit(1)

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        print(graph)

        self.graph = graph

def random_walk_update(graph, delete_ratio, adj_type):
    edge_weight = torch.ones(graph.num_edges())
    if adj_type == 'sym':
        norm = EdgeWeightNorm(norm='both')
    else:
        norm = EdgeWeightNorm(norm='left')
    graph.edata['w'] = norm(graph, edge_weight)
    # functions
    aggregate_fn = fn.u_mul_e('h', 'w', 'm')
    reduce_fn = fn.sum(msg='m', out='ay')

    graph.ndata['h'] = graph.ndata['pred_y']
    graph.update_all(aggregate_fn, reduce_fn)
    graph.ndata['ly'] = graph.ndata['pred_y'] - graph.ndata['ay']
    # graph.ndata['lyyl'] = torch.matmul(graph.ndata['ly'], graph.ndata['ly'].T)
    graph.apply_edges(inner_product_black)
    # graph.apply_edges(inner_product_white)
    black = graph.edata['inner_black']
    # white = graph.edata['inner_white']
    # delete
    threshold = int(delete_ratio * graph.num_edges())
    edge_to_move = set(black.sort()[1][:threshold].tolist())
    # edge_to_protect = set(white.sort()[1][-threshold:].tolist())
    edge_to_protect = set()
    graph_new = dgl.remove_edges(graph, list(edge_to_move.difference(edge_to_protect)))
    return graph_new

def inner_product_black(edges):
    return {'inner_black': (edges.src['ly'] * edges.dst['ly']).sum(axis=1)}

def inner_product_white(edges):
    return {'inner_white': (edges.src['ay'] * edges.dst['ay']).sum(axis=1)}

def find_inter(edges):
    return edges.src['label'] != edges.dst['label'] 

def cal_hetero(edges):
    return {'same': edges.src['label'] != edges.dst['label']}

def cal_hetero_normal(edges):
    return {'same_normal': (edges.src['label'] != edges.dst['label']) & (edges.src['label'] == 0)}

def cal_normal(edges):
    return {'normal': edges.src['label'] == 0}

def cal_hetero_anomal(edges):
    return {'same_anomal': (edges.src['label'] != edges.dst['label']) & (edges.src['label'] == 1)}

def cal_anomal(edges):
    return {'anomal': edges.src['label'] == 1}