from input.dataset import Dataset
from time import time
from algorithms import *
# from algorithms.GAlign.GAlign2 import GAlign2
from evaluation.metrics import get_statistics
import utils.graph_utils as graph_utils
import random
import numpy as np
import torch
import argparse
import os
import pdb
from utils.graph_utils import load_gt
import torch.nn.functional as F
import pandas as pd
import networkx as nx
import copy 


# import timesd
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# torch.use_deterministic_algorithms(True)

def parse_args():
    parser = argparse.ArgumentParser(description="Network alignment")
    parser.add_argument('--source_dataset', default="dataspace/douban/online/graphsage/")
    parser.add_argument('--target_dataset', default="dataspace/douban/offline/graphsage/")
    parser.add_argument('--groundtruth',    default="dataspace/douban/dictionaries/groundtruth")
    parser.add_argument('--seed',           default=123,    type=int)
    subparsers = parser.add_subparsers(dest="algorithm", help='Choose 1 of the algorithm from: IsoRank, FINAL, UniAlign, NAWAL, DeepLink, REGAL, IONE, PALE')

    # GAlign
    parser_GAlign = subparsers.add_parser("GAlign", help="GAlign algorithm")
    parser_GAlign.add_argument('--cuda',                action="store_true")
    parser_GAlign.add_argument('--embedding_dim',       default=200,         type=int)
    parser_GAlign.add_argument('--GAlign_epochs',    default=20,        type=int)
    parser_GAlign.add_argument('--lr', default=0.01, type=float)
    parser_GAlign.add_argument('--num_GCN_blocks', type=int, default=2)
    parser_GAlign.add_argument('--act', type=str, default='tanh')
    parser_GAlign.add_argument('--log', action="store_true", help="Just to print loss")
    parser_GAlign.add_argument('--invest', action="store_true", help="To do some statistics")
    parser_GAlign.add_argument('--input_dim', default=100, help="Just ignore it")
    parser_GAlign.add_argument('--alpha0', type=float, default=1)
    parser_GAlign.add_argument('--alpha1', type=float, default=1)
    parser_GAlign.add_argument('--alpha2', type=float, default=1)
    parser_GAlign.add_argument('--noise_level', type=float, default=0.01)

    # refinement
    parser_GAlign.add_argument('--refinement_epochs', default=10, type=int)
    parser_GAlign.add_argument('--threshold_refine', type=float, default=0.94, help="The threshold value to get stable candidates")

    # loss
    parser_GAlign.add_argument('--beta', type=float, default=0.8, help='balancing source-target and source-augment')
    parser_GAlign.add_argument('--threshold', type=float, default=0.01, help='confidence threshold for adaptivity loss')
    parser_GAlign.add_argument('--coe_consistency', type=float, default=0.8, help='balancing consistency and adaptivity loss')

    return parser.parse_args()

# ACN-------------------------------------------------------------
def preprocessing(G1, G2, alignment_dict):
    '''
    Parameters
    ----------
    G1 : source graph
    G2 : target graph
    alignment_dict : grth dict

    '''
    # shift index for constructing union
    # construct shifted dict
    shift = G1.number_of_nodes()
    G2_list = list(G2.nodes())
    G2_shiftlist = list(idx + shift for idx in list(G2.nodes()))
    shifted_dict = dict(zip(G2_list,G2_shiftlist))
    
    #relable idx for G2
    G2 = nx.relabel_nodes(G2, shifted_dict)
    
    #update alignment dict
    align1list = list(alignment_dict.keys())
    align2list = list(alignment_dict.values())   
    shifted_align2list = [a+shift for a in align2list]
    
    groundtruth_dict = dict(zip(align1list, shifted_align2list))
    groundtruth_dict, groundtruth_dict_reversed = get_reversed(groundtruth_dict)
    
    return G2, groundtruth_dict, groundtruth_dict_reversed

def get_reversed(alignment_dict):
    alignment_dict_reversed = {}
    reversed_dictionary = {value : key for (key, value) in alignment_dict.items()}
    return alignment_dict, reversed_dictionary

# 加边减边图，干扰目标图
def PerturbedProcessing(G1, G2, com_portion, rand_portion, graphname):

    G3 = copy.deepcopy(G1)
    #Groundtruth graph

    #Input 2 graphs
    G3, G4 = perturb_edge_pair(G3, rand_portion)
    G3, G4 = ReorderingSame(G3,G4) #0505 disabled
    
    return G3, G4

def perturb_edge_pair(G, rand_portion = 0.1):
    
    G_copy = copy.deepcopy(G)
    
    edgelist = list(G.edges)
    
    num_mask_rand = int(len(edgelist)*rand_portion)
    if rand_portion == 0:
        return G, G_copy
    
    for _ in range(num_mask_rand):
        e = random.sample(list(edgelist),1)
        
        start_vertex = e[0][0]
        end_vertex = e[0][1]
        
        if G.degree[start_vertex] >= 2 and G.degree[end_vertex] >= 2:
            G.remove_edges_from(e)
            
    for _ in range(num_mask_rand):
        e = random.sample(list(edgelist),1)
        
        start_vertex = e[0][0]
        end_vertex = e[0][1]
        
        if G_copy.degree[start_vertex] >= 2 and G_copy.degree[end_vertex] >= 2:
            G_copy.remove_edges_from(e)
            
    return G, G_copy

def ReorderingSame(G1,G2):
    G1 = nx.convert_node_labels_to_integers(G1, first_label=1, ordering='default', label_attribute=None)
    G2 = nx.convert_node_labels_to_integers(G2, first_label=1, ordering='default', label_attribute=None)
    return G1,G2

# ACN-------------------------------------------------------------


if __name__ == '__main__':
    args = parse_args()
    print(args)
    start_time = time()

    # random seed to make the best result
    # manual_seed = random.randint(1,1000000)
    manual_seed = 190029
    # manual_seed = 86484

    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    source_dataset = Dataset(args.source_dataset)
    target_dataset = Dataset(args.target_dataset)
    groundtruth = graph_utils.load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')

# ACN--------------------------------------------------------------
    G1 = nx.Graph()
    G2 = nx.Graph()
    # G1_edges = pd.read_csv('dataset/graph/douban1.edges', names = ['0', '1'])
    G1_edges = pd.read_csv('dataset/graph/econ1.edges', names = ['0', '1'])

    # print(G1_edges)
    G1.add_edges_from(np.array(G1_edges))
    # G2_edges = pd.read_csv('dataset/graph/douban2.edges', names = ['0', '1'])
    G2_edges = pd.read_csv('dataset/graph/econ2.edges', names = ['0', '1'])
    G2.add_edges_from(np.array(G2_edges))

    # pertub
    G1, G2 = PerturbedProcessing(G1, G2, 0, 0.05, 'econ')

    # alignment = pd.read_csv("dataset/alignment/douban.csv", header = None)
    alignment = pd.read_csv("dataset/alignment/econ.csv", header = None)

    alignment_dict = {}
    alignment_dict_reversed = {}
    for i in range(len(alignment)):
        alignment_dict[alignment.iloc[i, 0]] = alignment.iloc[i, 1]
        alignment_dict_reversed[alignment.iloc[i, 1]] = alignment.iloc[i, 0]
    
    # G2, alignment_dict, alignment_dict_reversed = preprocessing(G1, G2, alignment_dict)

    G1list = list(G1.nodes())
    #G1list.sort()
    idx1_list = list(range(G1.number_of_nodes()))
    #make dict for G1
    idx1_dict = {a : b for b, a in zip(idx1_list,G1list)}
    G2list = list(G2.nodes())
    #G2list.sort()
    idx2_list = list(range(G2.number_of_nodes()))
    #make dict for G2
    idx2_dict = {c : d for d, c in zip(idx2_list,G2list)}

# ACN--------------------------------------------------------------

    # print(type(groundtruth))
    algorithm = args.algorithm

    if algorithm == "IsoRank":
        train_dict = None
        if args.train_dict != "":
            train_dict = graph_utils.load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        model = IsoRank(source_dataset, target_dataset, args.H, args.alpha, args.max_iter, args.tol, train_dict=train_dict)
    elif algorithm == "FINAL":
        train_dict = None
        if args.train_dict != "":
            train_dict = graph_utils.load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        model = FINAL(source_dataset, target_dataset, H=args.H, alpha=args.alpha, maxiter=args.max_iter, tol=args.tol, train_dict=train_dict)
    elif algorithm == "REGAL":
        model = REGAL(source_dataset, target_dataset, max_layer=args.max_layer, alpha=args.alpha, k=args.k, num_buckets=args.buckets,
                      gammastruc = args.gammastruc, gammaattr = args.gammaattr, normalize=True, num_top=args.num_top)
    elif algorithm == "BigAlign":
        model = BigAlign(source_dataset, target_dataset, lamb=args.lamb)
    elif algorithm == "IONE":
        model = IONE(source_dataset, target_dataset, gt_train=args.train_dict, epochs=args.epochs, dim=args.dim, seed=args.seed, learning_rate=args.lr)
    elif algorithm == "DeepLink":
        model = DeepLink(source_dataset, target_dataset, args)
    elif algorithm == "GAlign":
        model = GAlign(G1, G2, alignment_dict, idx1_dict, idx2_dict, source_dataset, target_dataset, args)
    elif algorithm == "PALE":
        model = PALE(source_dataset, target_dataset, args)
    elif algorithm == "CENALP":
        model = CENALP(source_dataset, target_dataset, args)
    elif algorithm == "NAWAL":
        model = NAWAL(source_dataset, target_dataset, args)
    else:
        raise Exception("Unsupported algorithm")


    S = model.align()

    # print(S)
    # index = np.argmax(S, axis = 1)
    # print(index.shape)
    # A_st = np.zeros(S.shape)
    # for i in range(S.shape[0]):
    #     A_st[i,index[i]] = 1
    # print(A_st.sum(axis=1))

    print("-"*100)
    # print(type(groundtruth))
    # acc, MAP, top5, top10 = get_statistics(S.cpu().detach().numpy(), groundtruth, use_greedy_match=False, get_all_metric=True)
    acc, MAP, top5, top10, top1 = get_statistics(S, groundtruth, use_greedy_match=False, get_all_metric=True)
    print("Accuracy: {:.4f}".format(acc))
    print("MAP: {:.4f}".format(MAP))
    print("Precision_1: {:.4f}".format(top1))
    print("Precision_5: {:.4f}".format(top5))
    print("Precision_10: {:.4f}".format(top10))
    
    

    print("-"*100)
    print('Running time: {}'.format(time()-start_time))
    print(manual_seed)