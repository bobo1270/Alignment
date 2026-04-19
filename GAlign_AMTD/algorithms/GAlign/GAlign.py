from algorithms.network_alignment_model import NetworkAlignmentModel
from evaluation.metrics import get_statistics
from algorithms.GAlign.embedding_model import G_Align as Multi_Order, StableFactor
from input.dataset import Dataset
from utils.graph_utils import load_gt
import torch.nn.functional as F
import torch.nn as nn
from algorithms.GAlign.utils import *
from algorithms.GAlign.losses import *

import torch
import numpy as np
import networkx as nx
import random 
import numpy as np

import argparse
import os
import time
import sys

from torch.autograd import Variable
from tqdm import tqdm

from algorithms.GAlign import diffusion
import copy 
import math
import pandas as pd
from scipy.sparse import csr_array

class GAlign(NetworkAlignmentModel):
    """
    GAlign model for networks alignment task
    """
    def __init__(self, G1, G2, attr1, attr2, alignment_dict, idx1_dict, idx2_dict, source_dataset, target_dataset, args):
        """
        :params source_dataset: source graph
        :params target_dataset: target graph
        :params args: more config params
        """
        super(GAlign, self).__init__(source_dataset, target_dataset)
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.alphas = [args.alpha0, args.alpha1, args.alpha2]
        self.args = args
        # self.full_dict = load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        self.full_dict = load_gt('dataset/DataProcessing/allmv_tmdb/dictionaries/groundtruth', source_dataset.id2idx, target_dataset.id2idx, 'dict')
        # self.full_dict = alignment_dict

# ACN-------------------------------------------------------------------------
        self.G1 = G1
        self.G2 = G2
        self.alignment_dict = alignment_dict
        self.idx1_dict = idx1_dict
        self.idx2_dict = idx2_dict
        self.iter = 10
        self.alpha = G2.number_of_nodes() / G1.number_of_nodes()
        self.beta = 1
        self.train_ratio = 0.0
        self.eval_mode = True
        self.cea_mode = False
        self.attr_s = attr1
        self.attr_t = attr2

# ACN-------------------------------------------------------------------------
    def aggregated_adj(self, A):

        A_hat_list = []
        A_hat_list.append(None)  # empty element for future iteration
        for i in range(len(A)):
            A[i, i] = 1
        A = torch.FloatTensor(A)
        A_cand = A

        for l in range(2):

            D_ = torch.diag(torch.sum(A, 0)**(-0.5))
            A_hat = torch.matmul(torch.matmul(D_, A), D_)
            A_hat = A_hat.float()
            A_hat_list.append(A_hat)
            A_cand = torch.matmul(A, A_cand)
            A = A + A_cand

        return A_hat_list
# ACN-------------------------------------------------------------------------


    def align(self):
        """
        The main function of GAlign
        """
        source_A_hat, target_A_hat, source_feats, target_feats = self.get_elements()
        print(type(source_A_hat),type(source_feats))
        # print(source_A_hat)
        # print(source_feats)

# ACN -----------------------------------------------------------
        source_A_hat = nx.adjacency_matrix(self.G1)
        target_A_hat = nx.adjacency_matrix(self.G2)
        source_A_hat, _ = Laplacian_graph(source_A_hat.todense())
        target_A_hat, _ = Laplacian_graph(target_A_hat.todense())
        source_A_hat = source_A_hat.cuda()
        target_A_hat = target_A_hat.cuda()

        # source_A_hat = torch.tensor(source_A_hat.todense())
        # target_A_hat = target_A_hat.todense()
        # source_A_hat = source_A_hat.toarray()
        # target_A_hat = target_A_hat.toarray()
        # source_A_hat = torch.from_numpy(source_A_hat).float().cuda()
        # target_A_hat = torch.from_numpy(target_A_hat).float().cuda()
        # source_A_hat = torch.from_numpy(source_A_hat)
        # target_A_hat = torch.from_numpy(target_A_hat)
        # source_A_hat = torch.FloatTensor(source_A_hat)
        # target_A_hat = torch.FloatTensor(target_A_hat)


        # source_A_hat = source_A_hat.float()
        # indices = torch.nonzero(source_A_hat).t()
        # values = source_A_hat[indices[0], indices[1]]
        # source_A_hat = torch.sparse.FloatTensor(indices, values, source_A_hat.size())

        # target_A_hat = target_A_hat.float()
        # indices = torch.nonzero(target_A_hat).t()
        # values = source_A_hat[indices[0], indices[1]]
        # target_A_hat = torch.sparse.FloatTensor(indices, values, target_A_hat.size())

        source_feats = self.attr_s
        target_feats = self.attr_t
        source_feats = torch.FloatTensor(source_feats)
        target_feats = torch.FloatTensor(target_feats)
        if self.args.cuda:
            source_feats = source_feats.cuda()
            target_feats = target_feats.cuda()
        source_feats = F.normalize(source_feats)
        target_feats = F.normalize(target_feats)
        print(type(source_A_hat),type(source_feats))
        # print(source_A_hat)
        # print(source_feats)

# ACN -----------------------------------------------------------


        print("Running Multi-level embedding")
        GAlign = self.multi_level_embed(source_A_hat, target_A_hat, source_feats, target_feats)
        print("Running Refinement Alignment")
        S_GAlign = self.refinement_alignment(GAlign, source_A_hat, target_A_hat)

# ACN-----------------------------------------------------------------------------------------------
        iteration = 0

        seed_list1 = []
        seed_list2 = []

        index = sorted(list(self.G1.nodes())) 
        columns = sorted(list(self.G2.nodes()))

        start = time.time()

        while True:

            index = list(set(index) - set(seed_list1))
            # print(index)
            columns = list(set(columns) - set(seed_list2))
        
            seed_n_id_list = seed_list1 + seed_list2
            if len(columns) == 0 or len(index) == 0:
                break
            if len(self.alignment_dict) == len(seed_list1):
                break

            print('\n ------ The current iteration : {} ------'.format(iteration))
            # Update graph
            print('\n start adding a seed nodes')
            if iteration == 0:
                seed_list1, seed_list2, S, adj2, S_emb = self.AddSeeds_ver2_init(S_GAlign, index, columns, seed_list1, seed_list2, iteration)         
            else:
                seed_list1, seed_list2, S, adj2 = self.AddSeeds_ver2(S_emb, index, columns, seed_list1, seed_list2, iteration)
                # print(seed_list1)
                # print(seed_list2)
            iteration += 1

        # Evaluate Performance
        print("total time : {}sec".format(int(time.time() - start)))
        print('\n Start evaluation...')
        self.Evaluation(seed_list1, seed_list2)
        S_prime, result = self.FinalEvaluation(
            S, seed_list1, seed_list2, self.idx1_dict, self.idx2_dict, adj2)
        # self.normdif_checker(self.att_aug_s, self.att_aug_t,
        #                      embedding_aug1, embedding_aug2)
        
        return S
    
    def AddSeeds_ver2_init(self, S_GAlign, index, columns, seed_list1, seed_list2, iteration):
        S_fin = S_GAlign
        S_emb = copy.deepcopy(S_fin)

        # try:
        #     S_fin = S_fin + self.H
        # except:
        #     print("no prior anchors")
        #     pass
        print("no prior anchors")


        sim_matrix = np.zeros((len(index) * len(columns), 3))
        for i in range(len(index)):
            for j in range(len(columns)):
                sim_matrix[i * len(columns) + j, 0] = index[i]
                sim_matrix[i * len(columns) + j, 1] = columns[j]
                sim_matrix[i * len(columns) + j, 2] = S_fin[self.idx1_dict[index[i]], self.idx2_dict[columns[j]]]
        if len(seed_list1) != 0:
            print("Tversky sim calculation..")
            sim_matrix2 = ACN_sim(
                self.G1, self.G2, seed_list1, seed_list2, index, columns, alpha=self.alpha, beta=self.beta)
            sim_matrix[:, 2] *= sim_matrix2[:, 2]
        else:
            sim_matrix2 = 1  # no effect
        sim_matrix = sim_matrix[np.argsort(-sim_matrix[:, 2])]

        seed1, seed2 = [], []
        len_sim_matrix = len(sim_matrix)
        if len_sim_matrix != 0:
            T = align_func(version='const', a=int(len(self.alignment_dict) / self.iter), b=0, i=iteration)
            nodes1, nodes2, sims = sim_matrix[:, 0].astype(int), sim_matrix[:, 1].astype(int), sim_matrix[:, 2]
            idx = np.argsort(-sims)
            nodes1, nodes2, sims = nodes1[idx], nodes2[idx], sims[idx]
            while len(nodes1) > 0 and T > 0:
                T -= 1
                node1, node2 = nodes1[0], nodes2[0]
                seed1.append(node1)
                seed2.append(node2)
                mask = np.logical_and(nodes1 != node1, nodes2 != node2)
                nodes1, nodes2, sims = nodes1[mask], nodes2[mask], sims[mask]
            sim_matrix = np.column_stack((nodes1, nodes2, sims))
        anchor = len(seed_list1)
        seed_list1 += seed1
        seed_list2 += seed2
        print('Add seed nodes : {}'.format(len(seed1)))
        print(f'{iteration} iter completed')

        self.Evaluation(seed_list1, seed_list2)

        return seed_list1, seed_list2, S_fin, sim_matrix2, S_emb
    def AddSeeds_ver2(self, S_emb, index, columns, seed_list1, seed_list2, iteration):
        S_fin = S_emb
        sim_matrix = np.zeros((len(index) * len(columns), 3))
        for i in range(len(index)):
            for j in range(len(columns)):
                sim_matrix[i * len(columns) + j, 0] = index[i]
                sim_matrix[i * len(columns) + j, 1] = columns[j]
                sim_matrix[i * len(columns) + j, 2] = S_fin[self.idx1_dict[index[i]],
                                                            self.idx2_dict[columns[j]]]

        if len(seed_list1) != 0:
            print("ACN sim calculation..")
            sim_matrix2 = ACN_sim(
                self.G1, self.G2, seed_list1, seed_list2, index, columns, alpha=self.alpha, beta=self.beta)
            sim_matrix[:, 2] *= sim_matrix2[:, 2]
        else:
            sim_matrix2 = 1  # no effect
        sim_matrix = sim_matrix[np.argsort(-sim_matrix[:, 2])]

        seed1, seed2 = [], []
        len_sim_matrix = len(sim_matrix)
        if len_sim_matrix != 0:
            T = align_func(version='const', a=int(len(self.alignment_dict) / self.iter), b=0, i=iteration)
            nodes1, nodes2, sims = sim_matrix[:, 0].astype(int), sim_matrix[:, 1].astype(int), sim_matrix[:, 2]
            idx = np.argsort(-sims)
            nodes1, nodes2, sims = nodes1[idx], nodes2[idx], sims[idx]
            while len(nodes1) > 0 and T > 0:
                T -= 1
                node1, node2 = nodes1[0], nodes2[0]
                seed1.append(node1)
                seed2.append(node2)
                mask = np.logical_and(nodes1 != node1, nodes2 != node2)
                nodes1, nodes2, sims = nodes1[mask], nodes2[mask], sims[mask]
            sim_matrix = np.column_stack((nodes1, nodes2, sims))
        anchor = len(seed_list1)
        seed_list1 += seed1
        seed_list2 += seed2
        print('Add seed nodes : {}'.format(len(seed1)))

        print(f'{iteration} iter completed')

        self.Evaluation(seed_list1, seed_list2)

        return seed_list1, seed_list2, S_fin, sim_matrix2

    def Evaluation(self, seed_list1, seed_list2):
        count = 0

        for i in range(len(seed_list1)):
            try:
                if self.alignment_dict[seed_list1[i]] == seed_list2[i]:
                    count += 1
            except:
                continue

        train_len = int(self.train_ratio * len(self.alignment_dict))
        print('Prediction accuracy  at this iteration : %.2f%%' %
              (100 * (count-train_len) / (len(seed_list1)-train_len)))
        print('All accuracy : %.2f%%' %
              (100*(count / len(self.alignment_dict))))
        print('All prediction accuracy : %.2f%%' %
              (100*((count - train_len) / (len(self.alignment_dict)-train_len))))

    def FinalEvaluation(self, S, seed_list1, seed_list2, idx1_dict, idx2_dict, adj2):

        count = 0

        for i in range(len(seed_list1)):
            try:
                if self.alignment_dict[seed_list1[i]] == seed_list2[i]:
                    count += 1
            except:
                continue

        train_len = int(self.train_ratio * len(self.alignment_dict))
        print('All accuracy : %.2f%%' %
              (100*(count / len(self.alignment_dict))))
        acc = count / len(self.alignment_dict)

        #input embeddings are final embedding
        index = list(self.G1.nodes())
        columns = list(self.G2.nodes())
        if self.eval_mode == True:
            adj2 = calculate_Tversky_coefficient_final(
                self.G1, self.G2, seed_list1, seed_list2, index, columns, alpha=self.alpha, beta=self.beta)
            S_prime = self.adj2S(
                adj2, self.G1.number_of_nodes(), self.G2.number_of_nodes())
            S *= S_prime

        gt_dict = self.alignment_dict

        top_1 = top_k(S, 1)
        top_5 = top_k(S, 5)
        top_10 = top_k(S, 10)

        top1_eval = compute_precision_k(top_1, gt_dict, idx1_dict, idx2_dict)
        top5_eval = compute_precision_k(top_5, gt_dict, idx1_dict, idx2_dict)
        top10_eval = compute_precision_k(top_10, gt_dict, idx1_dict, idx2_dict)

        print('Success@1 : {:.4f}'.format(top1_eval))
        print('Success@5 : {:.4f}'.format(top5_eval))
        print('Success@10 : {:.4f}'.format(top10_eval))

        result = '@1:' + str(round(top1_eval, 4)) + ',  @5:' + str(round(top5_eval, 4)) + \
            ',  @10:' + str(round(top10_eval, 4)) + \
            ',  Acc:' + str(round(acc, 4))

        with open("./result.txt", "a") as file:
            file.write('\n All accuracy : %.2f%%' %
              (100*(count / len(self.alignment_dict))))
            file.write('\n Success@1 : {:.4f}'.format(top1_eval))
            file.write('\n Success@5 : {:.4f}'.format(top5_eval))
            file.write('\n Success@10 : {:.4f}'.format(top10_eval))
        return S, result

    def adj2S(self, adj, m, n):
        # m = # of nodes in G_s
        S = np.zeros((m, n))
        index = list(self.G1.nodes())
        columns = list(self.G2.nodes())
        for i in range(m):
            for j in range(n):
                S[self.idx1_dict[index[i]],
                    self.idx2_dict[columns[j]]] = adj[i * n + j, 2]
        return S

# ACN-----------------------------------------------------------------------------------------------

        # return S_GAlign


    def multi_level_embed(self, source_A_hat, target_A_hat, source_feats, target_feats):
        """
        Input: SourceGraph and TargetGraph
        Output: Embedding of those graphs using Multi_order_embedding model
        """
        GAlign = Multi_Order(
            activate_function = self.args.act,
            num_GCN_blocks = self.args.num_GCN_blocks,
            input_dim = self.args.input_dim,
            output_dim = self.args.embedding_dim,
            num_source_nodes = len(source_A_hat),
            num_target_nodes = len(target_A_hat),
            source_feats = source_feats,
            target_feats = target_feats
        )

        if self.args.cuda:
            GAlign = GAlign.cuda()

        structural_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, GAlign.parameters()), lr=self.args.lr)

        for epoch in range(self.args.GAlign_epochs):
            if self.args.log:
                print("Structure learning epoch: {}".format(epoch))
        
            for i in range(2):
                structural_optimizer.zero_grad()    
                if i == 0:
                    # Source loss
                    A_hat = source_A_hat
                    outputs, loss_dgi = GAlign(source_A_hat, 's')
                    # print(outputs.shape)

                    diff_A_hat = diffusion.diff_adj(A_hat, source_feats)
                    diff_outputs, loss_dgi_diff = GAlign(diff_A_hat, 's')

                    new_source_feats = diffusion.noise_x(source_feats)
                    diff_feats_outputs,loss_dgi_diff_feats = GAlign(A_hat, 's', new_source_feats)
                else:
                    # Target loss
                    A_hat = target_A_hat
                    outputs,loss_dgi = GAlign(target_A_hat, 't')

                    diff_A_hat = diffusion.diff_adj(A_hat, target_feats)
                    diff_outputs, loss_dgi_diff = GAlign(diff_A_hat, 't')

                    new_target_feats = diffusion.noise_x(target_feats)
                    diff_feats_outputs,loss_dgi_diff_feats = GAlign(A_hat, 't', new_target_feats)                                      

                consistency_loss = self.linkpred_loss(outputs[-1], A_hat) 
                diff_feats_consistency_loss = self.linkpred_loss(diff_feats_outputs[-1], A_hat)
                diff_consistency_loss = self.linkpred_loss(diff_outputs[-1], diff_A_hat)              

                consistency_loss = self.args.beta * consistency_loss + (1-self.args.beta) * (diff_consistency_loss + diff_feats_consistency_loss)
                # consistency_loss = self.args.beta * consistency_loss + (1-self.args.beta) * (diff_consistency_loss)
                

                diff = torch.abs(outputs[-1] - diff_feats_outputs[-1])
                noise_adaptivity_loss_1 = (diff[diff < self.args.threshold] ** 2).sum() / len(outputs)
                diff = torch.abs(outputs[-1] - diff_outputs[-1])
                noise_adaptivity_loss_2 = (diff[diff < self.args.threshold] ** 2).sum() / len(outputs)
                noise_adaptivity_loss = (noise_adaptivity_loss_1 + noise_adaptivity_loss_2) / 2
                # noise_adaptivity_loss = noise_adaptivity_loss_2     
                # Am-ID
                # loss = consistency_loss + loss_dgi
                # loss = self.args.coe_consistency * consistency_loss + (1 - self.args.coe_consistency) * noise_adaptivity_loss + 0.5*(loss_dgi + loss_dgi_diff + loss_dgi_diff_feats)
                # loss = diff_consistency_loss + diff_feats_consistency_loss + consistency_loss + noise_adaptivity_loss + loss_dgi + loss_dgi_diff + loss_dgi_diff_feats     

                # Douban       
                loss = diff_consistency_loss + diff_feats_consistency_loss + 0.0001 * (consistency_loss + noise_adaptivity_loss) + 0.00001*(loss_dgi + loss_dgi_diff + loss_dgi_diff_feats)
                # loss = diff_consistency_loss + diff_feats_consistency_loss + 0.0001 * (consistency_loss + noise_adaptivity_loss)
                
                # print(loss)

                """loop in backward"""    
                # if self.args.log:
                #     print("Loss: {:.4f}".format(loss.data))
                loss.backward()
                structural_optimizer.step()
        GAlign.eval()
        return GAlign

    def refinement_alignment(self, GAlign, source_A_hat, target_A_hat):
        source_A_hat = source_A_hat.to_dense()
        target_A_hat = target_A_hat.to_dense()
        GAlign_S = self.refine(GAlign, source_A_hat, target_A_hat, 0.94)
        return GAlign_S


    def get_elements(self):
        """
        Compute Normalized Laplacian matrix
        Preprocessing nodes attribute
        """
        source_A_hat, _ = Laplacian_graph(self.source_dataset.get_adjacency_matrix())
        target_A_hat, _ = Laplacian_graph(self.target_dataset.get_adjacency_matrix())
        if self.args.cuda:
            source_A_hat = source_A_hat.cuda()
            target_A_hat = target_A_hat.cuda()

        source_feats = self.source_dataset.features
        target_feats = self.target_dataset.features

        if source_feats is None:
            source_feats = np.zeros((len(self.source_dataset.G.nodes()), 1))
            target_feats = np.zeros((len(self.target_dataset.G.nodes()), 1))
        
        for i in range(len(source_feats)):
            if source_feats[i].sum() == 0:
                source_feats[i, -1] = 1
        for i in range(len(target_feats)):
            if target_feats[i].sum() == 0:
                target_feats[i, -1] = 1
        if source_feats is not None:
            source_feats = torch.FloatTensor(source_feats)
            target_feats = torch.FloatTensor(target_feats)
            if self.args.cuda:
                source_feats = source_feats.cuda()
                target_feats = target_feats.cuda()
        source_feats = F.normalize(source_feats)
        target_feats = F.normalize(target_feats)
        return source_A_hat, target_A_hat, source_feats, target_feats


    def linkpred_loss(self, embedding, A):
        pred_adj = torch.matmul(F.normalize(embedding), F.normalize(embedding).t())
        if self.args.cuda:
            pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]).cuda())), dim = 1)
        else:
            pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]))), dim = 1)
        #linkpred_losss = (pred_adj - A[index]) ** 2
        linkpred_losss = (pred_adj - A) ** 2
        linkpred_losss = linkpred_losss.sum() / A.shape[1]
        return linkpred_losss


    def refine(self, GAlign, source_A_hat, target_A_hat, threshold):
        refinement_model = StableFactor(len(source_A_hat), len(target_A_hat), self.args.cuda)
        if self.args.cuda: 
            refinement_model = refinement_model.cuda()
        S_max = None
        source_outputs,_ = GAlign(refinement_model(source_A_hat, 's'), 's')
        target_outputs,_ = GAlign(refinement_model(target_A_hat, 't'), 't')
        acc, S = get_acc(source_outputs, target_outputs, self.full_dict, self.alphas, just_S=True)
        score = np.max(S, axis=1).mean()
        acc_max = 0
        alpha_source_max = None
        alpha_target_max = None
        if 1:
        #if score > refinement_model.score_max:
            refinement_model.score_max = score
            alpha_source_max = refinement_model.alpha_source
            alpha_target_max = refinement_model.alpha_target
            acc_max = acc
            S_max = S
        print("Acc: {}, score: {:.4f}".format(acc, score))
        source_candidates, target_candidates = [], []            
        alpha_source_max = refinement_model.alpha_source + 0
        alpha_target_max = refinement_model.alpha_target + 0
        for epoch in range(self.args.refinement_epochs):
            if self.args.log:
                print("Refinement epoch: {}".format(epoch))
            source_candidates, target_candidates, len_source_candidates, count_true_candidates = self.get_candidate(source_outputs, target_outputs, threshold)
            
            refinement_model.alpha_source[source_candidates] *= 1.1
            refinement_model.alpha_target[target_candidates] *= 1.1
            source_outputs,_ = GAlign(refinement_model(source_A_hat, 's'), 's')
            target_outputs,_ = GAlign(refinement_model(target_A_hat, 't'), 't')
            acc, S = get_acc(source_outputs, target_outputs, self.full_dict, self.alphas, just_S=True)
            score = np.max(S, axis=1).mean()
            if score > refinement_model.score_max:
                refinement_model.score_max = score
                alpha_source_max = refinement_model.alpha_source + 0
                alpha_target_max = refinement_model.alpha_target + 0
                acc_max = acc
                S_max = S
            if self.args.log:
                print("Acc: {}, score: {:.4f}, score_max {:.4f}".format(acc, score, refinement_model.score_max))
            if epoch == self.args.refinement_epochs - 1:
                print("Numcandidate: {}, num_true_candidate: {}".format(len_source_candidates, count_true_candidates))
        print("Done refinement!")
        print("Acc with max score: {:.4f} is : {}".format(refinement_model.score_max, acc_max))
        refinement_model.alpha_source = alpha_source_max
        refinement_model.alpha_target = alpha_target_max
        self.GAlign_S = S_max
        # self.log_and_evaluate(GAlign, refinement_model, source_A_hat, target_A_hat)
        return self.GAlign_S


    def get_similarity_matrices(self, source_outputs, target_outputs):
        """
        Construct Similarity matrix in each layer
        :params source_outputs: List of embedding at each layer of source graph
        :params target_outputs: List of embedding at each layer of target graph
        """
        list_S = []
        for i in range(len(source_outputs)):
            source_output_i = source_outputs[i]
            target_output_i = target_outputs[i]

            S = torch.mm(F.normalize(source_output_i), F.normalize(target_output_i).t())
            list_S.append(S)
        return list_S


    def log_and_evaluate(self, embedding_model, refinement_model, source_A_hat, target_A_hat):
        embedding_model.eval()
        source_outputs = embedding_model(refinement_model(source_A_hat, 's'), 's')
        target_outputs = embedding_model(refinement_model(target_A_hat, 't'), 't')
        print("-"* 100)
        log, self.S = get_acc(source_outputs, target_outputs, self.full_dict, self.alphas)
        print(self.alphas)
        print(log)
        return source_outputs, target_outputs
    

    def get_candidate(self, source_outputs, target_outputs, threshold):
        List_S = self.get_similarity_matrices(source_outputs, target_outputs)[1:]
        source_candidates = []
        target_candidates = []
        count_true_candidates = 0
        if len(List_S) < 2:
            print("The current model doesn't support refinement for number of GCN layer smaller than 2")
            return torch.LongTensor(source_candidates), torch.LongTensor(target_candidates)

        num_source_nodes = len(self.source_dataset.G.nodes())
        num_target_nodes = len(self.target_dataset.G.nodes())
        for i in range(min(num_source_nodes, num_target_nodes)):
            node_i_is_stable = True
            for j in range(len(List_S)):
                if List_S[j][i].argmax() != List_S[j-1][i].argmax() or List_S[j][i].max() < threshold:
                    node_i_is_stable = False 
                    break
            if node_i_is_stable:
                tg_candi = List_S[-1][i].argmax()
                source_candidates.append(i)
                target_candidates.append(tg_candi)
                try:
                    if self.full_dict[i] == tg_candi:
                        count_true_candidates += 1
                except:
                    continue
        return torch.LongTensor(source_candidates), torch.LongTensor(target_candidates), len(source_candidates), count_true_candidates

# ACN --------------------------------------------------------------
def align_func(version, a, b, i):

    if version == "lin":
        return int(a*i + b)
    elif version == "exp":
        return int(a**i + b)
    elif version == "log":
        return int(math.log(a*i+b) + b)
    elif version == "const":
        return a


def ACN_sim(G1, G2, seed_list1, seed_list2, index, columns, alpha, beta, alignment_dict=None):

    start = time.time()

    shift = int(np.max([np.max(G1.nodes()), np.max(G2.nodes())]))
    seed1_dict = {}
    seed1_dict_reversed = {}
    seed2_dict = {}
    seed2_dict_reversed = {}
    for i in range(len(seed_list1)):
        seed1_dict[i + 2 * (shift + 1)] = seed_list1[i]
        seed1_dict_reversed[seed_list1[i]] = i + 2 * (shift + 1)
        seed2_dict[i + 2 * (shift + 1)] = seed_list2[i] + shift + 1
        seed2_dict_reversed[seed_list2[i] + shift + 1] = i + 2 * (shift + 1)
    G1_edges = pd.DataFrame(G1.edges())
    G1_edges.iloc[:, 0] = G1_edges.iloc[:, 0].apply(
        lambda x: to_seed(x, seed1_dict_reversed))
    G1_edges.iloc[:, 1] = G1_edges.iloc[:, 1].apply(
        lambda x: to_seed(x, seed1_dict_reversed))
    G2_edges = pd.DataFrame(G2.edges())
    G2_edges += shift + 1
    G2_edges.iloc[:, 0] = G2_edges.iloc[:, 0].apply(
        lambda x: to_seed(x, seed2_dict_reversed))
    G2_edges.iloc[:, 1] = G2_edges.iloc[:, 1].apply(
        lambda x: to_seed(x, seed2_dict_reversed))
    adj = nx.Graph()
    adj.add_edges_from(np.array(G1_edges))
    adj.add_edges_from(np.array(G2_edges))
    Tversky_dict = {}
    for G1_node in index:
        for G2_node in columns:
            if (G1_node, G2_node) not in Tversky_dict.keys():
                Tversky_dict[G1_node, G2_node] = 0
            try:
                #Tversky_dict[G1_node, G2_node] += calculate_Tversky(adj.neighbors(G1_node), adj.neighbors(G2_node + shift + 1), alpha, beta)
                Tversky_dict[G1_node, G2_node] += calculate_new(adj.neighbors(
                    G1_node), adj.neighbors(G2_node + shift + 1), alpha, beta)
            except:
                continue
    Tversky_dict = [[x[0][0], x[0][1], x[1]] for x in Tversky_dict.items()]
    sim_matrix = np.array(Tversky_dict)

   # print(f'{(time.time()-start):.2f} sec elapsed for Tversky')
    return sim_matrix

def calculate_Tversky_coefficient_final(G1, G2, seed_list1, seed_list2, index, columns, alpha, beta):
    shift = int(np.max([np.max(G1.nodes()), np.max(G2.nodes())]))
    seed1_dict = {}
    seed1_dict_reversed = {}
    seed2_dict = {}
    seed2_dict_reversed = {}
    for i in range(len(seed_list1)):
        seed1_dict[i + 2 * (shift + 1)] = seed_list1[i]
        seed1_dict_reversed[seed_list1[i]] = i + 2 * (shift + 1)
        seed2_dict[i + 2 * (shift + 1)] = seed_list2[i] + shift + 1
        seed2_dict_reversed[seed_list2[i] + shift + 1] = i + 2 * (shift + 1)
    G1_edges = pd.DataFrame(G1.edges())
    G1_edges.iloc[:, 0] = G1_edges.iloc[:, 0].apply(
        lambda x: to_seed(x, seed1_dict_reversed))
    G1_edges.iloc[:, 1] = G1_edges.iloc[:, 1].apply(
        lambda x: to_seed(x, seed1_dict_reversed))
    G2_edges = pd.DataFrame(G2.edges())
    G2_edges += shift + 1
    G2_edges.iloc[:, 0] = G2_edges.iloc[:, 0].apply(
        lambda x: to_seed(x, seed2_dict_reversed))
    G2_edges.iloc[:, 1] = G2_edges.iloc[:, 1].apply(
        lambda x: to_seed(x, seed2_dict_reversed))
    adj = nx.Graph()
    adj.add_edges_from(np.array(G1_edges))
    adj.add_edges_from(np.array(G2_edges))
    Tversky_dict = {}
    for G1_node in index:
        for G2_node in columns:
            Tversky_dict[G1_node, G2_node] = 0
            g1 = to_seed(G1_node, seed1_dict_reversed)
            g2 = to_seed(G2_node + shift + 1, seed2_dict_reversed)
            #Tversky_dict[G1_node, G2_node] += calculate_Tversky(adj.neighbors(g1), adj.neighbors(g2), alpha, beta)
            Tversky_dict[G1_node, G2_node] += calculate_new(
                adj.neighbors(g1), adj.neighbors(g2), alpha, beta)
    Tversky_dict = [[x[0][0], x[0][1], x[1]] for x in Tversky_dict.items()]
    sim_matrix = np.array(Tversky_dict)
    return sim_matrix

def to_seed(x, dictionary):
    try:
        return dictionary[x]
    except:
        return x

def top_k(S, k=1):
    """
    S: scores, numpy array of shape (M, N) where M is the number of source nodes,
        N is the number of target nodes
    k: number of predicted elements to return
    """
    top = np.argsort(-S)[:, :k]
    result = np.zeros(S.shape)
    for idx, target_elms in enumerate(top):
        for elm in target_elms:
            result[idx, elm] = 1

    return result


def compute_precision_k(top_k_matrix, gt, idx1_dict, idx2_dict):
    n_matched = 0

    if type(gt) == dict:
        for key, value in gt.items():
            if top_k_matrix[idx1_dict[key], idx2_dict[value]] == 1:
                n_matched += 1
        return n_matched/len(gt)

    return n_matched/n_nodes

def calculate_new(setA, setB, alpha, beta):
    setA = set(setA)
    setB = set(setB)

    ep = 0.01
    ep2 = max(len(setA), len(setB))

    ACNs = len(setA & setB) + ep

    #Tver = ACNs**2 / (abs(len(setA) - len(setB)) + ep2)

    return ACNs**2
