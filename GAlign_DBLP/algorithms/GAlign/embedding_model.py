import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
from torch.nn import init
import math

def init_weight(modules, activation):
    """
    Weight initialization
    :param modules: Iterable of modules
    :param activation: Activation function.
    """
    for m in modules:
        if isinstance(m, nn.Linear):
            if activation is None:
                m.weight.data = init.xavier_uniform_(m.weight.data) #, gain=nn.init.calculate_gain(activation.lower()))
            else:
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain(activation.lower()))
            if m.bias is not None:
                m.bias.data = init.constant_(m.bias.data, 0.0)


def get_act_function(activate_function):
    """
    Get activation function by name
    :param activation_fuction: Name of activation function 
    """
    if activate_function == 'sigmoid':
        activate_function = nn.Sigmoid()
    elif activate_function == 'relu':
        activate_function = nn.ReLU()
    elif activate_function == 'tanh':
        activate_function = nn.Tanh()
    else:
        return None
    return activate_function


class CombineModel(nn.Module):
    def __init__(self):
        super(CombineModel, self).__init__()
        self.thetas = nn.Parameter(torch.ones(3))

    
    def loss(self, S1, S2, S3, id2idx_augment):
        S = self.forward(S1, S2, S3)
        S_temp = torch.zeros(S.shape)
        for k,v in id2idx_augment.items():
            S_temp[int(k),v] = 1
        
        S = S / torch.sqrt((S**2).sum(dim=1)).view(S.shape[0],1)
        loss = -(S * S_temp).mean()
        return loss


    def forward(self, S1, S2, S3):
        theta_sum = torch.abs(self.thetas[0]) + torch.abs(self.thetas[1]) + torch.abs(self.thetas[2])
        return (torch.abs(self.thetas[0])/theta_sum) * S1 + (torch.abs(self.thetas[1])/theta_sum) * S2 + (torch.abs(self.thetas[2])/theta_sum) * S3


class Combine2Model(nn.Module):
    def __init__(self):
        super(Combine2Model, self).__init__()
        self.thetas = nn.Parameter(torch.ones(2))


    def loss(self, S1, S2, id2idx_augment):
        S = self.forward(S1, S2)
        S_temp = torch.zeros(S.shape)
        for k,v in id2idx_augment.items():
            S_temp[int(k),v] = 1
        
        S = S / torch.max(S, dim=1)[0].view(S.shape[0],1)
        loss = -(S * S_temp).mean()
        # loss = (S - 3 * torch.eye(len(S))).mean()
        return loss

    def forward(self, S1, S2):
        return torch.abs(self.thetas[0]) * S1 + torch.abs(self.thetas[1]) * S2


class GCN(nn.Module):
    """
    The GCN multistates block
    """
    def __init__(self, activate_function, input_dim, output_dim):
        """
        activate_function: Tanh
        input_dim: input features dimensions
        output_dim: output features dimensions
        """
        super(GCN, self).__init__()
        if activate_function is not None:
            self.activate_function = get_act_function(activate_function)
        else:
            self.activate_function = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        init_weight(self.modules(), activate_function)
    
    def forward(self, input, A_hat):
        output = self.linear(input)
        output = torch.matmul(A_hat, output)
        if self.activate_function is not None:
            output = self.activate_function(output)
        return output

class Encoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation, dropout):
        super(Encoder, self).__init__()
        self.conv = GCN_dgi(
            in_feats, n_hidden, n_hidden, n_layers, activation, dropout
        )

    def forward(self, features, adj,corrupt=False):
        if corrupt:
            perm = torch.randperm(features.shape[0])
            features = features[perm]
        features = self.conv(features,adj)
        return features

class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features
    
class G_Align(nn.Module):
    # """
    # Training a multilayer GCN model
    # """
    # def __init__(self, activate_function, num_GCN_blocks, input_dim, output_dim, \
    #             num_source_nodes, num_target_nodes, source_feats=None, target_feats=None):
    #     """
    #     :params activation_fuction: Name of activation function
    #     :params num_GCN_blocks: Number of GCN layers of model
    #     :params input_dim: The number of dimensions of input
    #     :params output_dim: The number of dimensions of output
    #     :params num_source_nodes: Number of nodes in source graph
    #     :params num_target_nodes: Number of nodes in target graph
    #     :params source_feats: Source Initialized Features
    #     :params target_feats: Target Initialized Features
    #     """
    #     super(G_Align, self).__init__()
    #     self.num_GCN_blocks = num_GCN_blocks 
    #     self.source_feats = source_feats
    #     self.target_feats = target_feats
    #     input_dim = self.source_feats.shape[1]
    #     self.input_dim = input_dim

    #     # GCN blocks (emb)
    #     self.GCNs = []
    #     for i in range(num_GCN_blocks):
    #         self.GCNs.append(GCN(activate_function, input_dim, output_dim))
    #         input_dim = self.GCNs[-1].output_dim
    #     self.GCNs = nn.ModuleList(self.GCNs)
    #     init_weight(self.modules(), activate_function)

    # def forward(self, A_hat, net='s', new_feats=None):
    #     """
    #     Do the forward
    #     :params A_hat: The sparse Normalized Laplacian Matrix 
    #     :params net: Whether forwarding graph is source or target graph
    #     """
    #     if new_feats is not None:
    #         input = new_feats
    #     elif net == 's':
    #         input = self.source_feats
    #     else:
    #         input = self.target_feats
    #     emb_input = input.clone()
    #     outputs = [emb_input]
    #     for i in range(self.num_GCN_blocks):
    #         GCN_output_i = self.GCNs[i](emb_input, A_hat)
    #         outputs.append(GCN_output_i)
    #         emb_input = GCN_output_i
    #     return outputs

    """
    Training a multilayer GCN model
    """
    def __init__(self, activate_function, num_GCN_blocks, input_dim, output_dim, \
                num_source_nodes, num_target_nodes, source_feats=None, target_feats=None):
        """
        :params activation_fuction: Name of activation function
        :params num_GCN_blocks: Number of GCN layers of model
        :params input_dim: The number of dimensions of input
        :params output_dim: The number of dimensions of output
        :params num_source_nodes: Number of nodes in source graph
        :params num_target_nodes: Number of nodes in target graph
        :params source_feats: Source Initialized Features
        :params target_feats: Target Initialized Features
        """
        super(G_Align, self).__init__()
        self.num_GCN_blocks = num_GCN_blocks 
        self.source_feats = source_feats
        self.target_feats = target_feats
        input_dim = self.source_feats.shape[1]
        self.input_dim = input_dim

        self.encoder = Encoder(
            input_dim, output_dim, num_GCN_blocks, activate_function, 0
        )


        self.discriminator = Discriminator(output_dim)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, A_hat, net='s', new_feats=None):
        """
        Do the forward
        :params A_hat: The sparse Normalized Laplacian Matrix 
        :params net: Whether forwarding graph is source or target graph
        """
        if new_feats is not None:
            input = new_feats
        elif net == 's':
            input = self.source_feats
        else:
            input = self.target_feats
        positive_ori = self.encoder(input,A_hat, corrupt=False)
        negative_ori = self.encoder(input,A_hat, corrupt=True)
        summary = torch.sigmoid(positive_ori[-1].mean(dim=0))

        positive = self.discriminator(positive_ori[-1], summary)
        negative = self.discriminator(negative_ori[-1], summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))
        positive_ori.insert(0, input)

        return positive_ori,l1 + l2

class StableFactor(nn.Module):
    """
    Stable factor following each node
    """
    def __init__(self, num_source_nodes, num_target_nodes, cuda=True):
        """
        :param num_source_nodes: Number of nodes in source graph
        :param num_target_nodes: Number of nodes in target graph
        """
        super(StableFactor, self).__init__()
        # self.alpha_source_trainable = nn.Parameter(torch.ones(num_source_nodes))
        self.alpha_source = torch.ones(num_source_nodes)
        self.alpha_target = torch.ones(num_target_nodes)
        self.score_max = 0
        self.alpha_source_max = None
        self.alpha_target_max = None
        if cuda:
            self.alpha_source = self.alpha_source.cuda()
            self.alpha_target = self.alpha_target.cuda()
        self.use_cuda = cuda
    
        
    def forward(self, A_hat, net='s'):
        """
        Do the forward 
        :param A_hat is the Normalized Laplacian Matrix
        :net: whether graph considering is source or target graph.
        """
        if net=='s':
            self.alpha = self.alpha_source
        else:
            self.alpha = self.alpha_target
        alpha_colum = self.alpha.reshape(len(self.alpha), 1)
        if self.use_cuda:
            alpha_colum = alpha_colum.cuda()
        A_hat_new = (alpha_colum * (A_hat * alpha_colum).t()).t()
        return A_hat_new 


def add_gaussian_noise(tensor, mean=0, std=0.00005):
    noise = torch.cuda.FloatTensor(tensor.size()) if torch.cuda.is_available() else torch.FloatTensor(tensor.size())
    torch.randn(tensor.size(), out=noise)
    noise = noise * std + mean
    noisy_tensor = tensor + noise
    return noisy_tensor

class GCN_dgi(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super(GCN_dgi, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        # hidden layers
        self.layers.append(GCN(activation, in_feats, n_hidden))
        for i in range(n_layers-1):
            self.layers.append(GCN(activation, n_hidden, n_hidden))
            
        # output layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features,adj):
        h_list = []
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h,adj)

            # random noise  robust
            # h = add_gaussian_noise(h)

            h_list.append(h)
        return h_list