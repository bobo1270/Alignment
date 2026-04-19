import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import math

import numpy as np

from torch_geometric.transforms import GDC 
from torch_sparse import coalesce
from torch_geometric.utils import dense_to_sparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    # out = torch.gather(v, index=t.unsqueeze(dim=1), dim=0).float()
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas)

def sample_q(t, x, alphas_bar, sqrt_one_minus_alphas_bar):

    miu, std = x.mean(dim=0), x.std(dim=0)
    noise = torch.randn_like(x, device=x.device)
    noise = noise * std + miu
    # noise = nn.LayerNorm(noise, elementwise_affine=False)
    noise = torch.tensor(noise)
    noise = torch.sign(x) * torch.abs(noise)
    x_t = (
            extract(alphas_bar, t, x.shape) * x +
            extract(sqrt_one_minus_alphas_bar, t, x.shape) * noise
            )
    return x_t



def noise_x(x):
    beta_schedule = 'linear'
    beta_1 = 0.0001
    beta_T = 0.02

    T = 100
    # beta = get_beta_schedule(beta_schedule, beta_1, beta_T, T)
    beta = np.linspace(
            beta_1, beta_T, T, dtype=np.float64
        )
    alphas = 1. - beta
    alphas = torch.from_numpy(alphas)
    alphas_bar = torch.cumprod(alphas, dim=0)
    alphas_bar = torch.sqrt(alphas_bar).to(device)
    sqrt_one_minus_alphas_bar = torch.sqrt(1. - alphas_bar)
    # alpha_l = 2

    with torch.no_grad():
        x = F.layer_norm(x, (x.shape[-1], ))

    t = torch.randint(T, size=(x.shape[0], ), device=x.device).to(device)
    x_t = sample_q(t, x, alphas_bar, sqrt_one_minus_alphas_bar)

    return x_t


def diff_adj(adj_s, features_s):
    gdc = GDC()
    diffusion_kwargs=dict(method='ppr', alpha=0.15)
    # diffusion_kwargs=dict(method='heat', t=8)
    id_s = adj_s.coalesce().indices()
    # edge_weight = torch.ones(id_s.size(1))
    edge_weight = adj_s.coalesce().values()

    edge_index, edge_weight = gdc.transition_matrix(id_s,edge_weight,adj_s.shape[0],'sym')

    diff_mat = gdc.diffusion_matrix_exact(edge_index, edge_weight,adj_s.shape[0],**diffusion_kwargs)

    # diff_mat = gdc.diffusion_matrix_approx(edge_index, edge_weight,adj_s.shape[0],'sym',**diffusion_kwargs)

    avg_degree = int(round(id_s.shape[1]*2/adj_s.shape[0]))
    edge_index,edge_weight = gdc.sparsify_dense(diff_mat,"threshold",**dict(avg_degree=avg_degree))
    edge_index, edge_weight = coalesce(edge_index, edge_weight, features_s.shape[0],features_s.shape[0])

    edge_index, edge_weight = gdc.transition_matrix(edge_index, edge_weight, features_s.shape[0], 'col')

    adj_s = torch.sparse_coo_tensor(edge_index, edge_weight, (features_s.shape[0],features_s.shape[0]))
    return adj_s