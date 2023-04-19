from thop import profile
import numpy as np

import torch
from torch import nn
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

# def normalize_keypoints(kpts, image_shape):
#     """ Normalize keypoints locations based on image image_shape"""
#     _, _, height, width = image_shape
#     one = kpts.new_tensor(1)
#     size = torch.stack([one*width, one*height])[None]
#     center = size / 2
#     scaling = size.max(1, keepdim=True).values * 0.7
#     return (kpts - center[:, None, :]) / scaling[:, None, :]

class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)
    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))

def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))

class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1

def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

default_config = {
    'descriptor_dim': 256,
    'nms_radius': 4,
    'keypoint_encoder': [32, 64, 128, 256],
    'GNN_layers': ['self', 'cross'] * 9,
    'sinkhorn_iterations': 100,
}

class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end
    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.
    """
    def __init__(self):
        super().__init__()
        # self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            default_config['descriptor_dim'], default_config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            default_config['descriptor_dim'], default_config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            default_config['descriptor_dim'], default_config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    # def forward(self, data):
    def forward(self, kpts0, scores0, desc0, kpts1, scores1, desc1):
        """Run SuperGlue on a pair of keypoints and descriptors"""

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, scores0)
        desc1 = desc1 + self.kenc(kpts1, scores1)

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        # scores = torch.bmm(torch.transpose(mdesc0, 1, 2), mdesc1)
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / default_config['descriptor_dim'] ** .5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=default_config['sinkhorn_iterations'])
        return scores

class MaxPool(nn.Module):
    def __init__(self, nms_radius: int):
        super(MaxPool, self).__init__()
        self.block = nn.MaxPool2d(kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        return torch.squeeze(self.block(x), dim=1)
def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    # def max_pool(x):
    #     return torch.nn.functional.max_pool2d(
    #         x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    max_pool = MaxPool(nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)

class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor
    """
    def __init__(self):
        super().__init__()
        # self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, default_config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)


    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        x = self.relu(self.conv1a(data))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, default_config['nms_radius'])

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        return scores, descriptors


weight_file_sp = '/home/hoangqc/Datasets/Weights/superpoint_v1.pth'
superpoint_model = SuperPoint()
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
superpoint_model.load_state_dict(torch.load(weight_file_sp, map_location=map_location))
superpoint_model.eval()

macs, params = profile(superpoint_model, inputs=(torch.randn(1, 1, 480, 640),))

print('SuperPoint-MACs: ', macs)
print('SuperPoint-Params: ', params)


# Initialize model with the pretrained weights
weight_file = '/home/hoangqc/Datasets/Weights/superglue_outdoor.pth'
superglue_model = SuperGlue()
superglue_model.load_state_dict(torch.load(weight_file, map_location=map_location))
superglue_model.eval()

num_kp0 = 256
x0 = torch.from_numpy(np.random.randint(low=0, high=639, size=(1, num_kp0)))
y0 = torch.from_numpy(np.random.randint(low=0, high=479, size=(1, num_kp0)))
kpts0 = torch.stack((x0, y0), 2).float()
scores0 = torch.randn(1, num_kp0)
desc0 = torch.randn(1, 256, num_kp0)

num_kp1 = 256
x1 = torch.from_numpy(np.random.randint(low=0, high=639, size=(1, num_kp1)))
y1 = torch.from_numpy(np.random.randint(low=0, high=479, size=(1, num_kp1)))
kpts1 = torch.stack((x1, y1), 2).float()
scores1 = torch.randn(1, num_kp1)
desc1 = torch.randn(1, 256, num_kp1)

# torch_out = superglue_model(kpts0, scores0, desc0, kpts1, scores1, desc1)
macs, params = profile(superglue_model, inputs=(kpts0, scores0, desc0, kpts1, scores1, desc1))

print('SuperGlue-MACs: ', macs)
print('SuperGlue-Params: ', params)
print('Magics: ', macs/(params*(num_kp0+num_kp1)))

from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator

flops, macs, params = get_model_profile(model=superpoint_model,  # model
                                        input_shape=(1, 1, 480, 640),
                                        args=None,  # list of positional arguments to the model.
                                        kwargs=None,  # dictionary of keyword arguments to the model.
                                        print_profile=True,
                                        detailed=True,  # print the detailed profile
                                        module_depth=-1,
                                        top_modules=1,  # the number of top modules to print aggregated profile
                                        warm_up=10,  # the number of warm-ups before measuring the time of each module
                                        as_string=True,
                                        output_file=None,
                                        ignore_modules=None)  # the list of modules to ignore in the profiling

flops, macs, params = get_model_profile(model=superglue_model,  # model
                                        input_shape=None,
                                        args=(kpts0, scores0, desc0, kpts1, scores1, desc1),  # list of positional arguments to the model.
                                        kwargs=None,  # dictionary of keyword arguments to the model.
                                        print_profile=True,
                                        detailed=True,  # print the detailed profile
                                        module_depth=-1,
                                        top_modules=1,  # the number of top modules to print aggregated profile
                                        warm_up=10,  # the number of warm-ups before measuring the time of each module
                                        as_string=True,
                                        output_file=None,
                                        ignore_modules=None)  # the list of modules to ignore in the profiling
