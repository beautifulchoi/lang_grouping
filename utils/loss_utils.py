#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
import random

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _rbf_kernel(x1, x2, gamma=1.0):
    diff = x1.unsqueeze(1) - x2.unsqueeze(0)
    squared_diff = torch.sum(diff**2, dim=-1)
    return torch.exp(-gamma * squared_diff)


# NOTE : only use feature_semgented mask
def contrastive_1d_loss(features, objects, gamma=0.01, num_samples=2048):
    """
    input:
        - features(torch.Tensor) : (f_dim, h, w)
        - objects(torch.Tensor) : (h,w)
    
    output:
        - contrastive loss value 
    """
    loss = 0.0
    sampled_indices = torch.randint(0, features.shape[0] * features.shape[1], (num_samples,))  # u개의 픽셀을 "전체에서" 뽑음
    sampled_indices = sampled_indices.to(objects.device)
    features_flatten = features.reshape(-1, features.shape[-1])
    sampled_features = features_flatten[sampled_indices]  # u  : shape must be (4096,3)

    for class_id in torch.unique(objects):  # obj id 0인경우 > class_objects에서 마스킹 됨
        if class_id == -1.:
            continue
        class_objects = objects == class_id  # 해당 클래스 마스크
        num_obj_pixel = torch.sum(class_objects)
        
        class_features = features * class_objects.unsqueeze(-1)  # 클래스 픽셀 마스킹 
        if num_obj_pixel <= 1:
            continue
        
        class_features = class_features.reshape(-1, class_features.shape[-1])  # (H*W , 3)
        
        class_objects_flatten = class_objects.flatten()
        sampled_cls_indices = sampled_indices[class_objects_flatten[sampled_indices] == 1.]  # 샘플링한 픽셀 중에서 클래스에 해당하는 픽셀들 
        sampled_class_features = class_features[sampled_cls_indices]  # u+
        
        positive_similarities = _rbf_kernel(sampled_class_features, sampled_class_features, gamma)  # u+ <-> u+  , (pos_pixel, pos_pixel)
        positive_similarities = positive_similarities.triu(diagonal=1)
        num_positive_pairs = torch.sum(positive_similarities > 0)
        if num_positive_pairs > 0:
            positive_loss = positive_similarities.sum()
            loss += positive_loss

    if loss == 0.0:
        raise ValueError("positive loss: zero value")
    # after adding all pos pair, we should take log scale
    loss = -torch.log(loss)

    negative_similarities = _rbf_kernel(sampled_features, sampled_features, gamma) #u <-> u, (sample, sample )
    negative_similarities = negative_similarities.triu(diagonal=1)

    negative_loss = torch.log(negative_similarities.sum())
    loss += negative_loss

    torch.cuda.empty_cache()

    return loss

# TODO
def weighted_contrastive_loss(ovl_features, seg_map, ovl_cls, gamma=0.01, num_samples=2048):
    """
    multi view contrastive loss which overlapped object get weighted loss

    input:
        - ovl_features(torch.Tensor) : object overlapped feature map, region not overlapped value 0(f_dim, h, w)
        - seg_map(torch.Tensor) : (h,w)
    
    output:
        - contrastive loss value 
    """
    
    loss = 0.0
    sampled_indices = torch.randint(0, ovl_features.shape[0] * ovl_features.shape[1], (num_samples,))  # u개의 픽셀을 "전체에서" 뽑음

    for class_id in torch.unique(seg_map):  # obj id 0인경우 > class_objects에서 마스킹 됨
        class_semantics = (seg_map == class_id).float()  # 해당 클래스에 대한 픽셀들
        num_semantic_pixel = torch.sum(class_semantics) 
        
        class_features = ovl_features * class_semantics.unsqueeze(-1)
        class_features = torch.where(class_features >= 0, class_features, 0)
        if num_semantic_pixel <= 1:
            continue
        
        class_features = class_features.reshape(-1, class_features.shape[-1]) # (H*W , 3)
        
        sampled_indices=sampled_indices.to(class_semantics.device)
        sampled_cls_indices = sampled_indices[class_semantics.view(-1,1)[sampled_indices].squeeze(-1) == 1.] #샘플링한 픽셀 중에서 클래스에 해당하는 픽셀들 
        sampled_class_features = class_features[sampled_cls_indices] # u+ 

        features_flatten = features.view(-1, features.shape[-1]) 
        sampled_features = features_flatten[sampled_indices] #u 
        
        positive_similarities = _rbf_kernel(sampled_features, sampled_class_features, gamma) # u+ <-> u  , (4096, pos_pixel)
        positive_similarities = positive_similarities.triu(diagonal=1) 
        num_positive_pairs = torch.sum(positive_similarities > 0)
        if num_positive_pairs > 0:
            positive_loss = -torch.log(positive_similarities.sum())
            loss += positive_loss

        negative_similarities = torch.exp(_rbf_kernel(sampled_features, sampled_features, gamma))
        negative_similarities = negative_similarities.triu(diagonal=1)
        num_negative_pairs = torch.sum(negative_similarities > 0)
        if num_negative_pairs > 0:
            negative_loss = torch.log(negative_similarities.sum())
            loss += negative_loss

    return loss/num_samples