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

def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()

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



def _rbf_kernel(x1, x2, diff_dim = -1, gamma=1.0):
    diff = x1.unsqueeze(1) - x2.unsqueeze(0)
    squared_diff = torch.sum(diff**2, dim=diff_dim)
    return torch.exp(-gamma * squared_diff)

# NOTE : use feature_semgented mask
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
        
        positive_similarities = _rbf_kernel(sampled_class_features, sampled_class_features, gamma=gamma)  # u+ <-> u+  , (pos_pixel, pos_pixel)
        positive_similarities = positive_similarities.triu(diagonal=1)
        num_positive_pairs = torch.sum(positive_similarities > 0)
        if num_positive_pairs > 0:
            positive_loss = positive_similarities.sum()
            loss += positive_loss

    if loss == 0.0:
        raise ValueError("positive loss: zero value")
    
    # after adding all pos pair, we should take log scale
    loss = -torch.log(loss)

    negative_similarities = _rbf_kernel(sampled_features, sampled_features, gamma=gamma) #u <-> u, (sample, sample )
    negative_similarities = negative_similarities.triu(diagonal=1)

    negative_loss = torch.log(negative_similarities.sum())
    loss += negative_loss

    torch.cuda.empty_cache()

    return loss / num_samples

def slowfast_contrastive_1d_loss(features_fast, features_slow, objects, gamma=0.01, num_samples=2048):
    """
    contrastive loss for slow fast update

    input:
        - features(torch.Tensor) : (h, w, 3)
        - seg_map(torch.Tensor) : (h,w)
    
    output:
        - contrastive loss value 
    """
    loss = 0.0
    h, w = features_fast.shape[0], features_fast.shape[1]
    sampled_indices = torch.randint(0, h*w, (num_samples,))  # sample pixels 
    sampled_indices = sampled_indices.to(objects.device)
    indices_fast, indices_slow = torch.split(sampled_indices, num_samples//2, dim=0) #non overlap pixels (n_pixels, )
    features_fast_flatten = features_fast.reshape(-1, features_fast.shape[-1]) #h*w, 3
    features_slow_flatten = features_slow.reshape(-1, features_fast.shape[-1])
    sampled_features_fast = features_fast_flatten[indices_fast]  # u  : shape must be (N_sample//2,3) 
    sampled_features_slow = features_slow_flatten[indices_slow] 


    class_ids = torch.unique(objects) 
    for class_id in class_ids:  # obj id 0인경우 > class_objects에서 마스킹 됨
        if class_id == -1.:
            continue
        class_objects = objects == class_id  # 해당 클래스 마스크
        num_obj_pixel = torch.sum(class_objects)
        if num_obj_pixel <= 1:
            continue
              
        class_features_fast = features_fast * class_objects.unsqueeze(-1)  # 클래스 픽셀 마스킹 
        class_features_slow = features_slow * class_objects.unsqueeze(-1)

        class_features_fast = class_features_fast.reshape(-1, class_features_fast.shape[-1])  # (H*W , 3)
        class_features_slow = class_features_slow.reshape(-1, class_features_slow.shape[-1])

        class_objects_flatten = class_objects.flatten()
        sampled_cls_indices_fast = indices_fast[class_objects_flatten[indices_fast] == 1.]  # 샘플링한 픽셀 중에서 클래스에 해당하는 픽셀들 
        sampled_cls_indices_slow = indices_slow[class_objects_flatten[indices_slow] == 1.]

        sampled_cls_features_fast = class_features_fast[sampled_cls_indices_fast]  # u+
        sampled_cls_features_slow = class_features_slow[sampled_cls_indices_slow]

        positive_similarities = _rbf_kernel(sampled_cls_features_fast, sampled_cls_features_slow, gamma=gamma)  # u+ <-> u'+  , (pos_pixel1, pos_pixel2)
        
        num_positive_pairs = torch.sum(positive_similarities > 0)
        if num_positive_pairs > 0:
            positive_loss = positive_similarities.sum()
            loss += positive_loss

    if loss == 0.0:
        raise ValueError("positive loss: zero value")
    
    # after adding all pos pair, we should take log scale
    loss = -torch.log(loss)

    similarities_total = _rbf_kernel(sampled_features_fast, sampled_features_slow, gamma=gamma)
    total_loss = torch.log(similarities_total.sum())
    loss += total_loss

    torch.cuda.empty_cache()
    return loss / (num_samples//2)  #divide ? 

# #TODO : need to debug - pos 가 total 보다 영향이 커서 전체 loss가 음수가 되는 현상이 있음
# def contrastive_1d_loss_vectorize(features, objects, gamma=0.01, num_samples=2048):
#     """
#     contrastive loss implementation with vectorizing
#     input:
#         - features(torch.Tensor) : (3, h, w)
#         - seg_map(torch.Tensor) : (h,w)
    
#     output:
#         - contrastive loss value 
#     """
#     loss = 0.0
#     h, w = features.shape[1:]
#     sampled_indices = torch.randint(0, h*w, (num_samples,))  # sample pixels 
#     sampled_indices = sampled_indices.to(objects.device)
#     features_flatten = features.permute(1,2,0).reshape(-1, features.shape[0]) #h*w, 3
#     sampled_features = features_flatten[sampled_indices]
#     class_masks = (objects.view(-1, 1) == torch.unique(objects).view(1, -1)).float() #(h*w, n_obj) each column is mask of specific class 
#     class_masks[:, torch.unique(objects) == -1] = 0 
    
#     class_features = class_masks.unsqueeze(1) * features_flatten.unsqueeze(-1) # h*w, f_dim, n_obj

#     #TODO 에러 잡음 -> class_feature에서 해당 오브젝트가 아닌 픽셀들은 0값이됨(마스킹 했으니까) -> 아닌거랑 아닌거 뺴면 0이되고 exp통과시 1로됨
#     num_obj_pixels = class_masks.sum(dim=0)
#     valid_classes = num_obj_pixels >= 1
    
#     class_features = class_features[:, :, valid_classes]

#     sampled_features = class_features[sampled_indices] # n_pixel, f_dim, n_obj

#     #같은 오브젝트들 끼리 -> 픽셀wise하게 rbf kernel 계산
#     similarities = _rbf_kernel(sampled_features, sampled_features, diff_dim=-2, gamma=gamma) # (npix,npix,nobj)
#     pos_sim = similarities.triu(diagonal=1)
#     #num_pos = torch.sum(pos_sim > 0)
#     loss += -torch.log(pos_sim.sum())

#     similarities_total = _rbf_kernel(sampled_features, sampled_features, gamma=gamma)
#     total_sim = similarities_total.triu(diagonal=1)
#     total_loss = torch.log(total_sim.sum())
#     loss += total_loss
#     torch.cuda.empty_cache()
#     return loss
