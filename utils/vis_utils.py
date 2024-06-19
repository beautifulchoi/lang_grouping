import sklearn
import sklearn.decomposition
import torch
from torch import nn
import numpy as np
import os
from PIL import Image
from pytorch_lightning import seed_everything


def feature_visualize_saving(feature, seed):
    fmap = feature[None, :, :, :] # torch.Size([1, 512, h, w])
    fmap = nn.functional.normalize(fmap, dim=1)
    pca = sklearn.decomposition.PCA(3, random_state=seed)
    f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy()
    transformed = pca.fit_transform(f_samples)
    feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
    feature_pca_components = torch.tensor(pca.components_).float().cuda()
    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)
    del f_samples
    vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
    vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu()
    return vis_feature


def get_feature_map(seg_dir, feature_dir, img_number):
    seg_path = os.path.join(seg_dir, f'{int(img_number)-1:0>5}'+ '.png')
    feature_path = os.path.join(feature_dir, "frame_" + f'{img_number:0>5}' + '_f.npy')
    seg_map = np.array(Image.open(seg_path))
    if not os.path.isfile(feature_path):
        feature_rel1 = np.load(os.path.join(feature_dir, "frame_" + f'{img_number+1:0>5}' + '_f.npy'))
        feature_rel2 = np.load(os.path.join(feature_dir, "frame_" + f'{img_number-1:0>5}' + '_f.npy'))
        feature_map = average_feature(np.concatenate([[feature_rel1], [feature_rel2]], axis=0))
    feature_map = np.load(feature_path) #256, 512
    h, w = seg_map.shape
    y, x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    seg = seg_map[y, x].squeeze(-1)
        
    point_feature = feature_map[seg[None]].squeeze(0)
    point_feature = torch.from_numpy(point_feature.reshape(h,w, -1)).permute(2, 0, 1)

    return point_feature

def average_feature(features): #(N, 256, 512)
    #n, num_obj, f_dim = features.shape
    avg_feature = np.sum(features, axis = 0)
    non_zero_mask = ~np.all(features == 0, axis=(1, 2))
    div_list = np.sum(non_zero_mask, axis=0)
    div_list = div_list[None, :]
    div_list = np.where(div_list!= 0, div_list, 10e-6)
    avg_feature /= div_list
    
    return avg_feature
        
