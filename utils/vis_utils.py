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
    seg_path = os.path.join(seg_dir, f'{img_number-1:0>5}'+ '.png')
    feature_path = os.path.join(feature_dir, "frame_" + f'{img_number:0>5}' + '_f.npy')
    seg_map = np.array(Image.open(seg_path))
    feature_map = np.load(feature_path) #256, 512
    h, w = seg_map.shape
    y, x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    seg = seg_map[y, x].squeeze(-1)
        
    point_feature = feature_map[seg[None]].squeeze(0)
    point_feature = torch.from_numpy(point_feature.reshape(h,w, -1)).permute(2, 0, 1)

    return point_feature

# test code
if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    seg_dir = '/home/splat/lang-grouping/lang-grouping/data/lerf_mask/figurines/object_mask/'
    feature_dir = '/home/splat/lang-grouping/lang-grouping/data/lerf_mask/figurines/language_features_instance/'
    save_path = '/home/splat/lang-grouping/lang-grouping/vis_instance_feature'

    image_names = []
    img_dir = '/home/splat/lang-grouping/lang-grouping/data/lerf_mask/figurines/images/'
    for filename in os.listdir(img_dir):
        _, file_extension = os.path.splitext(filename)
        image_name = os.path.splitext(filename)[0]
        image_names.append(image_name)

    for name in image_names:
        f_map = get_feature_map(seg_dir, feature_dir, name)
        vis_f = feature_visualize_saving(f_map)
        Image.fromarray((vis_f.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(save_path, name + "_feature_vis.png"))