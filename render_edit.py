# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-grouping, https://github.com/lkeab/gaussian-grouping
# GRAPHDECO research group, https://team.inria.fr/graphdeco


import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.vis_utils import get_feature_map
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import json
from torch import nn
import cv2
from PIL import Image
from sklearn.decomposition import PCA

from scipy.spatial import ConvexHull, Delaunay
#from render import feature_to_rgb, visualize_obj
import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from eval.openclip_encoder_grouping import OpenCLIPNetwork


@hydra.main(config_path="arguments/render", config_name="render_remove_config.yaml")
def run(cfg: DictConfig):
    #log.info(OmegaConf.to_yaml(cfg))
    print("Rendering " + cfg.dataset.model_path)
    seed_everything(0)
    
    removal(cfg.dataset, cfg.iteration, cfg.pipe, cfg.skip_train, cfg.skip_test, cfg.opt, cfg.removal_thresh)


def get_clip2obj_id(sem_map:torch.Tensor, seg_map:torch.Tensor, texts:list, device):
    clip_model = OpenCLIPNetwork(device)
    valid_map = clip_model.get_max_across(sem_map) # sem_map should be h,w,512
    clip_model.set_positives(texts)
    obj_ids = torch.tensor([], dtype=torch.int64).to(device)
    seg_map = seg_map + 1  #prevent 0 index is dleted

    for k in range(len(texts)):
        select_output = valid_map[k]
        scale = 30
        kernel = np.ones((scale,scale)) / (scale**2)
        np_relev = select_output.cpu().numpy()
        avg_filtered = cv2.filter2D(np_relev, -1, kernel)
        avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
        valid_map[k] = 0.5 * (avg_filtered + valid_map[k])

        mask = (valid_map[k] == torch.max(valid_map[k])).squeeze()
        
        is_inobj = torch.unique(mask)
        
        obj_id = torch.unique(mask * seg_map)
        obj_id = obj_id[obj_id!= 0]
        obj_id -= 1
        obj_ids = torch.cat((obj_ids, obj_id))
    
    obj_ids = obj_ids[obj_ids!= 0]
    print(obj_ids)
    return obj_ids


def points_inside_convex_hull(point_cloud, mask, remove_outliers=True, outlier_factor=1.0):
    """
    Given a point cloud and a mask indicating a subset of points, this function computes the convex hull of the 
    subset of points and then identifies all points from the original point cloud that are inside this convex hull.
    
    Parameters:
    - point_cloud (torch.Tensor): A tensor of shape (N, 3) representing the point cloud.
    - mask (torch.Tensor): A tensor of shape (N,) indicating the subset of points to be used for constructing the convex hull.
    - remove_outliers (bool): Whether to remove outliers from the masked points before computing the convex hull. Default is True.
    - outlier_factor (float): The factor used to determine outliers based on the IQR method. Larger values will classify more points as outliers.
    
    Returns:
    - inside_hull_tensor_mask (torch.Tensor): A mask of shape (N,) with values set to True for the points inside the convex hull 
                                              and False otherwise.
    """

    # Extract the masked points from the point cloud
    masked_points = point_cloud[mask].cpu().numpy() 

    # Remove outliers if the option is selected
    if remove_outliers:
        Q1 = np.percentile(masked_points, 25, axis=0)
        Q3 = np.percentile(masked_points, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR)) 
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
    else:
        filtered_masked_points = masked_points

    if masked_points is None:
        return -1
    
    # Compute the Delaunay triangulation of the filtered masked points
    delaunay = Delaunay(filtered_masked_points)

    # Determine which points from the original point cloud are inside the convex hull
    points_inside_hull_mask = delaunay.find_simplex(point_cloud.cpu().numpy()) >= 0

    # Convert the numpy mask back to a torch tensor and return
    inside_hull_tensor_mask = torch.tensor(points_inside_hull_mask, device='cuda')

    return inside_hull_tensor_mask

def removal_setup(gaussians, classifier, selected_obj_ids, removal_thresh):
    #selected_obj_ids = torch.tensor(selected_obj_ids).cuda()
    with torch.no_grad():
        logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
        prob_obj3d = torch.softmax(logits3d,dim=0)
        mask = prob_obj3d[selected_obj_ids, :, :] > removal_thresh # 

        mask3d = mask.any(dim=0).squeeze() 

        mask3d_convex = points_inside_convex_hull(gaussians._xyz.detach(),mask3d,outlier_factor=1.0)
        
        mask3d = torch.logical_or(mask3d,mask3d_convex)

        mask3d = mask3d.float()[:,None,None]

        # fix some gaussians
        mask3d = ~mask3d.bool().squeeze()

        xyz_sub = gaussians._xyz[mask3d].detach()
        features_dc_sub = gaussians._features_dc[mask3d].detach()
        features_rest_sub = gaussians._features_rest[mask3d].detach()
        opacity_sub = gaussians._opacity[mask3d].detach()
        scaling_sub = gaussians._scaling[mask3d].detach()
        rotation_sub = gaussians._rotation[mask3d].detach()
        objects_dc_sub = gaussians._objects_dc[mask3d].detach()

        # Construct nn.Parameters with specified gradients  tensor.detach().clone().requires_grad_(requires_grad)
        gaussians._xyz = nn.Parameter(xyz_sub.detach().clone().requires_grad_(False))
        gaussians._features_dc = nn.Parameter(features_dc_sub.detach().clone().requires_grad_(False))
        gaussians._features_rest = nn.Parameter(features_rest_sub.detach().clone().requires_grad_(False))
        gaussians._opacity = nn.Parameter(opacity_sub.detach().clone().requires_grad_(False))
        gaussians._scaling = nn.Parameter(scaling_sub.detach().clone().requires_grad_(False))
        gaussians._rotation = nn.Parameter(rotation_sub.detach().clone().requires_grad_(False))
        gaussians._objects_dc = nn.Parameter(objects_dc_sub.detach().clone().requires_grad_(False))
    
    return gaussians


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, opt, edit_command):
    render_path = os.path.join(model_path, name, "ours{}".format(iteration), f"{edit_command}_renders_{''.join(opt.select_texts)}")

    makedirs(render_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background, opt)
        rendering = results["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))


def removal(dataset : dict, iteration : int, pipeline : dict, 
            skip_train : bool, skip_test : bool, opt : OptimizationParams, removal_thresh : float):
    # 1. load gaussian checkpoint
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    num_classes = dataset.num_classes
    print("Num classes: ",num_classes)
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    classifier.cuda()
    classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(iteration),"classifier.pth")))
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # load gaussian
    checkpoint = os.path.join(dataset.model_path, f'chkpnt{iteration}.pth')
    (model_params, first_iter) = torch.load(checkpoint)
    check_include = {'lang': False, 'object': True }
    gaussians.restore(check_include, model_params, opt, mode='test')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_dir = os.path.join(dataset.model_path, "train/ours_30000/seg_map_pred")

    reference_img_number = f'{opt.reference_number:0>5}'
    select_texts = opt.select_texts

    seg_map = np.array(Image.open(os.path.join(seg_dir, f"{reference_img_number}.png")))
    seg_map = torch.from_numpy(seg_map).to(device)
    feat_dir = os.path.join(dataset.source_path, dataset.language_features_name)
    
    reference_img_number = opt.reference_number
    sem_map = get_feature_map(seg_dir, feat_dir, reference_img_number).to(device)
    
    select_obj_ids = get_clip2obj_id(sem_map.permute(1,2,0), seg_map, select_texts, device)

    # 2. remove selected object
    gaussians_edited = removal_setup(gaussians, classifier, select_obj_ids, removal_thresh)

    # 3. render new result
    with torch.no_grad():
        if not skip_train:
             render_set(dataset.model_path, "train", iteration, scene.getTrainCameras(), gaussians_edited, pipeline, background, opt, 'removal')

        if not skip_test:
             render_set(dataset.model_path, "test", iteration, scene.getTestCameras(), gaussians_edited, pipeline, background, opt, 'removal')



if __name__ == "__main__":
    run()
