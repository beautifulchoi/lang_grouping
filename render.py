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
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import hydra
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import seed_everything

@hydra.main(config_path="arguments/render", config_name="render_config.yaml")
def run(cfg: DictConfig):
    #log.info(OmegaConf.to_yaml(cfg))
    print("Rendering " + cfg.dataset.model_path)
    seed_everything(0)
    if cfg.dataset.feature_level:
        cfg.dataset.model_path = cfg.dataset.model_path + f"_{str(cfg.dataset.feature_level)}"
        cfg.dataset.lf_path = os.path.join(os.path.join(cfg.dataset.source_path, cfg.dataset.language_features_name))
    
    render_sets(cfg.dataset, cfg.iteration, cfg.pipe, cfg.skip_train, cfg.skip_test, cfg.opt)

#TODO : render set grouping에 맞게 변경
def render_set(dataset, name, iteration, views, gaussians, pipeline, background, opt):
    render_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration), "gt")
    render_npy_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration), "renders_npy")
    gts_npy_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration), "gt_npy")
    # colormask_path = os.path.join(dataset.model_pathh, name, "ours_{}".format(iteration), "objects_feature16")
    # gt_colormask_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration), "gt_objects_color")
    # pred_obj_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration), "objects_pred")

    makedirs(render_npy_path, exist_ok=True)
    makedirs(gts_npy_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    # makedirs(colormask_path, exist_ok=True)
    # makedirs(gt_colormask_path, exist_ok=True)
    # makedirs(pred_obj_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output = render(view, gaussians, pipeline, background, opt)

        if not opt.include_lang_feature:
            rendering = output["render"] 
        else:
            rendering = output["language_feature_image"]
            
        if not opt.include_lang_feature:
            gt = view.original_image[0:3, :, :]
            
        else:
            gt, mask = view.get_language_feature(os.path.join(dataset.source_path, dataset.language_features_name), feature_level=dataset.feature_level)

        np.save(os.path.join(render_npy_path, '{0:05d}'.format(idx) + ".npy"),rendering.permute(1,2,0).cpu().numpy())
        np.save(os.path.join(gts_npy_path, '{0:05d}'.format(idx) + ".npy"),gt.permute(1,2,0).cpu().numpy())
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
               
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, opt):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)

        num_classes = dataset.num_classes
        # classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        # classifier.cuda()
        # classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))

        checkpoint = os.path.join(dataset.model_path, f'chkpnt{iteration}.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt, mode='test')
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, opt)

        if not skip_test:
             render_set(dataset, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, opt)

if __name__ == "__main__":
    # Set up command line argument parser
    
    # parser = ArgumentParser(description="Testing script parameters")
    # model = ModelParams(parser, sentinel=True)
    # pipeline = PipelineParams(parser)
    # parser.add_argument("--iteration", default=-1, type=int)
    # parser.add_argument("--skip_train", action="store_true")
    # parser.add_argument("--skip_test", action="store_true")
    # parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--include_feature", action="store_true")

    # args = get_combined_args(parser)
    # print("Rendering " + args.model_path)

    # safe_state(args.quiet)

    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)
    run()