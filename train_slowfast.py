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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, contrastive_1d_loss, slowfast_contrastive_1d_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, find_overlap_cls
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from copy import deepcopy

#set seed automatically
from pytorch_lightning import seed_everything

#configuration setter
from omegaconf import DictConfig, OmegaConf
import logging
import hydra

import wandb
    
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(config_path="arguments/train", config_name="train_config")
def run(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    network_gui.init(cfg.ip, cfg.port)
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)
    seed_everything(0)

    if cfg.start_checkpoint:
        cfg.start_checkpoint = os.path.join(cfg.dataset.model_path, f"chkpnt{cfg.start_checkpoint}.pth")

    if cfg.dataset.feature_level and cfg.opt.include_lang_feature:
        cfg.dataset.model_path = cfg.dataset.model_path + f"_{str(cfg.dataset.feature_level)}"
        cfg.dataset.lf_path = os.path.join(os.path.join(cfg.dataset.source_path, cfg.dataset.language_features_name))
    
    if cfg.use_wandb:
        wandb.init(project="gaussian-splatting")
        wandb.config.args = cfg
        wandb.run.name = cfg.dataset.model_path

    #start train
    training(cfg.dataset, cfg.opt, cfg.pipe, cfg.test_iterations, 
             cfg.save_iterations, cfg.checkpoint_iterations, cfg.start_checkpoint, cfg.debug_from, cfg.use_wandb)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb = False):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if opt.include_lang_feature:
        if not checkpoint:
            raise ValueError("checkpoint missing!!!!!")
        
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        if len(model_params) == 13 and opt.include_lang_feature:
            first_iter = 0
        gaussians.restore(model_params, opt)
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    gaussians_slow = None
    
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, opt, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        rand_idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)

        if opt.contrastive.multi_view:
            if rand_idx >= len(viewpoint_stack):
                rand_idx -= 1
            if not viewpoint_stack:
                viewpoint_cam_related = viewpoint_cam
            else:
                viewpoint_cam_related = viewpoint_stack[rand_idx]  # near view selected

        #slow_fast initialization
        if opt.contrastive.slow_fast:
            if iteration == opt.contrastive.slow_fast.start_iter:
                gaussians_slow = deepcopy(gaussians)
                with torch.no_grad():
                    for param in gaussians_slow.capture(True):
                        if isinstance(param, torch.Tensor):
                            param.requires_grad = False
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, opt)
        image, language_feature, viewspace_point_tensor, visibility_filter, radii, objects =\
                render_pkg["render"], render_pkg["language_feature_image"], render_pkg["viewspace_points"], \
                render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"]
        
        # Loss
        contrast_loss = None
        if opt.include_lang_feature:
            gt_language_feature, language_feature_mask, seg_map = viewpoint_cam.get_language_feature(language_feature_dir=dataset.lf_path, 
                                                                                                     feature_level=dataset.feature_level,
                                                                                                     need_segmap=True)
            Ll1 = l1_loss(language_feature*language_feature_mask, gt_language_feature*language_feature_mask)        
            loss = Ll1
            log.info(f"feature - L1 loss: {Ll1}")
            if opt.contrastive and iteration % opt.contrastive.interval == 0 and iteration>=opt.contrastive.start_iter:
                gt_mask = (seg_map * language_feature_mask).squeeze(0)
                gt_mask = gt_mask.to(torch.uint8)

                # multi-view contrastive loss
                if opt.contrastive.multi_view:
                    gt_objects = viewpoint_cam.objects
                    gt_language_feature_related, language_feature_mask_related, seg_map_related = \
                        viewpoint_cam_related.get_language_feature(language_feature_dir=dataset.lf_path,
                                                                    feature_level=dataset.feature_level,
                                                                    need_segmap =True)
                    gt_objects = viewpoint_cam.objects
                    gt_objects_related = viewpoint_cam_related.objects
                    obj_mask, obj_mask_related , ovl_cls = find_overlap_cls(gt_objects, gt_objects_related)
                    if len(ovl_cls) <= 1:
                        print(f"ovl_cls num: {len(ovl_cls)}")
                        continue
                    render_pkg_related = render(viewpoint_cam_related, gaussians, pipe, background, opt)
                    language_feature_related = render_pkg_related["language_feature_image"]
                    ovl_lang_feature = language_feature*obj_mask
                    ovl_lang_feature_related = language_feature_related*obj_mask_related
                    gt_mask_related = (seg_map_related * language_feature_mask_related).squeeze(0)
                    gt_mask_related = gt_mask_related.to(torch.uint8)

                    #_contrast_loss = contrastive_1d_loss(ovl_lang_feature.permute(1,2,0), gt_mask, num_samples=1024) # lang feature: 3 728 986 // obj_mask : 728 986
                    #_contrast_loss_related = contrastive_1d_loss(ovl_lang_feature_related.permute(1,2,0), gt_mask_related, num_samples=1024)

                    _contrast_loss = contrastive_1d_loss(language_feature.permute(1,2,0), gt_mask, num_samples=1024) 
                    _contrast_loss_related = contrastive_1d_loss(ovl_lang_feature_related.permute(1,2,0), gt_mask_related, num_samples=1024)
                    obj_scale = obj_mask != 0.
                    obj_scale = obj_scale.sum()
                    obj_scale_related = obj_mask_related != 0.
                    obj_scale_related = obj_scale_related.sum()
                    
                    scale_weight = obj_scale/(obj_scale + obj_scale_related)
                    scale_weight_related = obj_scale_related/(obj_scale + obj_scale_related)

                    contrast_loss = (scale_weight * _contrast_loss) + (scale_weight_related * _contrast_loss_related)
                    if torch.isnan(contrast_loss):
                        print()
                    loss += contrast_loss
                    # TODO per object scale awareable weight needed
                    log.info(f"feature - contrastive loss: {contrast_loss}, weight : {scale_weight}, weight_related : {scale_weight_related}")
                
                # do slow fast implementation 
                elif opt.contrastive.slow_fast and gaussians_slow:
                    #slow 도 device에 함께 올라가 있어야함
                    render_pkg_slow = render(viewpoint_cam, gaussians_slow, pipe, background, opt)
                    language_feature_slow = render_pkg_slow["language_feature_image"]  #여기까지는 gt_mask가 살아있다가 함수에 들어가면 갑자기 값이 안보여
                    language_feature_slow = language_feature_slow.to(language_feature)
                    contrast_loss = slowfast_contrastive_1d_loss(language_feature.permute(1,2,0), language_feature_slow.permute(1,2,0), gt_mask, num_samples = 1024) #TODO error debuginng
                    loss += contrast_loss
                    log.info(f"feature - slowfast contrastive loss: {contrast_loss}")
                # 1 view contrastive loss for all semantics
                else: 
                    contrast_loss = contrastive_1d_loss(language_feature.permute(1,2,0), gt_mask, num_samples=1024)
                    loss += contrast_loss
                    log.info(f"feature - contrastive loss: {contrast_loss}")

        else:
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)            
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            log.info(f"photometric loss: {loss}")
            # obj loss
            # gt_obj = viewpoint_cam.objects.cuda().long()
            # logits = classifier(objects)
            # _obj_loss = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
            # _obj_loss = _obj_loss / torch.log(torch.tensor(num_classes))  # normalize to (0,1)
            # loss += _obj_loss
            # log.info(f"object loss: {_obj_loss}")
        
        loss.backward()
        iter_end.record()

        if gaussians_slow:
            # update slow network
            ema_update_slownet(gaussians_slow, gaussians, momentum= 0.9)
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(use_wandb, iteration, 
                            Ll1, loss, l1_loss,
                            iter_start.elapsed_time(iter_end), testing_iterations, scene, 
                            render, (pipe, background, opt), contrast_loss,
                            )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            #log on local too
            log.info(f"iteration {iteration}/{opt.iterations+1}, total loss: {loss}")
            # Densification
            if not opt.include_lang_feature:
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                # cls_optimizer.step() # 
                # cls_optimizer.zero_grad()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(opt.include_lang_feature), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
def prepare_output_and_logger(dataset):    # 
    if not dataset.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        dataset.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok = True)
    dataset_yaml = OmegaConf.to_yaml(dataset)
    with open(os.path.join(dataset.model_path, "cfg_args.yaml"), 'w') as cfg_log_f:
        cfg_log_f.write(dataset_yaml)


# use wanb for report
def training_report(use_wandb, iteration, Ll1, loss, l1_loss, 
                    elapsed, testing_iterations, scene : Scene, 
                    renderFunc, renderArgs, loss_contrast= None):

    if use_wandb:
        if loss_contrast:
            wandb.log({"train_loss_patches/l1_loss": Ll1.item(), "train_loss_patches/total_loss": loss.item(), 
                       "train_loss_patches/loss_contrastive": loss_contrast.item(), "iter_time": elapsed, "iter": iteration})
        else:
            wandb.log({"train_loss_patches/l1_loss": Ll1.item(), "train_loss_patches/total_loss": loss.item(), 
                       "iter_time": elapsed, "iter": iteration})
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if use_wandb:
                        if idx < 5:
                            wandb.log({config['name'] + "_view_{}/render".format(viewpoint.image_name): [wandb.Image(image)]})
                            if iteration == testing_iterations[0]:
                                wandb.log({config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name): [wandb.Image(gt_image)]})
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                
                if use_wandb:
                    wandb.log({config['name'] + "/loss_viewpoint - l1_loss": l1_test, config['name'] + "/loss_viewpoint - psnr": psnr_test})
        if use_wandb:
            wandb.log({"scene/opacity_histogram": scene.gaussians.get_opacity, "total_points": scene.gaussians.get_xyz.shape[0], "iter": iteration})
        torch.cuda.empty_cache()

# reference : https://github.com/yashbhalgat/Contrastive-Lift/blob/main/trainer/train_panopli_tensorf.py
def ema_update_slownet(slow_gs, fast_gs, momentum=0.9):
    # EMA update for the teacher
    with torch.no_grad():
        for param_q, param_k in zip(fast_gs.capture(True), slow_gs.capture(True)):
            if isinstance(param_q, torch.Tensor):
                param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)
                param_k.requires_grad = False
if __name__ == "__main__":
    run()
    print("\nTraining complete.")
