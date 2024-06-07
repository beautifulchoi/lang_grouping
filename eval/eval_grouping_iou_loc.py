import json
import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm
from PIL import Image

import sys
from utilities import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result
import colormaps
from openclip_encoder_grouping import OpenCLIPNetwork

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.vis_utils import get_feature_map
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything


@hydra.main(config_path="../arguments/eval", config_name="eval_grouping_config")
def run(cfg: DictConfig) -> None:
    seed_everything(0)

    # NOTE config setting
    dataset_name = cfg.dataset_name
    mask_thresh = cfg.mask_thresh
    feat_dir = cfg.feat_dir
    seg_dir = cfg.seg_dir
    output_path = os.path.join(cfg.output_dir, dataset_name)
    json_folder = os.path.join(cfg.json_folder, dataset_name)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, f'{timestamp}.log')
    logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)
    evaluate(feat_dir, seg_dir, output_path, json_folder, mask_thresh, logger)

def evaluate(feat_dir: str, seg_dir: str, output_path: str, json_folder: str, mask_thresh: float, logger: logging.Logger):
    """
    input:
        - feat_dir : clip feature directory(should be preprocessed) #TODO novel view(test view 인 경우) 에는 가까운 view들의 feature들로 할당 해줘야 함
        - seg_dir : rendered segmentation mask directory
        - output_path : output saved directory
        - json_folder : open voc test label's info json's path
        - mask_thresh : threshold of masking using relavancy map
        - logger : evaluation logging
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )

    gt_ann, image_shape, image_paths = eval_gt_lerfdata(Path(json_folder), Path(output_path))
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    sem_feat = np.zeros((len(eval_index_list), *image_shape, 512), dtype=np.float32)

    # instantiate openclip
    clip_model = OpenCLIPNetwork(device)

    chosen_iou_all = []
    acc_num = 0

    # 이미지 순서는 1번부터 시작이지만 렌더링 넘버링은 0번 부터 
    # loop iteration and eval each image's open voc 
    for j, idx in tqdm(enumerate(eval_index_list), desc="evaluate progress"): #should add 1 on idx
        img_name = os.path.basename(image_paths[j]).split('.')[0]
        rgb_img = cv2.imread(image_paths[j])[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)
        sem_feat = get_feature_map(seg_dir, feat_dir, idx+1).cuda()
        img_ann = gt_ann[f'{idx}']
        clip_model.set_positives(list(img_ann.keys()))
        
        save_name = Path(output_path) / f'{idx+1:0>5}'
        c_iou_list = activate_stream(sem_feat.permute(1,2,0), rgb_img, clip_model, save_name, img_ann,
                                        thresh=mask_thresh, colormap_options=colormap_options)
        chosen_iou_all.extend(c_iou_list)
        acc_num_img = lerf_localization(sem_feat.permute(1,2,0), rgb_img, clip_model, save_name, img_ann)
        acc_num += acc_num_img

    # # iou
    mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    logger.info(f'trunc thresh: {mask_thresh}')
    logger.info(f"iou chosen: {mean_iou_chosen:.4f}")
    # localization acc
    total_bboxes = 0
    for img_ann in gt_ann.values():
        total_bboxes += len(list(img_ann.keys()))
    acc = acc_num / total_bboxes
    logger.info("Localization accuracy: " + f'{acc:.4f}')


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def activate_stream(sem_map, 
                    image, 
                    clip_model, 
                    image_name: Path = None,
                    img_ann: Dict = None, 
                    thresh : float = 0.5, 
                    colormap_options = None):
    valid_map = clip_model.get_max_across(sem_map)                 # kx832x1264
    n_prompt, h, w = valid_map.shape

    # positive prompts
    chosen_iou_list = []
    for k in range(n_prompt):

        # NOTE Find the maximum point in the activation value graph after adding the filtering result.
        scale = 30
        kernel = np.ones((scale,scale)) / (scale**2)
        np_relev = valid_map[k].cpu().numpy()
        avg_filtered = cv2.filter2D(np_relev, -1, kernel) #box filter
        avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
        valid_map[k] = 0.5 * (avg_filtered + valid_map[k])
        
        output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}'
        output_path_relev.parent.mkdir(exist_ok=True, parents=True)
        colormap_saving(valid_map[k].unsqueeze(-1), colormap_options,
                        output_path_relev)
        
        # NOTE Consistent with lerf, activation values below 0.5 are considered background
        p_i = torch.clip(valid_map[k] - 0.5, 0, 1).unsqueeze(-1)
        valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
        mask = (valid_map[k] < 0.5).squeeze()
        valid_composited[mask, :] = image[mask, :] * 0.3
        output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}'
        output_path_compo.parent.mkdir(exist_ok=True, parents=True)
        colormap_saving(valid_composited, colormap_options, output_path_compo)
        
        # truncate the heatmap into mask
        output = valid_map[k]
        output = output - torch.min(output)
        output = output / (torch.max(output) + 1e-9)
        output = output * (1.0 - (-1.0)) + (-1.0)
        output = torch.clip(output, 0, 1)

        mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
        mask_pred = smooth(mask_pred)

        mask_gt = img_ann[clip_model.positives[k]]['mask'].astype(np.uint8)
        
        # calculate iou
        intersection = np.sum(np.logical_and(mask_gt, mask_pred))
        union = np.sum(np.logical_or(mask_gt, mask_pred))
        iou = np.sum(intersection) / np.sum(union)

        score = valid_map[k].max()
        chosen_iou_list.append(iou)
        
        # save for visulsization
        save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        vis_mask_save(mask_pred, save_path)

    return chosen_iou_list


def lerf_localization(sem_map, image, clip_model, image_name, img_ann):
    output_path_loca = image_name / 'localization'
    output_path_loca.mkdir(exist_ok=True, parents=True)

    valid_map = clip_model.get_max_across(sem_map)                 # kx832x1264
    # positive prompts
    acc_num = 0
    positives = list(img_ann.keys())
    for k in range(len(positives)):
        select_output = valid_map[k]
        
        # NOTE Find the maximum point in the smoothed activation value plot
        scale = 30
        kernel = np.ones((scale,scale)) / (scale**2)
        np_relev = select_output.cpu().numpy()
        avg_filtered = cv2.filter2D(np_relev, -1, kernel) # h, w

        score = avg_filtered.max()
        coord = np.nonzero(avg_filtered == score)
        coord_final = np.asarray(coord).transpose(1,0)[..., ::-1]
        
        for box in img_ann[positives[k]]['bboxes'].reshape(-1, 4):
            flag = 0
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            for cord_list in coord_final:
                if (cord_list[0] >= x_min and cord_list[0] <= x_max and 
                    cord_list[1] >= y_min and cord_list[1] <= y_max):
                    acc_num += 1
                    flag = 1
                    break
            if flag != 0:
                break
        
        # NOTE The averaged results are added to the original results to suppress noise and keep the activation boundaries clear
        avg_filtered = torch.from_numpy(avg_filtered).unsqueeze(-1).to(select_output.device)
        torch_relev = 0.5 * (avg_filtered + select_output.unsqueeze(-1)) # TODO error debugging
        p_i = torch.clip(torch_relev - 0.5, 0, 1)
        valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
        mask = (torch_relev < 0.5).squeeze()
        valid_composited[mask, :] = image[mask, :] * 0.3
        
        save_path = output_path_loca / f"{positives[k]}.png"
        show_result(valid_composited.cpu().numpy(), coord_final,
                    img_ann[positives[k]]['bboxes'], save_path)
    return acc_num

def eval_gt_lerfdata(json_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:
    """
    organise lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.json')))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.jpg')))
    gt_ann = {}
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)
        
        h, w = gt_data['info']['height'], gt_data['info']['width']
        idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1 
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']
            box = np.asarray(prompt_data['bbox']).reshape(-1)           # x1y1x2y2
            mask = polygon_to_mask((h, w), prompt_data['segmentation'])
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask
            
            # # save for visulsization
            save_path = ouput_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, (h, w), img_paths

if __name__ == "__main__":
    run()