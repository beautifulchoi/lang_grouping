import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy

import torch
import torchvision
from torch import nn

import hydra
from omegaconf import DictConfig

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"
from pytorch_lightning import seed_everything


@hydra.main(config_path="arguments/preprocess", config_name="preprocess_config.yaml")
def run(cfg: DictConfig):
    print("preprocessing" + cfg.source_path)
    seed_everything(cfg.seed)
    model = OpenCLIPNetwork(OpenCLIPNetworkConfig)

    dataset_path = cfg.source_path
    img_folder = os.path.join(dataset_path, 'images')
    seg_folder = os.path.join(dataset_path, 'object_mask')
    data_list = os.listdir(img_folder)
    data_list.sort()
    img_list = []
    seg_list = []
    WARNED = False
   
    for data_path in tqdm(data_list, desc="Preprocessing progress"):
        image_path = os.path.join(img_folder, data_path)
        seg_path = os.path.join(seg_folder, data_path)
        if os.path.splitext(seg_path)[1] != '.ppg':
            seg_path = os.path.splitext(seg_path)[0] + '.png'
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        seg = cv2.imread(seg_path)
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
        orig_w, orig_h = image.shape[1], image.shape[0]

        if orig_h > 1080:
            if not WARNED:
                print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                    "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                WARNED = True
            global_down = orig_h / 1080
        else:
            global_down = 1

            
        scale = float(global_down)
        resolution = (int( orig_w  / scale), int(orig_h / scale))
        
        image = cv2.resize(image, resolution)
        seg = cv2.resize(seg, resolution)
        image = torch.from_numpy(image)
        seg = torch.from_numpy(seg)
        img_list.append(image)
        seg_list.append(seg)
    images = [img_list[i][None, ...] for i in range(len(img_list))]
    imgs = torch.cat(images)
    segs = [seg_list[i][None, ...] for i in range(len(seg_list))]
    segs = torch.cat(segs)

    save_folder = os.path.join(dataset_path, 'language_features_instance')
    os.makedirs(save_folder, exist_ok=True)
    create(imgs, segs, data_list, model, save_folder)


@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)

def get_seg_img(mask, image): #여기 마스크는 위에서 처리해주자 (그 라벨 해당 부분은 1로 나머지 0으로 함수타기 전에 처리)
    image = image
    image = image * mask[..., None]
    mask = np.array(mask)
    image = np.array(image)
    coords= cv2.findNonZero(mask)
    x,y,w,h = cv2.boundingRect(coords)
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img  
    
def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def mask2segmap(masks, image):
    seg_img_list = []
    seg_map = -np.ones(image.shape[:2], dtype=np.int32)
    for i in range(len(masks)):
        mask = masks[i]
        seg_img = get_seg_img(mask, image)
        pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
        seg_img_list.append(pad_seg_img)

        seg_map[masks[i]['segmentation']] = i
    seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
    seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

    return seg_imgs, seg_map

def embed_clip_coarse_tiles(model, img, seg, num_obj): # seg : segmentation map, num_obj : 옵젝 총 갯수
    h, w = seg.shape
    tiles = np.zeros((num_obj, 224, 224, 3))
    for label in torch.unique(seg): # clip encoder로 넣어줄 때 타일을 넣는데, 이 타일이 어떤형태인지 파악할 필요가 있음
        mask = torch.where(seg==label, 1, 0)
        seg_img = get_seg_img(mask, img) #mask 와 이미지 내부에서 tensor -> numpy 로 만들어줘야함
        pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
        tiles[label] = pad_seg_img

    # [num_obj, 3, 224, 224] 사이즈로 tile을 만들어 주면 된다. 
    tiles = (torch.from_numpy(tiles.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')
    
    with torch.no_grad():
        clip_embeds = model.encode_image(tiles)
    clip_embeds /= clip_embeds.norm(dim=-1, keepdim=True)
    clip_embeds = clip_embeds.detach().cpu().half()
    #clip_embeds[clip_embeds != torch.unuque(seg)] = 
    return clip_embeds, torch.unique(seg) # (num_obj, 512) , 없는 물건은 0 처리

# num_obj, 512 채널의 object 단위로의 clip feature를 만들면 됨
def create(image_list, seg_list, data_list, model, save_folder):
    """
    input
        - image_list (torch.Tensor) : tensor list of image arrays(sorted)  (N, 3, h, w)
        - seg_list (torch.Tensor) : tensor list of seg map arrays(sorted)  (N, h, w)
        - data_list (list[str]) : list of image basename
        - model (nn.Module) : CLIP model  
        - save_folder (str) : path where to save
    """
    assert image_list is not None, "image_list must be provided to generate features"
    assert seg_list is not None, "seg_list must be provided to generate features"
    embed_size=512
    timer = 0
    img_embeds = torch.zeros((len(image_list), 300, embed_size))

    for i, img in tqdm(enumerate(image_list), desc="Embedding images", leave=False):
        timer += 1
        seg = seg_list[i]
        clip_embeds, cls = embed_clip_coarse_tiles(model, img, seg, 256) #(num_obj, 512)
        # make zero if not contained
        tot_cls = torch.arange(0,256)
        cls_mask = ~torch.isin(tot_cls, cls)
        clip_embeds[cls_mask] = 0 
        
        total_length = len(clip_embeds) # num_obj 
        
        if total_length > img_embeds.shape[1]:
            pad = total_length - img_embeds.shape[1]
            img_embeds = torch.cat([
                img_embeds,
                torch.zeros((len(image_list), pad, embed_size))
            ], dim=1)

        img_embeds[i, :total_length] = clip_embeds
        
    for i in range(img_embeds.shape[0]):
        save_path = os.path.join(save_folder, data_list[i].split('.')[0])
        save_path_f = save_path + '_f.npy'
        np.save(save_path_f,img_embeds[i].numpy())

if __name__ == '__main__':
    run()