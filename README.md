# Lang-Grouping: Object-centric semantic grouping for better understanding 3D scenes
<sub>
<b>Yongjun Choi</b>, <b>Seungoh Han</b>, <b>Junhee Lee</b><br>
</sub>

*(3D Vision Final Project, UNIST, 2024)*  
---

This repository contains the implementation of **LangGrouping**, a 3D vision project exploring language-guided Gaussian-based segmentation and grouping for multi-view scenes.  
The project extends the idea of **language-conditioned feature learning** in 3D Gaussian Splatting to perform semantic grouping without explicit supervision.

---

## Overview

**LangGrouping** combines multi-view visual reconstruction with text-guided reasoning to cluster and segment 3D regions.  
Our pipeline integrates:

- Using **SAM**'s highest level segmentation map and Tracker for object-wise multi-view consistency
- Inject **Language features** extracted from pretrained CLIP encoders in to **3D Gaussian splats**.  
- **Object-centric** contrastive learning for semantic region clustering  

This work is inspired by *LangSplat (CVPR 2024 Highlight)* and *Gaussian Grouping(ECCV 2024)*.

---

## Key Features
- Multi-view Gaussian-based 3D grouping with language guidance  
- Integration of CLIP feature embeddings for semantic understanding  
- Modular PyTorch-based training and visualization pipeline  
- Simple and extensible framework for new 3D-language tasks  
