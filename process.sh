#!/bin/bash

# get the language feature of the scene
python preprocess.py --dataset_name $dataset_path

# train the autoencoder
# cd autoencoder
# python train.py --dataset_path $dataset_path --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name ae_ckpt
# # e.g. python train.py --dataset_path ../data/sofa --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name lerf_mask/figurines

# # get the 3-dims language feature of the scene
# python test.py --dataset_name $dataset_path --dataset_name $dataset_name
# # e.g. python test.py --dataset_path ../data/sofa --dataset_name lerf_mask/figurines

# change hydra config's param to train vanilla GS
# python train.py dataset.model_path=lerf_mask/${casename} opt.include_lang_feature=False start_checkpoint=False

for level in 1 2 3
do
    python train.py dataset.source_path=${dataset_path} dataset.model_path=lerf_mask/${casename} opt.include_lang_feature=True start_checkpoint=30000 feature_level=${level}
    # e.g. python train.py -s data/sofa -m output/sofa --start_checkpoint data/sofa/sofa/chkpnt30000.pth --feature_level 3
done

for level in 1 2 3
do
    # render language features, if you want to render rgb, just set "include_lang_feature" to False
    python render.py dataset.source_path=${dataset_path} dataset.model_path=lerf_mask/${casename} opt.include_lang_feature=True feature_level=${level}
    # e.g. python render.py dataset.source_path="/home/gaussian-grouping/data/lerf_mask/figurines_{something other}" \
        #  dataset.model_path=lerf_mask/figurines_{something other}
        #  opt.include_lang_feature=True\
        #  feature_level=1
done