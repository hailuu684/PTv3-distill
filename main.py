import math
import torch
import torch.nn as nn
# from point_cloud_dataset import dummy_data, preprocessing
from torch.utils.data import Dataset, DataLoader
from dataloader import PTv3_Dataloader
from pointcept.utils.registry import Registry
from datasets import load_dataset
from PTv3_model import PointTransformerV3, load_weights_ptv3_nucscenes_seg
import pickle
import os

from pointcept.engines.train import TRAINERS

PRETRAINED_PATH = '/media/anda/hdd1/thomas/PTv3-distill/huggingface_model/PointTransformerV3/nuscenes-semseg-pt-v3m1-0-base/model/model_best.pth'
CONFIG_FILE = "configs/nuscenes/semseg-pt-v3m1-0-base.py"

def main():
    model = PointTransformerV3(
        in_channels=4,
        pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
        cls_mode=False,
        pdnorm_bn=False,
        mlp_ratio=4,
        qkv_bias=True,
        enable_flash=False,
        order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        upcast_attention=False,
        upcast_softmax=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True
        )

    # Load pretrained weights
    pretrained_model = load_weights_ptv3_nucscenes_seg(model, PRETRAINED_PATH)
    
    loader = PTv3_Dataloader(CONFIG_FILE)
    train_loader = loader.load_training_data()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model = pretrained_model.to(device)
    
    # Training loop
    pretrained_model.train()
    for batch_ndx, sample in enumerate(train_loader):
        print(sample["segment"].shape)
        
        # Move data to device
        sample = {k: v.to(device) for k, v in sample.items()}
        
        # Forward pass
        outputs = pretrained_model.forward(sample)
        print(outputs.keys())
        
        # Backward pass and optimization
        
        # ................
        break
    

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    