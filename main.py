import math
import torch
import torch.nn as nn
# from point_cloud_dataset import dummy_data, preprocessing
from torch.utils.data import Dataset, DataLoader
from PTv3_model import PointTransformerV3, load_weights_ptv3_nucscenes_seg
from datasets import load_dataset


def main():
    # dataset = load_dataset("Pointcept/s3dis-compressed",
    #                        cache_dir="/home/luu/projects/PTv3-distill/point_cloud_dataset/s3dis",
    #                        split="train",
    #                        download_mode="force_redownload")

    model = PointTransformerV3(in_channels=4, pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
                               cls_mode=False, pdnorm_bn=False, mlp_ratio=4, qkv_bias=True,
                               enable_flash=False, order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
                               stride=(2, 2, 2, 2), enc_depths=(2, 2, 2, 6, 2), enc_channels=(32, 64, 128, 256, 512),
                               enc_num_head=(2, 4, 8, 16, 32), enc_patch_size=(1024, 1024, 1024, 1024, 1024),
                               dec_depths=(2, 2, 2, 2), dec_channels=(64, 64, 128, 256), dec_num_head=(4, 4, 8, 16),
                               dec_patch_size=(1024, 1024, 1024, 1024), qk_scale=None, attn_drop=0.0, proj_drop=0.0,
                               drop_path=0.3, shuffle_orders=True, pre_norm=True, enable_rpe=False,
                               upcast_attention=False,
                               upcast_softmax=False, pdnorm_ln=False, pdnorm_decouple=True, pdnorm_adaptive=False,
                               pdnorm_affine=True)

    # Load pretrained weights
    pretrained_path = '/home/luu/projects/PTv3-distill/huggingface_model/PointTransformerV3/nuscenes-semseg-pt-v3m1-0-base/model/model_best.pth'

    pretrained_model = load_weights_ptv3_nucscenes_seg(model, pretrained_path)


def fix_specific_line(file_path, line_number, old_text, new_text):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()

    # Check if the line number is valid
    if line_number < len(lines):
        # Replace old_text with new_text in the specified line
        lines[line_number] = lines[line_number].replace(old_text, new_text)

    # Write the corrected lines back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

    print('done')


def preprocessed_s3dis():
    output_root = '/home/luu/projects/PTv3-distill/point_cloud_dataset/s3dis/preprocessed'
    dataset_root = '/home/luu/projects/PTv3-distill/point_cloud_dataset/s3dis/Stanford3dDataset_v1.2'

    "python3 pointcept/datasets/preprocessing/s3dis/ preprocess_s3dis.py " \
    "--dataset_root /home/luu/projects/PTv3-distill/point_cloud_dataset/s3dis/Stanford3dDataset_v1.2 " \
    "--output_root /home/luu/projects/PTv3-distill/point_cloud_dataset/s3dis/preprocessed"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

    # # Usage
    # file_path = '/home/luu/projects/PTv3-distill/point_cloud_dataset/s3dis/Stanford3dDataset_v1.2/Area_5/office_19/Annotations/ceiling'
    # line_number = 323473  # Index is zero-based, so 323474 in human terms is 323473 in Python
    # old_text = "103.0�0000"
    # new_text = "103.000000"

    # fix_specific_line(file_path, line_number, old_text, new_text)
