import math
import torch
import torch.nn as nn
# from point_cloud_dataset import dummy_data, preprocessing
from torch.utils.data import Dataset, DataLoader
from PTv3_model import PointTransformerV3, load_weights_ptv3_nucscenes_seg
import os


def main():
    # Create the dataset and DataLoader

    # # Initialize NuScenes instance and create a DataLoader
    # nusc = preprocessing.mini_nuscenes.NuScenes(version='v1.0-mini',
    #                                             dataroot='/home/luu/projects/distill_ptv3/point_cloud_dataset/mini_nuscenes',
    #                                             verbose=True)
    #
    # preprocessed_data_path = '/home/luu/projects/PTv3-distill/point_cloud_dataset' \
    #                          '/mini_nuscenes/preprocess_data/point_cloud_data.pkl'
    # if not os.path.exists(preprocessed_data_path):
    #
    #     save_preprocess_path = '/home/luu/projects/PTv3-distill/point_cloud_dataset/mini_nuscenes'
    #
    #     preprocessing.preprocess_point_cloud(nusc=nusc, out_dir=save_preprocess_path)
    #
    # # Create the dataset and DataLoader
    # nuscenes_dataset = preprocessing.mini_nuscenes.PreprocessedNuScenesDataset(preprocessed_data_path=preprocessed_data_path,
    #                                                                            use_augmentation=False)
    #
    # print(nuscenes_dataset[0])
    # mini_nuscene_dataloader = DataLoader(nuscenes_dataset, batch_size=4, shuffle=True, num_workers=4,
    #                                      collate_fn=None)
    #
    # # Iterate through the dataset
    # for i, (point_cloud, annotations) in enumerate(mini_nuscene_dataloader):
    #     print(f"Batch {i}:")
    #     print(f"Point Cloud Shape: {point_cloud.shape}")
    #     print(f"Annotations: {annotations}")
    #     break
    #
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

    # # Define a loss function and optimizer
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #
    # # Training loop
    # for epoch in range(10):  # Example of 10 epochs
    #     for batch_idx, (data_dict, labels) in enumerate(mini_nuscene_dataloader):
    #         optimizer.zero_grad()
    #
    #         # Forward pass
    #         outputs = model(data_dict)  # Passing data_dict through the model
    #
    #         # Calculate loss
    #         loss = criterion(outputs, labels)
    #
    #         print(outputs)
    #
    #         print(loss)
    #
    #         break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
