import torch
# from torch.utils.tensorboard import SummaryWriter
from dataloader import PTv3_Dataloader
from PTv3_model import PointTransformerV3TrainTeacher, load_weights_ptv3_nucscenes_seg
from pointcept.engines.defaults import default_config_parser, default_setup
from pointcept.models.losses import build_criteria
import os
from gpu_main import compute_miou, compute_iou_all_classes
from pointcept.engines import test
import numpy as np
import subprocess
from loguru import logger as custom_logger
import sys
from fvcore.nn import FlopCountAnalysis
from torch.profiler import profile, record_function, ProfilerActivity


# Pretrained model path and config file
PRETRAINED_PATH = './checkpoints/model_last.pth'
# PRETRAINED_PATH = './checkpoints/checkpoint_epoch_1.pth'
CONFIG_FILE = "configs/nuscenes/semseg-pt-v3m1-0-train-teacher.py"

# Remove default sink (optional)
custom_logger.remove()

# # Save the logs in loguru INFO
# log_path = "./logs/inference_log_student.txt"
# os.makedirs(os.path.dirname(log_path), exist_ok=True)
#
# # Add a file sink and a console sink
# custom_logger.add(log_path, mode="w", rotation="10 MB", enqueue=True, encoding="utf-8",
#                   format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}")
# custom_logger.add(sys.stderr, format="{time} | {level} | {message}")


def get_teacher_model(cfg):
    """
    Load the teacher model and load the pretrained weights.

    Returns:
        PointTransformerV3: The teacher model.
    """
    model_config = cfg.model.backbone
    return PointTransformerV3TrainTeacher(
        in_channels=model_config.in_channels,
        pdnorm_conditions=model_config.pdnorm_conditions,
        cls_mode=model_config.cls_mode,
        pdnorm_bn=model_config.pdnorm_bn,
        mlp_ratio=model_config.mlp_ratio,
        qkv_bias=model_config.qkv_bias,
        enable_flash=model_config.enable_flash,
        order=model_config.order,
        stride=model_config.stride,
        enc_depths=model_config.enc_depths,
        enc_channels=model_config.enc_channels,
        enc_num_head=model_config.enc_num_head,
        enc_patch_size=model_config.enc_patch_size,
        dec_depths=model_config.dec_depths,
        dec_channels=model_config.dec_channels,
        dec_num_head=model_config.dec_num_head,
        dec_patch_size=model_config.dec_patch_size,
        qk_scale=model_config.qk_scale,
        attn_drop=model_config.attn_drop,
        proj_drop=model_config.proj_drop,
        drop_path=model_config.drop_path,
        shuffle_orders=model_config.shuffle_orders,
        pre_norm=model_config.pre_norm,
        enable_rpe=model_config.enable_rpe,
        upcast_attention=model_config.upcast_attention,
        upcast_softmax=model_config.upcast_softmax,
        pdnorm_ln=model_config.pdnorm_ln,
        pdnorm_decouple=model_config.pdnorm_decouple,
        pdnorm_adaptive=model_config.pdnorm_adaptive,
        pdnorm_affine=model_config.pdnorm_affine
    )


def get_student_model(cfg):
    """
    Create a student model with the same architecture as the teacher model.

    Returns:
        PointTransformerV3: The student model.
    """
    model_config = cfg.model.backbone
    return PointTransformerV3TrainTeacher(
        in_channels=model_config.in_channels,
        pdnorm_conditions=model_config.pdnorm_conditions,
        cls_mode=model_config.cls_mode,
        pdnorm_bn=model_config.pdnorm_bn,
        mlp_ratio=model_config.mlp_ratio,
        qkv_bias=model_config.qkv_bias,
        enable_flash=model_config.enable_flash,
        order=model_config.order,
        stride=model_config.stride,
        enc_depths=model_config.enc_depths,  # Reduced depth
        enc_channels=model_config.enc_channels,  # Reduced channels
        enc_num_head=model_config.enc_num_head,  # Reduced heads
        enc_patch_size=model_config.enc_patch_size,  # Smaller patches
        dec_depths=model_config.dec_depths,  # Reduced decoder depth
        dec_channels=model_config.dec_channels,  # Reduced decoder channels
        dec_num_head=model_config.dec_num_head,  # Reduced decoder heads
        dec_patch_size=model_config.dec_patch_size,  # Smaller decoder patches
        qk_scale=model_config.qk_scale,
        attn_drop=model_config.attn_drop,
        proj_drop=model_config.proj_drop,
        drop_path=model_config.drop_path,
        shuffle_orders=model_config.shuffle_orders,
        pre_norm=model_config.pre_norm,
        enable_rpe=model_config.enable_rpe,
        upcast_attention=model_config.upcast_attention,
        upcast_softmax=model_config.upcast_softmax,
        pdnorm_ln=model_config.pdnorm_ln,
        pdnorm_decouple=model_config.pdnorm_decouple,
        pdnorm_adaptive=model_config.pdnorm_adaptive,
        pdnorm_affine=model_config.pdnorm_affine
    )


def train_student(log_file="test_one_frame_student.txt"):
    CONFIG_FILE_STUDENT = "configs/nuscenes/semseg-pt-v3m1-0-train-student.py"
    PRETRAINED_PATH_STUDENT = './checkpoints/checkpoint_batch_40001.pth'
    # Load configuration
    cfg = default_config_parser(CONFIG_FILE_STUDENT, None)
    cfg = default_setup(cfg)

    student_model = get_student_model(cfg)

    student_model = load_weights_ptv3_nucscenes_seg(student_model, PRETRAINED_PATH_STUDENT)

    # Data load
    loader = PTv3_Dataloader(cfg)
    train_loader = loader.load_training_data()

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = student_model.to(device)

    count_parameters(model)

    # Loss setup
    criteria = build_criteria(cfg.model.criteria)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    max_lr = cfg.scheduler.max_lr[0]  # Single value for max_lr

    # Scheduler setup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        pct_start=cfg.scheduler.pct_start,
        anneal_strategy=cfg.scheduler.anneal_strategy,
        div_factor=cfg.scheduler.div_factor,
        final_div_factor=cfg.scheduler.final_div_factor,
        total_steps=len(train_loader) * cfg.epoch,
    )

    # Training loop
    num_epochs = cfg.epoch
    os.makedirs('checkpoints', exist_ok=True)  # Ensure checkpoint directory exists

    num_classes = len(cfg.names)  # Assuming cfg.names contains class names

    tester = test.CustomSemSegTester(cfg=cfg, model=model)
    tester.test_one_frame()

    # for epoch in range(num_epochs):
    #     running_loss = 0.0
    #
    #     total_train_iou_scores = torch.zeros(num_classes, device=device)
    #
    #     # train
    #     model.train()
    #
    #     print("-----> Training")
    #     for batch_ndx, input_dict in enumerate(train_loader):
    #
    #         if batch_ndx == 1:
    #             print(">>>>>>>>>> Interrupted by setting stop batch == 1 <<<<<<<<<<<<")
    #             break
    #
    #         if batch_ndx > 40000:
    #             print("-------> Enough training, now evaluation")
    #             break
    #
    #         # Move input data to device
    #         input_dict = {k: v.to(device) for k, v in input_dict.items()}
    #         # coord_shape = input_dict['coord'].shape
    #         # segment_shape = input_dict['segment'].shape
    #         # feat_shape = input_dict['feat'].shape
    #         # with open("./debug/input_dict_train.txt", "w") as f:
    #         #     f.write("input_dict:\n")
    #         #     f.write(str(input_dict))
    #         #     f.write("\n\nShapes:\n")
    #         #     f.write(f"coord shape: {coord_shape}\n")
    #         #     f.write(f"segment shape: {segment_shape}\n")
    #         #     f.write(f"feat shape: {feat_shape}\n")
    #         # print("Done")
    #
    #         # Forward pass
    #         seg_logits = model(input_dict)
    #         # logits_tensor = seg_logits.get("feat", None)
    #         logits_tensor = seg_logits
    #
    #         # -----------------Start Profiling -----------------/
    #
    #         # Time in each layer
    #         activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
    #         sort_by_keyword = str(device) + "_time_total"
    #
    #         with profile(activities=activities, record_shapes=True) as prof:
    #             with record_function("model_inference"):
    #                 model(input_dict)
    #
    #         print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
    #
    #         # Memory in each layer
    #         with profile(activities=[ProfilerActivity.CPU],
    #                      profile_memory=True, record_shapes=True) as prof:
    #             model(input_dict)
    #
    #         print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    #         print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    #
    #         # -----------------End Profiling -----------------/
    #
    #         # Compute loss
    #         loss = criteria(logits_tensor, input_dict["segment"])
    #
    #         # Compute IoU
    #         ground_truth = input_dict["segment"]
    #
    #         with torch.no_grad():
    #             teacher_preds = torch.argmax(seg_logits, dim=1)
    #             teacher_iou_scores = compute_iou_all_classes(
    #                 teacher_preds,
    #                 ground_truth,
    #                 num_classes
    #             )
    #             teacher_miou = compute_miou(teacher_iou_scores)
    #             total_train_iou_scores += teacher_iou_scores
    #
    #         # Backward pass
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         # Accumulate loss
    #         running_loss += loss.item()
    #
    #         # Print progress every 10 batches
    #         if batch_ndx % 30 == 0:
    #             print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_ndx}/{len(train_loader)}], "
    #                   f"Loss: {loss.item():.2f}, IoU: {teacher_miou:.2f}")
    #
    #         # save model at batch:
    #         if batch_ndx % 10000 == 0:
    #             checkpoint_path = os.path.join('checkpoints', f'checkpoint_batch_{batch_ndx + 1}.pth')
    #             torch.save(model.state_dict(), checkpoint_path)
    #             print(f"Model checkpoint saved at batch {batch_ndx} in {checkpoint_path}")
    #
    #     # Step the scheduler
    #     scheduler.step()
    #
    #     # Print epoch loss and log to TensorBoard
    #     avg_loss = running_loss / len(train_loader)
    #     train_miou = compute_miou(total_train_iou_scores / len(train_loader))
    #     print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, mIoU: {train_miou:.4f}")
    #
    #     # Save checkpoints
    #     if (epoch + 1) % cfg.eval_epoch == 0 or (epoch + 1) == num_epochs:
    #         checkpoint_path = os.path.join('checkpoints', f'checkpoint_epoch_{epoch + 1}.pth')
    #         torch.save(model.state_dict(), checkpoint_path)
    #         print(f"Model checkpoint saved at {checkpoint_path}")
    #
    #     # ====================== TEST EVALUATION ======================
    #
    #     print("---------> Testing")
    #
    #     tester = test.CustomSemSegTester(cfg=cfg, model=model)
    #     tester.test()


def count_parameters(model):
    """Calculate total and trainable parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

    return total_params, trainable_params


def main():
    # Load configuration
    cfg = default_config_parser(CONFIG_FILE, None)
    cfg = default_setup(cfg)

    # Initialize TensorBoard
    # writer = SummaryWriter(log_dir='logs/ptv3_training')

    # Load teacher model
    teacher_model = get_teacher_model(cfg)

    # Load pretrained weights
    model = load_weights_ptv3_nucscenes_seg(teacher_model, PRETRAINED_PATH)

    # Data load
    loader = PTv3_Dataloader(cfg)
    train_loader = loader.load_training_data()
    # test_loader = loader.load_validation_data()

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    count_parameters(model)

    # Loss setup
    criteria = build_criteria(cfg.model.criteria)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    max_lr = cfg.scheduler.max_lr[0]  # Single value for max_lr

    # Scheduler setup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        pct_start=cfg.scheduler.pct_start,
        anneal_strategy=cfg.scheduler.anneal_strategy,
        div_factor=cfg.scheduler.div_factor,
        final_div_factor=cfg.scheduler.final_div_factor,
        total_steps=len(train_loader) * cfg.epoch,
    )

    # Training loop
    num_epochs = cfg.epoch
    os.makedirs('checkpoints', exist_ok=True)  # Ensure checkpoint directory exists

    num_classes = len(cfg.names)  # Assuming cfg.names contains class names

    tester = test.CustomSemSegTester(cfg=cfg, model=model)  # , model=model
    test_loader = tester.test_one_frame()

    # for epoch in range(num_epochs):
    #     running_loss = 0.0
    #
    #     total_train_iou_scores = torch.zeros(num_classes, device=device)
    #
    #     # train
    #     model.train()
    #
    #     print("-----> Training")
    #     for batch_ndx, input_dict in enumerate(train_loader):
    #
    #         if batch_ndx == 1:
    #             print(">>>>>>>>>> Interrupted by setting stop batch == 1 <<<<<<<<<<<<")
    #             break
    #
    #         # Move input data to device
    #         input_dict = {k: v.to(device) for k, v in input_dict.items()}
    #
    #         # Forward pass
    #         seg_logits = model(input_dict)
    #         # logits_tensor = seg_logits.get("feat", None)
    #
    #         # # -----------------Start Profiling -----------------/
    #         #
    #         # # Time in each layer
    #         # activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
    #         # sort_by_keyword = str(device) + "_time_total"
    #         #
    #         # with profile(activities=activities, record_shapes=True) as prof:
    #         #     with record_function("model_inference"):
    #         #         model(input_dict)
    #         #
    #         # print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
    #         #
    #         # # Memory in each layer
    #         # with profile(activities=[ProfilerActivity.CPU],
    #         #              profile_memory=True, record_shapes=True) as prof:
    #         #     model(input_dict)
    #         #
    #         # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    #         # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    #         #
    #         # # -----------------End Profiling -----------------/
    #
    #         logits_tensor = seg_logits
    #
    #         # Compute loss
    #         loss = criteria(logits_tensor, input_dict["segment"])
    #
    #         # Compute IoU
    #         ground_truth = input_dict["segment"]
    #
    #         # print(logits_tensor.shape)
    #         # print(ground_truth.shape)
    #         with torch.no_grad():
    #             teacher_preds = torch.argmax(seg_logits, dim=1)
    #             teacher_iou_scores = compute_iou_all_classes(
    #                 teacher_preds,
    #                 ground_truth,
    #                 num_classes
    #             )
    #             teacher_miou = compute_miou(teacher_iou_scores)
    #             total_train_iou_scores += teacher_iou_scores
    #
    #         # Backward pass
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         # Accumulate loss
    #         running_loss += loss.item()
    #
    #         # Log loss to TensorBoard
    #         # writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_ndx)
    #
    #         # Print progress every 10 batches
    #         if batch_ndx % 30 == 0:
    #             print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_ndx}/{len(train_loader)}], "
    #                   f"Loss: {loss.item():.2f}, IoU: {teacher_miou:.2f}")
    #
    #     # Step the scheduler
    #     scheduler.step()
    #
    #     # Print epoch loss and log to TensorBoard
    #     avg_loss = running_loss / len(train_loader)
    #     train_miou = compute_miou(total_train_iou_scores / len(train_loader))
    #     print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, mIoU: {train_miou:.4f}")
    #     # writer.add_scalar('Loss/epoch_avg', avg_loss, epoch + 1)
    #
    #     # Save checkpoints
    #     if (epoch + 1) % cfg.eval_epoch == 0 or (epoch + 1) == num_epochs:
    #         checkpoint_path = os.path.join('checkpoints', f'checkpoint_epoch_{epoch + 1}.pth')
    #         torch.save(model.state_dict(), checkpoint_path)
    #         print(f"Model checkpoint saved at {checkpoint_path}")
    #
    #     # ====================== TEST EVALUATION ======================
    #
    #     print("---------> Testing")
    #
    #     tester = test.CustomSemSegTester(cfg=cfg, model=model)
    #     tester.test()

    # # writer.close()


def calculate_gflops(
        in_channels=4,
        num_points=22764,
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        backbone_out_channels=64,
        num_classes=16,
        stride=(2, 2, 2, 2),
        enable_flash=True,
        model_name="Teacher"
):
    """
    Calculate GFLOPs for PointTransformerV3 model.

    Args:
        in_channels (int): Number of input channels (e.g., 4 for nuScenes).
        num_points (int): Number of input points (e.g., 22764).
        enc_depths (tuple): Encoder block depths per stage.
        enc_channels (tuple): Encoder channels per stage.
        enc_num_head (tuple): Encoder attention heads per stage.
        enc_patch_size (tuple): Encoder patch sizes per stage.
        dec_depths (tuple): Decoder block depths per stage.
        dec_channels (tuple): Decoder channels per stage.
        dec_num_head (tuple): Decoder attention heads per stage.
        dec_patch_size (tuple): Decoder patch sizes per stage.
        backbone_out_channels (int): Output channels of backbone.
        num_classes (int): Number of segmentation classes.
        stride (tuple): Downsampling strides.
        enable_flash (bool): Whether Flash Attention is enabled.
        model_name (str): Name of the model (e.g., "Teacher" or "Student").

    Returns:
        dict: FLOPs breakdown in GFLOPs.
    """
    # Point counts per stage (assuming stride halves points)
    points = [num_points]
    for s in stride:
        points.append(points[-1] // 2)

    # Embedding FLOPs
    embed_spconv = 2 * in_channels * enc_channels[0] * 125 * num_points / 1e9  # 5x5x5 kernel
    embed_linear = 2 * enc_channels[0] * enc_channels[0] * num_points / 1e9
    embedding_gflops = embed_spconv + embed_linear

    # Encoder FLOPs
    enc_attention_total = 0
    enc_mlp_total = 0
    enc_cpe_total = 0
    enc_pooling_total = 0

    for stage, (depth, channels, heads, patch_size) in enumerate(
            zip(enc_depths, enc_channels, enc_num_head, enc_patch_size)):
        p = points[stage]
        # CPE (spconv.SubMConv3d, 3x3x3=27)
        cpe = 2 * channels * channels * 27 * p * depth / 1e9
        enc_cpe_total += cpe

        # Attention
        qkv = 2 * channels * (3 * channels) * p / 1e9  # QKV linear
        proj = 2 * channels * channels * p / 1e9  # Projection linear
        if enable_flash:
            attn = 2 * p * min(patch_size, p) * heads / 3 / 1e9  # Flash Attention approximation
        else:
            attn = 2 * p * min(patch_size, p) * heads / 1e9  # Standard attention
        attention_per_block = (qkv + attn + proj) * depth
        enc_attention_total += attention_per_block

        # MLP (mlp_ratio=4)
        mlp_hidden = channels * 4
        mlp = 2 * (channels * mlp_hidden + mlp_hidden * channels) * p * depth / 1e9
        enc_mlp_total += mlp

        # Pooling (only for stages 1+)
        if stage > 0:
            pooling = 2 * enc_channels[stage - 1] * channels * points[stage - 1] / 1e9
            enc_pooling_total += pooling

    encoder_gflops = enc_cpe_total + enc_attention_total + enc_mlp_total + enc_pooling_total

    # Decoder FLOPs
    dec_attention_total = 0
    dec_mlp_total = 0
    dec_cpe_total = 0
    dec_unpooling_total = 0

    for stage, (depth, channels, heads, patch_size) in enumerate(
            zip(dec_depths, dec_channels, dec_num_head, dec_patch_size)):
        p_in = points[-(stage + 2)]  # Input points (upsampled)
        p_out = points[-(stage + 3)]  # Output points (before upsampling)

        # Unpooling
        unpool_in = 2 * enc_channels[-(stage + 1)] * channels * p_in / 1e9
        unpool_skip = 2 * enc_channels[-(stage + 2)] * channels * p_out / 1e9
        dec_unpooling_total += unpool_in + unpool_skip

        # CPE
        cpe = 2 * channels * channels * 27 * p_out * depth / 1e9
        dec_cpe_total += cpe

        # Attention
        qkv = 2 * channels * (3 * channels) * p_out / 1e9
        proj = 2 * channels * channels * p_out / 1e9
        if enable_flash:
            attn = 2 * p_out * min(patch_size, p_out) * heads / 3 / 1e9
        else:
            attn = 2 * p_out * min(patch_size, p_out) * heads / 1e9
        attention_per_block = (qkv + attn + proj) * depth
        dec_attention_total += attention_per_block

        # MLP
        mlp_hidden = channels * 4
        mlp = 2 * (channels * mlp_hidden + mlp_hidden * channels) * p_out * depth / 1e9
        dec_mlp_total += mlp

    decoder_gflops = dec_unpooling_total + dec_cpe_total + dec_attention_total + dec_mlp_total

    # Segmentation Head FLOPs
    seg_head_gflops = 2 * backbone_out_channels * num_classes * num_points / 1e9

    # Total FLOPs
    total_gflops = embedding_gflops + encoder_gflops + decoder_gflops + seg_head_gflops

    return {
        "Model": model_name,
        "Embedding": round(embedding_gflops, 2),
        "Encoder": round(encoder_gflops, 2),
        "Encoder Attention": round(enc_attention_total, 2),
        "Encoder MLP": round(enc_mlp_total, 2),
        "Decoder": round(decoder_gflops, 2),
        "Segmentation Head": round(seg_head_gflops, 2),
        "Total": round(total_gflops, 2)
    }


if __name__ == "__main__":
    # set up the GPU
    # subprocess.run(["bash", "run_set_up_GPU.sh"], check=True)

    # main()
    train_student()  # Val result: mIoU/mAcc/allAcc 0.7608/0.8255/0.9389
    # python main.py

    """
    teacher params: Total Parameters: 46160080
    Trainable Parameters: 46160080
    GPU used: 4.3 Gb
    
    student params: Total Parameters: 2778112
    Trainable Parameters: 2778112
    GPU used: 4.2 Gb
    """

    # # ------------ calculate config ------------- #
    # teacher_config = {
    #     "in_channels": 4,
    #     "num_points": 22764,
    #     "enc_depths": (2, 2, 2, 6, 2),
    #     "enc_channels": (32, 64, 128, 256, 512),
    #     "enc_num_head": (2, 4, 8, 16, 32),
    #     "enc_patch_size": (1024, 1024, 1024, 1024, 1024),
    #     "dec_depths": (2, 2, 2, 2),
    #     "dec_channels": (64, 64, 128, 256),
    #     "dec_num_head": (4, 4, 8, 16),
    #     "dec_patch_size": (1024, 1024, 1024, 1024),
    #     "backbone_out_channels": 64,
    #     "num_classes": 16,
    #     "stride": (2, 2, 2, 2),
    #     "enable_flash": True,
    #     "model_name": "Teacher"
    # }
    #
    # teacher_gflops = calculate_gflops(**teacher_config)
    # print("Teacher GFLOPs:", teacher_gflops)
    #
    # student_config = {
    #     "in_channels": 4,
    #     "num_points": 22764,
    #     "enc_depths": (1, 1, 1, 2, 1),
    #     "enc_channels": (16, 16, 32, 64, 128),
    #     "enc_num_head": (1, 1, 2, 4, 8),
    #     "enc_patch_size": (1024, 1024, 1024, 1024, 1024),
    #     "dec_depths": (1, 1, 1, 1),
    #     "dec_channels": (64, 64, 128, 128),
    #     "dec_num_head": (2, 2, 4, 8),
    #     "dec_patch_size": (1024, 1024, 1024, 1024),
    #     "backbone_out_channels": 64,
    #     "num_classes": 16,
    #     "stride": (2, 2, 2, 2),
    #     "enable_flash": True,
    #     "model_name": "Student"
    # }
    #
    # student_gflops = calculate_gflops(**student_config)
    # print("Student GFLOPs:", student_gflops)
