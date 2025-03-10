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

# Pretrained model path and config file
# PRETRAINED_PATH = './checkpoints/model_best.pth'
PRETRAINED_PATH = './checkpoints/checkpoint_epoch_1.pth'
CONFIG_FILE = "configs/nuscenes/semseg-pt-v3m1-0-train-teacher.py"


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

    # tester = test.CustomSemSegTester(cfg=cfg, model=model)
    # test_loader = tester.test()

    for epoch in range(num_epochs):
        running_loss = 0.0

        total_train_iou_scores = torch.zeros(num_classes, device=device)

        # train
        model.train()

        print("-----> Training")
        for batch_ndx, input_dict in enumerate(train_loader):
            # Move input data to device
            input_dict = {k: v.to(device) for k, v in input_dict.items()}

            # Forward pass
            seg_logits = model(input_dict)
            # logits_tensor = seg_logits.get("feat", None)
            logits_tensor = seg_logits

            # Compute loss
            loss = criteria(logits_tensor, input_dict["segment"])

            # Compute IoU
            ground_truth = input_dict["segment"]

            # print(logits_tensor.shape)
            # print(ground_truth.shape)
            with torch.no_grad():
                teacher_preds = torch.argmax(seg_logits, dim=1)
                teacher_iou_scores = compute_iou_all_classes(
                    teacher_preds,
                    ground_truth,
                    num_classes
                )
                # teacher_miou = compute_miou(teacher_iou_scores)
                total_train_iou_scores += teacher_iou_scores

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Log loss to TensorBoard
            # writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_ndx)

            # Print progress every 10 batches
            if batch_ndx % 30 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_ndx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Step the scheduler
        scheduler.step()

        # Print epoch loss and log to TensorBoard
        avg_loss = running_loss / len(train_loader)
        train_miou = compute_miou(total_train_iou_scores / len(train_loader))
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, mIoU: {train_miou:.4f}")
        # writer.add_scalar('Loss/epoch_avg', avg_loss, epoch + 1)

        # Save checkpoints
        if (epoch + 1) % cfg.eval_epoch == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join('checkpoints', f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

        # ====================== TEST EVALUATION ======================

        print("---------> Testing")

        tester = test.CustomSemSegTester(cfg=cfg, model=model)
        tester.test()


    # # writer.close()

if __name__ == "__main__":
    main()

    # python main.py

