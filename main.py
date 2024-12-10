import torch
from torch.utils.tensorboard import SummaryWriter
from dataloader import PTv3_Dataloader
from PTv3_model import PointTransformerV3, load_weights_ptv3_nucscenes_seg
from pointcept.engines.defaults import default_config_parser, default_setup
from pointcept.models.losses import build_criteria
import os

# from pytorch3d.ops import box3d_overlap

# Pretrained model path and config file
# PRETRAINED_PATH = '/home/thomle/PTv3-distill/huggingface_model/PointTransformerV3/nuscenes-semseg-pt-v3m1-0-base/model/model_best.pth'
PRETRAINED_PATH = './checkpoints/checkpoint_epoch_50_backup.pth'
CONFIG_FILE = "configs/nuscenes/semseg-pt-v3m1-0-base.py"

def student_models():

    student_model = PointTransformerV3(
        in_channels=4,
        pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
        cls_mode=False,
        pdnorm_bn=False,
        mlp_ratio=2,  # Reduced from 4 to 2
        qkv_bias=True,
        enable_flash=False,
        order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
        stride=(2, 2, 2, 2),
        enc_depths=(1, 1, 1, 3, 1),  # Reduced depth from (2, 2, 2, 6, 2)
        enc_channels=(16, 32, 64, 128, 256),  # Reduced number of channels
        enc_num_head=(1, 2, 4, 8, 16),  # Reduced number of attention heads
        enc_patch_size=(512, 512, 512, 512, 512),  # Smaller patch sizes
        dec_depths=(1, 1, 1, 1),  # Reduced decoder depth
        dec_channels=(32, 32, 64, 128),  # Reduced decoder channels
        dec_num_head=(2, 2, 4, 8),  # Reduced decoder attention heads
        dec_patch_size=(512, 512, 512, 512),  # Smaller patch sizes
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.1,  # Reduced drop path rate
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

    return student_model


def main():
    # Load configuration
    cfg = default_config_parser(CONFIG_FILE, None)
    cfg = default_setup(cfg)

    # Initialize TensorBoard
    writer = SummaryWriter(log_dir='logs/ptv3_training')

    # Load the model
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
    # model = student_models()
    # model = load_weights_ptv3_nucscenes_seg(student_model, PRETRAINED_PATH)
    model = load_weights_ptv3_nucscenes_seg(model, PRETRAINED_PATH)

    # Data load
    loader = PTv3_Dataloader(cfg)
    train_loader = loader.load_training_data()

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
    model.train()
    num_epochs = cfg.epoch
    os.makedirs('checkpoints', exist_ok=True)  # Ensure checkpoint directory exists

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_ndx, input_dict in enumerate(train_loader):
            # Move input data to device
            input_dict = {k: v.to(device) for k, v in input_dict.items()}

            # Forward pass
            seg_logits = model(input_dict)
            # logits_tensor = seg_logits.get("feat", None)
            logits_tensor = seg_logits

            # Compute loss
            loss = criteria(logits_tensor, input_dict["segment"])

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # box3d_overlap
            print(logits_tensor.shape, input_dict["segment"].shape)
            # intersection_vol, iou_3d = box3d_overlap(logits_tensor, input_dict["segment"])
            # print(intersection_vol, iou_3d)

            # Log loss to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_ndx)

            # Print progress every 10 batches
            if batch_ndx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_ndx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Step the scheduler
        scheduler.step()

        # Print epoch loss and log to TensorBoard
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/epoch_avg', avg_loss, epoch + 1)

        # Save checkpoints
        if (epoch + 1) % cfg.eval_epoch == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join('checkpoints', f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

    writer.close()

if __name__ == "__main__":
    main()
