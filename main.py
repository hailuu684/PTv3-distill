import torch
from dataloader import PTv3_Dataloader
from PTv3_model import PointTransformerV3, load_weights_ptv3_nucscenes_seg
from pointcept.engines.defaults import default_config_parser, default_setup
from pointcept.models.losses import build_criteria
import os

# Pretrained model path and config file
PRETRAINED_PATH = '/home/thomle/PTv3-distill/huggingface_model/PointTransformerV3/nuscenes-semseg-pt-v3m1-0-base/model/model_best.pth'
CONFIG_FILE = "configs/nuscenes/semseg-pt-v3m1-0-base.py"


def main():
    # Load configuration
    cfg = default_config_parser(CONFIG_FILE, None)
    cfg = default_setup(cfg)

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
    load_weights_ptv3_nucscenes_seg(model, PRETRAINED_PATH)

    # Data load
    loader = PTv3_Dataloader(cfg)
    train_loader = loader.load_training_data()

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss setup
    criteria = build_criteria(cfg.model.criteria)

    # Define optimizer with parameter groups (if needed)
    # if hasattr(model, "backbone") and hasattr(model, "head"):
    #     optimizer = torch.optim.AdamW([
    #         {"params": model.backbone.parameters(), "lr": 0.0002},
    #         {"params": model.head.parameters(), "lr": 0.002},
    #     ], weight_decay=cfg.optimizer.weight_decay)
    #     max_lr = cfg.scheduler.max_lr  # List of max_lr values for parameter groups
    # else:
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    #     max_lr = cfg.scheduler.max_lr[0]  # Single value for max_lr

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

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_ndx, input_dict in enumerate(train_loader):
            # Move input data to device
            input_dict = {k: v.to(device) for k, v in input_dict.items()}

            # Forward pass
            seg_logits = model(input_dict)

            # Extract the logits tensor for loss calculation
            # if isinstance(seg_logits, dict) or hasattr(seg_logits, 'keys'):
            #     logits_tensor = seg_logits.get("feat", None)  # Use the "feat" key
            #     if logits_tensor is None:
            #         raise KeyError("The key 'feat' is not present in seg_logits. Check your model's output structure.")
            # elif isinstance(seg_logits, torch.Tensor):  # If it's already a tensor
            #     logits_tensor = seg_logits
            # else:
            #     raise TypeError(f"Unexpected type for seg_logits: {type(seg_logits)}. Expected dict-like or torch.Tensor.")

            logits_tensor = seg_logits.get("feat", None)

            # Compute loss
            loss = criteria(logits_tensor, input_dict["segment"])

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Print progress every 10 batches
            if batch_ndx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_ndx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Step the scheduler
        scheduler.step()

        # Print epoch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(train_loader):.4f}")


if __name__ == "__main__":
    main()
