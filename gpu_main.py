import torch
from torch.utils.tensorboard import SummaryWriter
from dataloader import PTv3_Dataloader
from PTv3_model import PointTransformerV3, load_weights_ptv3_nucscenes_seg
from pointcept.engines.defaults import default_config_parser, default_setup
from pointcept.models.losses import build_criteria
import os
from pytorch3d.loss import chamfer_distance
import torch.nn.functional as F

# Pretrained model path and config file
PRETRAINED_PATH = './checkpoints/checkpoint_epoch_50_backup.pth'
CONFIG_FILE = "configs/nuscenes/semseg-pt-v3m1-0-base.py"

def compute_iou_all_classes(preds, labels, num_classes):
    """
    Compute IoU for all classes using GPU operations.

    Args:
        preds (torch.Tensor): Predicted labels of shape [N].
        labels (torch.Tensor): Ground truth labels of shape [N].
        num_classes (int): Number of classes.

    Returns:
        torch.Tensor: IoU scores for each class.
    """
    preds = preds.view(-1)
    labels = labels.view(-1)

    iou_scores = torch.zeros(num_classes, device=preds.device)

    for class_id in range(num_classes):
        pred_inds = preds == class_id
        target_inds = labels == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union > 0:
            iou_scores[class_id] = intersection / union
        else:
            iou_scores[class_id] = float('nan')  # Use NaN for classes not present

    return iou_scores

def compute_miou(iou_scores):
    """
    Compute the mean Intersection over Union (mIoU) from the IoU scores.

    Args:
        iou_scores (torch.Tensor): IoU scores for each class.

    Returns:
        torch.Tensor: The mean IoU.
    """
    # Exclude NaN values which represent classes not present
    valid_iou = iou_scores[~iou_scores.isnan()]
    miou = valid_iou.mean() if valid_iou.numel() > 0 else torch.tensor(0.0, device=iou_scores.device)
    return miou

def get_teacher_model(cfg):
    """
    Load the teacher model and load the pretrained weights.

    Returns:
        PointTransformerV3: The teacher model.
    """
    model_config = cfg.model.teacher_backbone
    return PointTransformerV3(
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
    model_config = cfg.model.student_backbone
    return PointTransformerV3(
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

def compute_chamfer_loss(persistence_diagram_1, persistence_diagram_2):
    """
    Compute the Chamfer loss between two persistence diagrams.

    Args:
        persistence_diagram_1 (list of torch.Tensor): Teacher's persistence diagrams.
        persistence_diagram_2 (list of torch.Tensor): Student's persistence diagrams.

    Returns:
        torch.Tensor: The Chamfer loss.
    """
    total_loss = 0.0

    for pd1, pd2 in zip(persistence_diagram_1, persistence_diagram_2):
        # Ensure the tensors are on the same device
        pd1 = pd1.to(persistence_diagram_1[0].device).unsqueeze(0)  # [1, N, D]
        pd2 = pd2.to(persistence_diagram_2[0].device).unsqueeze(0)  # [1, N, D]

        # Compute the Chamfer distance for the current part
        loss_chamfer, _ = chamfer_distance(pd1, pd2)
        total_loss += loss_chamfer

    return total_loss

def main():
    # Load configuration
    cfg = default_config_parser(CONFIG_FILE, None)
    cfg = default_setup(cfg)

    # Initialize TensorBoard
    writer = SummaryWriter(log_dir='logs/ptv3_training')

    # Load teacher model
    teacher_model = get_teacher_model(cfg)

    # Load student model
    student_model = get_student_model(cfg)

    # Load pretrained weights
    teacher_model = load_weights_ptv3_nucscenes_seg(teacher_model, PRETRAINED_PATH)

    # Data load
    loader = PTv3_Dataloader(cfg)
    train_loader = loader.load_training_data()
    

    # Move models to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    # print("load model done")

    # Loss function
    detection_loss_fn = build_criteria(cfg.model.criteria)
    # lambda_param = cfg.model.lambda_param  # Weight for distillation loss
    # print("Loss function done")

    # Optimizer for student model
    student_optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay
    )
    # print("student_optimizer done")
    # It's unclear why there's a teacher_optimizer since the teacher is in eval mode.
    # If you intend to train the teacher, uncomment and modify the following:
    # teacher_optimizer = torch.optim.AdamW(
    #     teacher_model.parameters(),
    #     lr=cfg.optimizer.lr,
    #     weight_decay=cfg.optimizer.weight_decay
    # )

    # Training loop
    teacher_model.eval()  # Freeze the teacher model
    student_model.train()
    print("Training loop done")

    num_epochs = cfg.epoch
    os.makedirs('checkpoints', exist_ok=True)

    num_classes = len(cfg.names)  # Assuming cfg.names contains class names
    
    for epoch in range(num_epochs):
        # print("loop")
        print(enumerate(train_loader))
        for batch_ndx, input_dict in enumerate(train_loader):
            # print("Loadding")
            # Move input data to device
            input_dict = {k: v.to(device, non_blocking=True) for k, v in input_dict.items()}
            # print(input_dict)
            ground_truth = input_dict["segment"]
            # print("Move input data to device done")

            # Forward pass through teacher model
            with torch.no_grad():
                teacher_seg_logits, teacher_latent_feature = teacher_model(input_dict)
                teacher_preds = torch.argmax(teacher_seg_logits, dim=1)
                teacher_iou_scores = compute_iou_all_classes(
                    teacher_preds,
                    ground_truth,
                    num_classes
                )
                teacher_miou = compute_miou(teacher_iou_scores)
            # print("teacher model move done")

            # Forward pass through student model
            student_seg_logits, student_latent_feature = student_model(input_dict)
            student_preds = torch.argmax(student_seg_logits, dim=1)
            student_iou_scores = compute_iou_all_classes(
                student_preds,
                ground_truth,
                num_classes
            )
            student_miou = compute_miou(student_iou_scores)
            # print("teacher model move done")

            # Compute detection loss for the teacher model (optional, since teacher is frozen)
            # If the teacher is frozen, this loss is not used for backpropagation
            teacher_loss = detection_loss_fn(teacher_seg_logits, ground_truth)

            # Compute detection loss for the student model
            student_loss = detection_loss_fn(student_seg_logits, ground_truth)

            # Chamfer loss between teacher and student latent features
            chamfer_loss = compute_chamfer_loss(teacher_latent_feature, student_latent_feature)

            # Combine detection loss and Chamfer loss for the student model
            student_loss_with_chamfer = student_loss + chamfer_loss

            # Backpropagation for student model
            student_optimizer.zero_grad()
            student_loss_with_chamfer.backward()
            student_optimizer.step()

            # Log training loss
            global_step = epoch * len(train_loader) + batch_ndx
            writer.add_scalar('Loss/student', student_loss.item(), global_step)
            writer.add_scalar('Loss/chamfer', chamfer_loss.item(), global_step)
            writer.add_scalar('Loss/teacher', teacher_loss.item(), global_step)
            writer.add_scalar('Metrics/student_mIoU', student_miou.item(), global_step)
            writer.add_scalar('Metrics/teacher_mIoU', teacher_miou.item(), global_step)

            # Print progress
            if batch_ndx % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_ndx}/{len(train_loader)}], "
                    f"Student Loss: {student_loss.item():.4f}, "
                    f"Student mIoU: {student_miou.item():.4f}, "
                    f"Teacher Loss: {teacher_loss.item():.4f}, "
                    f"Teacher mIoU: {teacher_miou.item():.4f}, "
                    f"Chamfer Loss: {chamfer_loss.item():.4f}"
                )

        # Save checkpoints
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join('checkpoints', f'student_checkpoint_epoch_{epoch + 1}.pth')
            torch.save(student_model.state_dict(), checkpoint_path)
            print(f"Student model checkpoint saved at {checkpoint_path}")

    writer.close()

if __name__ == "__main__":
    main()
