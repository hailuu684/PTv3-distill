import torch
from torch.utils.tensorboard import SummaryWriter
from dataloader import PTv3_Dataloader
from PTv3_model import PointTransformerV3, load_weights_ptv3_nucscenes_seg
from pointcept.engines.defaults import default_config_parser, default_setup
from pointcept.models.losses import build_criteria
import os
from pytorch3d.loss import chamfer_distance
import numpy as np
from sklearn.metrics import confusion_matrix

# from pytorch3d.ops import box3d_overlap

# Pretrained model path and config file
# PRETRAINED_PATH = '/home/thomle/PTv3-distill/huggingface_model/PointTransformerV3/nuscenes-semseg-pt-v3m1-0-base/model/model_best.pth'
PRETRAINED_PATH = './checkpoints/checkpoint_epoch_50_backup.pth'
CONFIG_FILE = "configs/nuscenes/semseg-pt-v3m1-0-base.py"

def compute_iou_all_classes(preds, labels, class_names):
    """
    Compute IoU for all classes.

    Args:
        preds (torch.Tensor): Predicted labels of shape [N].
        labels (torch.Tensor): Ground truth labels of shape [N].
        class_names (list): List of class names.

    Returns:
        dict: Dictionary with IoU scores for each class.
    """
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    iou_scores = {}
    for class_id, class_name in enumerate(class_names):
        # Calculate true positive, false positive, and false negative
        tp = np.sum((preds == class_id) & (labels == class_id))
        fp = np.sum((preds == class_id) & (labels != class_id))
        fn = np.sum((preds != class_id) & (labels == class_id))

        # Calculate IoU
        union = tp + fp + fn
        iou = tp / union if union > 0 else 0.0
        iou_scores[class_name] = iou

    return iou_scores

def compute_miou(iou_scores):
    """
    Compute the mean Intersection over Union (mIoU) from the IoU scores.

    Args:
        iou_scores (dict): Dictionary of IoU scores for each class.

    Returns:
        float: The mean IoU.
    """
    # Filter out IoUs that are zero (for classes that were not present in the batch)
    valid_iou_scores = [iou for iou in iou_scores.values() if iou > 0.0]
    
    # Calculate the mean of valid IoU scores
    miou = np.mean(valid_iou_scores) if valid_iou_scores else 0.0
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
        enc_depths=model_config.enc_depths,  # Reduced depth from (2, 2, 2, 6, 2)
        enc_channels=model_config.enc_channels,  # Reduced number of channels
        enc_num_head=model_config.enc_num_head,  # Reduced number of attention heads
        enc_patch_size=model_config.enc_patch_size,  # Smaller patch sizes
        dec_depths=model_config.dec_depths,  # Reduced decoder depth
        dec_channels=model_config.dec_channels,  # Reduced decoder channels
        dec_num_head=model_config.dec_num_head,  # Reduced decoder attention heads
        dec_patch_size=model_config.dec_patch_size,  # Smaller patch sizes
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
    
    
def compute_gradients(loss, feature_maps):
    """
    Compute the gradients of the loss with respect to feature maps.

    Args:
        loss (torch.Tensor): Loss value.
        feature_maps (list[torch.Tensor]): List of feature maps.

    Returns:
        list[torch.Tensor]: List of gradients for each feature map.
    """
    gradients = torch.autograd.grad(
        outputs=loss,
        inputs=feature_maps,
        grad_outputs=torch.ones_like(loss),
        create_graph=True,
        retain_graph=True
    )
    return gradients


def global_average_pooling(tensor):
    """
    Perform global average pooling on a tensor.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Globally averaged tensor.
    """
    return tensor.mean(dim=(2, 3), keepdim=True)  # Assuming 4D tensor


def compute_distillation_loss(teacher_features, student_features):
    """
    Compute distillation loss between teacher's and student's feature maps.

    Args:
        teacher_features (list[torch.Tensor]): Teacher's feature maps.
        student_features (list[torch.Tensor]): Student's feature maps.

    Returns:
        torch.Tensor: Distillation loss.
    """
    distillation_loss = 0.0
    for teacher, student in zip(teacher_features, student_features):
        distillation_loss += torch.nn.functional.mse_loss(teacher, student)
    return distillation_loss
    
def compute_chamfer_loss(persistence_diagram_1, persistence_diagram_2):
    """
    Compute the Chamfer loss between two persistence diagrams.

    Args:
        persistence_diagram_1 (list of torch.Tensor): A list of 3 tensors representing the persistence diagram.
        persistence_diagram_2 (list of torch.Tensor): A list of 3 tensors representing the persistence diagram.

    Returns:
        torch.Tensor: The Chamfer loss between the two persistence diagrams.
    """
    total_loss = 0.0

    # Iterate through each tensor (part of the persistence diagram)
    for pd1, pd2 in zip(persistence_diagram_1, persistence_diagram_2):
        # Chamfer distance takes point clouds, so ensure they're in the correct shape (B, N, D)
        # where B is the batch size, N is the number of points, and D is the dimension (2 for birth-death pairs)
        
        pd1 = pd1.unsqueeze(0)  # Shape: [1, N, 2]
        # print(pd1.shape)
        pd2 = pd2.unsqueeze(0)  # Shape: [1, N, 2]
        # print(pd2.shape)

        # Compute the Chamfer distance for the current part
        loss_chamfer, _ = chamfer_distance(pd1, pd2)
        
        # Add the result to the total loss
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
    # model = load_weights_ptv3_nucscenes_seg(student_model, PRETRAINED_PATH)
    teacher_model = load_weights_ptv3_nucscenes_seg(teacher_model, PRETRAINED_PATH)

    # Data load
    loader = PTv3_Dataloader(cfg)
    train_loader = loader.load_training_data()

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    # Loss function
    detection_loss_fn = build_criteria(cfg.model.criteria)
    # lambda_param = cfg.model.lambda_param  # Weight for distillation loss

    # Optimizer for student model
    student_optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay
    )
    # Optimizer for student model
    teacher_optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay
    )


    # Training loop
    teacher_model.eval()  # Freeze the teacher model
    student_model.train()

    num_epochs = cfg.epoch
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(num_epochs):
        for batch_ndx, input_dict in enumerate(train_loader):
            # Move input data to device
            input_dict = {k: v.to(device) for k, v in input_dict.items()}
            ground_truth = input_dict["segment"]

            # Forward pass through teacher model
            # print("Teacher model forward pass")
            with torch.no_grad():
                teacher_seg_logits, teacher_latent_feature = teacher_model(input_dict)
                teacher_iou_scores = compute_iou_all_classes(
                    torch.argmax(teacher_seg_logits, dim=1),
                    input_dict["segment"],
                    cfg.names
                    )
                teacher_miou = compute_miou(teacher_iou_scores)
                # teacher_iou_line = ", ".join([f"{class_name}: {iou:.4f}" for class_name, iou in teacher_iou_scores.items() if iou > 0.0])
                # print(f"Teacher IoU Scores: {teacher_iou_line}")
                
            
            # print(teacher_latent_feature)

            # Forward pass through student model
            # print("Student model forward pass")
            student_seg_logits, student_latent_feature = student_model(input_dict)
            student_iou_scores = compute_iou_all_classes(
                torch.argmax(student_seg_logits, dim=1),
                input_dict["segment"],
                cfg.names
                )
            student_miou = compute_miou(student_iou_scores)
            # student_iou_line = ", ".join([f"{class_name}: {iou:.4f}" for class_name, iou in student_iou_scores.items() if iou > 0.0])
            # print(f"Student IoU Scores: {student_iou_line}")

            # Compute detection loss for the teacher model
            teacher_loss = detection_loss_fn(teacher_seg_logits, ground_truth)
            # print('Detection loss for teacher model:', teacher_loss.item())

            # Chamfer loss
            chamfer_loss = compute_chamfer_loss(teacher_latent_feature, student_latent_feature)
            # print('Chamfer loss:', chamfer_loss)


            # Calculate gradients for the teacher's feature maps
            # gradients = compute_gradients(teacher_loss)

            # Compute feature importance weights
            # feature_importance_weights = [global_average_pooling(abs(grad)) for grad in gradients]

            # Apply gradient-based weighting to the student's feature maps
            # weighted_student_feature_maps = [
            #     student_map * weight
            #     for student_map, weight in zip(student_feature_maps, feature_importance_weights)
            # ]

            # Compute distillation loss
            # distillation_loss = compute_distillation_loss(teacher_feature_maps, weighted_student_feature_maps)

            # Combine detection loss and distillation loss for the student model
            student_loss = detection_loss_fn(student_seg_logits, ground_truth)
            student_loss_with_chamfer = student_loss + chamfer_loss

            # Backpropagation for student model
            student_optimizer.zero_grad()
            student_loss_with_chamfer.backward()
            student_optimizer.step()

            # Backpropagation for teacher model
            # teacher_optimizer.zero_grad()
            # teacher_loss.backward()
            # teacher_optimizer.step()

            # Log training loss
            writer.add_scalar('Loss/student', student_loss.item(), epoch * len(train_loader) + batch_ndx)
            writer.add_scalar('Loss/chamfer', chamfer_loss, epoch * len(train_loader) + batch_ndx)
            writer.add_scalar('Loss/teacher', teacher_loss.item(), epoch * len(train_loader) + batch_ndx)


            # Print progress
            if batch_ndx % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_ndx}/{len(train_loader)}], Student Loss: {student_loss.item():.4f}, Student mIoU: {student_miou:.4f}, Teacher Loss: {teacher_loss.item():.4f}, Teacher mIoU: {teacher_miou:.4f}, Chamfer Loss: {chamfer_loss:.4f}")

        # Save checkpoints
        if (epoch + 1) % cfg.eval_epoch == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join('checkpoints', f'student_checkpoint_epoch_{epoch + 1}.pth')
            torch.save(student_model.state_dict(), checkpoint_path)
            print(f"Student model checkpoint saved at {checkpoint_path}")

    writer.close()

if __name__ == "__main__":
    main()
