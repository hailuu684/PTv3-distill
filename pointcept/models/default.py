import torch.nn as nn
import torch_scatter
import torch
from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


# @MODELS.register_module()
# class DefaultSegmentorV2(nn.Module):
#     def __init__(
#         self,
#         num_classes,
#         backbone_out_channels,
#         backbone=None,
#         criteria=None,
#     ):
#         super().__init__()
#         self.seg_head = (
#             nn.Linear(backbone_out_channels, num_classes)
#             if num_classes > 0
#             else nn.Identity()
#         )
#         self.backbone = build_model(backbone)
#         self.criteria = build_criteria(criteria)

#     def forward(self, input_dict):
#         point = Point(input_dict)
#         point = self.backbone(point)
#         # print("point")
#         # print(point["feat"].shape)

#         # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
#         # TODO: remove this part after make all backbone return Point only.
#         if isinstance(point, Point):
#             feat = point.feat
#         else:
#             feat = point
#         print("feat shape:", feat.shape)
#         print("seg_head in_features:", self.seg_head.in_features)
#         print("seg_head out_features:", self.seg_head.out_features)
#         seg_logits = self.seg_head(feat)
#         # print(seg_logits.shape)
#         # train
#         if self.training:
#             loss = self.criteria(seg_logits, input_dict["segment"])
#             return dict(loss=loss)
#         # eval
#         elif "segment" in input_dict.keys():
#             loss = self.criteria(seg_logits, input_dict["segment"])
#             return dict(loss=loss, seg_logits=seg_logits)
#         # test
#         else:
#             return dict(seg_logits=seg_logits)

@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        # print("DefaultSegmentorV2 forward")
        # print(input_dict)
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)




@MODELS.register_module()
class DefaultTopologyDistill(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        teacher_ckpt_path=None,
        teacher_backbone=None,
        student_backbone=None,
        criteria=None,
        kld_distill=None,
        chamfer_distill=None,
        gkd_distill=None,
        freeze_backbone=False,
        freeze_teacher_backbone=True,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.teacher_backbone = build_model(teacher_backbone)
        self.teacher_backbone = self.load_teacher_weights(self.teacher_backbone, teacher_ckpt_path)
        self.student_backbone = build_model(student_backbone)
        self.criteria = build_criteria(criteria)
        self.kld_distill = build_criteria(kld_distill)
        self.chamfer_distill = build_criteria(chamfer_distill)
        self.gkd_distill = build_criteria(gkd_distill)
        self.freeze_backbone = freeze_backbone
        self.freeze_teacher_backbone = freeze_teacher_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if self.freeze_teacher_backbone:
            print("Freeze teacher model's parameters")
            for p in self.teacher_backbone.parameters():
                p.requires_grad = False
            self.teacher_backbone.eval()

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        ## make sure teacher model will not change
        self.teacher_backbone.eval()

        point_tea, latent_feature_tea = self.teacher_backbone(point)
        point_stu, latent_feature_stu = self.student_backbone(point)


        if isinstance(point_tea, Point):
            while "pooling_parent" in point_tea.keys():
                assert "pooling_inverse" in point_tea.keys()
                parent_tea = point_tea.pop("pooling_parent")
                inverse_tea = point_tea.pop("pooling_inverse")
                parent_tea.feat = torch.cat([parent_tea.feat, point_tea.feat[inverse]], dim=-1)
                point_tea = parent_tea
            feat_tea = point_tea.feat
        else:
            feat_tea = point_tea
        seg_logits_tea = self.seg_head(feat_tea)

        if isinstance(point_stu, Point):
            while "pooling_parent" in point_stu.keys():
                assert "pooling_inverse" in point_stu.keys()
                parent_stu = point_stu.pop("pooling_parent")
                inverse_stu = point_stu.pop("pooling_inverse")
                parent_stu.feat = torch.cat([parent_stu.feat, point_stu.feat[inverse]], dim=-1)
                point_stu = parent_stu
            feat_stu = point_stu.feat
        else:
            feat_stu = point_stu
        seg_logits_stu = self.seg_head(feat_stu)


        return_dict = dict()
        
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point_stu
        # train
        if self.training:
            student_loss = self.criteria(seg_logits_stu, input_dict["segment"])
            return_dict["stu_loss"] = student_loss
            total_loss = student_loss
            print("print student loss: ", student_loss)
            teacher_loss = self.criteria(seg_logits_stu, input_dict["segment"]) + torch.mean(latent_feature_tea ** 2)
            if self.kld_distill is not None:
                kld_distill_loss = self.kld_distill(seg_logits_stu,seg_logits_tea)
                return_dict["kld_distill_loss"] = kld_distill_loss
                total_loss = total_loss + kld_distill_loss
            if self.chamfer_distill is not None:
                chamfer_distill_loss = self.chamfer_distill(latent_feature_stu,latent_feature_tea)
                return_dict["chamfer_distill_loss"] = chamfer_distill_loss
                total_loss = total_loss + chamfer_distill_loss
            if self.gkd_distill is not None:
                teacher_loss = self.criteria(seg_logits_stu, input_dict["segment"]) + torch.mean(latent_feature_tea ** 2)
                return_dict["tea_loss"] = teacher_loss
                gkd_distill_loss = self.gkd_distill(student_loss, latent_feature_stu, teacher_loss, latent_feature_tea)
                return_dict["gkd_distill_loss"] = gkd_distill_loss
                total_loss = total_loss + gkd_distill_loss
            return_dict['loss'] = total_loss

        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits_stu, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits_stu
        # test
        else:
            return_dict["seg_logits"] = seg_logits_stu
        return return_dict

    def train(self, mode=True):
        ## rewrite the defalut train function to make sure that the teacher model will not update during the student training process.
        super().train(mode)
        if hasattr(self, "teacher_backbone") and self.teacher_backbone is not None:
            print('teacher model has been freezed and set to eval()')
            self.teacher_backbone.eval()  # freeze teacher model again. 
        return self

    # def load_teacher_weights(self, teacher_backbone, teacher_ckpt_path):
    #     device = next(teacher_backbone.parameters()).device
    #     print(f"Loading teacher weights to device: {device}")
    #     checkpoint = torch.load(teacher_ckpt_path, map_location=device)
        
    #     if "state_dict" in checkpoint:
    #         state_dict = checkpoint["state_dict"]
    #     else:
    #         state_dict = checkpoint

    #     try:
    #         teacher_backbone.load_state_dict(state_dict, strict=False)
    #         print(f"Loaded teacher weights from {teacher_ckpt_path} successfully.")
    #     except RuntimeError as e:
    #         print(f"Warning: Some keys do not match exactly. Error:\n{e}")
    #         teacher_backbone.load_state_dict(state_dict, strict=False)
        
    #     return teacher_backbone
    def load_teacher_weights(self, teacher_backbone, teacher_ckpt_path):
        device = next(teacher_backbone.parameters()).device
        print(f"Loading teacher weights to device: {device}")
        
        checkpoint = torch.load(teacher_ckpt_path, map_location=device)

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # 直接 strict=True，不捕捉异常，加载失败直接爆出错误
        teacher_backbone.load_state_dict(state_dict, strict=True)
        print(f"Loaded teacher weights from {teacher_ckpt_path} successfully with strict=True.")

        return teacher_backbone

