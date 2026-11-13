"""Single-modality base learners: RETFound image model, demographic MLP, biomarker MLP."""

from functools import partial

import timm.models.vision_transformer as timm_vit
import torch
import torch.nn as nn

try:
    from timm.models.layers import trunc_normal_
except ImportError:
    from timm.layers import trunc_normal_

import config
import utils


class RETFoundViT(timm_vit.VisionTransformer):
    """ViT-Large/16 with global average pooling over patch tokens (RETFound)."""

    def __init__(self, global_pool: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = kwargs["norm_layer"](kwargs["embed_dim"])
            del self.norm

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(b, -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        if self.global_pool:
            return self.fc_norm(x[:, 1:, :].mean(dim=1))
        return self.norm(x)[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def build_retfound_backbone(img_size: int = 224) -> RETFoundViT:
    return RETFoundViT(
        img_size=img_size,
        patch_size=config.RETFOUND_PATCH_SIZE,
        embed_dim=config.RETFOUND_EMBED_DIM,
        depth=config.RETFOUND_DEPTH,
        num_heads=config.RETFOUND_NUM_HEADS,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        global_pool=True,
        num_classes=0,
    )


class DemographicEncoder(nn.Module):
    """MLP over [age, gender] producing a 16-d embedding."""

    def __init__(self, input_dim: int = config.NUM_DEMOGRAPHIC_FEATURES,
                 embed_dim: int = config.DEMOGRAPHIC_EMBED_DIM, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(8, embed_dim)
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.output_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BiomarkerEncoder(nn.Module):
    """MLP over the standardized biomarker vector."""

    def __init__(self, input_dim: int, hidden_dims: list, embed_dim: int, dropout: float = 0.3):
        super().__init__()
        layers = [nn.BatchNorm1d(input_dim)]
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, embed_dim))
        self.net = nn.Sequential(*layers)
        self.output_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TriModalCIMTModel(nn.Module):
    """One base learner: encoder for a single modality followed by a two-layer classifier.

    mode is "image_only", "demographic_only" or "biomarker_only". The three
    probabilities are combined later in fusion.py.
    """

    VALID_MODES = ("image_only", "demographic_only", "biomarker_only")

    def __init__(self, num_biomarkers: int = config.NUM_BIOMARKERS,
                 num_classes: int = config.NUM_CLASSES, mode: str = config.MODEL_MODE,
                 retfound_checkpoint=None, freeze_backbone: bool = config.FREEZE_BACKBONE):
        super().__init__()
        assert mode in self.VALID_MODES, f"Unknown mode {mode}; expected one of {self.VALID_MODES}"
        self.mode = mode

        self.image_backbone = None
        if mode == "image_only":
            self.image_backbone = build_retfound_backbone(img_size=config.IMG_SIZE)
            if retfound_checkpoint is not None:
                utils.load_retfound_backbone(self.image_backbone, retfound_checkpoint)
            if freeze_backbone:
                for p in self.image_backbone.parameters():
                    p.requires_grad = False

        self.demographic_encoder = DemographicEncoder() if mode == "demographic_only" else None
        self.biomarker_encoder = (
            BiomarkerEncoder(num_biomarkers, config.BIOMARKER_HIDDEN_DIMS,
                             config.BIOMARKER_EMBED_DIM, config.DROPOUT)
            if mode == "biomarker_only" else None
        )

        head_input_dim = {
            "image_only": config.RETFOUND_EMBED_DIM,
            "demographic_only": config.DEMOGRAPHIC_EMBED_DIM,
            "biomarker_only": config.BIOMARKER_EMBED_DIM,
        }[mode]

        self.classifier = nn.Sequential(
            nn.Linear(head_input_dim, config.FUSION_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FUSION_HIDDEN_DIM, num_classes),
        )
        trunc_normal_(self.classifier[-1].weight, std=2e-5)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, image=None, demographics=None, biomarkers=None):
        if self.mode == "image_only":
            feat = self.image_backbone.forward_features(image)
        elif self.mode == "demographic_only":
            feat = self.demographic_encoder(demographics)
        else:
            feat = self.biomarker_encoder(biomarkers)
        return self.classifier(feat)


def build_model(mode: str = None, load_pretrained: bool = True) -> TriModalCIMTModel:
    mode = mode or config.MODEL_MODE
    checkpoint = config.RETFOUND_CHECKPOINT_PATH if (load_pretrained and mode == "image_only") else None
    return TriModalCIMTModel(mode=mode, retfound_checkpoint=checkpoint)
