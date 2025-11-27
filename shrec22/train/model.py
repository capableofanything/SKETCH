"""
Model definitions for SHREC22
"""
import torch
import torch.nn as nn
from timm import create_model


class SwinWithRegressor(nn.Module):
    """
    Swin Transformer based model with classification and regression heads

    This model:
    - Uses a pretrained Swin Transformer as backbone
    - Adds gesture start/end regressors
    - Incorporates coordinate information from skeletal sequences

    Args:
        num_classes: Number of gesture classes
        pretrained_model: Name of the pretrained model to use
    """

    def __init__(self, num_classes, pretrained_model):
        super(SwinWithRegressor, self).__init__()

        # Load pretrained Swin Transformer
        self.swin = create_model(
            pretrained_model,
            pretrained=True,
            num_classes=num_classes
        )

        # Enable gradient checkpointing to save memory
        self.swin.set_grad_checkpointing(True)

        # Get feature dimension from the backbone
        feature_dim = self.swin.num_features

        # MLP for processing coordinate information
        self.coord_mlp = nn.Sequential(
            nn.Linear(16 * 26, 16 * 26),
            nn.BatchNorm1d(16 * 26),
            nn.ReLU(inplace=True),
            nn.Linear(16 * 26, 16 * 26),
            nn.BatchNorm1d(16 * 26),
            nn.ReLU(inplace=True),
            nn.Linear(16 * 26, 1)
        )

        # Replace original classifier head
        self.swin.head = nn.Identity()
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Regression heads for gesture start/end prediction
        self.regressor_start = nn.Linear(feature_dim, 1)
        self.regressor_end = nn.Linear(feature_dim, 1)

    def forward(self, x, seq):
        """
        Forward pass

        Args:
            x: Input images [B, C, H, W]
            seq: Sequence data [B, 16, 3, 26]

        Returns:
            classification_output: Class logits [B, num_classes]
            gesture_start: Predicted gesture start positions [B]
            gesture_end: Predicted gesture end positions [B]
            features: Extracted features [B, feature_dim]
        """
        B = x.size(0)

        # 1) Process sequence data to get coordinate representation [B, 3]
        # seq: [B,16,3,26] → [B,3,16,26] → [B*3,16*26]
        coords = seq.permute(0, 2, 1, 3).reshape(B * 3, -1)
        coord_feats = self.coord_mlp(coords)  # [B*3, 1]
        representation_coord = coord_feats.view(B, 3)  # [B, 3]

        # 2) Add coordinate information to input image
        # x: [B, C_in, H, W]
        _, C_in, H, W = x.shape
        chunk = H // 3

        # Create additive map: [B, 1, H, 1]
        add_map = torch.zeros(B, 1, H, 1, device=x.device, dtype=x.dtype)

        for i in range(3):
            start = i * chunk
            end = (i + 1) * chunk if i < 2 else H
            add_map[:, :, start:end, :] = representation_coord[:, i].view(B, 1, 1, 1)

        # Add coordinate information and process through backbone
        x_mod = x + add_map  # → [B, C_in, H, W]
        features = self.swin(x_mod)

        # Generate outputs
        classification_output = self.classifier(features)
        gesture_start = self.regressor_start(features).squeeze(1)
        gesture_end = self.regressor_end(features).squeeze(1)

        return classification_output, gesture_start, gesture_end, features

