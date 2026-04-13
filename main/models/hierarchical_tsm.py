import torch
from torch import nn
import timm
from .utils import TSM

class DeepfakeEdgeModel_Residual_HTSM(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # ===== Backbone =====
        backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=True,
            num_classes=0,
            global_pool=""
        )

        self.backbone = backbone
        feat_dim = 1024

        # ===== Hierarchical TSM =====
        self.tsm_intra = TSM(feat_dim, n_segment=5, mode="diff")   # within region
        self.tsm_inter = TSM(feat_dim, n_segment=4, mode="shift")   # across regions

        # ===== CNN blocks (residual branches) =====
        self.cnn_intra = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.Hardswish()
        )

        self.cnn_inter = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.Hardswish()
        )

        # init CNN weights
        for cnn in [self.cnn_intra, self.cnn_inter]:
            for m in cnn.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # ===== Pool + classifier =====
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        x: (B, 80, 3, H, W)
        structured as:
            8 regions × 10 frames
        """

        B, T, C, H, W = x.shape  # T = 80

        # ===== Backbone =====
        x = x.view(B * T, C, H, W)
        x = self.backbone(x)  # (B*T, C_feat, h, w)

        _, C_feat, h, w = x.shape

        # ===== Reshape: (B, 8 regions, 10 frames) =====
        x = x.view(B, 8, 10, C_feat, h, w)

        # =========================================================
        # 🔹 INTRA TSM (within each region) — Residual
        # =========================================================
        x = x.view(B * 8 * 10, C_feat, h, w)

        identity = x
        out = self.tsm_intra(x)
        out = self.cnn_intra(out)
        x = identity + out

        # ===== Pool within region =====
        x = self.gap(x)                         # (B*8*10, C, 1, 1)
        x = x.view(B * 8, 10, C_feat)
        x = x.mean(dim=1)                      # (B*8, C)

        # =========================================================
        # 🔹 INTER TSM (across regions) — Residual
        # =========================================================
        x = x.view(B, 8, C_feat)
        x = x.view(B * 8, C_feat, 1, 1)

        identity = x
        out = self.tsm_inter(x)
        out = self.cnn_inter(out)
        x = identity + out

        # ===== Final aggregation =====
        x = x.view(B, 8, C_feat)
        x = x.mean(dim=1)  # (B, C)

        logits = self.classifier(x)

        return logits.view(-1, 1)