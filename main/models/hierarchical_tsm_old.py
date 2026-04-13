import torch
from torch import nn
import timm
from .utils import TSM

class DeepfakeEdgeModel_HTSM(nn.Module):
    def __init__(self, num_classes=1, use_fft=False):
        super().__init__()

        self.use_fft = use_fft

        backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=True,
            num_classes=0,
            global_pool=""
        )

        feat_dim = 1024
        self.backbone = backbone

        if self.use_fft:
            first_conv = self.backbone.conv_stem

            new_conv = nn.Conv2d(
                4,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False
            )

            with torch.no_grad():
                new_conv.weight[:, :3] = first_conv.weight
                nn.init.kaiming_normal_(new_conv.weight[:, 3:])

            self.backbone.conv_stem = new_conv

        # ===== TSM blocks =====
        self.tsm_intra = TSM(feat_dim, n_segment=5)
        self.tsm_inter = TSM(feat_dim, n_segment=4)

        # ===== CNN after intra =====
        self.cnn_intra = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.Hardswish()
        )

        # ===== CNN after inter =====
        self.cnn_inter = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.Hardswish()
        )

        # init CNNs
        for cnn in [self.cnn_intra, self.cnn_inter]:
            for m in cnn.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        if not self.use_fft and C == 4:
            x = x[:, :, :3, :, :]

        # ===== Backbone =====
        x = x.view(B * T, x.shape[2], H, W)
        x = self.backbone(x)  # (B*T, C, h, w)

        _, C_feat, h, w = x.shape

        # ===== Reshape into regions =====
        x = x.view(B, 4, 5, C_feat, h, w)

        # ===== INTRA TSM =====
        x = x.view(B * 4 * 5, C_feat, h, w)
        x = self.tsm_intra(x)

        # ✅ CNN after intra
        x = self.cnn_intra(x)

        # ===== Pool within region =====
        x = self.gap(x)
        x = x.view(B * 4, 5, C_feat)
        x = x.mean(dim=1)  # (B*4, C)

        # ===== INTER TSM =====
        x = x.view(B, 4, C_feat)
        x = x.view(B * 4, C_feat, 1, 1)

        x = self.tsm_inter(x)

        # ✅ CNN after inter
        x = self.cnn_inter(x)

        # ===== Final aggregation =====
        x = x.view(B, 4, C_feat)
        x = x.mean(dim=1)  # (B, C)

        return self.classifier(x).view(-1, 1)