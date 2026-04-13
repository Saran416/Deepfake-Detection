import torch
from torch import nn
import timm
from .utils import TSM  # (not used here but fine to keep for consistency)

class DeepfakeEdgeModel_Mobile(nn.Module):
    def __init__(self, n_segment=20, num_classes=1, use_fft=False):
        super().__init__()
        self.n_segment = n_segment
        self.use_fft = use_fft

        backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=True,
            num_classes=0,
            global_pool=""
        )

        feat_dim = 1024
        self.backbone = backbone

        # ==== FFT handling ====
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

        self.cnn = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.Hardswish()
        )

        for m in self.cnn.modules():
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
        b, t, c, h, w = x.shape

        if not self.use_fft and c == 4:
            x = x[:, :, :3, :, :]

        x = x.view(b * t, x.shape[2], h, w)

        x = self.backbone(x)   # (b*t, C, H, W)

        x = self.cnn(x)        # (b*t, C, H, W)

        x = self.gap(x)        # (b*t, C, 1, 1)

        x = x.view(b, t, -1)   # (B, t, C)
        x = x.mean(dim=1)      # (B, C)

        logits = self.classifier(x)
        return logits.view(-1, 1)