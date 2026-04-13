import torch
from torch import nn
import timm

class DeepfakeEdgeModel_Xception(nn.Module):
    def __init__(self, n_segment=20, num_classes=1, use_fft=False):
        super().__init__()
        self.n_segment = n_segment
        self.use_fft = use_fft

        # No global pooling → returns feature maps
        self.backbone = timm.create_model(
            "legacy_xception",
            pretrained=True,
            num_classes=0,
            global_pool=""   # important
        )

        feat_dim = 2048

        # GAP (now REQUIRED)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # FFT modification (optional)
        if self.use_fft:
            first_conv = self.backbone.conv1

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

            self.backbone.conv1 = new_conv

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        b, t, c, h, w = x.shape

        if not self.use_fft and c == 4:
            x = x[:, :, :3, :, :]

        x = x.view(b * t, x.shape[2], h, w)

        x = self.backbone(x)   # (b*t, C, H, W)
        x = self.gap(x)        # (b*t, C, 1, 1)

        x = x.view(b, t, -1)   # (B, t, feat_dim)
        x = x.mean(dim=1)      # (B, feat_dim)

        logits = self.classifier(x) # (B, 1)
        logits = logits.view(-1, 1)

        return logits