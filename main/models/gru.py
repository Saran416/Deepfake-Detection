import torch
from torch import nn
import timm
from .utils import TSM

class DeepfakeEdgeModel_GRU(nn.Module):
    def __init__(self, n_segment=20, num_classes=1, use_fft=False, hidden_size=256, num_layers=2):
        super().__init__()
        self.n_segment = n_segment
        self.use_fft = use_fft

        # Backbone (same as before)
        backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=True,
            num_classes=0,
            global_pool=""
        )

        feat_dim = 1024
        self.backbone = backbone

        # FFT input modification (same as before)
        if self.use_fft:
            first_conv = self.backbone.conv_stem

            new_conv = nn.Conv2d(
                4,  # RGB + FFT
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

        # Spatial pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # GRU replaces TSM
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )

        self.temporal_norm = nn.LayerNorm(feat_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        b, t, c, h, w = x.shape

        if not self.use_fft and c == 4:
            x = x[:, :, :3, :, :]  # keep only RGB

        # Merge batch and time for CNN
        x = x.view(b * t, x.shape[2], h, w)

        # Backbone feature extraction
        x = self.backbone(x)              # (B*T, C, H, W)
        x = self.gap(x)                  # (B*T, C, 1, 1)
        x = x.view(b, t, -1)             # (B, T, C)

        x = self.temporal_norm(x)

        # GRU temporal modeling
        out, h_n = self.gru(x)           # out: (B, T, hidden)

        # Option 1: last hidden state
        # x = h_n[-1]                      # (B, hidden)

        # Option 2 (alternative): mean pooling
        x = out.mean(dim=1)

        return self.classifier(x).view(-1, 1)