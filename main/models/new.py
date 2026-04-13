import torch
from torch import nn
import timm
from .utils import TSM


class BlockWithTSM(nn.Module):
    def __init__(self, block, channels, n_segment):
        super().__init__()
        self.block = block
        self.tsm = TSM(channels, n_segment)

    def forward(self, x):
        x = self.tsm(x)
        x = self.block(x)
        return x


class DeepfakeEdgeModel_STSM(nn.Module):
    def __init__(self, n_segment=20, num_classes=1):
        super().__init__()
        self.n_segment = n_segment

        # Backbone
        backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=True,
            num_classes=0,
            global_pool=""
        )

        self.backbone = backbone
        feat_dim = self.backbone.num_features

        # ----- Insert TSM into backbone blocks -----
        new_blocks = []
        for blk in self.backbone.blocks:
            if hasattr(blk, "conv_pwl"):
                out_ch = blk.conv_pwl.out_channels
            elif hasattr(blk, "conv_pw"):
                out_ch = blk.conv_pw.out_channels
            elif hasattr(blk, "bn3"):
                out_ch = blk.bn3.num_features
            else:
                out_ch = feat_dim

            new_blocks.append(BlockWithTSM(blk, out_ch, self.n_segment))

        self.backbone.blocks = nn.Sequential(*new_blocks)

        # Pool + classifier
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        b, t, c, h, w = x.shape

        # Ensure RGB
        if c > 3:
            x = x[:, :, :3, :, :]

        x = x.view(b * t, 3, h, w)

        # Backbone forward
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)

        x = self.backbone.blocks(x)   # TSM inside

        x = self.backbone.conv_head(x)
        x = self.backbone.act2(x)

        # Pool
        x = self.gap(x)

        # Temporal aggregation
        x = x.view(b, t, -1)
        x = x.mean(dim=1)

        logits = self.classifier(x)
        return logits.view(-1, 1)