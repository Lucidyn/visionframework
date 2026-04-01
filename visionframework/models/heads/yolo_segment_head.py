"""
YOLO instance-segmentation heads (YOLO11 ``Proto`` + ``cv4``, YOLO26 ``Proto26``).

Outputs are consumed by :class:`~visionframework.algorithms.segmentation.yolo_segmenter`
together with letterbox decode + NMS + ``process_mask_native``.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from visionframework.core.registry import HEADS
from visionframework.layers import ConvBNAct
from visionframework.models.heads.yolo_head import YOLOHead


class MaskProto(nn.Module):
    """Ultralytics ``Proto`` — mask prototypes from P3 features."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        super().__init__()
        self.cv1 = ConvBNAct(c1, c_, 3, 1)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = ConvBNAct(c_, c_, 3, 1)
        self.cv3 = ConvBNAct(c_, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Proto26(nn.Module):
    """Ultralytics ``Proto26`` — fused multi-scale prototypes + semantic head (unused at inference)."""

    def __init__(self, ch: tuple, c_: int = 256, c2: int = 32, nc: int = 80):
        super().__init__()
        self.nc = nc
        # Base Proto path uses c1=c_ so first conv is c_ -> c_
        self.cv1 = ConvBNAct(c_, c_, 3, 1)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = ConvBNAct(c_, c_, 3, 1)
        self.cv3 = ConvBNAct(c_, c2, 1, 1)
        self.feat_refine = nn.ModuleList(ConvBNAct(x, ch[0], 1, 1) for x in ch[1:])
        self.feat_fuse = ConvBNAct(ch[0], c_, 3, 1)
        self.semseg = nn.Sequential(
            ConvBNAct(ch[0], c_, 3, 1),
            ConvBNAct(c_, c_, 3, 1),
            nn.Conv2d(c_, nc, 1),
        )

    def forward(self, x: List[torch.Tensor], return_semseg: bool = False):
        feat = x[0]
        for i, f in enumerate(self.feat_refine):
            up_feat = f(x[i + 1])
            up_feat = F.interpolate(up_feat, size=feat.shape[2:], mode="nearest")
            feat = feat + up_feat
        fused = self.feat_fuse(feat)
        p = self.cv3(self.cv2(self.upsample(self.cv1(fused))))
        if self.training and return_semseg and getattr(self, "semseg", None) is not None:
            return p, self.semseg(feat)
        return p


@HEADS.register("YOLOSegmentHead")
class YOLOSegmentHead(nn.Module):
    """YOLO11 segment head: :class:`YOLOHead` + ``MaskProto`` + mask coefficient branches ``cv4``."""

    def __init__(
        self,
        in_channels: List[int],
        num_classes: int = 80,
        reg_max: int = 16,
        nm: int = 32,
        npr: int = 256,
        **_kw,
    ):
        super().__init__()
        self.nm = nm
        self.npr = npr
        self.det = YOLOHead(in_channels, num_classes, reg_max)
        ch = tuple(in_channels)
        self.proto = MaskProto(ch[0], npr, nm)
        c4 = max(ch[0] // 4, nm)
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                ConvBNAct(x, c4, 3, 1),
                ConvBNAct(c4, c4, 3, 1),
                nn.Conv2d(c4, nm, 1),
            )
            for x in ch
        )

    def forward(self, features: List[torch.Tensor]):
        det_out = self.det(features)
        proto = self.proto(features[0])
        bs = features[0].shape[0]
        parts = []
        for i, _ in enumerate(features):
            parts.append(self.cv4[i](features[i]).view(bs, self.nm, -1))
        mask_coeff = torch.cat(parts, dim=2)
        return {"det": det_out, "proto": proto, "mask_coeff": mask_coeff}


@HEADS.register("YOLO26SegmentHead")
class YOLO26SegmentHead(nn.Module):
    """YOLO26 segment head: :class:`YOLOHead` + :class:`Proto26` + ``cv4``."""

    def __init__(
        self,
        in_channels: List[int],
        num_classes: int = 80,
        reg_max: int = 1,
        nm: int = 32,
        npr: int = 256,
        **_kw,
    ):
        super().__init__()
        self.nm = nm
        self.npr = npr
        ch = tuple(in_channels)
        self.det = YOLOHead(in_channels, num_classes, reg_max)
        self.proto = Proto26(ch, npr, nm, num_classes)
        c4 = max(ch[0] // 4, nm)
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                ConvBNAct(x, c4, 3, 1),
                ConvBNAct(c4, c4, 3, 1),
                nn.Conv2d(c4, nm, 1),
            )
            for x in ch
        )

    def forward(self, features: List[torch.Tensor]):
        det_out = self.det(features)
        proto = self.proto(features)
        bs = features[0].shape[0]
        parts = []
        for i, _ in enumerate(features):
            parts.append(self.cv4[i](features[i]).view(bs, self.nm, -1))
        mask_coeff = torch.cat(parts, dim=2)
        return {"det": det_out, "proto": proto, "mask_coeff": mask_coeff}
