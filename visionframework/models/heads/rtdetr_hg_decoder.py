"""Standalone RT-DETR decoder (Ultralytics-compatible ``state_dict`` layout; inference + optional training w/o CDN)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from visionframework.core.registry import HEADS
from visionframework.models.layers.rtdetr_hg_ops import bias_init_with_prob, meshgrid_ij
from visionframework.models.layers.rtdetr_hg_transformer import (
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    MLP,
)


def get_cdn_group(
    batch: dict[str, Any] | None,
    num_classes: int,
    num_queries: int,
    class_embed: torch.Tensor,
    num_dn: int = 100,
    cls_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0,
    training: bool = False,
):
    """Ultralytics CDN helper — only needed for full training; inference returns ``None`` s."""
    if (not training) or num_dn <= 0 or batch is None:
        return None, None, None, None
    raise NotImplementedError(
        "Contrastive denoising (CDN) training is not implemented in the standalone RT-DETR head."
    )


class RTDETRDecoderCore(nn.Module):
    """Mirrors Ultralytics ``RTDETRDecoder`` submodule names for weight loading."""

    export = False
    shapes: list = []
    anchors = torch.empty(0)
    valid_mask = torch.empty(0)
    dynamic = False

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,
        nq: int = 300,
        ndp: int = 4,
        nh: int = 8,
        ndl: int = 6,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = None,
        eval_idx: int = -1,
        nd: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
    ):
        super().__init__()
        if act is None:
            act = nn.ReLU()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl
        self.input_proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd, eps=0.001, momentum=0.03)) for x in ch
        )
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2, act=nn.ReLU, out_norm=nn.Identity())
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3, act=nn.ReLU, out_norm=nn.Identity())
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3, act=nn.ReLU, out_norm=nn.Identity()) for _ in range(ndl)])
        self._reset_parameters()

    def forward(self, x: list, batch: dict | None = None):
        feats, shapes = self._get_encoder_input(x)
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )
        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        out = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return out
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, out)

    @staticmethod
    def _generate_anchors(
        shapes: list[list[int]],
        grid_size: float = 0.05,
        dtype=torch.float32,
        device: str = "cpu",
        eps: float = 1e-2,
    ):
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = meshgrid_ij(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))
        anchors = torch.cat(anchors, 1)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x: list):
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            shapes.append([h, w])
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        bs = feats.shape[0]
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
            self.shapes = shapes
        features = self.enc_output(self.valid_mask * feats)
        enc_outputs_scores = self.enc_score_head(features)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype, device=topk_ind.device).unsqueeze(-1).repeat(1, self.num_queries).view(-1)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors
        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        if self.learnt_init_query:
            embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            embeddings = top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)
        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def _reset_parameters(self):
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


@HEADS.register("RTDETRHGDecoder")
class RTDETRHGDecoder(nn.Module):
    """Thin wrapper so ``head.decoder`` matches converted checkpoints."""

    def __init__(self, variant: str = "l", num_classes: int = 80, **_kw):
        super().__init__()
        variant = str(variant).lower()
        if variant not in ("l", "x"):
            raise ValueError("RTDETRHGDecoder variant must be 'l' or 'x'")
        ch = (256, 256, 256) if variant == "l" else (384, 384, 384)
        self.decoder = RTDETRDecoderCore(nc=num_classes, ch=ch)

    def forward(self, feats, batch=None):
        return self.decoder(feats, batch)
