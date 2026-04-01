"""Instance mask helpers aligned with ultralytics ``process_mask_native`` / ``scale_masks``."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def crop_mask(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Crop N masks (N, H, W) to xyxy boxes (N, 4) in pixel coords."""
    if boxes.device != masks.device:
        boxes = boxes.to(masks.device)
    n, h, w = masks.shape
    if n < 50 and not masks.is_cuda:
        for i, (x1, y1, x2, y2) in enumerate(boxes.round().int()):
            masks[i, :y1] = 0
            masks[i, y2:] = 0
            masks[i, :, :x1] = 0
            masks[i, :, x2:] = 0
        return masks
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def scale_masks(
    masks: torch.Tensor,
    shape: tuple[int, int],
    ratio_pad: tuple[tuple[float, float], tuple[float, float]] | None = None,
    padding: bool = True,
) -> torch.Tensor:
    """Rescale segment masks (N, C, H, W) to ``shape`` (H0, W0), letterbox-aware."""
    im1_h, im1_w = masks.shape[2:]
    im0_h, im0_w = shape[:2]
    if im1_h == im0_h and im1_w == im0_w:
        return masks

    if ratio_pad is None:
        gain = min(im1_h / im0_h, im1_w / im0_w)
        pad_w = (im1_w - round(im0_w * gain)) / 2
        pad_h = (im1_h - round(im0_h * gain)) / 2
    else:
        pad_w, pad_h = ratio_pad[1]

    top = round(pad_h - 0.1) if padding else 0
    left = round(pad_w - 0.1) if padding else 0
    bottom = im1_h - round(pad_h + 0.1)
    right = im1_w - round(pad_w + 0.1)
    return F.interpolate(
        masks[..., top:bottom, left:right].float(),
        shape,
        mode="bilinear",
    )


def process_mask_native(
    protos: torch.Tensor,
    masks_in: torch.Tensor,
    bboxes: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    """Binary masks (N, H0, W0) from proto (nm, mh, mw), coeffs (N, nm), boxes xyxy."""
    c, mh, mw = protos.shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = scale_masks(masks[None], shape)[0]
    masks = crop_mask(masks, bboxes)
    return masks.gt(0.0).byte()
