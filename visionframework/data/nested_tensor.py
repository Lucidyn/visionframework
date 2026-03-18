from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class NestedTensor:
    tensors: torch.Tensor
    mask: Optional[torch.Tensor]

    def to(self, device):
        t = self.tensors.to(device)
        m = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(t, m)

    def decompose(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.tensors, self.mask


def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor] | torch.Tensor) -> NestedTensor:
    if isinstance(tensor_list, torch.Tensor):
        t = tensor_list
        b, _, h, w = t.shape
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=t.device)
        return NestedTensor(t, mask)

    if len(tensor_list) == 0:
        raise ValueError("tensor_list is empty")

    max_h = max(x.shape[-2] for x in tensor_list)
    max_w = max(x.shape[-1] for x in tensor_list)
    batch = torch.stack(
        [torch.nn.functional.pad(x, (0, max_w - x.shape[-1], 0, max_h - x.shape[-2])) for x in tensor_list],
        dim=0,
    )
    mask = torch.ones((len(tensor_list), max_h, max_w), dtype=torch.bool, device=batch.device)
    for i, x in enumerate(tensor_list):
        mask[i, : x.shape[-2], : x.shape[-1]] = False
    return NestedTensor(batch, mask)

