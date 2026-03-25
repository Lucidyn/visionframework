"""
Convert Ultralytics ``rtdetr-l.pt`` / ``rtdetr-x.pt`` (HGNet) to VisionFramework ``ModelWrapper`` state_dict.

Only ``torch`` is required to run this script. Official ``.pt`` files are published by Ultralytics (AGPL-3.0);
see ``NOTICE`` in the repository root.

Mapping (Ultralytics ``RTDETRDetectionModel.model`` index)::

    model.0 .. model.{N-1}  ->  backbone.layers.0 .. backbone.layers.{N-1}
    model.N                 ->  head.decoder.*

where ``N = 28`` for ``l`` and ``N = 32`` for ``x``.

Usage::

    python -m visionframework.tools.convert_ultralytics_rtdetr_hg \\
        --weights path/to/rtdetr-l.pt --variant l --out weights/detection/rtdetr/rtdetr_l_vf.pth --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from visionframework.core.builder import build_model
from visionframework.core.config import resolve_config


def vf_key_to_ultralytics(k: str, variant: str) -> str | None:
    """Inverse of :func:`ultralytics_key_to_vf` (for tests / round-trip)."""
    variant = variant.lower()
    n_enc = 28 if variant == "l" else 32
    if k.startswith("backbone.layers."):
        suffix = k[len("backbone.layers.") :]
        idx_str, rest = suffix.split(".", 1)
        return f"model.{int(idx_str)}.{rest}"
    if k.startswith("head.decoder."):
        rest = k[len("head.decoder.") :]
        return f"model.{n_enc}.{rest}"
    return None


def ultralytics_key_to_vf(k: str, variant: str) -> str | None:
    if not k.startswith("model."):
        return None
    parts = k.split(".")
    if len(parts) < 3:
        return None
    idx = int(parts[1])
    rest = ".".join(parts[2:])
    n_enc = 28 if variant == "l" else 32
    if idx < n_enc:
        return f"backbone.layers.{idx}.{rest}"
    if idx == n_enc:
        return f"head.decoder.{rest}"
    return None


def convert_checkpoint(
    ultra_path: str,
    out_path: str,
    variant: str,
    verify: bool = False,
    verify_config: str | None = None,
) -> None:
    variant = variant.lower()
    if variant not in ("l", "x"):
        raise ValueError("variant must be 'l' or 'x'")

    ckpt = torch.load(ultra_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        raw = ckpt["model"]
        sd = raw.state_dict() if isinstance(raw, torch.nn.Module) else raw
    elif isinstance(ckpt, dict):
        sd = ckpt.get("state_dict", ckpt)
    else:
        sd = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt

    new_sd = {}
    for k, v in sd.items():
        nk = ultralytics_key_to_vf(k, variant)
        if nk is not None:
            new_sd[nk] = v

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_sd, out)
    print(f"Wrote {len(new_sd)} tensors to {out}")

    if verify:
        cfg_name = verify_config or (
            "configs/detection/rtdetr/rtdetr_l.yaml" if variant == "l" else "configs/detection/rtdetr/rtdetr_x.yaml"
        )
        cfg = resolve_config(cfg_name)
        model = build_model(cfg, weights=None)
        missing, unexpected = model.load_state_dict(new_sd, strict=True)
        print(f"verify strict: missing={len(missing)} unexpected={len(unexpected)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=str, required=True, help="rtdetr-l.pt or rtdetr-x.pt")
    p.add_argument("--variant", type=str, required=True, choices=("l", "x"), help="Must match checkpoint")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--verify-config", type=str, default=None)
    args = p.parse_args()
    convert_checkpoint(args.weights, args.out, args.variant, verify=args.verify, verify_config=args.verify_config)


if __name__ == "__main__":
    main()
