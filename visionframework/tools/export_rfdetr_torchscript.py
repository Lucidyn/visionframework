from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _build_rfdetr(model_size: str):
    # Import inside to keep runtime dependency optional.
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium, RFDETRLarge

    size = model_size.lower()
    if size == "nano":
        return RFDETRNano()
    if size == "small":
        return RFDETRSmall()
    if size == "base":
        return RFDETRBase()
    if size == "medium":
        return RFDETRMedium()
    if size == "large":
        return RFDETRLarge()
    raise ValueError(f"Unknown model size: {model_size}")


def export_one(model_size: str, out_path: str) -> None:
    """
    Export RF-DETR official checkpoint to a `.pth` file.

    This tool requires the `rfdetr` package. It downloads the official weights
    (if missing) and writes a framework-friendly checkpoint dict:
      {"model": state_dict, "meta": {...}}
    """
    m = _build_rfdetr(model_size)
    # Ensure official weights exist (ModelConfig downloads on init).
    # We save a clean checkpoint containing only model weights.
    state = m.model.model.state_dict()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": state,
            "meta": {"model_size": model_size, "resolution": m.model.resolution},
        },
        str(out),
    )
    print(f"Saved PTH: {out} (resolution={m.model.resolution})")


def main() -> None:
    p = argparse.ArgumentParser(description="Export RF-DETR official `.pth` checkpoint (requires `rfdetr`).")
    p.add_argument("--size", choices=["nano", "small", "base", "medium", "large"], required=True)
    p.add_argument("--out", required=True, help="Output .pth path for torch.save()")
    args = p.parse_args()

    export_one(args.size, args.out)


if __name__ == "__main__":
    main()

