#!/usr/bin/env python3
"""Print L2 norms of CLIP routing anchors (and optional boundaries) saved in a checkpoint directory."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import torch

_SAME_PREFIX = "mcitbox.same."
_LIST_KEYS = ("image_anchors", "text_anchors", "image_boundary", "text_boundary")


def _tensor_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach().float().reshape(-1)).item())


def _anchors_from_state_dict(state: Dict[str, Any]) -> Dict[str, List[torch.Tensor]]:
    """Extract list-valued anchor tensors from a nested state dict (e.g. same_state.bin)."""
    out: Dict[str, List[torch.Tensor]] = {}
    if any(isinstance(k, str) and k.startswith("_base_model.") for k in state.keys()):
        state = {k[len("_base_model.") :] if str(k).startswith("_base_model.") else k: v for k, v in state.items()}
    for lk in _LIST_KEYS:
        seq = state.get(lk)
        if not isinstance(seq, (list, tuple)):
            continue
        tensors = [t for t in seq if isinstance(t, torch.Tensor)]
        if tensors:
            out[lk] = tensors
    return out


def _anchors_from_safetensors_flat(flat: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
    """Rebuild anchor lists from merged SAME keys in adapter_model.safetensors."""
    sub = {k: v for k, v in flat.items() if isinstance(k, str) and k.startswith(_SAME_PREFIX)}
    if not sub:
        return {}
    state: Dict[str, List[torch.Tensor]] = {}
    for lk in _LIST_KEYS:
        pref_list = f"{_SAME_PREFIX}{lk}."
        idx_tensors: Dict[int, torch.Tensor] = {}
        for k, v in sub.items():
            if not k.startswith(pref_list):
                continue
            tail = k[len(pref_list) :]
            if tail.isdigit():
                idx_tensors[int(tail)] = v
        if idx_tensors:
            mx = max(idx_tensors)
            lst = [idx_tensors[i] for i in range(mx + 1) if i in idx_tensors]
            if lst:
                state[lk] = lst
    return state


def _try_load_hide_disco(pt_path: str) -> Dict[str, List[torch.Tensor]]:
    blob = torch.load(pt_path, map_location="cpu")
    if not isinstance(blob, dict):
        return {}
    return _anchors_from_state_dict(blob)


def load_anchor_lists(checkpoint_dir: str) -> Tuple[str, Dict[str, List[torch.Tensor]]]:
    """Return (source_description, anchors_dict)."""
    d = os.path.expanduser(checkpoint_dir)

    p_same = os.path.join(d, "same_state.bin")
    if os.path.isfile(p_same):
        st = torch.load(p_same, map_location="cpu")
        if isinstance(st, dict):
            got = _anchors_from_state_dict(st)
            if got:
                return (p_same, got)

    p_st = os.path.join(d, "adapter_model.safetensors")
    if os.path.isfile(p_st):
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise RuntimeError("Reading adapter_model.safetensors requires `safetensors`") from e
        flat = load_file(p_st)
        got = _anchors_from_safetensors_flat(flat)
        if got:
            return (p_st, got)

    for name in ("hide_state.pt", "disco_state.pt"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            got = _try_load_hide_disco(p)
            if got:
                return (p, got)

    return ("", {})


def _print_group(title: str, tensors: List[torch.Tensor]) -> None:
    print(title)
    for i, t in enumerate(tensors):
        print(f"  [{i}] shape={tuple(t.shape)}  L2_norm={_tensor_norm(t):.6f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "checkpoint_dir",
        type=str,
        help="Directory containing same_state.bin and/or adapter_model.safetensors (or hide/disco_state.pt)",
    )
    ap.add_argument(
        "--include-boundary",
        action="store_true",
        help="Also print norms for image_boundary / text_boundary if present",
    )
    args = ap.parse_args()

    src, groups = load_anchor_lists(args.checkpoint_dir)
    if not groups:
        print(
            "No anchor tensors found. Expected one of:\n"
            "  - same_state.bin (lists image_anchors / text_anchors)\n"
            "  - adapter_model.safetensors with mcitbox.same.image_anchors.* keys\n"
            "  - hide_state.pt / disco_state.pt",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"source: {src}\n")
    if "image_anchors" in groups:
        _print_group("image_anchors", groups["image_anchors"])
        print()
    if "text_anchors" in groups:
        _print_group("text_anchors", groups["text_anchors"])
        print()

    if args.include_boundary:
        if "image_boundary" in groups:
            _print_group("image_boundary", groups["image_boundary"])
            print()
        if "text_boundary" in groups:
            _print_group("text_boundary", groups["text_boundary"])
            print()

    # Summary line for quick comparison
    if "image_anchors" in groups and "text_anchors" in groups:
        ni, nt = len(groups["image_anchors"]), len(groups["text_anchors"])
        print(f"counts: image_anchors={ni}  text_anchors={nt}")
        if ni == nt:
            print("pairwise |img - txt| L2 (per task index):")
            for i in range(ni):
                a = groups["image_anchors"][i].detach().float().reshape(-1)
                b = groups["text_anchors"][i].detach().float().reshape(-1)
                m = min(a.numel(), b.numel())
                d = float(torch.linalg.vector_norm(a[:m] - b[:m]).item())
                print(f"  [{i}] diff_L2={d:.6f}")


if __name__ == "__main__":
    main()
