"""对 ``torch.utils.checkpoint.checkpoint`` 默认传入 ``use_reentrant=False``。

Transformers LLaMA 在 ``gradient_checkpointing`` 下调用 HuggingFace 内建的 checkpoint，
默认 ``use_reentrant=True`` 会在分段反向里再次走过同一批参数；DeepSpeed ZeRO Stage 2
在梯度钩子里「归约一次即标记完成」，易触发 ``already been reduced``。

``use_reentrant=False`` 走另一套分段反向实现，通常不再以相同方式嵌套触发二次钩子。
旧版 PyTorch 不支持该关键字时自动回退。"""

from __future__ import annotations

_PATCHED = False


def apply_gradient_checkpoint_use_reentrant_false() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import torch.utils.checkpoint as cu

    _orig = cu.checkpoint

    def _checkpoint(function, *args, **kwargs):
        kwargs.setdefault("use_reentrant", False)
        try:
            return _orig(function, *args, **kwargs)
        except TypeError:
            kwargs.pop("use_reentrant", None)
            return _orig(function, *args, **kwargs)

    cu.checkpoint = _checkpoint  # type: ignore[assignment]
    _PATCHED = True
