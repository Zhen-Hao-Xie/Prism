#!/usr/bin/env python3
"""
验证 O-LoRA 组件是否按设计生效：正交 loss 非零、专家可训练掩码、integration 修改 loss。

用法（在仓库根目录）:
  python scripts/verify_olora_components.py
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _test_fake_named_params() -> None:
    """嵌套子模块产生与真实 PEFT 一致的参数名，检验正则与正交累加。"""
    import torch
    import torch.nn as nn
    from PEFT.tuners.custom.olora import (
        apply_olora_expert_trainable_mask,
        compute_olora_orthogonal_loss,
    )

    class ExpertA(nn.Module):
        def __init__(self, fill: float) -> None:
            super().__init__()
            self.mlp = nn.Linear(8, 2, bias=False)
            nn.init.constant_(self.mlp.weight, fill)

    class ExpertB(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Linear(2, 16, bias=False)
            nn.init.zeros_(self.mlp.weight)

    class FakeLoraA(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.loraA = nn.ModuleList([ExpertA(1.0), ExpertA(2.0)])

    class FakeLoraB(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.loraB = nn.ModuleList([ExpertB(), ExpertB()])

    class FakeQProj(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A = nn.ModuleDict({"default": FakeLoraA()})
            self.lora_B = nn.ModuleDict({"default": FakeLoraB()})

    layer0 = nn.Module()
    layer0.self_attn = nn.Module()
    layer0.self_attn.q_proj = FakeQProj()
    root = nn.Module()
    root.base_model = nn.Module()
    root.base_model.model = nn.Module()
    root.base_model.model.layers = nn.ModuleList([layer0])

    pfx = "base_model.model.layers.0.self_attn.q_proj."
    z0 = compute_olora_orthogonal_loss(root, cur_task=0, adapter_name="default")
    assert float(z0.item()) == 0.0, "cur_task=0 时不应有正交项"

    z1 = compute_olora_orthogonal_loss(root, cur_task=1, adapter_name="default")
    # A0 全 1、A1 全 2，形状 (2,8)：(A1 @ A0^T) 每个元素为 sum_k 2*1=16，2x2 共 64
    assert abs(float(z1.item()) - 64.0) < 1e-3, z1.item()

    apply_olora_expert_trainable_mask(root, cur_task=1, adapter_name="default")
    grads = {n: p.requires_grad for n, p in root.named_parameters() if "lora_" in n}
    assert grads[pfx + "lora_A.default.loraA.0.mlp.weight"] is False
    assert grads[pfx + "lora_A.default.loraA.1.mlp.weight"] is True
    assert grads[pfx + "lora_B.default.loraB.0.mlp.weight"] is False
    assert grads[pfx + "lora_B.default.loraB.1.mlp.weight"] is True
    print("[ok] 伪造模块: 正交 loss 与专家 requires_grad 掩码")


def _test_tiny_llama_peft() -> None:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from PEFT import get_peft_model
    from PEFT.tuners.custom.olora import (
        apply_olora_expert_trainable_mask,
        compute_olora_orthogonal_loss,
        sync_olora_cur_task,
    )
    from method.custom.olora.integration import OloraIntegration

    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    base = LlamaForCausalLM(cfg)
    peft_config = __import__("PEFT.tuners.custom.olora", fromlist=["OLoRAConfig"]).OLoRAConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        expert_num=2,
        cur_task=1,
        bias="none",
    )
    pm = get_peft_model(base, peft_config)
    apply_olora_expert_trainable_mask(pm, cur_task=1, adapter_name="default")
    sync_olora_cur_task(pm, 1)

    a_names = [
        n
        for n, _ in pm.named_parameters()
        if "lora_A.default.loraA" in n and (n.endswith("mlp.weight") or n.endswith(".weight"))
    ]
    assert a_names, "应存在 lora_A.default.loraA.{i}.mlp.weight 参数"
    print(f"[info] 示例 A 参数名: {a_names[0][:80]}...")

    orth = compute_olora_orthogonal_loss(pm, 1, adapter_name="default")
    assert orth.numel() == 1 and orth.item() >= 0.0
    print(f"[info] compute_olora_orthogonal_loss(cur_task=1) = {float(orth.item()):.6f}")

    input_ids = torch.randint(0, 128, (2, 8))
    labels = input_ids.clone()
    pm.train()
    out = pm(input_ids=input_ids, labels=labels)
    assert out.loss is not None
    loss_lm = out.loss.detach().item()

    integ = OloraIntegration(
        SimpleNamespace(
            task_num=2,
            cur_task=1,
            olora_lambda=0.05,
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            peft_target_modules="attn_only",
            exclude_module_path_segments=[],
        )
    )

    class _Wrap:
        """模拟 CLModel：integration 会读 ``training`` 与 ``_base_model``。"""

        training = True
        _base_model = pm

    out2 = integ.on_forward_end(_Wrap(), out, SimpleNamespace(task_id=1))
    expected = loss_lm + 0.05 * float(orth.detach().item())
    got = float(out2.loss.detach().item())
    assert abs(got - expected) < 1e-4 * max(1.0, abs(expected)), (got, expected)
    print(f"[ok] Tiny Llama PEFT: LM loss={loss_lm:.4f}, +0.05*orth => total={got:.4f}")

    # 梯度：正交项应对当前任务 A/B 有梯度，历史 A 无梯度（已 detach）
    out2.loss.backward()
    hist_a = [n for n, p in pm.named_parameters() if "lora_A.default.loraA.0.mlp" in n and p.grad is not None]
    cur_a = [n for n, p in pm.named_parameters() if "lora_A.default.loraA.1.mlp" in n and p.grad is not None]
    assert not hist_a, f"历史专家 A 不应有 grad: {hist_a[:1]}"
    assert cur_a, "当前任务 A 应有 grad"
    print("[ok] 反传: 历史 A 无 grad，当前 A 有 grad")


def main() -> None:
    _test_fake_named_params()
    _test_tiny_llama_peft()
    print("\n全部检查通过：O-LoRA 正交项、掩码、integration.loss 与梯度行为符合设计。")


if __name__ == "__main__":
    main()
