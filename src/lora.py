"""LoRA from scratch (plan section 5, leg 1).

A frozen linear layer W0 gets a trainable low-rank update:

    y = W0 x + (alpha / r) * B (A x)

with A (r x d_in) noise-initialized and B (d_out x r) zero-initialized, so
the adapted model is exactly the base model at step 0. Trainable params per
adapted matrix: r * (d_in + d_out) — i.e. 2 * r * d for square attention
projections. That count is asserted in tests, not just quoted.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        assert isinstance(base, nn.Linear) and base.bias is None
        self.base = base
        self.r = r
        self.scale = alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.base.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * F.linear(F.linear(x, self.A), self.B)

    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + self.scale * (self.B @ self.A)


def apply_lora(model: nn.Module, r: int, alpha: float, targets=("wq", "wv")) -> int:
    """Freeze the whole model, then wrap every Linear named in `targets`
    with a LoRALinear. Returns the trainable parameter count."""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in model.modules():
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every adapter into its base weight and swap back plain Linears —
    zero inference overhead, identical outputs."""
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                child.base.weight.data = child.merged_weight().data
                setattr(module, name, child.base)
    return model


def lora_state_dict(model: nn.Module) -> dict:
    """Just the adapter tensors — a LoRA checkpoint is a few hundred KB."""
    return {k: v for k, v in model.state_dict().items() if ".A" in k or ".B" in k}
