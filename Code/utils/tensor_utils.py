# Code/quantum/tensor_utils.py
# -*- coding: utf-8 -*-
"""
tensor_utils — маленькі утиліти для узгодження типів/форм і батч-викликів QNode.
"""

from __future__ import annotations
from typing import Any, Optional, Iterable
import torch


def to_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Приводить x до dtype/device як у ref (зберігає граф)."""
    if x.dtype != ref.dtype or x.device != ref.device:
        return x.to(device=ref.device, dtype=ref.dtype)
    return x


def ensure_tensor(x: Any, *, dtype: Optional[torch.dtype] = None,
                  device: Optional[torch.device] = None, logger=None) -> torch.Tensor:
    """
    Перетворює x (Tensor | list/tuple | scalar/np) на torch.Tensor з dtype/device, якщо задані.
    Якщо x не Tensor, можливе втрачання градієнтів — попереджаємо в лог.
    """
    if torch.is_tensor(x):
        t = x
    elif isinstance(x, (list, tuple)):
        if all(torch.is_tensor(v) for v in x):
            t = torch.stack([v for v in x])
        else:
            if logger:
                logger.warning("QNode returned non-tensor items; converting via torch.tensor (grad may be lost).")
            t = torch.tensor(x)
    else:
        if logger:
            logger.warning("QNode returned non-tensor value; converting via torch.as_tensor (grad may be lost).")
        t = torch.as_tensor(x)

    if dtype is not None or device is not None:
        t = t.to(dtype=(dtype or t.dtype), device=(device or t.device))
    return t


def tensor1d(x: Any, *, ref: torch.Tensor, logger=None) -> torch.Tensor:
    """
    x → 1D torch.Tensor, dtype/device як у ref (зберігаємо граф, якщо x вже Tensor).
    """
    t = ensure_tensor(x, dtype=ref.dtype, device=ref.device, logger=logger)
    if t.ndim != 1:
        t = t.reshape(-1)
    return t


def batched_apply(qnode, angles_batch: torch.Tensor, theta: torch.Tensor, *, logger=None) -> torch.Tensor:
    """
    Викликає qnode(angles_i, theta) для кожного елемента батчу:
      • приводить angles_i до dtype/device θ,
      • узгоджує вихід до 1D тензора dtype/device θ,
      • стекає у (B, out_dim).
    """
    B = int(angles_batch.shape[0])
    out_list = []
    for i in range(B):
        a_i = to_like(angles_batch[i], theta)
        out_i = qnode(a_i, theta)
        out_i = tensor1d(out_i, ref=theta, logger=logger)
        out_list.append(out_i)
    return torch.stack(out_list, dim=0)
