# Code/quantum/metrics_bridge.py
# -*- coding: utf-8 -*-
"""
metrics_bridge.py — місток між квантовим шаром і Code/angle_metrics.py.

- Метрики по кутам φ: compute_phi_metrics(...)
- Метрики по градієнтах θ: compute_theta_grad_metrics(...)
- Статистика виходів: compute_expval_metrics(...)
- Пакування та лог: pack_metrics(...), pack_and_log(...)
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
import math
import torch

# ваш модуль з метриками
from Code.utils import angle_metrics as AM


# ----------------------------- Утиліти ----------------------------- #

def _to_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def _basic_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach()
    return {
        "min": _to_float(x.min()),
        "max": _to_float(x.max()),
        "mean": _to_float(x.mean()),
        "std": _to_float(x.std(unbiased=False) if x.numel() > 1 else torch.tensor(0.0)),
    }


def merge_dicts(*ds: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in ds:
        out.update(d)
    return out


# ------------------------- Метрики по φ (angles) ------------------------- #

def compute_phi_metrics(
    phi_batch: torch.Tensor,
    *,
    include_per_pc_std: bool = False,
    angle_max: Optional[float] = math.pi,
) -> Dict[str, Any]:
    """
    phi_batch: Tensor[B, n_qubits] у [-π, π]
    Повертає ключі 'phi/*': базова статистика + (за наявності) saturation/live та std_per_pc агрегати.
    """
    if not isinstance(phi_batch, torch.Tensor) or phi_batch.ndim != 2:
        raise ValueError(f"phi_batch must be Tensor[B, n_qubits], got {type(phi_batch)} with shape {getattr(phi_batch,'shape',None)}")

    B, n_qubits = phi_batch.shape
    flat = phi_batch.view(-1)
    stats = _basic_stats(flat)

    out: Dict[str, Any] = {
        "phi/batch": int(B),
        "phi/n_qubits": int(n_qubits),
        "phi/min": stats["min"],
        "phi/max": stats["max"],
        "phi/mean": stats["mean"],
        "phi/std": stats["std"],
    }

    # AM.saturation_rate_phi і AM.live_rate_phi повертають (overall, per_pc)
    try:
        sat_overall, _ = AM.saturation_rate_phi(phi_batch, angle_max=angle_max if angle_max is not None else math.pi)
        out["phi/saturation_rate"] = float(sat_overall)
    except Exception:
        pass

    try:
        live_overall, _ = AM.live_rate_phi(phi_batch)
        out["phi/live_rate"] = float(live_overall)
    except Exception:
        pass

    if include_per_pc_std:
        try:
            per_pc = AM.std_phi(phi_batch)  # 1D: (n_qubits,)
            # узагальнюємо, щоб не засмічувати логи
            t = per_pc if isinstance(per_pc, torch.Tensor) else torch.as_tensor(per_pc)
            out["phi/std_per_pc_mean"] = _to_float(t.mean())
            out["phi/std_per_pc_max"] = _to_float(t.max())
        except Exception:
            pass

    return out


# ---------------------- Метрики по градієнтах θ ---------------------- #

def compute_theta_grad_metrics(theta: torch.Tensor) -> Dict[str, Any]:
    """
    theta: Tensor[L, n_qubits]; викликати ПІСЛЯ loss.backward().
    Повертає 'theta/*' і базові grad-статистики.
    """
    if not isinstance(theta, torch.Tensor) or theta.ndim != 2:
        raise ValueError(f"theta must be Tensor[L, n_qubits], got {type(theta)} with shape {getattr(theta,'shape',None)}")

    out: Dict[str, Any] = {
        "theta/L": int(theta.shape[0]),
        "theta/n_qubits": int(theta.shape[1]),
        "theta/has_grad": theta.grad is not None,
    }

    g = theta.grad
    if g is not None:
        # Базові ручні метрики
        out.update({
            "theta/grad_mean": _to_float(g.mean()),
            "theta/grad_std": _to_float(g.std(unbiased=False) if g.numel() > 1 else torch.tensor(0.0)),
            "theta/grad_max": _to_float(g.abs().max()),
            "theta/grad_l2": _to_float(torch.linalg.vector_norm(g)),
        })
        # Додаткові агрегати від AM.grad_stats (працює по iterable параметрів/градієнтів)
        try:
            extra = AM.grad_stats([theta])  # бере theta.grad всередині
            # нормалізуємо імена під наш префікс
            for k, v in extra.items():
                out[f"theta/grad.{k}"] = float(v)
        except Exception:
            pass

    return out


# ---------------------- Метрики виходів (expvals) ---------------------- #

def compute_expval_metrics(outputs: torch.Tensor, *, prefix: str = "out") -> Dict[str, Any]:
    """
    outputs: Tensor з ndim>=1 (типово B×n_qubits).
    Повертає базові статистики під ключами {prefix}/*.
    """
    if not isinstance(outputs, torch.Tensor) or outputs.ndim < 1:
        raise ValueError(f"outputs must be Tensor with ndim>=1, got {type(outputs)} with shape {getattr(outputs,'shape',None)}")

    flat = outputs.detach().view(-1)
    s = _basic_stats(flat)
    return {
        f"{prefix}/min": s["min"],
        f"{prefix}/max": s["max"],
        f"{prefix}/mean": s["mean"],
        f"{prefix}/std": s["std"],
        f"{prefix}/numel": int(flat.numel()),
    }


# ---------------------- Пакування та лог ---------------------- #

def pack_metrics(metrics: Mapping[str, Any], *, prefix: Optional[str] = None) -> Dict[str, float]:
    """
    Якщо задано prefix — використовує Code.angle_metrics.pack_metrics_dict(prefix, **metrics),
    інакше повертає плаский dict як є.
    """
    if prefix is not None:
        try:
            return AM.pack_metrics_dict(prefix, **dict(metrics))
        except Exception:
            # фолбек: якщо з якоїсь причини pack_metrics_dict впав — повертаємо як є
            return dict(metrics)  # type: ignore[return-value]
    return dict(metrics)


def pack_and_log(
    logger,
    metrics: Mapping[str, Any],
    *,
    prefix: Optional[str] = None,
    level: str = "INFO",
) -> None:
    """
    Пакує метрики (через AM.pack_metrics_dict, якщо задано prefix) і логує їх.
    """
    payload = pack_metrics(metrics, prefix=prefix)
    msg = f"{prefix} | {payload}" if prefix and not str(payload).startswith(f"{prefix}/") else str(payload)
    getattr(logger, level.lower(), logger.info)(msg)
