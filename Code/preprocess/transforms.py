# Code/transforms.py
"""
Перетворення після PCA для тренування:
    X_PC  ->  z (скейлінг за preprocess_stats.yaml)  ->  φ (кутове кодування)

Склад:
- PCScaler:     онлайн-стандартизація/масштабування (Z-score, Robust, Global L2, Max-Abs).
- AngleEncoder: кутове кодування φ = clip((π/k)·z, ±angle_max).
- encode_from_pca: тонкий хелпер X_PC -> φ (корисно для smoke-тестів).
- (опц.) метрики: clip_rate, live_rate, std_phi.
- (опц.) валідація z-кешу: validate_z_cache_from_stats.

Залежності: numpy, pyyaml; torch — опційно (якщо використовуєш тензори у тренуванні).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# Опційна підтримка torch
try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# Локальні утиліти I/O та fingerprint
from Code.utils.io_utils import load_yaml, fingerprint_matches

ArrayLike = Union[np.ndarray, "torch.Tensor"]


# ---------- дрібні утиліти типів ----------

def _is_torch(x: Any) -> bool:
    return _HAS_TORCH and isinstance(x, torch.Tensor)  # type: ignore[name-defined]


def _as_numpy(x: ArrayLike) -> np.ndarray:
    if _is_torch(x):
        return x.detach().cpu().numpy()  # type: ignore[union-attr]
    return np.asarray(x)


def _same_type_like(x: ArrayLike, y_np: np.ndarray) -> ArrayLike:
    """
    Повернути y_np у тому ж типі/пристрої, що і x.
    """
    if _is_torch(x):
        # переносимо dtype/device як у x, зберігаємо gradient flow (через torch.tensor)
        return torch.as_tensor(y_np, dtype=x.dtype, device=x.device)  # type: ignore[union-attr]
    return y_np


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


# ---------- PCScaler ----------

@dataclass
class PCScaler:
    """
    Онлайн-скейлер після PCA. Відтворює перетворення, зафіксоване у preprocess_stats.yaml.
    Підтримувані методи:
        - "zscore":  (x - mu) / (sigma + eps)
        - "robust":  (x - median) / (denom + eps)   # denom = MAD*c_mad або IQR/c_iqr
        - "global_l2":  x / (||x||_2 + eps)
        - "maxabs":     x / (max|x| + eps)
    """
    method: str
    pca_dim: int
    eps: float
    dtype_str: str = "float32"
    # статистики per-PC (для "zscore" / "robust"); None для глобальних методів
    mu: Optional[np.ndarray] = None
    sigma: Optional[np.ndarray] = None
    median: Optional[np.ndarray] = None
    denom: Optional[np.ndarray] = None  # робастна оцінка σ
    # службове: fingerprint з yaml (не обов’язковий)
    fingerprint: Optional[Dict[str, Any]] = None

    # ---- фабрики ----
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PCScaler":
        payload = load_yaml(path)
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PCScaler":
        method = str(d.get("method", "zscore")).lower()
        pca_dim = int(d.get("pca_dim")) #fails here
        eps = float(d.get("eps", 1e-8))
        dtype_str = str(d.get("dtype", "float32"))
        stats = d.get("stats", {}) or {}

        mu = sigma = median = denom = None
        if method == "zscore":
            mu = np.asarray(stats["mu"], dtype=np.float64)
            sigma = np.asarray(stats["sigma"], dtype=np.float64)
            _require(mu.shape == (pca_dim,), f"mu shape {mu.shape} != (pca_dim,)")
            _require(sigma.shape == (pca_dim,), f"sigma shape {sigma.shape} != (pca_dim,)")
        elif method == "robust":
            median = np.asarray(stats["median"], dtype=np.float64)
            denom = np.asarray(stats["denom"], dtype=np.float64)
            _require(median.shape == (pca_dim,), f"median shape {median.shape} != (pca_dim,)")
            _require(denom.shape == (pca_dim,), f"denom shape {denom.shape} != (pca_dim,)")
        elif method in ("global_l2", "maxabs"):
            pass  # статистики не потрібні
        else:
            raise ValueError(f"Unknown scaler.method: {method}")

        return cls(
            method=method,
            pca_dim=pca_dim,
            eps=eps,
            dtype_str=dtype_str,
            mu=mu, sigma=sigma,
            median=median, denom=denom,
            fingerprint=d.get("fingerprint"),
        )

    # ---- основний API ----
    def transform(self, x: ArrayLike) -> ArrayLike:
        """
        Приймає вектор (p,) або батч (N,p). Повертає того ж типу й форми.
        """
        # перевірка форми
        x_np = _as_numpy(x)
        _require(x_np.ndim in (1, 2), f"Expected 1D or 2D array, got shape={x_np.shape}")
        p = x_np.shape[-1]
        _require(p == self.pca_dim, f"Expected last dim = pca_dim={self.pca_dim}, got {p}")

        method = self.method
        eps = float(self.eps)

        if method == "zscore":
            _require(self.mu is not None and self.sigma is not None, "mu/sigma are required for zscore")
            # broadcasting: (..., p) - (p,)
            z_np = (x_np - self.mu) / (self.sigma + eps)
        elif method == "robust":
            _require(self.median is not None and self.denom is not None, "median/denom are required for robust")
            z_np = (x_np - self.median) / (self.denom + eps)
        elif method == "global_l2":
            if x_np.ndim == 1:
                norm = float(np.linalg.norm(x_np, ord=2))
                z_np = x_np / (norm + eps)
            else:
                norms = np.linalg.norm(x_np, ord=2, axis=1, keepdims=True)
                z_np = x_np / (norms + eps)
        elif method == "maxabs":
            if x_np.ndim == 1:
                m = float(np.max(np.abs(x_np)))
                z_np = x_np / (m + eps)
            else:
                m = np.max(np.abs(x_np), axis=1, keepdims=True)
                z_np = x_np / (m + eps)
        else:
            raise ValueError(f"Unknown scaler.method: {method}")

        return _same_type_like(x, z_np)


# ---------- AngleEncoder ----------

@dataclass
class AngleEncoder:
    """
    Кутове кодування:
        φ = clip((angle_max / k) * z, -angle_max, +angle_max)
    Якщо no_clip=True — повертає «сирі» (angle_max/k)*z без обрізання (для діагностики).
    """
    k: float
    angle_max: float = math.pi
    no_clip: bool = False

    def encode(self, z: ArrayLike) -> ArrayLike:
        z_np = _as_numpy(z)
        scale = float(self.angle_max) / float(self.k)
        phi_np = scale * z_np
        if not self.no_clip:
            np.clip(phi_np, -self.angle_max, self.angle_max, out=phi_np)
        return _same_type_like(z, phi_np)


# ---------- Комбінатор ----------

def encode_from_pca(
    x_pc: ArrayLike,
    stats_path: Union[str, Path],
    k: float,
    angle_max: float = math.pi,
    no_clip: bool = False,
) -> ArrayLike:
    """
    Зручний хелпер: X_PC -> z (через preprocess_stats.yaml) -> φ.
    """
    scaler = PCScaler.from_yaml(stats_path)
    z = scaler.transform(x_pc)
    ang = AngleEncoder(k=k, angle_max=angle_max, no_clip=no_clip)
    return ang.encode(z)


# ---------- (опц.) Метрики для тренування ----------

def clip_rate(z: ArrayLike, k: float) -> Tuple[float, ArrayLike]:
    """
    Повертає (overall, per_pc) для умови |z| > k.
    overall — float; per_pc — того ж типу, що і вхід (np.ndarray або torch.Tensor).
    """
    if _is_torch(z):
        zt = z  # type: ignore[assignment]
        mask = torch.gt(torch.abs(zt), float(k))  # type: ignore[attr-defined]
        overall = float(mask.float().mean().item())
        per_pc = mask.float().mean(dim=0)
        return overall, per_pc
    else:
        zn = np.asarray(z)
        mask = (np.abs(zn) > float(k))
        overall = float(mask.mean())
        per_pc = mask.mean(axis=0) if zn.ndim == 2 else np.asarray(mask, dtype=np.float32)
        return overall, per_pc


def live_rate(phi: ArrayLike, lo: float = math.pi/3, hi: float = 2*math.pi/3) -> Tuple[float, ArrayLike]:
    """
    Частка «живих» кутів: lo ≤ |φ| ≤ hi. (overall, per_pc)
    """
    if _is_torch(phi):
        pt = phi  # type: ignore[assignment]
        m = (torch.abs(pt) >= float(lo)) & (torch.abs(pt) <= float(hi))  # type: ignore[attr-defined]
        overall = float(m.float().mean().item())
        per_pc = m.float().mean(dim=0)
        return overall, per_pc
    else:
        pn = np.asarray(phi)
        m = (np.abs(pn) >= float(lo)) & (np.abs(pn) <= float(hi))
        overall = float(m.mean())
        per_pc = m.mean(axis=0) if pn.ndim == 2 else np.asarray(m, dtype=np.float32)
        return overall, per_pc


def std_phi(phi: ArrayLike) -> ArrayLike:
    """
    Стандартне відхилення кутів per-PC (уздовж осі batch).
    """
    if _is_torch(phi):
        return torch.std(phi, dim=0, unbiased=False)  # type: ignore[call-arg]
    else:
        pn = np.asarray(phi)
        if pn.ndim == 1:
            return np.array([pn.std(ddof=0)], dtype=pn.dtype)
        return pn.std(axis=0, ddof=0)


# ---------- (опц.) Перевірка z-кешу через fingerprint ----------

def validate_z_cache_from_stats(
    stats_yaml_path: Union[str, Path],
    sidecar_fp_path: Union[str, Path],
    *,
    keys: Tuple[str, ...] = ("method", "pca_dim", "eps", "seed", "stats_hash"),
) -> bool:
    """
    Порівнює fingerprint з preprocess_stats.yaml і sidecar-файла кешу (Z_*.npy.fp.yaml).
    Повертає True, якщо кеш сумісний.
    """
    stats = load_yaml(stats_yaml_path)
    expected_fp = stats.get("fingerprint", {})
    current_fp = load_yaml(sidecar_fp_path)
    return fingerprint_matches(current_fp, expected_fp, keys=keys)


# ---------- smoke-тест ----------

if __name__ == "__main__":
    # Невеликий самотест без зовнішніх файлів.
    rng = np.random.default_rng(0)
    p = 8
    N = 1024

    # згенеруємо псевдо-PCA простір із різними σ по осях
    true_sigma = np.linspace(0.5, 3.0, p)  # різні масштаби осей
    X = rng.normal(loc=0.0, scale=true_sigma, size=(N, p)).astype(np.float32)

    # сфабрикуємо "yaml-подібний" словник для Z-score
    fake_yaml = {
        "method": "zscore",
        "pca_dim": p,
        "eps": 1e-8,
        "dtype": "float32",
        "stats": {
            "mu": X.mean(axis=0).astype(np.float64).tolist(),
            "sigma": X.std(axis=0, ddof=0).astype(np.float64).tolist(),
        },
    }

    scaler = PCScaler.from_dict(fake_yaml)
    Z = scaler.transform(X)  # стандарталізуємо

    m = Z.mean(axis=0)
    s = Z.std(axis=0, ddof=0)
    print("[SMOKE] mean(z) ~ 0  ->", np.round(m, 3))
    print("[SMOKE] std(z)  ~ 1  ->", np.round(s, 3))

    ang = AngleEncoder(k=2.5, angle_max=math.pi, no_clip=False)
    PHI = ang.encode(Z)
    overall_clip, perpc_clip = clip_rate(Z, k=2.5)
    print(f"[SMOKE] clip_rate(k=2.5) overall={overall_clip:.4f}  max_pc={float(np.max(_as_numpy(perpc_clip))):.4f}")

    lo, hi = math.pi/3, 2*math.pi/3
    overall_live, _ = live_rate(PHI, lo=lo, hi=hi)
    print(f"[SMOKE] live_rate(φ in [{lo:.2f},{hi:.2f}]) overall={overall_live:.4f}")

__all__ = ["PCScaler","AngleEncoder","encode_from_pca"]