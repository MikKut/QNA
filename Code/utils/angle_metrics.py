# Code/angle_metrics.py
"""
Метрики для кутового кодування та стабільності навчання.

Що тут є:
- Z-простір (після скейлера): clip_rate_z, clip_rate_grid_z, z_stats, suggest_k_from_target_clip
- φ-простір (після AngleEncoder): saturation_rate_phi, live_rate_phi, std_phi, hist_phi
- Хелпери: compute_phi_from_z, angle_scale
- Градієнти (PyTorch): grad_stats
- EMA-трекер: EMA
- Утиліта для плоских логів: pack_metrics_dict

Працює з NumPy і, за наявності, з PyTorch без зайвих копій.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

# Опційна підтримка torch
try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

ArrayLike = Union[np.ndarray, "torch.Tensor"]


# ---------- дрібні утиліти типів ----------

def _is_torch(x: Any) -> bool:
    return _HAS_TORCH and isinstance(x, torch.Tensor)  # type: ignore[name-defined]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if _is_torch(x):
        return x.detach().cpu().numpy()  # type: ignore[union-attr]
    return np.asarray(x)


def _same_type_like(x_ref: ArrayLike, arr_np: np.ndarray) -> ArrayLike:
    if _is_torch(x_ref):
        return torch.as_tensor(arr_np, dtype=x_ref.dtype, device=x_ref.device)  # type: ignore[union-attr]
    return arr_np


def angle_scale(k: float, angle_max: float = math.pi) -> float:
    """
    Аналітична похідна некліпленого кодування: dφ/dz = angle_max / k.
    """
    return float(angle_max) / float(k)


# ---------- Z-простір: clip / статистики / пропозиція k ----------

def clip_rate_z(z: ArrayLike, k: float) -> Tuple[float, ArrayLike]:
    """
    Загальний і per-PC кліпінг для умови |z| > k.
    Повертає: (overall: float, per_pc: np.ndarray | torch.Tensor)
    """
    if _is_torch(z):
        zt = z  # type: ignore[assignment]
        mask = torch.gt(torch.abs(zt), float(k))  # type: ignore[attr-defined]
        overall = float(mask.float().mean().item())
        per_pc = mask.float().mean(dim=0) if zt.ndim == 2 else mask.float()
        return overall, per_pc
    else:
        zn = np.asarray(z)
        mask = (np.abs(zn) > float(k))
        overall = float(mask.mean())
        per_pc = mask.mean(axis=0) if zn.ndim == 2 else mask.astype(np.float32)
        return overall, per_pc


def clip_rate_grid_z(z: ArrayLike, k_list: Sequence[float]) -> Dict[str, Any]:
    """
    Рахує clip_rate_z для набору k. Повертає словник:
      {
        "k": np.array [K],
        "overall": np.array [K],
        "per_pc": np.array [K, p]  (або None, якщо z 1D)
      }
    """
    z_np = _to_numpy(z)
    is_2d = (z_np.ndim == 2)
    p = z_np.shape[-1]

    k_arr = np.asarray(list(k_list), dtype=np.float64)
    overall = np.zeros_like(k_arr, dtype=np.float64)
    per_pc = np.zeros((k_arr.size, p), dtype=np.float64) if is_2d else None

    for i, kval in enumerate(k_arr):
        mask = (np.abs(z_np) > kval)
        overall[i] = mask.mean()
        if is_2d:
            per_pc[i, :] = mask.mean(axis=0)

    return {"k": k_arr, "overall": overall, "per_pc": per_pc}


def z_stats(
    z: ArrayLike,
    *,
    quantiles: Sequence[float] = (0.25, 0.5, 0.75, 0.95, 0.99),
    use_abs_for_quantiles: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Пер-PC статистики для батча z (N,p):
      mean(z), std(z), min(z), max(z), і квантилі (за замовчуванням — для |z|).
    Повертає словник np.ndarray форми (p,).
    """
    zn = _to_numpy(z)
    if zn.ndim == 1:
        zn = zn[None, :]  # (1,p)

    mean = zn.mean(axis=0)
    std = zn.std(axis=0, ddof=0)
    zmin = zn.min(axis=0)
    zmax = zn.max(axis=0)

    base = np.abs(zn) if use_abs_for_quantiles else zn
    q_vals = np.quantile(base, q=np.asarray(quantiles, dtype=np.float64), axis=0)

    out: Dict[str, np.ndarray] = {
        "mean": mean.astype(np.float64),
        "std": std.astype(np.float64),
        "min": zmin.astype(np.float64),
        "max": zmax.astype(np.float64),
    }
    for qi, q in enumerate(quantiles):
        out[f"q{int(round(100*q)):02d}"] = q_vals[qi, :].astype(np.float64)
    return out


def suggest_k_from_target_clip(
    z: ArrayLike,
    target: float = 0.02,
) -> float:
    """
    Пропонує k*, який дає ~target для P(|z| > k).
    Для двобічної умови |z| > k достатньо взяти квантиль 1 - target по |z|.
    """
    zn = _to_numpy(z)
    flat_abs = np.abs(zn).reshape(-1)
    q = float(np.quantile(flat_abs, 1.0 - float(target)))
    return q


# ---------- φ-простір: метрики насичення / «живості» / розкиду / гістограми ----------

def compute_phi_from_z(
    z: ArrayLike,
    k: float,
    *,
    angle_max: float = math.pi,
    no_clip: bool = False,
) -> ArrayLike:
    """
    Обчислює φ = clip((angle_max/k)*z, ±angle_max) або сирі значення при no_clip=True.
    """
    z_np = _to_numpy(z)
    scale = float(angle_max) / float(k)
    phi_np = scale * z_np
    if not no_clip:
        np.clip(phi_np, -angle_max, angle_max, out=phi_np)
    return _same_type_like(z, phi_np)


def saturation_rate_phi(
    phi: ArrayLike,
    *,
    angle_max: float = math.pi,
    tol: float = 1e-7,
) -> Tuple[float, ArrayLike]:
    """
    Частка насичення (кліпу) у φ: |φ| ≈ angle_max (з допуском tol).
    Повертає (overall, per_pc).
    """
    if _is_torch(phi):
        pt = phi  # type: ignore[assignment]
        mask = torch.abs(torch.abs(pt) - float(angle_max)) <= float(tol)  # type: ignore[attr-defined]
        overall = float(mask.float().mean().item())
        per_pc = mask.float().mean(dim=0) if pt.ndim == 2 else mask.float()
        return overall, per_pc
    else:
        pn = np.asarray(phi)
        mask = np.abs(np.abs(pn) - float(angle_max)) <= float(tol)
        overall = float(mask.mean())
        per_pc = mask.mean(axis=0) if pn.ndim == 2 else mask.astype(np.float32)
        return overall, per_pc


def live_rate_phi(
    phi: ArrayLike,
    *,
    lo: float = math.pi/3,
    hi: float = 2*math.pi/3,
) -> Tuple[float, ArrayLike]:
    """
    Частка «живих» кутів: lo ≤ |φ| ≤ hi.
    Повертає (overall, per_pc).
    """
    if _is_torch(phi):
        pt = phi  # type: ignore[assignment]
        m = (torch.abs(pt) >= float(lo)) & (torch.abs(pt) <= float(hi))  # type: ignore[attr-defined]
        overall = float(m.float().mean().item())
        per_pc = m.float().mean(dim=0) if pt.ndim == 2 else m.float()
        return overall, per_pc
    else:
        pn = np.asarray(phi)
        m = (np.abs(pn) >= float(lo)) & (np.abs(pn) <= float(hi))
        overall = float(m.mean())
        per_pc = m.mean(axis=0) if pn.ndim == 2 else m.astype(np.float32)
        return overall, per_pc


def std_phi(phi: ArrayLike) -> ArrayLike:
    """
    Пер-PC стандартне відхилення φ уздовж батча.
    """
    if _is_torch(phi):
        return torch.std(phi, dim=0, unbiased=False)  # type: ignore[call-arg]
    pn = _to_numpy(phi)
    if pn.ndim == 1:
        return np.array([pn.std(ddof=0)], dtype=pn.dtype)
    return pn.std(axis=0, ddof=0)


def hist_phi(
    phi: ArrayLike,
    *,
    bins: int = 20,
    range: Tuple[float, float] = (-math.pi, math.pi),
    per_pc: bool = False,
    density: bool = False,
) -> Dict[str, Any]:
    """
    Гістограми φ.
      - per_pc=False: одна загальна гістограма по всіх значеннях φ.
      - per_pc=True: окрема гістограма на кожну PC (повертає матрицю [p, bins]).
    Повертає словник з ключами:
      {"bins": np.array [bins+1], "hist": np.array [...], "per_pc": bool}
    """
    pn = _to_numpy(phi)
    if pn.ndim == 1:
        pn = pn[None, :]  # (1,p)

    if not per_pc:
        hist, edges = np.histogram(pn.reshape(-1), bins=bins, range=range, density=density)
        return {"bins": edges, "hist": hist, "per_pc": False}
    else:
        p = pn.shape[1]
        all_hists = np.zeros((p, bins), dtype=np.float64)
        edges = None
        for j in range(p):
            hj, edges = np.histogram(pn[:, j], bins=bins, range=range, density=density)
            all_hists[j, :] = hj
        assert edges is not None
        return {"bins": edges, "hist": all_hists, "per_pc": True}


# ---------- Градієнтні метрики (PyTorch) ----------

def grad_stats(
    params_or_grads: Iterable["torch.nn.Parameter"],
    *,
    clamp_inf: bool = True,
) -> Dict[str, float]:
    """
    Агреговані метрики для градієнтів моделі (PyTorch):
      mean|g|, std|g|, l2_norm, max|g|, zero_frac.
    Передавай або params (з .grad), або самі тензори градієнтів.
    """
    if not _HAS_TORCH:
        raise RuntimeError("grad_stats requires PyTorch installed.")

    abs_vals: List[torch.Tensor] = []
    l2_sq: float = 0.0
    n_total: int = 0
    n_zeros: int = 0
    gmax: float = 0.0

    for item in params_or_grads:
        g = item.grad if hasattr(item, "grad") else item
        if g is None:
            continue
        if not torch.is_tensor(g):  # type: ignore[attr-defined]
            continue
        # на випадок inf/nan — або відкинемо, або затиснемо
        g_flat = g.detach().abs().reshape(-1)
        if clamp_inf:
            g_flat = torch.nan_to_num(g_flat, nan=0.0, posinf=1e12, neginf=1e12)  # type: ignore[attr-defined]
        abs_vals.append(g_flat)
        n = g_flat.numel()
        n_total += int(n)
        n_zeros += int((g_flat == 0).sum().item())
        gmax = max(gmax, float(g_flat.max().item()))
        l2_sq += float((g_flat ** 2).sum().item())

    if n_total == 0:
        return {"mean_abs": 0.0, "std_abs": 0.0, "l2": 0.0, "max_abs": 0.0, "zero_frac": 0.0}

    cat = torch.cat(abs_vals, dim=0)
    mean_abs = float(cat.mean().item())
    std_abs = float(cat.std(unbiased=False).item())
    l2 = math.sqrt(max(l2_sq, 0.0))
    zero_frac = float(n_zeros) / float(n_total)

    return {
        "mean_abs": mean_abs,
        "std_abs": std_abs,
        "l2": l2,
        "max_abs": gmax,
        "zero_frac": zero_frac,
    }


# ---------- EMA-трекер для згладжування метрик ----------

@dataclass
class EMA:
    beta: float = 0.9
    state: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.state is None:
            self.state = {}

    def update(self, key: str, value: float) -> float:
        """
        Оновлює EMA для ключа й повертає згладжене значення.
        """
        if key not in self.state:
            self.state[key] = float(value)
        else:
            self.state[key] = self.beta * self.state[key] + (1.0 - self.beta) * float(value)
        return self.state[key]

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        return self.state.get(key, default)


# ---------- Утиліта для плоских логів ----------

def pack_metrics_dict(prefix: str, **metrics: Any) -> Dict[str, float]:
    """
    Перетворює вкладені метрики у плоский словник із префіксом, придатний для логів.
    Приклад:
        pack_metrics_dict("train",
            loss=0.1,
            clip_overall=0.012,
            grad={"l2": 3.4, "max_abs": 0.7}
        )
    ->
        {"train/loss": 0.1, "train/clip_overall": 0.012, "train/grad.l2": 3.4, "train/grad.max_abs": 0.7}
    """
    out: Dict[str, float] = {}

    def _walk(base: str, obj: Any):
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                _walk(f"{base}.{k}" if base else str(k), v)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _walk(f"{base}[{i}]", v)
        else:
            try:
                out[f"{prefix}/{base}"] = float(obj)
            except Exception:
                pass  # пропустити нечислові

    for k, v in metrics.items():
        _walk(k, v)
    return out


# ---------- мінімальний smoke-тест ----------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, p = 2000, 8
    true_sigma = np.linspace(0.5, 3.0, p)
    Z = rng.normal(0.0, true_sigma, size=(N, p)).astype(np.float32)

    # Z-простір
    out_grid = clip_rate_grid_z(Z, [1.5, 2.0, 2.5, 3.0, 3.5])
    k_suggest = suggest_k_from_target_clip(Z, target=0.02)
    stats = z_stats(Z)
    print("[SMOKE] grid overall:", np.round(out_grid["overall"], 4))
    print("[SMOKE] k*≈", round(float(k_suggest), 3))
    print("[SMOKE] z_stats std[:3]:", np.round(stats["std"][:3], 3))

    # φ-простір
    k = 2.5
    PHI = compute_phi_from_z(Z, k=k, angle_max=math.pi, no_clip=False)
    sat_overall, _ = saturation_rate_phi(PHI)
    live_overall, _ = live_rate_phi(PHI)
    std_per_pc = std_phi(PHI)
    print(f"[SMOKE] saturation overall (k={k}):", round(sat_overall, 4))
    print(f"[SMOKE] live_rate overall (k={k}):", round(live_overall, 4))
    print("[SMOKE] std_phi[:3]:", np.round(_to_numpy(std_per_pc)[:3], 3))

    # гістограма
    H = hist_phi(PHI, bins=10, per_pc=False)
    print("[SMOKE] hist bins:", H["bins"].shape, "hist:", H["hist"].shape)
