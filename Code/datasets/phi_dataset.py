# Code/datasets/phi_dataset.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import inspect
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

from Code.utils.io_utils import load_project_config
from Code.logger import setup_logger

# У твоєму репо AngleEncoder живе у Code/transforms.py і має метод .encode(...)
try:
    from Code.preprocess.transforms import AngleEncoder  # expects .encode(Z) -> phi
except Exception:
    AngleEncoder = None  # fallback буде застосовано нижче


class PhiDataset(Dataset):
    """
    Dataset, що повертає (φ, y) для split ∈ {'train','val','test'}.

    Джерела:
      A) Готові φ у файлі .npz/.pt з ключами {'phi','y'}  (legacy-сумісність)
      B) Директорія препроцесингу з файлами:
           - Z_{split}_std.npy(.fp)  АБО  X_{split}_pca.npy
           - y_{split}.npy
           - pca_meta.npz / preprocess_stats(.npz|.npy) для (mu, sigma) якщо беремо X_pca
         → φ обчислюється на льоту: Z → AngleEncoder → φ

    Конфіг:
      - pca.dim                    : n_qubits (ширина φ)
      - data.{train,val,test}_path : шлях до файлу або директорії
      - data.n_classes             : (опц.) верхня межа класів
      - data.strict_dim            : сувора перевірка розмірності (bool, default=True)
      - angles.*                   : параметри для AngleEncoder (k, angle_max, no_clip)
      - angles.angle_max           : використовується також у fallback
      - project.seed, data.dummy_size: для dummy-режиму
    """

    # ------------------------------ INIT ---------------------------------- #
    def __init__(
        self,
        config: Dict[str, Any],
        mode: str = "train",
        path: Optional[str] = None,
        cache_in_memory: bool = True,
        dummy: Optional[bool] = None,
        logger=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mode = self._normalize_mode(mode)
        self.logger = logger or setup_logger(__name__)

        # Основні параметри
        self.n_qubits: int = int(self._cfg("pca.dim", 0))
        if self.n_qubits <= 0:
            raise ValueError("Config 'pca.dim' має бути > 0 (кількість кубітів / ширина φ).")

        self.n_classes_cfg: Optional[int] = self._cfg("data.n_classes", None)
        self.strict_dim: bool = bool(self._cfg("data.strict_dim", True))
        self.angle_max: float = float(self._cfg("angles.angle_max", np.pi))
        self.project_seed: Optional[int] = self._cfg("project.seed", None)

        # Шлях може бути файлом (.npz/.pt) або директорією препроцесингу
        self.path = path or self._cfg(f"data.{self.mode}_path", None)
        if dummy is None:
            dummy = self.path is None
        self.is_dummy = bool(dummy)

        # Публічні тензори (в cached-режимі)
        self.phi: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None

        # Lazy-поля
        self._lazy_mode: bool = False
        self._src: Optional[str] = None  # "Z" | "X_pca" | "file" | "dummy"
        self._N: int = 0
        self._y_np: Optional[np.ndarray] = None
        self._Z_np: Optional[np.ndarray] = None
        self._Xp_np: Optional[np.ndarray] = None
        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None
        self._angle_encoder: Optional[Any] = None
        self._angles_cfg: Dict[str, Any] = {}
        # lazy для контейнерів
        self._phi_np_lazy: Optional[np.ndarray] = None
        self._phi_t_lazy: Optional[torch.Tensor] = None
        self._y_t_lazy: Optional[torch.Tensor] = None

        # Завантаження
        if self.is_dummy:
            self._load_dummy()
        else:
            if not isinstance(self.path, str) or not os.path.exists(self.path):
                raise FileNotFoundError(f"[PhiDataset] Не знайдено шлях: {self.path}")

            if os.path.isdir(self.path):
                # Прямо директорія препроцесингу
                self._load_from_preproc_dir(self.path, cache_in_memory=cache_in_memory)
            else:
                ext = os.path.splitext(self.path)[1].lower()
                if ext in (".npy",) or self.path.endswith(".npy.fp"):
                    # Користувач подав окремий .npy; візьмемо його директорію
                    root_dir = os.path.dirname(self.path) or "."
                    self.logger.info(
                        "[PhiDataset] Інтерпретую '%s' як директорію препроцесингу '%s' (mode=%s).",
                        self.path, root_dir, self.mode
                    )
                    self._load_from_preproc_dir(root_dir, cache_in_memory=cache_in_memory)
                else:
                    # Контейнер з {'phi','y'}
                    self._load_file(self.path, cache_in_memory=cache_in_memory)

        self._validate_shapes_and_types()
        self._infer_or_validate_n_classes()
        self._log_basic_stats()
        self._maybe_log_angle_metrics()

    # --------------------------- LOADERS ---------------------------------- #
    def _load_from_preproc_dir(self, root_dir: str, cache_in_memory: bool = True) -> None:
        # 1) labels
        y_path = os.path.join(root_dir, f"y_{self.mode}.npy")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"[PhiDataset] Не знайдено мітки: {y_path}")
        y_np = self._load_npy(y_path, dtype=np.int64, allow_mmap=True)
        if y_np.ndim != 1:
            raise ValueError(f"[PhiDataset] y має бути 1D, отримано {y_np.shape}")

        # 2) пріоритезуємо Z
        z_path = self._prefer_file([
            os.path.join(root_dir, f"Z_{self.mode}_std.npy"),
            os.path.join(root_dir, f"Z_{self.mode}_std.npy.fp"),
        ])

        if z_path:
            if z_path.endswith(".fp"):
                self.logger.info("[PhiDataset] Завантажую memmapped NPY (.fp) через np.load(mmap_mode='r').")
            Z = self._load_npy(z_path, dtype=np.float32, allow_mmap=True)
            if Z.shape[0] != y_np.shape[0]:
                raise ValueError(f"[PhiDataset] N у Z ({Z.shape[0]}) != N у y ({y_np.shape[0]})")
            p = Z.shape[1]
            if p != self.n_qubits:
                msg = f"[PhiDataset] Z dim={p} != pca.dim(n_qubits)={self.n_qubits}"
                if self.strict_dim:
                    raise ValueError(msg)
                self.logger.warning(msg + " (продовжую)")
            source = "Z"
        else:
            x_path = os.path.join(root_dir, f"X_{self.mode}_pca.npy")
            if not os.path.exists(x_path):
                raise FileNotFoundError(
                    f"[PhiDataset] Немає ні Z_{self.mode}_std.npy(.fp), ні X_{self.mode}_pca.npy у {root_dir}"
                )
            Xp = self._load_npy(x_path, dtype=np.float32, allow_mmap=True)
            if Xp.shape[0] != y_np.shape[0]:
                raise ValueError(f"[PhiDataset] N у X_pca ({Xp.shape[0]}) != N у y ({y_np.shape[0]})")
            p = Xp.shape[1]
            if p != self.n_qubits:
                msg = f"[PhiDataset] X_pca dim={p} != pca.dim(n_qubits)={self.n_qubits}"
                if self.strict_dim:
                    raise ValueError(msg)
                self.logger.warning(msg + " (продовжую)")
            mu, sigma = self._load_pca_stats(root_dir)
            source = "X_pca"

        # 3) AngleEncoder
        angles_cfg = self._cfg("angles", {}) or {}
        self._angles_cfg = dict(angles_cfg)
        encoder, enc_desc = self._make_angle_encoder(angles_cfg)

        # 4) cache vs lazy
        if cache_in_memory:
            if source == "Z":
                phi_np = self._encode_np(Z, encoder=encoder, angles_cfg=angles_cfg, angle_max=self.angle_max)
            else:
                Z_std = (Xp - mu) / np.clip(sigma, 1e-12, None)  # type: ignore[name-defined]
                phi_np = self._encode_np(Z_std, encoder=encoder, angles_cfg=angles_cfg, angle_max=self.angle_max)

            if phi_np.ndim != 2 or phi_np.shape[1] != self.n_qubits:
                raise ValueError(f"[PhiDataset] Очікуємо φ.shape == (N,{self.n_qubits}), отримано {phi_np.shape}")
            if phi_np.shape[0] != y_np.shape[0]:
                raise ValueError(f"[PhiDataset] N у φ ({phi_np.shape[0]}) != N у y ({y_np.shape[0]})")

            self.phi = torch.from_numpy(phi_np.astype(np.float32, copy=False)).contiguous()
            self.y = torch.from_numpy(y_np.astype(np.int64, copy=False)).contiguous()
            self._lazy_mode = False
            self._src = source
            self._N = self.phi.size(0)
        else:
            # Lazy: зберігаємо лише необхідні джерела
            self._lazy_mode = True
            self._src = source
            self._N = int(y_np.shape[0])
            self._y_np = y_np
            if source == "Z":
                self._Z_np = Z
            else:
                self._Xp_np = Xp
                self._mu = mu.astype(np.float32, copy=False)      # type: ignore[union-attr]
                self._sigma = np.clip(sigma.astype(np.float32, copy=False), 1e-12, None)  # type: ignore[union-attr]
            self._angle_encoder = encoder

        # Резюме по діапазону
        finite_min, finite_max, clip_rate = self._phi_summary_cached_or_source()
        self.logger.info(
            "[PhiDataset] Побудовано φ з %s: %s | φ∈[%.3f, %.3f], clip-rate≈%.2f%% | mode=%s | cache=%s",
            self._src, enc_desc, finite_min, finite_max, 100.0 * clip_rate, self.mode, not self._lazy_mode
        )

        # Метадані
        self.meta = {
            "mode": self.mode,
            "source": self._src,
            "encoder": enc_desc,
            "n_qubits": self.n_qubits,
            "cached": not self._lazy_mode,
        }

    def _load_file(self, path: str, cache_in_memory: bool = True) -> None:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npz":
            npz = np.load(path, allow_pickle=False, mmap_mode=None if cache_in_memory else "r")
            if "phi" not in npz or "y" not in npz:
                raise KeyError(f"[PhiDataset] У {path} очікуються ключі 'phi' та 'y'.")
            phi_np = np.asarray(npz["phi"], dtype=np.float32)
            y_np = np.asarray(npz["y"], dtype=np.int64)
            if cache_in_memory:
                self.phi = torch.from_numpy(phi_np.copy()).to(dtype=torch.float32).contiguous()
                self.y = torch.from_numpy(y_np.copy()).to(dtype=torch.int64).contiguous()
                self._lazy_mode = False
                self._src = "file"
                self._N = self.phi.size(0)
            else:
                self._lazy_mode = True
                self._src = "file"
                self._N = int(y_np.shape[0])
                self._y_np = y_np
                self._phi_np_lazy = phi_np
        elif ext == ".pt":
            obj = torch.load(path, map_location="cpu")
            if not isinstance(obj, dict) or "phi" not in obj or "y" not in obj:
                raise KeyError(f"[PhiDataset] У {path} очікується dict з ключами 'phi' та 'y'.")
            phi = obj["phi"].detach().cpu().to(dtype=torch.float32).contiguous()
            y = obj["y"].detach().cpu().to(dtype=torch.int64).contiguous()
            if cache_in_memory:
                self.phi, self.y = phi, y
                self._lazy_mode = False
                self._src = "file"
                self._N = self.phi.size(0)
            else:
                self._lazy_mode = True
                self._src = "file"
                self._N = int(y.size(0))
                self._phi_t_lazy = phi
                self._y_t_lazy = y
        else:
            raise ValueError(f"[PhiDataset] Непідтримуваний формат: {ext} (очікуємо .npz або .pt)")

    def _load_dummy(self) -> None:
        N = int(self._cfg("data.dummy_size", 128))
        rng = np.random.default_rng(int(self.project_seed or 0))
        phi_np = rng.uniform(low=-self.angle_max, high=self.angle_max, size=(N, self.n_qubits)).astype(np.float32)
        y_np = rng.integers(low=0, high=int(self._cfg("data.n_classes", 10)), size=N, endpoint=False, dtype=np.int64)

        self.phi = torch.from_numpy(phi_np)
        self.y = torch.from_numpy(y_np)
        self._lazy_mode = False
        self._src = "dummy"
        self._N = N
        self.meta = {"mode": self.mode, "source": "dummy", "encoder": "uniform", "n_qubits": self.n_qubits, "cached": True}

    # --------------------------- HELPERS ---------------------------------- #
    def _normalize_mode(self, mode: str) -> str:
        m = str(mode).lower()
        if m in {"valid", "validation", "dev"}:
            return "val"
        if m not in {"train", "val", "test"}:
            raise ValueError(f"[PhiDataset] Невідомий split/mode: {mode}")
        return m

    def _prefer_file(self, candidates: List[str]) -> Optional[str]:
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _load_npy(self, path: str, dtype, allow_mmap: bool = True) -> np.ndarray:
        mmap_mode = "r" if allow_mmap else None
        arr = np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"[PhiDataset] Очікувався numpy.ndarray у {path}, отримано {type(arr)}")
        return np.asarray(arr, dtype=dtype)

    def _load_pca_stats(self, root_dir: str) -> tuple[np.ndarray, np.ndarray]:
        # 1) preprocess_stats.npz
        stats_npz = os.path.join(root_dir, "preprocess_stats.npz")
        if os.path.exists(stats_npz):
            obj = np.load(stats_npz, allow_pickle=False)
            mu = obj.get("pca_mu", None); sigma = obj.get("pca_sigma", None)
            if mu is not None and sigma is not None:
                return np.asarray(mu, np.float32), np.asarray(sigma, np.float32)

        # 2) preprocess_stats.npy (dict)
        stats_npy = os.path.join(root_dir, "preprocess_stats.npy")
        if os.path.exists(stats_npy):
            try:
                d = np.load(stats_npy, allow_pickle=True).item()
                mu = d.get("pca_mu", None); sigma = d.get("pca_sigma", None)
                if mu is not None and sigma is not None:
                    return np.asarray(mu, np.float32), np.asarray(sigma, np.float32)
            except Exception:
                pass

        # 3) pca_meta.npz
        meta_path = os.path.join(root_dir, "pca_meta.npz")
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=False)
            mu = None; sigma = None
            for k in ("mu", "mean", "pca_mu"):
                if k in meta: mu = meta[k]; break
            for k in ("sigma", "std", "pca_sigma"):
                if k in meta: sigma = meta[k]; break
            if mu is not None and sigma is not None:
                return np.asarray(mu, np.float32), np.asarray(sigma, np.float32)

        raise FileNotFoundError("[PhiDataset] Не знайдено (mu, sigma) у preprocess_stats(.npz|.npy) чи pca_meta.npz")

    def _make_angle_encoder(self, angles_cfg: Dict[str, Any]):
        """Створює AngleEncoder або повертає None + опис. Підтримує лише відомі kwargs."""
        if AngleEncoder is None:
            return None, "fallback(angle_max-aware)"

        # Дізнаємось підтримувані kwargs з підпису __init__
        sig = inspect.signature(AngleEncoder.__init__)
        allowed = {p.name for p in sig.parameters.values()
                   if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        allowed.discard("self")

        enc_kwargs = {k: v for k, v in angles_cfg.items() if k in allowed}
        dropped = {k: v for k, v in angles_cfg.items() if k not in allowed}
        if dropped:
            self.logger.warning(
                "[PhiDataset] AngleEncoder: ігнорую невідомі kwargs: %s; дозволені=%s",
                list(dropped.keys()), sorted(allowed)
            )

        try:
            enc = AngleEncoder(**enc_kwargs)
        except TypeError as e:
            self.logger.error("AngleEncoder init провалився: kwargs=%s; signature=%s", enc_kwargs, sig)
            raise
        return enc, f"AngleEncoder({enc_kwargs})"

    def _encode_np(self, Z: np.ndarray, *, encoder: Any, angles_cfg: Dict[str, Any], angle_max: float) -> np.ndarray:
        """Векторизоване кодування Z → φ. Використовує AngleEncoder.encode/transform/forward або fallback."""
        if encoder is not None:
            if hasattr(encoder, "encode"):
                phi_np = encoder.encode(Z)
            elif hasattr(encoder, "transform"):
                phi_np = encoder.transform(Z)
            else:
                phi_np = encoder.forward(Z)
        else:
            # Fallback: основний шлях φ = clip((angle_max / k) * Z, ±angle_max)
            if "k" in angles_cfg:
                k = float(angles_cfg.get("k", 2.5))
                phi_np = np.clip((angle_max / k) * Z, -angle_max, angle_max)
            else:
                # запасний варіант
                alpha = float(angles_cfg.get("alpha", angle_max / 3))
                clip_sigmas = float(angles_cfg.get("clip_sigmas", 3.0))
                Zc = np.clip(Z, -clip_sigmas, +clip_sigmas)
                phi_np = np.clip(alpha * (Zc / clip_sigmas), -angle_max, angle_max)
        return np.asarray(phi_np, dtype=np.float32)

    def _phi_summary_cached_or_source(self) -> Tuple[float, float, float]:
        """(min, max, clip_rate) для φ з кешу або по семплу джерела."""
        if self.phi is not None:
            pn = self.phi.detach().cpu().numpy()
            finite = np.isfinite(pn)
            min_v = float(np.min(pn[finite])) if finite.any() else float("nan")
            max_v = float(np.max(pn[finite])) if finite.any() else float("nan")
            clip_rate = float(np.mean((pn <= -self.angle_max + 1e-6) | (pn >= self.angle_max - 1e-6)))
            return min_v, max_v, clip_rate

        # Lazy: семпл до 1024 рядків
        K = min(self._N, 1024)
        if K == 0:
            return 0.0, 0.0, 0.0
        idx = np.linspace(0, self._N - 1, num=K, dtype=int)

        if self._src == "file":
            if self._phi_np_lazy is not None:
                pn = np.asarray(self._phi_np_lazy)[idx]
            else:
                pt = self._phi_t_lazy
                pn = pt[idx].detach().cpu().numpy()  # type: ignore[index]
        else:
            if self._src == "Z":
                Zs = np.asarray(self._Z_np)[idx]
            else:
                Xs = np.asarray(self._Xp_np)[idx]
                Zs = (Xs - self._mu) / self._sigma  # type: ignore[operator]
            pn = self._encode_np(Zs, encoder=self._angle_encoder, angles_cfg=self._angles_cfg, angle_max=self.angle_max)

        finite = np.isfinite(pn)
        min_v = float(np.min(pn[finite])) if finite.any() else float("nan")
        max_v = float(np.max(pn[finite])) if finite.any() else float("nan")
        clip_rate = float(np.mean((pn <= -self.angle_max + 1e-6) | (pn >= self.angle_max - 1e-6)))
        return min_v, max_v, clip_rate

    # ----------------------- VALIDATION / LOGGING ------------------------- #
    def _validate_shapes_and_types(self) -> None:
        if self._lazy_mode:
            if self._N <= 0:
                raise ValueError("[PhiDataset] Lazy-режим, але N <= 0.")
            return

        if not isinstance(self.phi, torch.Tensor) or not isinstance(self.y, torch.Tensor):
            raise TypeError("[PhiDataset] 'phi' та 'y' мають бути torch.Tensor у cached-режимі.")

        if self.phi.ndim != 2 or self.phi.size(1) != self.n_qubits:
            raise ValueError(f"[PhiDataset] Очікуємо phi.shape == (N, {self.n_qubits}), отримано {tuple(self.phi.shape)}")
        if self.y.ndim != 1 or self.y.size(0) != self.phi.size(0):
            raise ValueError(f"[PhiDataset] Очікуємо y.shape == (N,), отримано {tuple(self.y.shape)}; Nphi={self.phi.size(0)}")

        if self.phi.dtype != torch.float32:
            self.logger.warning("[PhiDataset] Приводжу phi.dtype %s → float32", str(self.phi.dtype))
            self.phi = self.phi.to(dtype=torch.float32)
        if self.y.dtype != torch.int64:
            self.logger.warning("[PhiDataset] Приводжу y.dtype %s → int64", str(self.y.dtype))
            self.y = self.y.to(dtype=torch.int64)

    def _infer_or_validate_n_classes(self) -> None:
        if self._lazy_mode:
            if self._y_np is not None:
                max_label = int(np.max(self._y_np)) if self._N > 0 else -1
            elif self._y_t_lazy is not None:
                max_label = int(self._y_t_lazy.max().item()) if self._N > 0 else -1
            else:
                max_label = -1
        else:
            max_label = int(torch.as_tensor(self.y.max()).item()) if self.y.numel() > 0 else -1

        self.n_classes = int(self.n_classes_cfg) if self.n_classes_cfg else max_label + 1
        if self.n_classes_cfg is not None and max_label >= int(self.n_classes_cfg):
            raise ValueError(f"[PhiDataset] Значення y виходить за межі n_classes={self.n_classes_cfg}: max(y)={max_label}")
        if self.n_classes <= 0:
            raise ValueError("[PhiDataset] n_classes має бути > 0.")

    def _log_basic_stats(self) -> None:
        N = int(self._N if self._lazy_mode else self.phi.size(0))  # type: ignore[union-attr]
        # Розподіл класів
        try:
            if self._lazy_mode and self._y_np is not None:
                binc = np.bincount(self._y_np, minlength=self.n_classes)
                counts = [int(x) for x in binc.tolist()]
            elif self._lazy_mode and self._y_t_lazy is not None:
                binc = torch.bincount(self._y_t_lazy, minlength=self.n_classes)
                counts = [int(x) for x in binc.tolist()]
            else:
                with torch.no_grad():
                    binc = torch.bincount(self.y, minlength=self.n_classes)  # type: ignore[arg-type]
                    counts = [int(x) for x in binc.tolist()]
        except Exception:
            counts = []

        # Діапазон φ
        if self._lazy_mode:
            min_v, max_v, _ = self._phi_summary_cached_or_source()
            mean_abs = float("nan")
        else:
            with torch.no_grad():
                mean_abs = float(self.phi.abs().mean().item())  # type: ignore[union-attr]
            pn = self.phi.detach().cpu().numpy()  # type: ignore[union-attr]
            finite = np.isfinite(pn)
            min_v = float(np.min(pn[finite])) if finite.any() else float("nan")
            max_v = float(np.max(pn[finite])) if finite.any() else float("nan")

        self.logger.info(
            "[PhiDataset] mode=%s | N=%d, n_qubits=%d, n_classes=%d | |phi|_min=%.5f, |phi|_max=%.5f, |phi|_mean≈%s",
            self.mode, N, self.n_qubits, self.n_classes, min_v, max_v,
            f"{mean_abs:.5f}" if mean_abs == mean_abs else "—"
        )
        if counts:
            self.logger.info("[PhiDataset] Розподіл класів: %s", counts)

    def _maybe_log_angle_metrics(self) -> None:
        try:
            min_v, max_v, clip_rate = self._phi_summary_cached_or_source()
            self.logger.info(
                "[PhiDataset] φ clip-rate≈%.4f (|φ| близько angle_max=%.5f), діапазон≈[%.5f, %.5f]",
                clip_rate, self.angle_max, min_v, max_v
            )
        except Exception as e:
            self.logger.warning("[PhiDataset] Не вдалося порахувати метрики φ: %s", str(e))

    # ------------------------------- API ---------------------------------- #
    def __len__(self) -> int:
        return int(self._N if self._lazy_mode else self.phi.size(0))  # type: ignore[union-attr]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._lazy_mode:
            return self.phi[idx], self.y[idx]  # type: ignore[index]

        # Lazy-гілки
        if self._y_np is not None:
            y_t = torch.as_tensor(self._y_np[idx], dtype=torch.int64)
        else:
            y_t = self._y_t_lazy[idx].to(dtype=torch.int64)  # type: ignore[index]

        if self._src == "file":
            if self._phi_np_lazy is not None:
                phi_np = np.asarray(self._phi_np_lazy[idx]).astype(np.float32, copy=False)
                phi_t = torch.from_numpy(phi_np)
            else:
                phi_t = self._phi_t_lazy[idx].to(dtype=torch.float32)  # type: ignore[index]
            return phi_t, y_t

        # Обчислення φ на льоту
        if self._src == "Z":
            z = np.asarray(self._Z_np[idx]).astype(np.float32, copy=False)
        else:
            xp = np.asarray(self._Xp_np[idx]).astype(np.float32, copy=False)
            z = (xp - self._mu) / self._sigma  # type: ignore[operator]

        if self._angle_encoder is not None:
            enc = self._angle_encoder
            row = z[None, :]
            if hasattr(enc, "encode"):
                phi_np = enc.encode(row).astype(np.float32, copy=False)[0]
            elif hasattr(enc, "transform"):
                phi_np = enc.transform(row).astype(np.float32, copy=False)[0]
            else:
                phi_np = enc.forward(row).astype(np.float32, copy=False)[0]
        else:
            angles_cfg = self._angles_cfg
            if "k" in angles_cfg:
                k = float(angles_cfg.get("k", 2.5))
                phi_np = np.clip((self.angle_max / k) * z, -self.angle_max, self.angle_max)
            else:
                alpha = float(angles_cfg.get("alpha", self.angle_max / 3))
                clip_sigmas = float(angles_cfg.get("clip_sigmas", 3.0))
                zc = np.clip(z, -clip_sigmas, +clip_sigmas)
                phi_np = np.clip(alpha * (zc / clip_sigmas), -self.angle_max, self.angle_max)

        return torch.from_numpy(phi_np), y_t

    def class_weights(self, normalize: bool = True) -> torch.Tensor:
        if self._lazy_mode:
            if self._y_np is not None:
                binc = np.bincount(self._y_np, minlength=self.n_classes).astype(np.float32)
                binc[binc <= 0] = 1.0
                w = 1.0 / binc
                if normalize:
                    w = w * (self.n_classes / w.sum())
                return torch.from_numpy(w)
            if self._y_t_lazy is not None:
                with torch.no_grad():
                    binc = torch.bincount(self._y_t_lazy, minlength=self.n_classes).to(torch.float32)
                    binc = torch.where(binc > 0, binc, torch.ones_like(binc))
                    w = 1.0 / binc
                    if normalize:
                        w = w * (self.n_classes / w.sum())
                return w
            return torch.ones(self.n_classes, dtype=torch.float32)

        with torch.no_grad():
            binc = torch.bincount(self.y, minlength=self.n_classes).to(torch.float32)  # type: ignore[arg-type]
            binc = torch.where(binc > 0, binc, torch.ones_like(binc))
            w = 1.0 / binc
            if normalize:
                w = w * (self.n_classes / w.sum())
        return w

    # ------------------------------ CONFIG -------------------------------- #
    def _cfg(self, path: str, default: Any = None) -> Any:
        cur: Any = self.config
        for key in path.split("."):
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    @classmethod
    def from_config_path(
        cls, config_path: str, mode: str = "train", path: Optional[str] = None, **kw,
    ) -> "PhiDataset":
        cfg = load_project_config(config_path)
        return cls(cfg, mode=mode, path=path, **kw)
