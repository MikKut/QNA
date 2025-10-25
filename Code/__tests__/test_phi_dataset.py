# tests/test_phi_dataset.py
# -*- coding: utf-8 -*-
import json
import math
from pathlib import Path

import numpy as np
import torch
import pytest

# ---- допоміжні writer-и -------------------------------------------------

def write_preprocess_stats_yaml(path: Path, mu: np.ndarray, sigma: np.ndarray, *,
                                method: str = "zscore",
                                pca_dim: int | None = None,
                                eps: float = 1e-8,
                                dtype: str = "float32") -> None:
    """
    Створює preprocess_stats.yaml у форматі, який очікує PCScaler.from_yaml():
      {
        "method": "zscore" | "robust" | "global_l2" | "maxabs",
        "pca_dim": <int>,
        "eps": <float>,
        "dtype": "float32",
        "stats": {
          "mu": [...],
          "sigma": [...]
        }
      }
    Записуємо JSON (валідний підмножинний синтаксис YAML) — додаткових пакунків не треба.
    """
    path = Path(path)
    if pca_dim is None:
        pca_dim = int(mu.shape[0])
    data = {
        "method": method,
        "pca_dim": int(pca_dim),
        "eps": float(eps),
        "dtype": dtype,
        "stats": {
            "mu": np.asarray(mu, dtype=np.float64).tolist(),
            "sigma": np.asarray(sigma, dtype=np.float64).tolist(),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def write_phi_npz(path: Path, phi: np.ndarray, y: np.ndarray) -> None:
    path = Path(path)
    np.savez(path, phi=phi.astype(np.float32), y=y.astype(np.int64))

def write_phi_pt(path: Path, phi: np.ndarray, y: np.ndarray) -> None:
    path = Path(path)
    torch.save({"phi": torch.tensor(phi, dtype=torch.float32),
                "y": torch.tensor(y, dtype=torch.int64)}, str(path))


# ---- фікстури ------------------------------------------------------------

P = 8
NTR, NVAL, NTE = 100, 40, 30

@pytest.fixture
def tmp_preproc(tmp_path: Path):
    """
    Мінімальний препроц-каталог для v2:
    - X_{train,val,test}_pca.npy
    - y_{train,val,test}.npy
    - preprocess_stats.yaml (ключі method/pca_dim/eps/dtype/stats{mu,sigma})
    """
    root = tmp_path / "preproc"
    root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)

    Xtr = rng.normal(0, 1, size=(NTR, P)).astype(np.float32)
    Xva = rng.normal(0, 1, size=(NVAL, P)).astype(np.float32)
    Xte = rng.normal(0, 1, size=(NTE, P)).astype(np.float32)

    ytr = rng.integers(0, 10, size=(NTR,), endpoint=False, dtype=np.int64)
    yva = rng.integers(0, 10, size=(NVAL,), endpoint=False, dtype=np.int64)
    yte = rng.integers(0, 10, size=(NTE,), endpoint=False, dtype=np.int64)

    np.save(root / "X_train_pca.npy", Xtr)
    np.save(root / "X_val_pca.npy",   Xva)
    np.save(root / "X_test_pca.npy",  Xte)
    np.save(root / "y_train.npy", ytr)
    np.save(root / "y_val.npy",   yva)
    np.save(root / "y_test.npy",  yte)

    # Статистики (по train)
    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    write_preprocess_stats_yaml(root / "preprocess_stats.yaml", mu, sigma, pca_dim=P)

    return root


@pytest.fixture
def base_cfg(tmp_preproc: Path):
    """
    Базовий конфіг для v2.
    """
    return {
        "pca": {"dim": P},
        "data": {
            "train_path": str(tmp_preproc),
            "val_path":   str(tmp_preproc),
            "n_classes": 10,
            # "target_classes": [...], "use_class_suffix": True, "full_classes": 10
        },
        "angles": {"k": 2.5, "angle_max": math.pi, "no_clip": False},
        "project": {"seed": 42},
    }


# ---- тести ---------------------------------------------------------------

def test_from_xpca_cached(base_cfg):
    """X_pca + preprocess_stats.yaml (cached)."""
    from Code.datasets.phi_dataset import PhiDataset
    ds = PhiDataset(base_cfg, mode="train", cache_in_memory=True)
    assert len(ds) == NTR
    assert ds.meta["source"] == "X_pca"
    assert ds.phi.shape == (NTR, P)
    assert ds.y.shape == (NTR,)
    assert ds.phi.dtype == torch.float32
    assert ds.y.dtype == torch.int64


def test_from_xpca_lazy(base_cfg):
    """Той самий сценарій, але lazy=ON (перевіряємо transform через PCScaler)."""
    from Code.datasets.phi_dataset import PhiDataset
    ds = PhiDataset(base_cfg, mode="train", cache_in_memory=False)
    assert len(ds) == NTR
    x0, y0 = ds[0]
    assert x0.shape[0] == P
    assert x0.dtype == torch.float32
    assert y0.dtype == torch.int64
    assert ds.meta["source"] == "X_pca"


def test_from_z_cached_prioritized(tmp_preproc: Path, base_cfg):
    """Якщо є Z_* — він має пріоритезуватись над X_pca."""
    # Побудуємо Z з того ж YAML
    with open(tmp_preproc / "preprocess_stats.yaml", "r", encoding="utf-8") as f:
        d = json.load(f)
    mu = np.asarray(d["stats"]["mu"], dtype=np.float32)
    sg = np.asarray(d["stats"]["sigma"], dtype=np.float32)

    Xtr = np.load(tmp_preproc / "X_train_pca.npy")
    Xva = np.load(tmp_preproc / "X_val_pca.npy")
    Ztr = (Xtr - mu) / np.clip(sg, 1e-12, None)
    Zva = (Xva - mu) / np.clip(sg, 1e-12, None)
    np.save(tmp_preproc / "Z_train_std.npy", Ztr.astype(np.float32))
    np.save(tmp_preproc / "Z_val_std.npy",   Zva.astype(np.float32))

    from Code.datasets.phi_dataset import PhiDataset
    ds = PhiDataset(base_cfg, mode="train", cache_in_memory=True)
    assert ds.meta["source"] == "Z"
    assert ds.phi.shape == (NTR, P)


def test_target_classes_with_suffix_and_Z(tmp_preproc: Path, base_cfg):
    """
    Підмножина класів + файловий суфікс: використовуємо Z_*_cls3589.npy, y_*_cls3589.npy.
    """
    target = [3, 5, 8, 9]
    suf = "_cls3589"

    # Згенеруємо підмножину train/val:
    with open(tmp_preproc / "preprocess_stats.yaml", "r", encoding="utf-8") as f:
        d = json.load(f)
    mu = np.asarray(d["stats"]["mu"], dtype=np.float32)
    sg = np.asarray(d["stats"]["sigma"], dtype=np.float32)

    ytr = np.load(tmp_preproc / "y_train.npy")
    yva = np.load(tmp_preproc / "y_val.npy")
    Xtr = np.load(tmp_preproc / "X_train_pca.npy")
    Xva = np.load(tmp_preproc / "X_val_pca.npy")

    Ztr_all = (Xtr - mu) / np.clip(sg, 1e-12, None)
    Zva_all = (Xva - mu) / np.clip(sg, 1e-12, None)

    keep_tr = np.isin(ytr, target)
    keep_va = np.isin(yva, target)
    Ztr = Ztr_all[keep_tr]
    Zva = Zva_all[keep_va]
    ytr_sub = ytr[keep_tr]
    yva_sub = yva[keep_va]

    # Файли із суфіксом
    np.save(tmp_preproc / f"Z_train_std{suf}.npy", Ztr.astype(np.float32))
    np.save(tmp_preproc / f"Z_val_std{suf}.npy",   Zva.astype(np.float32))
    np.save(tmp_preproc / f"y_train{suf}.npy",     ytr_sub.astype(np.int64))
    np.save(tmp_preproc / f"y_val{suf}.npy",       yva_sub.astype(np.int64))

    cfg = dict(base_cfg)
    cfg["data"] = dict(cfg["data"],
                       target_classes=target,
                       use_class_suffix=True,
                       full_classes=10)

    from Code.datasets.phi_dataset import PhiDataset
    ds = PhiDataset(cfg, mode="train", cache_in_memory=True)

    # Очікування
    assert ds.meta["source"] == "Z"
    assert int(ds.y.min()) == 0
    assert int(ds.y.max()) == 3
    assert len(ds) == int(keep_tr.sum())
    assert ds.n_classes == 4


def test_target_classes_without_suffix_filters_internally(base_cfg):
    """Без суфікса: фільтрація та ремап робляться всередині (X_pca + YAML)."""
    cfg = dict(base_cfg)
    cfg["data"] = dict(cfg["data"],
                       target_classes=[0, 1],
                       use_class_suffix=False,
                       full_classes=10)
    from Code.datasets.phi_dataset import PhiDataset
    ds = PhiDataset(cfg, mode="train", cache_in_memory=True)
    assert ds.meta["source"] == "X_pca"
    assert ds.n_classes == 2
    assert int(ds.y.min()) == 0 and int(ds.y.max()) == 1
    uniq = torch.unique(ds.y).tolist()
    assert set(uniq) <= {0, 1}

def test_missing_stats_with_only_xpca_raises(tmp_preproc: Path, base_cfg):
    """Є тільки X_pca і немає ні Z, ні preprocess_stats.yaml → FileNotFoundError."""
    # Видаляємо YAML і гарантуємо відсутність Z
    for name in ["preprocess_stats.yaml", "Z_train_std.npy", "Z_val_std.npy", "Z_test_std.npy"]:
        p = tmp_preproc / name
        if p.exists():
            p.unlink()

    from Code.datasets.phi_dataset import PhiDataset
    with pytest.raises(FileNotFoundError):
        PhiDataset(base_cfg, mode="train", cache_in_memory=True)