# Code/preprocess_pca.py
"""
PCA препроцес для MNIST:
- Завантажити MNIST (train/test), масштабувати пікселі (опц.), розплющити вектори 784.
- Розбити train на train/val (стратифіковано).
- Fit PCA лише на train, transform для train/val/test (n_components = p).
- Зберегти X_*_pca.npy, y_*.npy.
- Зберегти pca_meta.npz:
    * якщо pca.save_extras=true → повний набір полів (components_, mean_, EV, EVR, singular_values_ тощо);
    * якщо false → мінімум (pca_dim, whiten, random_state, evr_sum [+ опц. перша частина EVR]).
- Логи: shapes, class counts, сумарна пояснена дисперсія, базові статистики проєкцій train-PC.

Додатково:
- pca.meta_compressed (bool, опц., дефолт=true): використовувати np.savez_compressed для pca_meta.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# локальні утиліти
from Code.utils.io_utils import (
    load_project_config, set_seed, ensure_parent_dir, save_npy, save_npz,
    assert_no_nan, as_dtype, check_overwrite
)
from ..logger import setup_logger

# Опціонально: torchvision може бути не встановлений у деяких середовищах
try:
    import torch
    from torchvision import datasets, transforms
except Exception as e:
    raise RuntimeError(
        "Цьому скрипту потрібні torchvision/torch. Встанови пакети або "
        "заміни завантаження MNIST на свій шлях до даних."
    ) from e


def _load_mnist_as_arrays(root: Path, scale_to_unit: bool, dtype: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Завантажує MNIST з torchvision і повертає (X_train, y_train, X_test, y_test) як NumPy."""
    to_tensor = transforms.ToTensor()  # дає тензор у [0,1]
    train_ds = datasets.MNIST(root=str(root), train=True, download=True, transform=to_tensor)
    test_ds  = datasets.MNIST(root=str(root), train=False, download=True, transform=to_tensor)

    X_train = train_ds.data.numpy().astype(np.float32)  # (N,28,28)
    y_train = train_ds.targets.numpy().astype(np.int64)
    X_test  = test_ds.data.numpy().astype(np.float32)
    y_test  = test_ds.targets.numpy().astype(np.int64)

    if scale_to_unit:
        X_train /= 255.0
        X_test  /= 255.0

    # flatten до 784
    Ntr, H, W = X_train.shape
    Nte = X_test.shape[0]
    X_train = X_train.reshape(Ntr, H * W)
    X_test  = X_test.reshape(Nte, H * W)

    # dtype із конфіга
    X_train = as_dtype(X_train, dtype=dtype)
    X_test  = as_dtype(X_test,  dtype=dtype)

    return X_train, y_train, X_test, y_test


def main() -> None:
    ap = argparse.ArgumentParser(description="MNIST → PCA препроцес")
    ap.add_argument("--config", type=str, default="./project.yaml", help="Шлях до project.yaml")
    ap.add_argument("--overwrite", action="store_true", help="Дозволити перезапис артефактів (перекриє safety.overwrite)")
    args = ap.parse_args()
    cfg = load_project_config(args.config)

    # ---------- Логер ----------
    log_path = cfg.get("logging", {}).get("preprocess_log", "./runs/preproc/preprocess.log")
    log_level = cfg.get("logging", {}).get("level", "INFO")
    logger = setup_logger(log_file=log_path, logger_name="preprocess", level=log_level, use_utc=True)

    # ---------- Безпека / опції ----------
    safety = cfg.get("safety", {})
    overwrite_cfg = bool(safety.get("overwrite", True))
    overwrite = bool(args.overwrite or overwrite_cfg)
    check_nan = bool(safety.get("assert_no_nan", True))

    # ---------- Загальне ----------
    proj = cfg.get("project", {})
    seed = int(proj.get("seed", 42))
    set_seed(seed)

    paths = cfg.get("paths", {})
    data_root = Path(paths.get("data_root", "./data"))

    X_train_pca_path = paths.get("X_train_pca", "./data/X_train_pca.npy")
    X_val_pca_path   = paths.get("X_val_pca",   "./data/X_val_pca.npy")
    X_test_pca_path  = paths.get("X_test_pca",  "./data/X_test_pca.npy")
    y_train_path = paths.get("y_train", "./data/y_train.npy")
    y_val_path   = paths.get("y_val",   "./data/y_val.npy")
    y_test_path  = paths.get("y_test",  "./data/y_test.npy")
    pca_meta_path = paths.get("pca_meta", "./data/pca_meta.npz")

    # запобігти випадковому перезапису
    for out_path in [X_train_pca_path, X_val_pca_path, X_test_pca_path, y_train_path, y_val_path, y_test_path, pca_meta_path]:
        check_overwrite(out_path, overwrite=overwrite)

    # ---------- Налаштування пікселів і типів ----------
    pixels = cfg.get("pixels", {})
    scale_to_unit = bool(pixels.get("scale_to_unit", True))
    dtype = str(pixels.get("dtype", "float32"))

    # ---------- Спліт ----------
    split = cfg.get("split", {})
    make_val = bool(split.get("make_val_split", True))
    stratify = bool(split.get("stratify", True))
    shuffle  = bool(split.get("shuffle", True))
    split_seed = int(split.get("split_seed", seed))
    val_size  = split.get("val_size", 10000)
    val_fraction = split.get("val_fraction", None)
    if val_fraction is not None and not make_val:
        logger.warning("split.val_fraction задано, але make_val_split=False — val не буде створено.")

    # ---------- PCA налаштування ----------
    pca_cfg = cfg.get("pca", {})
    p = int(pca_cfg.get("dim", 8))
    whiten = bool(pca_cfg.get("whiten", False))
    svd_solver = str(pca_cfg.get("svd_solver", "auto"))
    pca_random_state = int(pca_cfg.get("random_state", seed))
    save_extras = bool(pca_cfg.get("save_extras", True))
    meta_compressed = bool(pca_cfg.get("meta_compressed", True))  # нове: стислий npz

    logger.info("Початок препроцесу: seed=%d, pca_dim=%d, whiten=%s, save_extras=%s, meta_compressed=%s",
                seed, p, whiten, save_extras, meta_compressed)

    # ---------- Завантаження MNIST ----------
    X_train_raw, y_train_full, X_test_raw, y_test = _load_mnist_as_arrays(
        root=data_root, scale_to_unit=scale_to_unit, dtype=dtype
    )
    logger.info("MNIST завантажено. Train=%d, Test=%d, dim=%d", X_train_raw.shape[0], X_test_raw.shape[0], X_train_raw.shape[1])

    # ---------- Train/Val split ----------
    if make_val:
        if val_fraction is not None:
            test_size = float(val_fraction)
        else:
            test_size = float(val_size) / float(X_train_raw.shape[0])
        logger.info("Робимо валідаційний спліт: test_size=%.4f (stratify=%s, shuffle=%s, seed=%d)",
                    test_size, stratify, shuffle, split_seed)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_raw, y_train_full,
            test_size=test_size,
            random_state=split_seed,
            shuffle=shuffle,
            stratify=(y_train_full if stratify else None),
        )
    else:
        X_train, y_train = X_train_raw, y_train_full
        X_val = np.empty((0, X_train.shape[1]), dtype=X_train.dtype)
        y_val = np.empty((0,), dtype=y_train.dtype)
        logger.warning("make_val_split=False — валідаційний набір НЕ створено.")

    # ---------- PCA: fit на train, transform усіх ----------
    pca = PCA(n_components=p, whiten=whiten, svd_solver=svd_solver, random_state=pca_random_state)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_val_pca   = pca.transform(X_val)   if X_val.shape[0] > 0 else X_val
    X_test_pca  = pca.transform(X_test_raw)

    # ---------- Перевірки ----------
    if check_nan:
        for arr, name in [
            (X_train_pca, "X_train_pca"), (X_val_pca, "X_val_pca"), (X_test_pca, "X_test_pca")
        ]:
            assert_no_nan(arr, name=name)

    # ---------- Логування саніті-чеків ----------
    def _class_counts(y: np.ndarray) -> dict:
        vals, cnts = np.unique(y, return_counts=True)
        return {int(k): int(v) for k, v in zip(vals, cnts)}

    logger.info("Класи train: %s", _class_counts(y_train))
    if X_val.shape[0] > 0:
        logger.info("Класи val:   %s", _class_counts(y_val))
    logger.info("Класи test:  %s", _class_counts(y_test))

    evr_sum = float(np.sum(pca.explained_variance_ratio_))
    logger.info("Сумарна пояснена дисперсія top-%d ПК: %.4f", p, evr_sum)
    logger.info("Форми: X_train_pca=%s, X_val_pca=%s, X_test_pca=%s",
                tuple(X_train_pca.shape), tuple(X_val_pca.shape), tuple(X_test_pca.shape))

    # ---------- Збереження X/y ----------
    save_npy(X_train_pca, X_train_pca_path, overwrite=overwrite, dtype=dtype)
    save_npy(X_val_pca,   X_val_pca_path,   overwrite=overwrite, dtype=dtype)
    save_npy(X_test_pca,  X_test_pca_path,  overwrite=overwrite, dtype=dtype)
    save_npy(y_train, y_train_path, overwrite=overwrite)
    save_npy(y_val,   y_val_path,   overwrite=overwrite)
    save_npy(y_test,  y_test_path,  overwrite=overwrite)

    # ---------- Збереження PCA-метаданих ----------
    # Мінімум завжди:
    meta_min = {
        "whiten":        np.array([int(whiten)], dtype=np.int32),
        "pca_dim":       np.array([p], dtype=np.int32),
        "random_state":  np.array([pca_random_state], dtype=np.int32),
        "evr_sum":       np.array([evr_sum], dtype=np.float32),
    }

    # Якщо потрібні «важкі» поля — додаємо:
    meta_full = {}
    meta_str  = {}
    if save_extras:
        meta_full.update({
            "components_":               pca.components_.astype(np.float32, copy=False),
            "mean_":                     pca.mean_.astype(np.float32, copy=False),
            "explained_variance_":       pca.explained_variance_.astype(np.float32, copy=False),
            "explained_variance_ratio_": pca.explained_variance_ratio_.astype(np.float32, copy=False),
        })
        # singular_values_ може бути відсутнім
        sv = getattr(pca, "singular_values_", None)
        if sv is not None:
            meta_full["singular_values_"] = np.asarray(sv, dtype=np.float32)

        # рядковий solver кладемо як bytes
        meta_str["svd_solver_str"] = np.frombuffer(svd_solver.encode("utf-8"), dtype=np.uint8)

        # маленький бонус: збережемо перші кілька EVR як скаляри (навіть якщо full вимкнутий, це можна додати)
        # але тут ми вже додали весь EVR у meta_full; тож ок.

    # Об'єднати й записати (з опц. стисненням)
    arrays_to_save = {**meta_min, **meta_full, **meta_str}
    ensure_parent_dir(pca_meta_path)
    if meta_compressed:
        # напряму через numpy, щоб не міняти io_utils API
        np.savez_compressed(Path(pca_meta_path), **arrays_to_save)
    else:
        save_npz(pca_meta_path, overwrite=overwrite, **arrays_to_save)

    logger.info(
        "Артефакти збережено:\n  %s\n  %s\n  %s\n  %s\n  %s\n  %s\n  %s (extras=%s, compressed=%s)",
        X_train_pca_path, X_val_pca_path, X_test_pca_path,
        y_train_path, y_val_path, y_test_path, pca_meta_path,
        save_extras, meta_compressed
    )
    logger.info("Готово.")

if __name__ == "__main__":
    main()
