# Code/preprocess_fit_scaler.py
"""
Fit скейлера після PCA (лише на train) і, опційно, збереження Z-кешу.

Методи:
  - "zscore":   z_j = (x_j - μ_j) / (σ_j + eps)
  - "robust":   median ± MAD/IQR → шкалюємо до ≈σ (для нормалі):
                  kind="mad": denom_j = c_mad * MAD_j,      c_mad≈1.4826
                  kind="iqr": denom_j = IQR_j / c_iqr,      c_iqr≈1.349
                z_j = (x_j - median_j) / (denom_j + eps)
  - "global_l2":  z = x / (||x||_2 + eps)        (нормалізація всього вектора)
  - "maxabs":     z = x / (max(|x|) + eps)       (нормалізація всього вектора)

Виходи:
  - preprocess_stats.yaml  (метод + параметри/статистики + fingerprint)
  - (опц.) Z_*_std.npy     (кеш стандартизованих ознак) + .fp.yaml «паспорт» кешу

Примітки:
  - Fit — ТІЛЬКИ на train. Transform — train/val/test (для кешу).
  - Для "global_*" методу fit-статистики не потрібні (зберігаємо лише метадані).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from Code.utils.io_utils import (
    load_project_config, set_seed, load_npy, save_npy, save_yaml,
    assert_no_nan, as_dtype, ensure_parent_dir, build_fingerprint, save_fingerprint
)
from Code.logger import setup_logger
from Code.preprocess.transforms import PCScaler  # для побудови z-кешу 1:1 з тренуванням


# ---------- утиліти обчислення статистик (fit на train) ----------

def _fit_zscore_stats(x_train_pc: np.ndarray, eps: float) -> Dict[str, np.ndarray]:
    """Пер-осева (per-PC) μ та σ на train з захистом від нуля."""
    mu = np.mean(x_train_pc, axis=0)
    sigma = np.std(x_train_pc, axis=0, ddof=0)
    sigma = np.maximum(sigma, eps)  # захист від нуля
    return {"mu": mu, "sigma": sigma}


def _fit_robust_stats(
    x_train_pc: np.ndarray,
    kind: str,
    c_mad: float,
    c_iqr: float,
    eps: float
) -> Dict[str, np.ndarray]:
    """Робастні пер-осеві статистики: median та масштаб (MAD чи IQR, перетворені в оцінку σ)."""
    med = np.median(x_train_pc, axis=0)
    kind_l = kind.lower()
    if kind_l == "mad":
        # MAD_j = median(|x_j - med_j|)
        mad = np.median(np.abs(x_train_pc - med), axis=0)
        denom = c_mad * mad
    elif kind_l == "iqr":
        q25 = np.percentile(x_train_pc, 25, axis=0)
        q75 = np.percentile(x_train_pc, 75, axis=0)
        iqr = q75 - q25
        # σ ≈ IQR / 1.349 → denom = IQR / c_iqr
        denom = iqr / max(c_iqr, 1e-12)
    else:
        raise ValueError(f"Unknown robust.kind: {kind}")
    denom = np.maximum(denom, eps)
    return {
        "median": med,
        "denom": denom,
        "robust_kind": kind_l,
        "c_mad": c_mad,
        "c_iqr": c_iqr
    }


# ---------- допоміжне безпечне завантаження сплітів для кешу ----------

def _load_split_or_empty(path: str, p: int, logger, check_nan: bool, dtype: str) -> np.ndarray:
    """
    Завантажити .npy або повернути порожній (0, p), якщо файл відсутній.
    """
    try:
        arr = load_npy(path)
        if check_nan:
            assert_no_nan(arr, f"{Path(path).name}")
        if arr.size == 0:
            return np.empty((0, p), dtype=dtype)
        # sanity: остання вісь = p
        if arr.shape[-1] != p:
            raise ValueError(f"{path}: PCA dim mismatch: got {arr.shape[-1]} vs expected {p}")
        return arr
    except FileNotFoundError:
        logger.warning("Файл не знайдено, пропускаємо спліт: %s", path)
        return np.empty((0, p), dtype=dtype)


# ---------- основний CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Fit скейлера на train PCA-проєкціях і (опц.) збереження Z-кешу")
    ap.add_argument("--config", type=str, default="config.yaml", help="Шлях до config.yaml")
    ap.add_argument("--overwrite", action="store_true", help="Дозволити перезапис артефактів (перекриє safety.overwrite)")
    args = ap.parse_args()

    cfg = load_project_config(args.config)

    # ---------- Логер ----------
    log_path = cfg.get("logging", {}).get("preprocess_log", "./runs/preproc/preprocess.log")
    log_level = cfg.get("logging", {}).get("level", "INFO")
    logger = setup_logger(log_file=log_path, logger_name="preprocess.fit_scaler", level=log_level, use_utc=True)

    # ---------- Загальне/безпека ----------
    proj = cfg.get("project", {})
    seed = int(proj.get("seed", 42))
    set_seed(seed)

    safety = cfg.get("safety", {})
    overwrite_cfg = bool(safety.get("overwrite", True))
    overwrite = bool(args.overwrite or overwrite_cfg)
    check_nan = bool(safety.get("assert_no_nan", True))

    # ---------- PCA/шляхи ----------
    pca_dim = int(cfg.get("pca", {}).get("dim", 8))
    paths = cfg.get("paths", {})
    X_train_pca_path = paths.get("X_train_pca", "./data/X_train_pca.npy")
    X_val_pca_path   = paths.get("X_val_pca",   "./data/X_val_pca.npy")
    X_test_pca_path  = paths.get("X_test_pca",  "./data/X_test_pca.npy")
    stats_out_path   = paths.get("preprocess_stats", "./data/preprocess_stats.yaml")

    # (опц.) Z-кеш
    zc = cfg.get("z_cache", {})
    zcache_enable = bool(zc.get("enable", False))
    zcache_fingerprint = bool(zc.get("fingerprint", True))
    z_train_path = cfg["paths"].get("z_cache", {}).get("train", "./data/Z_train_std.npy")
    z_val_path   = cfg["paths"].get("z_cache", {}).get("val",   "./data/Z_val_std.npy")
    z_test_path  = cfg["paths"].get("z_cache", {}).get("test",  "./data/Z_test_std.npy")
    z_fp_suffix = ".fp.yaml"  # sidecar до .npy

    # ---------- Налаштування dtype ----------
    dtype = str(cfg.get("pixels", {}).get("dtype", "float32"))

    # ---------- Налаштування скейлера ----------
    sc = cfg.get("scaler", {})
    method = str(sc.get("method", "zscore")).lower()
    eps = float(sc.get("eps", 1e-8))
    robust_kind = str(sc.get("robust", {}).get("kind", "mad")).lower()
    c_mad = float(sc.get("robust", {}).get("c_mad", 1.4826))
    c_iqr = float(sc.get("robust", {}).get("c_iqr", 1.349))
    global_axis = str(sc.get("global", {}).get("axis", "vector")).lower()
    if global_axis != "vector":
        logger.warning("scaler.global.axis=%s не підтримується; використовуємо 'vector'.", global_axis)

    # ---------- Завантаження train PCA-проєкцій (fit тільки на train) ----------
    X_train_pc = load_npy(X_train_pca_path)
    logger.info("Завантажено X_train_pca: shape=%s, dtype=%s", tuple(X_train_pc.shape), X_train_pc.dtype)
    if check_nan:
        assert_no_nan(X_train_pc, "X_train_pca")
    # sanity: розмірність має збігатися з pca_dim
    if X_train_pc.shape[-1] != pca_dim:
        raise ValueError(
            f"PCA dim mismatch: X_train_pca has last dim {X_train_pc.shape[-1]} "
            f"but config.pca.dim={pca_dim}. Перерахуй PCA (preprocess_pca.py) або виправ config."
        )

    # ---------- FIT ----------
    stats: Dict[str, object] = {}
    low_var_indices: Tuple[np.ndarray, np.ndarray] | None = None

    if method == "zscore":
        s = _fit_zscore_stats(X_train_pc, eps=eps)
        stats.update(s)
        # Лог перших значень
        logger.info("Z-score (before): μ[0]=%.6f, σ[0]=%.6f", float(s["mu"][0]), float(s["sigma"][0]))
        # Попередження про low-variance осі
        lv_mask = s["sigma"] <= (eps * 1.000001)
        if np.any(lv_mask):
            idx = np.where(lv_mask)[0]
            logger.warning("Low-variance PC (σ≈eps) на осях: %s", idx.tolist())
    elif method == "robust":
        s = _fit_robust_stats(X_train_pc, kind=robust_kind, c_mad=c_mad, c_iqr=c_iqr, eps=eps)
        stats.update(s)
        logger.info("Robust(%s): median[0]=%.6f, denom[0]=%.6f", robust_kind, float(s["median"][0]), float(s["denom"][0]))
        lv_mask = s["denom"] <= (eps * 1.000001)
        if np.any(lv_mask):
            idx = np.where(lv_mask)[0]
            logger.warning("Low-variance PC (denom≈eps) на осях: %s", idx.tolist())
    elif method in ("global_l2", "maxabs"):
        logger.info("Global method '%s': fit-статистики відсутні (нормалізація по вектору).", method)
    else:
        raise ValueError(f"Невідомий метод скейлера: {method}")

    # ---------- Запис preprocess_stats.yaml з fingerprint ----------
    fp = build_fingerprint(
        method=method, pca_dim=pca_dim, eps=eps, seed=seed, stats=stats,
        extras={"robust_kind": robust_kind, "c_mad": c_mad, "c_iqr": c_iqr}
    )
    stats_payload: Dict[str, object] = {
        "method": method,
        "eps": eps,
        "pca_dim": pca_dim,
        "seed": seed,
        "dtype": dtype,
        "stats": {},
        "fingerprint": fp.to_dict(),
    }
    if method == "zscore":
        stats_payload["stats"] = {
            "mu": np.asarray(stats["mu"]).astype(np.float64).tolist(),
            "sigma": np.asarray(stats["sigma"]).astype(np.float64).tolist(),
        }
    elif method == "robust":
        stats_payload["stats"] = {
            "median": np.asarray(stats["median"]).astype(np.float64).tolist(),
            "denom": np.asarray(stats["denom"]).astype(np.float64).tolist(),
            "robust_kind": robust_kind,
            "c_mad": c_mad,
            "c_iqr": c_iqr,
        }
    else:
        stats_payload["stats"] = {"global_axis": global_axis}

    ensure_parent_dir(stats_out_path)
    save_yaml(stats_payload, stats_out_path, overwrite=overwrite)
    logger.info("Збережено preprocess_stats.yaml → %s", stats_out_path)

    # ---------- (Опція) побудова Z-кешу через PCScaler ----------
    if zcache_enable:
        logger.info("Z-кеш УВІМКНЕНО. Будуємо Z для train/val/test за методом '%s' через PCScaler...", method)

        # Завантажимо val/test безпечно
        X_val_pc  = _load_split_or_empty(X_val_pca_path, pca_dim, logger, check_nan, dtype)
        X_test_pc = _load_split_or_empty(X_test_pca_path, pca_dim, logger, check_nan, dtype)

        # Трансформація 1:1 як у тренуванні
        scaler = PCScaler.from_dict(stats_payload)
        Z_train = scaler.transform(X_train_pc)
        Z_val   = scaler.transform(X_val_pc)  if X_val_pc.size  > 0 else X_val_pc
        Z_test  = scaler.transform(X_test_pc) if X_test_pc.size > 0 else X_test_pc

        # Перевірки
        if check_nan:
            assert_no_nan(Z_train, "Z_train_std")
            if Z_val.size > 0:
                assert_no_nan(Z_val, "Z_val_std")
            if Z_test.size > 0:
                assert_no_nan(Z_test, "Z_test_std")

        try:
            m3 = np.round(Z_train.mean(axis=0)[:3], 6).tolist()
            s3 = np.round(Z_train.std(axis=0, ddof=0)[:3], 6).tolist()
            logger.info("Post-scale sanity (train Z): mean(z)[0:3]=%s, std(z)[0:3]=%s", m3, s3)
        except Exception:
            pass
    
        # Приведення dtype та збереження
        Z_train = as_dtype(Z_train, dtype=dtype)
        Z_val   = as_dtype(Z_val, dtype=dtype)   if Z_val.size  > 0 else Z_val
        Z_test  = as_dtype(Z_test, dtype=dtype)  if Z_test.size > 0 else Z_test

        save_npy(Z_train, z_train_path, overwrite=overwrite, dtype=dtype)
        if Z_val.size > 0:
            save_npy(Z_val,   z_val_path,   overwrite=overwrite, dtype=dtype)
        else:
            logger.info("Val-кеш пропущено (порожньо або відсутній вхід).")
        if Z_test.size > 0:
            save_npy(Z_test,  z_test_path,  overwrite=overwrite, dtype=dtype)
        else:
            logger.warning("Test-кеш пропущено (порожньо або відсутній вхід).")

        logger.info("Z-кеш збережено:\n  %s\n  %s\n  %s", z_train_path, z_val_path, z_test_path)

        # Sidecar fingerprint для кожного кешу (за налаштуванням)
        if zcache_fingerprint:
            for path, split_name in [(z_train_path, "train"), (z_val_path, "val"), (z_test_path, "test")]:
                out_path = Path(path)
                if not out_path.exists():
                    continue  # відповідний кеш не створено — пропускаємо
                sidecar = str(out_path.with_suffix(out_path.suffix + z_fp_suffix))
                fp_split = build_fingerprint(
                    method=method,
                    pca_dim=pca_dim,
                    eps=eps,
                    seed=seed,
                    stats=stats,
                    extras={"split": split_name, "dtype": dtype}
                )
                save_fingerprint(fp_split, sidecar, overwrite=overwrite)
            logger.info("Паспорти кешу (.fp.yaml) збережено поруч із .npy")

    logger.info("Готово: fit скейлера (%s) завершено.", method)


if __name__ == "__main__":
    main()
