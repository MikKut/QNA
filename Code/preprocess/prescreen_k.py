# Code/preprocess/prescreen_k.py
"""
Prescreen k: швидко оцінює k для кутового кодування на обраному спліті,
використовуючи z-кеш (з fingerprint-перевіркою) або обчислення z на льоту.

Вихід:
- друк у консоль: рекомендований k, емпіричний k*, clip_rate на k
- YAML-звіт у paths.diag_dir: prescreen_k_{split}.yaml
- (опц.) --write-config: оновлює angles.k у config.yaml

Використання (приклад):
  python Code/prescreen_k.py \
      --config config.yaml \
      --split auto \
      --k-grid 1.5,2.0,2.5,3.0,3.5 \
      --use-cache auto \
      --target-clip 0.02 \
      --write-config
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from Code.utils.io_utils import (
    load_project_config, load_yaml, save_yaml, load_npy, save_json,
    ensure_dir, assert_no_nan, class_suffix
)
from Code.logger import setup_logger
from Code.preprocess.transforms import PCScaler, validate_z_cache_from_stats
from Code.utils.angle_metrics import clip_rate_grid_z, suggest_k_from_target_clip


# ---------------------- CLI ----------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prescreen k (кутовий бюджет) за валід/трейн split")
    ap.add_argument("--config", type=str, default="config.yaml", help="Шлях до config.yaml")
    ap.add_argument("--split", type=str, default="auto", help="train|val|test|auto (auto = val якщо є, інакше train)")
    ap.add_argument("--k-grid", type=str, default="1.5,2.0,2.5,3.0,3.5", help="Сітка k, через кому")
    ap.add_argument("--use-cache", type=str, default="auto", choices=["auto", "yes", "no"], help="Читати Z_*_std.npy")
    ap.add_argument("--target-clip", type=float, default=0.02, help="Цільовий clip_rate(|z|>k)")
    ap.add_argument("--write-config", action="store_true", help="Оновити angles.k у config.yaml рекомендованим значенням")
    return ap.parse_args()


# ---------------------- helpers ----------------------

def _paths_for_split(cfg: Dict, split: str) -> Tuple[str, str, str]:
    p = cfg.get("paths", {})
    zc = p.get("z_cache", {})
    if split == "train":
        return (p.get("X_train_pca", "./data/X_train_pca.npy"),
                zc.get("train", "./data/Z_train_std.npy"),
                p.get("y_train", "./data/y_train.npy"))
    if split == "val":
        return (p.get("X_val_pca", "./data/X_val_pca.npy"),
                zc.get("val", "./data/Z_val_std.npy"),
                p.get("y_val", "./data/y_val.npy"))
    if split == "test":
        return (p.get("X_test_pca", "./data/X_test_pca.npy"),
                zc.get("test", "./data/Z_test_std.npy"),
                p.get("y_test", "./data/y_test.npy"))
    raise ValueError(f"Unknown split: {split}")


def _load_or_compute_z(
    cfg: Dict,
    stats_yaml_path: str,
    scaler: PCScaler,
    split: str,
    use_cache: Literal["auto", "yes", "no"],
    check_nan: bool,
    logger,
) -> Optional[np.ndarray]:
    X_path, Z_path, y_path = _paths_for_split(cfg, split)
    pca_dim = int(cfg.get("pca", {}).get("dim", 8))

    # 0) зчитаємо y (для маски класів — і при Z-кеші, і при X_pca)
    target_classes = cfg.get("data", {}).get("target_classes", None)
    y = None
    if target_classes:
        if not Path(y_path).exists():
            logger.warning("Відсутній y_* для split=%s: %s — фільтрування класів неможливе.", split, y_path)
        else:
            y = load_npy(y_path).astype(np.int64, copy=False)

    # 1) спроба взяти Z з кешу
    if use_cache in ("auto", "yes"):
        sidecar = str(Path(Z_path).with_suffix(Path(Z_path).suffix + ".fp.yaml"))
        if Path(Z_path).exists():
            ok = True
            if Path(sidecar).exists():
                try:
                    ok = validate_z_cache_from_stats(stats_yaml_path, sidecar)
                except Exception as e:
                    logger.warning("Fingerprint перевірка кешу впала (%s). Ігноруємо кеш.", e)
                    ok = False
            elif use_cache == "auto":
                ok = False
            if ok or use_cache == "yes":
                try:
                    Z = load_npy(Z_path)
                    if check_nan:
                        assert_no_nan(Z, f"{Path(Z_path).name}")
                    if Z.ndim == 2 and Z.shape[1] == pca_dim:
                        logger.info("Використовуємо Z-кеш для split=%s: %s", split, Z_path)
                        # ⬇️ застосуємо фільтр класів, якщо треба
                        if y is not None and isinstance(target_classes, (list, tuple)) and len(target_classes) < 10:
                            mask = np.isin(y, list(map(int, target_classes)))
                            Z = Z[mask]
                            logger.info("Фільтр класів застосовано до Z: залишилось %d рядків.", Z.shape[0])
                        return np.asarray(Z)
                    else:
                        logger.warning("Форма Z не відповідає pca_dim; перераховуємо на льоту.")
                except FileNotFoundError:
                    pass

    # 2) на льоту від X_pca
    if not Path(X_path).exists():
        logger.warning("Відсутній X_*_pca для split=%s: %s", split, X_path)
        return None
    X = load_npy(X_path)
    if X.size == 0:
        logger.warning("Порожній split=%s (0 рядків).", split)
        return None
    if X.ndim != 2 or X.shape[1] != pca_dim:
        raise ValueError(f"{X_path}: очікувана форма (*,{pca_dim}), отримано {X.shape}")
    if check_nan:
        assert_no_nan(X, f"{Path(X_path).name}")

    # ⬇️ фільтр класів для X, якщо треба
    if y is not None and isinstance(target_classes, (list, tuple)) and len(target_classes) < 10:
        mask = np.isin(y, list(map(int, target_classes)))
        X = X[mask]
        logger.info("Фільтр класів застосовано до X_pca: залишилось %d рядків.", X.shape[0])

    Z = scaler.transform(X)
    if check_nan:
        assert_no_nan(Z, f"Z_{split}_std")
    return np.asarray(Z)

def _choose_k_from_grid(k_arr: np.ndarray, overall_clip: np.ndarray, target: float) -> float:
    ok = np.where(overall_clip < float(target))[0]
    if ok.size > 0:
        return float(k_arr[int(ok[0])])
    return float(k_arr[int(np.argmin(overall_clip))])


# ---------------------- main ----------------------

def main() -> None:
    args = _parse_args()
    cfg = load_project_config(args.config)

    # логер
    log_path = cfg.get("logging", {}).get("preprocess_log", "./runs/preproc/preprocess.log")
    log_level = cfg.get("logging", {}).get("level", "INFO")
    logger = setup_logger(log_file=log_path, logger_name="preprocess.prescreen_k", level=log_level, use_utc=True)

    # базове
    stats_path = cfg.get("paths", {}).get("preprocess_stats", "./data/preprocess_stats.yaml")
    diag_dir = cfg.get("paths", {}).get("diag_dir", "./runs/preproc/diag")
    ensure_dir(diag_dir)

    safety = cfg.get("safety", {})
    check_nan = bool(safety.get("assert_no_nan", True))

    # split
    split = args.split.lower()
    if split == "auto":
        x_val = cfg.get("paths", {}).get("X_val_pca", "./data/X_val_pca.npy")
        split = "val" if Path(x_val).exists() else "train"

    # завантажуємо скейлер
    stats_yaml = load_yaml(stats_path)
    scaler = PCScaler.from_dict(stats_yaml)

    # дістаємо z
    Z = _load_or_compute_z(
        cfg=cfg,
        stats_yaml_path=stats_path,
        scaler=scaler,
        split=split,
        use_cache=args.use_cache,  # type: ignore[arg-type]
        check_nan=check_nan,
        logger=logger,
    )
    if Z is None:
        raise SystemExit(f"[prescreen_k] Немає даних для split={split}. Спочатку згенеруй PCA та/або scaler.")

    # грид по k
    k_grid = [float(x) for x in args.k_grid.split(",") if x.strip()]
    grid = clip_rate_grid_z(Z, k_grid)
    k_arr = np.asarray(grid["k"], dtype=float)
    overall = np.asarray(grid["overall"], dtype=float)

    # вибір k
    k_rec = _choose_k_from_grid(k_arr, overall, target=args.target_clip)
    k_emp = float(suggest_k_from_target_clip(Z, target=args.target_clip))
    clip_at_krec = float(overall[int(np.argmin(np.abs(k_arr - k_rec)))])

    # звіт
    out_yaml = Path(diag_dir) / f"prescreen_k_{split}.yaml"
    payload = {
        "split": split,
        "target_clip": float(args.target_clip),
        "k_grid": [float(x) for x in k_arr],
        "k_recommended": float(k_rec),
        "k_empirical": float(k_emp),
        "clip_overall_at_k_recommended": clip_at_krec,
    }
    save_yaml(payload, str(out_yaml), True)
    print(f"[prescreen_k] split={split}  k_rec={k_rec:.3f}  k_emp~={k_emp:.3f}  clip@k_rec={clip_at_krec:.4f}")
    print(f"[prescreen_k] report → {out_yaml}")

    # (опц.) оновити angles.k в конфігу
    if args.write_config:
        
        cfg_mod = load_yaml(args.config)
        angles = dict(cfg_mod.get("angles", {}))
        angles["k"] = float(k_rec)
        cfg_mod["angles"] = angles
        save_yaml(cfg_mod, args.config)
        print(f"[prescreen_k] config.yaml оновлено: angles.k={k_rec:.3f}")


if __name__ == "__main__":
    import numpy as np  # noqa: F401  (гарантуємо наявність np)
    main()
