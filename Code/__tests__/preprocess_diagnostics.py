# Code/preprocess_diagnostics.py
"""
Офлайнова діагностика препроцесу після PCA:
- читає preprocess_stats.yaml;
- бере z з кешу (з fingerprint-верифікацією) або обчислює на льоту через PCScaler;
- рахує метрики:
    * у z-просторі: clip_rate для сітки k, базові статистики per-PC;
    * у φ-просторі: saturation/live/std для вибраних k (типово — вся сітка);
- пропонує k* під цільовий clip_rate;
- зберігає таблиці/звіти/гістограми/графіки.

Запуск (приклад):
    python Code/preprocess_diagnostics.py \
        --config ./project.yaml \
        --splits val,train \
        --k-grid 1.5,2.0,2.5,3.0,3.5 \
        --use-cache auto \
        --hist-bins 40

Залежності: numpy, pyyaml; matplotlib (опційно для графіків); torch — не обов'язково.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from Code.utils.io_utils import (
    save_csv, 
    load_project_config, load_yaml, load_npy, save_yaml, save_json, save_npz,
    ensure_dir, ensure_parent_dir, assert_no_nan
)
from Code.logger import setup_logger
from Code.preprocess.transforms import PCScaler, validate_z_cache_from_stats
from Code.utils.angle_metrics import (
    clip_rate_grid_z, z_stats, suggest_k_from_target_clip,
    compute_phi_from_z, saturation_rate_phi, live_rate_phi, std_phi, hist_phi
)


# ---------------------------- CLI / Config ----------------------------

@dataclass
class CLIArgs:
    config: str
    splits: List[str]
    k_grid: List[float]
    use_cache: Literal["auto", "yes", "no"]
    hist_bins: int
    no_plots: bool
    target_clip: float


def _parse_args() -> CLIArgs:
    ap = argparse.ArgumentParser(description="Діагностика препроцесу після PCA")
    ap.add_argument("--config", type=str, default="./project.yaml", help="Шлях до project.yaml")
    ap.add_argument("--splits", type=str, default="", help="Список сплітів через кому (наприклад: val,train)")
    ap.add_argument("--k-grid", type=str, default="1.5,2.0,2.5,3.0,3.5", help="Сітка k (через кому)")
    ap.add_argument("--use-cache", type=str, default="auto", choices=["auto", "yes", "no"], help="Використовувати Z-кеш")
    ap.add_argument("--hist-bins", type=int, default=40, help="К-сть бінів для гістограм φ")
    ap.add_argument("--no-plots", action="store_true", help="Не зберігати PNG-графіки")
    ap.add_argument("--target-clip", type=float, default=0.02, help="Цільовий clip_rate для вибору k*")
    args = ap.parse_args()

    k_grid = [float(x) for x in args.k_grid.split(",") if x.strip()]
    splits = [s.strip().lower() for s in args.splits.split(",") if s.strip()]
    if args.use_cache not in ("auto", "yes", "no"):
        raise ValueError("--use-cache must be in {'auto','yes','no'}")

    return CLIArgs(
        config=args.config,
        splits=splits,
        k_grid=k_grid,
        use_cache=args.use_cache,  # type: ignore[arg-type]
        hist_bins=args.hist_bins,
        no_plots=bool(args.no_plots),
        target_clip=float(args.target_clip),
    )


# ---------------------------- I/O helpers ----------------------------

def _plot_clip_over_k(out_png: str, k: np.ndarray, overall: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt  # lazy import
        ensure_parent_dir(out_png)
        plt.figure()
        plt.plot(k, overall, marker="o")
        plt.xlabel("k")
        plt.ylabel("clip_rate (|z| > k)")
        plt.title("Clip rate vs k")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.savefig(out_png, bbox_inches="tight", dpi=140)
        plt.close()
    except Exception:
        # безпечно пропускаємо, якщо matplotlib недоступний
        pass


def _plot_phi_hist(out_png: str, bins: np.ndarray, hist: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
        ensure_parent_dir(out_png)
        centers = 0.5 * (bins[1:] + bins[:-1])
        plt.figure()
        plt.bar(centers, hist, width=(bins[1]-bins[0]))
        plt.xlabel("φ")
        plt.ylabel("count")
        plt.title("Histogram of φ")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.savefig(out_png, bbox_inches="tight", dpi=140)
        plt.close()
    except Exception:
        pass


def _plot_std_phi_per_pc(out_png: str, std_per_pc: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
        ensure_parent_dir(out_png)
        plt.figure()
        x = np.arange(std_per_pc.size)
        plt.bar(x, std_per_pc)
        plt.xlabel("PC index")
        plt.ylabel("std(φ)")
        plt.title("std(φ) per PC")
        plt.grid(True, axis="y", linestyle="--", alpha=0.3)
        plt.savefig(out_png, bbox_inches="tight", dpi=140)
        plt.close()
    except Exception:
        pass


# ---------------------------- Core helpers ----------------------------

def _paths_for_split(cfg: Dict, split: str) -> Tuple[str, str]:
    """Повертає (X_split_pca.npy, Z_split_std.npy) шляхи."""
    p = cfg.get("paths", {})
    zc = p.get("z_cache", {})
    if split == "train":
        return p.get("X_train_pca", "./data/X_train_pca.npy"), zc.get("train", "./data/Z_train_std.npy")
    if split == "val":
        return p.get("X_val_pca", "./data/X_val_pca.npy"), zc.get("val", "./data/Z_val_std.npy")
    if split == "test":
        return p.get("X_test_pca", "./data/X_test_pca.npy"), zc.get("test", "./data/Z_test_std.npy")
    raise ValueError(f"Unknown split: {split}")


def _load_or_compute_z_for_split(
    cfg: Dict,
    scaler: PCScaler,
    stats_yaml_path: str,
    split: str,
    use_cache: Literal["auto", "yes", "no"],
    check_nan: bool,
    logger,
) -> Optional[np.ndarray]:
    """
    Повертає z для спліта (N,p) або None, якщо вхідні файли відсутні.
    """
    X_path, Z_path = _paths_for_split(cfg, split)
    pca_dim = int(cfg.get("pca", {}).get("dim", 8))

    # 1) use-cache?
    if use_cache in ("auto", "yes"):
        z_sidecar = str(Path(Z_path).with_suffix(Path(Z_path).suffix + ".fp.yaml"))
        if Path(Z_path).exists():
            ok_fp = True
            if Path(z_sidecar).exists():
                try:
                    ok_fp = validate_z_cache_from_stats(stats_yaml_path, z_sidecar)
                except Exception as e:
                    logger.warning("Fingerprint перевірка кешу впала (%s). Ігноруємо кеш.", e)
                    ok_fp = False
            elif use_cache == "auto":
                # у режимі auto, якщо нема сайдкару — краще порахувати на льоту
                ok_fp = False

            if ok_fp or use_cache == "yes":
                try:
                    Z = load_npy(Z_path)
                    if check_nan:
                        assert_no_nan(Z, f"{Path(Z_path).name}")
                    if Z.ndim != 2 or Z.shape[1] != pca_dim:
                        logger.warning("Форма Z не відповідає pca_dim; рахуємо z на льоту.")
                    else:
                        logger.info("Використовуємо валідний Z-кеш для split=%s → %s", split, Z_path)
                        return Z
                except FileNotFoundError:
                    pass  # впадемо до обчислення на льоту

    # 2) обчислення на льоту
    if not Path(X_path).exists():
        logger.warning("PCA-проєкції для split=%s відсутні: %s — пропускаємо спліт.", split, X_path)
        return None

    X = load_npy(X_path)
    if X.size == 0:
        logger.info("Порожній split=%s (X має 0 рядків) — пропускаємо.", split)
        return None
    if X.ndim != 2 or X.shape[1] != pca_dim:
        raise ValueError(f"{X_path}: форма {X.shape} не відповідає pca_dim={pca_dim}")
    if check_nan:
        assert_no_nan(X, f"{Path(X_path).name}")

    Z = scaler.transform(X)
    if check_nan:
        assert_no_nan(Z, f"Z_{split}_std")
    return np.asarray(Z)


def _diag_z_split(
    z: np.ndarray,
    k_grid: Sequence[float],
) -> Dict[str, object]:
    """
    Обчислює:
      - grid по k: clip_overall, clip_per_pc [K,p], clip_mean_pc, clip_max_pc
      - базові z_stats per-PC
    """
    grid = clip_rate_grid_z(z, k_grid)  # {"k": [K], "overall": [K], "per_pc": [K,p] or None}
    k_arr: np.ndarray = grid["k"]
    overall: np.ndarray = grid["overall"]
    per_pc = grid["per_pc"]

    rows: List[Dict[str, float]] = []
    clip_mean_pc = np.zeros_like(overall)
    clip_max_pc = np.zeros_like(overall)

    if per_pc is None:
        # 1D випадок: трактуємо як (1,p), але таблиця буде тільки з overall
        for i in range(k_arr.size):
            rows.append({
                "k": float(k_arr[i]),
                "clip_overall": float(overall[i]),
            })
    else:
        for i in range(k_arr.size):
            per_pc_i = per_pc[i, :]
            clip_mean_pc[i] = float(per_pc_i.mean())
            clip_max_pc[i] = float(per_pc_i.max())
            row = {
                "k": float(k_arr[i]),
                "clip_overall": float(overall[i]),
                "clip_mean_pc": float(clip_mean_pc[i]),
                "clip_max_pc": float(clip_max_pc[i]),
            }
            # також додамо per-PC колонки
            for j in range(per_pc_i.size):
                row[f"clip_pc{j}"] = float(per_pc_i[j])
            rows.append(row)

    stats = z_stats(z, quantiles=(0.25, 0.5, 0.75, 0.95, 0.99), use_abs_for_quantiles=True)

    return {
        "k_table_rows": rows,
        "k": k_arr,
        "clip_overall": overall,
        "clip_mean_pc": clip_mean_pc,
        "clip_max_pc": clip_max_pc,
        "z_stats": stats,
    }


def _choose_k_recommended(
    k_arr: np.ndarray,
    clip_overall: np.ndarray,
    target_clip: float,
) -> float:
    """
    Вибирає найменший k зі сітки, де clip_overall < target_clip.
    Якщо такого немає — бере k з мінімальним clip_overall.
    """
    ok = np.where(clip_overall < float(target_clip))[0]
    if ok.size > 0:
        return float(k_arr[int(ok[0])])
    # fallback — k з мінімальним clip
    idx = int(np.argmin(clip_overall))
    return float(k_arr[idx])


def _diag_phi_split(
    z: np.ndarray,
    k_list: Sequence[float],
    angle_max: float,
    hist_bins: int,
) -> Dict[str, object]:
    """
    Обчислює φ-метрики для кожного k зі списку:
      - saturation_overall/per_pc, live_overall/per_pc, std_phi (per_pc)
      - гістограма φ (загальна), повертаємо бінування та counts
    """
    rows: List[Dict[str, float]] = []
    per_k_hist: Dict[str, object] = {}
    per_k_stdpc: Dict[str, List[float]] = {}

    for k in k_list:
        phi = compute_phi_from_z(z, k=k, angle_max=angle_max, no_clip=False)
        sat_overall, _ = saturation_rate_phi(phi, angle_max=angle_max)
        live_overall, _ = live_rate_phi(phi, lo=math.pi/3, hi=2*math.pi/3)
        std_pc = std_phi(phi)
        std_pc_np = np.asarray(std_pc)

        # Запишемо рядок-агрегат (overall + середній/макс std по PC)
        rows.append({
            "k": float(k),
            "sat_overall": float(sat_overall),
            "live_overall": float(live_overall),
            "std_phi_mean_pc": float(std_pc_np.mean()),
            "std_phi_max_pc": float(std_pc_np.max()),
        })

        # Збережемо пер-PC std для зручності
        per_k_stdpc[f"k={k:.3f}"] = std_pc_np.tolist()

        # Гістограма φ для цього k (загальна)
        H = hist_phi(phi, bins=hist_bins, per_pc=False)
        per_k_hist[f"k={k:.3f}"] = {
            "bins": H["bins"].tolist(),
            "hist": np.asarray(H["hist"]).astype(float).tolist()
        }

    return {
        "phi_table_rows": rows,
        "phi_std_pc": per_k_stdpc,
        "phi_hists": per_k_hist,
    }


# ---------------------------- main ----------------------------

def main() -> None:
    args = _parse_args()
    cfg = load_project_config(args.config)

    # Логер
    log_path = cfg.get("logging", {}).get("preprocess_log", "./runs/preproc/preprocess.log")
    log_level = cfg.get("logging", {}).get("level", "INFO")
    logger = setup_logger(log_file=log_path, logger_name="preprocess.diag", level=log_level, use_utc=True)

    # Де складати результати
    out_dir = cfg.get("paths", {}).get("diag_dir", "./runs/preproc/diag")
    ensure_dir(out_dir)

    # Базові налаштування
    pca_dim = int(cfg.get("pca", {}).get("dim", 8))
    stats_path = cfg.get("paths", {}).get("preprocess_stats", "./data/preprocess_stats.yaml")
    angle_max = float(cfg.get("angles", {}).get("angle_max", math.pi))
    safety = cfg.get("safety", {})
    check_nan = bool(safety.get("assert_no_nan", True))

    # CSV-налаштування (дружні до Excel у UA/EU локалях)
    csv_cfg = cfg.get("csv", {})
    csv_delim: str = str(csv_cfg.get("delimiter", ";"))
    csv_sep_hint: bool = bool(csv_cfg.get("excel_sep_hint", True))
    csv_utf8_bom: bool = bool(csv_cfg.get("utf8_bom", True))

    # Які спліти брати за замовчуванням
    splits = args.splits
    if not splits:
        # якщо є val — беремо val, інакше train
        x_val_path = cfg.get("paths", {}).get("X_val_pca", "./data/X_val_pca.npy")
        splits = ["val"] if Path(x_val_path).exists() else ["train"]

    # Завантажуємо параметри скейлера
    stats_yaml = load_yaml(stats_path)
    scaler = PCScaler.from_dict(stats_yaml)

    # Діагностика по кожному спліту
    for split in splits:
        logger.info("=== Діагностика split=%s ===", split)
        z = _load_or_compute_z_for_split(
            cfg=cfg,
            scaler=scaler,
            stats_yaml_path=stats_path,
            split=split,
            use_cache=args.use_cache,
            check_nan=check_nan,
            logger=logger,
        )
        if z is None:
            logger.warning("split=%s пропущено (немає даних).", split)
            continue

        if z.ndim != 2 or z.shape[1] != pca_dim:
            raise ValueError(f"split={split}: z shape {z.shape} не відповідає pca_dim={pca_dim}")
        N = z.shape[0]
        logger.info("z shape=%s; прикладів N=%d; p=%d", tuple(z.shape), N, pca_dim)

        # --- Z-метрики по сітці k ---
        diag_z = _diag_z_split(z, args.k_grid)
        k_arr: np.ndarray = diag_z["k"]  # type: ignore[assignment]
        overall: np.ndarray = diag_z["clip_overall"]  # type: ignore[assignment]
        rows_k: List[Dict[str, float]] = diag_z["k_table_rows"]  # type: ignore[assignment]
        stats_z: Dict[str, np.ndarray] = diag_z["z_stats"]  # type: ignore[assignment]

        # збережемо CSV з k-гридом
        k_csv = str(Path(out_dir) / f"diag_{split}_k.csv")
        save_csv(
            rows_k, k_csv,
            delimiter=csv_delim,
            excel_sep_hint=csv_sep_hint,
            utf8_bom=csv_utf8_bom
        )
        logger.info("Збережено таблицю по k → %s", k_csv)

        # і JSON з пер-PC статистиками z
        z_json = str(Path(out_dir) / f"diag_{split}_z_stats.json")
        # перетворимо np.ndarray у звичайні списки
        z_json_payload = {k: np.asarray(v).astype(float).tolist() for k, v in stats_z.items()}
        save_json(z_json_payload, z_json)
        logger.info("Збережено z-статистики → %s", z_json)

        # --- Вибір рекомендованого k ---
        k_rec = _choose_k_recommended(k_arr, overall, target_clip=args.target_clip)
        k_suggest = suggest_k_from_target_clip(z, target=args.target_clip)
        logger.info("Рекомендований k_rec=%.3f (з гриду), емпірична пропозиція k*≈%.3f", k_rec, k_suggest)

        # --- φ-метрики для всієї сітки (щоб не вгадувати топ-3) ---
        diag_phi = _diag_phi_split(z, k_arr.tolist(), angle_max, args.hist_bins)
        rows_phi: List[Dict[str, float]] = diag_phi["phi_table_rows"]  # type: ignore[assignment]
        phi_std_pc: Dict[str, List[float]] = diag_phi["phi_std_pc"]  # type: ignore[assignment]
        phi_hists: Dict[str, Dict[str, List[float]]] = diag_phi["phi_hists"]  # type: ignore[assignment]

        phi_csv = str(Path(out_dir) / f"diag_{split}_phi.csv")
        save_csv(
            rows_phi, phi_csv,
            delimiter=csv_delim,
            excel_sep_hint=csv_sep_hint,
            utf8_bom=csv_utf8_bom
        )
        logger.info("Збережено φ-метрики по k → %s", phi_csv)

        # збережемо гістограми φ у .npz (щоб швидко малювати потім)
        phi_npz = str(Path(out_dir) / f"diag_{split}_phi_hists.npz")
        # перетворимо у плоскі масиви для np.savez
        npz_payload = {}
        for key, d in phi_hists.items():
            bins = np.asarray(d["bins"], dtype=np.float64)
            hist = np.asarray(d["hist"], dtype=np.float64)
            npz_payload[f"{key}/bins"] = bins
            npz_payload[f"{key}/hist"] = hist
        save_npz(phi_npz, **npz_payload)
        logger.info("Збережено гістограми φ → %s", phi_npz)

        # підсумковий YAML
        # знайдемо індекс k_rec у гриді:
        irec = int(np.argmin(np.abs(k_arr - k_rec)))
        # обчислимо std_phi_mean на k_rec
        k_key = f"k={k_arr[irec]:.3f}"
        std_mean_at_k = float(np.mean(np.asarray(phi_std_pc[k_key], dtype=np.float64)))
        summary = {
            "split": split,
            "N": int(N),
            "p": int(pca_dim),
            "k_grid": [float(x) for x in k_arr],
            "target_clip": float(args.target_clip),
            "k_recommended": float(k_arr[irec]),
            "k_empirical_suggest": float(k_suggest),
            "clip_overall_at_k": float(overall[irec]),
            "std_phi_mean_at_k": std_mean_at_k,
        }
        sum_yaml = str(Path(out_dir) / f"diag_{split}_summary.yaml")
        save_yaml(summary, sum_yaml)
        logger.info("Збережено summary → %s", sum_yaml)

        # --- PNG-плоти (якщо дозволено) ---
        if not args.no_plots:
            png_clip = str(Path(out_dir) / f"plot_clip_over_k_{split}.png")
            _plot_clip_over_k(png_clip, k_arr, overall)

            # гістограма для k_rec (загальна)
            bins = np.asarray(phi_hists[k_key]["bins"])
            hist = np.asarray(phi_hists[k_key]["hist"])
            png_hist = str(Path(out_dir) / f"plot_phi_hist_{split}_k{float(k_arr[irec]):.2f}.png")
            _plot_phi_hist(png_hist, bins, hist)

            # std_phi per PC для k_rec
            std_pc_arr = np.asarray(phi_std_pc[k_key], dtype=np.float64)
            png_std = str(Path(out_dir) / f"plot_std_phi_per_pc_{split}_k{float(k_arr[irec]):.2f}.png")
            _plot_std_phi_per_pc(png_std, std_pc_arr)

    logger.info("Готово: діагностика завершена.")


if __name__ == "__main__":
    main()
