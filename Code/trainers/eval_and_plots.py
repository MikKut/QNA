# Code/scripts/eval_and_plots.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

# --- проектні імпорти (ті ж, що у тренері) ---
from Code.utils.io_utils import load_project_config
from Code.logger import setup_logger
from Code.datasets.phi_dataset import PhiDataset
from Code.models.quantum_classifier import QuantumClassifier

_LOG = setup_logger("eval_plots")


@dataclass
class RunIO:
    run_dir: str
    metrics_csv: str
    model_pt: str
    cfg_snapshot: str
    figs_dir: str
    report_md: str
    test_json: str


def _discover_runio(run_dir: str) -> RunIO:
    metrics_csv = os.path.join(run_dir, "metrics.csv")
    model_pt = os.path.join(run_dir, "model.pt")
    cfg_snapshot = os.path.join(run_dir, "config.snapshot.yaml")
    figs_dir = os.path.join(run_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    return RunIO(
        run_dir=run_dir,
        metrics_csv=metrics_csv,
        model_pt=model_pt,
        cfg_snapshot=cfg_snapshot,
        figs_dir=figs_dir,
        report_md=os.path.join(run_dir, "report.md"),
        test_json=os.path.join(run_dir, "test_metrics.json"),
    )


def _fmt(x: Optional[float], n=6) -> str:
    return "" if x is None else f"{float(x):.{n}f}"


def _load_metrics_df(path: str) -> pd.DataFrame:
    # перший рядок у нас "sep=;" — pandas це зрозуміє, але без engine="python" може сваритися на крапку з комою.
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
    sep = ";" if "sep=;" in first else ","
    df = pd.read_csv(path, sep=sep, comment="s", engine="python")
    # нормалізуємо типи
    for col in ["epoch", "step", "time_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def _build_loaders(cfg: Dict[str, Any], batch_size: Optional[int] = None):
    bs = int(batch_size or cfg.get("training", {}).get("batch_size", 64))
    test_ds = PhiDataset(cfg, mode="test", logger=_LOG)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=bs, shuffle=False, num_workers=int(cfg.get("training", {}).get("num_workers", 0)),
        pin_memory=False, persistent_workers=False
    )
    return test_loader


def _build_model(cfg: Dict[str, Any]) -> QuantumClassifier:
    return QuantumClassifier(cfg, logger=_LOG)


@torch.no_grad()
def _evaluate(model: torch.nn.Module, loader, crit) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    steps = 0
    for xb, yb in loader:
        expvals = model.quantum(xb)
        logits = model.classifier(expvals)
        loss = crit(logits, yb)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == yb).float().mean().item()
        loss_sum += float(loss.item())
        acc_sum += float(acc)
        steps += 1
    return (loss_sum / max(steps, 1), acc_sum / max(steps, 1))


def _maybe_set_eval_shots(model, shots: Optional[int]):
    try:
        if shots is None:
            return
        if hasattr(model, "set_shots"):
            model.set_shots(shots)  # type: ignore[attr-defined]
        else:
            getattr(model, "quantum").set_shots(shots)  # type: ignore[attr-defined]
        _LOG.info("Eval shots → %s", str(shots))
    except Exception as e:
        _LOG.warning("Не вдалося встановити eval_shots=%s: %s", str(shots), e)


def _plot_curves(df: pd.DataFrame, figs_dir: str):
    df = df.copy()

    # зведемо до епох
    train_ep = df[df["split"] == "train_epoch"]
    val_ep   = df[df["split"] == "val_epoch"]
    # опційні поля
    have_vt = "Vtilde" in df.columns and df["Vtilde"].notna().any()
    have_lr = "lr" in df.columns and df["lr"].notna().any()
    have_shots = "shots" in df.columns and df["shots"].notna().any()
    have_gn = "grad_norm" in df.columns and df["grad_norm"].notna().any()
    have_gn_raw = "grad_norm_raw" in df.columns and df["grad_norm_raw"].notna().any()
    have_clip = "clipped" in df.columns and (df["split"] == "train").any()

    def savefig(name: str):
        path = os.path.join(figs_dir, name)
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        return path

    # 1) Loss
    plt.figure(figsize=(6, 4))
    if not train_ep.empty:
        plt.plot(train_ep["epoch"], train_ep["loss"], label="train")
    if not val_ep.empty:
        plt.plot(val_ep["epoch"], val_ep["loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.legend()
    loss_png = savefig("loss.png")

    # 2) Accuracy
    plt.figure(figsize=(6, 4))
    if not train_ep.empty:
        plt.plot(train_ep["epoch"], train_ep["acc"], label="train")
    if not val_ep.empty:
        plt.plot(val_ep["epoch"], val_ep["acc"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy")
    plt.legend()
    acc_png = savefig("acc.png")

    # 3) Grad norms
    gn_png = None
    if have_gn or have_gn_raw:
        plt.figure(figsize=(6, 4))
        if have_gn_raw and not train_ep.empty:
            plt.plot(train_ep["epoch"], train_ep["grad_norm_raw"], label="‖∇‖ raw")
        if have_gn and not train_ep.empty:
            plt.plot(train_ep["epoch"], train_ep["grad_norm"], label="‖∇‖ clipped")
        plt.xlabel("epoch")
        plt.ylabel("norm")
        plt.title("Gradient norms")
        plt.legend()
        gn_png = savefig("grad_norms.png")

    # 4) Clip rate
    clip_png = None
    if have_clip:
        # по батчах трену: частка clipped у межах епохи вже підрахована як train_epoch.clip_rate,
        # але щоб не залежати від формату, підрахуємо з сирих рядків:
        train_batches = df[df["split"] == "train"].copy()
        if not train_batches.empty and "epoch" in train_batches and "clipped" in train_batches:
            grp = train_batches.groupby("epoch")["clipped"].mean()
            plt.figure(figsize=(6, 4))
            plt.plot(grp.index.values, grp.values)
            plt.xlabel("epoch")
            plt.ylabel("clip_rate")
            plt.title("Gradient clipping rate (train)")
            clip_png = savefig("clip_rate.png")

    # 5) Vtilde
    vt_png = None
    if have_vt:
        # беремо середній Vtilde по батчах на епоху (якщо train_epoch має, теж ок)
        if "split" in df.columns and (df["split"] == "train").any():
            vt = df[(df["split"] == "train") & df["Vtilde"].notna()].groupby("epoch")["Vtilde"].mean()
            if not vt.empty:
                plt.figure(figsize=(6, 4))
                plt.plot(vt.index.values, vt.values)
                plt.xlabel("epoch")
                plt.ylabel("Vtilde (mean over train batches)")
                plt.title("Noise proxy Ṽ (train)")
                vt_png = savefig("vtilde.png")

    # 6) Shots
    shots_png = None
    if have_shots:
        ep_shots = None
        # віддаємо перевагу train_epoch рядкам; якщо їх нема — беремо train батчі і усереднюємо
        if not train_ep.empty and "shots" in train_ep:
            ep_shots = train_ep[["epoch", "shots"]].dropna().drop_duplicates("epoch")
        elif (df["split"] == "train").any():
            ep_shots = df[df["split"] == "train"].groupby("epoch")["shots"].first().reset_index()
        if ep_shots is not None and len(ep_shots) > 0:
            plt.figure(figsize=(6, 3.5))
            plt.step(ep_shots["epoch"], ep_shots["shots"], where="post")
            plt.xlabel("epoch")
            plt.ylabel("shots")
            plt.title("Shots schedule")
            shots_png = savefig("shots.png")

    # 7) LR (за першою групою)
    lr_png = None
    if have_lr and not train_ep.empty:
        # якщо lr логувався лише по батчах — візьмемо перший batch кожної епохи
        tmp = df[(df["split"] == "train") & df["lr"].notna()].copy()
        if tmp.empty:
            tmp = train_ep[["epoch", "lr"]].copy()
        else:
            tmp = tmp.groupby("epoch").first().reset_index()[["epoch", "lr"]]
        if not tmp.empty:
            plt.figure(figsize=(6, 3.5))
            plt.plot(tmp["epoch"], tmp["lr"])
            plt.xlabel("epoch")
            plt.ylabel("lr")
            plt.title("Learning rate (group 0)")
            lr_png = savefig("lr.png")

    return {
        "loss_png": loss_png,
        "acc_png": acc_png,
        "gn_png": gn_png,
        "clip_png": clip_png,
        "vt_png": vt_png,
        "shots_png": shots_png,
        "lr_png": lr_png,
    }


def _write_report(md_path: str, figs: Dict[str, Optional[str]], test_metrics: Optional[Dict[str, Any]], cfg: Dict[str, Any]):
    lines = []
    lines.append(f"# Eval report\n")
    # короткий конфіг
    q = cfg.get("quantum", {})
    t = cfg.get("training", {})
    lines.append(f"- device: `{q.get('device','default.qubit')}`, shots: `{q.get('shots',None)}`")
    lines.append(f"- n_qubits: `{cfg.get('pca',{}).get('dim','NA')}`, n_layers: `{q.get('n_layers','NA')}`, encoding: `{q.get('encoding','ry')}`, meas: `{q.get('measurement', q.get('measurements','Z'))}`")
    lines.append(f"- optimizer: `{cfg.get('optim',{}).get('name','adam')}`, lr: `{t.get('lr', cfg.get('optim',{}).get('lr','NA'))}`\n")

    if test_metrics is not None:
        lines.append("## Test metrics\n")
        lines.append(f"- test_loss: **{_fmt(test_metrics.get('loss'))}**")
        lines.append(f"- test_acc:  **{_fmt(test_metrics.get('acc'), 4)}**\n")

    lines.append("## Figures\n")
    for k, p in figs.items():
        if p:
            lines.append(f"### {k.replace('_',' ').title()}\n")
            rel = os.path.relpath(p, os.path.dirname(md_path))
            lines.append(f"![{k}]({rel})\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate test set and plot training curves")
    parser.add_argument("--run_dir", type=str, required=True, help="Папка конкретного рану (містить metrics.csv, model.pt, config.snapshot.yaml)")
    parser.add_argument("--eval_shots", type=int, default=None, help="Перевизначити shots для eval/test (опційно)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size для тесту (опційно)")
    args = parser.parse_args(argv)

    io = _discover_runio(args.run_dir)

    # 1) Конфіг
    if not os.path.exists(io.cfg_snapshot):
        _LOG.error("Не знайдено snapshot конфігу: %s", io.cfg_snapshot)
        return 2
    cfg = load_project_config(io.cfg_snapshot)
    _LOG.info("Завантажено конфіг: %s", io.cfg_snapshot)

    # 2) Модель + чекпойнт
    if not os.path.exists(io.model_pt):
        _LOG.error("Не знайдено модель: %s", io.model_pt)
        return 2
    model = _build_model(cfg, )
    sd = torch.load(io.model_pt, map_location="cpu")
    model.load_state_dict(sd)
    _maybe_set_eval_shots(model, args.eval_shots if args.eval_shots is not None else cfg.get("quantum", {}).get("eval_shots", None))

    # 3) Test loader (якщо є)
    test_metrics = None
    try:
        test_loader = _build_loaders(cfg, batch_size=args.batch_size)
        crit = torch.nn.CrossEntropyLoss()
        test_loss, test_acc = _evaluate(model, test_loader, crit)
        test_metrics = {"loss": float(test_loss), "acc": float(test_acc), "shots": (args.eval_shots or cfg.get("quantum", {}).get("eval_shots", None))}
        with open(io.test_json, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2, ensure_ascii=False)
        _LOG.info("TEST: loss=%.4f, acc=%.3f", test_loss, test_acc)
    except Exception as e:
        _LOG.warning("Тест-сет недоступний або оцінка не виконана: %s", e)

    # 4) Плоти з metrics.csv (якщо є)
    figs = {}
    if os.path.exists(io.metrics_csv):
        try:
            df = _load_metrics_df(io.metrics_csv)
            figs = _plot_curves(df, io.figs_dir)
            _LOG.info("Графіки збережено до: %s", io.figs_dir)
        except Exception as e:
            _LOG.warning("Не вдалося побудувати графіки: %s", e)
    else:
        _LOG.warning("Не знайдено metrics.csv: %s", io.metrics_csv)

    # 5) Короткий markdown-звіт
    try:
        _write_report(io.report_md, figs, test_metrics, cfg)
        _LOG.info("Report: %s", io.report_md)
    except Exception as e:
        _LOG.warning("Не вдалося записати report.md: %s", e)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())