# Code/trainers/train_quantum.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader

# --- Проєктні імпорти ---
from Code.utils.io_utils import load_project_config  # YAML loader (корінь або Code/)
try:
    # Бажано мати єдину точку сідування
    from Code.utils.io_utils import seed_everything  # type: ignore[attr-defined]
except Exception:
    seed_everything = None  # fallback нижче

from Code.logger import setup_logger
from Code.datasets.phi_dataset import PhiDataset
from Code.models.quantum_classifier import QuantumClassifier

# Опційний реєстр оптимізаторів (QNA-Adam тощо)
try:
    from Code.optim.registry import get_optimizer as get_opt_from_registry  # type: ignore
except Exception:
    get_opt_from_registry = None

# Опційні квантові метрики (міст до angle_metrics)
try:
    from Code.quantum_layer.metrics_bridge import (
        compute_phi_metrics,
        compute_expval_metrics,
        compute_theta_grad_metrics,
        merge_dicts,
    )
    _HAS_Q_METRICS = True
except Exception:
    _HAS_Q_METRICS = False


# -------------------------- Допоміжні структури ---------------------------

@dataclass
class RunPaths:
    root: str
    metrics_csv: str
    model_pt: str
    config_snapshot: str
    profiler_json: str  # JSON Lines (.jsonl)
    exp_name: str
    optimizer: str


# --------------------------- Утиліти/допоміжні ----------------------------

def _now_stamp() -> str:
    # YYYYMMDD_HHMMSS
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_tag(x: Any) -> str:
    s = str(x).lower()
    return "".join(ch for ch in s if ch.isalnum()) or "na"


def _prepare_run_dir(cfg: Dict[str, Any], override_opt_name: Optional[str] = None) -> RunPaths:
    """Готує папку ран-а та імена артефактів; включає теги в ім'я експерименту."""
    runs_dir = cfg.get("logging", {}).get("dir", "runs")
    os.makedirs(runs_dir, exist_ok=True)

    # Компоненти імені
    k = cfg.get("angles", {}).get("k", "NA")
    L = cfg.get("quantum", {}).get("n_layers", "NA")
    shots = cfg.get("quantum", {}).get("shots", "NA")
    opt = override_opt_name if override_opt_name is not None else cfg.get("optim", {}).get("name", "adam")
    meas = cfg.get("quantum", {}).get("measurements", cfg.get("quantum", {}).get("measurement", "Z"))
    enc = cfg.get("quantum", {}).get("encoding", "ry")
    nq = cfg.get("pca", {}).get("dim", "NA")

    exp_name = (
        f"EXP_{_now_stamp()}"
        f"_opt{_normalize_tag(opt)}"
        f"_meas{_normalize_tag(meas)}"
        f"_enc{_normalize_tag(enc)}"
        f"_nq{_normalize_tag(nq)}"
        f"_k{_normalize_tag(k)}"
        f"_L{_normalize_tag(L)}"
        f"_shots{_normalize_tag(shots)}"
    )

    root = os.path.join(runs_dir, exp_name)
    os.makedirs(root, exist_ok=True)

    return RunPaths(
        root=root,
        metrics_csv=os.path.join(root, "metrics.csv"),
        model_pt=os.path.join(root, "model.pt"),
        config_snapshot=os.path.join(root, "config.snapshot.yaml"),
        profiler_json=os.path.join(root, "profiler.jsonl"),
        exp_name=exp_name,
        optimizer=_normalize_tag(opt),
    )


def make_worker_init_fn(base_seed: int):
    """Сідує random/np/torch у кожному DataLoader-воркері: base_seed + worker_id."""
    def _init_fn(worker_id: int):
        worker_seed = int(base_seed) + int(worker_id)
        try:
            import random
            random.seed(worker_seed)
        except Exception:
            pass
        try:
            np.random.seed(worker_seed)
        except Exception:
            pass
        try:
            torch.manual_seed(worker_seed)
        except Exception:
            pass
    return _init_fn


def _save_config_snapshot(cfg: Dict[str, Any], path: str, logger) -> None:
    # Спершу пробуємо io_utils.save_yaml (якщо є)
    try:
        from Code.utils.io_utils import save_yaml  # type: ignore
        save_yaml(cfg, path)
        logger.info("Збережено snapshot конфігу: %s", path)
        return
    except Exception:
        pass

    # Fallback на PyYAML або JSON
    try:
        import yaml  # type: ignore
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        logger.info("Збережено snapshot конфігу (YAML): %s", path)
    except Exception:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        logger.info("Збережено snapshot конфігу (JSON): %s", path)


def _fallback_seed_everything(seed: int, deterministic_torch: bool = False, logger=None):
    if logger:
        logger.info("Fallback seeding torch/numpy/random (seed=%d, deterministic_torch=%s)", seed, deterministic_torch)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic_torch:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def _set_all_seeds(cfg: Dict[str, Any], logger) -> int:
    """Єдиний вхід для сідування: намагаємось використати io_utils.seed_everything; інакше fallback."""
    seed = int(cfg.get("project", {}).get("seed", 42))
    det_torch = bool(cfg.get("project", {}).get("deterministic_torch", False))
    if seed_everything is not None:
        try:
            seed_everything(seed, deterministic_torch=det_torch, logger=logger)  # type: ignore[misc]
            logger.info("Сіди через seed_everything(seed=%d, deterministic_torch=%s)", seed, det_torch)
            return seed
        except Exception:
            logger.warning("seed_everything впав — використовую fallback.")
    _fallback_seed_everything(seed, deterministic_torch=det_torch, logger=logger)
    return seed


def _grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach()
        if g.numel() == 0:
            continue
        total += float(torch.sum(g * g).item())
    return math.sqrt(total)


def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        correct = (preds == y).sum().item()
        total = y.numel()
        return correct / max(total, 1)


def _linear_shot_schedule(epoch: int, total_epochs: int, start: int, end: int) -> int:
    if total_epochs <= 1:
        return int(end)
    alpha = min(max(epoch / (total_epochs - 1), 0.0), 1.0)
    return int(round((1 - alpha) * start + alpha * end))


def _log_config_summary(cfg: Dict[str, Any], logger) -> None:
    q = cfg.get("quantum", {})
    t = cfg.get("training", {})
    d = cfg.get("data", {})
    a = cfg.get("angles", {})

    train_path = d.get("train_phi") or d.get("train_path")
    val_path = d.get("val_phi") or d.get("val_path")

    logger.info(
        "Конфіг | n_qubits=%s | n_layers=%s | topology=%s | enc=%s | meas=%s | device=%s | shots=%s | diff=%s",
        cfg.get("pca", {}).get("dim", "NA"),
        q.get("n_layers", "NA"),
        q.get("topology", "NA"),
        q.get("encoding", "ry"),
        q.get("measurements", q.get("measurement", "Z")),
        q.get("device", "default.qubit"),
        str(q.get("shots", None)),
        q.get("diff_method", "adjoint"),
    )
    logger.info(
        "Навчання | batch_size=%s | epochs=%s | lr=%s | grad_clip=%.2f | n_classes=%s",
        t.get("batch_size", "NA"),
        t.get("epochs", "NA"),
        t.get("lr", cfg.get("optim", {}).get("lr", "NA")),
        float(t.get("grad_clip", 1.0)),
        d.get("n_classes", "NA"),
    )
    logger.info(
        "Дані    | train=%s | val=%s | angle_max=%s | k=%s",
        train_path or "dummy",
        val_path or "—",
        a.get("angle_max", "π"),
        a.get("k", "2.5"),
    )


def _get_current_shots(model) -> Optional[int]:
    try:
        q = getattr(model, "quantum", None)
        if q is None:
            return None
        if hasattr(q, "shots"):
            return q.shots  # type: ignore[attr-defined]
        # інколи shots у spec
        spec = getattr(q, "spec", None)
        if spec is not None and hasattr(spec, "shots"):
            return spec.shots  # type: ignore[attr-defined]
    except Exception:
        pass
    return None


def _set_model_shots(model, shots: Optional[int], logger=None) -> None:
    try:
        if hasattr(model, "set_shots"):
            model.set_shots(shots)  # type: ignore[attr-defined]
        else:
            getattr(model, "quantum").set_shots(shots)  # type: ignore[attr-defined]
        if logger:
            logger.info("Встановлено shots → %s", str(shots))
    except Exception as e:
        if logger:
            logger.warning("Не вдалось встановити shots=%s: %s", str(shots), e)


def _set_model_seed(model, seed: int, logger=None) -> None:
    try:
        if hasattr(model, "set_seed"):
            model.set_seed(seed)  # type: ignore[attr-defined]
        else:
            getattr(model, "quantum").set_seed(seed)  # type: ignore[attr-defined]
        if logger:
            logger.info("Reseed quantum device RNG → %d", seed)
    except Exception as e:
        if logger:
            logger.warning("Не вдалось reseed девайс: %s", e)


def _build_loaders(cfg: Dict[str, Any], seed: int, logger) -> Tuple[DataLoader, Optional[DataLoader]]:
    batch_size = int(cfg.get("training", {}).get("batch_size", 64))
    num_workers = int(cfg.get("training", {}).get("num_workers", 0))

    train_ds = PhiDataset(cfg, mode="train", logger=logger)
    val_path = cfg.get("data", {}).get("val_phi") or cfg.get("data", {}).get("val_path")
    val_ds = PhiDataset(cfg, mode="val", logger=logger)

    # Детермінізм для shuffle та воркерів
    g = torch.Generator().manual_seed(seed)
    worker_init = make_worker_init_fn(seed) if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=worker_init,
        generator=g,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=worker_init,
        generator=g,
        persistent_workers=(num_workers > 0),
    ) if val_ds is not None else None

    return train_loader, val_loader


def _build_model(cfg: Dict[str, Any], logger) -> QuantumClassifier:
    # Узгоджено з твоїм класом: приймає готовий dict-конфіг
    return QuantumClassifier(cfg, logger=logger)


def _build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module, logger) -> torch.optim.Optimizer:
    optim_cfg = cfg.get("optim", {})  # бажаний блок для опцій
    name = str(optim_cfg.get("name", "adam")).lower()

    # Базові опції Adam/AdamW
    lr = float(optim_cfg.get("lr", cfg.get("training", {}).get("lr", 3e-4)))
    wd = float(optim_cfg.get("weight_decay", 0.0))
    betas = tuple(optim_cfg.get("betas", (0.9, 0.999)))  # type: ignore
    eps = float(optim_cfg.get("eps", 1e-8))
    amsgrad = bool(optim_cfg.get("amsgrad", False))
    maximize = bool(optim_cfg.get("maximize", False))

    # Param groups (за бажанням різні lr для кванту та голови)
    pg: List[dict] = []
    lrs_cfg = optim_cfg.get("lrs", {}) or cfg.get("training", {}).get("lrs", {})
    if lrs_cfg:
        q_lr = float(lrs_cfg.get("quantum", lr))
        h_lr = float(lrs_cfg.get("head", lr))
        if hasattr(model, "quantum"):
            pg.append({"params": list(model.quantum.parameters()), "lr": q_lr})  # type: ignore[attr-defined]
        if hasattr(model, "classifier"):
            pg.append({"params": list(model.classifier.parameters()), "lr": h_lr})  # type: ignore[attr-defined]
        if not pg:
            pg.append({"params": model.parameters()})
    else:
        pg.append({"params": model.parameters()})

    # Додаткові kwargs (під кастомні оптимізатори типу QNA-Adam)
    extra_kwargs = {
        k: v for k, v in optim_cfg.items()
        if k not in {"name", "lr", "weight_decay", "betas", "eps", "amsgrad", "maximize", "lrs"}
    }

    if get_opt_from_registry is not None:
        opt = get_opt_from_registry(
            name=name,
            params=pg,
            lr=lr,
            weight_decay=wd,
            betas=betas,
            eps=eps,
            amsgrad=amsgrad,
            maximize=maximize,
            **extra_kwargs,
        )
        logger.info("Оптимізатор з registry: %s", name)
        return opt

    # Вбудовані варіанти
    if name in ("adamw",):
        opt = torch.optim.AdamW(pg, lr=lr, weight_decay=wd, betas=betas, eps=eps, amsgrad=amsgrad, maximize=maximize)
    else:
        opt = torch.optim.Adam(pg, lr=lr, weight_decay=wd, betas=betas, eps=eps, amsgrad=amsgrad, maximize=maximize)
    logger.info("Оптимізатор: %s (lr=%.3g, wd=%.2g, betas=%s, eps=%g, amsgrad=%s)", name, lr, wd, betas, eps, amsgrad)
    return opt

def _fmt(x, fmt=".6f"):
    """Робить 1,234 замість 1.234 з потрібною точністю."""
    if x is None:
        return ""
    # цілі — як є
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    # інше → float з потрібним форматом
    s = format(float(x), fmt)
    return s.replace(".", ",")
# --------------------------------- MAIN -----------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train quantum classifier (PennyLane + Torch)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (supports root config.yaml or Code/config.yaml)",
    )
    args = parser.parse_args(argv)
    logger = setup_logger("train_quantum")

    try:
        # Завантаження конфігу
        config = load_project_config(args.config)
        logger.info("Конфіг завантажено з: %s", args.config)
        _log_config_summary(config, logger)

        # Сіди
        seed = _set_all_seeds(config, logger)

        # Визначаємо фактичне ім'я оптимізатора для тега ран-а
        requested_opt = str(config.get("optim", {}).get("name", "adam")).lower()
        if get_opt_from_registry is None and requested_opt not in ("adam", "adamw"):
            resolved_opt = "adam"
        else:
            resolved_opt = requested_opt

        # Папка запуску та snapshot конфігу
        run_paths = _prepare_run_dir(config, override_opt_name=resolved_opt)
        logger.info("Run name: %s", run_paths.exp_name)
        logger.info("Run dir:  %s", run_paths.root)
        _save_config_snapshot(config, run_paths.config_snapshot, logger)
        logger.info("Артефакти будуть у: %s", run_paths.root)

        # Дані
        train_loader, val_loader = _build_loaders(config, seed, logger)

        # Модель
        model = _build_model(config, logger)  # CPU; default.qubit
        _set_model_seed(model, seed, logger=logger)  # синхронізуємо стохастику девайса

        # Оптимізатор
        optimizer = _build_optimizer(config, model, logger)
        criterion = torch.nn.CrossEntropyLoss()

        # Grad clipping
        grad_clip = float(config.get("training", {}).get("grad_clip", 1.0))
        # Якщо оптимізатор кліпить всередині (QNA-Adam) — вимикаємо зовнішній кліпінг
        name_norm = "".join(ch for ch in requested_opt if ch.isalnum())
        clip_in_opt = bool(config.get("optim", {}).get("qna", {}).get("clip_in_optimizer", False))
        if name_norm in ("qnaadam",) and clip_in_opt:
            grad_clip = 0.0
            logger.info("Grad clipping: external OFF (handled inside optimizer).")

        # Shot-annealing / eval_shots
        schedules = config.get("schedules", {})
        shots_sched = schedules.get("shots", {})
        shots_mode = str(shots_sched.get("mode", "none")).lower()
        shots_start = shots_sched.get("start", config.get("quantum", {}).get("shots", None))
        shots_end = shots_sched.get("end", shots_start)
        shots_epochs = int(shots_sched.get("epochs", config.get("training", {}).get("epochs", 1)))
        reseed_each_epoch = bool(config.get("quantum", {}).get("reseed_each_epoch", False))
        eval_shots = config.get("quantum", {}).get("eval_shots", None)

        # Початкові shots
        if shots_mode != "none" and shots_start is not None:
            if _get_current_shots(model) != shots_start:
                _set_model_shots(model, shots_start, logger=logger)
                logger.info("Початкові shots → %s (режим аннілінгу: %s)", str(shots_start), shots_mode)

        # CSV заголовок
        with open(run_paths.metrics_csv, "w", newline="", encoding="utf-8") as csv_file:
            csv_file.write("sep=;\n")
            csv_writer = csv.writer(csv_file, delimiter=";")
            csv_writer.writerow(
                ["epoch", "step", "split", "loss", "acc",
                 "grad_norm_raw", "grad_norm", "clipped",
                 "lr", "shots", "time_ms"]
            )
            csv_file.flush()

            # Тренувальний цикл
            epochs = int(config.get("training", {}).get("epochs", 20))
            best_val = float("inf")
            save_every = int(config.get("logging", {}).get("save_every_epochs", 0))

            logger.info(
                "Старт тренування: epochs=%d, batch_size=%d, grad_clip=%.2f, shots=%s",
                epochs,
                int(config.get("training", {}).get("batch_size", 64)),
                grad_clip,
                str(config.get("quantum", {}).get("shots", None)),
            )

            for ep in range(1, epochs + 1):
                epoch_t0 = time.time()

                # Опц. reseed по епохах (для контрольованої shot-noise послідовності)
                if reseed_each_epoch:
                    _set_model_seed(model, seed + ep, logger=logger)

                # Shot annealing
                if shots_mode == "linear" and (shots_start is not None) and (shots_end is not None):
                    new_shots = _linear_shot_schedule(ep - 1, shots_epochs, int(shots_start), int(shots_end))
                    if _get_current_shots(model) != new_shots:
                        _set_model_shots(model, new_shots, logger=logger)

                # ---- TRAIN ----
                model.train()
                running_loss = running_acc = 0.0
                running_gn_raw = running_gn = 0.0
                clip_count = 0
                steps = 0
                t0 = time.time()

                for step, (xb, yb) in enumerate(train_loader, start=1):
                    steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = criterion(logits, yb)

                    # Guard від не-фінтних лоссів
                    if not torch.isfinite(loss):
                        logger.warning("Non-finite loss @ epoch=%d step=%d → пропускаю крок.", ep, step)
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    loss.backward()

                    # (Легкі) квантові метрики — лише на 1-му батчі епохи (за наявності містка)
                    if _HAS_Q_METRICS and step == 1:
                        try:
                            with torch.no_grad():
                                expvals = model.quantum(xb)  # (B, out_dim)  # type: ignore[attr-defined]
                            q_metrics = merge_dicts(
                                compute_phi_metrics(xb, angle_max=float(config.get("angles", {}).get("angle_max", math.pi))),
                                compute_expval_metrics(expvals),
                                # Якщо у шарі параметри називаються інакше — блок спрацює під try/except
                                compute_theta_grad_metrics(getattr(model.quantum, "theta", None)),  # type: ignore[attr-defined]
                            )
                            # Акуратно округляємо числові значення
                            safe_qm = {k: (round(float(v), 6) if isinstance(v, (int, float)) else v) for k, v in q_metrics.items()}
                            logger.info("qmetrics: %s", safe_qm)
                        except Exception:
                            pass

                    # Норма ∇ ДО кліпінгу
                    gn_raw = _grad_norm(model.parameters())
                    clipped_flag = 0
                    if grad_clip and grad_clip > 0:
                        unclipped = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        if float(unclipped) > float(grad_clip):
                            clipped_flag = 1

                    optimizer.step()

                    # Норма ∇ ПІСЛЯ кліпінгу (для інформації)
                    gn = _grad_norm(model.parameters())
                    acc = _accuracy_from_logits(logits, yb)

                    running_loss += float(loss.item())
                    running_acc += acc
                    running_gn_raw += gn_raw
                    running_gn += gn
                    clip_count += clipped_flag

                    # CSV по батчах
                    elapsed_ms = int((time.time() - t0) * 1000)
                    current_shots = str(_get_current_shots(model))
                    lr_used = None
                    for g in optimizer.param_groups:
                        lr_used = g.get("lr", None)
                        break
                    csv_writer.writerow([
                        ep, step, "train",
                        _fmt(loss.item(), ".6f"),   # loss
                        _fmt(acc, ".4f"),           # acc
                        _fmt(gn_raw, ".6f"),        # grad_norm_raw
                        _fmt(gn, ".6f"),            # grad_norm
                        int(clipped_flag),          # clipped (ціле)
                        (_fmt(lr_used, ".6g") if lr_used is not None else ""),  # lr
                        (_fmt(current_shots, ".0f") if isinstance(current_shots, (int, float)) else (current_shots or "")),  # shots
                        int(elapsed_ms),            # time_ms (ціле)
                    ])

                # Агрегати трену
                train_loss = running_loss / max(steps, 1)
                train_acc = running_acc / max(steps, 1)
                train_gn_raw = running_gn_raw / max(steps, 1)
                train_gn = running_gn / max(steps, 1)
                clip_rate = clip_count / max(steps, 1)
                current_shots = str(_get_current_shots(model))
                logger.info(
                    "[Epoch %d] train: loss=%.4f, acc=%.3f, ‖∇‖raw=%.3f, ‖∇‖=%.3f, clip_rate=%.2f, shots=%s",
                    ep, train_loss, train_acc, train_gn_raw, train_gn, clip_rate, current_shots
                )
                csv_writer.writerow([
                    ep, "E", "train_epoch",
                    _fmt(train_loss, ".6f"),
                    _fmt(train_acc, ".4f"),
                    _fmt(train_gn_raw, ".6f"),
                    _fmt(train_gn, ".6f"),
                    _fmt(clip_rate, ".4f"),
                    "",                         # lr (немає)
                    (_fmt(current_shots, ".0f") if isinstance(current_shots, (int, float)) else (current_shots or "")),
                    0,
                ])
                csv_file.flush()

                # ---- VAL (опційно) ----
                if val_loader is not None:
                    original_shots = _get_current_shots(model)
                    try:
                        if eval_shots is not None:
                            _set_model_shots(model, eval_shots, logger=logger)

                        model.eval()
                        v_running_loss, v_running_acc, v_steps = 0.0, 0.0, 0
                        with torch.no_grad():
                            for xb, yb in val_loader:
                                logits = model(xb)
                                loss = criterion(logits, yb)
                                v_running_loss += float(loss.item())
                                v_running_acc += _accuracy_from_logits(logits, yb)
                                v_steps += 1
                        val_loss = v_running_loss / max(v_steps, 1)
                        val_acc = v_running_acc / max(v_steps, 1)
                        logger.info("[Epoch %d]   val: loss=%.4f, acc=%.3f (eval_shots=%s)",
                                    ep, val_loss, val_acc, str(eval_shots) if eval_shots is not None else current_shots)
                        shots_val = eval_shots if eval_shots is not None else current_shots
                        csv_writer.writerow([
                            ep, "E", "val_epoch",
                            _fmt(val_loss, ".6f"),
                            _fmt(val_acc,  ".4f"),
                            "", "", "",
                            "",                       # lr
                            _fmt(shots_val, ".0f"),   # shots
                            ""                        # time_ms
                        ])
                        csv_file.flush()

                        # Найкращий чекпойнт по val_loss
                        if val_loss < best_val:
                            best_val = val_loss
                            torch.save(model.state_dict(), run_paths.model_pt)
                            logger.info("Збережено найкращу модель (val_loss=%.6f) → %s", best_val, run_paths.model_pt)
                    finally:
                        # Повертаємо shots навіть у разі виключення
                        if eval_shots is not None:
                            _set_model_shots(model, original_shots, logger=logger)

                # Періодичне збереження (коли без валідації)
                if (val_loader is None) and (save_every > 0) and (ep % save_every == 0):
                    torch.save(model.state_dict(), run_paths.model_pt)
                    logger.info("Збережено модель (кожні %d епох) → %s", save_every, run_paths.model_pt)

                # Профайлінг епохи → JSONL
                epoch_ms = int((time.time() - epoch_t0) * 1000)
                profile_entry = {"epoch": ep, "time_ms": epoch_ms, "shots": _get_current_shots(model)}
                with open(run_paths.profiler_json, "a", encoding="utf-8") as f:
                    f.write(json.dumps(profile_entry) + "\n")

            # Якщо нічого не зберегли — збережемо фінальну
            if not os.path.exists(run_paths.model_pt):
                torch.save(model.state_dict(), run_paths.model_pt)
                logger.info("Збережено фінальну модель → %s", run_paths.model_pt)

        logger.info("Готово. Артефакти: %s", run_paths.root)
        return 0

    except Exception as e:
        logger.exception("Помилка під час тренування: %s", str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
