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
import copy
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

# Прямий імпорт QNAAdam як fallback, якщо реєстру нема
try:
    from Code.optim.qna_adam import QNAAdam  # type: ignore
    _HAS_QNA = True
except Exception:
    QNAAdam = None  # type: ignore
    _HAS_QNA = False

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


class EarlyStopper:
    """
    Проста рання зупинка.
      monitor: 'val_loss' (mode='min') або 'val_acc' (mode='max')
      patience: скільки епох без покращення терпіти
      min_delta: мінімальний зсув, щоб вважати покращенням
      warmup_epochs: скільки епох не перевіряти критерій
      restore_best: відкотити найкращий state_dict моделі
    """
    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 5,
        min_delta: float = 0.0,
        warmup_epochs: int = 0,
        restore_best: bool = True,
        logger=None,
    ) -> None:
        assert mode in ("min", "max")
        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.warmup_epochs = int(warmup_epochs)
        self.restore_best = bool(restore_best)
        self.logger = logger or setup_logger("earlystop")

        self.best_value: Optional[float] = None
        self.best_state: Optional[Dict[str, Any]] = None
        self.bad_epochs: int = 0

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < (self.best_value - self.min_delta)
        else:
            return value > (self.best_value + self.min_delta)

    def step(self, value: float, epoch: int, model: torch.nn.Module) -> bool:
        # теплий старт
        if epoch <= self.warmup_epochs:
            return False
        if self.best_value is None or self._is_better(value):
            self.best_value = float(value)
            self.bad_epochs = 0
            if self.restore_best:
                # робимо легкий deepcopy state_dict
                self.best_state = copy.deepcopy(model.state_dict())
            if self.logger:
                self.logger.info("[EarlyStop] Новий найкращий %s = %.6f на епосі %d",
                                 self.monitor, self.best_value, epoch)
            return False
        else:
            self.bad_epochs += 1
            if self.logger:
                self.logger.info("[EarlyStop] Немає покращення (%d/%d), поточне=%.6f, найкраще=%.6f",
                                 self.bad_epochs, self.patience, float(value), float(self.best_value))
            if self.bad_epochs >= self.patience:
                if self.restore_best and self.best_state is not None:
                    try:
                        model.load_state_dict(self.best_state)
                        if self.logger:
                            self.logger.info("[EarlyStop] Відновлено найкращий стейт моделі.")
                    except Exception:
                        pass
                return True
            return False


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


def _enforce_torch_determinism(enabled: bool, logger) -> None:
    """Додає практичні прапорці детермінізму для CUDA/Torch."""
    if not enabled or not torch.cuda.is_available():
        return
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        # Вимкнути TF32 для стабільності (не впливає на CPU)
        torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
        torch.backends.cudnn.allow_tf32 = False        # type: ignore[attr-defined]
    except Exception:
        pass
    # Для детермінізму на деяких CUDA-опах; не критично, але корисно зафіксувати
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    logger.info("Torch determinism flags applied on CUDA (cudnn.deterministic=True, benchmark=False, TF32=OFF).")

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
        q.get("diff_method", "parameter-shift"),
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

def _validate_quantum_backend(cfg: Dict[str, Any], model, logger) -> None:
    """Легка валідація налаштувань квантового бекенду й узгодження з конфігом."""
    qcfg = cfg.get("quantum", {}) or {}
    pca_dim = int(cfg.get("pca", {}).get("dim", -1))
    n_qubits = getattr(getattr(model, "quantum", None), "n_qubits", None)
    if isinstance(n_qubits, int) and pca_dim > 0 and n_qubits != pca_dim:
        logger.warning("n_qubits (%s) != pca.dim (%s) — перевірте узгодженість енкодингу.", n_qubits, pca_dim)
    # diff_method sanity
    allowed_diff = {"parameter-shift", "finite-diff", "backprop", "adjoint"}
    dm = str(qcfg.get("diff_method", "parameter-shift")).lower()
    if dm not in allowed_diff:
        logger.warning("Непідтримуваний diff_method='%s' — перевірте конфіг (очікувано одне з %s).", dm, sorted(allowed_diff))
    # shots тип
    shots = qcfg.get("shots", None)
    if shots is not None and not isinstance(shots, int):
        logger.warning("shots має бути int, зараз %r — бекенд може впасти пізніше.", shots)
    # device info
    try:
        dev = getattr(getattr(model, "quantum", None), "device", None)
        logger.info("Quantum backend: %s | shots=%s", getattr(dev, "name", str(dev)), str(_get_current_shots(model)))
    except Exception:
        pass

def _sanity_check_model(cfg: Dict[str, Any], model, train_loader: DataLoader, criterion, logger) -> None:
    """Одноразовий dry-run: перевіряє форми, NaN/Inf і узгодженість n_classes."""
    try:
        n_classes = int(cfg.get("data", {}).get("n_classes", -1))
        # ВАЖЛИВО: не використовувати iter(train_loader), щоб не зсувати RNG/порядок!
        train_ds = train_loader.dataset  # type: ignore[attr-defined]
        bs = int(cfg.get("training", {}).get("batch_size", 64))
        take = min(len(train_ds), max(1, bs))  # перші 'take' елементів без shuffle
        xs, ys = [], []
        for i in range(take):
            xi, yi = train_ds[i]
            xs.append(xi)
            ys.append(yi)
        xb = torch.stack(xs, dim=0)
        yb = torch.as_tensor(ys)
        t0 = time.time()
        with torch.no_grad():
            E = model.quantum(xb)
            logits = model.classifier(E)
            _ = criterion(logits, yb)  # перевірка сумісності форми
        if not torch.isfinite(E).all() or not torch.isfinite(logits).all():
            logger.warning("Sanity: виявлено не-фінтні значення у expvals/логітах.")
        if n_classes > 0 and logits.shape[-1] != n_classes:
            logger.warning("Sanity: logits.shape[-1]=%d ≠ n_classes=%d — перевірте head/конфіг.",
                           int(logits.shape[-1]), n_classes)
        logger.info("Sanity: E=%s → logits=%s | n_classes=%s | dry-run %.1f ms",
                    tuple(E.shape), tuple(logits.shape), (n_classes if n_classes>0 else "NA"),
                    (time.time() - t0)*1000.0)
    except StopIteration:
        logger.warning("Sanity: train_loader порожній — пропускаю перевірку.")
    except Exception as e:
        logger.warning("Sanity: перевірка впала: %s", e)

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
    batch_size  = int(cfg.get("training", {}).get("batch_size", 64))
    num_workers = int(cfg.get("training", {}).get("num_workers", 0))
    pin_mem     = bool(cfg.get("training", {}).get("pin_memory", torch.cuda.is_available()))
    drop_last   = bool(cfg.get("training", {}).get("drop_last", True))
    prefetch    = int(cfg.get("training", {}).get("prefetch_factor", 2))
    if batch_size <= 0:
        logger.error("training.batch_size має бути ≥ 1, зараз: %d", batch_size)
        raise ValueError("Invalid batch_size")
    
    train_ds = PhiDataset(cfg, mode="train", logger=logger)
    val_ds   = PhiDataset(cfg, mode="val",   logger=logger)

    if len(train_ds) == 0:
        logger.error("Training split is empty — перевірте шляхи або препроцесинг.")
        raise RuntimeError("Empty training dataset")

    # Детермінізм для shuffle та воркерів
    g = torch.Generator().manual_seed(seed)
    worker_init = make_worker_init_fn(seed) if num_workers > 0 else None
    dl_kwargs = {"persistent_workers": (num_workers > 0)}
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = prefetch

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        worker_init_fn=worker_init,
        generator=g,
        drop_last=drop_last,
        **dl_kwargs,
    )

    # Створюємо val_loader ТІЛЬКИ якщо є зразки
    if len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
            worker_init_fn=worker_init,
            generator=g,
            drop_last=False,
            **dl_kwargs,
        )
    else:
        val_loader = None
        logger.warning("Validation split is empty — працюємо без валідації / EarlyStopping.")

    try:
        logger.info("Train size: %d | Val size: %s", len(train_ds), (len(val_ds) if val_loader is not None else "—"))
    except Exception:
        pass
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

    # Спроба через registry
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

    # Вбудовані варіанти (fallback)
    if name in ("qna_adam", "qnaadam") and _HAS_QNA:
        qna = optim_cfg.get("qna", {}) or {}
        opt = QNAAdam(
            pg,
            lr=lr,
            weight_decay=wd,
            betas=betas,
            eps=eps,
            amsgrad=amsgrad,
            maximize=maximize,
            lambda_var=float(qna.get("lambda_var", 0.0)),
            lr_min_mult=float(qna.get("lr_min_mult", 0.0)),
            lr_max_mult=float(qna.get("lr_max_mult", 1.0)),
            vtilde_key=str(qna.get("vtilde_key", "qna_vtilde")),
            clip_in_optimizer=bool(qna.get("clip_in_optimizer", False)),
            max_norm=float(qna.get("max_norm", 1.0)),
            error_if_nonfinite=bool(qna.get("error_if_nonfinite", False)),
            grad_eps=float(qna.get("grad_eps", 1e-6)),
        )
        logger.info("Оптимізатор: QNAAdam (lr=%.3g, wd=%.2g, betas=%s, eps=%g, amsgrad=%s)", lr, wd, betas, eps, amsgrad)
        return opt

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


def _push_qna_var_to_optimizer(
    optimizer: torch.optim.Optimizer,
    module: torch.nn.Module,
    var_param: torch.Tensor,
    logger
) -> None:
    """
    Кладемо пер-параметрний шум V саме в state[theta]["qna_var_param"].
    Вимагаємо збіг шейпу або робимо керований fallback (із логами).
    Додатково гарантуємо finite та ≥0.
    """
    if var_param is None:
        return

    # 1) Знайти іменований параметр 'theta' без рекурсії (у твоєму QuantumLayer він верхнього рівня)
    theta = None
    theta_name = None
    for n, p in module.named_parameters(recurse=False):
        if n == "theta":
            theta, theta_name = p, n
            break
    if theta is None:
        logger.warning("QNA: не знайдено параметр 'theta' у quantum-модулі.")
        return

    # 2) Переконатися, що theta оптимізується
    if not any(theta in g.get("params", []) for g in optimizer.param_groups):
        logger.warning("QNA: 'theta' відсутній у optimizer.param_groups — var_param буде проігноровано.")
        return

    with torch.no_grad():
        # 3) Санітаризація та приведення типів/девайсу
        V = var_param.to(dtype=theta.dtype, device=theta.device, copy=False)
        V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
        V.clamp_min_(0.0)

        # 4) Узгодження форми: або точний збіг, або скаляр, або fallback=mean
        did_mismatch = False
        used_mean = None

        if V.shape != theta.shape:
            did_mismatch = True
            if V.numel() == 1:
                V = V.expand_as(theta)
                logger.info("QNA: var_param скаляр — розширю до theta.shape=%s.", tuple(theta.shape))
            else:
                used_mean = float(V.mean().item())
                V = torch.full_like(theta, used_mean)
                logger.warning(
                    "QNA: var_param.shape=%s ≠ theta.shape=%s — fallback на mean=%.3e.",
                    tuple(var_param.shape), tuple(theta.shape), used_mean
                )

        # 5) Запис у state рівно одного параметра (θ)
        optimizer.state[theta]["qna_var_param"] = V

        # 6) Діагностика
        Vw = optimizer.state[theta]["qna_var_param"]
        if did_mismatch and used_mean is not None:
            logger.warning(
                "QNA PUSH: shape mismatch for %s: V%s != P%s → fallback mean=%.3e",
                theta_name, tuple(var_param.shape), tuple(theta.shape), used_mean
            )
        logger.debug(
            "QNA PUSH: wrote V into state[%s]: shape=%s | min/mean/max=[%.3e, %.3e, %.3e]",
            theta_name, tuple(Vw.shape),
            float(Vw.min().item()), float(Vw.mean().item()), float(Vw.max().item())
        )

def _extract_qna_stats_for_module(optimizer: torch.optim.Optimizer, module: torch.nn.Module) -> Dict[str, Any]:
    """
    Повертає group['qna_stats'] саме для тієї param-group, що містить параметри module (квантова група).
    Якщо оптимізатор не QNAAdam або статистики немає — повертає порожній словник.
    """
    try:
        if not isinstance(optimizer, QNAAdam):
            return {}
    except Exception:
        return {}

    qparams = set(p for p in module.parameters())
    for g in optimizer.param_groups:
        gparams = set(p for p in g.get("params", []))
        if qparams & gparams:
            stats = g.get("qna_stats", {}) or {}
            # повертаємо лише очікувані ключі (щоб було стабільно у CSV)
            allow = {
                "scale_mean", "scale_p90", "scale_min", "scale_max",
                "scale_at_min_frac", "scale_at_max_frac",
                "update_norm", "grad_norm", "cos_grad_update"
            }
            return {k: stats.get(k, None) for k in allow}
    return {}


# --------------------------------- MAIN -----------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train quantum classifier (PennyLane + Torch, Inline-DocPS + QNA)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (supports root config.yaml or Code/config.yaml)",
    )
    args = parser.parse_args(argv)
    
    try:
        # Завантаження конфігу
        config = load_project_config(args.config)
        a = config.get("logging", {})
        b = a.get("level", "INFO")
        logger = setup_logger("train_quantum", level = config.get("logging", {}).get("level", "INFO"))
        logger.info("Конфіг завантажено з: %s", args.config)
        _log_config_summary(config, logger)

        # Сіди
        seed = _set_all_seeds(config, logger)
        _enforce_torch_determinism(bool(config.get("project", {}).get("deterministic_torch", False)), logger)

        # Визначаємо фактичне ім'я оптимізатора для тега ран-а
        requested_opt = str(config.get("optim", {}).get("name", "adam")).lower()
        resolved_opt = requested_opt
        if get_opt_from_registry is None and requested_opt not in ("adam", "adamw", "qna_adam", "qnaadam"):
            resolved_opt = "adam"

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
        _validate_quantum_backend(config, model, logger)
        _sanity_check_model(config, model, train_loader, criterion=torch.nn.CrossEntropyLoss(), logger=logger)
        # Оптимізатор
        optimizer = _build_optimizer(config, model, logger)
        criterion = torch.nn.CrossEntropyLoss()

        # Grad clipping (зовнішній). Якщо в QNAAdam увімкнений внутрішній — зовнішній OFF.
        grad_clip = float(config.get("training", {}).get("grad_clip", 1.0))
        qna_clip_inside = False
        if isinstance(optimizer, QNAAdam) and getattr(optimizer, "param_groups", None):
            # зчитуємо будь-яку групу — якщо десь увімкнено, вважаємо "всередині"
            qna_clip_inside = any(bool(g.get("clip_in_optimizer", False)) for g in optimizer.param_groups)
        if qna_clip_inside:
            grad_clip = 0.0
            logger.info("Grad clipping: external OFF (обробляється усередині QNAAdam).")

        # Early Stopping
        es_cfg = (config.get("training", {}).get("early_stopping", {}) or {})
        es_enabled = bool(es_cfg.get("enabled", False)) and (val_loader is not None)
        if es_enabled:
            monitor = str(es_cfg.get("monitor", "val_loss")).lower()
            mode = "min" if "loss" in monitor else str(es_cfg.get("mode", "max")).lower()
            patience = int(es_cfg.get("patience", 5))
            min_delta = float(es_cfg.get("min_delta", 0.0))
            warmup = int(es_cfg.get("warmup_epochs", 0))
            restore_best = bool(es_cfg.get("restore_best", True))
            stopper = EarlyStopper(monitor=monitor, mode=mode, patience=patience,
                                   min_delta=min_delta, warmup_epochs=warmup,
                                   restore_best=restore_best, logger=logger)
            logger.info("EarlyStopping: enabled | monitor=%s | mode=%s | patience=%d | min_delta=%.3g | warmup=%d | restore_best=%s",
                        monitor, mode, patience, min_delta, warmup, restore_best)
        else:
            stopper = None
            if not es_cfg:
                logger.info("EarlyStopping: disabled (no config).")
            elif val_loader is None:
                logger.info("EarlyStopping: disabled (немає валідації).")
            else:
                logger.info("EarlyStopping: disabled (enabled=false).")

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
                 "lr", "shots", "Vtilde",
                 # --- нове: QNA телеметрія (на рівні квантової групи) ---
                 "scale_mean", "scale_p90", "scale_min", "scale_max",
                 "scale_at_min", "scale_at_max",
                 "upd_norm", "cos_gu",
                 "time_ms"]
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

            qmetrics_enabled = bool(config.get("quantum", {}).get("metrics", {}).get("enabled", False)) and _HAS_Q_METRICS
            qmetrics_path = os.path.join(run_paths.root, "qmetrics.jsonl")
            if qmetrics_enabled:
                # Touch-файл і повідомлення в лог — зручно для артефактів
                open(qmetrics_path, "a", encoding="utf-8").close()
                logger.info("Q-metrics enabled → %s", qmetrics_path)
            
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
                running_vt = 0.0
                vt_batches = 0
                # агрегати qna телеметрії
                s_mean = s_p90 = s_min = s_max = 0.0
                s_at_min = s_at_max = 0.0
                upd_norm_acc = cos_gu_acc = 0.0

                steps = 0
                t0 = time.time()

                for step, (xb, yb) in enumerate(train_loader, start=1):
                    steps += 1
                    optimizer.zero_grad(set_to_none=True)

                    # 1) Forward: квант → голова
                    expvals = model.quantum(xb)                       # (B, out_dim), requires_grad=True
                    logits = model.classifier(expvals)                # (B, n_classes)
                    loss = criterion(logits, yb)

                    # Guard від не-фінтних лоссів
                    if not torch.isfinite(loss):
                        logger.warning("Non-finite loss @ epoch=%d step=%d → пропускаю крок.", ep, step)
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    # 2) ∂L/∂E — без глобального backward
                    dL_dE = torch.autograd.grad(loss, expvals, retain_graph=False, create_graph=False)[0]

                    # 3) Inline-DocPS: заповнити theta.grad і отримати пер-параметрний шум
                    doc = model.quantum.backward_inline_docps(
                        xb, dL_dE,
                        reduce_var="param",  # для QNA потрібен пер-параметрний
                    )
                    Vtilde = doc.get("Vtilde", None)
                    var_param = doc.get("var_param", None)

                    if qmetrics_enabled and step == 1:
                        collect_qmetrics(logger, config, model, qmetrics_path, ep, step, xb, expvals)

                    # 4) Передати пер-параметрний шум у QNAAdam (через state[p])
                    if isinstance(optimizer, QNAAdam) and var_param is not None:
                        try:
                            _push_qna_var_to_optimizer(optimizer, model.quantum, var_param, logger)  # type: ignore[attr-defined]
                            for name, p in model.quantum.named_parameters():
                                if name.lower() == "theta":
                                    st = optimizer.state.get(p, {})
                                    V = st.get("qna_var_param", None)
                                    logger.debug(
                                        "QNA push check: param=%s | in_state=%s | V.shape=%s | V.dtype=%s | "
                                        "V[min,mean,p90,max]=[%.3e, %.3e, %.3e, %.3e] | V@device=%s",
                                        name, ("yes" if V is not None else "no"),
                                        (tuple(V.shape) if V is not None else None),
                                        (str(V.dtype) if V is not None else None),
                                        (float(V.min().item()) if V is not None else float("nan")),
                                        (float(V.mean().item()) if V is not None else float("nan")),
                                        (float(torch.quantile(V.flatten(), torch.tensor(0.90)).item()) if V is not None else float("nan")),
                                        (float(V.max().item()) if V is not None else float("nan")),
                                        (str(V.device) if V is not None else None),
                                    )
                                    break

                        except Exception as e:
                            logger.warning("Не вдалось push var_param у QNAAdam: %s", e)

                    # 5) Класична голова: відсікаємо квантову гілку, backward лише в голову
                    expvals_detached = expvals.detach()
                    logits2 = model.classifier(expvals_detached)
                    loss2 = criterion(logits2, yb)
                    loss2.backward()  # градієнти підуть тільки в classifier.*

                    # Норма ∇ ДО кліпінгу
                    gn_raw = _grad_norm(model.parameters())
                    clipped_flag = 0
                    if grad_clip and grad_clip > 0:
                        unclipped = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        if float(unclipped) > float(grad_clip):
                            clipped_flag = 1

                    optimizer.step()

                    # Після step: зчитуємо qna-статистику саме для квантової групи
                    qstats = _extract_qna_stats_for_module(optimizer, model.quantum)  # type: ignore[attr-defined]
                    logger.debug(
                        "QNA STATS: %s",
                        (qstats if qstats else {"warn": "no qna_stats for quantum group"})
                    )
                    # Норма ∇ ПІСЛЯ кліпінгу (для інформації)
                    gn = _grad_norm(model.parameters())
                    acc = _accuracy_from_logits(logits, yb)

                    # Акумулятори
                    running_loss += float(loss.item())
                    running_acc += acc
                    running_gn_raw += gn_raw
                    running_gn += gn
                    clip_count += clipped_flag
                    if Vtilde is not None:
                        running_vt += float(Vtilde)
                        vt_batches += 1

                    # qna агрегати (якщо є)
                    if qstats:
                        s_mean   += float(qstats.get("scale_mean") or 0.0)
                        s_p90    += float(qstats.get("scale_p90") or 1.0)
                        s_min    += float(qstats.get("scale_min") or 1.0)
                        s_max    += float(qstats.get("scale_max") or 1.0)
                        s_at_min += float(qstats.get("scale_at_min_frac") or 0.0)
                        s_at_max += float(qstats.get("scale_at_max_frac") or 0.0)
                        upd_norm_acc += float(qstats.get("update_norm") or 0.0)
                        cos_gu_acc   += float(qstats.get("cos_grad_update") or 0.0)

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
                        (_fmt(Vtilde, ".3e") if Vtilde is not None else ""),
                        # qna stats (квантова група) — можуть бути порожні якщо не QNAAdam
                        _fmt(qstats.get("scale_mean") if qstats else None, ".4f"),
                        _fmt(qstats.get("scale_p90")  if qstats else None, ".4f"),
                        _fmt(qstats.get("scale_min")  if qstats else None, ".4f"),
                        _fmt(qstats.get("scale_max")  if qstats else None, ".4f"),
                        _fmt(qstats.get("scale_at_min_frac") if qstats else None, ".4f"),
                        _fmt(qstats.get("scale_at_max_frac") if qstats else None, ".4f"),
                        _fmt(qstats.get("update_norm") if qstats else None, ".6f"),
                        _fmt(qstats.get("cos_grad_update") if qstats else None, ".4f"),
                        int(elapsed_ms),            # time_ms (ціле)
                    ])

                # Агрегати трену
                train_loss = running_loss / max(steps, 1)
                train_acc = running_acc / max(steps, 1)
                train_gn_raw = running_gn_raw / max(steps, 1)
                train_gn = running_gn / max(steps, 1)
                clip_rate = clip_count / max(steps, 1)
                vt_epoch = (running_vt / vt_batches) if vt_batches > 0 else None

                # усереднення qna-метрик по батчах
                if steps > 0:
                    s_mean_e   = s_mean / steps
                    s_p90_e    = s_p90 / steps
                    s_min_e    = s_min / steps
                    s_max_e    = s_max / steps
                    s_at_min_e = s_at_min / steps
                    s_at_max_e = s_at_max / steps
                    upd_norm_e = upd_norm_acc / steps
                    cos_gu_e   = cos_gu_acc / steps
                else:
                    s_mean_e = s_p90_e = s_min_e = s_max_e = s_at_min_e = s_at_max_e = upd_norm_e = cos_gu_e = None

                current_shots = str(_get_current_shots(model))
                logger.info(
                    "[Epoch %d] train: loss=%.4f, acc=%.3f, ‖∇‖raw=%.3f, ‖∇‖=%.3f, clip_rate=%.2f, V~=%s, shots=%s | scale_mean=%.3f p90=%.3f",
                    ep, train_loss, train_acc, train_gn_raw, train_gn, clip_rate,
                    (f"{vt_epoch:.3e}" if vt_epoch is not None else "—"),
                    current_shots,
                    (s_mean_e if s_mean_e is not None else float("nan")),
                    (s_p90_e if s_p90_e is not None else float("nan")),
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
                    (_fmt(vt_epoch, ".3e") if vt_epoch is not None else ""),
                    _fmt(s_mean_e, ".4f"),
                    _fmt(s_p90_e,  ".4f"),
                    _fmt(s_min_e,  ".4f"),
                    _fmt(s_max_e,  ".4f"),
                    _fmt(s_at_min_e, ".4f"),
                    _fmt(s_at_max_e, ".4f"),
                    _fmt(upd_norm_e, ".6f"),
                    _fmt(cos_gu_e, ".4f"),
                    0,
                ])
                csv_file.flush()

                # ---- VAL (опційно) ----
                val_loss = None
                val_acc = None
                if val_loader is not None:
                    original_shots = _get_current_shots(model)
                    try:
                        if eval_shots is not None:
                            _set_model_shots(model, eval_shots, logger=logger)

                        model.eval()
                        v_running_loss, v_running_acc, v_steps = 0.0, 0.0, 0
                        with torch.no_grad():
                            for xb, yb in val_loader:
                                expvals_v = model.quantum(xb)
                                logits_v = model.classifier(expvals_v)
                                loss_v = criterion(logits_v, yb)
                                v_running_loss += float(loss_v.item())
                                v_running_acc += _accuracy_from_logits(logits_v, yb)
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
                            "",                       # Vtilde
                            "", "", "", "", "", "", "", "",  # заповнювачі для qna-колонок + time_ms
                        ])
                        csv_file.flush()

                        # Найкращий чекпойнт по val_loss (для зворотної сумісності)
                        if val_loss < best_val:
                            best_val = val_loss
                            torch.save(model.state_dict(), run_paths.model_pt)
                            logger.info("Збережено найкращу модель (val_loss=%.6f) → %s", best_val, run_paths.model_pt)

                        # Early stopping (якщо увімкнено)
                        if stopper is not None:
                            monitor_value = float(val_loss) if (stopper.monitor == "val_loss") else float(val_acc)
                            if stopper.step(monitor_value, ep, model):
                                logger.info("Early stopping спрацював на епосі %d.", ep)
                                # Після відновлення best_state — збережемо модель
                                torch.save(model.state_dict(), run_paths.model_pt)
                                break

                    finally:
                        # Повертаємо shots навіть у разі виключення
                        if eval_shots is not None:
                            _set_model_shots(model, original_shots, logger=logger)

                # Періодичне збереження (коли без валідації)
                if (val_loader is None) and (save_every > 0) and (ep % save_every == 0):
                    torch.save(model.state_dict(), run_paths.model_pt)
                    logger.info("Збережено модель (кожні %d епох) → %s", save_every, run_paths.model_pt)

                # Early stop вихід із епох-лупу
                if stopper is not None and stopper.best_state is not None and stopper.bad_epochs >= stopper.patience:
                    break

                # Профайлінг епохи → JSONL
                epoch_ms = int((time.time() - epoch_t0) * 1000)
                profile_entry = {"epoch": ep, "time_ms": epoch_ms, "shots": _get_current_shots(model), "acc": val_acc, "val_loss": val_loss}
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

def collect_qmetrics(logger, config, model, qmetrics_path, ep, step, xb, expvals):
    try:
        with torch.no_grad():
                                # 1) φ-метрики (якщо xb вже є кутами)
            phi_m = compute_phi_metrics(xb, include_per_pc_std=False,
                                                            angle_max=config.get("angles", {}).get("angle_max", None))
                                # 2) виходи (expvals)
            out_m = compute_expval_metrics(expvals, prefix="out")
                            # 3) градієнти θ (після backward_inline_docps, до step)
        theta_param = None
        for p in model.quantum.parameters():  # type: ignore[attr-defined]
            if isinstance(p, torch.nn.Parameter) and p.ndim == 2:
                theta_param = p
                break
        theta_m = compute_theta_grad_metrics(theta_param) if theta_param is not None else {}
        payload = merge_dicts({"epoch": ep, "step": step}, phi_m, out_m, theta_m)
        with open(qmetrics_path, "a", encoding="utf-8") as fqm:
            fqm.write(json.dumps(payload) + "\n")
    except Exception as qe:
        logger.warning("Q-metrics skip: %s", qe)


if __name__ == "__main__":
    sys.exit(main())
