# Code/quantum/__examples__/sanity_check.py
# -*- coding: utf-8 -*-

import os
import math
import argparse
from typing import Dict, Any

import torch

from Code.logger import setup_logger
from Code.quantum_layer.devices import seed_everything
from Code.quantum_layer.quantum_layer import QuantumLayer
from Code.quantum_layer.metrics_bridge import (
    compute_phi_metrics,
    compute_expval_metrics,
    compute_theta_grad_metrics,
    merge_dicts,
    pack_and_log,
)

# Прагматичне читання конфіга: спершу пробуємо наш util, інакше — YAML напряму
def load_cfg(path: str) -> Dict[str, Any]:
    try:
        from Code.utils.io_utils import load_project_config  # ваш утил
        return load_project_config(path)
    except Exception:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def guess_config_path(cli_path: str | None) -> str:
    if cli_path and os.path.exists(cli_path):
        return cli_path
    for cand in ("config.yaml", "project.yaml"):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError("Не знайдено config.yaml / project.yaml. Передайте шлях через --config")


def main():
    parser = argparse.ArgumentParser(description="Sanity-check квантового шару")
    parser.add_argument("--config", type=str, default=None, help="Шлях до YAML-конфіга (config.yaml або project.yaml)")
    parser.add_argument("--batch", type=int, default=4, help="Розмір батчу для перевірки")
    parser.add_argument("--shots", type=int, default=None, help="Оверрайд shots (за замовчуванням з конфіга)")
    parser.add_argument("--seed", type=int, default=None, help="Оверрайд сид (за замовчуванням з конфіга)")
    args = parser.parse_args()

    cfg_path = guess_config_path(args.config)
    log = setup_logger("sanity.quantum")

    log.info("=== SANITY: start ===")
    log.info("Config path: %s", cfg_path)
    cfg = load_cfg(cfg_path)

    # Сідимо всі RNG (додатково до сидів пристрою)
    try:
        base_seed = int(cfg.get("project", {}).get("seed", 42))
    except Exception:
        base_seed = 42
    if args.seed is not None:
        base_seed = int(args.seed)
    seed_everything(base_seed)
    log.info("Global RNG seeded with %d", base_seed)

    # Створюємо шар із конфіга
    layer = QuantumLayer.from_config(cfg, logger=log)
    if args.shots is not None:
        layer.set_shots(int(args.shots))  # перевибудує QNode під капотом

    # Друкуемо мета-інфо
    meta = layer.meta
    log.info("Layer meta: n_qubits=%d, n_layers=%d, output_dim=%d, device=%s, shots=%s, encoding=%s, reupload=%s, topology=%s",
             layer.n_qubits, layer.n_layers, layer.output_dim, meta.get("device"), str(layer.spec.shots),
             meta.get("encoding"), str(meta.get("reupload")), meta.get("topology"))

    # Готуємо випадкові φ ∈ [-π, π]
    B = int(args.batch)
    n_qubits = layer.n_qubits
    g = torch.Generator().manual_seed(base_seed + 1)
    phi = (torch.rand((B, n_qubits), generator=g, dtype=torch.float32) * (2 * math.pi)) - math.pi

    # Forward
    out = layer(phi)

    # Метрики φ та виходів
    phi_metrics = compute_phi_metrics(phi, angle_max=cfg.get("angles", {}).get("angle_max", math.pi))
    out_metrics = compute_expval_metrics(out, prefix="expval")

    # Backward для sanity (простий loss)
    loss = (out ** 2).sum()
    loss.backward()
    theta_metrics = compute_theta_grad_metrics(layer.theta)

    all_metrics = merge_dicts(
        {"loss": float(loss.item())},
        phi_metrics,
        out_metrics,
        theta_metrics,
    )

    pack_and_log(log, all_metrics, prefix="sanity/quantum", level="INFO")

    # Короткий друк у консоль (щоб було видно без лог-файлу)
    print(f"[SANITY] out.shape={tuple(out.shape)}, out.min={float(out.min()):.4f}, out.max={float(out.max()):.4f}")
    print(f"[SANITY] loss={float(loss.item()):.6f}")

    log.info("=== SANITY: done ===")


if __name__ == "__main__":
    main()