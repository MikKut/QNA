# Code/quantum/devices.py
# -*- coding: utf-8 -*-
"""
devices.py — єдине місце правди про квантовий пристрій у проєкті.

Призначення:
  • Централізовано створювати qml.device(...) з заданими name/wires/shots/seed.
  • Давати зручні утиліти для відтворюваності (seed_everything) і маніпуляцій зі shots.
  • Інкапсулювати перевірки/інваріанти (n_qubits > 0, shots > 0 або None).

Нотатки:
  • QNode міцно прив'язаний до конкретного device. Якщо ви змінюєте shots/seed —
    створіть НОВИЙ device і перевиберіть QNode (через qnode_factory).
  • У аналітичному режимі (shots is None) деякі версії PennyLane не люблять,
    коли їм явно передають shots=None або seed — тому ми їх не передаємо.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Mapping, Any, Dict
import logging
import random
import numpy as np

try:
    from Code.logger import setup_logger
    _log = setup_logger("quantum.devices")
except Exception:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _log = logging.getLogger("quantum.devices")

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # дозволяє використовувати модуль без Torch (лише для seed_everything)

import pennylane as qml


# --------------------------- Публічні типи/структури ---------------------------

@dataclass(frozen=True)
class DeviceSpec:
    """
    Опис конфігурації квантового пристрою.

    Args:
        name:     назва backend'а (у нашому проєкті — "default.qubit")
        n_qubits: кількість дротів (узгоджено з PCA/AngleEncoder)
        shots:    None => аналітичний режим; або додатне ціле (кількість вимірювань)
        seed:     сид для відтворюваності вимірювань (device-level seed)
    """
    name: str = "default.qubit"
    n_qubits: int = 8
    shots: Optional[int] = 100
    seed: int = 42


# --------------------------- Внутрішні валідації ---------------------------

def _validate_spec(spec: DeviceSpec) -> None:
    if not isinstance(spec.name, str) or not spec.name:
        raise ValueError("DeviceSpec.name must be a non-empty string.")
    if spec.n_qubits <= 0:
        raise ValueError(f"DeviceSpec.n_qubits must be > 0, got {spec.n_qubits}.")
    if spec.shots is not None and (not isinstance(spec.shots, int) or spec.shots <= 0):
        raise ValueError(f"DeviceSpec.shots must be None or positive int, got {spec.shots}.")
    # Політика проєкту: працюємо на default.qubit — але не забороняємо явно
    if spec.name != "default.qubit":
        _log.warning("Non-standard device name '%s' (project targets 'default.qubit').", spec.name)


# --------------------------- Фабрика пристрою ---------------------------

def _build_device_kwargs(spec: DeviceSpec) -> Dict[str, Any]:
    """
    Формує kwargs для qml.device з урахуванням особливостей PL:
    • в аналітичному режимі НЕ передаємо 'shots' і 'seed';
    • у шот-режимі передаємо і 'shots', і 'seed' (якщо є).
    """
    kwargs: Dict[str, Any] = {"wires": spec.n_qubits}
    if spec.shots is not None:
        kwargs["shots"] = int(spec.shots)
        # seed має сенс лише коли є дискретні вимірювання (shots)
        if isinstance(spec.seed, int):
            kwargs["seed"] = int(spec.seed)
    return kwargs


def make_device(spec: DeviceSpec) -> qml.Device:
    """
    Створює і повертає qml.device(spec.name, wires=..., [shots=...], [seed=...]).

    У аналітичному режимі (shots=None) не передаємо ні shots, ні seed.
    """
    _validate_spec(spec)
    kwargs = _build_device_kwargs(spec)
    _log.debug(
        "Creating device: name=%s, wires=%d, %s",
        spec.name, spec.n_qubits,
        "analytic" if spec.shots is None else f"shots={spec.shots}, seed={spec.seed}"
    )

    try:
        dev = qml.device(spec.name, **kwargs)
    except Exception as e:  # pragma: no cover
        _log.exception("qml.device(...) failed | name=%s kwargs=%s", spec.name, kwargs)
        raise

    # Перевірка кількості дротів
    try:
        n_wires = len(dev.wires)
    except Exception:  # pragma: no cover
        n_wires = getattr(dev, "num_wires", spec.n_qubits)
    if n_wires != spec.n_qubits:
        raise RuntimeError(
            f"Device reports {n_wires} wires, but spec.n_qubits={spec.n_qubits}."
        )

    return dev


# --------------------------- Утиліти для керування Spec ---------------------------

def with_shots(spec: DeviceSpec, shots: Optional[int]) -> DeviceSpec:
    """
    Повертає новий DeviceSpec з оновленими shots (None або додатне ціле).
    Після цього СТВОРІТЬ новий device і перевиберіть QNode (QNode прив'язаний до device).
    """
    if shots is not None and (not isinstance(shots, int) or shots <= 0):
        raise ValueError(f"shots must be None or positive int, got {shots}.")
    return replace(spec, shots=shots)


def with_seed(spec: DeviceSpec, seed: int) -> DeviceSpec:
    """
    Повертає новий DeviceSpec з оновленим seed (для ресемплу шуму вимірювань).
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed}.")
    return replace(spec, seed=seed)


def is_analytic(spec: DeviceSpec) -> bool:
    """True, якщо обрано аналітичний режим (shots is None)."""
    return spec.shots is None


def device_summary(dev: qml.Device) -> Dict[str, Any]:
    """
    Повертає коротку інформацію про пристрій (для логів/діагностики).
    Сумісно з різними версіями PL, де shots може бути об'єктом.
    """
    try:
        n_wires = len(dev.wires)
    except Exception:  # pragma: no cover
        n_wires = getattr(dev, "num_wires", None)

    shots_val = getattr(dev, "shots", None)
    try:
        # Якщо це не int/None, спробуємо взяти total_shots / copies / shots
        if shots_val is not None and not isinstance(shots_val, int):
            shots_val = (
                getattr(shots_val, "total_shots", None)
                or getattr(shots_val, "copies", None)
                or getattr(shots_val, "shots", None)
                or str(shots_val)
            )
    except Exception:  # pragma: no cover
        pass

    return {
        "name": getattr(dev, "name", type(dev).__name__),
        "n_wires": n_wires,
        "shots": shots_val,
        "analytic": shots_val in (None, 0),
    }


# --------------------------- Сидінг для відтворюваності ---------------------------

def seed_everything(seed: int) -> None:
    """
    Виставляє сид для Python, NumPy і (за наявності) Torch.
    Це доповнює seed пристрою (який задається у DeviceSpec).
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed}.")

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)
    _log.debug("Global RNGs seeded | seed=%d (python, numpy%s)", seed, ", torch" if torch is not None else "")


# --------------------------- Зручний адаптер із config.yaml ---------------------------

def _parse_yaml_shots(val: Any) -> Optional[int]:
    """
    Перетворює значення з YAML у коректний shots:
      • None / "none" / "null" / "" -> None (аналітичний режим)
      • додатне ціле -> int
    """
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("none", "null", ""):
            return None
        try:
            ival = int(s)
        except Exception:
            raise ValueError(f"Invalid 'quantum.shots' string value: {val!r}")
        else:
            val = ival
    if isinstance(val, (int, np.integer)):
        ival = int(val)
        if ival <= 0:
            raise ValueError(f"'quantum.shots' must be positive or null, got {ival}.")
        return ival
    raise ValueError(f"Unsupported 'quantum.shots' type: {type(val)}")


def spec_from_config(cfg: Mapping[str, Any]) -> DeviceSpec:
    """
    Будує DeviceSpec з dict-конфіга (вже завантаженого з YAML).

    Очікувані ключі:
      cfg["quantum"]["device"] -> str (наприклад, "default.qubit")
      cfg["quantum"]["shots"]  -> int | null | "none"
      cfg["pca"]["dim"]        -> int (n_qubits)
      cfg["project"]["seed"]   -> int
    """
    q = cfg.get("quantum", {}) or {}
    pca = cfg.get("pca", {}) or {}
    proj = cfg.get("project", {}) or {}

    name = str(q.get("device", "default.qubit"))
    n_qubits = int(pca.get("dim", 8))
    shots = _parse_yaml_shots(q.get("shots", 100))
    seed = int(proj.get("seed", 42))

    spec = DeviceSpec(name=name, n_qubits=n_qubits, shots=shots, seed=seed)
    _validate_spec(spec)
    return spec


__all__ = [
    "DeviceSpec",
    "make_device",
    "with_shots",
    "with_seed",
    "is_analytic",
    "device_summary",
    "seed_everything",
    "spec_from_config",
]
