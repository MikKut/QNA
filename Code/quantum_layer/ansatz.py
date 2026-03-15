# Code/quantum/ansatz.py
# -*- coding: utf-8 -*-
"""
ansatz.py — тренована (дані-незалежна) частина квантової схеми.
Вміст:
  - Ентанглери: ring_entangle / linear_entangle
  - Шар HEA: hardware_efficient_layer(theta_row, topology=...)
  - Повтор шарів: hardware_efficient_ansatz(theta, topology=...)

Принципи:
  - НЕ кодує дані (це робить encodings.py).
  - НЕ вимірює (це робиться у QNode/фабриці).
  - Працює виключно з тренованими параметрами θ.

Параметри:
  - theta: torch.Tensor форми (L, n_qubits)
  - theta_row: torch.Tensor форми (n_qubits,)
"""

from __future__ import annotations

from typing import Literal
import torch
import pennylane as qml

Topology = Literal["ring", "linear"]


# --------- Перевірки форм --------- #

def _assert_theta_row(theta_row: torch.Tensor) -> None:
    if not isinstance(theta_row, torch.Tensor):
        raise TypeError(f"theta_row must be torch.Tensor, got {type(theta_row)}.")
    if theta_row.ndim != 1:
        raise ValueError(f"theta_row must be 1D (n_qubits,), got shape={tuple(theta_row.shape)}.")
    if theta_row.numel() == 0:
        raise ValueError("theta_row must have positive length (n_qubits > 0).")


def _assert_theta(theta: torch.Tensor) -> None:
    if not isinstance(theta, torch.Tensor):
        raise TypeError(f"theta must be torch.Tensor, got {type(theta)}.")
    if theta.ndim != 2:
        raise ValueError(f"theta must be 2D (L, n_qubits), got shape={tuple(theta.shape)}.")
    L, n_qubits = theta.shape
    if L <= 0 or n_qubits <= 0:
        raise ValueError(f"theta must have L>0 and n_qubits>0, got L={L}, n_qubits={n_qubits}.")


# --------- Ентанглери --------- #

def ring_entangle(n_qubits: int) -> None:
    """
    CNOT-кільце: 0→1, 1→2, ..., (n-2)→(n-1), (n-1)→0.
    Викликати всередині qfunc/QNode.
    """
    if n_qubits <= 1:
        return
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])


def linear_entangle(n_qubits: int) -> None:
    """
    Лінійний ланцюжок CNOT: 0→1, 1→2, ..., (n-2)→(n-1).
    """
    if n_qubits <= 1:
        return
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])


def _apply_entangler(n_qubits: int, topology: Topology = "ring") -> None:
    if topology == "ring":
        ring_entangle(n_qubits)
    elif topology == "linear":
        linear_entangle(n_qubits)
    else:
        raise ValueError(f"Unsupported topology '{topology}'. Use 'ring' or 'linear'.")


# --------- HEA: шар і повний анзац --------- #

def hardware_efficient_layer(
    theta_row: torch.Tensor,
    topology: Topology = "ring",
) -> None:
    """
    Один шар HEA: RY(θ_i) на кожному кубіті i, далі шар CNOT за заданою топологією.

    Args:
        theta_row: (n_qubits,), trainable кути для одно-кубітних RY.
        topology:  'ring' | 'linear' — схема заплутування.
    """
    _assert_theta_row(theta_row)
    n_qubits: int = theta_row.shape[0]

    # Одно-кубітні треновані обертання
    for i in range(n_qubits):
        qml.RY(theta_row[i], wires=i)

    # Заплутування
    _apply_entangler(n_qubits, topology=topology)


def hardware_efficient_ansatz(
    theta: torch.Tensor,
    topology: Topology = "ring",
) -> None:
    """
    Повний HEA з L шарів.

    Args:
        theta:    (L, n_qubits) — всі треновані параметри.
        topology: 'ring' | 'linear'
    """
    _assert_theta(theta)
    L, _ = theta.shape
    for l in range(L):
        hardware_efficient_layer(theta[l], topology=topology)


__all__ = [
    "Topology",
    "ring_entangle",
    "linear_entangle",
    "hardware_efficient_layer",
    "hardware_efficient_ansatz",
]
