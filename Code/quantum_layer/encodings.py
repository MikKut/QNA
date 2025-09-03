# Code/quantum/encodings.py
# -*- coding: utf-8 -*-
"""
encodings.py — тонкий прошарок між готовими кутами φ та квантовою схемою.
Використовується всередині QNode (qml.qnode) для введення даних у вигляді
обертань на кожному кубіті.

Ключові принципи:
- НЕ змінює дані: жодних нормалізацій/clip — їх вже зроблено у AngleEncoder.
- Працює з одним зразком за раз: angles.shape == (n_qubits,).
- Дає прозоре API: apply_ry_encoding / rx / rz, а також encode_data(..) з диспетчером.
- Має зручний re-uploading: encode_data_reupload(angles, L, kind="ry").

Приклади використання в QNode:
    apply_ry_encoding(angles)                       # разове кодування
    encode_data_reupload(angles, L, kind="ry")      # φ перед кожним з L шарів

Автор: QNA проект
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Callable, Literal, Dict

import torch
import pennylane as qml

EncoderKind = Literal["ry", "rx", "rz", "xy"]


# --------- Внутрішні утиліти --------- #

def _normalize_wires(wires: Optional[Sequence[int]], n_qubits: int) -> Sequence[int]:
    """Повертає послідовність дротів довжини n_qubits."""
    if wires is None:
        return list(range(n_qubits))
    if len(wires) != n_qubits:
        raise ValueError(f"len(wires) must equal n_qubits={n_qubits}, got {len(wires)}.")
    return wires


def _assert_angles_vector(angles: torch.Tensor) -> None:
    """Гарантує, що angles — 1D-вектор розміру (n_qubits,)."""
    if not isinstance(angles, torch.Tensor):
        raise TypeError(f"angles must be a torch.Tensor, got {type(angles)}.")
    if angles.ndim != 1:
        raise ValueError(f"angles must be 1D (n_qubits,), got shape={tuple(angles.shape)}.")
    # Діапазон перевіряти НЕ обов'язково (AngleEncoder гарантує [-π, π]),
    # але лишимо легкий дебаг-асерт:
    # if torch.any(angles < -math.pi) or torch.any(angles > math.pi):
    #     raise ValueError("angles must be in [-π, π].")


# --------- Базові кодування --------- #

def apply_ry_encoding(angles: torch.Tensor, wires: Optional[Sequence[int]] = None) -> None:
    """
    Кодує φ у повороти RY: на дроті wires[i] застосовує qml.RY(angles[i]).
    Викликати лише всередині qfunc/QNode.

    Args:
        angles: torch.Tensor, shape (n_qubits,), значення у [-π, π].
        wires:  послідовність дротів довжини n_qubits або None (0..n_qubits-1).
    """
    _assert_angles_vector(angles)
    n_qubits = angles.shape[0]
    wires = _normalize_wires(wires, n_qubits)

    for i, w in enumerate(wires):
        qml.RY(angles[i], wires=w)


def apply_rx_encoding(angles: torch.Tensor, wires: Optional[Sequence[int]] = None) -> None:
    """
    Кодує φ у повороти RX: qml.RX(angles[i]) на кожному дроті.
    """
    _assert_angles_vector(angles)
    n_qubits = angles.shape[0]
    wires = _normalize_wires(wires, n_qubits)

    for i, w in enumerate(wires):
        qml.RX(angles[i], wires=w)


def apply_rz_encoding(angles: torch.Tensor, wires: Optional[Sequence[int]] = None) -> None:
    """
    Кодує φ у повороти RZ: qml.RZ(angles[i]) на кожному дроті.
    Зауваження: чисте RZ не змінює популяції у базисі Z до заплутування/інших осей,
    але може бути корисним у комбінації з HEA/entanglement.
    """
    _assert_angles_vector(angles)
    n_qubits = angles.shape[0]
    wires = _normalize_wires(wires, n_qubits)

    for i, w in enumerate(wires):
        qml.RZ(angles[i], wires=w)


def apply_xy_encoding(
    angles: torch.Tensor,
    wires: Optional[Sequence[int]] = None,
    alpha: float = 0.5,
) -> None:
    """
    Комбіноване кодування: RX(alpha*φ), потім RY((1-alpha)*φ) на кожному дроті.
    Дає трохи вищу виразність без зміни кількості параметрів.

    Args:
        angles: (n_qubits,)
        wires:  послідовність дротів або None
        alpha:  частка кута для RX у [0,1]. За замовчуванням 0.5.
    """
    _assert_angles_vector(angles)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}.")
    n_qubits = angles.shape[0]
    wires = _normalize_wires(wires, n_qubits)

    for i, w in enumerate(wires):
        if alpha != 0.0:
            qml.RX(alpha * angles[i], wires=w)
        if alpha != 1.0:
            qml.RY((1.0 - alpha) * angles[i], wires=w)


# --------- Диспетчер кодувань --------- #

_ENCODERS: Dict[str, Callable[..., None]] = {
    "ry": apply_ry_encoding,
    "rx": apply_rx_encoding,
    "rz": apply_rz_encoding,
    "xy": apply_xy_encoding,
}


def get_encoder(kind: EncoderKind | str) -> Callable[..., None]:
    """
    Повертає функцію-кодувальник за назвою.
    Підтримуються: "ry", "rx", "rz", "xy".
    """
    try:
        return _ENCODERS[str(kind).lower()]
    except KeyError as e:
        supported = ", ".join(sorted(_ENCODERS.keys()))
        raise ValueError(f"Unsupported encoder kind='{kind}'. Supported: {supported}.") from e


def encode_data(
    angles: torch.Tensor,
    kind: EncoderKind | str = "ry",
    wires: Optional[Sequence[int]] = None,
    **kwargs,
) -> None:
    """
    Універсальний вхід: кодує φ за допомогою обраного енкодера (разове кодування).
    Викликає одну з apply_*_encoding(...).

    Args:
        angles: (n_qubits,), torch.Tensor
        kind:   "ry" | "rx" | "rz" | "xy"
        wires:  None або послідовність довжини n_qubits
        kwargs: додаткові параметри для конкретного енкодера (напр., alpha для "xy")
    """
    encoder = get_encoder(kind)
    encoder(angles, wires=wires, **kwargs)


# --------- Re-uploading (повторне введення φ у кожному шарі) --------- #

def encode_data_reupload(
    angles: torch.Tensor,
    L: int,
    kind: EncoderKind | str = "ry",
    wires: Optional[Sequence[int]] = None,
    **kwargs,
) -> None:
    """
    Вводить ті самі кути φ перед КОЖНИМ з L шарів анзацу.
    Це зручно викликати безпосередньо в QNode, коли між шарами йде тренований блок.

    Приклад у QNode (ескіз):
        for l in range(L):
            encode_data(angles, kind="ry")        # ← це робить ця функція
            hardware_efficient_layer(theta[l])    # ваш тренований блок + entanglement

    Args:
        angles: torch.Tensor, (n_qubits,)
        L:      кількість повторів (шарів HEA)
        kind:   тип енкодера: "ry" | "rx" | "rz" | "xy"
        wires:  None або послідовність довжини n_qubits
        kwargs: додаткові параметри для енкодера (напр., alpha для "xy")
    """
    if not isinstance(L, int) or L <= 0:
        raise ValueError(f"L must be a positive int, got {L}.")

    encoder = get_encoder(kind)
    _assert_angles_vector(angles)
    n_qubits = angles.shape[0]
    wires = _normalize_wires(wires, n_qubits)

    for _ in range(L):
        encoder(angles, wires=wires, **kwargs)


__all__ = [
    "apply_ry_encoding",
    "apply_rx_encoding",
    "apply_rz_encoding",
    "apply_xy_encoding",
    "get_encoder",
    "encode_data",
    "encode_data_reupload",
]
