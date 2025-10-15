# Code/quantum/qnode_factory.py
# -*- coding: utf-8 -*-
"""
qnode_factory.py — фабрика QNode для PennyLane з інтерфейсом Torch.

Роль:
  • Збирає одно-зразкову квантову схему (кодування φ → HEA-анзац → вимірювання).
  • Інкапсулює бекенд (device, shots, seed), тип кодування, топологію і набір вимірювань.
  • Повертає callable qnode(angles, theta) та meta-інформацію.

Базова поведінка проєкту:
  • encoding="ry", reupload=True, topology="ring", measurement="Z"/"ZX",
    diff_method="parameter-shift", interface="torch".

Інтерфейси:
  angles: torch.Tensor форми (n_qubits,) — кути з AngleEncoder ([-π, π])
  theta:  torch.Tensor форми (L, n_qubits) — треновані параметри HEA

Вихід:
  При measurement_mode="expval":
      torch.Tensor форми (n_meas * n_qubits,)
  При measurement_mode="expval_var":
      torch.Tensor форми (2 * n_meas * n_qubits,)
      (спершу всі expval у стандартному порядку, потім усі var у тому ж порядку)
"""

from __future__ import annotations

from typing import Optional, Tuple, Sequence, Dict, Any, Literal, Callable, List
import logging

import torch
import pennylane as qml

from .devices import DeviceSpec, make_device, device_summary
from .encodings import encode_data
from .ansatz import hardware_efficient_layer
from Code.logger import setup_logger
logger = setup_logger("quantum.devices")


Topology = Literal["ring", "linear"]
EncodingKind = Literal["ry", "rx", "rz", "xy"]
MeasurementMode = Literal["expval", "expval_var"]  # expval only, або expval+var за один виклик
MeasurementKind = Literal["Z", "ZX", "ZXY"]


# ----------------------------- Внутрішні утиліти ----------------------------- #

def _normalize_measurement(measurement: str | Sequence[str]) -> Tuple[Tuple[str, ...], int]:
    """
    Перетворює вхідну специфікацію вимірювань у кортеж осей ('Z','X','Y') та їх кількість.
    Дозволені варіанти: "Z", "ZX", "ZXY", або послідовність {"Z","X","Y"}.

    Повертає:
        axes:  кортеж осей, у верхньому регістрі, без дублікатів, у заданому порядку
        n_meas: len(axes)
    """
    if isinstance(measurement, (list, tuple)):
        axes = tuple(str(x).upper() for x in measurement)
    elif isinstance(measurement, str):
        axes = tuple(ch for ch in measurement.upper())
    else:
        raise ValueError(f"Unsupported measurement spec type: {type(measurement)}.")

    allowed = {"Z", "X", "Y"}
    for a in axes:
        if a not in allowed:
            raise ValueError(f"Unsupported measurement axis '{a}'. Allowed: {sorted(allowed)}")

    # Прибираємо можливі дублікати, зберігаючи порядок:
    seen = set()
    uniq_axes: List[str] = []
    for a in axes:
        if a not in seen:
            uniq_axes.append(a)
            seen.add(a)
    return tuple(uniq_axes), len(uniq_axes)


def _expected_output_dim(n_qubits: int, n_meas: int) -> int:
    return n_meas * n_qubits


def _assert_shapes(angles: torch.Tensor, theta: torch.Tensor, n_layers_expected: int, n_qubits: int) -> None:
    """
    Легка перевірка форм усередині qfunc; викликається в рантаймі для зрозумілих помилок.
    """
    if not isinstance(angles, torch.Tensor):
        raise TypeError(f"angles must be torch.Tensor, got {type(angles)}.")
    if not isinstance(theta, torch.Tensor):
        raise TypeError(f"theta must be torch.Tensor, got {type(theta)}.")

    if angles.ndim != 1 or angles.shape[0] != n_qubits:
        raise ValueError(f"angles must have shape (n_qubits,), got {tuple(angles.shape)}; n_qubits={n_qubits}.")

    if theta.ndim != 2 or theta.shape[0] != n_layers_expected or theta.shape[1] != n_qubits:
        raise ValueError(
            f"theta must have shape (L, n_qubits)=(%d,%d), got %s."
            % (n_layers_expected, n_qubits, tuple(theta.shape))
        )

def _axis_to_obs(axis: str, wire: int) -> qml.operation.Operator:
    """Відповідність осі Паулі до спостережуваного оператора PennyLane."""
    if axis == "Z":
        return qml.PauliZ(wire)
    if axis == "X":
        return qml.PauliX(wire)
    if axis == "Y":
        return qml.PauliY(wire)
    raise RuntimeError(f"Unexpected axis '{axis}' in measurement.")

def build_measurement_ops(axes: Tuple[str, ...], n_qubits: int) -> tuple[list, list]:
    """
    Побудувати впорядковані списки вимірювань для expval та var у стандартному порядку:
      спочатку всі Z по дротах 0..n-1, потім X, потім Y.

    Повертає:
        exp_ops: [qml.expval(Pauli*), ...] довжини n_meas*n_qubits
        var_ops: [qml.var(Pauli*),    ...] довжини n_meas*n_qubits (той самий порядок)
    """
    exp_ops = [qml.expval(_axis_to_obs(ax, i)) for ax in axes for i in range(n_qubits)]
    var_ops = [qml.var(_axis_to_obs(ax, i)) for ax in axes for i in range(n_qubits)]
    return exp_ops, var_ops

# ----------------------------- Публічна фабрика ------------------------------ #

def create_qnode(
    spec: DeviceSpec,
    n_layers: int,
    topology: Topology = "ring",
    encoding: EncodingKind | str = "ry",
    *,
    reupload: bool = True,
    measurement: MeasurementKind | Sequence[str] = "Z",
    diff_method: Literal["parameter-shift", "best"] = "parameter-shift",
    interface: Literal["torch"] = "torch",
    encoding_kwargs: Optional[Dict[str, Any]] = None,
    measurement_mode: MeasurementMode = "expval"
) -> Tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Dict[str, Any]]:
    """
    Створює QNode з інтерфейсом Torch для одно-зразкової схеми:
      [encode φ] → [L шарів HEA] → [expval по осях measurement].

    Args:
        spec:          DeviceSpec (backend, wires, shots, seed)
        n_layers:      глибина HEA (L > 0)
        topology:      'ring' | 'linear' — етанглер у HEA
        encoding:      'ry'|'rx'|'rz'|'xy' — тип кодування φ
        reupload:      True → вводити φ перед кожним шаром HEA; False → один раз на початку
        measurement:   'Z' (база), 'ZX', 'ZXY' або послідовність осей ('Z','X',...)
        diff_method:   'parameter-shift' (рекомендовано з shots) або 'best'
        interface:     лише 'torch' підтримується в цьому проєкті
        encoding_kwargs: додаткові параметри енкодера (напр., alpha для 'xy')
        measurement_mode: "expval" (тільки середні) або "expval_var" (середні + дисперсії за один виклик)

    Returns:
        qnode:  callable (angles: (n_qubits,), theta: (L,n_qubits)) -> torch.Tensor[dim]
               де dim = out_dim (для "expval") або 2*out_dim (для "expval_var")
        meta:   словник з довідковою інформацією (axes, output_dim, device info, тощо)
    """
    if n_layers <= 0:
        raise ValueError(f"n_layers must be > 0, got {n_layers}.")
    if interface != "torch":
        raise ValueError(f"Only interface='torch' is supported, got '{interface}'.")

    axes, n_meas = _normalize_measurement(measurement)
    n_qubits = spec.n_qubits
    out_dim = _expected_output_dim(n_qubits, n_meas)
    want_var = (measurement_mode == "expval_var")

    dev = make_device(spec)
    dev_info = device_summary(dev)
    enc_kwargs = dict(encoding_kwargs or {})

    # ------------------------- Визначаємо qfunc ------------------------- #

    @qml.qnode(dev, interface=interface, diff_method=diff_method)
    def qfunc(angles: torch.Tensor, theta: torch.Tensor) -> Any:
        """
        angles: (n_qubits,) — кути φ
        theta:  (L, n_qubits) — параметри HEA
        """
        _assert_shapes(angles, theta, n_layers_expected=n_layers, n_qubits=n_qubits)

        if reupload:
            # Повторне введення φ перед кожним шаром
            for l in range(n_layers):
                encode_data(angles, kind=encoding, **enc_kwargs)
                hardware_efficient_layer(theta[l], topology=topology)
        else:
            # Разове кодування φ на початку
            encode_data(angles, kind=encoding, **enc_kwargs)
            for l in range(n_layers):
                hardware_efficient_layer(theta[l], topology=topology)

        # Вимірювання: конкатенуємо expval по запитаних осях
        # Порядок: спершу всі Z по дротах, потім усі X, потім усі Y (як у axes).
        if want_var:
            exp_ops, var_ops = build_measurement_ops(axes, n_qubits)
            return [*exp_ops, *var_ops]
            
        return [qml.expval(_axis_to_obs(ax, i)) for ax in axes for i in range(n_qubits)]
        
    out_dim_total = out_dim * (2 if want_var else 1)
    meta: Dict[str, Any] = {
        "axes": axes,
        "n_meas": n_meas,
        "output_dim": out_dim_total,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "encoding": encoding,
        "reupload": reupload,
        "topology": topology,
        "diff_method": diff_method,
        "device": dev_info,
        "measurement_mode": measurement_mode,
    }
    logger.debug("QNode created: %s", meta)

    # Підказка типу для виклику з Torch (angles, theta) -> torch.Tensor[out_dim]
    def qnode_typed(angles: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return qfunc(angles, theta)  # type: ignore[return-value]

    return qnode_typed, meta


__all__ = [
    "Topology",
    "EncodingKind",
    "MeasurementKind",
    "MeasurementMode",
    "build_measurement_ops",
    "create_qnode",
]