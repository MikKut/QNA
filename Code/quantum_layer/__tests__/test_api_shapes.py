# Code/quantum/__tests__/test_api_shapes.py
# -*- coding: utf-8 -*-

import math
import pytest
import torch

from Code.quantum_layer.quantum_layer import QuantumLayer


# ---------- Хелпери ----------

def random_phi(batch: int, n_qubits: int, *, seed: int = 123) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # рівномірно з [-π, π]
    return (torch.rand((batch, n_qubits), generator=g, dtype=torch.float32) * (2 * math.pi)) - math.pi


# ---------- Фікстури ----------

@pytest.fixture(scope="module")
def n_qubits():
    return 8


@pytest.fixture(scope="module")
def layer_Z(n_qubits):
    # Невеликі shots для швидких тестів
    return QuantumLayer(
        n_qubits=n_qubits,
        n_layers=2,
        measurement="Z",
        shots=10,
        seed=42,
        reupload=True,                 # базова поведінка
        diff_method="parameter-shift", # дефолт для shot-режиму
        dtype=torch.float32,
    )


@pytest.fixture(scope="module")
def layer_ZX(n_qubits):
    return QuantumLayer(
        n_qubits=n_qubits,
        n_layers=2,
        measurement="ZX",  # Z + X -> подвоєний вихід
        shots=10,
        seed=42,
        reupload=True,
        diff_method="parameter-shift",
        dtype=torch.float32,
    )


# ---------- Тести форми вхід/вихід ----------

def test_forward_shape_Z(layer_Z, n_qubits):
    B = 4
    phi = random_phi(B, n_qubits)
    out = layer_Z(phi)
    assert out.shape == (B, n_qubits), "Для measurement='Z' вихід має бути (B, n_qubits)"
    assert out.dtype == layer_Z.theta.dtype
    assert out.device == layer_Z.theta.device


def test_forward_shape_ZX(layer_ZX, n_qubits):
    B = 3
    phi = random_phi(B, n_qubits, seed=777)
    out = layer_ZX(phi)
    assert out.shape == (B, 2 * n_qubits), "Для measurement='ZX' вихід має бути (B, 2*n_qubits)"
    assert out.dtype == layer_ZX.theta.dtype
    assert out.device == layer_ZX.theta.device


def test_invalid_input_shape_1d(layer_Z, n_qubits):
    # angles повинен бути 2D (B, n_qubits)
    with pytest.raises(ValueError):
        _ = layer_Z(torch.zeros(n_qubits, dtype=torch.float32))


def test_invalid_input_shape_wrong_nq(layer_Z, n_qubits):
    B = 2
    # неправильна друга розмірність
    with pytest.raises(ValueError):
        _ = layer_Z(torch.zeros(B, n_qubits + 1, dtype=torch.float32))


def test_output_dim_property_matches(layer_Z, n_qubits):
    # Для 'Z' вихідна розмірність дорівнює n_qubits
    assert layer_Z.output_dim == n_qubits


def test_dtype_device_consistency(layer_Z, n_qubits):
    # Перевіряємо, що forward коректно працює з іншим dtype/device вхідних кутів
    B = 2
    phi_fp64 = random_phi(B, n_qubits).to(dtype=torch.float64)  # інший dtype
    out = layer_Z(phi_fp64)
    # Вихід має відповідати dtype/device параметрів θ
    assert out.dtype == layer_Z.theta.dtype
    assert out.device == layer_Z.theta.device
