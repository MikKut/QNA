# Code/quantum/__tests__/test_profile_shots.py
# -*- coding: utf-8 -*-

import math
import time
import logging
import pytest
import torch

from Code.quantum_layer.quantum_layer import QuantumLayer

# Спробуємо ваш кастомний логер; якщо нема — stdlib
try:
    from Code.logger import setup_logger  # type: ignore
    LOG = setup_logger("tests.profile")
except Exception:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger("tests.profile")


# ---------- Хелпери ----------

def random_phi(batch: int, n_qubits: int, *, seed: int = 123, dtype=torch.float32) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand((batch, n_qubits), generator=g, dtype=dtype) * (2 * math.pi)) - math.pi


def profile_forward(layer: QuantumLayer, phi: torch.Tensor, *, repeats: int = 3) -> float:
    """
    Повертає середній час (сек) на forward для заданого шару і фіксованого phi.
    """
    # прогрів
    _ = layer(phi)
    torch.cuda.synchronize() if torch.cuda.is_available() else None  # на випадок GPU

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = layer(phi)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    return (t1 - t0) / float(repeats)


# ---------- Параметри профілю ----------

SHOTS_LIST = [50, 100, 500, 1000]   # як у ТЗ
BATCH = 8
N_QUBITS = 8
N_LAYERS = 2
REPEATS = 3


# ---------- Власне тест ----------

@pytest.mark.parametrize("shots", SHOTS_LIST)
def test_profile_forward_time_and_shape(shots):
    """
    Профілює forward для різних shots, перевіряє форми/типи виходу,
    логує середній час на батч.
    """
    layer = QuantumLayer(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        measurement="Z",
        shots=shots,
        seed=42,
        reupload=True,
        diff_method="parameter-shift",
        dtype=torch.float32,
    )

    phi = random_phi(BATCH, N_QUBITS, seed=2025)
    # Перевірка форми/типу виходу
    out = layer(phi)
    assert out.shape == (BATCH, N_QUBITS)
    assert out.dtype == layer.theta.dtype
    assert out.device == layer.theta.device

    # Профіль часу
    avg_sec = profile_forward(layer, phi, repeats=REPEATS)
    # Час має бути додатним і розумним (менше ~5 секунд для такого розміру)
    assert avg_sec > 0.0
    assert avg_sec < 5.0, f"Forward занадто повільний для shots={shots}: {avg_sec:.3f}s"

    LOG.info(
        "PROFILE | shots=%4d | batch=%d | n_qubits=%d | layers=%d | avg_time=%.6f s",
        shots, BATCH, N_QUBITS, N_LAYERS, avg_sec
    )


def test_profile_summary_table(capsys):
    """
    Додатковий зручний підсумок: будує таблицю часів для всіх shots.
    Не має жорстких асертів по відносинах часу (щоб уникнути флаків),
    лише друкує зведення.
    """
    layer = QuantumLayer(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        measurement="Z",
        shots=SHOTS_LIST[0],
        seed=42,
        reupload=True,
        diff_method="parameter-shift",
        dtype=torch.float32,
    )
    phi = random_phi(BATCH, N_QUBITS, seed=2025)

    results = []
    for s in SHOTS_LIST:
        layer.set_shots(s)  # перевибирає QNode під капотом
        avg_sec = profile_forward(layer, phi, repeats=REPEATS)
        results.append((s, avg_sec))

    # Форматована таблиця
    header = " shots | avg_time (s)\n" + "-" * 24
    rows = [f"{s:>6} | {t:>12.6f}" for s, t in results]
    table = header + "\n" + "\n".join(rows)

    LOG.info("PROFILE SUMMARY\n%s", table)
    print(table)  # щоб було видно і без логера

    # Мінімальні sanity-перевірки
    for s, t in results:
        assert t > 0.0
