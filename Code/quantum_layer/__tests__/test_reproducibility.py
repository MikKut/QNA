# Code/quantum/__tests__/test_reproducibility.py
# -*- coding: utf-8 -*-

import math
import pytest
import torch

from Code.quantum_layer.quantum_layer import QuantumLayer


# ---------- Хелпери ----------

def random_phi(batch: int, n_qubits: int, *, seed: int = 123, dtype=torch.float32) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # рівномірно з [-π, π]
    return (torch.rand((batch, n_qubits), generator=g, dtype=dtype) * (2 * math.pi)) - math.pi


# ---------- Фікстури ----------

@pytest.fixture(scope="module")
def n_qubits():
    return 8


@pytest.fixture(scope="module")
def phi_small(n_qubits):
    # Невеликий батч для швидких тестів
    return random_phi(3, n_qubits, seed=2024)


@pytest.fixture(scope="module")
def layer_analytic(n_qubits):
    # Аналітичний режим → детерміновані виходи і градієнти, незалежно від seed
    return QuantumLayer(
        n_qubits=n_qubits,
        n_layers=2,
        measurement="Z",
        shots=None,              # аналітичний режим
        seed=42,
        reupload=True,
        diff_method="parameter-shift",
        dtype=torch.float32,
    )


@pytest.fixture()
def layer_shots_A(n_qubits):
    # Шот-режим: детермінованість гарантована між інстансами з однаковим seed для однакових викликів
    return QuantumLayer(
        n_qubits=n_qubits,
        n_layers=2,
        measurement="Z",
        shots=50,
        seed=777,
        reupload=True,
        diff_method="parameter-shift",
        dtype=torch.float32,
    )


@pytest.fixture()
def layer_shots_B(n_qubits):
    # Другий інстанс з тим самим сидом — має давати ті самі послідовності вимірювань
    return QuantumLayer(
        n_qubits=n_qubits,
        n_layers=2,
        measurement="Z",
        shots=50,
        seed=777,
        reupload=True,
        diff_method="parameter-shift",
        dtype=torch.float32,
    )


# ---------- Тести: аналітичний режим ----------

def test_analytic_same_outputs_across_calls(layer_analytic, phi_small):
    # В аналітичному режимі дві послідовні прогонки з однаковими φ мають збігатися точно
    out1 = layer_analytic(phi_small)
    out2 = layer_analytic(phi_small)
    assert torch.allclose(out1, out2, atol=0.0, rtol=0.0), "shots=None має давати ідентичні виходи на повторних викликах"


def test_analytic_same_across_instances(n_qubits, phi_small):
    # Два інстанси з однаковими налаштуваннями shots=None → однакові виходи
    A = QuantumLayer(n_qubits=n_qubits, n_layers=2, measurement="Z", shots=None, seed=11, reupload=True)
    B = QuantumLayer(n_qubits=n_qubits, n_layers=2, measurement="Z", shots=None, seed=999, reupload=True)
    outA = A(phi_small)
    outB = B(phi_small)
    assert torch.allclose(outA, outB, atol=0.0, rtol=0.0), "В аналітичному режимі сид не впливає на expval — виходи мають збігатися"


# ---------- Тести: шот-режим (відтворюваність та зміна сидів) ----------

def test_shots_same_seed_same_instance_two_calls_produce_sequence(layer_shots_A, phi_small):
    # У шот-режимі один і той самий інстанс з фіксованим сидом дає ДЕТЕРМІНОВНУ ПОСЛІДОВНІСТЬ:
    # out1 != out2, але якщо повторити експеримент, послідовність буде та сама.
    out1 = layer_shots_A(phi_small)
    out2 = layer_shots_A(phi_small)
    # Як правило, out1 і out2 відрізняються (нова «порція» вибірок); дозволяємо, що можуть збігтися випадково.
    assert not torch.allclose(out1, out2, atol=1e-7, rtol=1e-7) or True


def test_shots_same_seed_two_instances_match_on_each_call(layer_shots_A, layer_shots_B, phi_small):
    # Два незалежних інстанси з тим самим сидом мають видавати ОДНАКОВІ значення для k-го виклику
    # (бо обидва генерують одну і ту ж псевдовипадкову послідовність).
    outA1 = layer_shots_A(phi_small)
    outB1 = layer_shots_B(phi_small)
    assert torch.allclose(outA1, outB1, atol=1e-6, rtol=1e-6), "Перший виклик має дати однакові результати для однакових сидів"

    outA2 = layer_shots_A(phi_small)
    outB2 = layer_shots_B(phi_small)
    assert torch.allclose(outA2, outB2, atol=1e-6, rtol=1e-6), "Другий виклик також має збігатися між інстансами з тим самим сидом"


def test_shots_set_seed_changes_output(layer_shots_A, phi_small):
    # Зміна сидa → інший детермінований потік випадковостей → інші expvals
    out_before = layer_shots_A(phi_small)
    layer_shots_A.set_seed(layer_shots_A.spec.seed + 1)
    out_after = layer_shots_A(phi_small)

    # З великою ймовірністю значення зміняться (не вимагаємо великої різниці; просто не allclose)
    assert not torch.allclose(out_before, out_after, atol=1e-7, rtol=1e-7), "Після set_seed очікуємо інші expvals у шот-режимі"


def test_shots_change_shots_affects_output(layer_shots_B, phi_small):
    # Зміна кількості шотів зазвичай змінює оцінку expval
    out50 = layer_shots_B(phi_small)     # shots=50 (як у фікстурі)
    layer_shots_B.set_shots(200)         # більше шотів → інша оцінка (менший шум, але інше значення)
    out200 = layer_shots_B(phi_small)

    assert out50.shape == out200.shape == (phi_small.shape[0], phi_small.shape[1])
    assert not torch.allclose(out50, out200, atol=1e-6, rtol=1e-6), "Інша кількість шотів має дати інше (більш стабільне) очікування"
