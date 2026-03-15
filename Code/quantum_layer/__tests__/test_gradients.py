# Code/quantum/__tests__/test_gradients.py
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
def layer_analytic(n_qubits):
    # Аналітичний режим: shots=None -> стабільні та точні градієнти
    layer = QuantumLayer(
        n_qubits=n_qubits,
        n_layers=2,
        measurement="Z",
        shots=None,              # важливо: жодного стокастичного шуму
        seed=42,                 # сид усе одно фіксуємо
        reupload=True,           # базова політика
        diff_method="parameter-shift",
        dtype=torch.float32,
    )
    return layer


# ---------- Тести градієнтів ----------

def test_theta_requires_grad(layer_analytic):
    assert layer_analytic.theta.requires_grad, "θ має бути trainable параметром (requires_grad=True)"


def test_gradients_exist_and_shape(layer_analytic, n_qubits):
    B = 4
    phi = random_phi(B, n_qubits)
    out = layer_analytic(phi)                 # (B, n_qubits)
    # Скаляримо loss; беремо суму квадратів, щоб уникнути випадкових нульових градів
    loss = (out ** 2).sum()
    loss.backward()

    assert layer_analytic.theta.grad is not None, "Після backward() θ.grad не має бути None"
    assert layer_analytic.theta.grad.shape == (layer_analytic.n_layers, n_qubits), \
        "Форма градієнтів θ має бути (L, n_qubits)"
    # Перевірка, що градієнти скінченні
    assert torch.isfinite(layer_analytic.theta.grad).all(), "Градієнти θ мають бути скінченні (без NaN/Inf)"
    # І що вони не всі нулі (при випадкових φ і reupload=True це має триматись)
    assert not torch.allclose(layer_analytic.theta.grad, torch.zeros_like(layer_analytic.theta.grad)), \
        "Градієнти θ не очікуються всюди нульовими для випадкових φ"

    # Приберемо градієнти для наступних тестів
    layer_analytic.zero_grad(set_to_none=True)


def test_second_backward_after_zero_grad_is_consistent(layer_analytic, n_qubits):
    # Той самий phi і аналітичний режим -> відтворювані значення градів між прогоном 1 і 2
    B = 3
    phi = random_phi(B, n_qubits, seed=999)

    # Прогін 1
    out1 = layer_analytic(phi)
    loss1 = (out1 ** 2).sum()
    loss1.backward()
    g1 = layer_analytic.theta.grad.detach().clone()
    layer_analytic.zero_grad(set_to_none=True)

    # Прогін 2 (ті самі дані, ті самі θ)
    out2 = layer_analytic(phi)
    loss2 = (out2 ** 2).sum()
    loss2.backward()
    g2 = layer_analytic.theta.grad.detach().clone()
    layer_analytic.zero_grad(set_to_none=True)

    # В аналітичному режимі градієнти мають збігатися з високою точністю
    assert torch.allclose(g1, g2, atol=1e-6, rtol=1e-6), "Градієнти θ мають бути відтворюваними у shots=None"


def test_simple_gradient_step_reduces_loss(layer_analytic, n_qubits):
    # Перевіримо, що крок у напрямку -grad зменшує loss (малий крок)
    B = 5
    phi = random_phi(B, n_qubits, seed=2024)

    # Початкові значення
    out0 = layer_analytic(phi)
    loss0 = (out0 ** 2).sum()

    # Градієнт і крок
    loss0.backward()
    g = layer_analytic.theta.grad.detach()
    assert g is not None
    # Малий крок (SGD без оптимізатора)
    lr = 1e-2
    with torch.no_grad():
        layer_analytic.theta -= lr * g

    layer_analytic.zero_grad(set_to_none=True)

    # Новий loss має бути не гірший (з запасом на чисельні похибки)
    out1 = layer_analytic(phi)
    loss1 = (out1 ** 2).sum()

    assert float(loss1) <= float(loss0) + 1e-7, "Крок у напрямку -grad має не збільшувати loss"
