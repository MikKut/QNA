# Code/tests/test_quantum_grads.py
import math
import pytest
import torch
import numpy as np

from Code.models.quantum_classifier import QuantumClassifier
from Code.datasets.phi_dataset import PhiDataset

# (опційно) єдині сиди через проєктні утиліти
try:
    from Code.utils.io_utils import seed_everything  # type: ignore
except Exception:
    seed_everything = None


def _base_config(n_qubits=4, n_classes=3, shots=None, diff_method="adjoint"):
    """
    Базовий конфіг для градієнтних тестів.
    n_qubits зроблено невеликим, щоб тести були швидкі навіть на CI.
    """
    return {
        "pca": {"dim": int(n_qubits)},
        "angles": {"k": 2.5, "angle_max": float(math.pi)},
        "project": {"seed": 123},
        "data": {
            "n_classes": int(n_classes),
            "dummy_size": 32,  # невеликий набір
        },
        "quantum": {
            "device": "default.qubit",
            "shots": shots,              # None = аналітичний режим
            "n_layers": 2,
            "topology": "ring",
            "encoding": "ry",
            "measurements": "Z",
            "diff_method": diff_method,  # 'adjoint' для shots=None
        },
        "training": {"batch_size": 8, "lr": 3e-4, "grad_clip": 1.0},
    }


@pytest.fixture(autouse=True)
def _set_seeds():
    # детермінізуємо середовище для відтворюваності
    if seed_everything is not None:
        seed_everything(123)
    else:
        torch.manual_seed(123)
        np.random.seed(123)


def _grad_norm(params):
    total_sq = 0.0
    for p in params:
        if p.grad is not None:
            g = p.grad.detach()
            if g.numel() > 0:
                total_sq += float(torch.sum(g * g).item())
    return math.sqrt(total_sq)


def _has_any_param(module: torch.nn.Module) -> bool:
    try:
        next(module.parameters())
        return True
    except StopIteration:
        return False


def test_grads_exist_and_nonzero_analytic():
    """
    Перевірка: при shots=None та diff_method='adjoint' градієнти проходять
    і є ненульові як у квантового шару, так і у лінійного.
    """
    cfg = _base_config(n_qubits=4, n_classes=3, shots=None, diff_method="adjoint")

    # Дані (dummy) і батч
    ds = PhiDataset(cfg, mode="train", dummy=True)
    B = cfg["training"]["batch_size"]
    X = torch.stack([ds[i][0] for i in range(B)], dim=0)      # (B, n_qubits)
    y = torch.stack([ds[i][1] for i in range(B)], dim=0)      # (B,)

    # Модель і оптимізатор (для узгодженості, хоч крок не обов'язково робити)
    model = QuantumClassifier(cfg)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    logits = model(X)                     # (B, n_classes)
    loss = criterion(logits, y)
    loss.backward()

    # Перевіряємо, що параметри існують
    assert _has_any_param(model), "У моделі немає параметрів"
    assert _has_any_param(model.classifier), "У classifier немає параметрів"
    assert _has_any_param(model.quantum), "У QuantumLayer немає параметрів"

    # Загальна норма градієнтів моделі
    total_gn = _grad_norm(model.parameters())
    assert total_gn > 0.0, "Градієнти моделі нульові"

    # Окремо: градієнти квантового шару
    q_gn = _grad_norm(model.quantum.parameters())
    assert q_gn > 0.0, "Градієнти квантового шару нульові"

    # Окремо: градієнти лінійного шару
    c_gn = _grad_norm(model.classifier.parameters())
    assert c_gn > 0.0, "Градієнти classifier нульові"

    # Відсутність NaN/Inf у градієнтах
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "У градієнтах знайдено NaN/Inf"


@pytest.mark.parametrize(
    "shots",
    [
        pytest.param(50, marks=pytest.mark.skipif(True, reason="Опціонально: повільніше в CI")),
        pytest.param(100, marks=pytest.mark.skipif(True, reason="Опціонально: повільніше в CI")),
    ],
)
def test_grads_exist_finite_shots_smoke(shots):
    """
    (Опціональний саніті-тест) Для невеликих shots перевіряємо, що градієнти існують.
    За замовчуванням пропущено, щоб не сповільнювати CI.
    """
    cfg = _base_config(n_qubits=3, n_classes=2, shots=shots, diff_method="parameter-shift")

    ds = PhiDataset(cfg, mode="train", dummy=True)
    X = torch.stack([ds[i][0] for i in range(4)], dim=0)  # невеликий батч
    y = torch.stack([ds[i][1] for i in range(4)], dim=0)

    model = QuantumClassifier(cfg)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()

    # Мінімальна перевірка: хоч якісь градієнти пройшли
    total_gn = _grad_norm(model.parameters())
    assert total_gn > 0.0, "Градієнти нульові при shots>0"
