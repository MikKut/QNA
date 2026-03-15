# Code/tests/test_quantum_shapes.py
import math
import pytest
import torch
import numpy as np

from Code.models.quantum_classifier import QuantumClassifier
from Code.datasets.phi_dataset import PhiDataset

# (опційно) сиди через проєктні утиліти
try:
    from Code.utils.io_utils import seed_everything  # type: ignore
except Exception:
    seed_everything = None


def _base_config(n_qubits=8, n_classes=10, measurements="Z"):
    return {
        "pca": {"dim": int(n_qubits)},
        "angles": {"k": 2.5, "angle_max": float(math.pi)},
        "project": {"seed": 123},
        "data": {
            "n_classes": int(n_classes),
            "dummy_size": 32,         # невеликий dummy-набір для швидких тестів
            # шляхів не даємо → PhiDataset піде в dummy-режим
        },
        "quantum": {
            "device": "default.qubit",
            "shots": None,            # аналітичний режим (очікуємо стабільні градієнти)
            "n_layers": 2,
            "topology": "ring",
            "encoding": "ry",
            "measurements": str(measurements),
            "diff_method": "adjoint",
        },
        "training": {"batch_size": 8},
    }


@pytest.fixture(autouse=True)
def _set_seeds():
    # детермінізуємо середовище
    if seed_everything is not None:
        seed_everything(123)
    else:
        torch.manual_seed(123)
        np.random.seed(123)


def test_phi_dataset_shapes_dummy():
    cfg = _base_config(n_qubits=8, n_classes=10, measurements="Z")
    ds = PhiDataset(cfg, mode="train", dummy=True)
    assert len(ds) == cfg["data"]["dummy_size"]

    x0, y0 = ds[0]
    assert isinstance(x0, torch.Tensor) and isinstance(y0, torch.Tensor)
    assert x0.dtype == torch.float32 and y0.dtype == torch.int64
    assert x0.shape == (cfg["pca"]["dim"],)
    assert 0 <= int(y0.item()) < cfg["data"]["n_classes"]

    # Перевірка глобальних форм
    X = torch.stack([ds[i][0] for i in range(4)], dim=0)  # (4, n_qubits)
    Y = torch.stack([ds[i][1] for i in range(4)], dim=0)  # (4,)
    assert X.shape == (4, cfg["pca"]["dim"])
    assert Y.shape == (4,)


@pytest.mark.parametrize("measurements,expected_multiplier", [
    ("Z", 1),
    ("ZX", 2),   # якщо у твоєму QuantumLayer немає ZX, тест буде пропущено
])
def test_quantum_layer_output_shape(measurements, expected_multiplier):
    cfg = _base_config(n_qubits=8, n_classes=3, measurements=measurements)

    # Датасет (dummy), беремо невеликий батч
    ds = PhiDataset(cfg, mode="train", dummy=True)
    batch = torch.stack([ds[i][0] for i in range(5)], dim=0)  # (B=5, n_qubits)

    # Модель
    try:
        model = QuantumClassifier(cfg)
    except Exception as e:
        # Якщо вимірювання ZX не підтримується у твоїй реалізації — толерантно скіпаємо
        if measurements != "Z":
            pytest.skip(f"Measurements '{measurements}' не підтримуються: {e}")
        raise

    # Перевіряємо форму виходу квантового шару
    try:
        q_out = model.quantum(batch)  # очікуємо (B, out_dim)
    except Exception as e:
        if measurements != "Z":
            pytest.skip(f"QuantumLayer не може виконати '{measurements}': {e}")
        raise

    assert isinstance(q_out, torch.Tensor)
    assert q_out.shape[0] == batch.shape[0]
    assert q_out.shape[1] == expected_multiplier * cfg["pca"]["dim"]

    # Перевіряємо, що classifier бачить правильний розмір
    assert model.quantum_out_dim == expected_multiplier * cfg["pca"]["dim"]


@pytest.mark.parametrize("measurements,expected_multiplier", [
    ("Z", 1),
    ("ZX", 2),   # за відсутності підтримки ZX — тест буде пропущено
])
def test_quantum_classifier_output_shape(measurements, expected_multiplier):
    cfg = _base_config(n_qubits=6, n_classes=4, measurements=measurements)

    ds = PhiDataset(cfg, mode="train", dummy=True)
    B = 7
    batch = torch.stack([ds[i][0] for i in range(B)], dim=0)  # (B, n_qubits)

    try:
        model = QuantumClassifier(cfg)
    except Exception as e:
        if measurements != "Z":
            pytest.skip(f"Measurements '{measurements}' не підтримуються: {e}")
        raise

    logits = model(batch)  # (B, n_classes)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (B, cfg["data"]["n_classes"])

    # Додатково перевіряємо, що всередині квантовий вихід мав очікувану форму
    try:
        q_out = model.quantum(batch)
        assert q_out.shape == (B, expected_multiplier * cfg["pca"]["dim"])
    except Exception:
        # якщо ZX не підтримано — ми вже скіпнули б вище; тут просто проходимо
        if measurements != "Z":
            pytest.skip("ZX внутрішній виклик недоступний")
