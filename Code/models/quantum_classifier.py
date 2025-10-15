# Code/models/quantum_classifier.py
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# Проєктні імпорти (існуючі у тебе)
from Code.quantum_layer.quantum_layer import QuantumLayer 
from Code.utils.io_utils import load_project_config 
from Code.logger import setup_logger 


class QuantumClassifier(nn.Module):
    """
    Квантово-класичний класифікатор:
        QuantumLayer(φ[B, n_qubits]) -> expvals[B, out_dim] -> Linear(out_dim -> n_classes)
    де out_dim залежить від режиму вимірювання (Z | ZX | ZXY).

    Примітки:
      - Вхід: Готові кутові ознаки φ (PCA + Z-score + масштабування у ±angle_max), БЕЗ дублю AngleEncoder.
      - Пристрій: QuantumLayer (default.qubit) працює на CPU; тримай увесь пайплайн на CPU.
      - Градієнти: підтримуються через PennyLane (interface='torch', diff_method з конфігу).
    """

    def __init__(self, config: Dict[str, Any], logger=None) -> None:
        super().__init__()
        self.config = config
        self.logger = logger or setup_logger(__name__)

        # --- Читаємо ключові параметри з конфігу ---
        self.n_qubits: int = int(self._cfg("pca.dim", 8))
        self.n_classes: int = int(self._cfg("data.n_classes", 2))

        qcfg = {
            "device": self._cfg("quantum.device", "default.qubit"),
            "shots": self._cfg("quantum.shots", None),
            "n_layers": int(self._cfg("quantum.n_layers", 2)),
            "topology": str(self._cfg("quantum.topology", "ring")),
            "encoding": str(self._cfg("quantum.encoding", "ry")),
            "measurements": str(self._cfg("quantum.measurements", "Z")).upper(),
            "diff_method": self._cfg("quantum.diff_method", "adjoint"),
            "param_seed": self._cfg("project.seed", None),
            "reupload": self._cfg("quantum.reupload", True)
        }

        # --- Створюємо квантовий шар ---
        self.quantum: QuantumLayer
        if hasattr(QuantumLayer, "from_config") and callable(getattr(QuantumLayer, "from_config")):
            self.quantum = QuantumLayer.from_config(config, logger=self.logger)
        else:
            self.quantum = QuantumLayer(
                n_qubits=self.n_qubits,
                n_layers=qcfg["n_layers"],
                topology=qcfg["topology"],
                encoding=qcfg["encoding"],
                measurement=qcfg["measurements"],
                device_name=qcfg["device"],
                shots=qcfg["shots"],
                diff_method=qcfg["diff_method"],
                param_seed=qcfg["param_seed"],
                reupload=qcfg["reupload"],
                logger=self.logger,
            )

        # --- Визначаємо розмірність виходу квантового шару ---
        q_out_dim = getattr(self.quantum, "out_dim", None)
        if q_out_dim is None:
            q_out_dim = self._infer_out_dim_from_measurements(self.n_qubits, qcfg["measurements"])
        self.quantum_out_dim: int = int(q_out_dim)

        # --- Класичний лінійний шар для логітів ---
        self.classifier = nn.Linear(self.quantum_out_dim, self.n_classes)
        self._init_classifier_weights(param_seed=self._cfg("project.seed", None))

        self._last_phi: Optional[torch.Tensor] = None
        self._last_expvals: Optional[torch.Tensor] = None
        self._warned_no_shots: bool = False

        # Лог рядок-конфіг
        self.logger.info(
            "[QuantumClassifier] n_qubits=%d, out_dim=%d, n_classes=%d | device=%s, shots=%s, "
            "layers=%d, topology=%s, encoding=%s, meas=%s, diff=%s, reupload=%s",
            self.n_qubits,
            self.quantum_out_dim,
            self.n_classes,
            qcfg["device"],
            str(qcfg["shots"]),
            qcfg["n_layers"],
            qcfg["topology"],
            qcfg["encoding"],
            qcfg["measurements"],
            qcfg["diff_method"],
            str(qcfg["reupload"]),
        )

    # ----------------------------------------------------------------------
    # ПУБЛІЧНІ API-методи (зручно для тренера/колбеків)
    # ----------------------------------------------------------------------
    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phi: тензор кутів форми (B, n_qubits), dtype=float32

        Returns:
            logits: (B, n_classes)
        """
        # Перевірка розмірів на ранньому етапі
        if phi.dim() != 2 or phi.size(1) != self.n_qubits:
            raise ValueError(
                f"Expected phi of shape (B, {self.n_qubits}), got {tuple(phi.shape)}"
            )
        
        shots = self._cfg("quantum.shots", None)
        if shots is None and not self._warned_no_shots:
            self.logger.warning(
                "[QuantumClassifier] quantum.shots is None (analytic mode). "
                "Шум від шотів відсутній; DocPS/шум-орієнтований QNA не матиме ефекту."
            )
            self._warned_no_shots = True

        self._last_phi = phi.detach()

        # Основний квантовий forward
        expvals: torch.Tensor = self.quantum(phi)

        expvals: torch.Tensor = self.quantum(phi)  # (B, out_dim)
        if expvals.dim() != 2 or expvals.size(1) != self.quantum_out_dim:
            raise RuntimeError(
                f"QuantumLayer returned shape {tuple(expvals.shape)}; "
                f"expected (B, {self.quantum_out_dim})."
            )
        self._last_expvals = expvals
        logits = self.classifier(expvals)  # (B, n_classes)
        return logits

    def set_shots(self, shots: Optional[int]) -> None:
        """Прокидаємо керування стохастикою у QuantumLayer (для shot-annealing)."""
        if hasattr(self.quantum, "set_shots"):
            self.quantum.set_shots(shots)
            self.logger.info("[QuantumClassifier] shots set to %s", str(shots))
        else:
            self.logger.warning("QuantumLayer has no set_shots(...) method.")

    def set_seed(self, seed: int) -> None:
        """Встановлює сид для детермінованості всередині квантового шару."""
        if hasattr(self.quantum, "set_seed"):
            self.quantum.set_seed(seed)
            self.logger.info("[QuantumClassifier] param/device seed set to %s", str(seed))
        else:
            self.logger.warning("QuantumLayer has no set_seed(...) method.")

    # ----------------------------------------------------------------------
    # Допоміжні методи
    # ----------------------------------------------------------------------
    def _init_classifier_weights(self, param_seed: Optional[int]) -> None:
        """
        Детермінована ініціалізація лінеарного шару (за наявності project.seed),
        інакше — стандартна (PyTorch default).
        """
        if param_seed is None:
            return
        g = torch.Generator(device="cpu")
        # невеликий зсув, щоб не співпасти з ініціалізацією θ у QuantumLayer
        g.manual_seed(int(param_seed) + 13)

        # kaiming_uniform для ваг
        nn.init.kaiming_uniform_(self.classifier.weight, a=math.sqrt(5), generator=g)
        if self.classifier.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.classifier.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.classifier.bias, -bound, bound, generator=g)

    @staticmethod
    def _infer_out_dim_from_measurements(n_qubits: int, measurements: str) -> int:
        m = (measurements or "Z").upper()
        if m == "Z":
            return n_qubits
        if m == "ZX":
            return 2 * n_qubits
        if m == "ZXY":
            return 3 * n_qubits
        # Фолбек, якщо додали щось нове
        return n_qubits

    def _cfg(self, path: str, default: Any = None) -> Any:
        """
        Безпечне читання з вкладеного конфігу: "a.b.c" -> config["a"]["b"]["c"].
        """
        cur: Any = self.config
        for key in path.split("."):
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    # ----------------------------------------------------------------------
    # Зручні конструктори
    # ----------------------------------------------------------------------
    @classmethod
    def from_config_path(cls, config_path: str, logger=None) -> "QuantumClassifier":
        """
        Завантажує YAML (корінь або Code/config.yaml) через io_utils.load_project_config
        і створює QuantumClassifier.
        """
        cfg = load_project_config(config_path)
        return cls(cfg, logger=logger)

    # Для зручного друку у логах
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_qubits={self.n_qubits}, "
            f"quantum_out_dim={self.quantum_out_dim}, n_classes={self.n_classes})"
        )