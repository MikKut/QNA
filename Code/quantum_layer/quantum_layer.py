# Code/quantum/quantum_layer.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence, Tuple
import logging
import torch
import torch.nn as nn

from Code.logger import setup_logger  # ← ваш кастомний логер
from .devices import DeviceSpec, spec_from_config, with_seed, with_shots
from .qnode_factory import create_qnode
from Code.utils.tensor_utils import batched_apply
Topology = Literal["ring", "linear"]
EncodingKind = Literal["ry", "rx", "rz", "xy"]
MeasurementKind = Literal["Z", "ZX", "ZXY"]


class QuantumLayer(nn.Module):
    def __init__(
        self,
        n_qubits: int,
        *,
        n_layers: int = 2,
        topology: Topology = "ring",
        encoding: EncodingKind | str = "ry",
        reupload: bool = True,
        measurement: MeasurementKind | Sequence[str] = "Z",
        diff_method: Literal["parameter-shift", "best"] = "parameter-shift",
        spec: Optional[DeviceSpec] = None,
        device_name: str = "default.qubit",
        shots: Optional[int] = 100,
        seed: int = 42,
        dtype: torch.dtype = torch.float32,
        logger: Optional[logging.Logger] = None, 
        param_seed: Optional[int] = 0,            
        param_init_std: float = 0.1,
    ) -> None:
        super().__init__()

        if n_qubits <= 0:
            raise ValueError(f"n_qubits must be > 0, got {n_qubits}.")
        if n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {n_layers}.")

        # Логер модуля (idempotent у вашій реалізації setup_logger)
        self.logger = logger or setup_logger("quantum.layer")

        # Гіперпараметри схеми
        self.n_qubits: int = int(n_qubits)
        self.n_layers: int = int(n_layers)
        self.topology: Topology = topology
        self.encoding: str = str(encoding).lower()
        self.reupload: bool = bool(reupload)
        self.measurement: MeasurementKind | Sequence[str] = measurement
        self.diff_method: Literal["parameter-shift", "best"] = diff_method

        # Треновані параметри θ
        theta = torch.empty(self.n_layers, self.n_qubits, dtype=dtype)
        if param_seed is not None:
            g = torch.Generator(device="cpu").manual_seed(int(param_seed))
            nn.init.normal_(theta, mean=0.0, std=float(param_init_std), generator=g)
        else:
            nn.init.normal_(theta, mean=0.0, std=float(param_init_std))
        self.theta: nn.Parameter = nn.Parameter(theta)
        self._param_seed = param_seed
        self._param_init_std = float(param_init_std)

        # DeviceSpec
        self.spec: DeviceSpec = spec if spec is not None else DeviceSpec(
            name=device_name, n_qubits=self.n_qubits, shots=shots, seed=seed
        )

        self._qnode, self._meta = self._build_qnode()
        self.logger.debug(
            "Init QuantumLayer | n_qubits=%d, n_layers=%d, topology=%s, encoding=%s, reupload=%s, "
            "measurement=%s, shots=%s, seed=%d, param_seed=%s, param_init_std=%.3f",
            self.n_qubits, self.n_layers, self.topology, self.encoding, self.reupload,
            str(self.spec.shots), self.spec.seed, str(self._param_seed), self._param_init_std
        )
        self.logger.debug("QNode meta: %s", self._meta)

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        *,
        n_layers: Optional[int] = None,
        topology: Topology = "ring",
        encoding: EncodingKind | str = "ry",
        reupload: bool = True,
        measurement: MeasurementKind | Sequence[str] = "Z",
        diff_method: Literal["parameter-shift", "best"] = "parameter-shift",
        dtype: torch.dtype = torch.float32,
        logger: Optional[logging.Logger] = None,
    ) -> "QuantumLayer":
        spec = spec_from_config(cfg)
        n_qubits = int(spec.n_qubits)
        L = int(n_layers if n_layers is not None else int(cfg.get("quantum", {}).get("n_layers", 2)))
        proj_seed = int(cfg.get("project", {}).get("seed", 42))
        return cls(
            n_qubits=n_qubits,
            n_layers=L,
            topology=topology,
            encoding=encoding,
            reupload=reupload,
            measurement=measurement,
            diff_method=diff_method,
            spec=spec,
            dtype=dtype,
            logger=logger,
            param_seed=proj_seed
        )

    def set_shots(self, shots: Optional[int]) -> None:
        old = self.spec.shots
        self.spec = with_shots(self.spec, shots)
        self._qnode, self._meta = self._build_qnode()
        self.logger.info("Switch shots: %s -> %s | meta=%s", str(old), str(shots), self._meta)

    def set_seed(self, seed: int) -> None:
        old = self.spec.seed
        self.spec = with_seed(self.spec, seed)
        self._qnode, self._meta = self._build_qnode()
        self.logger.info("Reseed device: %d -> %d | meta=%s", old, seed, self._meta)

    def rebuild_qnode(
        self,
        *,
        topology: Optional[Topology] = None,
        encoding: Optional[EncodingKind | str] = None,
        reupload: Optional[bool] = None,
        measurement: Optional[MeasurementKind | Sequence[str]] = None,
        diff_method: Optional[Literal["parameter-shift", "best"]] = None,
        spec: Optional[DeviceSpec] = None,
    ) -> None:
        if topology is not None:
            self.topology = topology
        if encoding is not None:
            self.encoding = str(encoding).lower()
        if reupload is not None:
            self.reupload = bool(reupload)
        if measurement is not None:
            self.measurement = measurement
        if diff_method is not None:
            self.diff_method = diff_method
        if spec is not None:
            self.spec = spec

        self._qnode, self._meta = self._build_qnode()
        self.logger.info("Rebuilt QNode | meta=%s", self._meta)

    @property
    def output_dim(self) -> int:
        return int(self._meta.get("output_dim", self.n_qubits))

    @property
    def meta(self) -> Dict[str, Any]:
        return dict(self._meta)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        self._assert_input_shape(angles)
        self.logger.debug(
            "Forward | batch=%d, n_qubits=%d, output_dim=%d, shots=%s",
            angles.shape[0], self.n_qubits, self.output_dim, str(self.spec.shots)
        )

        return batched_apply(self._qnode, angles, self.theta, logger=self.logger)
    
    def _build_qnode(self) -> Tuple[Any, Dict[str, Any]]:
        qnode, meta = create_qnode(
            spec=self.spec,
            n_layers=self.n_layers,
            topology=self.topology,
            encoding=self.encoding,
            reupload=self.reupload,
            measurement=self.measurement,
            diff_method=self.diff_method,
            interface="torch",
            encoding_kwargs=None,
        )
        return qnode, meta
    
    def _assert_input_shape(self, angles: torch.Tensor) -> None:
        if not isinstance(angles, torch.Tensor):
            raise TypeError(f"angles must be a torch.Tensor, got {type(angles)}.")
        if angles.ndim != 2 or angles.shape[1] != self.n_qubits:
            raise ValueError(
                f"angles must have shape (B, n_qubits) with n_qubits={self.n_qubits}, "
                f"got {tuple(angles.shape)}."
            )


__all__ = ["QuantumLayer", "Topology", "EncodingKind", "MeasurementKind"]