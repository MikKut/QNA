# Code/quantum_layer/quantum_layer.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Callable
import logging
import math
import torch
import torch.nn as nn

from Code.logger import setup_logger
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

        # DeviceSpec тренувального тракту
        self.spec: DeviceSpec = spec if spec is not None else DeviceSpec(
            name=device_name, n_qubits=self.n_qubits, shots=shots, seed=seed
        )

        # Основний QNode (expval-only)
        self._qnode, self._meta = self._build_qnode()

        # --- Статистичний QNode для Inline-DocPS (expval+var, ледача ініціалізація) ---
        self._qnode_stats: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
        self._meta_stats: Optional[Dict[str, Any]] = None
        self._spec_stats: Optional[DeviceSpec] = None
        self._stats_seed_offset: int = 0  # опційний зсув сид лише для зонда
        self._probe_shots_override: Optional[int] = None  # опційні shots тільки для зонда

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

    # ------------------------ Публічні сервісні методи ------------------------

    def set_shots(self, shots: Optional[int]) -> None:
        old = self.spec.shots
        self.spec = with_shots(self.spec, shots)
        self._qnode, self._meta = self._build_qnode()
        # Інвалідовуємо stats-QNode, щоб підхопив нові shots/seed
        self._qnode_stats = None
        self._meta_stats = None
        self._spec_stats = None
        self.logger.info("Switch shots: %s -> %s | meta=%s", str(old), str(shots), self._meta)

    def set_seed(self, seed: int) -> None:
        old = self.spec.seed
        self.spec = with_seed(self.spec, seed)
        self._qnode, self._meta = self._build_qnode()
        # Інвалідовуємо stats-QNode (щоб узгодити сиди)
        self._qnode_stats = None
        self._meta_stats = None
        self._spec_stats = None
        self.logger.info("Reseed device: %d -> %d | meta=%s", old, self.spec.seed, self._meta)

    def set_probe_seed_offset(self, offset: Optional[int]) -> None:
        """
        Встановлює/скидає зсув seed лише для зонда (stats-QNode). Інвалідовує зонд.
        None або 0 → вимкнути зсув.
        """
        self._stats_seed_offset = int(offset or 0)
        self._qnode_stats = None
        self._meta_stats = None
        self._spec_stats = None
        self.logger.info("[Inline-DocPS] probe seed offset → %d (stats-QNode will be rebuilt)", self._stats_seed_offset)

    def set_probe_shots(self, shots: Optional[int]) -> None:
        """
        Встановлює кількість шотів лише для зонда (stats-QNode). Тренувальний shots не змінюється.
        """
        self._probe_shots_override = None if shots is None else int(shots)
        self._qnode_stats = None
        self._meta_stats = None
        self._spec_stats = None
        self.logger.info("[Inline-DocPS] probe shots override → %s", str(self._probe_shots_override))

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
        # Інвалідовуємо stats-QNode (конфіг змінився)
        self._qnode_stats = None
        self._meta_stats = None
        self._spec_stats = None
        self.logger.info("Rebuilt QNode | meta=%s", self._meta)

    # ----------------------------- Властивості --------------------------------

    @property
    def output_dim(self) -> int:
        return int(self._meta.get("output_dim", self.n_qubits))

    @property
    def meta(self) -> Dict[str, Any]:
        return dict(self._meta)

    # ------------------------------- Forward ----------------------------------

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        self._assert_input_shape(angles)
        self.logger.debug(
            "Forward | batch=%d, n_qubits=%d, output_dim=%d, shots=%s",
            angles.shape[0], self.n_qubits, self.output_dim, str(self.spec.shots)
        )
        return batched_apply(self._qnode, angles, self.theta, logger=self.logger)

    # -------------------------- Inline-DocPS утиліти --------------------------

    def _ensure_stats_qnode(self, shots: Optional[int] = None) -> None:
        """
        Ледаче створення окремого QNode для зондів (expval+var).
        Не чіпає тренувальний self._qnode / self.spec.
        Якщо статистичний QNode ще не існує — будує його.
        """
        if self._qnode_stats is not None and self._meta_stats is not None and self._spec_stats is not None:
            return

        # Окремий DeviceSpec для статистики: ті самі або перевизначені shots; опц. зсув сидa
        base = self.spec
        use_shots = shots if shots is not None else (self._probe_shots_override if self._probe_shots_override is not None else base.shots)
        spec_stats = with_shots(base, use_shots)
        if self._stats_seed_offset:
            spec_stats = with_seed(spec_stats, int(spec_stats.seed) + int(self._stats_seed_offset))
        self._spec_stats = spec_stats

        qnode_stats, meta_stats = create_qnode(
            spec=spec_stats,
            n_layers=self.n_layers,
            topology=self.topology,
            encoding=self.encoding,
            reupload=self.reupload,
            measurement=self.measurement,
            diff_method=self.diff_method,
            interface="torch",
            encoding_kwargs=None,
            measurement_mode="expval_var",  # головна відмінність: повертає [expval..., var...]
        )
        self._qnode_stats = qnode_stats
        self._meta_stats = meta_stats

        self.logger.info(
            "[Inline-DocPS] stats-QNode ready | shots=%s, expval_dim=%d, var_dim=%d, device=%s",
            str(self._spec_stats.shots),
            int(self._meta_stats.get("expval_dim", 0)),
            int(self._meta_stats.get("var_dim", 0)),
            str(self._meta_stats.get("device")),
        )
        if self._spec_stats.shots is None:
            self.logger.warning("[Inline-DocPS] shots=None (аналітика) — шумова метрика буде неінформативною.")

    def _split_ev(self, result: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Розрізати вихід stats-QNode на (E, Var) відповідно до meta_stats.
        Якщо var_dim == 0 — обчислити Var = 1 - E**2 (Паулі-випадок).
        """
        assert self._meta_stats is not None, "stats meta is not initialized"
        expdim: int = int(self._meta_stats.get("expval_dim", 0))
        vardim: int = int(self._meta_stats.get("var_dim", 0))
        if expdim <= 0:
            raise RuntimeError("Invalid expval_dim in stats meta.")
        if result.dim() != 2 or result.size(1) < expdim + max(vardim, 0):
            raise RuntimeError(f"Unexpected stats result shape {tuple(result.shape)} for expdim={expdim}, vardim={vardim}.")

        E = result[:, :expdim].clamp(-1.0, 1.0)
        if vardim > 0:
            V = result[:, expdim:expdim + vardim]
        else:
            # Для Паулі-спостережень: Var = 1 - E^2 (у межах [-1,1] для expval)
            V = 1.0 - E.pow(2)
        return E, V

    @torch.no_grad()
    def docps_shift_stats(
        self,
        angles_batch: torch.Tensor,  # (B, n_qubits)
        layer: int,
        wire: int,
        shift: float = math.pi / 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Атомік для одного параметра: на тому ж batch φ повертає (E+, E-, V+, V-), усі (B, out_dim).
        """
        if angles_batch.dim() != 2 or angles_batch.size(1) != self.n_qubits:
            raise ValueError(f"angles_batch must have shape (B, {self.n_qubits}), got {tuple(angles_batch.shape)}")
        if not (0 <= layer < self.n_layers and 0 <= wire < self.n_qubits):
            raise IndexError(f"(layer={layer}, wire={wire}) out of bounds for (L={self.n_layers}, nq={self.n_qubits})")

        self._ensure_stats_qnode()
        assert self._qnode_stats is not None

        # Готуємо зсунуті копії параметрів (без зміни self.theta)
        th = self.theta.detach()
        theta_plus = th.clone()
        theta_minus = th.clone()
        theta_plus[layer, wire] = theta_plus[layer, wire] + shift
        theta_minus[layer, wire] = theta_minus[layer, wire] - shift

        # Один і той самий batch φ
        out_plus = batched_apply(self._qnode_stats, angles_batch, theta_plus, logger=self.logger)   # (B, 2*out_dim)
        out_minus = batched_apply(self._qnode_stats, angles_batch, theta_minus, logger=self.logger) # (B, 2*out_dim)

        E_plus, V_plus = self._split_ev(out_plus)
        E_minus, V_minus = self._split_ev(out_minus)
        return E_plus, E_minus, V_plus, V_minus

    @torch.no_grad()
    def backward_inline_docps(
        self,
        angles_batch: torch.Tensor,   # (B, n_qubits) — той самий batch φ
        dL_dE: torch.Tensor,          # (B, out_dim) — ∂L/∂E з автограду
        *,
        shift: float = math.pi / 2,
        shots: Optional[int] = None,
        subset: Optional[Sequence[Tuple[int, int]]] = None,
        reduce_var: Literal["param"] = "param",
        return_epm: bool = False,
    ) -> Dict[str, Any]:
        """
        Обчислює theta.grad через параметр-шифт та повертає пер-параметрну шумову метрику var_param.

        Returns:
            {
              "grad_theta": torch.Tensor(L, n_qubits),
              "var_param": torch.Tensor(L, n_qubits),   # карта V_i (документна формула)
              "Vtilde": None,                           # лишається для сумісності ключів
              "shots": int,
              "expval_dim": int,
              (опц.) "Eplus_mean": float,
              (опц.) "Eminus_mean": float,
            }
        """
        B, nq = angles_batch.shape
        if nq != self.n_qubits:
            raise ValueError(f"angles_batch second dim must be {self.n_qubits}, got {nq}.")
        if dL_dE.dim() != 2 or dL_dE.size(1) != self.output_dim:
            raise ValueError(f"dL_dE must have shape (B, {self.output_dim}), got {tuple(dL_dE.shape)}.")

        if reduce_var != "param":
            raise NotImplementedError('Only reduce_var="param" is supported for document-correct experiments.')

        self._ensure_stats_qnode(shots=shots)
        assert self._qnode_stats is not None and self._meta_stats is not None and self._spec_stats is not None

        expdim: int = int(self._meta_stats.get("expval_dim", self.output_dim))
        if expdim != self.output_dim:
            # Безпека: тренувальний qnode повертає expval_dim; stats qnode має співпасти
            self.logger.warning("expval_dim mismatch: train=%d, stats=%d", self.output_dim, expdim)

        # Коеф. параметр-шифту: стандартно 1/2 для ±π/2; інакше 1/(2*sin(shift))
        if abs(float(shift) - (math.pi / 2)) < 1e-8:
            c_ps = 0.5
        else:
            s = math.sin(float(shift))
            if abs(s) < 1e-12:
                raise ValueError("Invalid shift for parameter-shift: sin(shift)≈0.")
            c_ps = 1.0 / (2.0 * s)

        # Підмножина параметрів (якщо не задано — всі)
        if subset is None:
            indices: Sequence[Tuple[int, int]] = [(l, w) for l in range(self.n_layers) for w in range(self.n_qubits)]
        else:
            indices = list(subset)
            if len(indices) == 0:
                raise ValueError("subset for backward_inline_docps must not be empty.")

        device      = self.theta.device
        theta_dtype = self.theta.dtype

        grad_theta_work = torch.zeros(self.n_layers, self.n_qubits, dtype=torch.float32, device=device)
        var_param       = torch.zeros(self.n_layers, self.n_qubits, dtype=torch.float32, device=device)

        # dL/dE → float32 на потрібному пристрої
        dL_dE_f32 = dL_dE.to(device=device, dtype=torch.float32, copy=False)

        # 1/(4M) для документної формули
        M = self._spec_stats.shots
        inv_4M = (1.0 / (4.0 * float(M))) if (M is not None and float(M) > 0) else 0.0
        if inv_4M == 0.0:
            self.logger.warning("[Inline-DocPS] shots=None or M<=0 → V_i неінформативні; трактуємо як 0.")

        # Діагностика середніх експектацій (якщо запитано)
        ep_acc = em_acc = 0.0
        ep_cnt = em_cnt = 0

        for (l, w) in indices:
            E_plus, E_minus, V_plus, V_minus = self.docps_shift_stats(angles_batch, l, w, shift=shift)

            E_plus  = E_plus.to(device=device, dtype=torch.float32, copy=False)
            E_minus = E_minus.to(device=device, dtype=torch.float32, copy=False)
            V_plus  = V_plus.to(device=device, dtype=torch.float32, copy=False)
            V_minus = V_minus.to(device=device, dtype=torch.float32, copy=False)

            V_plus.clamp_min_(0.0)
            V_minus.clamp_min_(0.0)

            # dE/dtheta (B, out_dim)
            dE_dth = c_ps * (E_plus - E_minus)
            # Проєкція на dL/dE: сумування по batch та out_dim
            g_lw = torch.sum(dE_dth * dL_dE_f32)
            grad_theta_work[l, w] = g_lw

            # Документна оцінка шуму для параметра i
            v_lw = inv_4M * float(torch.mean(V_plus + V_minus).item()) if inv_4M > 0.0 else 0.0
            var_param[l, w] = v_lw

            if return_epm:
                ep_acc += float(E_plus.mean().item()); ep_cnt += 1
                em_acc += float(E_minus.mean().item()); em_cnt += 1

        # Пишемо градієнт у параметр (torch-сумісно)
        if self.theta.grad is not None:
            self.theta.grad.detach_()
            self.theta.grad.zero_()

        grad_theta_out = torch.zeros_like(self.theta)
        grad_theta_out.copy_(grad_theta_work.to(dtype=theta_dtype))
        self.theta.grad = grad_theta_out

        out: Dict[str, Any] = {
            "grad_theta": grad_theta_out,
            "var_param": var_param,        # карта V_i
            "Vtilde": None,               # для сумісності ключів (не використовується у param-режимі)
            "shots": int(self._spec_stats.shots) if self._spec_stats.shots is not None else -1,
            "expval_dim": int(expdim),
        }
        if return_epm and ep_cnt > 0 and em_cnt > 0:
            out["Eplus_mean"] = ep_acc / ep_cnt
            out["Eminus_mean"] = em_acc / em_cnt

        self.logger.debug(
            "[Inline-DocPS] backward: B=%d, subset=%d, shift=%.4f, shots=%s, reduce=param",
            B, len(indices), float(shift), str(self._spec_stats.shots),
        )
        return out

    # ------------------------------- Helpers ----------------------------------

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
            # ВАЖЛИВО: тренувальний forward повертає лише expval
            # (статистичний шлях — окремим QNode у _ensure_stats_qnode)
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
