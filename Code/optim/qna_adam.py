# Code/optim/qna_adam.py
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch.optim import Optimizer

from Code.logger import setup_logger

logger = setup_logger("qnaadam")


class QNAAdam(Optimizer):
    """
    QNAAdam — це звичайний Adam із додатковим шум-адаптивним масштабуванням кроку.

    Базові формули оновлення — як у torch.optim.Adam (включно з bias-correction
    та AMSGrad). Єдина відмінність: перед застосуванням кроку ми множимо базовий
    lr групи на scale_λ, де

        scale_λ = 1 / (1 + lambda_var * Vtilde),   клемимо в [lr_min_mult, lr_max_mult].

    Значення Vtilde очікується в param-group під ключем `vtilde_key` (за замовчуванням
    "qna_vtilde"), яке тренер виставляє після вимірювання шуму (Inline-DocPS).

    Додатково (опційно) є внутрішній глобальний L2 grad clipping (clip_in_optimizer).

    Параметри
    ---------
    params : iterable
        Список параметрів або param_groups як у torch.optim.
    lr : float
        Базова швидкість навчання.
    betas : Tuple[float, float]
        Коефіцієнти EMA для першого та другого моментів.
    eps : float
        Малий термін для числової стабільності.
    weight_decay : float
        L2-регуляризація (звичайний Adam, НЕ decoupled).
    amsgrad : bool
        Якщо True, використовуємо AMSGrad-випуклення.
    maximize : bool
        Якщо True, робимо градієнтний підйом (змінюємо знак града).

    --- QNA-специфічні ---
    lambda_var : float
        Коефіцієнт штрафу шуму. 0 → вимкнути.
    lr_min_mult, lr_max_mult : float
        Межі для scale_λ.
    vtilde_key : str
        Ім'я ключа у param-group, звідки брати Vtilde (наприклад, "qna_vtilde").

    --- Кліпінг усередині оптимізатора (опційно) ---
    clip_in_optimizer : bool
        Якщо True — виконується глобальний L2 clipping до Adam-кроку.
    max_norm : float
        Межа L2-норми градієнта.
    error_if_nonfinite : bool
        Якщо True — кидати помилку при NaN/Inf у нормі; інакше пропускати крок.
    grad_eps : float
        Епсилон у дільнику при масштабуванні під час кліпінгу.

    Зауваження
    ----------
    • Якщо param-group не має ключа `vtilde_key`, вважаємо Vtilde=0 → scale_λ=1,
      і така група оновлюється ідентично до звичайного Adam.
    • Щоб порівняння з Adam було чесним, тримайте однакові гіперпараметри і спосіб
      отримання градієнтів; QNAAdam лише масштабує крок для “квантової” групи.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        maximize: bool = False,
        # QNA:
        lambda_var: float = 1e-4,
        lr_min_mult: float = 0.0,
        lr_max_mult: float = 1.0,
        vtilde_key: str = "qna_vtilde",
        # внутрішній кліпінг:
        clip_in_optimizer: bool = False,
        max_norm: float = 1.0,
        error_if_nonfinite: bool = False,
        grad_eps: float = 1e-6,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if lr_min_mult < 0.0 or lr_max_mult <= 0.0 or lr_min_mult > lr_max_mult:
            raise ValueError("Invalid lr_min_mult/lr_max_mult range.")

        defaults: Dict[str, Any] = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            # QNA:
            lambda_var=lambda_var,
            lr_min_mult=lr_min_mult,
            lr_max_mult=lr_max_mult,
            vtilde_key=vtilde_key,
            # clipping:
            clip_in_optimizer=clip_in_optimizer,
            max_norm=max_norm,
            error_if_nonfinite=error_if_nonfinite,
            grad_eps=grad_eps,
        )
        super().__init__(params, defaults)

        self._step: int = 0
        self._logged_once: bool = False

    # ----------------------------- helpers ---------------------------------

    def _log_once_options(self) -> None:
        if self._logged_once:
            return
        self._logged_once = True

        try:
            any_clip = any(g.get("clip_in_optimizer", False) for g in self.param_groups)
            max_norms = sorted(set(float(g.get("max_norm", 1.0)) for g in self.param_groups))
            err_nonf = any(g.get("error_if_nonfinite", False) for g in self.param_groups)
            grad_epss = sorted(set(float(g.get("grad_eps", 1e-6)) for g in self.param_groups))
            lambdas = sorted(set(float(g.get("lambda_var", 0.0)) for g in self.param_groups))
            lrmins = sorted(set(float(g.get("lr_min_mult", 0.0)) for g in self.param_groups))
            lrmaxs = sorted(set(float(g.get("lr_max_mult", 1.0)) for g in self.param_groups))
            keys = sorted(set(str(g.get("vtilde_key", "qna_vtilde")) for g in self.param_groups))

            logger.info(
                "[QNAAdam] options: clip_in_optimizer=%s | max_norm=%s | error_if_nonfinite=%s "
                "| grad_eps=%s | lambda_var=%s | lr_min_mult=%s | lr_max_mult=%s | vtilde_key(s)=%s",
                any_clip, max_norms, err_nonf, grad_epss, lambdas, lrmins, lrmaxs, keys
            )
        except Exception:
            pass

    def _global_grad_clip(self) -> Tuple[bool, float]:
        """
        Повертає (clipped, total_norm). Якщо зустріли не-скінченність:
          - або кидаємо помилку (error_if_nonfinite=True),
          - або пропускаємо step (clipped=False, total_norm=nan) — це обробляє step().
        """
        total_sq = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Норму рахуємо у float32 для стабільності (без переходу в float64).
                g = p.grad.detach()
                if g.numel() == 0:
                    continue
                total_sq += float((g.to(torch.float32) ** 2).sum().item())

        if not math.isfinite(total_sq):
            # Некоректна сума квадратів → далі поведінка залежно від прапора
            return False, float("nan")

        total_norm = math.sqrt(max(total_sq, 0.0))
        # Знайдемо межі/eps (беремо найменший grad_eps, найбільший max_norm серед груп)
        max_norm = max(float(g.get("max_norm", 1.0)) for g in self.param_groups)
        grad_eps = min(float(g.get("grad_eps", 1e-6)) for g in self.param_groups)

        if total_norm > max_norm:
            scale = max_norm / (total_norm + grad_eps)
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    p.grad.detach().mul_(scale)
            logger.debug("[QNAAdam] global grad clip: ‖g‖=%.6f → scaled by %.6f", total_norm, scale)
            return True, total_norm
        return False, total_norm

    @staticmethod
    def _scale_from_vtilde(group: Dict[str, Any]) -> Tuple[float, float]:
        lam = float(group.get("lambda_var", 0.0))
        if lam <= 0.0:
            return 1.0, 0.0

        key = str(group.get("vtilde_key", "qna_vtilde"))
        vtilde = float(group.get(key, 0.0) or 0.0)

        scale = 1.0 / (1.0 + lam * vtilde)
        # клема:
        s_min = float(group.get("lr_min_mult", 0.0))
        s_max = float(group.get("lr_max_mult", 1.0))
        if scale < s_min:
            scale = s_min
        if scale > s_max:
            scale = s_max
        return float(scale), float(vtilde)

    # ------------------------------ API ------------------------------------

    def step(self, closure: Optional[Any] = None):
        """Виконує один крок оновлення параметрів (Adam + шумовий scale_λ)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step += 1
        self._log_once_options()

        # Внутрішній кліпінг (за потреби)
        if any(g.get("clip_in_optimizer", False) for g in self.param_groups):
            was_clipped, total_norm = self._global_grad_clip()
            self._last_clip_applied = bool(was_clipped)
            if not math.isfinite(total_norm):
                # Не-скінченні градієнти
                if any(g.get("error_if_nonfinite", False) for g in self.param_groups):
                    raise RuntimeError("QNAAdam: non-finite gradient norm (inf or NaN).")
                logger.warning("QNAAdam: non-finite gradient norm — skipping step.")
                return loss
            # Якщо просто відбулось масштабування — йдемо далі (короткий лог уже є).

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            amsgrad = bool(group.get("amsgrad", False))
            maximize = bool(group.get("maximize", False))

            # Визначаємо ефективний lr із шумовим масштабом
            scale, vtilde = self._scale_from_vtilde(group)
            lr_eff = lr * scale

            # Оновлення параметрів — стандартний Adam
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("QNAAdam does not support sparse gradients")

                if maximize:
                    grad = -grad

                # Ініціалізація стану
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # Зберігаємо стани у dtype/пристрої параметра
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]

                state["step"] += 1
                step_t = state["step"]

                # weight decay (L2) як у Adam
                if weight_decay != 0.0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # EMA моментів
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if amsgrad:
                    max_exp_avg_sq: torch.Tensor = state.get("max_exp_avg_sq")
                    if "max_exp_avg_sq" not in state:
                        max_exp_avg_sq = torch.zeros_like(exp_avg_sq)
                        state["max_exp_avg_sq"] = max_exp_avg_sq
                    else:
                        # узгодження dtype/device
                        state["max_exp_avg_sq"] = state["max_exp_avg_sq"].to(exp_avg_sq.dtype).to(exp_avg_sq.device)
                        max_exp_avg_sq = state["max_exp_avg_sq"]

                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)

                # Bias-correction
                bias_c1 = 1.0 - beta1 ** step_t
                bias_c2 = 1.0 - beta2 ** step_t
                step_size = lr_eff * math.sqrt(bias_c2) / bias_c1

                # Оновлення параметра
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

            # Невеликий лог для прозорості (раз на крок — норм)
            logger.debug(
                "[QNAAdam] group step: lr=%.3g, scale=%.4f, Vtilde=%.3e, amsgrad=%s, wd=%.2g",
                lr, scale, vtilde, amsgrad, weight_decay
            )

        return loss
