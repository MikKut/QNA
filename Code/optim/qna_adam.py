# Code/optim/qna_adam.py
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch.optim import Optimizer

from Code.logger import setup_logger

logger = setup_logger("qnaadam")


class QNAAdam(Optimizer):
    r"""
    QNAAdam — Adam з шум-адаптивним масштабуванням кроку (document-style).

    ✅ Що додає відносно torch.optim.Adam:
      • Масштабування оновлення кожного параметра елемент-по-елементу за формулою
            scale_λ = 1 / (1 + λ_var · Ṽ)
        де Ṽ — оцінка дисперсії shot-noise. Масштаб клемиться у
        [lr_min_mult, lr_max_mult].

      • Два рівні джерела Ṽ:
          1) **Per-param (рекомендовано):** тренер кладе для кожного `p`
             `optimizer.state[p]["qna_var_param"] = V_like_p` (shape як у `p.data`
             або скаляр). Це чесно відповідає формулі «для кожного параметра θᵢ».
          2) **Group:** якщо для `p` немає `state[p]["qna_var_param"]`, беремо
             скаляр `group[vtilde_key]` (напр., середнє по параметрах на зонді).

      • (Опційно) глобальний L2 grad clipping усередині оптимізатора.

    ⚠️ Важливо:
      • Масштаб застосовується **до самого оновлення** (елемент-по-елементу),
        а не вбудовується в `lr`: так ми підтримуємо пер-параметрне (навіть
        пер-елементне) шкалування при збереженні формул Adam.
      • Після `step()` одноразові шуми очищаються:
          - `state[p]["qna_var_param"]` видаляється,
          - `group[vtilde_key]` теж очищується.
        Тренер має виставляти їх наново перед кожним кроком, щоб не було "протікання"
        значень між кроками.

    Параметри
    ---------
    params : iterable
        Iterable параметрів або param_groups.
    lr : float
        Базовий learning rate (до шумового масштабу).
    betas : Tuple[float, float]
        EMA коефіцієнти для першого та другого моментів.
    eps : float
        Числова стабільність.
    weight_decay : float
        L2-регуляризація (класичний Adam, НЕ decoupled).
    amsgrad : bool
        Увімкнути AMSGrad.
    maximize : bool
        Якщо True — робити підйом (змінити знак градієнта).

    --- QNA ---
    lambda_var : float
        Коефіцієнт штрафу шуму λ_var. 0 → QNA вимкнено.
    lr_min_mult, lr_max_mult : float
        Межі клеми для scale_λ.
    vtilde_key : str
        Ключ у param-group для групового Ṽ (скаляр), якщо немає пер-параметрного.

    --- Внутрішній кліпінг (опційно) ---
    clip_in_optimizer : bool
        Якщо True — робимо глобальний L2 clip до кроку Adam.
    max_norm : float
        Межа L2-норми градієнта.
    error_if_nonfinite : bool
        Якщо True — кидати помилку при NaN/Inf нормі; інакше пропускати крок.
    grad_eps : float
        Епсилон у дільнику під час кліпінгу.

    Як подавати Ṽ (document-style)
    -------------------------------
    • Після вимірювання шуму (Inline-DocPS) тренер може:
        # пер-параметрно (рекомендовано):
        optimizer.state[p]["qna_var_param"] = V_like_p  # shape як p, або скаляр
        group["qna_mode"] = "param"
        # бек-ап груповим:
        group["qna_vtilde"] = float(Vparam.mean().item())

        # або лише груповий:
        group["qna_vtilde"] = float(Vtilde)
        group["qna_mode"] = "group"

    • Після step() оптимізатор автоматично очищає ці одноразові поля.

    Сумісність із Adam
    ------------------
    Якщо λ_var = 0 і не задано Ṽ — це звичайний Adam (формули, bias-correction та AMSGrad
    без змін). Для чесного порівняння з базою використовуйте однаковий гр. кліпінг, wd і
    той самий спосіб обчислення градієнтів (у нас — Inline-DocPS).
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
                g = p.grad.detach()
                if g.numel() == 0:
                    continue
                total_sq += float((g.to(torch.float32) ** 2).sum().item())

        if not math.isfinite(total_sq):
            return False, float("nan")

        total_norm = math.sqrt(max(total_sq, 0.0))
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
    def _nan_to_num_(t: torch.Tensor, value_for_nan: float = 0.0) -> torch.Tensor:
        """In-place: NaN→value_for_nan, ±Inf→clamped (Inf → very large, treated by clamp later)."""
        return torch.nan_to_num_(t, nan=value_for_nan)

    @staticmethod
    def _make_tensor_like_param(src: Any, like: torch.Tensor) -> torch.Tensor:
        """
        Перенесе src (float|Tensor) на девайс/dtype 'like'. Допускає скаляр або
        той самий шейп; якщо шейп відрізняється і не скаляр — кидає помилку.
        """
        if isinstance(src, torch.Tensor):
            t = src.to(device=like.device, dtype=like.dtype, copy=False)
            if t.numel() == 1:
                return t.expand_as(like)
            if t.shape == like.shape:
                return t
            raise ValueError(
                f"qna_var_param tensor must be scalar or same shape as param: got {tuple(t.shape)} vs {tuple(like.shape)}"
            )
        else:
            # трактуємо як скаляр
            return torch.as_tensor(src, device=like.device, dtype=like.dtype).expand_as(like)

    @staticmethod
    def _build_scale_from_var(V: torch.Tensor, *, lam: float, s_min: float, s_max: float) -> torch.Tensor:
        """
        Повертає тензор масштабу: 1 / (1 + lam * clamp(V, min=0)), із клемою [s_min, s_max].
        """
        V_ = torch.clamp(V, min=0.0)
        scale = torch.reciprocal(1.0 + float(lam) * V_)
        if s_min > 0.0 or s_max < float("inf"):
            scale = torch.clamp(scale, min=float(s_min), max=float(s_max))
        return scale

    def _get_scale_tensor(self, group: Dict[str, Any], p: torch.nn.Parameter) -> Tuple[torch.Tensor, str]:
        """
        Повертає (scale_tensor, mode_used) для параметра p:
          • "param": використано state[p]["qna_var_param"] (тензор/скаляр)
          • "group": використано скаляр group[vtilde_key]
          • "off":   λ_var <= 0 або Ṽ відсутній → тензор з одиниць
        """
        lam = float(group.get("lambda_var", 0.0))
        if lam <= 0.0:
            return torch.ones_like(p.data), "off"

        s_min = float(group.get("lr_min_mult", 0.0))
        s_max = float(group.get("lr_max_mult", 1.0))
        key = str(group.get("vtilde_key", "qna_vtilde"))
        requested_mode = str(group.get("qna_mode", "") or "")

        # 1) пер-параметрний шлях через state[p]
        st = self.state[p]
        Vp = st.get("qna_var_param", None)
        if Vp is not None:
            try:
                V_like = self._make_tensor_like_param(Vp, p.data)  # broadcast/shape check
            except Exception as e:
                logger.warning("[QNAAdam] invalid qna_var_param for param '%s': %s → fallback to group/off",
                               p.__class__.__name__, e)
                Vp = None
            else:
                self._nan_to_num_(V_like, 0.0)
                scale = self._build_scale_from_var(V_like, lam=lam, s_min=s_min, s_max=s_max)
                return scale, "param"

        # 2) груповий шлях
        gv = group.get(key, None)
        if gv is not None:
            try:
                gv_t = self._make_tensor_like_param(gv, p.data)  # скаляр ок → розшириться
                self._nan_to_num_(gv_t, 0.0)
                scale = self._build_scale_from_var(gv_t, lam=lam, s_min=s_min, s_max=s_max)
                # Якщо просили "param", але немає Vp — попереджаємо один раз на групу
                if requested_mode.lower() == "param":
                    # не засмічуємо логи — короткий debug:
                    logger.debug("[QNAAdam] qna_mode='param' but state[p]['qna_var_param'] missing; used 'group'.")
                return scale, "group"
            except Exception as e:
                logger.warning("[QNAAdam] invalid group vtilde '%s': %s → fallback off", key, e)

        # 3) вимкнено/немає даних
        return torch.ones_like(p.data), "off"

    # ------------------------------ API ------------------------------------

    def step(self, closure: Optional[Any] = None):
        """Один крок оновлення (Adam + шум-адаптивний scale_λ, per-param коли є)."""
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

        # --- Головний цикл оновлення ---
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            amsgrad = bool(group.get("amsgrad", False))
            maximize = bool(group.get("maximize", False))

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("QNAAdam does not support sparse gradients")

                # gradient ascent?
                if maximize:
                    grad = -grad

                # --- state init ---
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]

                # weight decay (L2) — як у класичному Adam
                if weight_decay != 0.0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # --- моментні оновлення ---
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if amsgrad:
                    if "max_exp_avg_sq" not in state:
                        state["max_exp_avg_sq"] = torch.zeros_like(exp_avg_sq)
                    # узгодити dtype/device на випадок переносів
                    state["max_exp_avg_sq"] = state["max_exp_avg_sq"].to(exp_avg_sq.dtype).to(exp_avg_sq.device)
                    max_exp_avg_sq: torch.Tensor = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)

                # Bias correction
                state["step"] += 1
                step_t = state["step"]
                bias_c1 = 1.0 - beta1 ** step_t
                bias_c2 = 1.0 - beta2 ** step_t
                step_size = lr * math.sqrt(bias_c2) / bias_c1  # скаляр групи

                # --- QNA масштаб (тензор) для цього параметра ---
                scale_tensor, mode_used = self._get_scale_tensor(group, p)

                # --- застосувати оновлення Adam з пер-параметрним scale ---
                # Базовий "градієнтний" крок (тензор)
                update = exp_avg / denom  # не модифікуємо exp_avg/denom місці
                # Пер-елементне шум-адаптивне масштабування
                update = update.mul(scale_tensor)
                # Крок
                p.data.add_(update, alpha=-step_size)

                # Очистити одноразовий шум для цього параметра (не протікає на наступний step)
                if "qna_var_param" in state:
                    state.pop("qna_var_param", None)

            # Після групи: очистити груповий vtilde (він одноразовий)
            key = str(group.get("vtilde_key", "qna_vtilde"))
            if key in group:
                try:
                    group.pop(key, None)
                except Exception:
                    pass

            logger.debug(
                "[QNAAdam] group step: lr=%.3g, amsgrad=%s, wd=%.2g, qna_mode=%s, lam=%.2e",
                lr, amsgrad, weight_decay, str(group.get("qna_mode", "auto")), float(group.get("lambda_var", 0.0))
            )

        return loss
