# Code/optim/registry.py
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import torch
from torch.optim import Optimizer

from Code.logger import setup_logger 

_LOG = setup_logger(__name__)

# ----------------------------- helpers ------------------------------------


def _normalize_name(name: str) -> str:
    """make 'AdamW', 'adam-w', 'adam_w' → 'adamw' (alnum only, lower)."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _pop_nested(d: MutableMapping[str, Any], key: str, default=None):
    return d.pop(key, default) if key in d else default


def _filter_kwargs(allowed: set[str], kwargs: Dict[str, Any], *, strict: bool, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    for k, v in list(kwargs.items()):
        if k in allowed:
            out[k] = v
        else:
            extra[k] = v
    if extra:
        msg = f"[optim.registry] Ignoring unsupported kwargs for {prefix}optimizer: {sorted(extra.keys())}"
        if strict:
            raise ValueError(msg)
        _LOG.warning(msg)
    return out


def _named_parameters_from(model: Optional[torch.nn.Module], params: Iterable[torch.nn.Parameter]) -> Mapping[str, torch.nn.Parameter]:
    """
    Try to get name→param mapping. Prefer model.named_parameters(); otherwise
    synthesize names param_0, param_1, ... for bare iterables.
    """
    if model is not None:
        return dict(model.named_parameters())
    # fallback: fabricate names (substring matching will be near-useless without names)
    return {f"param_{i}": p for i, p in enumerate(params)}


def _match_names(names: List[str], pattern: str) -> List[str]:
    """Return subset of names matching pattern (substring or regex:/.../)."""
    if len(pattern) >= 2 and pattern.startswith("/") and pattern.endswith("/"):
        rgx = re.compile(pattern[1:-1])
        return [n for n in names if rgx.search(n)]
    # default: case-insensitive substring
    pat = pattern.lower()
    return [n for n in names if pat in n.lower()]


def _build_param_groups(
    base_params: Iterable[torch.nn.Parameter],
    *,
    model: Optional[torch.nn.Module],
    base_kwargs: Dict[str, Any],
    groups_cfg: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build torch-style param_groups from config like:
      - match: "quantum" | "/classifier\\.weight/"
        lr_mult: 0.5
        weight_decay: 0.0
        ...
    Unmatched params go to a default group with base_kwargs.
    """
    named = _named_parameters_from(model, base_params)
    all_names = list(named.keys())
    assigned: set[str] = set()
    groups: List[Dict[str, Any]] = []

    for gi, g in enumerate(groups_cfg):
        pattern = str(g.get("match", "")).strip()
        if not pattern:
            _LOG.warning("[optim.registry] param_groups[%d] has empty 'match' — skipping", gi)
            continue
        matched = _match_names(all_names, pattern)
        matched = [n for n in matched if n not in assigned]
        if not matched:
            _LOG.warning("[optim.registry] param_groups[%d] pattern='%s' matched 0 params", gi, pattern)
            continue

        # derive group kwargs
        g_kwargs = dict(base_kwargs)  # start from base
        # explicit lr or lr_mult
        if "lr" in g:
            g_kwargs["lr"] = float(g["lr"])
        elif "lr_mult" in g and "lr" in g_kwargs:
            try:
                g_kwargs["lr"] = float(g["lr_mult"]) * float(g_kwargs["lr"])
            except Exception:
                _LOG.warning("[optim.registry] param_groups[%d] bad lr_mult=%s", gi, g.get("lr_mult"))

        # allow any other optimizer kw override present in group config
        for k, v in g.items():
            if k in {"match", "lr_mult"}:
                continue
            g_kwargs[k] = v

        groups.append({
            "params": [named[n] for n in matched],
            **g_kwargs,
        })
        assigned.update(matched)

    # default group with remaining params
    remaining = [named[n] for n in all_names if n not in assigned]
    if remaining:
        groups.append({"params": remaining, **base_kwargs})

    # log groups summary
    _LOG.info(
        "[optim.registry] built %d param group(s); sizes=%s",
        len(groups), [len(g["params"]) for g in groups]
    )
    return groups


# ----------------------------- public API ---------------------------------


def available() -> Dict[str, str]:
    """Return map of normalized optimizer names to human-readable labels."""
    return {
        "adam": "torch.optim.Adam",
        "adamw": "torch.optim.AdamW",
        "sgd": "torch.optim.SGD",
        "rmsprop": "torch.optim.RMSprop",
        "qnaadam": "Code.optim.qna_adam.QNAAdam (if available)",
    }


def register(name: str):
    """
    (Extension hook) Decorator to register custom optimizers at runtime.
    Example:
        @register("fancy_opt")
        class FancyOpt(torch.optim.Optimizer): ...
    """
    norm = _normalize_name(name)

    def _decorator(cls_or_fn):
        _CUSTOM[norm] = cls_or_fn
        _LOG.info("[optim.registry] registered custom optimizer '%s' -> %s", norm, cls_or_fn)
        return cls_or_fn

    return _decorator


# pre-seeded custom registry (empty by default)
_CUSTOM: Dict[str, Any] = {}


def get_optimizer(
    name: str,
    params: Iterable[torch.nn.Parameter],
    *,
    # common kwargs from YAML:
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    betas: Optional[Tuple[float, float]] = None,
    eps: Optional[float] = None,
    amsgrad: Optional[bool] = None,
    momentum: Optional[float] = None,
    alpha: Optional[float] = None,
    # optional advanced:
    param_groups: Optional[List[Dict[str, Any]]] = None,
    model: Optional[torch.nn.Module] = None,
    strict: bool = False,
    **kwargs: Any,
) -> Optimizer:
    """
    Factory: returns a torch.optim.Optimizer instance by name.
    Supports built-ins (adam/adamw/sgd/rmsprop) and 'qna_adam' if available.

    Parameters
    ----------
    name : str
        Optimizer name (case/format agnostic, e.g. 'AdamW', 'adam-w', 'adam_w').
    params : iterable of Parameters OR a pre-built list of param groups
        Model parameters. If you pass pre-built param groups, `param_groups` arg is ignored.
    param_groups : optional list of dicts
        High-level config for groups with 'match' and optional overrides; requires `model` to resolve names.
    model : optional nn.Module
        Model whose named_parameters() will be used to resolve param_groups.
    strict : bool
        If True, unknown kwargs will raise ValueError; otherwise they are ignored with a warning.

    Returns
    -------
    torch.optim.Optimizer
    """
    norm = _normalize_name(name)
    cfg: Dict[str, Any] = dict(
        lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, amsgrad=amsgrad,
        momentum=momentum, alpha=alpha,
    )
    # merge any loose kwargs into cfg (so users can pass optimizer-specific keys)
    cfg.update(kwargs or {})

    # if user already provided low-level torch param groups, respect them entirely
    # (detect by first element being dict with 'params')
    params_list: Any = list(params) if not isinstance(params, list) else params
    user_provided_groups = bool(params_list and isinstance(params_list[0], dict) and "params" in params_list[0])

    # flatten nested 'qna' config for qna_adam
    if norm == "qnaadam":
        qna_nested = _pop_nested(cfg, "qna", None)
        if isinstance(qna_nested, dict):
            cfg.update(qna_nested)

    # select backend and allowed keys
    allowed: set[str]
    ctor: Any = None

    if norm in _CUSTOM:
        ctor = _CUSTOM[norm]
        allowed = set(cfg.keys())  # assume custom handles provided keys
    elif norm == "adam":
        from torch.optim import Adam as _Adam
        ctor = _Adam
        allowed = {"lr", "betas", "eps", "weight_decay", "amsgrad", "maximize"}
    elif norm == "adamw":
        from torch.optim import AdamW as _AdamW
        ctor = _AdamW
        allowed = {"lr", "betas", "eps", "weight_decay", "amsgrad", "maximize"}
    elif norm == "sgd":
        from torch.optim import SGD as _SGD
        ctor = _SGD
        allowed = {"lr", "momentum", "weight_decay", "dampening", "nesterov"}
    elif norm == "rmsprop":
        from torch.optim import RMSprop as _RMSprop
        ctor = _RMSprop
        allowed = {"lr", "momentum", "alpha", "eps", "centered", "weight_decay"}
    elif norm == "qnaadam":
        try:
            from Code.optim.qna_adam import QNAAdam as _QNAAdam  # type: ignore
        except Exception as e:
            raise ImportError(
                "Requested optimizer 'qna_adam' but Code/optim/qna_adam.py (QNAAdam) is missing. "
                "Add it, or switch optim.name to 'adam'."
            ) from e
        ctor = _QNAAdam
        allowed = {
            # Adam-параметри:
            "lr", "betas", "eps", "weight_decay", "amsgrad", "maximize",
            # QNA-параметри:
            "lambda_var", "lr_min_mult", "lr_max_mult", "vtilde_key",
            "clip_in_optimizer", "max_norm", "error_if_nonfinite", "grad_eps",
        }


    else:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {sorted(available().keys())}")

    # prepare kwargs for ctor
    ctor_kwargs = _filter_kwargs(allowed, cfg, strict=strict, prefix=f"{norm}/")

    # build param groups if high-level config provided and user didn't pass low-level groups
    if (not user_provided_groups) and param_groups:
        try:
            groups = _build_param_groups(
                base_params=params_list,
                model=model,
                base_kwargs=ctor_kwargs,
                groups_cfg=param_groups,
            )
            params_for_ctor = groups
        except Exception as e:
            _LOG.warning("[optim.registry] Failed to build param_groups (%s). Falling back to flat params.", e)
            params_for_ctor = params_list
    else:
        params_for_ctor = params_list

    # instantiate optimizer
    opt: Optimizer = ctor(params_for_ctor, **ctor_kwargs)

    # Logging summary
    try:
        n_groups = len(params_for_ctor) if isinstance(params_for_ctor, list) and params_for_ctor and isinstance(params_for_ctor[0], dict) else 1
    except Exception:
        n_groups = 1
    _LOG.info(
        "[optim.registry] created optimizer '%s' (groups=%d) with keys=%s",
        norm, n_groups, sorted(ctor_kwargs.keys())
    )
    return opt


# ------------------------ (optional) LR schedulers -------------------------

def get_lr_scheduler(optimizer: Optimizer, **sched_cfg: Any):
    """
    Optional helper to create a torch LR scheduler from config, e.g.:
      type: cosine|step|multistep|plateau
      T_max: 50
      step_size: 10
      gamma: 0.5
      milestones: [30, 60]
      patience: 5
    Returns a scheduler or None if type is missing/unknown.
    """
    t = _normalize_name(sched_cfg.get("type", "")) if isinstance(sched_cfg, dict) else ""
    if not t:
        return None
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR, StepLR, MultiStepLR, ReduceLROnPlateau
    )
    if t == "cosine":
        return CosineAnnealingLR(optimizer, T_max=int(sched_cfg.get("tmax", sched_cfg.get("T_max", 50))))
    if t == "step":
        return StepLR(optimizer, step_size=int(sched_cfg.get("step_size", 10)), gamma=float(sched_cfg.get("gamma", 0.1)))
    if t == "multistep":
        ms = sched_cfg.get("milestones", [])
        return MultiStepLR(optimizer, milestones=list(ms), gamma=float(sched_cfg.get("gamma", 0.1)))
    if t == "plateau":
        return ReduceLROnPlateau(optimizer, mode=str(sched_cfg.get("mode", "min")), patience=int(sched_cfg.get("patience", 5)))
    _LOG.warning("[optim.registry] Unknown scheduler.type=%s", t)
    return None
