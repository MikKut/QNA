"""
io_utils.py — утиліти вводу/виводу та допоміжні функції для препроцесингу.

Функціональність:
- Читання/запис YAML, NPY, NPZ, JSON.
- Перевірки NaN/Inf, приведення dtype.
- Створення тек (ensure_dir), контроль перезапису.
- Встановлення seed для reproducibility.
- 'Fingerprint' артефактів (метадані + хеш статистик), щоб не підхопити застарілий кеш.

Залежності: pyyaml, numpy (json — стандартна бібліотека)
"""

from __future__ import annotations
import json
import hashlib
import random

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Dict, Iterable, Mapping, Optional, Tuple, Union
from pathlib import Path
from collections.abc import Mapping
import dataclasses, datetime
import numpy as np
import yaml
import csv

import os
import time
import random
from typing import Optional

try:
    # наш проєктний логер
    from Code.logger import setup_logger
except Exception:  # fallback, якщо логер недоступний у момент імпорту
    def setup_logger(name: str):
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)
    
# Опціональна підтримка torch (не обов'язково встановлений)
try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


PathLike = Union[str, Path]


# ---------- Файлова допомога ----------

def ensure_dir(path: PathLike) -> Path:
    """Створює теку (якщо її немає). Повертає Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent_dir(file_path: PathLike) -> None:
    """Створює батьківську теку для файла (якщо її немає)."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def check_overwrite(file_path: PathLike, overwrite: bool) -> None:
    """
    Кидає помилку, якщо файл існує і overwrite=False.
    Інакше мовчки дозволяє перезапис.
    """
    p = Path(file_path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"File exists and overwrite=False: {p}")


# ---------- YAML / JSON ----------

def load_yaml(path: PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}



def _yaml_safe(x):
    try:
        import numpy as np
    except Exception:
        np = None

    if isinstance(x, str):
        return str(x)

    # Базові скаляри без змін
    if isinstance(x, (int, float, bool, type(None))):
        return x

    import dataclasses, enum, datetime as dt
    from pathlib import Path
    from collections.abc import Mapping, Sequence

    if dataclasses.is_dataclass(x):
        x = dataclasses.asdict(x)
    if isinstance(x, enum.Enum):
        return _yaml_safe(x.value)

    if np is not None:
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):  # всі numpy-скаляри (включно з np.str_)
            return _yaml_safe(x.item())  # рекурсивно, щоб пройти через гілку str вище

    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dt.datetime):
        return x.isoformat()
    if isinstance(x, Mapping):
        return { _yaml_safe(k): _yaml_safe(v) for k, v in x.items() }
    if isinstance(x, Sequence) and not isinstance(x, (bytes, bytearray)):
        # bytes/bytearray не вважаємо "послідовністю" для YAML; за потреби — явно обробіть
        return [ _yaml_safe(i) for i in x ]
    # fallback — рядкове подання будь-чого
    return str(x)


def save_yaml(data, path, overwrite=True):
    p = Path(path)
    ensure_parent_dir(p)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} already exists; use overwrite=True")
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(_yaml_safe(data), f, sort_keys=False, allow_unicode=True)


def save_json(data: Mapping[str, Any], path: PathLike, overwrite: bool = True, indent: int = 2) -> None:
    check_overwrite(path, overwrite)
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(path: PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- NumPy (NPY / NPZ) ----------

def save_npy(arr: np.ndarray, path: PathLike, overwrite: bool = True, dtype: Optional[str] = None) -> None:
    """
    Зберігає масив у .npy. Опційно кастує dtype ("float32"/"float64"/...).
    """
    check_overwrite(path, overwrite)
    ensure_parent_dir(path)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    np.save(Path(path), arr)


def load_npy(path: PathLike) -> np.ndarray:
    return np.load(Path(path), allow_pickle=False)


def save_npz(path: PathLike, overwrite: bool = True, **arrays: np.ndarray) -> None:
    """
    Зберігає декілька масивів у .npz:
        save_npz("meta.npz", components=..., mean=..., explained_variance=...)
    """
    check_overwrite(path, overwrite)
    ensure_parent_dir(path)
    np.savez(Path(path), **arrays)


def load_npz(path: PathLike) -> Dict[str, np.ndarray]:
    with np.load(Path(path)) as data:
        return {k: data[k] for k in data.files}



def save_csv(
    rows: List[Dict[str, Any]],
    path: PathLike,
    delimiter: str = ";",
    excel_sep_hint: bool = True,
    utf8_bom: bool = True,
    float_format: str = ".6g",
    decimal_comma: bool = True,
) -> None:
    """
    Збереження CSV з керованим роздільником + коректною серіалізацією чисел для Excel.
    За замовчуванням: delimiter=';', sep-хінт і UTF-8 з BOM — оптимально для UA/EU локалей.
    """
    ensure_parent_dir(path)
    encoding = "utf-8-sig" if utf8_bom else "utf-8"

    if not rows:
        with open(path, "w", newline="", encoding=encoding) as f:
            if excel_sep_hint:
                f.write(f"sep={delimiter}\n")
            f.write("empty\n")
        return

    header: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                header.append(k)

    # Серіалізатор значень у текст
    def _ser(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            s = format(float(v), float_format)      # напр. '2', '1.5'
            if decimal_comma:
                s = s.replace(".", ",")             # → '2' або '1,5'
            return s
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if v is None:
            return ""
        return str(v)

    with open(path, "w", newline="", encoding=encoding) as f:
        if excel_sep_hint:
            f.write(f"sep={delimiter}\n")
        writer = csv.DictWriter(f, fieldnames=header, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _ser(r.get(k, "")) for k in header})


# ---------- Перевірки коректності ----------

def assert_no_nan(arr: np.ndarray, name: str = "array") -> None:
    """
    Кидає ValueError, якщо знайдені NaN/Inf.
    """
    finite_mask = np.isfinite(arr) 
    if not finite_mask.all():
        n_bad = int((~finite_mask).sum())
        raise ValueError(f"{name}: found {n_bad} non-finite entries (NaN/Inf)")


def as_dtype(arr: np.ndarray, dtype: str = "float32") -> np.ndarray:
    """Привести масив до заданого dtype без копії, якщо можливо."""
    return arr.astype(dtype, copy=False)


# ---------- Seed / reproducibility ----------

def set_seed(seed: int) -> None:
    """
    Фіксує генератори випадковостей (python, numpy, torch*).
    """
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Опційно більш детермінований режим (може бути повільнішим)
        try:
            torch.backends.cudnn.deterministic = True  # type: ignore
            torch.backends.cudnn.benchmark = False     # type: ignore
        except Exception:
            pass

def seed_everything(
    seed: Optional[int],
    *,
    deterministic_torch: bool = False,
    logger=None,
) -> int:
    """
    Виставляє сиди для random / numpy / torch (+CUDA, якщо доступно) і, за можливості,
    для pennylane.numpy. Опційно вмикає детермінізм у PyTorch.

    Args:
        seed: Бажаний цілий сид. Якщо None — буде згенеровано.
        deterministic_torch: Якщо True — вмикає torch.use_deterministic_algorithms(True)
                             та відповідні флаги cudnn (корисно для детермінізму).
        logger: Проєктний логер; якщо None — створиться локально.

    Returns:
        int: фактично використаний сид.
    """
    log = logger or setup_logger(__name__)

    # 1) нормалізуємо seed
    if seed is None:
        # простий, але достатній генератор резервного сиду
        seed = int(time.time_ns() % (2**31 - 1))
        log.warning("seed_everything: seed=None → використано згенерований сид=%d", seed)
    else:
        try:
            seed = int(seed)
        except Exception as e:
            raise ValueError(f"seed_everything: seed має бути int/None, отримано {seed!r}") from e

    # 2) системні та інтерпретаторні сид-и
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 3) torch (CPU/CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        # У нашому проєкті ми працюємо на CPU (default.qubit), але хай буде опція:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
        except Exception:
            pass

    # 4) pennylane.numpy (не критично, але корисно для узгодженості)
    try:
        from pennylane import numpy as pnp  # type: ignore
        try:
            pnp.random.seed(seed)
        except Exception:
            # у деяких версіях pnp це звичайний numpy-аліас і вже посіданий
            pass
    except Exception:
        # PennyLane може бути ще не інстальовано/не потрібно у цьому контексті
        pass

    log.info("seed_everything: встановлено сид=%d (random/numpy/torch%s)", seed,
             "/cudnn-deterministic" if deterministic_torch else "")

    return seed

# ---------- Fingerprint для артефактів ----------

@dataclasses.dataclass(frozen=True)
class Fingerprint:
    """
    Паспорт артефакта: допомагає уникати «тихого» використання застарілих файлів.
    """
    method: str
    pca_dim: int
    eps: float
    seed: int
    stats_hash: str  # sha256 від статистик (μ/σ або медіани/IQR)
    created_at: str  # ISO-час
    numpy_version: str
    torch_version: Optional[str]
    extras: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "pca_dim": self.pca_dim,
            "eps": self.eps,
            "seed": self.seed,
            "stats_hash": self.stats_hash,
            "created_at": self.created_at,
            "numpy_version": self.numpy_version,
            "torch_version": self.torch_version,
            "extras": self.extras or {},
        }


def _jsonable_stats(stats: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Перетворює масиви у списки (з округленням), щоб стабільно хешувати.
    """
    out: Dict[str, Any] = {}
    for k, v in stats.items():
        if isinstance(v, np.ndarray):
            # Округляємо, щоб уникнути флюктуацій останніх бітів
            arr = v.astype(np.float64, copy=False)
            out[k] = np.round(arr, decimals=12).tolist()
        elif isinstance(v, (list, tuple)):
            out[k] = v
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        else:
            # спробуємо рекурсивно
            try:
                out[k] = _jsonable_stats(v)  # type: ignore[arg-type]
            except Exception:
                out[k] = str(v)
    return out


def compute_stats_hash(stats: Mapping[str, Any]) -> str:
    """
    Рахує sha256-хеш від статистик скейлера / метаданих PCA (стабільно).
    """
    jsonable = _jsonable_stats(stats)
    payload = json.dumps(jsonable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_fingerprint(
    *,
    method: str,
    pca_dim: int,
    eps: float,
    seed: int,
    stats: Mapping[str, Any],
    extras: Optional[Dict[str, Any]] = None,
) -> Fingerprint:
    return Fingerprint(
        method=method,
        pca_dim=pca_dim,
        eps=eps,
        seed=seed,
        stats_hash=compute_stats_hash(stats),
        created_at=datetime.now(timezone.utc).isoformat(),
        numpy_version=np.__version__,
        torch_version=(torch.__version__ if _HAS_TORCH else None),
        extras=extras,
    )


def save_fingerprint(fp: Fingerprint, path: PathLike, overwrite: bool = True) -> None:
    """
    Зберігає fingerprint у YAML/JSON (визначається за розширенням).
    """
    ext = Path(path).suffix.lower()
    data = fp.to_dict()
    if ext in (".yaml", ".yml"):
        save_yaml(data, path, overwrite=overwrite)
    elif ext == ".json":
        save_json(data, path, overwrite=overwrite, indent=2)
    else:
        raise ValueError(f"Unsupported fingerprint extension: {ext}")


def load_fingerprint(path: PathLike) -> Dict[str, Any]:
    ext = Path(path).suffix.lower()
    if ext in (".yaml", ".yml"):
        return load_yaml(path)
    if ext == ".json":
        return load_json(path)
    raise ValueError(f"Unsupported fingerprint extension: {ext}")


def fingerprint_matches(
    current: Mapping[str, Any],
    expected: Mapping[str, Any],
    keys: Iterable[str] = ("method", "pca_dim", "eps", "seed", "stats_hash"),
) -> bool:
    """
    Порівнює два «паспорти» артефактів по ключам.
    """
    for k in keys:
        if str(current.get(k)) != str(expected.get(k)):
            return False
    return True


# ---------- Завантаження конфіга проекту ----------

def load_project_config(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    tried = []

    if not p.is_absolute():
        print('It is not absolute')
        candidates = [
            Path.cwd() / p,                                         # поточна директорія
            Path(__file__).resolve().parents[2] / p,                # корінь репо + відносний шлях
            Path(__file__).resolve().parents[2] / "Code" / p ,                # корінь репо + відносний шлях
            Path(__file__).resolve().parents[2] / "Code" / "configs" / p.name,  # типовий каталог
        ]
        for c in candidates:
            print (c)
            tried.append(c)
            if c.exists():
                p = c
                break
        else:
            raise FileNotFoundError(
                "Config file not found. Tried:\n" + "\n".join(map(str, tried))
            )
    elif not p.exists():
        print (p)
        raise FileNotFoundError(f"Config file not found: {p}")

    cfg = load_yaml(p)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping (dict), got: {type(cfg)}")
    return cfg
