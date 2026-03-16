from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Base config at {path} is not a mapping.")
    return data


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def ensure_path(d: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    cur = d
    for k in keys:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    return cur


def make_variant(
    base: Dict[str, Any],
    *,
    seed: int,
    shots: int,
    optimizer_name: str,
    lr: float,
    lambda_var: float,
    epochs: int,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)

    project = ensure_path(cfg, "project")
    quantum = ensure_path(cfg, "quantum")
    training = ensure_path(cfg, "training")
    optim = ensure_path(cfg, "optim")
    qna = ensure_path(cfg, "optim", "qna")

    # keep seed fixed for scouting
    project["seed"] = int(seed)

    # training regime
    quantum["shots"] = int(shots)
    quantum["eval_shots"] = int(quantum.get("eval_shots", 2048))
    quantum["reseed_each_epoch"] = False

    training["epochs"] = int(epochs)

    # for compatibility with code that may inspect both sections
    training["optimizer"] = str(optimizer_name)

    training["lr"] = float(lr)

    optim["name"] = str(optimizer_name)
    optim["lr"] = float(lr)
    
    qna["lambda_var"] = float(lambda_var)

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate scouting configs for Adam / Adam-matched / QNA on low-shot regime."
    )
    parser.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Path to base YAML config.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where generated YAML configs will be written.",
    )
    args = parser.parse_args()

    base = load_yaml(args.base)
    out_dir: Path = args.out_dir

    seed = 42
    epochs = 35

    # fixed scouting grid
    shots_grid: List[int] = [7, 15, 25]
    qna_lambdas: List[int] = [8, 16, 32]

    # base learning rates
    lr_adam = 3.0e-4
    lr_adam_matched = 2.7e-4
    lr_qna = 3.0e-4

    generated: List[Path] = []

    # Adam baseline
    for shots in shots_grid:
        cfg = make_variant(
            base,
            seed=seed,
            shots=shots,
            optimizer_name="adam",
            lr=lr_adam,
            lambda_var=0.0,
            epochs=epochs,
        )
        path = out_dir / f"adam_s{shots}_seed{seed}.yaml"
        save_yaml(path, cfg)
        generated.append(path)

    # Adam-matched
    for shots in shots_grid:
        cfg = make_variant(
            base,
            seed=seed,
            shots=shots,
            optimizer_name="adam",
            lr=lr_adam_matched,
            lambda_var=0.0,
            epochs=epochs,
        )
        path = out_dir / f"adam_matched_s{shots}_seed{seed}.yaml"
        save_yaml(path, cfg)
        generated.append(path)

    # QNA
    for shots in shots_grid:
        for lam in qna_lambdas:
            cfg = make_variant(
                base,
                seed=seed,
                shots=shots,
                optimizer_name="qna_adam",
                lr=lr_qna,
                lambda_var=float(lam),
                epochs=epochs,
            )
            path = out_dir / f"qna_s{shots}_lam{lam}_seed{seed}.yaml"
            save_yaml(path, cfg)
            generated.append(path)

    # matrix helper file
    matrix_path = out_dir / "_matrix.tsv"
    with matrix_path.open("w", encoding="utf-8") as f:
        f.write("config\toptimizer\tshots\tlr\tlambda_var\tseed\tepochs\n")
        for shots in shots_grid:
            f.write(f"adam_s{shots}_seed{seed}.yaml\tadam\t{shots}\t{lr_adam}\t0\t{seed}\t{epochs}\n")
        for shots in shots_grid:
            f.write(
                f"adam_matched_s{shots}_seed{seed}.yaml\tadam_matched\t{shots}\t{lr_adam_matched}\t0\t{seed}\t{epochs}\n"
            )
        for shots in shots_grid:
            for lam in qna_lambdas:
                f.write(
                    f"qna_s{shots}_lam{lam}_seed{seed}.yaml\tqna_adam\t{shots}\t{lr_qna}\t{lam}\t{seed}\t{epochs}\n"
                )

    print("Generated configs:")
    for p in generated:
        print(f" - {p}")
    print(f"Matrix file: {matrix_path}")


if __name__ == "__main__":
    main()