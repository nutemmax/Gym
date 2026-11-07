#!/usr/bin/env python3
"""Sample IFGym v3 splits and wrap them as NeMo Gym JSONL episodes."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path(
    "/capstor/store/cscs/swissai/infra01/reasoning/data/if-gym/datasets/v3-multiturn-if"
)


def gym_row(episode: dict[str, Any]) -> dict[str, Any]:
    turns = episode.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("source row must have at least one turn")
    prompt = turns[0].get("rendered_prompt")
    sample_id = episode.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("source row must have a non-empty sample_id")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"sample {sample_id!r} has no rendered first-turn prompt")
    if episode.get("n_turns") != len(turns):
        raise ValueError(f"sample {sample_id!r} has an inconsistent n_turns value")

    return {
        "id": sample_id,
        "responses_create_params": {
            "input": [{"role": "user", "content": prompt}],
        },
        "agent_ref": {"type": "responses_api_agents", "name": "ifgym_agent"},
        "episode": episode,
    }


def reservoir_sample(path: Path, count: int, seed: int) -> list[dict[str, Any]]:
    """Return a deterministic, uniform sample without loading the split in memory."""
    if count < 1:
        raise ValueError("sample count must be positive")
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    seen = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            row = gym_row(episode)
            seen += 1
            if len(selected) < count:
                selected.append(row)
                continue
            replacement = rng.randrange(seen)
            if replacement < count:
                selected[replacement] = row
    if not selected:
        raise ValueError(f"no episodes found in {path}")
    return selected


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary_path, path)


def prepare(
    source_dir: Path,
    output_dir: Path,
    *,
    train_samples: int = 100,
    validation_samples: int = 64,
    seed: int = 42,
) -> tuple[Path, Path]:
    train_rows = reservoir_sample(source_dir / "train/data.jsonl", train_samples, seed)
    validation_rows = reservoir_sample(
        source_dir / "val/data.jsonl", validation_samples, seed + 1
    )
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    write_jsonl(train_rows, train_path)
    write_jsonl(validation_rows, validation_path)
    return train_path, validation_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("SCRATCH", "/tmp"))
        / "ifgym-data/v3-multiturn-if",
    )
    parser.add_argument("--train-samples", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path, validation_path = prepare(
        args.source_dir,
        args.output_dir,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        seed=args.seed,
    )
    print(f"Wrote {args.train_samples} training samples to {train_path}")
    print(f"Wrote {args.validation_samples} validation samples to {validation_path}")


if __name__ == "__main__":
    main()
