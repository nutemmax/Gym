#!/usr/bin/env python3
"""Wrap generated IFGym phases, or sample legacy v3 splits, as NeMo Gym JSONL."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
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


def _validate_generated_episode(episode: object, path: Path, line_number: int) -> dict[str, Any]:
    """Validate one generated episode without changing its serialized content."""
    location = f"{path}:{line_number}"
    if not isinstance(episode, dict):
        raise ValueError(f"expected a JSON object at {location}")
    turns = episode.get("turns")
    if not isinstance(episode.get("sample_id"), str) or not episode["sample_id"]:
        raise ValueError(f"episode at {location} must have a non-empty sample_id")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"episode at {location} must have at least one turn")
    if len(turns) > 10:
        raise ValueError(f"episode at {location} has more than 10 turns")
    if episode.get("n_turns") != len(turns):
        raise ValueError(f"episode at {location} has an inconsistent n_turns value")
    if any(not isinstance(turn, dict) for turn in turns):
        raise ValueError(f"episode at {location} contains a malformed turn")
    if any(
        not isinstance(turn.get("rendered_prompt"), str)
        or not turn["rendered_prompt"].strip()
        for turn in turns
    ):
        raise ValueError(f"episode at {location} contains a turn without a rendered prompt")
    return episode


def _write_generated_split(source_path: Path, output_path: Path) -> int:
    """Stream one generated split to its own atomically replaced output file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            with source_path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        raise ValueError(f"blank line at {source_path}:{line_number}")
                    try:
                        episode = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"invalid JSON at {source_path}:{line_number}") from error
                    episode = _validate_generated_episode(episode, source_path, line_number)
                    output.write(json.dumps(gym_row(episode), ensure_ascii=False) + "\n")
                    count += 1
            if not count:
                raise ValueError(f"no episodes found in {source_path}")
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return count


def prepare_generated_root(source_root: Path, output_root: Path) -> dict[tuple[str, str], tuple[Path, int]]:
    """Adapt generated single/random/correlated train and val files without sampling.

    Outputs are written as ``<output_root>/<phase>/{train,validation}.jsonl``.
    """
    planned: list[tuple[str, str, Path, Path]] = []
    for phase in ("single", "random", "correlated"):
        for source_split, output_split in (("train", "train"), ("val", "validation")):
            source_path = source_root / phase / source_split / "data.jsonl"
            output_path = output_root / phase / f"{output_split}.jsonl"
            planned.append((phase, source_split, source_path, output_path))

    source_paths = {source_path.resolve() for _, _, source_path, _ in planned}
    for _, _, _, output_path in planned:
        if output_path.resolve() in source_paths:
            raise ValueError(f"output path overlaps a generated input path: {output_path}")

    outputs: dict[tuple[str, str], tuple[Path, int]] = {}
    for phase, source_split, source_path, output_path in planned:
        count = _write_generated_split(source_path, output_path)
        outputs[(phase, source_split)] = (output_path, count)
    return outputs


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
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


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
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    source.add_argument("--generated-root", type=Path)
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

    if args.generated_root is not None:
        outputs = prepare_generated_root(args.generated_root, args.output_dir)
        for (phase, split), (path, count) in outputs.items():
            print(f"Wrote {count} {phase}/{split} episodes to {path}")
        return

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
