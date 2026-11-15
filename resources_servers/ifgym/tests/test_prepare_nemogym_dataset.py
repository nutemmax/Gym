import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parent.parent / "prepare_nemogym_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_nemogym_dataset", SCRIPT)
prepare_dataset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_dataset)


def source_episode(sample_id: str) -> dict:
    prompt = f"Prompt for {sample_id}"
    return {
        "sample_id": sample_id,
        "split_type": "test",
        "n_turns": 1,
        "turns": [{"rendered_prompt": prompt, "active_constraints": [], "events": []}],
    }


def write_source_split(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for index in range(count):
            output.write(json.dumps(source_episode(f"sample-{index}")) + "\n")


def test_prepare_samples_and_wraps_train_and_validation_splits(tmp_path):
    source_dir = tmp_path / "source"
    write_source_split(source_dir / "train/data.jsonl", 20)
    write_source_split(source_dir / "val/data.jsonl", 10)

    train_path, validation_path = prepare_dataset.prepare(
        source_dir, tmp_path / "output", train_samples=5, validation_samples=3, seed=17
    )
    train_rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    validation_rows = [
        json.loads(line) for line in validation_path.read_text().splitlines()
    ]

    assert len(train_rows) == 5
    assert len(validation_rows) == 3
    assert len({row["id"] for row in train_rows}) == 5
    assert len({row["id"] for row in validation_rows}) == 3
    assert not {row["id"] for row in train_rows} & {
        row["id"] for row in validation_rows
    }
    row = train_rows[0]
    assert row["responses_create_params"]["input"] == [
        {"role": "user", "content": row["episode"]["turns"][0]["rendered_prompt"]}
    ]
    assert row["agent_ref"] == {
        "type": "responses_api_agents",
        "name": "ifgym_agent",
    }
    assert row["episode"]["sample_id"] == row["id"]


def test_reservoir_sampling_is_reproducible(tmp_path):
    source = tmp_path / "train.jsonl"
    write_source_split(source, 30)

    first = prepare_dataset.reservoir_sample(source, 7, seed=9)
    second = prepare_dataset.reservoir_sample(source, 7, seed=9)

    assert [row["id"] for row in first] == [row["id"] for row in second]


def generated_episode(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "split_type": "generated",
        "n_turns": 2,
        "turns": [
            {"turn_idx": 1, "rendered_prompt": f"First {sample_id}", "raw": [1, 2]},
            {"turn_idx": 2, "rendered_prompt": f"Second {sample_id}", "extra": {"kept": True}},
        ],
        "metadata": {"preserve": ["all", "fields"]},
    }


def test_prepare_generated_root_streams_all_phases_and_splits(tmp_path):
    source_root = tmp_path / "generated"
    expected: dict[tuple[str, str], list[dict]] = {}
    for phase in ("single", "random", "correlated"):
        for split in ("train", "val"):
            episodes = [generated_episode(f"{phase}-{split}-{index}") for index in range(2)]
            expected[(phase, split)] = episodes
            path = source_root / phase / split / "data.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("".join(json.dumps(episode) + "\n" for episode in episodes))

    outputs = prepare_dataset.prepare_generated_root(source_root, tmp_path / "output")

    assert set(outputs) == set(expected)
    for (phase, source_split), episodes in expected.items():
        path, count = outputs[(phase, source_split)]
        assert count == len(episodes)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row["id"] for row in rows] == [episode["sample_id"] for episode in episodes]
        assert [row["episode"] for row in rows] == episodes
        assert all(
            row["responses_create_params"]["input"]
            == [{"role": "user", "content": episode["turns"][0]["rendered_prompt"]}]
            for row, episode in zip(rows, episodes, strict=True)
        )
        assert all(
            row["agent_ref"] == {"type": "responses_api_agents", "name": "ifgym_agent"}
            for row in rows
        )
        assert path.parent.name == phase
        assert path.name == ("train.jsonl" if source_split == "train" else "validation.jsonl")


def test_prepare_generated_root_failure_keeps_existing_output_atomic(tmp_path):
    source_root = tmp_path / "generated"
    source_path = source_root / "single/train/data.jsonl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        json.dumps(generated_episode("valid")) + "\n" + "{malformed json}\n"
    )
    output_path = tmp_path / "output/single/train.jsonl"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("existing output\n")

    try:
        prepare_dataset.prepare_generated_root(source_root, tmp_path / "output")
    except ValueError as error:
        assert "invalid JSON" in str(error)
    else:
        raise AssertionError("malformed generated input should fail")

    assert output_path.read_text() == "existing output\n"
    assert list(output_path.parent.glob("*.tmp")) == []


def test_prepare_generated_root_rejects_blank_lines_and_overlong_episodes(tmp_path):
    source_root = tmp_path / "generated"
    path = source_root / "single/train/data.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(generated_episode("valid")) + "\n\n")

    try:
        prepare_dataset.prepare_generated_root(source_root, tmp_path / "output")
    except ValueError as error:
        assert "blank line" in str(error)
    else:
        raise AssertionError("blank line should fail")

    overlong = generated_episode("too-long")
    overlong["turns"] *= 6
    overlong["n_turns"] = len(overlong["turns"])
    path.write_text(json.dumps(overlong) + "\n")
    try:
        prepare_dataset.prepare_generated_root(source_root, tmp_path / "output")
    except ValueError as error:
        assert "more than 10 turns" in str(error)
    else:
        raise AssertionError("overlong episode should fail")


def test_prepare_generated_root_rejects_empty_splits_without_replacing_output(tmp_path):
    source_root = tmp_path / "generated"
    path = source_root / "single/train/data.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("")
    output_path = tmp_path / "output/single/train.jsonl"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("existing output\n")

    with pytest.raises(ValueError, match="no episodes found"):
        prepare_dataset.prepare_generated_root(source_root, tmp_path / "output")

    assert output_path.read_text() == "existing output\n"


def test_prepare_generated_root_rejects_any_turn_without_prompt_without_leaking_it(tmp_path):
    source_root = tmp_path / "generated"
    path = source_root / "single/train/data.jsonl"
    path.parent.mkdir(parents=True)
    episode = generated_episode("sample")
    episode["turns"][0]["rendered_prompt"] = "PRIVATE PROMPT TEXT"
    episode["turns"][1]["rendered_prompt"] = "   "
    path.write_text(json.dumps(episode) + "\n")

    with pytest.raises(ValueError) as error:
        prepare_dataset.prepare_generated_root(source_root, tmp_path / "output")

    assert "rendered prompt" in str(error.value)
    assert "PRIVATE PROMPT TEXT" not in str(error.value)


def test_prepare_generated_root_rejects_output_aliasing_raw_input(tmp_path):
    source_root = tmp_path / "generated"
    source_path = source_root / "single/train/data.jsonl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(json.dumps(generated_episode("sample")) + "\n")
    output_root = tmp_path / "output"
    (output_root / "single").mkdir(parents=True)
    (output_root / "single/train.jsonl").symlink_to(source_path)

    with pytest.raises(ValueError, match="overlaps a generated input path"):
        prepare_dataset.prepare_generated_root(source_root, output_root)

    assert source_path.read_text() == json.dumps(generated_episode("sample")) + "\n"
