import importlib.util
import json
from pathlib import Path


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
    path.parent.mkdir(parents=True)
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
