# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay saved Gym rollout checks against the original IFGym checker registry."""

import argparse
import json
import math
from pathlib import Path

from resources_servers.ifgym.episode import Episode, initial_user_text


def verify_rollout(row: dict, result: dict) -> None:
    info = result["info"]
    if info["sample_id"] != row["episode"]["sample_id"]:
        raise ValueError("rollout sample_id does not match the dataset")
    episode = Episode(row["episode"])
    first_prompt = episode.turns[0]["rendered_prompt"]
    if initial_user_text(row["responses_create_params"]) != first_prompt:
        raise ValueError("dataset initial prompt differs from the episode")
    if initial_user_text(result["responses_create_params"]) != first_prompt:
        raise ValueError("rollout initial prompt differs from the episode")
    details = info["turn_details"]
    if not details and not (
        result.get("truncated")
        and info.get("truncation_reason") == "context_exhausted"
        and (result.get("instance_config") or {}).get("mask_sample")
    ):
        raise ValueError("rollout has no assistant turns")
    assistant_text = [""]
    followups = []
    for item in result["response"]["output"]:
        role = item.get("role")
        content = item.get("content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if part.get("type") in {"input_text", "output_text"}
            )
        if role == "user":
            followups.append(content)
            assistant_text.append("")
        elif role == "assistant" and item.get("type") == "message":
            assistant_text[-1] += content
    if result.get("truncated") and len(assistant_text) == len(details) + 1 and not assistant_text[-1]:
        assistant_text.pop()
    if assistant_text != [detail["response"] for detail in details]:
        raise ValueError("assistant transcript differs from recorded turn responses")
    if followups != [turn["rendered_prompt"] for turn in episode.turns[1 : len(followups) + 1]]:
        raise ValueError("follow-up transcript differs from the dataset")
    for recorded in details:
        _, _, _, replayed = episode.step(recorded["response"])
        if recorded != replayed["turn_details"][-1]:
            raise ValueError(f"verification differs on turn {recorded['turn_idx']}")
    if info["turn_rewards"] != episode.rewards:
        raise ValueError("recorded turn rewards differ from replay")
    if bool(result["terminated"]) != (episode.index == len(episode.turns)):
        raise ValueError("termination does not match the replayed turns")
    if bool(result.get("truncated")) == bool(result["terminated"]):
        raise ValueError("rollout must be either terminated or truncated")
    mean = sum(episode.rewards) / len(episode.rewards) if episode.rewards else 0.0
    if not math.isclose(result["reward"], mean, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("trajectory mean differs from replay")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--rollouts", required=True, type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    dataset = {row["episode"]["sample_id"]: row for row in rows}
    if len(dataset) != len(rows):
        raise ValueError("dataset contains duplicate sample IDs")
    results = [json.loads(line) for line in args.rollouts.read_text().splitlines() if line.strip()]
    if not results:
        raise ValueError("rollout file is empty")
    for index, result in enumerate(results, 1):
        sample_id = result["info"]["sample_id"]
        try:
            verify_rollout(dataset[sample_id], result)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"rollout {index} ({sample_id}): {error}") from error
    print(f"Verified {len(results)} rollouts against {args.dataset}")


if __name__ == "__main__":
    main()
