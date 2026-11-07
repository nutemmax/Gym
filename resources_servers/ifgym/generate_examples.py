# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate reviewable synthetic episodes using IFGym's original schedule builder."""

import json
from pathlib import Path

from ifgym.data.conversation import ConversationInternalRepr
from ifgym.templates.activation_rules import (
    ConstraintIntroduction as Intro,
)
from ifgym.templates.activation_rules import (
    ConstraintRevocation as Revoke,
)
from ifgym.templates.activation_rules import (
    Conversation,
    EveryNthTurn,
    NextNTurns,
    ThisTurnOnly,
    Turn,
    UntilRevoked,
)


def add(handle, constraint_id, kwargs=None, rule=None):
    return Intro(handle, constraint_id, kwargs or {}, rule or UntilRevoked())


def examples():
    scenarios = [
        [
            (
                "Describe gardening.",
                "Describe gardening. Include seed and do not use commas in every response until changed.",
                [
                    add("word1", "keywords:existence", {"keywords": ["seed"]}),
                    add("comma", "punctuation:no_comma"),
                ],
            ),
            (
                "Give another gardening fact.",
                "Give another gardening fact. For this response include bloom exactly once; keep the earlier rules.",
                [
                    add("once", "keywords:word_once", {"keyword": "bloom"}, ThisTurnOnly()),
                ],
            ),
            (
                "Describe a flower.",
                "Stop requiring seed. Include flower instead for the rest of the conversation. Do not use commas.",
                [
                    Revoke("word1"),
                    add("word2", "keywords:existence", {"keywords": ["flower"]}),
                ],
            ),
            (
                "Finish.",
                "Finish with a gardening fact ending in All done. Keep the other active rules.",
                [
                    add("ending", "startend:end_checker", {"end_phrase": "All done."}, ThisTurnOnly()),
                ],
            ),
        ],
        [
            (
                "Describe an orchard.",
                "Describe an orchard in lowercase English and do not use the word banana. Keep both "
                "rules until changed.",
                [
                    add("case1", "change_case:english_lowercase"),
                    add("ban", "keywords:forbidden_words", {"forbidden_words": ["banana"]}),
                ],
            ),
            (
                "Describe a tree.",
                "Describe a tree. Include orchard in this response and the next; keep the earlier rules.",
                [
                    add("orchard", "keywords:existence", {"keywords": ["orchard"]}, NextNTurns(2)),
                ],
            ),
            (
                "Describe fruit.",
                "Now use uppercase English instead of lowercase. Describe fruit, keeping the other active rules.",
                [
                    Revoke("case1"),
                    add("case2", "change_case:english_capital"),
                ],
            ),
        ],
        [
            (
                "Give gardening tips.",
                "Give exactly two bullet points about gardening; include water. Keep both rules until changed.",
                [
                    add("bullets1", "detectable_format:number_bullet_lists", {"num_bullets": 2}),
                    add("water", "keywords:existence", {"keywords": ["water"]}),
                ],
            ),
            (
                "Give more tips.",
                "Give exactly three bullet points instead of two. Keep including water.",
                [
                    Revoke("bullets1"),
                    add("bullets2", "detectable_format:number_bullet_lists", {"num_bullets": 3}),
                ],
            ),
            (
                "Give a final tip.",
                "Stop requiring bullet points. Give a final tip with no commas; continue to include water.",
                [
                    Revoke("bullets2"),
                    add("comma", "punctuation:no_comma", rule=ThisTurnOnly()),
                ],
            ),
        ],
        [
            (
                "Describe a planet.",
                "Reply as JSON without commas. Keep both rules until changed.",
                [
                    add("json", "detectable_format:json_format"),
                    add("comma", "punctuation:no_comma"),
                ],
            ),
            (
                "Describe Mars.",
                "Reply about Mars and include mars exactly once. Keep the JSON and no-comma rules.",
                [
                    add("mars", "keywords:word_once", {"keyword": "mars"}, ThisTurnOnly()),
                ],
            ),
            (
                "Describe space.",
                "Stop using JSON. Start your response with Answer; continue to avoid commas.",
                [
                    Revoke("json"),
                    add("first", "first_word:first_word_answer", {"first_word": "Answer"}),
                ],
            ),
        ],
        [
            (
                "Describe a river.",
                "Describe a river with a title in <<double angle brackets>> and include river. Keep these rules.",
                [
                    add("title", "detectable_format:title"),
                    add("river", "keywords:existence", {"keywords": ["river"]}),
                ],
            ),
            (
                "Describe the riverbank.",
                "Describe the riverbank without using pollution. Keep the earlier rules.",
                [
                    add("ban", "keywords:forbidden_words", {"forbidden_words": ["pollution"]}),
                ],
            ),
            (
                "Describe its flow.",
                "Stop requiring a title. End the response with the word flow and keep the other rules.",
                [
                    Revoke("title"),
                    add("last", "last_word:last_word_answer", {"last_word": "flow"}),
                ],
            ),
        ],
        [
            (
                "Describe a book.",
                "Begin with Summary and end with Done. Keep both rules until changed.",
                [
                    add("first1", "first_word:first_word_answer", {"first_word": "Summary"}),
                    add("end", "startend:end_checker", {"end_phrase": "Done."}),
                ],
            ),
            (
                "Describe its author.",
                "Begin with Note instead of Summary. Keep ending with Done.",
                [
                    Revoke("first1"),
                    add("first2", "first_word:first_word_answer", {"first_word": "Note"}),
                ],
            ),
            (
                "Describe its setting.",
                "Describe its setting without commas for this response; keep the existing rules.",
                [
                    add("comma", "punctuation:no_comma", rule=ThisTurnOnly()),
                ],
            ),
        ],
        [
            (
                "Describe colors.",
                "Include blue at least twice and never use red. Keep these rules until changed.",
                [
                    add(
                        "freq",
                        "keywords:frequency",
                        {"keyword": "blue", "frequency": 2, "relation": "at least"},
                    ),
                    add("ban", "keywords:forbidden_words", {"forbidden_words": ["red"]}),
                ],
            ),
            (
                "Describe a landscape.",
                "Stop requiring blue. Include green exactly once instead; continue to avoid red.",
                [
                    Revoke("freq"),
                    add("once", "keywords:word_once", {"keyword": "green"}),
                ],
            ),
            (
                "Describe another landscape.",
                "Start with the same first word as your previous answer. Include a word from an earlier "
                "answer that was absent from the previous one. Keep the active rules.",
                [
                    add("history", "history:same_first_word_as_previous", rule=ThisTurnOnly()),
                    add("earlier", "history:reuse_earlier_word", rule=ThisTurnOnly()),
                ],
            ),
        ],
        [
            (
                "Describe music.",
                "Write in uppercase English without commas until I change the case rule.",
                [
                    add("upper", "change_case:english_capital"),
                    add("comma", "punctuation:no_comma"),
                ],
            ),
            (
                "Describe rhythm.",
                "Use lowercase English instead of uppercase. Keep avoiding commas.",
                [
                    Revoke("upper"),
                    add("lower", "change_case:english_lowercase"),
                ],
            ),
            (
                "List instruments.",
                "Give exactly two bullet points in lowercase English without commas.",
                [
                    add(
                        "bullets",
                        "detectable_format:number_bullet_lists",
                        {"num_bullets": 2},
                        ThisTurnOnly(),
                    ),
                ],
            ),
        ],
        [
            (
                "Describe the sky.",
                "Include earth and sky in every response. Use no commas in this response and the next.",
                [
                    add("words", "keywords:existence", {"keywords": ["earth", "sky"]}),
                    add("comma", "punctuation:no_comma", rule=NextNTurns(2)),
                ],
            ),
            (
                "Describe a sunset.",
                "Describe a sunset; keep the active rules.",
                [],
            ),
            (
                "Describe a sunrise.",
                "Describe a sunrise, and put a title in <<double angle brackets>> for this response. "
                "The no-comma rule has expired.",
                [
                    add("title", "detectable_format:title", rule=ThisTurnOnly()),
                ],
            ),
        ],
        [
            (
                "Describe a journey.",
                "Start with Begin and include path exactly once. Keep both rules until changed.",
                [
                    add("first1", "first_word:first_word_answer", {"first_word": "Begin"}),
                    add("once", "keywords:word_once", {"keyword": "path"}),
                ],
            ),
            (
                "Describe a route.",
                "Describe the route. On every other response starting now, do not use commas; keep the earlier rules.",
                [
                    add("comma", "punctuation:no_comma", rule=EveryNthTurn(2)),
                ],
            ),
            (
                "Describe the destination.",
                "Begin with Arrive instead of Begin. Keep the other active rules.",
                [
                    Revoke("first1"),
                    add("first2", "first_word:first_word_answer", {"first_word": "Arrive"}),
                ],
            ),
            (
                "Finish the journey.",
                "End with Safe travels. Keep the other active rules.",
                [
                    add("end", "startend:end_checker", {"end_phrase": "Safe travels."}, ThisTurnOnly()),
                ],
            ),
        ],
    ]
    rows = []
    for index, scenario in enumerate(scenarios, 1):
        sample_id = f"ifgym-synthetic-{index:02d}"
        conversation = Conversation(
            key=sample_id, turns=[Turn(task_text=task, events=events) for task, _, events in scenario]
        )
        episode = ConversationInternalRepr(
            sample_id=sample_id,
            split_type="example",
            n_turns=len(scenario),
            conversation=conversation,
            turn_rules=[
                prompt.removeprefix(task).strip() if prompt.startswith(task) else prompt
                for task, prompt, _ in scenario
            ],
            rendered_prompts=[prompt for _, prompt, _ in scenario],
            metadata={"synthetic": True},
        ).to_dict()
        rows.append(
            {
                "id": sample_id,
                "responses_create_params": {"input": [{"role": "user", "content": scenario[0][1]}]},
                "agent_ref": {"type": "responses_api_agents", "name": "ifgym_agent"},
                "episode": episode,
            }
        )
    return rows


if __name__ == "__main__":
    path = Path(__file__).parent / "data/example.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples()))
