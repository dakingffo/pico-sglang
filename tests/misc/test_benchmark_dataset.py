import json

import pytest

from benchmarks.bench_common import (
    _dataset_prompt_len,
    _take_dataset_prompts,
    load_dataset_prompts,
    resolve_dataset_path,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize
        assert add_generation_prompt
        return f"<user>{messages[0]['content']}<assistant>"

    def encode(self, text):
        return text.split()


def test_load_spec_bench():
    try:
        path = resolve_dataset_path("spec-bench")
    except FileNotFoundError:
        pytest.skip("Spec-Bench is an optional local benchmark dataset")
    assert path.endswith("benchmarks/datasets/spec_bench.jsonl")
    assert len(load_dataset_prompts(path)) == 480
    assert len(load_dataset_prompts("spec_bench", "coding")) == 10
    assert len(load_dataset_prompts("spec_bench", "qa,rag")) == 160


def test_dataset_uses_first_turn(tmp_path):
    path = tmp_path / "multi_turn.jsonl"
    path.write_text(
        json.dumps({"category": "test", "turns": ["first", "second"]}) + "\n",
        encoding="utf-8",
    )
    assert load_dataset_prompts(str(path)) == ["first"]


def test_dataset_sampling_is_deterministic_and_never_repeats(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps([{"prompt": str(i)} for i in range(4)]),
        encoding="utf-8",
    )
    first = _take_dataset_prompts(str(path), None, 3, seed=7)
    assert first == _take_dataset_prompts(str(path), None, 3, seed=7)
    assert len(set(first)) == 3
    with pytest.raises(ValueError, match="refusing to repeat prompts"):
        _take_dataset_prompts(str(path), None, 5, seed=7)


def test_dataset_token_length_includes_chat_template():
    tokenizer = FakeTokenizer()
    assert _dataset_prompt_len(tokenizer, "one two three") == 3
