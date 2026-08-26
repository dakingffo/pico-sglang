import json

import pytest

from benchmarks.bench_common import (
    make_server_cmd,
    save_json,
    split_warmup_prompt,
    summarize,
)


def test_warmup_prompt_is_not_in_measured_batch():
    warmup, prompts, input_lens = split_warmup_prompt(
        ["measured-0", "measured-1", "warmup"], [1, 2, 3], True
    )
    assert warmup == "warmup"
    assert prompts == ["measured-0", "measured-1"]
    assert input_lens == [1, 2]


def test_ar_online_metrics_match_sglang_tpot():
    # start, three content chunks, finish_reason, DONE
    stats = summarize([[0.0, 2.0, 3.0, 4.0, 4.1, 4.2]], [8], [3], mtp=False)
    assert stats["ttft_ms"][0] == pytest.approx(2000.0)
    assert stats["tpot_ms"][0] == pytest.approx(1100.0)
    assert stats["output_tok_per_s"] == pytest.approx(3 / 4.2)


def test_mtp_acceptance_excludes_mandatory_target_token():
    # One prefill token plus two 2-token verify rounds => one accepted draft/round.
    stats = summarize([[0.0, 2.0, 3.0, 4.0, 4.1, 4.2]], [8], [5], mtp=True)
    assert stats["avg_accept"] == pytest.approx(1.0)
    assert stats["rounds"] == 2
    assert stats["tpot_ms"][0] == pytest.approx(550.0)


def test_dt_server_flag_is_rebuilt_once():
    cmd = make_server_cmd(
        ["--model-path", "model", "--enable-dt-separation"],
        port=1234,
        enable_specualtive_decoding=True,
        num_spec_tokens=3,
        enable_dt=True,
    )
    assert cmd.count("--enable-dt-separation") == 1


def test_save_json_persists_each_completed_point(tmp_path):
    path = tmp_path / "result.json"
    save_json(str(path), {"mtp": {"1": {"output_tok_per_s": 10.0}}})
    assert json.loads(path.read_text()) == {
        "mtp": {"1": {"output_tok_per_s": 10.0}}
    }
    assert not path.with_suffix(".json.tmp").exists()
