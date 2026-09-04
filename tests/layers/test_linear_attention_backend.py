from types import SimpleNamespace

import pytest
import torch

from picosgl.core import Batch, Context, Request, clear_global_ctx, set_global_ctx
from picosgl.layers.linear_attention_backend.fla import FlashLinearAttentionBackend
from picosgl.layers.linear_attention_backend.native import NativeLinearAttentionBackend


def _make_inputs(seq_len: int) -> dict:
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size, num_k_heads, num_v_heads = 2, 2, 4
    head_k_dim = head_v_dim = 64
    return dict(
        query=torch.randn(
            batch_size, seq_len, num_k_heads, head_k_dim,
            device=device, dtype=dtype,
        ),
        key=torch.randn(
            batch_size, seq_len, num_k_heads, head_k_dim,
            device=device, dtype=dtype,
        ),
        value=torch.randn(
            batch_size, seq_len, num_v_heads, head_v_dim,
            device=device, dtype=dtype,
        ),
        gate=torch.randn(
            batch_size, seq_len, num_v_heads,
            device=device, dtype=dtype,
        ),
        beta=torch.randn(
            batch_size, seq_len, num_v_heads,
            device=device, dtype=dtype,
        ),
        A_log=torch.randn(num_v_heads, device=device, dtype=torch.float32),
        dt_bias=torch.randn(num_v_heads, device=device, dtype=torch.float32),
        initial_state=torch.randn(
            batch_size, num_v_heads, head_k_dim, head_v_dim,
            device=device, dtype=dtype,
        ),
    )


def _make_request(table_idx: int, cached_len: int, device_len: int) -> Request:
    return Request(
        input_ids=torch.zeros(device_len, dtype=torch.int32),
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=1,
        uid=table_idx,
        sampling_params=None,  # type: ignore
        cache_handle=None,  # type: ignore
        max_device_len=device_len + 1,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_decode_graph_metadata_uses_capture_buffers() -> None:
    device = torch.device("cuda")
    backend = NativeLinearAttentionBackend()
    ctx = Context(page_size=64)
    ctx.linear_state = SimpleNamespace(device=device)  # type: ignore
    ctx.state_table = torch.tensor(
        [[3, 4], [5, 6], [9, 9]], dtype=torch.int32, device=device
    )
    real_req = _make_request(0, cached_len=64, device_len=65)
    dummy_req = _make_request(2, cached_len=0, device_len=1)
    batch = Batch(reqs=[real_req], phase="decode")
    batch.padded_reqs = [real_req, dummy_req]
    capture_batch = Batch(reqs=[dummy_req, dummy_req], phase="decode")
    capture_batch.padded_reqs = capture_batch.reqs

    set_global_ctx(ctx)
    try:
        backend.init_capture_graph(max_seq_len=128, bs_list=[2])
        backend.prepare_for_capture(capture_batch)
        backend.prepare_metadata(batch)
        backend.prepare_for_replay(batch)
    finally:
        clear_global_ctx()

    assert backend.capture is not None
    assert backend.capture.read_slots.tolist() == [3, 9]
    assert backend.capture.write_slots.tolist() == [4, 9]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(("method", "seq_len"), [("decode", 1), ("prefill", 64)])
def test_fla_matches_native(method: str, seq_len: int) -> None:
    inputs = _make_inputs(seq_len)
    with torch.no_grad():
        expected_output, expected_state = getattr(
            NativeLinearAttentionBackend(), f"_{method}"
        )(**inputs)
        actual_output, actual_state = getattr(
            FlashLinearAttentionBackend(), f"_{method}"
        )(**inputs)

    torch.testing.assert_close(actual_output, expected_output, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual_state, expected_state, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_verify_matches_recurrent_reference() -> None:
    inputs = _make_inputs(seq_len=5)
    batch_size, seq_len, num_heads, head_dim = inputs["value"].shape
    write_slots = torch.arange(
        batch_size * seq_len,
        dtype=torch.int32,
        device=inputs["value"].device,
    ).view(batch_size, seq_len)
    state_pool = torch.empty(
        batch_size * seq_len,
        num_heads,
        head_dim,
        head_dim,
        dtype=inputs["value"].dtype,
        device=inputs["value"].device,
    )
    backend = NativeLinearAttentionBackend()

    with torch.no_grad():
        expected_output, expected_state = backend._decode(**inputs)
        actual_output = backend._verify(
            **inputs,
            write_slots=write_slots,
            state_pool=state_pool,
        )

    torch.testing.assert_close(actual_output, expected_output, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        state_pool[write_slots[:, -1].to(torch.int64)],
        expected_state.to(state_pool.dtype),
        rtol=1e-3,
        atol=1e-2,
    )
