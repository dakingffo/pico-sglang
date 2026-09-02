import pytest
import torch

from picosgl.layers.linear_attention_backend import GatedDeltaInput
from picosgl.layers.linear_attention_backend.fla import FlashLinearAttentionBackend
from picosgl.layers.linear_attention_backend.native import NativeLinearAttentionBackend


def _make_inputs(seq_len: int) -> GatedDeltaInput:
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size, num_k_heads, num_v_heads = 2, 2, 4
    head_k_dim = head_v_dim = 64
    return GatedDeltaInput(
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(("method", "seq_len"), [("decode", 1), ("prefill", 64)])
def test_fla_matches_native(method: str, seq_len: int) -> None:
    inputs = _make_inputs(seq_len)
    with torch.no_grad():
        expected_output, expected_state = getattr(NativeLinearAttentionBackend(), method)(inputs)
        actual_output, actual_state = getattr(FlashLinearAttentionBackend(), method)(inputs)

    torch.testing.assert_close(actual_output, expected_output, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual_state, expected_state, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_verify_matches_recurrent_reference() -> None:
    inputs = _make_inputs(seq_len=5)
    batch_size, seq_len, num_heads, head_dim = inputs.value.shape
    write_slots = torch.arange(
        batch_size * seq_len,
        dtype=torch.int32,
        device=inputs.value.device,
    ).view(batch_size, seq_len)
    state_pool = torch.empty(
        batch_size * seq_len,
        num_heads,
        head_dim,
        head_dim,
        dtype=inputs.value.dtype,
        device=inputs.value.device,
    )
    backend = NativeLinearAttentionBackend()

    with torch.no_grad():
        expected_output, expected_state = backend.decode(inputs)
        actual_output = backend.verify(inputs, write_slots, state_pool)

    torch.testing.assert_close(actual_output, expected_output, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        state_pool[write_slots[:, -1].to(torch.int64)],
        expected_state.to(state_pool.dtype),
        rtol=1e-3,
        atol=1e-2,
    )
