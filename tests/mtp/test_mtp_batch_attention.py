from types import SimpleNamespace

import torch

from picosgl.distributed import DistributedInfo, set_tp_info


def _reset_tp() -> None:
    import picosgl.distributed.info as info

    info._TP_INFO = None
    set_tp_info(DistributedInfo(rank=0, size=1))


def _make_attention():
    _reset_tp()
    from picosgl.layers.qwen3_5 import Qwen3_5Attention

    config = SimpleNamespace(
        head_dim=8,
        hidden_size=32,
        num_qo_heads=4,
        num_kv_heads=2,
        rms_norm_eps=1e-6,
        rotary_config=SimpleNamespace(rotary_dim=8, max_position=128, base=10_000.0),
    )
    attention = Qwen3_5Attention(config, layer_id=0, paged=False)
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for tensor in attention.state_dict().values():
            tensor.copy_(torch.randn(tensor.shape, generator=generator) * 0.05)
    return attention


def test_batch_attention_matches_individual_variable_length() -> None:
    attention = _make_attention()
    lengths = [2, 4, 3]
    max_len = max(lengths)
    x = torch.zeros(len(lengths), max_len, 32)
    positions = torch.zeros(len(lengths), max_len, dtype=torch.int64)
    valid = torch.zeros(len(lengths), max_len, dtype=torch.bool)
    individual = []

    generator = torch.Generator().manual_seed(23)
    for i, length in enumerate(lengths):
        row = torch.randn(length, 32, generator=generator)
        pos = torch.arange(10 + i, 10 + i + length)
        start = max_len - length
        x[i, start:] = row
        positions[i, start:] = pos
        valid[i, start:] = True
        individual.append(attention.forward_with_kv(row, pos))

    batch_out, batch_kv, batch_valid = attention.forward_with_kv_batch(
        x, positions, valid
    )
    for i, length in enumerate(lengths):
        start = max_len - length
        expected_out, (expected_k, expected_v) = individual[i]
        torch.testing.assert_close(batch_out[i, start:], expected_out)
        torch.testing.assert_close(batch_kv[0][i, :, start:], expected_k)
        torch.testing.assert_close(batch_kv[1][i, :, start:], expected_v)
        assert torch.equal(batch_valid[i], valid[i])


def test_batch_attention_cached_step_matches_individual() -> None:
    attention = _make_attention()
    lengths = [2, 4]
    max_len = max(lengths)
    x = torch.zeros(2, max_len, 32)
    positions = torch.zeros(2, max_len, dtype=torch.int64)
    valid = torch.zeros(2, max_len, dtype=torch.bool)
    rows = []
    generator = torch.Generator().manual_seed(29)

    for i, length in enumerate(lengths):
        row = torch.randn(length, 32, generator=generator)
        pos = torch.arange(length)
        start = max_len - length
        x[i, start:] = row
        positions[i, start:] = pos
        valid[i, start:] = True
        rows.append((row, pos))

    _, batch_kv, batch_valid = attention.forward_with_kv_batch(x, positions, valid)
    next_x = torch.randn(2, 1, 32, generator=generator)
    next_pos = torch.tensor([[lengths[0]], [lengths[1]]])
    batch_out, _, _ = attention.forward_with_kv_batch(
        next_x,
        next_pos,
        torch.ones(2, 1, dtype=torch.bool),
        batch_kv,
        batch_valid,
    )

    for i, (row, pos) in enumerate(rows):
        _, kv = attention.forward_with_kv(row, pos)
        expected, _ = attention.forward_with_kv(
            next_x[i], next_pos[i], kv
        )
        torch.testing.assert_close(batch_out[i], expected)


def test_indexed_native_kv_pool_matches_dense_batch_attention() -> None:
    from picosgl.speculator.drafters.mtp.attention import MTPAttentionBackend
    from picosgl.speculator.drafters.mtp.pool import MTPKVPool

    attention = _make_attention()
    lengths = [2, 4, 3]
    offsets = [0, 2, 6]
    total = sum(lengths)
    generator = torch.Generator().manual_seed(31)
    x = torch.randn(total, 32, generator=generator)
    positions = torch.cat([torch.arange(length) for length in lengths])

    query, gate, key, value = attention.project_for_cache(x, positions)
    pool = MTPKVPool(
        max_running_req=3,
        window_size=max(lengths),
        max_batch_size=3,
        num_spec_tokens=1,
        num_kv_heads=attention.num_kv_heads,
        head_dim=attention.head_dim,
        dtype=key.dtype,
        device=torch.device("cpu"),
    )
    pool.k[:total].copy_(key)
    pool.v[:total].copy_(value)
    indices = torch.zeros(3, max(lengths), dtype=torch.int32)
    valid = torch.zeros(3, max(lengths), dtype=torch.bool)
    last_rows = []
    for i, (offset, length) in enumerate(zip(offsets, lengths)):
        indices[i, :length] = torch.arange(offset, offset + length, dtype=torch.int32)
        valid[i, :length] = True
        last_rows.append(offset + length - 1)

    last_rows = torch.tensor(last_rows)
    backend = MTPAttentionBackend(
        "auto",
        attention.num_qo_heads,
        attention.num_kv_heads,
        attention.head_dim,
        key.dtype,
        torch.device("cpu"),
    )
    pooled = backend.forward(query[last_rows], pool, indices, valid, lengths)
    pooled = pooled * torch.sigmoid(gate[last_rows])
    pooled = attention.o_proj.forward(pooled.reshape(len(lengths), -1))
    expected = []
    for offset, length in zip(offsets, lengths):
        row = x[offset : offset + length]
        pos = positions[offset : offset + length]
        output, _ = attention.forward_with_kv(row, pos)
        expected.append(output[-1])

    torch.testing.assert_close(pooled, torch.stack(expected))
