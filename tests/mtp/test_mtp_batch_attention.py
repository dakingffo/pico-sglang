import torch
import torch.nn.functional as F

from picosgl.distributed import DistributedInfo, set_tp_info


def _reset_tp() -> None:
    import picosgl.distributed.info as info

    info._TP_INFO = None
    set_tp_info(DistributedInfo(rank=0, size=1))


def _make_attention():
    _reset_tp()
    from picosgl.layers import GatedRotaryAttention

    attention = GatedRotaryAttention(
        hidden_size=32,
        head_dim=8,
        num_qo_heads=4,
        num_kv_heads=2,
        layer_id=0,
        rotary_dim=8,
        max_position=128,
        rope_base=10_000.0,
        rope_scaling=None,
        qk_norm_eps=1e-6,
        zero_centered_norm=True,
    )
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for tensor in attention.state_dict().values():
            tensor.copy_(torch.randn(tensor.shape, generator=generator) * 0.05)
    return attention


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
    num_repeats = attention.num_qo_heads // attention.num_kv_heads
    for offset, length, last_row in zip(offsets, lengths, last_rows):
        query_row = query[last_row]
        key_row = key[offset : offset + length].repeat_interleave(
            num_repeats, dim=1
        ).transpose(0, 1)
        value_row = value[offset : offset + length].repeat_interleave(
            num_repeats, dim=1
        ).transpose(0, 1)
        score = torch.matmul(query_row.unsqueeze(1), key_row.transpose(-1, -2))
        score = F.softmax(
            score * attention.head_dim**-0.5, dim=-1, dtype=torch.float32
        )
        output = torch.matmul(score.to(value_row.dtype), value_row).squeeze(1)
        output *= torch.sigmoid(gate[last_row])
        expected.append(attention.o_proj.forward(output.flatten()))

    torch.testing.assert_close(pooled, torch.stack(expected))
