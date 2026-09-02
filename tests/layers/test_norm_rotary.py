import torch

from picosgl.layers import RMSNorm
from picosgl.layers.rotary import RotaryEmbedding


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def test_zero_centered_rmsnorm() -> None:
    norm = RMSNorm(8, eps=1e-6, zero_centered=True)
    norm.weight.copy_(torch.linspace(-0.2, 0.2, 8))
    x = torch.randn(3, 8, generator=torch.Generator().manual_seed(7))

    expected = x.float()
    expected = expected * torch.rsqrt(expected.pow(2).mean(-1, keepdim=True) + norm.eps)
    expected = (expected * (1.0 + norm.weight.float())).type_as(x)
    torch.testing.assert_close(norm.forward(x), expected)


def test_partial_rotary_embedding() -> None:
    rope = RotaryEmbedding(
        head_size=8,
        rotary_dim=4,
        max_position_embeddings=32,
        base=10_000.0,
    )
    positions = torch.tensor([[0, 3, 7], [2, 5, 11]])
    generator = torch.Generator().manual_seed(11)
    query = torch.randn(2, 3, 4, 8, generator=generator)
    key = torch.randn(2, 3, 2, 8, generator=generator)
    expected_query = query.clone()
    expected_key = key.clone()

    cache = rope._cos_sin_cache
    flat_positions = positions.reshape(-1)
    cos = torch.cat((cache[flat_positions, :2], cache[flat_positions, :2]), dim=-1)
    sin = torch.cat((cache[flat_positions, 2:], cache[flat_positions, 2:]), dim=-1)
    for tensor in (expected_query, expected_key):
        flat = tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])
        x = flat[..., :4]
        flat[..., :4] = x * cos.unsqueeze(1) + _rotate_half(x) * sin.unsqueeze(1)

    actual_query, actual_key = rope.forward(positions, query, key)
    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_key, expected_key)
    assert torch.equal(actual_query[..., 4:], expected_query[..., 4:])
    assert torch.equal(actual_key[..., 4:], expected_key[..., 4:])
