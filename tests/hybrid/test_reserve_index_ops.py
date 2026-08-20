"""Verify reserve index ops: CacheManager.state_commit_verify (pure index, zero memcpy).

The MTP verify round writes each candidate's post-state into the K+1 reserve columns
R[0..K]. Committing ``num_sampled`` accepted tokens only SHUFFLES slot ids:

  * no page crossing: swap R[0] <-> R[num_sampled-1] so the accepted state becomes the
    next round's baseline (R[0]).
  * page crossing (K+1 <= 5 < page_size, so at most one page): the boundary candidate's
    state is PINNED into state_table[uid, page], the page slot is freed/refilled, and the
    accepted state moves to R[0].
  * boundary token is the LAST accepted: the pinned page slot IS the next baseline; the
    reserve is left untouched and the baseline is the page slot (early return).

No state data is ever copied -- we assert pool contents are byte-identical before/after.

Run: /home/daking/.conda/envs/daking/bin/python tests/hybrid/test_reserve_index_ops.py
"""
import os
import sys

_REPO = "/home/daking/PROJECT/pico-sglang"
sys.path.insert(0, os.path.join(_REPO, "python"))
sys.path.insert(0, _REPO)

import torch

import picosgl.core as core
from picosgl.core import Context, Request, SamplingParams, set_global_ctx
from picosgl.scheduler.cache import CacheManager
from picosgl.cache.linear.state_pool import LinearStatePool


def setup_ctx() -> None:
    """Fresh 64-token context per test (pytest imports all modules up front, so the
    module level must stay side-effect free)."""
    core._GLOBAL_CTX = None
    set_global_ctx(Context(page_size=64))


def make_req(table_idx, cached_len):
    return Request(
        input_ids=torch.arange(200, dtype=torch.int32),
        table_idx=table_idx, cached_len=cached_len, output_len=10, uid=table_idx,
        sampling_params=SamplingParams(), cache_handle=None,
    )


def make_cm(st, nslots):
    pool = LinearStatePool(
        num_slots=nslots, num_linear_layers=1, conv_dim=8, kernel_size=4,
        num_v_heads=4, head_k_dim=4, head_v_dim=4,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    cm = CacheManager(
        num_pages=10, page_size=64,
        page_table=torch.zeros((8, 512), dtype=torch.int32),
        type="hybrid_radix", num_states=nslots,
        state_table=st, state_pool=pool, draft_offset=8,
    )
    return cm, pool


def setup_page(uid, page_slot, reserve):
    """state_table row: page 1 holds `page_slot`; R[0..K] hold `reserve`."""
    st = torch.full((8, 12), -1, dtype=torch.int32)
    st[uid, 1] = page_slot
    st[uid, 8: 8 + len(reserve)] = torch.tensor(reserve, dtype=torch.int32)
    return st


def mark_in_use(cm, slot_ids):
    """Take `slot_ids` out of the manager's free list (they are pre-populated)."""
    ids = torch.tensor(slot_ids, dtype=torch.int32)
    cm.free_states = cm.free_states[~torch.isin(cm.free_states, ids)]


# -------------------------------------------------------------------------------------
# 4a. no page crossing: C=120, num_sampled=2 (positions 120,121). B=128 > C_end=122.
#     R[0] and R[num_sampled-1] swap; the page slot is untouched.
# -------------------------------------------------------------------------------------
def test_no_cross_swap():
    setup_ctx()
    st = setup_page(2, 40, [50, 51, 52, 53])
    cm, pool = make_cm(st, nslots=200)
    mark_in_use(cm, [40, 50, 51, 52, 53])
    r = make_req(2, 120)
    snap = (pool.conv_state.clone(), pool.recurrent_state.clone())
    cm.state_commit_verify(r, 120, 2)
    assert r.baseline_slot == 51, (r.baseline_slot, st[2].tolist())  # accepted = R[num_sampled-1]
    assert st[2, 8] == 51 and st[2, 9] == 50, st[2].tolist()         # R[0] <-> R[1]
    assert st[2, 1] == 40, st[2].tolist()                            # page untouched
    assert len(cm.free_states) == 200 - 5                            # no free/refill in no-cross
    assert torch.equal(pool.conv_state, snap[0]) and torch.equal(pool.recurrent_state, snap[1])
    print("[4a] commit no-cross swap OK")


# -------------------------------------------------------------------------------------
# 4b. page crossing, boundary candidate NOT the last: C=125, num_sampled=4 (125..128).
#     B=128, j_b = 127-125 = 2 < num_sampled-1 = 3. Pin R[2] -> page, refill R[2],
#     accepted R[3] -> R[0].
# -------------------------------------------------------------------------------------
def test_cross_pin_refill_swap():
    setup_ctx()
    st = setup_page(2, 40, [50, 51, 52, 53])
    cm, pool = make_cm(st, nslots=300)
    mark_in_use(cm, [40, 50, 51, 52, 53])
    r = make_req(2, 125)
    snap = (pool.conv_state.clone(), pool.recurrent_state.clone())
    cm.state_commit_verify(r, 125, 4)
    assert st[2, 1] == 52, st[2].tolist()      # page pinned = R[j_b=2]
    assert st[2, 8] == 53, st[2].tolist()      # accepted (R[3]) -> next baseline R[0]
    assert st[2, 9] == 51, st[2].tolist()      # R[1] untouched
    assert st[2, 10] >= 0 and st[2, 10] != 52, st[2].tolist()  # R[2] refilled fresh
    assert st[2, 11] == 50, st[2].tolist()     # old baseline R[0] swapped out to R[3]
    assert r.baseline_slot == 53, r.baseline_slot
    # page slot 40 freed + one refill allocated -> net zero slots consumed
    assert len(cm.free_states) == 300 - 5, len(cm.free_states)
    assert torch.equal(pool.conv_state, snap[0]) and torch.equal(pool.recurrent_state, snap[1])
    print("[4b] commit crossing pin/refill/swap OK")


# -------------------------------------------------------------------------------------
# 4c. boundary token is the LAST accepted: C=126, num_sampled=2 (126,127).
#     B=128 = C_end => j_b = 1 = num_sampled-1. Pin R[1] -> page, refill R[1],
#     baseline = page slot, early return (no swap).
# -------------------------------------------------------------------------------------
def test_cross_boundary_last():
    setup_ctx()
    st = setup_page(2, 40, [50, 51, 52, 53])
    cm, pool = make_cm(st, nslots=300)
    mark_in_use(cm, [40, 50, 51, 52, 53])
    r = make_req(2, 126)
    snap = (pool.conv_state.clone(), pool.recurrent_state.clone())
    cm.state_commit_verify(r, 126, 2)
    assert st[2, 1] == 51, st[2].tolist()             # pinned = R[j_b=1] (state after 127)
    assert r.baseline_slot == 51, r.baseline_slot     # baseline = page slot (early return)
    assert st[2, 8] == 50, st[2].tolist()             # R[0] untouched (no swap)
    assert st[2, 9] >= 0 and st[2, 9] != 51, st[2].tolist()  # R[1] refilled fresh
    assert st[2, 10] == 52 and st[2, 11] == 53, st[2].tolist()  # R[2], R[3] untouched
    assert len(cm.free_states) == 300 - 5             # page slot freed + one refill -> net zero
    assert torch.equal(pool.conv_state, snap[0]) and torch.equal(pool.recurrent_state, snap[1])
    print("[4c] commit boundary-last pin+baseline OK")


if __name__ == "__main__":
    test_no_cross_swap()
    test_cross_pin_refill_swap()
    test_cross_boundary_last()
    print("\nALL RESERVE INDEX OP TESTS PASSED")
