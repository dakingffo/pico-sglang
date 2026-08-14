from __future__ import annotations

import heapq
import time
from itertools import count
from dataclasses import dataclass
from typing import Any, Callable, TypeAlias

import torch

from picosgl.core import get_global_ctx
from picosgl.utils import align_down

from .base import (
    BaseCacheHandle, 
    BasePrefixCache, 
    InsertResult, 
    MatchResult, 
    SizeInfo
)


KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    else:
        return lambda x: tuple(x[:page_size].tolist())

    
class RadixTreeNode:
    counter = count(0)

    def __init__(self, key_fn: KEY_FN, page_size: int = 1, tic: int | None = None) -> None:
        self.uuid = next(RadixTreeNode.counter)
        self.key_fn = key_fn
        self.page_size = page_size
        self.children: dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.timestamp = tic or time.monotonic_ns()

        # these fields should be updated later
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int
        # one linear-state pool slot per cached page (only for hybrid_radix)
        self._state_value: torch.Tensor | None = None

    def set_key_value(
        self,
        key        : torch.Tensor,
        value      : torch.Tensor,
        state_value: torch.Tensor | None = None,
    ) -> None:
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)
        if state_value is not None:
            assert len(state_value) == len(value) // self.page_size, (
                f"state_value ({len(state_value)}) must be one entry per page "
                f"({len(value)} // {self.page_size})"
            )
            self._state_value = state_value

    def set_parent(self, parent: RadixTreeNode) -> None:
        assert isinstance(parent, RadixTreeNode)
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from picosgl.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        new_node = RadixTreeNode(self.key_fn, self.page_size, self.timestamp)
        if self._state_value is not None:
            split = pos // self.page_size
            new_node.set_key_value(
                self._key[:pos], self._value[:pos], self._state_value[:split]
            )
        else:
            new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        if self._state_value is not None:
            split = pos // self.page_size
            self.set_key_value(
                self._key[pos:], self._value[pos:], self._state_value[split:]
            )
        else:
            self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp
    

@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode

    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        value_list: list[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        if not value_list:
            return torch.empty(0, dtype=torch.int32, device=self.node._key.device)
        return torch.cat(value_list)

    def get_matched_state_slots(self) -> torch.Tensor | None:
        """Pool slot ids of the matched pages, in page order. None if the cache holds
        no state info (non-hybrid radix) or nothing was matched."""
        node = self.node
        state_list: list[torch.Tensor] = []
        while not node.is_root():
            if node._state_value is None:
                return None
            state_list.append(node._state_value)
            node = node.parent
        state_list.reverse()
        if not state_list:
            return None
        return torch.cat(state_list)


class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.page_size = get_global_ctx().page_size
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn, self.page_size)
        self.root_node.ref_count = 1  # root is always protected

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        node, prefix_len = self._tree_walk(input_ids)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def insert_prefix(
        self,
        input_ids   : torch.Tensor,
        indices     : torch.Tensor,
        state_slots : torch.Tensor | None = None,
    ) -> InsertResult:
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            new_node = RadixTreeNode(self.key_fn, self.page_size)
            if state_slots is not None:
                new_node.set_key_value(
                    input_ids[prefix_len:], indices[prefix_len:].clone(),
                    state_slots[prefix_len // self.page_size : insert_len // self.page_size].clone(),
                )
            else:
                new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Evict prefixes. Returns (evicted KV page indices, evicted state slots).

        For the non-hybrid radix cache the state slots are None.
        """
        if size == 0:
            return self.empty_tensor, None
        assert size <= self.evictable_size, f"Can't evict {size}, only {self.evictable_size}"

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: list[torch.Tensor] = []
        evicted_state  : list[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert leave_nodes, f"Can't evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            if node._state_value is not None:
                evicted_state.append(node._state_value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices), (
            torch.cat(evicted_state) if evicted_state else None
        )

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        pass

    def total_state_pages(self) -> int:
        """Total number of linear-state slots owned by the tree (one per cached page)."""
        total = 0
        stack: list[RadixTreeNode] = [self.root_node]
        while stack:
            node = stack.pop()
            if node._state_value is not None:
                total += len(node._state_value)
            stack.extend(node.children.values())
        return total

    def _collect_leave_nodes_for_evict(self) -> list[RadixTreeNode]:
        nodes: list[RadixTreeNode] = [self.root_node]
        leave_nodes: list[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def _tree_walk(self, input_ids: torch.Tensor) -> tuple[RadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                break
            node = child_node  # walk to child node

            # NOTE: at least 1 page is matched, so match_len >= page_size
            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                break

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len


class HybridRadixPrefixCache(RadixPrefixCache):
    def insert_prefix(
        self,
        input_ids   : torch.Tensor,
        indices     : torch.Tensor,
        state_slots : torch.Tensor | None = None,
    ) -> InsertResult:
        insert_len = align_down(len(input_ids), self.page_size)
        if insert_len > 0:
            assert state_slots is not None, "hybrid_radix requires per-page state slots"
        return super().insert_prefix(input_ids, indices, state_slots)
