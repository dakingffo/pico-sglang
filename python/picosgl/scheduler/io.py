from __future__ import annotations

from typing import TYPE_CHECKING, Final

import msgpack
import numpy as np
import torch

from picosgl.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg, DetokenizeMsg, DraftReplyMsg
from picosgl.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)


class SchedulerIOMixin:
    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        tp_info = config.tp_info
        self.tp_cpu_group: Final = tp_cpu_group
        if config.offline_mode:
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return  # early exit

        if tp_info.is_primary():
            self._recv_from_tokenizer: Final = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,
                decoder=BaseBackendMsg.decoder,
            )
            self._send_into_tokenizer: Final = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=True,
                encoder=BaseTokenizerMsg.encoder,
            )

        if tp_info.size > 1:
            if tp_info.is_primary():
                recv = self._recv_msg_multi_rank0
                send = self._reply_tokenizer_rank0
                self._send_into_ranks: Final = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, 
                    create=True, 
                    encoder=BaseBackendMsg.encoder
                )
            else:
                recv = self._recv_msg_multi_rank1
                send = self._reply_tokenizer_rank1
                self._recv_from_rank0: Final = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr,
                    create=False,
                    decoder=BaseBackendMsg.decoder,
                )
        else:
            recv = self._recv_msg_single_rank
            send = self._reply_tokenizer_rank0

        self.receive_msg = recv
        self.send_result = send

    def run_when_idle(self):
        raise NotImplementedError("should be implemented")

    def offline_receive_msg(self, blocking: bool = False) -> list[BaseBackendMsg]:
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: list[DetokenizeMsg]) -> None:
        raise NotImplementedError("should be implemented")

    def sync_all_ranks(self) -> None:
        self.tp_cpu_group.barrier().wait()

    def _recv_msg_single_rank(self, blocking: bool = False) -> list[BaseBackendMsg]:
        pending_msgs: list[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_tokenizer.get())
        while not self._recv_from_tokenizer.empty():
            pending_msgs.append(self._recv_from_tokenizer.get())
        return pending_msgs

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> list[BaseBackendMsg]:
        pending_msgs: list[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            raw = self._recv_from_tokenizer.get_raw()
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))

        pending_raw_msgs: list[bytes] = []
        while not self._recv_from_tokenizer.empty():
            pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # broadcast the number of raw messages to all ranks
        src_tensor = torch.tensor(len(pending_raw_msgs))
        self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

        for raw in pending_raw_msgs:
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))
        return pending_msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> list[BaseBackendMsg]:
        pending_msgs: list[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_rank0.get())

        # ensure all ranks have the same number of raw messages
        dst_tensor = torch.tensor(-1)
        self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()
        dst_length = int(dst_tensor.item())

        for _ in range(dst_length):
            pending_msgs.append(self._recv_from_rank0.get())
        return pending_msgs

    def _reply_tokenizer_rank0(self, reply: list[DetokenizeMsg]) -> None:
        num_reply = len(reply)
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        if num_reply == 1:
            self._send_into_tokenizer.put(reply[0])
        elif num_reply > 1:
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))  # type: ignore

    def _reply_tokenizer_rank1(self, reply: list[DetokenizeMsg]) -> None:
        pass

    def _send_draft_to_ranks(
        self,
        reply: DraftReplyMsg,
        probs: torch.Tensor | None,
    ) -> None:
        """Rank0 broadcasts a step's draft results to the other TP ranks.

        Only rank0 talks to the drafter (zmq + NCCL); every rank's VerifyManager schedules
        the same requests, so rank>0 needs the identical DraftReplyMsg + draft_probs to
        backfill its own DraftState. Mirror of ``_recv_msg_multi_rank0/1``: rank0 PUBs the
        raw messages and gloo-broadcasts the count; the probs tensor crosses as raw fp32
        bytes (the drafter side already did the GPU→CPU trip).
        """
        n = 2 if probs is not None else 1
        src = torch.tensor(n)
        self.tp_cpu_group.broadcast(src, root=0).wait()
        self._send_into_ranks.put_raw(msgpack.packb(reply.encoder(), use_bin_type=True))
        if probs is not None:
            self._send_into_ranks.put_raw(probs.cpu().contiguous().numpy().tobytes())

    def _recv_draft_from_rank0(
        self,
        vocab_size: int,
    ) -> tuple[DraftReplyMsg, torch.Tensor | None]:
        """Non-primary-rank receive side of ``_send_draft_to_ranks``."""
        dst = torch.tensor(-1)
        self.tp_cpu_group.broadcast(dst, root=0).wait()
        n = int(dst.item())
        assert n in (1, 2), f"bad draft broadcast count {n}"
        reply = DraftReplyMsg.decoder(
            msgpack.unpackb(self._recv_from_rank0.get_raw(), raw=False)
        )
        probs = None
        if n == 2:
            data = np.frombuffer(self._recv_from_rank0.get_raw(), dtype=np.float32)
            probs = torch.from_numpy(data).reshape(-1, vocab_size).to(self.device)
        return reply, probs
