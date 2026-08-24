from __future__ import annotations

from ..utils import deserialize_type, serialize_type


class BaseDrafterMsg:
    """Control-plane message between the target (rank0) and the speculator process.

    Payloads are scalars / int lists / SamplingParams only — the heavy tensors
    (carry_hidden, appended hidden, draft_probs) cross the selected data plane, keyed by the
    counts carried here. Same encode/decode scheme as BaseBackendMsg; each message module
    uses its own ``globals()`` so ``deserialize_type`` can resolve nested dataclasses.
    """

    def encoder(self) -> dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: dict) -> BaseDrafterMsg:
        return deserialize_type(globals(), json)
