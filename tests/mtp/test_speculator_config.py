import torch

from picosgl.distributed import DistributedInfo
from picosgl.scheduler import SchedulerConfig
from picosgl.speculator import (
    DataPlaneSizes,
    MTPHiddenFeature,
    MTPSpeculatorConfig,
    SpeculatorReserve,
    make_data_plane_sizes,
)


def test_mtp_speculator_config() -> None:
    speculator_config = MTPSpeculatorConfig(
        num_draft_tokens=3,
        window_size=96,
    )
    config = SchedulerConfig(
        model_path="unused",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=8,
        speculative_algorithm="MTP",
        speculative_draft_model_path="unused",
        speculator_config=speculator_config,
    )

    assert config.decode_batch_budget == 12
    assert speculator_config.make_reserve(config.max_running_req) == SpeculatorReserve(
        num_state_slots=32,
        state_slots_per_request=4,
    )
    assert make_data_plane_sizes(config, 1024, 100, 96) == DataPlaneSizes(
        max_hidden_rows=96,
        hidden_size=1024,
        max_prob_rows=12,
        vocab_size=100,
    )

    hidden = torch.arange(12).view(4, 3)
    hidden_feature = speculator_config.make_hidden_feature(hidden)
    assert isinstance(hidden_feature, MTPHiddenFeature)
    selected = hidden_feature.select(slice(1, 3))
    assert torch.equal(selected.full_hidden, hidden[1:3])
