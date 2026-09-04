from picosgl.utils import Registry

from ..args import SpeculatorArgumentParserBase
from ..base import BaseSpeculatorConfig
from ..hidden_captor import HiddenCaptorBase
from .mtp.args import MTPArgumentParser as _MTPArgumentParser
from .mtp.hidden_captor import MTPHiddenCaptor as _MTPHiddenCaptor


SUPPORTED_SPECULATOR_ARGUMENT_PARSERS = Registry[
    type[SpeculatorArgumentParserBase]
]("Speculator Argument Parser")
SUPPORTED_SPECULATOR_ARGUMENT_PARSERS.register("MTP")(_MTPArgumentParser)
SUPPORTED_HIDDEN_CAPTORS = Registry[type[HiddenCaptorBase]]("Hidden Captor")
SUPPORTED_HIDDEN_CAPTORS.register("MTP")(_MTPHiddenCaptor)


def make_speculator_argument_parser(
    algorithm: str,
) -> SpeculatorArgumentParserBase:
    return SUPPORTED_SPECULATOR_ARGUMENT_PARSERS[algorithm]()


def make_hidden_captor(
    algorithm: str,
    config   : BaseSpeculatorConfig,
) -> HiddenCaptorBase:
    return SUPPORTED_HIDDEN_CAPTORS[algorithm](config)


__all__ = [
    "SUPPORTED_SPECULATOR_ARGUMENT_PARSERS",
    "SUPPORTED_HIDDEN_CAPTORS",
    "make_hidden_captor",
    "make_speculator_argument_parser",
]
