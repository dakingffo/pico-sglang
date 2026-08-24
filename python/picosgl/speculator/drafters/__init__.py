from picosgl.utils import Registry

from ..args import SpeculatorArgumentParserBase
from .mtp.args import MTPArgumentParser as _MTPArgumentParser


SUPPORTED_SPECULATOR_ARGUMENT_PARSERS = Registry[
    type[SpeculatorArgumentParserBase]
]("Speculator Argument Parser")
SUPPORTED_SPECULATOR_ARGUMENT_PARSERS.register("MTP")(_MTPArgumentParser)


def make_speculator_argument_parser(
    algorithm: str,
) -> SpeculatorArgumentParserBase:
    return SUPPORTED_SPECULATOR_ARGUMENT_PARSERS[algorithm]()


__all__ = [
    "SUPPORTED_SPECULATOR_ARGUMENT_PARSERS",
    "make_speculator_argument_parser",
]
