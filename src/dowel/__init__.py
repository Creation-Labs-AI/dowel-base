"""Logger module.

This module instantiates a global logger singleton.
"""
from dowel.histogram import Histogram
from dowel.logger import Logger, LoggerWarning, LogOutput
from dowel.simple_outputs import StdOutput, TextOutput
from dowel.tabular_input import TabularInput
from dowel.csv_output import CsvOutput  # noqa: I100
from dowel.tensor_board_output import TensorBoardOutput
import fsspec

logger = Logger()
tabular = TabularInput()
_FILESYSTEM = fsspec.filesystem(protocol='file')

def set_filesystem(fs: fsspec.AbstractFileSystem):
    """Set the filesystem to use for logging.

    :param fs: The filesystem to use for logging.
    """
    global _FILESYSTEM
    _FILESYSTEM = fs

def get_filesystem() -> fsspec.AbstractFileSystem:
    """Get the filesystem used for logging.

    :return: The filesystem used for logging.
    """
    return _FILESYSTEM

__all__ = [
    'Histogram',
    'Logger',
    'CsvOutput',
    'StdOutput',
    'TextOutput',
    'LogOutput',
    'LoggerWarning',
    'TabularInput',
    'TensorBoardOutput',
    'logger',
    'tabular',
]
