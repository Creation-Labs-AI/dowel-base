"""Logger module.

This module instantiates a global logger singleton.
"""
from dowel.histogram import Histogram
from dowel.logger import Logger
from dowel.log_output import LogOutput
from dowel.logger_warning import LoggerWarning
from dowel.simple_outputs import StdOutput, TextOutput
from dowel.tabular_input import TabularInput
from dowel.csv_output import CsvOutput  # noqa: I100
from dowel.tensor_board_output import TensorBoardOutput, ValueType
from dowel.torch_tensorboard_output import TorchTensorBoardOutput
from dowel.torch_tensorboard_output import ValueType as TorchValueType
from dowel.video_output import RolloverToken, VideoOutput, VideoOutputs

logger = Logger()
tabular = TabularInput[ValueType]()

__all__ = [
    'Histogram',
    'Logger',
    'CsvOutput',
    'StdOutput',
    'TextOutput',
    'LogOutput',
    'LoggerWarning',
    'RolloverToken',
    'TabularInput',
    'TensorBoardOutput',
    'TorchTensorBoardOutput',
    'TorchValueType',
    'logger',
    'tabular',
    'ValueType',
    'VideoOutput',
    'VideoOutputs'
]
