"""A `dowel.logger.LogOutput` for tensorboard.

It receives the input data stream from `dowel.logger`, then add them to
tensorboard summary operations through tensorboardX.

Note:
    Neither TensorboardX nor TensorBoard supports log parametric
    distributions. We add this feature by sampling data from a
    `tfp.distributions.Distribution` object.

"""
import functools
import warnings
from numbers import Number
from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import tensorboardX as tbX

from dowel.histogram import Histogram
from dowel.logger_warning import LoggerWarning
from dowel.log_output import LogOutput
from dowel.tabular_input import TabularInput
from dowel.utils import colorize

try:
    import tensorflow as tf
    RecordType = Union[tf.Graph, TabularInput]
    GraphType = tf.Graph
except ImportError:
    tf = None
    RecordType = TabularInput
    GraphType = None

ValueType = Union[Number, np.integer, np.floating, plt.Figure, scipy.stats.distributions.rv_frozen, scipy.stats._multivariate.multi_rv_frozen, Histogram]

class TensorBoardOutput(LogOutput[RecordType]):
    """TensorBoard output for logger.

    Args:
        log_dir(str): The save location of the tensorboard event files.
        x_axis(str): The name of data used as x-axis for scalar tabular.
            If None, x-axis will be the number of dump() is called.
        additional_x_axes(list[str]): Names of data to used be as additional
            x-axes.
        flush_secs(int): How often, in seconds, to flush the added summaries
            and events to disk.
        histogram_samples(int): Number of samples to generate when logging
            random distribution.

    """

    def __init__(self,
                 log_dir: str,
                 x_axis: Optional[str] = None,
                 additional_x_axes: Optional[List[str]] = None,
                 flush_secs: int = 120,
                 histogram_samples: int = 1e3):
        if x_axis is None:
            assert not additional_x_axes, (
                'You have to specify an x_axis if you want additional axes.')

        additional_x_axes = additional_x_axes or []

        self._writer = tbX.SummaryWriter(log_dir, flush_secs=flush_secs)
        self._x_axis = x_axis
        self._additional_x_axes = additional_x_axes
        self._default_step = 0
        self._histogram_samples = int(histogram_samples)
        self._added_graph = False
        self._waiting_for_dump: List[Callable[[int], None]] = []
        # Used in tests to emulate Tensorflow not being installed.
        self._tf = tf

        self._warned_once = set()
        self._disable_warnings = False

    @property
    def types_accepted(self):
        """Return the types that the logger may pass to this output."""
        always_accepted = (TabularInput,)
        if self._tf is None:
            return always_accepted
        else:
            return always_accepted + (self._tf.Graph,)

    def record(self, data: RecordType, prefix: str = ""):
        """Add data to tensorboard summary.

        Args:
            data: The data to be logged by the output.
            prefix(str): A prefix placed before a log entry in text outputs.

        """
        if isinstance(data, TabularInput):
            self._waiting_for_dump.append(
                functools.partial(self._record_tabular, data))
        elif self._tf is not None and isinstance(data, self._tf.Graph):
            self._record_graph(data)
        else:
            raise ValueError('Unacceptable type.')

    def _record_tabular(self, data: TabularInput, step: int):
        if self._x_axis:
            nonexist_axes = []
            for axis in [self._x_axis] + self._additional_x_axes:
                if axis not in data.as_dict:
                    nonexist_axes.append(axis)
            if nonexist_axes:
                self._warn('{} {} exist in the tabular data.'.format(
                    ', '.join(nonexist_axes),
                    'do not' if len(nonexist_axes) > 1 else 'does not'))

        for key, value in data.as_dict.items():
            if isinstance(value,
                          np.ScalarType) and self._x_axis in data.as_dict:
                if self._x_axis != key:
                    x = data.as_dict[self._x_axis]
                    self._record_kv(key, value, x)

                for axis in self._additional_x_axes:
                    if key != axis and key in data.as_dict:
                        x = data.as_dict[axis]
                        self._record_kv('{}/{}'.format(key, axis), value, x)
            else:
                self._record_kv(key, value, step)
            data.mark(key)

    def _record_kv(self, key: str, value: ValueType, step):
        if isinstance(value, np.ScalarType):
            self._writer.add_scalar(key, value, step)
        elif isinstance(value, plt.Figure):
            self._writer.add_figure(key, value, step)
        elif isinstance(value, scipy.stats.distributions.rv_frozen):
            shape = (self._histogram_samples, ) + value.mean().shape
            self._writer.add_histogram(key, value.rvs(shape), step)
        elif isinstance(value, scipy.stats._multivariate.multi_rv_frozen):
            self._writer.add_histogram(key, value.rvs(self._histogram_samples),
                                       step)
        elif isinstance(value, Histogram):
            self._writer.add_histogram(key, value, step)
            self._writer.add_video()

    def _record_graph(self, graph: GraphType):
        graph_def = graph.as_graph_def(add_shapes=True)
        event = tbX.proto.event_pb2.Event(
            graph_def=graph_def.SerializeToString())
        self._writer.file_writer.add_event(event)

    def dump(self, step: Optional[int] = None):
        """Flush summary writer to disk."""
        # Log the tabular inputs, now that we have a step
        for p in self._waiting_for_dump:
            p(step or self._default_step)
        self._waiting_for_dump.clear()

        # Flush output files
        for w in self._writer.all_writers.values():
            w.flush()

        self._default_step += 1

    def close(self):
        """Flush all the events to disk and close the file."""
        self._writer.close()

    def _warn(self, msg):
        """Warns the user using warnings.warn.

        The stacklevel parameter needs to be 3 to ensure the call to logger.log
        is the one printed.
        """
        if not self._disable_warnings and msg not in self._warned_once:
            warnings.warn(colorize(msg, 'yellow'),
                          NonexistentAxesWarning,
                          stacklevel=3)
        self._warned_once.add(msg)
        return msg


class NonexistentAxesWarning(LoggerWarning):
    """Raise when the specified x axes do not exist in the tabular."""
