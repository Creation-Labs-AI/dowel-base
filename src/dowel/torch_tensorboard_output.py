import functools
import operator
import warnings
from dataclasses import dataclass
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
from typing_extensions import Self

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing
import scipy.stats

from dowel.histogram import Histogram
from dowel.log_output import LogOutput
from dowel.tabular_input import TabularInput
from dowel.tensor_board_output import NonexistentAxesWarning
from dowel.utils import colorize

try:
    import open3d

    _has_open3d = True
    TriangleMeshType = open3d.geometry.TriangleMesh

    class Open3DInterface:
        @classmethod
        def calculate_zy_rotation_for_arrow(cls: Type[Self], direction: numpy.typing.NDArray[np.floating]) -> Tuple[numpy.typing.NDArray[np.floating], numpy.typing.NDArray[np.floating]]:
            # Based on https://stackoverflow.com/questions/59026581/create-arrows-in-open3d
            gamma: float = np.arctan(direction[1] / direction[0])
            Rz = np.array(
                [
                    [np.cos(gamma), -np.sin(gamma), 0.0],
                    [np.sin(gamma), np.cos(gamma), 0.0],
                    [0.0, 0.0, 1.0]
                ]
            )

            # Rotate direction to calculate next rotation
            direction = np.einsum("ij,i->j", Rz, direction)

            # Rotation over y axis of the FOR
            beta = np.arctan(direction[0] / direction[2])
            Ry = np.array(
                [
                    [np.cos(beta), 0.0, np.sin(beta)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(beta), 0.0, np.cos(beta)]
                ]
            )

            return Rz, Ry

        @classmethod
        def create_arrow(cls: Type[Self], scale: float) -> open3d.geometry.TriangleMesh:
            cone_height = scale * 0.0166
            cylinder_height = scale * 0.0333
            cone_radius = 0.25
            cylinder_radius = 0.15
            mesh_frame = open3d.geometry.TriangleMesh.create_arrow(
                cone_radius=cone_radius,
                cone_height=cone_height,
                cylinder_radius=cylinder_radius,
                cylinder_height=cylinder_height
            )

            return mesh_frame

        @classmethod
        def get_arrow(cls: Type[Self], position: numpy.typing.NDArray[np.floating], velocity: numpy.typing.NDArray[np.floating], colour: Optional[numpy.typing.NDArray[np.floating]] = None) -> open3d.geometry.TriangleMesh:
            scale = np.linalg.norm(velocity)
            direction = velocity / (scale if scale > 0.0 else 1.0)

            Rz, Ry = cls.calculate_zy_rotation_for_arrow(direction)
            mesh = cls.create_arrow(scale)

            mesh.rotate(Ry)
            mesh.rotate(Rz)

            mesh.translate(position)

            if colour is not None:
                mesh.paint_uniform_color(colour)

            return mesh
except:
    _has_open3d = False
    TriangleMeshType = type(None)

    class Open3DInterface:
        @classmethod
        def calculate_zy_rotation_for_arrow(cls: Type[Self], direction: numpy.typing.NDArray[np.floating]) -> Tuple[numpy.typing.NDArray[np.floating], numpy.typing.NDArray[np.floating]]:
            raise NotImplementedError

        @classmethod
        def create_arrow(cls: Type[Self], scale: float) -> open3d.geometry.TriangleMesh:
            raise NotImplementedError

        @classmethod
        def get_arrow(cls: Type[Self], position: numpy.typing.NDArray[np.floating], velocity: numpy.typing.NDArray[np.floating], colour: Optional[numpy.typing.NDArray[np.floating]] = None) -> open3d.geometry.TriangleMesh:
            raise NotImplementedError

@dataclass
class Trajectory:
    positions: numpy.typing.NDArray[np.floating]
    velocities: numpy.typing.NDArray[np.floating]
    colour: numpy.typing.NDArray[np.floating]

    def to_meshes(self):
        return [
            Open3DInterface.get_arrow(position, velocity, colour=self.colour)
            for position, velocity in zip(self.positions, self.velocities)
        ]

try:
    import torch
    import torch.nn
    import torch.utils.tensorboard

    if torch.__version__ < (2, 0):
        from tensorboard.compat.proto.summary_pb2 import Summary, HistogramProto
        from torch.utils.tensorboard._convert_np import make_np

        def make_histogram(values: np.ndarray, bins: Optional[Sequence[Number]], max_bins: Optional[int] = None):
            """Convert values into a histogram proto using logic from histogram.cc."""
            if values.size == 0:
                raise ValueError("The input has no element.")
            values = values.reshape(-1)
            counts, limits = np.histogram(values, bins=bins)
            num_bins = len(counts)
            if max_bins is not None and num_bins > max_bins:
                subsampling = num_bins // max_bins
                subsampling_remainder = num_bins % subsampling
                if subsampling_remainder != 0:
                    counts = np.pad(
                        counts,
                        pad_width=[[0, subsampling - subsampling_remainder]],
                        mode="constant",
                        constant_values=0,
                    )
                counts = counts.reshape(-1, subsampling).sum(axis=-1)
                new_limits = np.empty((counts.size + 1,), limits.dtype)
                new_limits[:-1] = limits[:-1:subsampling]
                new_limits[-1] = limits[-1]
                limits = new_limits

            # Find the first and the last bin defining the support of the histogram:

            cum_counts = np.cumsum(np.greater(counts, 0))
            start, end = np.searchsorted(cum_counts, [0, cum_counts[-1] - 1], side="right")
            start = int(start)
            end = int(end) + 1
            del cum_counts

            # TensorBoard only includes the right bin limits. To still have the leftmost limit
            # included, we include an empty bin left.
            # If start == 0, we need to add an empty one left, otherwise we can just include the bin left to the
            # first nonzero-count bin:
            counts = (
                counts[start - 1 : end] if start > 0 else np.concatenate([[0], counts[:end]])
            )
            limits = limits[start : end + 1]

            if counts.size == 0 or limits.size == 0:
                raise ValueError("The histogram is empty, please file a bug report.")

            sum_sq = values.dot(values)
            return HistogramProto(
                min=values.min(),
                max=values.max(),
                num=len(values),
                sum=values.sum(),
                sum_squares=sum_sq,
                bucket_limit=limits.tolist(),
                bucket=counts.tolist(),
            )

        def histogram(name: str, values: Union[torch.Tensor, np.ndarray], bins: Sequence[float], max_bins: Optional[int] = None):
            # pylint: disable=line-too-long
            """Outputs a `Summary` protocol buffer with a histogram.
            The generated
            [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
            has one summary value containing a histogram for `values`.
            This op reports an `InvalidArgument` error if any value is not finite.
            Args:
            name: A name for the generated node. Will also serve as a series name in
                TensorBoard.
            values: A real numeric `Tensor`. Any shape. Values to use to
                build the histogram.
            Returns:
            A scalar `Tensor` of type `string`. The serialized `Summary` protocol
            buffer.
            """
            values = make_np(values)
            hist = make_histogram(values.astype(float), bins, max_bins)
            return Summary(value=[Summary.Value(tag=name, histo=hist)])

        def add_histogram(
            self: torch.utils.tensorboard.SummaryWriter,
            tag: str,
            values: Union[torch.Tensor, np.ndarray, str],
            global_step: Optional[int] = None,
            bins: str = "tensorflow",
            walltime: Optional[float] = None,
            max_bins: Optional[int] = None
        ):
            """Add histogram to summary.

            Args:
                tag (str): Data identifier
                values (torch.Tensor, numpy.ndarray, or string/blobname): Values to build histogram
                global_step (int): Global step value to record
                bins (str): One of {'tensorflow','auto', 'fd', ...}. This determines how the bins are made. You can find
                other options in: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
                walltime (float): Optional override default walltime (time.time())
                seconds after epoch of event

            Examples::

                from torch.utils.tensorboard import SummaryWriter
                import numpy as np
                writer = SummaryWriter()
                for i in range(10):
                    x = np.random.random(1000)
                    writer.add_histogram('distribution centers', x + i, i)
                writer.close()

            Expected result:

            .. image:: _static/img/tensorboard/add_histogram.png
            :scale: 50 %

            """
            torch._C._log_api_usage_once("tensorboard.logging.add_histogram")
            if self._check_caffe2_blob(values):
                from caffe2.python import workspace

                values = workspace.FetchBlob(values)
            if isinstance(bins, str) and bins == "tensorflow":
                bins = self.default_bins
            self._get_file_writer().add_summary(
                histogram(tag, values, bins, max_bins=max_bins), global_step, walltime
            )
    else:
        add_histogram = torch.utils.tensorboard.SummaryWriter.add_histogram_raw

    @dataclass
    class Traceable:
        model: torch.nn.Module
        inputs: Optional[Sequence[torch.Tensor]] = None
        verbose: bool = False
        use_strict_trace: bool = True

    @dataclass
    class Mesh:
        vertices: Union[np.ndarray, torch.Tensor] # [B, N, 3]
        colours: Optional[Union[np.ndarray, torch.Tensor]] = None # [B, N, 3]
        faces: Optional[Union[np.ndarray, torch.Tensor]] = None # [B, N, 3]

        @classmethod
        def from_mesh_list(cls: Type[Self], meshes: Sequence[TriangleMeshType]):
            if not _has_open3d:
                raise NotImplementedError

            concatenated_mesh = functools.reduce(operator.add, meshes)
            return cls(
                vertices=np.asarray(concatenated_mesh.vertices)[np.newaxis, ...],
                colours=np.asarray(concatenated_mesh.vertex_colors)[np.newaxis, ...],
                faces=np.asarray(concatenated_mesh.triangles)[np.newaxis, ...]
            )

    ValueType = Union[
        Number, np.integer, np.floating,
        plt.Figure,
        scipy.stats.distributions.rv_frozen,
        scipy.stats._multivariate.multi_rv_frozen,
        torch.Tensor,
        Mesh,
        Histogram
    ]

    RecordType = Union[TabularInput[ValueType], Traceable]

    class TorchTensorBoardOutput(LogOutput[RecordType]):
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
                    max_queue: int = 10,
                    flush_secs: int = 120,
                    histogram_samples: int = 1000,
                    mesh_config_dict: Optional[Dict[str, Any]] = None,
                    video_fps: int = 20):
            if x_axis is None:
                assert not additional_x_axes, (
                    'You have to specify an x_axis if you want additional axes.')

            additional_x_axes = additional_x_axes or []

            self._writer = torch.utils.tensorboard.SummaryWriter(
                log_dir,
                max_queue=max_queue,
                flush_secs=flush_secs
            )
            self._x_axis = x_axis
            self._additional_x_axes = additional_x_axes
            self._default_step = 0
            self._histogram_samples = int(histogram_samples)
            self._mesh_config_dict = mesh_config_dict
            self._video_fps = video_fps

            self._waiting_for_dump: List[Callable[[int], None]] = []

            self._warned_once = set()
            self._disable_warnings = False

        @property
        def types_accepted(self):
            """Return the types that the logger may pass to this output."""
            return (TabularInput, Traceable)

        def record(self, data: RecordType, prefix: str = ""):
            """Add data to tensorboard summary.

            Args:
                data: The data to be logged by the output.
                prefix(str): A prefix placed before a log entry in text outputs.

            """
            if isinstance(data, TabularInput):
                self._waiting_for_dump.append(
                    functools.partial(self._record_tabular, data))
            elif isinstance(data, Traceable):
                self._writer.add_graph(
                    model=data.model,
                    input_to_model=data.inputs,
                    verbose=data.verbose,
                    use_strict_trace=data.use_strict_trace
                )
            else:
                raise ValueError('Unacceptable type.')

        def _record_tabular(self, data: TabularInput[ValueType], step: int):
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

        def _record_kv(self, key: str, value: ValueType, step: int):
            if isinstance(value, np.ScalarType):
                self._writer.add_scalar(key, value, step)
            elif isinstance(value, plt.Figure):
                self._writer.add_figure(key, value, step)
            elif isinstance(value, scipy.stats.distributions.rv_frozen):
                shape = (self._histogram_samples,) + value.mean().shape
                add_histogram(self._writer, key, value.rvs(shape), step)
            elif isinstance(value, scipy.stats._multivariate.multi_rv_frozen):
                add_histogram(
                    self._writer,
                    key,
                    value.rvs(self._histogram_samples),
                    step
                )
            elif isinstance(value, Histogram):
                add_histogram(self._writer, key, np.array(value), step)
            elif isinstance(value, torch.Tensor):
                if value.ndim == 4:
                    self._writer.add_image(key, value, step)
                elif value.ndim == 5:
                    self._writer.add_video(key, value, step, fps=self._video_fps)
            elif isinstance(value, Mesh):
                self._writer.add_mesh(
                    key,
                    value.vertices,
                    (value.colours * 255.0).astype(np.uint8) if value.colours is not None else None,
                    value.faces,
                    config_dict=self._mesh_config_dict,
                    global_step=step
                )

        def dump(self, step: Optional[int] = None):
            """Flush summary writer to disk."""
            # Log the tabular inputs, now that we have a step
            for dump_item_fn in self._waiting_for_dump:
                dump_item_fn(step or self._default_step)

            self._waiting_for_dump.clear()

            # Flush output files
            for writer in self._writer.all_writers.values():
                writer.flush()

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
except:
    TorchTensorBoardOutput = None