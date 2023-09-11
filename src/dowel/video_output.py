from dataclasses import dataclass
from typing import Optional, Union

import av
import av.container
import av.video
import numpy as np
import numpy.typing

from dowel.log_output import LogOutput
from dowel.simple_outputs import FileOutput

class VideoOutput(FileOutput[numpy.typing.NDArray[np.integer]]):
    def __init__(
            self,
            file_name: str,
            format: Optional[str] = None,
            options: Optional[dict] = None,
            container_options: Optional[dict] = None,
            buffer_size: int = 32768,
            stream_format: str = "hevc_nvenc",
            pixel_format: Optional[str] = None,
            frame_rate: int = 20,
            **stream_options):
        super().__init__(file_name, mode="wb")

        self.container: av.container.Container = av.open(
            self._log_file,
            mode="w",
            format=format,
            options=options,
            container_options=container_options,
            buffer_size=buffer_size
        )
        self.video_output_stream: av.video.VideoStream = self.container.add_stream(stream_format, rate=frame_rate, options=stream_options)

        if pixel_format is None:
            self.codec = av.Codec(stream_format, mode="w")
            self.default_format: str = self.codec.video_formats[0].name
            self.pixel_format = self.default_format

        self.video_output_stream.pix_fmt = self.pixel_format

        self.frame_width, self.frame_height = (None, None)

    @property
    def types_accepted(self):
        """Accept TabularInput objects only."""
        return (np.ndarray,)

    def record(self, data: numpy.typing.NDArray[np.integer], prefix: str = ""):
        if not isinstance(data, self.types_accepted):
            raise TypeError(f"Unacceptable type: {type(data)}")

        if self.frame_width is None and self.frame_height is None:
            self.frame_height, self.frame_width = data.shape[0:2]
            self.video_output_stream.width = self.frame_width
            self.video_output_stream.height = self.frame_height

        av_frame: av.VideoFrame = av.VideoFrame.from_ndarray(data, format="rgb24")
        for packet in self.video_output_stream.encode(av_frame):
            self.container.mux(packet)

    def close(self):
        for packet in self.video_output_stream.encode():
            self.container.mux(packet)

        self.container.close()

        super().close()

@dataclass
class RolloverToken:
    label: Optional[str] = None

class VideoOutputs(LogOutput[Union[RolloverToken, numpy.typing.NDArray[np.integer]]]):
    def __init__(
            self,
            root_dir: str,
            extension: Optional[str] = "mp4",
            format: Optional[str] = None,
            options: Optional[dict] = None,
            container_options: Optional[dict] = None,
            buffer_size: int = 32768,
            stream_format: str = "hevc_nvenc",
            pixel_format: Optional[str] = None,
            frame_rate: int = 20,
            **stream_options):
        super().__init__()

        self.root_dir = root_dir
        self.extension = extension
        if not self._fs.exists(self.root_dir):
            self._fs.makedirs(self.root_dir, exist_ok=True)

        self.index = 0
        self.next_label: Optional[str] = None

        self.extra_args = {
            "format": format,
            "options": options,
            "container_options": container_options,
            "buffer_size": buffer_size,
            "stream_format": stream_format,
            "pixel_format": pixel_format,
            "frame_rate": frame_rate
        }
        self.stream_options = stream_options

        self.video_output: Optional[VideoOutput] = None

    @property
    def types_accepted(self):
        return (np.ndarray, RolloverToken)

    @property
    def next_tag(self):
        if self.next_label is not None:
            return self.next_label
        else:
            return str(self.index)

    def rollover(self, rollover_token: RolloverToken):
        self.close()

        if rollover_token.label is not None:
            self.next_label = rollover_token.label

        self.index += 1

    def open_video(self):
        self.video_output = VideoOutput(
            f"{self.root_dir}/{self.next_tag}.{self.extension}",
            **self.extra_args,
            **self.stream_options
        )

    def record(self, data: Union[RolloverToken, numpy.typing.NDArray[np.integer]], prefix: str = ""):
        if isinstance(data, RolloverToken):
            self.rollover(data)
        else:
            if self.video_output is None:
                self.open_video()

            self.video_output.record(data, prefix)

    def close(self):
        if self.video_output is not None:
            self.video_output.close()
            self.video_output = None