from typing import Optional

import av
import av.container
import av.video
import numpy as np
import numpy.typing

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