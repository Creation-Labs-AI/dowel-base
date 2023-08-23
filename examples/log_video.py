import numpy as np
import dowel
from dowel import logger, tabular

logger.add_output(dowel.StdOutput())
logger.add_output(dowel.CsvOutput("dowel_video_progress.csv"))
logger.add_output(dowel.VideoOutput("dowel_video.mp4"))

def generate_frame(index: int):
    u, v = np.meshgrid(np.arange(256), np.arange(256), indexing="xy")
    z = np.full_like(u, 255)

    u = (u + index) % 255
    v = (v - index) % 255
    return np.stack((u, v, z), axis=2).astype(np.uint8)

logger.log("Starting up...")
for index in range(1000):
    logger.push_prefix(f"Iteration {index}: ")
    logger.log("Running training step")

    tabular.record("iteration", index)
    tabular.record("loss", 100.0 / (2.0 + float(index)))
    logger.log(tabular)
    logger.log(generate_frame(index))

    logger.pop_prefix()
    logger.dump_all()

logger.remove_all()