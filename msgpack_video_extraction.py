from pathlib import Path

import msgpack
import cv2
import numpy as np

# Input/output paths
RECORDING_DIR      = Path("E:/data/noark_data_may_15/linear_t1_may_15")
FRAMES_MSGPACK     = RECORDING_DIR / "webcam_color.msgpack"
TIMESTAMPS_MSGPACK = RECORDING_DIR / "webcam_timestamp.msgpack"
CAM_CSV_OUT        = RECORDING_DIR / "camera_data.csv"
msgpack_file = RECORDING_DIR / "webcam_color.msgpack"
output_video = RECORDING_DIR / "output.mp4"

fps = 100
height = 800
width = 1280

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height),
    isColor=False
)

frames_written = 0

with open(msgpack_file, "rb") as f:

    unpacker = msgpack.Unpacker(f, raw=True)

    for data in unpacker:

        frame_data = data[b'data']

        frame = np.frombuffer(frame_data, dtype=np.uint8)

        frame = frame.reshape((800, 1280))

        out.write(frame)

        frames_written += 1

        print(f"Written frame: {frames_written}", end="\r")

out.release()

print(f"\nSaved: {output_video}")
print(f"Total frames: {frames_written}")