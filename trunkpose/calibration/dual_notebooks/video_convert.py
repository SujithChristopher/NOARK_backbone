import cv2
import msgpack as mp
import msgpack_numpy as mpn
import os
from pathlib import Path
from tqdm.auto import tqdm
import argparse


def get_video_unpacker(vidf):
    _video_file = open(vidf, "rb")
    return mp.Unpacker(_video_file, object_hook=mpn.decode)


def count_frames(vidf):
    unpacker = get_video_unpacker(vidf)
    return sum(1 for _ in unpacker)


def msgpack_to_video(msgpack_path, output_path, fps=30):
    unpacker = get_video_unpacker(msgpack_path)

    first_frame = next(unpacker)
    h, w = first_frame.shape[:2]
    is_color = len(first_frame.shape) == 3

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h), isColor=is_color)

    # write first frame
    frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR) if is_color else first_frame
    writer.write(frame_bgr)

    total = count_frames(msgpack_path) - 1  # already consumed one
    unpacker = get_video_unpacker(msgpack_path)
    next(unpacker)  # skip first, already written

    for frame in tqdm(unpacker, total=total, desc=output_path.name):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if is_color else frame
        writer.write(frame_bgr)

    writer.release()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert dual-cam msgpack files to mp4")
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Path to calibration data folder (default: auto-detect from project root)",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS (default: 30)")
    parser.add_argument(
        "--cam",
        choices=["both", "cam0", "cam1"],
        default="both",
        help="Which camera(s) to convert (default: both)",
    )
    args = parser.parse_args()

    if args.folder:
        calib_data_folder = Path(args.folder)
    else:
        project_root = Path(__file__).parents[3]
        calib_data_folder = (
            project_root
            / "data"
            / "calibration"
            / "dual_160"
            / "dual_cam_calibration_checker_sz_30mm"
        )
        calib_data_folder = (
            project_root
            / "data"
            / "dual_data"
            / "dual_camera_trunk_test"
        )

    if not calib_data_folder.exists():
        raise FileNotFoundError(f"Data folder not found: {calib_data_folder}")

    cam_files = {
        "cam0": calib_data_folder / "cam0_imx219.msgpack",
        "cam1": calib_data_folder / "cam1_ov9281.msgpack",
    }

    targets = ["cam0", "cam1"] if args.cam == "both" else [args.cam]

    for cam in targets:
        src = cam_files[cam]
        if not src.exists():
            print(f"Skipping {cam}: {src} not found")
            continue
        dst = calib_data_folder / f"{cam}.mp4"
        print(f"\nConverting {src.name} -> {dst.name}")
        msgpack_to_video(src, dst, fps=args.fps)


if __name__ == "__main__":
    main()
