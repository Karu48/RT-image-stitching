import argparse
import time
from pathlib import Path
import cv2
import numpy as np

from .stitcher import ProgressiveStitcher


def folder_stream(folder: Path, poll_interval: float, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    seen = set()
    while True:
        files = [p for p in folder.iterdir() if p.suffix.lower() in exts]
        files.sort(key=lambda p: p.stat().st_mtime)
        new_files = [p for p in files if p not in seen]
        for p in new_files:
            seen.add(p)
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            yield img, str(p)
        time.sleep(poll_interval)


def camera_stream(index: int, skip: int = 1):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % skip != 0:
                continue
            yield frame, None
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description="Progressive Image Stitcher")
    parser.add_argument("--source", choices=["folder", "camera"], required=True)
    parser.add_argument("--folder", type=str, help="Folder to watch for images")
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Polling interval for folder (s)")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index for VideoCapture")
    parser.add_argument("--camera-skip", type=int, default=2, help="Use each Nth frame from camera")
    parser.add_argument("--resize", type=float, default=1.0, help="Resize factor (<1.0 speeds up)")
    parser.add_argument("--no-view", action="store_true", help="Do not show UI window")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save latest panorama after each update")
    parser.add_argument("--no-feather", action="store_true", help="Disable feather blending")

    args = parser.parse_args()

    stitcher = ProgressiveStitcher(resize=args.resize, feather=(not args.no_feather))

    if args.source == "folder":
        if not args.folder:
            raise SystemExit("--folder is required when --source folder")
        folder = Path(args.folder)
        if not folder.exists():
            raise SystemExit(f"Folder not found: {folder}")
        stream = folder_stream(folder, poll_interval=args.poll_interval)
    else:
        stream = camera_stream(args.camera_index, skip=args.camera_skip)

    last_update = time.time()
    for frame, origin in stream:
        pano, ok, err = stitcher.add_image(frame)
        if not ok and err:
            # Drop frame on failure, continue
            continue

        now = time.time()
        # Throttle UI updates a bit
        if pano is not None and (not args.no_view) and (now - last_update > 0.02):
            cv2.imshow("Progressive Panorama", pano)
            last_update = now
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if args.save_path and pano is not None:
            cv2.imwrite(args.save_path, pano)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
