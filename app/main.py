import argparse
import time
from pathlib import Path
import cv2

from .stitcher import ProgressiveStitcher


def folder_stream(folder: Path, poll_interval: float, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    """
    Poll a folder and yield new images as they appear.
    Additionally, when the folder becomes empty (after previously having files),
    emit a reset signal by yielding (None, "__RESET__").
    """
    seen = set()
    had_files = False
    while True:
        files = [p for p in folder.iterdir() if p.suffix.lower() in exts]
        files.sort(key=lambda p: p.stat().st_mtime)

        if len(files) == 0:
            if had_files:
                # Folder cleared → signal a reset
                yield None, "__RESET__"
                had_files = False
                seen.clear()
            time.sleep(poll_interval)
            continue

        had_files = True
        new_files = [p for p in files if p not in seen]
        for p in new_files:
            seen.add(p)
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            yield img, str(p)
        time.sleep(poll_interval)


# Camera source removed for headless server operation.


def main():
    parser = argparse.ArgumentParser(description="Progressive Image Stitcher (folder-only, headless)")
    parser.add_argument("--folder", type=str, required=True, help="Folder to watch for images")
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Polling interval for folder (s)")
    parser.add_argument("--resize", type=float, default=1.0, help="Resize factor (<1.0 speeds up)")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save latest panorama after each update")
    parser.add_argument("--no-feather", action="store_true", help="Disable feather blending")

    args = parser.parse_args()

    stitcher = ProgressiveStitcher(resize=args.resize, feather=(not args.no_feather))
    # Default: reset on start
    stitcher.reset()

    folder = Path(args.folder)
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")
    stream = folder_stream(folder, poll_interval=args.poll_interval)

    for frame, origin in stream:
        # Folder-based reset: if the folder gets cleared, reset the stitcher
        if origin == "__RESET__":
            stitcher.reset()
            # Also delete current output file if configured
            if args.save_path:
                try:
                    Path(args.save_path).unlink()
                except Exception:
                    pass
            continue

        pano, ok, err = stitcher.add_image(frame)
        if not ok and err:
            # Drop frame on failure, continue
            continue

        if args.save_path and pano is not None:
            cv2.imwrite(args.save_path, pano)



if __name__ == "__main__":
    main()
