# RT Image Stitching (Progressive)

A minimal real-time (progressive) image stitcher that incrementally builds a panorama as images arrive (from a folder or camera stream). Uses OpenCV features + RANSAC homography and simple feather blending.

## Features
- Incremental homography composition relative to a keyframe
- Automatic canvas growth with coordinate offset tracking
- Simple feather blending in overlaps
- Folder polling (new images auto-stitched) or camera stream

## Quick Start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run watching a folder for new images (headless):
```powershell
python -m app.main --folder .\samples --poll-interval 0.5 --resize 0.5 --save-path panorama.jpg
```

Notes:
- Use `--resize` (< 1.0) to speed up matching with high-res frames.
- Headless by default: no window is created; use `--save-path` to persist the latest panorama after each update.

## Limitations & Tips
- Works best for planar scenes or pure rotational motion. For significant parallax, expect artifacts.
- If you frequently move back over the same area, consider adding loop-closure + pose-graph optimization (not included in this minimal prototype).
- Use consistent exposure/focus; drastic changes reduce match quality.
- If stitching panoramas from rotation, consider adding cylindrical projection for reduced distortion.

## Project Structure
```
RT-image-stitching/
  app/
    __init__.py
    main.py
    stitcher.py
  requirements.txt
  README.md
```

## Runtime Behavior (Final Version)
- Folder-only: the runner polls an input folder and stitches images as they appear.
- Reset on start: the internal stitcher state is cleared when the process starts.
- Folder-based reset: if the folder becomes empty after previously having images, the stitcher resets automatically.
- Output file cleanup: when a folder-based reset occurs, the runner removes the current output file specified by `--save-path` to start clean.
- Saving: if `--save-path` is set, the latest panorama is written after each successful update.

## Example Commands
```powershell
# Start stitching from a folder and save the panorama
python -m app.main --folder .\samples --poll-interval 0.5 --resize 0.5 --save-path panorama.jpg

# Reset by clearing the folder contents; the stitcher resets automatically and deletes the output file
Remove-Item .\samples\* -Force

# Add new images to begin a new session
Copy-Item .\new_session\*.jpg .\samples\
```

## License
This sample is provided as-is for educational purposes.