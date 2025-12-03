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

Run with a camera:
```powershell
python -m app.main --source camera --camera-index 0 --resize 0.75
```

Run watching a folder for new images:
```powershell
python -m app.main --source folder --folder .\samples --poll-interval 0.5 --resize 0.75
```

Notes:
- Use `--resize` (< 1.0) to speed up matching with high-res frames.
- Press `q` in the display window to quit. Use `--no-view` to run headless and optionally save to `--save-path`.

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

## License
This sample is provided as-is for educational purposes.