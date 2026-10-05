# NeuroPest - Source Fly Animation & Skin Art Assets

This directory contains raw source sprite sheets for top-down fly skins and animation cycles.

## File Manifest

| File | Description | Perspective | Animation States / Purpose |
|---|---|---|---|
| `05_cartoon_fly_topdown_dorsal.jpg` | **Cartoon Fly (Top-Down)** | **Dorsal (Top-Down)** | Flight (wing stroke) & Walk (tripod gait) |
| `06_chubby_fly_topdown_flight.jpg` | **Chubby Chibi Flight Sheet** | **Dorsal (Top-Down)** | Flight cycle (Frames 1–12) |
| `07_chubby_fly_topdown_walk.jpg` | **Chubby Chibi Walk Sheet** | **Dorsal (Top-Down)** | Walking cycle (body bob & leg stepping) |
| `08_pastel_candy_fly_topdown.jpg` | **Pastel / Candy Fairy Fly** | **Dorsal (Top-Down)** | Flight (glittering fairy wings) & Walk |
| `09_cyborg_fly_topdown.jpg` | **Cyborg / Mecha Fly Drone** | **Dorsal (Top-Down)** | Flight (cyan energy wings) & Robotic Walk |

## Code Integration & Asset Processing

- In the desktop overlay coordinate frame (`neuropest/app.py` and `neuropest/render.py`), fly heading is measured relative to the `+x` axis (East).
- Raw dorsal frames facing North are rotated $90^\circ$ clockwise during processing and centered on a $128\times 128$ canvas.
- To process and extract these raw sheets into clean transparent PNG runtime frames:
  ```bash
  uv run python tools/prepare_all_skins.py
  ```
  Generated runtime frames are output directly to `neuropest/assets/skins/`.
