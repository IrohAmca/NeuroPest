# NeuroPest - Runtime Fly Skins & Sprite Assets

This directory contains the runtime fly skins and animated sprite frames used by the desktop simulation and overlay renderer.

## Directory Structure

- `thumbnails/`: 64×64 preview badges displayed on the visual skin cards in the Control Panel's "Appearance" section (`classic.png`, `chubby.png`, `cyborg.png`, `candy.png`, `cartoon.png`).
- `chubby/`: **Chubby Fly (Chibi)** – 16-step walking gait cycle (`walk_*.png`), 12-frame flight cycle (`fly_*.png`), and resting pose (`idle.png`).
- `cyborg/`: **Cyborg Fly (Sci-Fi)** – 6-step walking cycle and 6-frame glowing energy wing flight cycle.
- `candy/`: **Pastel Fly (Fairy / Candy)** – 8-step walking cycle and 6-frame fluttering fairy wings.
- `cartoon/`: **Cartoon Fly (Retro)** – 8-step walking cycle and 8-frame flight cycle.

## Technical Specifications

1. **Coordinate System Alignment:** All sprites are oriented facing `+x` (East / right) to match the simulation's heading convention ($0\text{ rad} = \text{facing right}$).
2. **Center of Rotation:** The thorax (physical center of rotation) is aligned to the canvas center $(64, 64)$ for seamless angular rotation.
3. **Alpha Defringing & Edge Ramping:** Processed via mathematical defringing to eliminate white boundary halos; smooth continuous alpha ramping prevents leg glowing artifacts on dark or transparent desktop surfaces.
4. **Regeneration Pipeline:** Sprites can be rebuilt and processed from raw sheets at any time using:
   ```bash
   uv run python tools/prepare_all_skins.py
   ```
   To visually inspect and verify all frames in a grid:
   ```bash
   uv run python tools/preview_all_skins.py
   ```
