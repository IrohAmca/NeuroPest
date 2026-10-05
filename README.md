# NeuroPest

A desktop fly pet that roams your screen and reacts dynamically to your mouse cursor. Its behavior is directly driven by a Leaky Integrate-and-Fire (LIF) simulation of the fruit fly (*Drosophila melanogaster*) connectome extracted from **FlyWire (v783)**.

**Current State (v0.6+):** The mouse cursor is converted into three primary sensory streams, transformed into motor behaviors by the biological connectome:
- **Looming / approach velocity** $\rightarrow$ backward walking (MDN) when approached gently; explosive escape takeoff (Giant Fiber / DNp01) when approached rapidly.
- **Cursor bearing / proximity** $\rightarrow$ left/right steering turns (DNa02).
- **Motility & foraging drive** $\rightarrow$ tonic current into walking command neurons (DNp09 / P9).
- **Touch / hover on head** $\rightarrow$ antennal and head grooming (aDN1 / aDN2) via cephalic mechanosensory bristles and Johnston's organ.
- **Metabolism & Hunger** $\rightarrow$ state-dependent energy consumption; starvation drives DNp09 hyperactivity, increased foraging, and proboscis extension feeding.
- **Physical boundaries & Nociception** $\rightarrow$ strict screen clamping with elastic bounce; collisions trigger PPL1 dopaminergic punishment signals that teach the Mushroom Body to avoid screen edges.
- **Visual Skin Customization** $\rightarrow$ selectable visual themes (Classic Vector, Chubby Chibi, Cyborg, Pastel Fairy, Cartoon Retro) with smooth 16-frame dorsal locomotion cycles.
- **Real Screen Vision (optional)** $\rightarrow$ the fly perceives the desktop through ommatidia and medulla column receptive fields, with hardware-isolated screen capture (`BitBlt`).

---

## Quickstart

```bash
uv sync
uv run neuropest
```

If the pre-built FlyWire connectome is not found on first launch, a toy circuit (146 neurons) runs as a fallback. See the [Real Data (FlyWire v783)](#real-data-flywire-v783) section below to build the full or reduced connectome tiers.

### Key Behaviors & Controls

- **Interactive Cursor Reactions:**
  - Approaching slowly prompts the fly to **walk backwards (retreat)**.
  - Approaching rapidly triggers an immediate **escape takeoff (fly)**.
  - Moving the cursor near the fly causes it to **turn towards or away** depending on learned valence.
  - Skittishness slider adjusts loom sensitivity. Cursors farther than 700 px are ignored (idling motor).
- **Hunger & Metabolism:**
  - The fly consumes energy based on its behavioral state:
    - Resting / grooming: $\approx 0.005 / \text{s}$ (basal rate)
    - Walking: $\approx 0.015 / \text{s}$
    - Flight: $\approx 0.080 / \text{s}$ (~$16\times$ basal consumption!)
  - Satiated flies groom and rest. Hungry flies experience internal walking drive (DNp09 hyperactivity) and increased food odor sensitivity.
  - Approaching food spots triggers proboscis extension, emerald crop glow, and floating "+Reward" effects.
  - Instant "Feed" and "Starve" actions are available in the control panel; live hunger percentage is badged in the system tray.
- **Screen Boundaries & Wall Avoidance:**
  - Strict 2 px safety margin ensures fly body and wings never clip outside visible screen bounds.
  - Elastic collision bounce reflects velocity inwards upon hitting screen edges.
  - Each wall impact injects a nociceptive pain stimulus (PPL1 dopaminergic pulse), training the Mushroom Body to avoid borders over time (aversive conditioning).
  - Multi-monitor support enables smooth traversal across taskbars and display boundaries (or pinning to a single monitor in View settings).
- **Visual Appearance & Skins:**
  - Real-time skin selector with custom icon cards in the View settings:
    - **Classic (Vector):** Anatomical geometry; proboscis, head, thorax, and 6-legged alternating tripod gait.
    - **Chubby (Chibi):** Soft rounded body, cute wings, 16-step dorsal walk and flight animations.
    - **Cyborg (Sci-Fi):** Mecha robotic chassis, glowing cyan wings, cybernetic joints.
    - **Pastel (Fairy):** Translucent fairy wings, soft pastel palette, delicate footsteps.
    - **Cartoon (Retro):** Expressive compound eyes, retro stylized dorsal patterns.
- **Visual Input (Eye View):**
  - Toggle "See desktop through fly eyes" in the control panel to replace cursor coordinate math with optical sampling through ommatidia receptive fields.
- **Control Panel:**
  - Dark-themed card interface displaying real-time telemetry (Giant Fiber, MDN, Steering, real-time factor, CPU load, active neuron count, valence, hunger).
  - Collapsible **Advanced Behavior Settings** (Motility at 65%, Skittishness $\times 1.00$, Wall Pain $\times 1.00$, with a single-click reset).
  - Closing the control window minimizes it to the system tray; click the tray icon to restore.
- **System Tray Icon:**
  - Dynamic fly eye color reflects internal state: Green = Walking, Orange = Retreating, Red = Escape Flight.
  - Tooltip shows current behavioral state and live hunger percentage.
- **UI Preview:**
  - Run `uv run python tools/ui_preview.py` to render the control interface to a PNG without starting the simulation engine.

---

## Real Data (FlyWire v783)

Raw datasets and connectome files are ignored by git (`data/raw/`, `data/circuits/`). Download sources and git blob hashes (`git hash-object`):

| File | Source | Size | SHA-1 |
|---|---|---|---|
| `Connectivity_783.parquet` | [github.com/philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) | 100.8 MB | `d386555d1a5f40ebfa1380bcb05b1fab044855fd` |
| `Completeness_783.csv` | same repository | 3.3 MB | `b5a26b82b69a3c2e7fd99cd6810c36fc8f0e492b` |
| `Supplemental_file1_neuron_annotations.tsv` | [github.com/flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations) (`supplemental_files/`) | 31.7 MB | `02e72f6c8161d3465f77fec0edf96c5d98027a9e` |

### Build Pipeline

```bash
mkdir -p data/raw
git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model.git /tmp/shiu          # ~190 MB
git clone --depth 1 https://github.com/flyconnectome/flywire_annotations.git /tmp/fwann
cp /tmp/shiu/Connectivity_783.parquet /tmp/shiu/Completeness_783.csv data/raw/
cp /tmp/fwann/supplemental_files/Supplemental_file1_neuron_annotations.tsv data/raw/
git hash-object data/raw/*               # verify matches table above

uv run python tools/build_flywire.py     # ~30s: builds data/circuits/flywire_v783.npz (125 MB) + tiers/
uv run python tools/build_eye.py         # eye models: data/circuits/eye.npz and field.npz (for visual input)
uv run python tools/build_mushroom.py    # optional: biological Mushroom Body (KC/MBON/DAN/PN), data/circuits/mushroom_flywire.npz (~200 kB)
uv run python tools/fidelity.py          # ~5min: data/circuits/tiers.json (fidelity and speed metrics)
# optional probes: tools/fidelity.py 5000 15000 30000 --dt 0.1 0.5 1.0 | --image | --mirror
```

> **Note on shallow clones:** Standard shallow cloning is sufficient; files in these repos are regular git objects, not Git LFS. `sez_neurons.pickle` is deliberately omitted because pickle can execute arbitrary code and is not needed for circuit construction.

**`tiers.json` is machine-specific.** Error columns (takeoff, retreat, steering, descending neuron correlation) are intrinsic to the circuit and drivers. Speed columns reflect single-core throughput measured on the benchmarking host; run `tools/fidelity.py` locally to calibrate for your CPU.

### License and Attribution

- Code: Shiu et al. repository is MIT licensed.
- **FlyWire data is CC BY-NC 4.0** (Attribution required, Non-Commercial use only; confirmed at flywire.ai/guidelines).
- This project is distributed strictly for non-commercial research and educational use. Commercial use requires permission from FlyWire.
- **Citations:**
  - Dorkenwald et al. 2024 (*Nature*, FlyWire connectome).
  - Schlegel et al. 2024 (*Nature*, cell types and annotations).
  - Shiu et al. 2024 (*Nature*, whole-brain LIF model).

---

## Neural Circuit Architecture

The full connectome consists of **138,639 neurons** and **15.1 million synaptic edges**, with synaptic weight $\text{sign} \times \text{synapse count} \times 0.275\text{ mV}$. Mouse cursor activity is translated into Poisson spike trains on sensory projection neurons, and motor outputs are decoded from descending neurons (DNs):

| Cursor Stimulus | Sensory Neurons | Motor Readout | Resulting Behavior | Reference |
|---|---|---|---|---|
| Approach speed (looming) | LPLC2 (size) + LC4 (velocity) | DNp01 (Giant Fiber) | Escape flight / takeoff | Ache 2019; von Reyn 2017 |
| Approach speed (mild) | LPC1 (functional surrogate) | MDN | Backward walking | Bidaye 2014 |
| Bearing & Proximity | LC10a/c-2/d (ipsilateral) | DNa02 (ipsilateral) | Ipsilateral steering turn | Rayshubskiy et al. |
| Motility slider | Tonic current | DNp09 / P9 | Forward walking | Bidaye 2020 |
| Head hover / touch | Cephalic bristles (BM_*) + JO-C/E | DNg62 (aDN1) + DNge078 (aDN2) | Head & antenna grooming | Hampel 2015; Shiu 2024 |
| Hunger state | Metabolic deficit tonic drive | DNp09 | Hyperactivity & foraging | Krashes 2009 |
| Boundary collision | Border impact | PPL1 dopaminergic cluster | Nociceptive pain conditioning | Claridge-Chang 2009 |

### Motor Control & Latency

Takeoff is decoded as an **event** rather than a continuous firing rate: in biological fruit flies, 1–2 spikes from the Giant Fiber trigger immediate escape jumping. Thus, two GF spikes within a 10 ms window (`BrainSpec.gf_event_spikes`) immediately switch state to `FLY`. An 80 ms exponentially filtered rate pathway acts as an exit condition. Measured latency (`tools/latency.py`, 15k tier, 8 seeds):
- Takeoff takes 8 ms (down from 16 ms) at $20/\text{s}$ expansion.
- Takeoff takes 14 ms at $10/\text{s}$ expansion.
- Takeoff threshold begins at $\approx 4/\text{s}$ expansion.

### Behavioral States & Precedence

1. **Takeoff / Fly** (`FLY`)
2. **Retreat / Backward Walk** (`RETREAT`)
3. **Freeze** (`FREEZE`)
4. **Groom** (`GROOM`)
5. **Walk** (`WALK`) / **Stand** (`STAND`)

Higher priority states immediately interrupt lower states; state transitions enforce minimal durations and hysteresis to prevent jitter. 

- **Freezing:** Model has no single dedicated freeze neuron; freezing is an engineered behavioral rule (`BrainSpec.freeze_on = 0.7`). If looming rate exceeds 0.7/s without triggering escape or retreat, the fly freezes for $\ge 1.2\text{ s}$ until expansion drops below 0.3/s.
- **Escape Direction:** DNp02/DNp11 encode lateral LC4 gradients. Escape trajectory is directed opposite to cursor heading (`fly.py`).
- **Left-Right Symmetrization:** Synaptic weights and connection signs are bilaterally balanced (`flywire.py`, `lif_wgpu.py`) to eliminate unilateral steering biases under symmetrical stimuli.

---

## Reduced Connectome Tiers

Because the network exhibits minimal spontaneous baseline activity without input, neurons that never fire can be pruned without affecting circuit output. Neurons are ranked by total spike counts across training input protocols (with input/anchor neurons always retained). Tier $N$ represents the top $N$ ranked neurons.

Benchmarked on Intel Core i5-10300H (single-core, $\text{d}t = 0.5\text{ ms}$; error relative to full 138k connectome):

| Neurons | Takeoff (GF) Error | Retreat (MDN) Error | Steering (DNa02) Error | DN Correlation | Speed (Avg / Worst-case) |
|---:|---:|---:|---:|---:|---:|
| 2,000 | 29% | 75% | 98% | 0.743 | $\times 65$ / $\times 28$ |
| 5,000 | 13% | 20% | 23% | 0.993 | $\times 27$ / $\times 7.0$ |
| 10,000 | 9% | 6% | 10% | 0.997 | $\times 20$ / $\times 4.4$ |
| **15,000 (Default)** | **0.6%** | **5.4%** | **0.8%** | **0.999** | **$\times 11$ / $\times 2.1$** |
| 20,000 | 0.1% | 0.5% | 1.2% | 1.000 | $\times 5.7$ / $\times 1.7$ |
| 50,000 | 0% | 0% | 0.9% | 0.999 | $\times 4.0$ / $\times 0.5$ |
| 138,639 (Full) | 0% | 0% | 0% | 1.000 | $\times 2.5$ / $\times 0.4$ |

> **Key takeaway:** At **15,000 neurons**, fidelity is virtually indistinguishable from the whole brain ($r = 0.999$) while maintaining an average speedup of **$\times 11$ real-time** on a single CPU core. Idle CPU load is negligible ($\approx 0\%$) when the mouse cursor is far away.

---

## Visual Input & Desktop Capture

When "See desktop through fly eyes" is enabled, the fly views the screen through an ommatidial cone matching Drosophila optics. Each medulla column points in a specific azimuth and elevation, sampling the desktop plane.

```bash
uv run python tools/build_eye.py       # builds eye.npz and field.npz (+ eye_map.png)
uv run python tools/vision_runner.py   # end-to-end worker test with scripted motions
```

### Screen Capture Pipeline (`neuropest/capture.py`)

- **Subprocess Isolation:** Screen capture runs in a dedicated worker process communicating via zero-lock shared memory (`multiprocessing.Array`).
- **GDI BitBlt:** Captures a 480×480 px window centered around the fly ($\approx 5\text{ ms/frame}$, converted to grayscale uint8 with Numba).
- **Dynamic Framerate:** 8 fps during movement ($\approx 4\%$ CPU core); drops to 2 fps when scene is static ($\approx 1\%$ CPU core).
- **Overlay Isolation:** Uses `SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)` (0x11) so the fly's own sprite is completely excluded from desktop capture—the fly never mistakes itself for a predator!
- **Multi-Monitor & DPI:** Per-monitor DPI awareness handles virtual screen offsets, including negative coordinates (e.g. `-1920, 0`).

### Retinotopic Motion Detectors (`neuropest/vision.py`)

- **Looming Expansion Detector (LPLC2 / LC4):** Detects symmetric outward edge expansion across four quadrants. LPLC2 receives receptive-field area coverage (angular size), while LC4 receives expansion velocity.
- **Small Target Motion Detector (LC10):** Measures localized dark or bright contrast relative to surrounding background.

---

## Mushroom Body & Reinforcement Learning

The Mushroom Body mediates associative learning via three-factor synaptic plasticity at Kenyon Cell $\to$ MBON synapses, gated by Dopaminergic Neurons (DANs).

- **Dopamine as Prediction Error (RPE):** Dopamine signal represents reinforcement minus cue valence already learned ($\text{Reinforcement} - \text{Valence}$). A fully anticipated reward or threat produces zero net learning.
- **Extinction Learning:** An expected reinforcement that does not arrive triggers active memory extinction (`eta_ext`), decaying associative weights.
- **Valence Modulation:** Learned valence ($V \in [-1, +1]$) dynamically modulates behavior:
  - Positive valence ($V > 0$): increases forward walking drive and attraction towards the cursor.
  - Negative valence ($V < 0$): heightens skittishness, reverses steering away from the cursor, and triggers defensive takeoff.
- **Visual Learning from Connectome PNs:** When visual input is active, visual projection neurons reaching Kenyon cells (aMe12, MTe32, MTe30, LTe25, MTe40 for cursor; aMe26, LTe72, MTe37 for looming) directly stimulate KCs through connectome receptive fields (`tools/probe_mb_vision.py`).
- **Real FlyWire Wiring Mode (Opt-in):** Setting `NEUROPEST_MB=flywire` loads real anatomical connectivity (`tools/build_mushroom.py`) with 5,177 Kenyon cells, 96 MBONs (35 types), and biological PAM/PPL1 innervation.
- **Persistence:** Learned memories persist between launches in `data/memory/fly_memory.npz` (or `fly_memory_flywire.npz`), with a one-click "Forget" button in the Behavior tab.

---

## Simulation Engine & Hardware Acceleration

The core simulation engine implements leaky integrate-and-fire dynamics with exact exponential integration and 1.8 ms synaptic delay:

- `LIFEngine` (`neuropest/engine/lif.py`): Event-driven Numba CPU engine. Only updates active neurons each step; computational cost scales with active spikes rather than total network size.
- `ReferenceEngine`: Dense NumPy implementation used as an oracle for unit test verification.
- `WGPUEngine` (`neuropest/engine/lif_wgpu.py`): Cross-platform GPU acceleration via WebGPU (`wgpu`). Supports Vulkan, Metal, and DirectX on NVIDIA, AMD, Intel, and Apple Silicon without requiring CUDA.

### Optional GPU Acceleration

```bash
uv sync --extra gpu
uv run neuropest
```

Select Compute in the control panel: **Auto / CPU / GPU**. "Auto" automatically delegates circuits of 50,000+ neurons to the GPU while running smaller tiers on the lightweight CPU engine.

Benchmarked on full connectome (138,639 neurons, 15.1M synapses, $\text{d}t = 0.5\text{ ms}$):

| Input Load (Spikes/s) | CPU (Numba) | NVIDIA GTX 1650 (Vulkan) | Intel UHD Graphics (Vulkan) |
|---|---:|---:|---:|
| 10% Retina @ 20 Hz (21k) | $\times 9 - 11$ | $\times 12.8$ | $\times 3.4$ |
| 50% Retina @ 20 Hz (105k) | $\times 1.9$ | $\times 13.7$ | $\times 3.0$ |
| 50% Retina @ 50 Hz (265k) | $\times 1.0 - 1.4$ | $\times 11.7$ | $\times 3.1$ |
| 100% Retina @ 50 Hz (538k) | $\times 0.5$ | $\times 9.1$ | $\times 2.5$ |
| 100% Retina @ 100 Hz (1.1M) | $\times 0.2 - 0.3$ | $\times 7.0$ | $\times 2.1$ |
| Looming 50 Hz (33k) | $\times 0.4$ | $\times 8.6$ | $\times 2.7$ |

---

## Repository Structure

```
NeuroPest/
├── neuropest/
│   ├── app.py              # Main desktop application, transparent overlay & Qt lifecycle
│   ├── brain.py            # Neural network coordinator, sensory encoding & motor decoding
│   ├── capture.py          # Isolated BitBlt screen capture worker with shared memory
│   ├── control.py          # Dark-themed control panel & telemetry window
│   ├── eyebuild.py         # Medulla column alignment & ommatidial cone projection
│   ├── fly.py              # Kinematics, heading, tripod gait phase, screen boundary physics
│   ├── flywire.py          # FlyWire connectome loader, ranking, tiers & symmetrization
│   ├── metabolism.py       # Energy consumption, hunger states & proboscis extension
│   ├── mushroom.py         # Mushroom Body RL (3-factor plasticity, prediction error, MBONs)
│   ├── paths.py            # Standard dataset, cache, and asset directory paths
│   ├── pheromone.py        # Odor diffusion field modeling & bilateral antennal tropotaxis
│   ├── render.py           # Procedural vector fly & multi-skin sprite renderer
│   ├── runner.py           # Background simulation worker process & rate controller
│   ├── states.py           # Fly behavioral state constants (FLY, WALK, RETREAT, GROOM, etc.)
│   ├── theme.py            # Modern UI styling, color palettes & tray icons
│   ├── vision.py           # Retinotopic feature detectors (looming & small target)
│   ├── visual.py           # Optical projection neuron drive & visual learning cues
│   ├── assets/skins/       # Sprite animations (idle, walk, fly) and UI thumbnails
│   └── engine/             # LIF engines (Numba CPU, WGPU WebGPU, NumPy reference)
├── tools/                  # Connectome builders, fidelity probes, vision calibration & benchmarks
└── tests/                  # Pytest verification test suite (150+ unit and integration tests)
```

---

## Testing

Run the full automated test suite:

```bash
uv run pytest
```

Tests validate engine equivalence, connectome tier indexing, screen capture memory isolation, Mushroom Body learning dynamics, metabolism states, pheromone tropotaxis, multi-skin rendering, and physical margin reflections.
