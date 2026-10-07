"""Does a spiking optic lobe, driven through the lamina by a screen image, see looming and motion?

A fly stands on the screen plane facing +x. Scenes are drawn on the plane (a dark disk on mid-gray).
Lamina cells (L1/L2/L3, or photoreceptors) fire at a rate set by the luminance their column looks at
(darker = more, as the histamine synapses invert the photoreceptor signal) after a slow adaptation, the full
brain runs on the GPU, and the response of every visual projection neuron type is recorded.

Run: uv run python tools/vision_experiment.py [--adapter 0] [--h 60] [--r 40] [--kinds L1,L2,L3]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from neuropest import flywire
from neuropest.engine import create_engine
from neuropest.vision import KINDS, Retina, sample_scene

FRAME_MS = 1000.0 / 60.0


def disk(cx, cy, radius, dark=0.0, back=0.5):
    def scene(px, py):
        inside = (px - cx) ** 2 + (py - cy) ** 2 <= radius ** 2
        return np.where(inside, dark, back).astype(np.float32)
    return scene


def uniform(level):
    return lambda px, py: np.full(px.shape, level, np.float32)


def stimuli(r):
    """name -> function(t seconds) -> scene, over t in [0, 2.0]; the stimulus runs from t = 0.5 s."""
    def loom(t):                                              # dark disk approaches along the heading
        s = np.clip((t - 0.5) / 1.5, 0, 1)
        return disk(600 - 560 * s, 0, r)

    def recede(t):
        s = np.clip((t - 0.5) / 1.5, 0, 1)
        return disk(40 + 560 * s, 0, r)

    def slide(t):                                             # same disk crossing at 120 px, no expansion
        s = np.clip((t - 0.5) / 1.5, 0, 1)
        return disk(120, -400 + 800 * s, r)

    def flash(t):
        return disk(150, 0, r) if t >= 0.5 else uniform(0.5)

    def step_dark(t):
        return uniform(0.5 if t < 0.5 else 0.2)

    return {"uniform": lambda t: uniform(0.5), "loom": loom, "recede": recede, "slide": slide, "flash": flash,
            "global dark step": step_dark}


class Rig:
    def __init__(self, net, retina, kinds, backend, adapter, eye_height, r0=30.0, gain=120.0, tau_adapt=0.3,
                 bias_mv=0.0, bias_w=0.4):
        self.net, self.h = net, eye_height
        mask = np.isin(retina.kind, [KINDS[k] for k in kinds])
        self.retina = retina.subset(mask)
        self.r0, self.gain, self.tau = r0, gain, tau_adapt
        self.engine = create_engine(net, dt=0.5, seed=1, backend=backend, adapter=adapter)
        ann = flywire.load_annotations(net.ids)               # rows aligned with the network index
        self.groups = {}
        cell = ann.cell_type.fillna("").to_numpy()
        sc = ann.super_class.fillna("").to_numpy()
        if bias_mv > 0:
            # graded neurons sit above rest in the fly: emulate with tonic Poisson synaptic input of mean bias_mv
            # (steady-state depolarisation = rate * w * tau_syn); only optic lobe and visual projection neurons
            tonic = np.flatnonzero(np.isin(sc, ["optic", "visual_projection", "visual_centrifugal"])).astype(np.int32)
            # deterministic: one event of size w every step; steady state g = w / (1 - exp(-dt/tau_syn)) = bias
            w = bias_mv * (1.0 - np.exp(-0.5 / 5.0))
            self.engine.set_current_drive(tonic, 1e9, w)
            self.bias_idx = tonic
        for ct in np.unique(cell[sc == "visual_projection"]):
            self.groups[ct] = np.flatnonzero((cell == ct) & (sc == "visual_projection")).astype(np.int32)
        for k in ("GF", "MDN", "WALK"):
            self.groups[k] = net.groups[k]
        self.groups["DNa02"] = np.concatenate([net.groups["DNa02_L"], net.groups["DNa02_R"]])
        for pref in ("T4", "T5"):
            self.groups[pref] = np.flatnonzero(np.char.startswith(cell.astype(str), pref)).astype(np.int32)
        for ct in ("Mi1", "Tm1", "Tm2", "Tm3", "Tm4", "Tm9", "L1", "L2", "L3"):
            self.groups[ct] = np.flatnonzero(cell == ct).astype(np.int32)
        self.names = list(self.groups)
        self.mon_idx = np.concatenate([self.groups[n] for n in self.names]).astype(np.int32)
        self.edges = np.cumsum([0] + [len(self.groups[n]) for n in self.names])
        self.adapt = np.full(len(self.retina), 0.5, np.float32)

    def frame(self, scene, adapt=True):
        lum = sample_scene(self.retina, scene, 0.0, 0.0, 0.0, self.h)
        a = 1 - np.exp(-(FRAME_MS / 1000.0) / self.tau)
        self.adapt += a * (lum - self.adapt)
        c = lum - (self.adapt if adapt else 0.5)
        rate = np.clip(self.r0 - self.gain * c, 0.0, 200.0)
        self.engine.set_drive(self.retina.idx, rate)
        self.engine.advance(FRAME_MS)
        counts = self.engine.pop_counts(self.mon_idx)
        cs = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])
        return np.array([(cs[self.edges[i + 1]] - cs[self.edges[i]]) / max(1, len(self.groups[n])) /
                         (FRAME_MS / 1000.0) for i, n in enumerate(self.names)])        # Hz per neuron

    def run(self, make_scene, seconds=2.0, adapt=True):
        out = []
        for i in range(int(seconds * 60)):
            out.append(self.frame(make_scene(i / 60.0), adapt))
        return np.array(out)                                                          # (frames, groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=int, default=0)
    ap.add_argument("--backend", default="gpu")
    ap.add_argument("--h", type=float, default=60.0, help="eye height above the screen, px")
    ap.add_argument("--r", type=float, default=40.0, help="disk radius, px")
    ap.add_argument("--kinds", default="L1,L2,L3")
    ap.add_argument("--r0", type=float, default=30.0)
    ap.add_argument("--gain", type=float, default=120.0)
    ap.add_argument("--no-adapt", action="store_true")
    ap.add_argument("--top", type=int, default=14)
    ap.add_argument("--bias", type=float, default=0.0, help="tonic depolarisation of optic lobe neurons, mV")
    ap.add_argument("--bias-w", type=float, default=0.4, help="size of one tonic synaptic event, mV")
    a = ap.parse_args()

    net = flywire.load_cache()
    retina = Retina.load()
    rig = Rig(net, retina, a.kinds.split(","), a.backend, a.adapter, a.h, a.r0, a.gain, bias_mv=a.bias, bias_w=a.bias_w)
    print(f"{len(rig.retina)} driven cells ({a.kinds}), eye height {a.h} px, disk radius {a.r} px, backend {a.backend}, "
          f"tonic bias {a.bias} mV")
    rig.run(lambda t: uniform(0.5), 1.0, not a.no_adapt)                               # settle
    results = {}
    for name, fn in stimuli(a.r).items():
        t0 = time.perf_counter()
        s0 = rig.engine.total_spikes
        results[name] = rig.run(lambda t, fn=fn: fn(t), 2.0, not a.no_adapt)
        print(f"  ran '{name}' in {time.perf_counter() - t0:.1f} s; whole brain {(rig.engine.total_spikes - s0) / 2.0 / 1000:.0f}k spikes/s",
              flush=True)
    names = rig.names
    base = {k: v[15:30].mean(0) for k, v in results.items()}                          # 0.25-0.5 s: before onset
    late = {k: v[84:120].mean(0) for k, v in results.items()}                          # 1.4-2.0 s: late in the stimulus
    peak = {k: v[30:120].max(0) for k, v in results.items()}
    print("\nmean Hz per neuron, late in the stimulus (after baseline of the same run in brackets)")
    cols = list(results)
    print(f"{'type':10s} " + " ".join(f"{c[:16]:>16s}" for c in cols))
    interest = ["LPLC2", "LC4", "LC6", "LPLC1", "LPC1", "LC17", "LC15", "LC10a", "LC10c-2", "LC10d", "LC11", "LC12",
                "LC16", "LC9", "T4", "T5", "Mi1", "Tm1", "Tm3", "L1", "GF", "MDN", "DNa02", "WALK"]
    for n in interest:
        if n in names:
            j = names.index(n)
            print(f"{n:10s} " + " ".join(f"{late[c][j]:7.1f} ({base[c][j]:5.1f})" for c in cols))
    # response selectivity over all visual projection types: loom vs uniform and loom vs slide
    vpn = [n for n in names if n in rig.groups and n not in ("GF", "MDN", "WALK", "DNa02", "T4", "T5", "Mi1", "Tm1", "Tm2",
                                                         "Tm3", "Tm4", "Tm9", "L1", "L2", "L3")]
    j = {n: names.index(n) for n in vpn}
    score = sorted(((late["loom"][j[n]] - late["uniform"][j[n]], n) for n in vpn), reverse=True)[:a.top]
    print("\nvisual projection types most driven by the looming disk (late rate minus the uniform scene):")
    for s, n in score:
        print(f"  {n:10s} loom {late['loom'][j[n]]:6.1f}  uniform {late['uniform'][j[n]]:6.1f}  slide {late['slide'][j[n]]:6.1f}  "
              f"recede {late['recede'][j[n]]:6.1f}  flash {late['flash'][j[n]]:6.1f}  peak(loom) {peak['loom'][j[n]]:6.1f}")
    from neuropest.paths import CIRCUITS
    np.savez(CIRCUITS / "vision_experiment.npz", names=np.array(names), **{k.replace(" ", "_"): v for k, v in results.items()})


if __name__ == "__main__":
    main()
