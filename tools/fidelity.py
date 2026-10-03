"""Fidelity and cost of each FlyWire tier against the full-brain simulation.

For stimulus levels the ranking was NOT built from (flywire.test_protocols: looming, retreat input,
left/right bearing input, and mixtures), drive the input groups in the full network and in each tier
(top-N neurons by the stored relevance order) and compare:
  GF err     looming protocols:  mean |GF_tier - GF_full| / max(GF_full, 5 Hz)       (take-off command)
  MDN err    retreat protocols:  same for MDN                                         (backward walking)
  DNa02 err  bearing protocols:  same for the DNa02 on the driven side                (steering)
  groom err  touch protocols:    same for the grooming DNs aDN1/aDN2 (GROOM)          (grooming)
  mix err    mixed protocols:    mean over GF, MDN and both DNa02
  DN corr    Pearson correlation of per-DN rates, DNs present in the tier, all protocols pooled
  DN kept    share of the full network's descending-neuron spiking that lies on retained DNs
  active %   mean share of neurons updated per step; realtime x = simulated s per wall s (mean and worst)
Writes data/circuits/tiers.json (read by the GUI).

Two further measurements, which write nothing into tiers.json (their tables go to data/probes/ as CSV):
  --dt 0.1 0.5 1.0   the same protocols at other integration steps: each (tier, dt) against the full brain at the
                     FIRST dt listed (take it as the reference, 0.1), plus the seed-to-seed noise floor of the
                     full brain at that dt; the speed column says what a coarser step buys
  --image            image-driven protocols: cursor scenes go through the REAL VisionDrive, so only the projection
                     neurons whose receptive field the object covers fire (sparse, retinotopic, time-varying), unlike
                     the group-wide protocols above, which drive every LPLC2/LC4/LPC1/LC10 neuron at one rate

The tier size n counts the pinned neurons (stimulus inputs and anchor outputs, which every tier keeps first);
`free` = n - pinned is what the ranking actually chose.

  --mirror           equal LC10_L / LC10_R drive in the full brain: do both sides steer their DNa02 equally?

Run: uv run python tools/fidelity.py [n ...] [--dt 0.1 0.5 1.0] [--image] [--mirror]   (tier sizes BEFORE --dt: it takes every number after it)
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time

import numpy as np

from neuropest import flywire, paths
from neuropest.engine import LIFEngine

DEFAULT_TIERS = (1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 50000, 100000)
DT = 0.5
ANCHORS = ("GF", "MDN", "DNa02_L", "DNa02_R", "GROOM")


def machine_name() -> str:
    """CPU the speed columns were measured on (they are machine specific)."""
    name = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo", encoding="utf8"):
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{name}, {platform.system()}"


def run(net, protocols, warm_ms=300.0, run_ms=1000.0, dt=DT, seed=21):
    g = net.groups
    res = []
    every = np.arange(net.n)
    for proto in protocols:
        e = LIFEngine(net, dt=dt, seed=seed)
        for name, rate in proto.items():
            e.add_drive(g[name], float(rate))
        e.advance(warm_ms)
        e.pop_counts(every)
        u0, st0 = e.neuron_updates, e.steps
        t = time.perf_counter()
        e.advance(run_ms)
        wall = time.perf_counter() - t
        counts = e.pop_counts(every) / (run_ms / 1000.0)
        res.append(dict(counts=counts, rt=run_ms / 1000.0 / wall,
                        active=100.0 * (e.neuron_updates - u0) / (e.steps - st0) / net.n))
    return res


def anchor_rates(net, res):
    g = net.groups
    return np.array([[r["counts"][g[a]].mean() for a in ANCHORS] for r in res])      # (protocols, anchors)


def rel_err(tier, full):
    return np.abs(tier - full) / np.maximum(full, 5.0)


def families(protocols):
    fam = {k: [i for i, p in enumerate(protocols) if list(p) == [k]]
           for k in ("LOOM", "RETREAT_IN", "LC10_L", "LC10_R", "TOUCH_L", "TOUCH_R")}
    return fam, [i for i, p in enumerate(protocols) if len(p) > 1]


def errors(a_tier, a_full, fam, mixed):
    e = rel_err(a_tier, a_full)
    return dict(
        gf_err=100 * float(e[fam["LOOM"], 0].mean()),
        mdn_err=100 * float(e[fam["RETREAT_IN"], 1].mean()),
        steer_err=100 * float(np.mean([e[fam["LC10_L"], 2].mean(), e[fam["LC10_R"], 3].mean()])),
        groom_err=100 * float(e[fam["TOUCH_L"] + fam["TOUCH_R"], 4].mean()),
        mix_err=100 * float(e[mixed].mean()))


def pinned_count(net) -> int:
    """Neurons every tier keeps first: the stimulus input groups and the anchor (output) groups."""
    return len(np.unique(np.concatenate([net.groups[k] for k in flywire.ANCHOR_GROUPS + flywire.INPUT_GROUPS])))


# ------------------------------------------------------------------------------------------------ dt sweep
def dt_sweep(dts, tiers, full):
    """Each (tier, dt) against the full brain at dts[0]; the noise floor is the full brain at dts[0] with another seed."""
    protocols = flywire.test_protocols()
    fam, mixed = families(protocols)
    ref = run(full, protocols, dt=dts[0])
    a_ref = anchor_rates(full, ref)
    rows = []
    head = (f"{'neurons':>8} {'dt ms':>6} | {'GF %':>6} {'MDN %':>6} {'DNa02 %':>7} {'groom %':>7} {'mix %':>6} | "
            f"{'active %':>8} {'x mean':>7} {'x worst':>7}")
    print(f"reference: full brain, dt {dts[0]:g} ms, seed 21 ({machine_name()})")
    print(head)

    def add(label, net, dt, res):
        err = errors(anchor_rates(net, res), a_ref, fam, mixed)
        rts = [r["rt"] for r in res]
        row = dict(n=label, dt=dt, **err, active=float(np.mean([r["active"] for r in res])),
                   realtime=float(np.mean(rts)), rt_min=float(min(rts)))
        rows.append(row)
        print(f"{label:>8} {dt:>6g} | {err['gf_err']:>6.1f} {err['mdn_err']:>6.1f} {err['steer_err']:>7.1f} "
              f"{err['groom_err']:>7.1f} {err['mix_err']:>6.1f} | {row['active']:>8.1f} {row['realtime']:>7.2f} "
              f"{row['rt_min']:>7.2f}", flush=True)

    add("noise", full, dts[0], run(full, protocols, dt=dts[0], seed=22))     # same circuit and dt, other Poisson stream
    for dt in dts:
        add("full", full, dt, run(full, protocols, dt=dt))
    for n in tiers:
        net = full.prefix(n)
        for dt in dts:
            add(str(n), net, dt, run(net, protocols, dt=dt))
    return rows


# ------------------------------------------------------------------------------------------ image protocols
IMAGE_FRAME_S = 0.02
IMAGE_BIN_S = 0.2                  # DNa02 is ONE neuron per side: shorter bins only count single spikes


def image_scenes():
    """name -> (cursor path, seconds, disc radius px). The cursor paths of tools/vision_calibrate.py."""
    import vision_calibrate as vc

    keep = ("still, ahead, 150 px", "still, right, 150 px", "approach 150 px/s", "approach 400 px/s", "approach 800 px/s",
            "approach 1500 px/s", "approach 800 px/s, r 15", "approach 400 px/s, r 60",
            "approach 800 px/s from 60 deg right", "approach 800 px/s from 100 deg left", "slide 400 px/s at 150 px",
            "recede after 3 s near")
    scen = vc.paths([150.0, 400.0, 800.0, 1500.0])
    return {k: scen[k] for k in keep if k in scen}      # the disc-size scenes exist once vision_calibrate has them


def image_run(net, field, scenes, seed=21):
    """-> {scene: dict(GF, MDN, GROOM = peak of the binned rate (Hz), steer = mean DNa02_R - DNa02_L over the scene (Hz),
    driven = neurons forced at any time)}"""
    from neuropest.visual import VisionDrive, VisionParams

    drive = VisionDrive(net, VisionParams(), field)
    g = net.groups
    out = {}
    for name, (path, seconds, *radius) in scenes.items():
        from dataclasses import replace
        base = drive.params
        if radius:
            drive.params = replace(base, halo_px=radius[0])
        e = LIFEngine(net, dt=DT, seed=seed)
        drive.reset()
        bins, acc, driven = [], np.zeros(len(ANCHORS)), set()
        per_bin = int(round(IMAGE_BIN_S / IMAGE_FRAME_S))
        for f in range(int(seconds / IMAGE_FRAME_S)):
            t = f * IMAGE_FRAME_S
            idx, rates = drive.step_cursor(path(t), path(t - IMAGE_FRAME_S), 0.0, 0.0, 0.0, IMAGE_FRAME_S)
            e.set_drive(idx, rates)
            driven.update(idx[rates > 0.5].tolist())
            e.advance(IMAGE_FRAME_S * 1000.0)
            acc += np.array([e.pop_counts(g[a]).mean() for a in ANCHORS]) / IMAGE_FRAME_S
            if (f + 1) % per_bin == 0:
                bins.append(acc / per_bin)
                acc = np.zeros(len(ANCHORS))
        b = np.array(bins)                                           # (bins, anchors)
        steer = float((b[:, ANCHORS.index("DNa02_R")] - b[:, ANCHORS.index("DNa02_L")]).mean())
        out[name] = dict(GF=float(b[:, 0].max()), MDN=float(b[:, 1].max()), steer=steer, GROOM=float(b[:, 4].max()),
                         driven=len(driven))
        drive.params = base
    return out


def image_protocols(tiers, full):
    from neuropest.vision import VisualField

    field = VisualField.load()
    scenes = image_scenes()
    ref = image_run(full, field, scenes)
    print(f"reference: full brain, image-driven scenes ({machine_name()}); GF / MDN = peak of the 200 ms rate, steer = mean right - left DNa02 (Hz)")
    print(f"{'scene':38} | {'GF':>6} {'MDN':>6} {'steer':>6} | driven neurons per tier ->")
    for k, v in ref.items():
        print(f"{k:38} | {v['GF']:>6.1f} {v['MDN']:>6.1f} {v['steer']:>6.1f} | {v['driven']}", flush=True)
    rows = [dict(n=full.n, scene=k, **v) for k, v in ref.items()]
    print(f"\n{'neurons':>8} | {'GF %':>6} {'MDN %':>6} {'steer %':>7}   (mean over scenes of |tier - full| / max(full, 5 Hz))")
    summary = []
    for n in ["noise"] + list(tiers):                       # noise: the full brain again with another Poisson stream
        res = image_run(full if n == "noise" else full.prefix(n), field, scenes, seed=22 if n == "noise" else 21)
        gf = [abs(res[k]["GF"] - ref[k]["GF"]) / max(ref[k]["GF"], 5.0) for k in ref]
        mdn = [abs(res[k]["MDN"] - ref[k]["MDN"]) / max(ref[k]["MDN"], 5.0) for k in ref]
        st = [abs(res[k]["steer"] - ref[k]["steer"]) / max(abs(ref[k]["steer"]), 5.0) for k in ref]
        row = dict(n=n, gf_err=100 * np.mean(gf), mdn_err=100 * np.mean(mdn), steer_err=100 * np.mean(st))
        summary.append(row)
        rows += [dict(n=n, scene=k, **v) for k, v in res.items()]
        print(f"{str(n):>8} | {row['gf_err']:>6.1f} {row['mdn_err']:>6.1f} {row['steer_err']:>7.1f}", flush=True)
    return rows, summary


def mirror_probe(full, rates=(18.0, 35.0, 75.0), seconds=2.0):
    """Equal drive of LC10_L and LC10_R in the full brain: does each side steer its own DNa02 equally?"""
    g, rows = full.groups, []
    print(f"{'LC10 rate':>9} {'driven side':>11} | {'DNa02_L Hz':>10} {'DNa02_R Hz':>10}")
    for rate in rates:
        for side in ("LC10_L", "LC10_R"):
            e = LIFEngine(full, dt=DT, seed=21)
            e.add_drive(g[side], float(rate))
            e.advance(300.0)
            e.pop_counts(np.arange(full.n))
            e.advance(seconds * 1000.0)
            left, right = (e.pop_counts(g[k]).mean() / seconds for k in ("DNa02_L", "DNa02_R"))
            rows.append(dict(lc10_rate_hz=rate, driven=side, dna02_l_hz=float(left), dna02_r_hz=float(right),
                             n_lc10_l=len(g["LC10_L"]), n_lc10_r=len(g["LC10_R"]), n_dna02_l=len(g["DNa02_L"]),
                             n_dna02_r=len(g["DNa02_R"])))
            print(f"{rate:>9g} {side:>11} | {left:>10.1f} {right:>10.1f}", flush=True)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiers", type=int, nargs="*")
    ap.add_argument("--dt", type=float, nargs="+", help="integration steps (ms) to compare; the first is the reference")
    ap.add_argument("--image", action="store_true", help="image-driven protocols through the real VisionDrive")
    ap.add_argument("--mirror", action="store_true", help="left/right symmetry of LC10 -> DNa02 in the full brain")
    a = ap.parse_args()
    full = flywire.load_cache()
    if a.mirror:
        write_csv(paths.ROOT / "data" / "probes" / "steer_symmetry.csv",
                  [dict(machine=machine_name(), **r) for r in mirror_probe(full)])
        if not (a.dt or a.image):
            return
    if a.dt or a.image:
        tiers = a.tiers or [5000, 10000, 15000, 20000, 30000]
        probes = paths.ROOT / "data" / "probes"
        if a.dt:
            rows = dt_sweep(a.dt, tiers, full)
            write_csv(probes / "fidelity_dt.csv", [dict(machine=machine_name(), **r) for r in rows])
        if a.image:
            rows, summary = image_protocols(tiers, full)
            write_csv(probes / "fidelity_image.csv", [dict(machine=machine_name(), **r) for r in rows])
            write_csv(probes / "fidelity_image_summary.csv", [dict(machine=machine_name(), **r) for r in summary])
        return
    protocols = flywire.test_protocols()
    fam, mixed = families(protocols)
    tiers = a.tiers or [n for n in DEFAULT_TIERS if n < full.n]
    pinned = pinned_count(full)
    ref = run(full, protocols)
    a_full = anchor_rates(full, ref)
    dn_full = np.stack([r["counts"][full.groups["DN"]] for r in ref])
    dn_ids = full.groups["DN"]

    rts = [r["rt"] for r in ref]
    out = [dict(n=full.n, pinned=pinned, free=full.n - pinned, edges=int(full.nnz), **errors(a_full, a_full, fam, mixed),
                dn_corr=1.0, dn_kept=1.0, active=float(np.mean([r["active"] for r in ref])), realtime=float(np.mean(rts)),
                rt_min=float(min(rts)))]
    head = (f"{'neurons':>8} {'free':>8} {'edges':>9} | {'GF %':>6} {'MDN %':>6} {'DNa02 %':>7} {'groom %':>7} {'mix %':>6} | "
            f"{'DN corr':>7} {'DN kept %':>9} | {'active %':>8} {'x mean':>7} {'x worst':>7}")
    print(f"pinned (stimulus inputs + anchor outputs, in every tier): {pinned}")
    print(head)
    print(f"{full.n:>8} {full.n - pinned:>8} {full.nnz:>9} | {0:>6.1f} {0:>6.1f} {0:>7.1f} {0:>7.1f} {0:>6.1f} | {1:>7.3f} {100:>9.1f} | "
          f"{out[0]['active']:>8.1f} {out[0]['realtime']:>7.2f} {out[0]['rt_min']:>7.2f}  (full brain)", flush=True)
    for n in tiers:
        sub = full.prefix(n)
        res = run(sub, protocols)
        err = errors(anchor_rates(sub, res), a_full, fam, mixed)
        pos = np.full(full.n, -1, np.int64)
        pos[full.order[:n]] = np.arange(n)
        have = pos[dn_ids] >= 0
        x = dn_full[:, have].ravel()
        b = np.stack([r["counts"][pos[dn_ids[have]]] for r in res]).ravel()
        corr = float(np.corrcoef(x, b)[0, 1]) if have.sum() > 2 and x.std() > 0 and b.std() > 0 else float("nan")
        kept = float(dn_full[:, have].sum() / dn_full.sum())
        rts = [r["rt"] for r in res]
        row = dict(n=n, pinned=min(pinned, n), free=max(n - pinned, 0), edges=int(sub.nnz), **err, dn_corr=corr, dn_kept=kept,
                   active=float(np.mean([r["active"] for r in res])), realtime=float(np.mean(rts)), rt_min=float(min(rts)))
        out.append(row)
        print(f"{n:>8} {row['free']:>8} {sub.nnz:>9} | {err['gf_err']:>6.1f} {err['mdn_err']:>6.1f} {err['steer_err']:>7.1f} {err['groom_err']:>7.1f} {err['mix_err']:>6.1f} | "
              f"{corr:>7.3f} {100 * kept:>9.1f} | {row['active']:>8.1f} {row['realtime']:>7.2f} {row['rt_min']:>7.2f}", flush=True)
    paths.TIERS.write_text(json.dumps(dict(dt=DT, machine=machine_name(), pinned=pinned, protocols=protocols,
                                           tiers=sorted(out, key=lambda r: r["n"])), indent=1))
    print("wrote", paths.TIERS)


if __name__ == "__main__":
    main()
