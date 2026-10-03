"""Print what a circuit does for a few cursor scenarios (calibration aid).

Run: uv run python tools/calibrate.py [toy|flywire] [n_neurons] [seconds]
bearing: radians, positive = cursor to the fly's right.
"""
from __future__ import annotations

import sys
from collections import Counter

from neuropest.brain import Brain

# name, distance px, closing speed px/s, walk bias, bearing rad
SCENARIOS = [
    ("cursor far, still", 900, 0, 0.0, 0.0),
    ("cursor near, still, ahead", 150, 0, 0.0, 0.0),
    ("cursor near, still, right", 150, 0, 0.0, 1.5),
    ("cursor near, still, left", 150, 0, 0.0, -1.5),
    ("slow approach 300 px/s @300px", 300, 300, 0.0, 0.0),
    ("steady approach 500 px/s @300px", 300, 500, 0.0, 0.0),
    ("steady approach 700 px/s @300px", 300, 700, 0.0, 0.0),
    ("brisk approach 1200 px/s @300px", 300, 1200, 0.0, 0.0),
    ("fast approach 3000 px/s @150px", 150, 3000, 0.0, 0.0),
    ("far, walk bias 0.6", 900, 0, 0.6, 0.0),
    ("far, walk bias 0.8", 900, 0, 0.8, 0.0),
]


def make_net(kind: str, n: int):
    if kind == "flywire":
        from neuropest import flywire
        return flywire.load_cache().prefix(n)
    from neuropest.toy_circuit import build
    return build(n)


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "toy"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else (2000 if kind == "flywire" else 146)
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    steps = int(seconds * 250)
    net = make_net(kind, n)
    print(f"{kind} circuit, {net.n:,} neurons")
    for name, dist, closing, bias, bearing in SCENARIOS:
        b = Brain(net)
        b.set_stimulus(1e6, 0, 0.0, bearing=0.0)
        for _ in range(250):           # 1 s settle
            b.advance(4.0)
        b.set_stimulus(dist, closing, bias, bearing=bearing)
        seen, acc, first_fly, switches, prev, steer = Counter(), Counter(), None, 0, None, 0.0
        for step in range(steps):
            s = b.advance(4.0)
            seen[s] += 1
            switches += prev is not None and s != prev
            prev = s
            if s == "fly" and first_fly is None:
                first_fly = step * 4
            for k in ("GF", "MDN", "WALK"):
                acc[k] += b.rates[k]
            steer += b.steer
        pct = 100.0 / steps
        print(f"{name:34s} GF {acc['GF'] / steps:5.1f} MDN {acc['MDN'] / steps:5.1f} WALK {acc['WALK'] / steps:5.1f} "
              f"steer {steer / steps:+6.1f} | stand {seen['stand'] * pct:3.0f}% walk {seen['walk'] * pct:3.0f}% "
              f"retreat {seen['retreat'] * pct:3.0f}% fly {seen['fly'] * pct:3.0f}% | {switches / seconds:3.1f} sw/s"
              + (f" | fly after {first_fly} ms" if first_fly is not None else ""), flush=True)


if __name__ == "__main__":
    main()
