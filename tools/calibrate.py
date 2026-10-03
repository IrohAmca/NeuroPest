"""Print what a circuit does for a few cursor scenarios (calibration aid).

Run: uv run python tools/calibrate.py [toy|flywire] [n_neurons] [seconds]
"""
from __future__ import annotations

import sys
from collections import Counter

from neuropest.brain import Brain

SCENARIOS = [
    ("cursor far, still", 900, 0, 0.0),
    ("cursor near, still", 150, 0, 0.0),
    ("slow approach 300 px/s @300px", 300, 300, 0.0),
    ("steady approach 600 px/s @300px", 300, 600, 0.0),
    ("brisk approach 1200 px/s @300px", 300, 1200, 0.0),
    ("fast approach 3000 px/s @150px", 150, 3000, 0.0),
    ("far, walk bias 0.3", 900, 0, 0.3),
    ("far, walk bias 0.6", 900, 0, 0.6),
    ("far, walk bias 0.7", 900, 0, 0.7),
    ("far, walk bias 0.8", 900, 0, 0.8),
    ("far, walk bias 1.0", 900, 0, 1.0),
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
    for name, dist, closing, bias in SCENARIOS:
        b = Brain(net)
        b.set_stimulus(1e6, 0, 0.0)
        for _ in range(250):           # 1 s settle
            b.advance(4.0)
        b.set_stimulus(dist, closing, bias)
        seen, rates, first_fly, switches, prev = Counter(), [0.0, 0.0, 0.0], None, 0, None
        for step in range(steps):
            s = b.advance(4.0)
            seen[s] += 1
            switches += prev is not None and s != prev
            prev = s
            if s == "fly" and first_fly is None:
                first_fly = step * 4
            r = b.rates
            rates[0] += r["GF"]; rates[1] += r["WALK"]; rates[2] += r["REST"]
        pct = 100.0 / steps
        print(f"{name:34s} GF {rates[0] / steps:6.1f}  WALK {rates[1] / steps:6.1f}  REST {rates[2] / steps:6.1f} Hz | "
              f"stand {seen['stand'] * pct:4.0f}% walk {seen['walk'] * pct:4.0f}% fly {seen['fly'] * pct:4.0f}% "
              f"| {switches / seconds:4.1f} sw/s"
              + (f" | first fly after {first_fly} ms" if first_fly is not None else ""), flush=True)


if __name__ == "__main__":
    main()
