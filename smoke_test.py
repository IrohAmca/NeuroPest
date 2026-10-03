import time

from neuropest.brain import Brain

b = Brain()
t = time.perf_counter()
for _ in range(100):
    b.step(16, 600, 0)
print("idle:", b.state, f"{(time.perf_counter() - t) * 10:.1f} ms/step")
seen = set()
for _ in range(100):
    seen.add(b.step(16, 150, 0))
print("near, still:", seen)
seen = set()
for _ in range(60):
    seen.add(b.step(16, 120, 3000))
print("looming:", seen, "GF Hz", float(b.c.rate[b.g["GF"]].mean()))
