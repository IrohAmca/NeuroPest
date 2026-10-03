"""How long until the engine worker is ready, per circuit size (warm numba cache).

Run: uv run python tools/startup_time.py
"""
import time

from neuropest.runner import EngineConfig, Runner


def main():
    r = Runner(EngineConfig("toy", 146, 0.5))
    try:
        for cfg in (EngineConfig("toy", 146), EngineConfig("flywire", 2_000), EngineConfig("flywire", 20_000),
                    EngineConfig("flywire", 138_639)):
            t = time.perf_counter()
            r.start(cfg)
            while not (r.ready or r.failed):
                time.sleep(0.02)
            print(f"{cfg.circuit:8s} n={cfg.n:>7,}: ready={r.ready} after {time.perf_counter() - t:5.2f} s", flush=True)
    finally:
        r.stop()


if __name__ == "__main__":
    main()
