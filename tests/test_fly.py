import random

from neuropest.brain import FLY, STAND, WALK
from neuropest.fly import Fly

# a 1920x1080 monitor with a 48 px taskbar: usable area ends at y = 1032 - margin
RECT = (25.0, 25.0, 1895.0, 1007.0)


def _simulate(seed, minutes=10, cursor_mode="random"):
    rnd = random.Random(seed)
    random.seed(seed)
    fly = Fly(960, 500)
    state, left = STAND, 0
    ys, xs = [], []
    cur = (rnd.uniform(0, 1920), rnd.uniform(0, 1080))
    for _ in range(minutes * 60 * 60):
        left -= 1
        if left <= 0:
            state = rnd.choices([STAND, WALK, FLY], [5, 4, 1])[0]
            left = rnd.randint(30, 400) if state != FLY else rnd.randint(20, 120)
            cur = (rnd.uniform(0, 1920), rnd.uniform(0, 1080))
        fly.update(1 / 60, state, cur, RECT)
        assert RECT[0] <= fly.x <= RECT[2] and RECT[1] <= fly.y <= RECT[3], (fly.x, fly.y)
        ys.append(fly.y)
        xs.append(fly.x)
    return xs, ys


def test_never_leaves_usable_area():
    for seed in range(5):
        _simulate(seed)


def test_not_pinned_to_the_bottom():
    _, ys = _simulate(11, minutes=20)
    near_bottom = sum(y > RECT[3] - 12 for y in ys) / len(ys)
    near_top = sum(y < RECT[1] + 12 for y in ys) / len(ys)
    assert near_bottom < 0.08, near_bottom
    assert near_bottom < 3 * max(near_top, 0.01)
    assert 0.35 < sum(ys) / len(ys) / RECT[3] < 0.65


def test_clamp_after_taskbar_grows():
    fly = Fly(500, 1000)
    fly.clamp((25.0, 25.0, 1895.0, 900.0))
    assert fly.y == 900.0
