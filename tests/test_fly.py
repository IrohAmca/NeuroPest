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


def test_retreat_walks_backward_and_steering_turns_toward_the_cursor_side():
    import math

    from neuropest.states import RETREAT

    fly = Fly(500, 500)
    fly.heading = 0.0                                   # facing +x
    fly.update(0.5, RETREAT, (900, 500), RECT)
    assert fly.x < 500 and abs(fly.y - 500) < 1e-6      # moved backward, along -x
    f2 = Fly(500, 500)
    f2.heading, f2.turn, f2.turn_t = 0.0, 0.0, 10.0     # no random wander during the step
    f2.update(0.1, WALK, (500, 900), RECT, steer=50.0)
    assert f2.heading > 0.05                            # positive steer turns right (clockwise, toward +y)
    f3 = Fly(500, 500)
    f3.heading = 0.0
    assert abs(f3.bearing_of((500, 900)) - math.pi / 2) < 1e-9      # cursor below a fly facing +x: 90 deg right


def test_clamp_after_taskbar_grows():
    fly = Fly(500, 1000)
    fly.clamp((25.0, 25.0, 1895.0, 900.0))
    assert fly.y == 900.0


def test_grooming_fly_stays_in_place():
    from neuropest.states import GROOM

    fly = Fly(500, 500)
    x, y, heading = fly.x, fly.y, fly.heading
    for _ in range(60):
        fly.update(1 / 60, GROOM, (500, 500), RECT)
    assert (fly.x, fly.y) == (x, y) and abs(fly.heading - heading) < 1e-9 and fly.phase != 0.0
