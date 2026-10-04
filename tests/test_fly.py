import math
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


def test_frozen_fly_stays_in_place():
    from neuropest.states import FREEZE

    fly = Fly(500, 500)
    x, y, heading = fly.x, fly.y, fly.heading
    for _ in range(60):
        fly.update(1 / 60, FREEZE, (900, 500), RECT)
    assert (fly.x, fly.y) == (x, y) and abs(math.remainder(fly.heading - heading, math.tau)) < 1e-9 and fly.phase == 0.0


def test_multi_screen_seam_crossing():
    from neuropest.fly import PlayArea, ScreenBox

    s0 = ScreenBox(
        id=0, name="Main",
        raw_l=0.0, raw_t=0.0, raw_r=2560.0, raw_b=1392.0,
        phys_l=0.0, phys_t=0.0, phys_r=2560.0, phys_b=1392.0,
        dpr=1.0, margin=18.0,
    )
    s1 = ScreenBox(
        id=1, name="Left",
        raw_l=-1920.0, raw_t=0.0, raw_r=-384.0, raw_b=816.0,
        phys_l=-1920.0, phys_t=0.0, phys_r=0.0, phys_b=1020.0,
        dpr=1.25, margin=18.0,
    )
    area = PlayArea([s0, s1])

    # 1. Fly on Screen 0 walking left towards Screen 1 (at physical Y = 500)
    fly = Fly(10.0, 500.0)
    fly.heading = math.pi  # facing left
    fly.turn, fly.turn_t = 0.0, 10.0
    # Move across seam (speed 70 px/s, dt 0.2s -> delta 14 px left)
    fly.update(0.2, WALK, (10.0, 500.0), area)
    # Overshoot past 0 by 4 px -> on s1: raw_r (-384) - 4 / 1.25 = -387.2
    assert fly.x < -384.0, f"Expected fly on s1, got {fly.x}"
    # Physical Y (500) translated to s1 logical Y = 500 / 1.25 = 400.0
    assert abs(fly.y - 400.0) < 0.1, f"Expected y=400, got {fly.y}"
    assert abs(math.remainder(fly.heading - math.pi, math.tau)) < 0.1  # heading continues left without bouncing

    # 2. Fly on Screen 1 walking right towards Screen 0
    fly2 = Fly(-390.0, 400.0)
    fly2.heading = 0.0  # facing right
    fly2.turn, fly2.turn_t = 0.0, 10.0
    fly2.update(0.2, WALK, (-390.0, 400.0), area)
    # Overshoot past -384 by 8 px -> on s0: raw_l (0) + 8 * 1.25 = 10.0
    assert fly2.x > 0.0, f"Expected fly on s0, got {fly2.x}"
    # Physical Y (400 * 1.25 = 500) translated to s0 logical Y = 500
    assert abs(fly2.y - 500.0) < 0.1, f"Expected y=500, got {fly2.y}"
    assert abs(fly2.heading - 0.0) < 0.1  # heading continues right


def test_multi_screen_taskbar_and_void_bounds():
    from neuropest.fly import PlayArea, ScreenBox

    s0 = ScreenBox(
        id=0, name="Main",
        raw_l=0.0, raw_t=0.0, raw_r=2560.0, raw_b=1392.0,
        phys_l=0.0, phys_t=0.0, phys_r=2560.0, phys_b=1392.0,
        dpr=1.0, margin=18.0,
    )
    s1 = ScreenBox(
        id=1, name="Left",
        raw_l=-1920.0, raw_t=0.0, raw_r=-384.0, raw_b=816.0,
        phys_l=-1920.0, phys_t=0.0, phys_r=0.0, phys_b=1020.0,
        dpr=1.25, margin=18.0,
    )
    area = PlayArea([s0, s1])

    # 1. Fly on Screen 1 hitting bottom taskbar (y = 816 - margin = 798)
    fly = Fly(-1000.0, 790.0)
    fly.heading = math.pi / 2  # moving downwards (+y)
    fly.update(0.5, WALK, (-1000.0, 790.0), area)
    assert fly.y <= 816.0 - 18.0  # bounced off taskbar
    assert fly.heading < 0  # bounced upwards

    # 2. Fly on Screen 0 at Y = 1200 (below Screen 1's height of 1020 physical) moving left
    fly2 = Fly(25.0, 1200.0)
    fly2.heading = math.pi  # moving left (-x)
    fly2.update(0.5, WALK, (25.0, 1200.0), area)
    assert fly2.x >= 18.0  # bounced off solid left edge
    assert abs(fly2.heading) < math.pi / 2  # bounced rightwards

    # 3. Wall push is 0 on open portal, but positive on solid wall
    # At Y = 500 (portal exists to left):
    wx_portal, _ = area.wall_push(25.0, 500.0, 110.0)
    assert wx_portal == 0.0, f"Expected 0 wall push at open portal, got {wx_portal}"
    # At Y = 1200 (no portal, void to left):
    wx_wall, _ = area.wall_push(25.0, 1200.0, 110.0)
    assert wx_wall > 0.0, f"Expected inward wall push at solid wall, got {wx_wall}"


def test_multi_screen_physical_continuity():
    from neuropest.fly import Fly, PlayArea, ScreenBox
    from neuropest.states import WALK

    # Two monitors touching at x = 0 in physical desktop coordinates (matching Overlay HWND)
    s0 = ScreenBox(
        id=0, name="Main",
        raw_l=0.0, raw_t=0.0, raw_r=2560.0, raw_b=1440.0,
        phys_l=0.0, phys_t=0.0, phys_r=2560.0, phys_b=1440.0,
        dpr=1.0, margin=2.0,
    )
    s1 = ScreenBox(
        id=1, name="Left",
        raw_l=-1920.0, raw_t=0.0, raw_r=0.0, raw_b=1080.0,
        phys_l=-1920.0, phys_t=0.0, phys_r=0.0, phys_b=1080.0,
        dpr=1.0, margin=2.0,
    )
    area = PlayArea([s0, s1])

    # 1. Fly steps from Main (x=5) leftwards into Left screen (dt=0.2s, speed=70 -> delta 14px)
    fly = Fly(5.0, 500.0)
    fly.heading = math.pi
    fly.turn, fly.turn_t = 0.0, 10.0
    fly.update(0.2, WALK, (5.0, 500.0), area)
    # 5.0 - 14.0 = -9.0. Exactly continuous across seam, no gap!
    assert abs(fly.x - (-9.0)) < 0.1, f"Expected x=-9.0, got {fly.x}"
    assert abs(fly.y - 500.0) < 0.1
    assert abs(math.remainder(fly.heading - math.pi, math.tau)) < 0.1

    # 2. Fly steps from Left (x=-5) rightwards into Main screen
    fly2 = Fly(-5.0, 500.0)
    fly2.heading = 0.0
    fly2.turn, fly2.turn_t = 0.0, 10.0
    fly2.update(0.2, WALK, (-5.0, 500.0), area)
    # -5.0 + 14.0 = 9.0. Continuous across seam!
    assert abs(fly2.x - 9.0) < 0.1, f"Expected x=9.0, got {fly2.x}"
    assert abs(fly2.y - 500.0) < 0.1
    assert abs(fly2.heading - 0.0) < 0.1

    # 3. Fly can freely occupy the previously-skipped zone [-384, 0] on the left screen
    fly3 = Fly(-200.0, 500.0)
    fly3.heading = 0.0
    fly3.turn, fly3.turn_t = 0.0, 10.0
    fly3.update(0.1, WALK, (-200.0, 500.0), area)
    assert abs(fly3.x - (-193.0)) < 0.1


def test_tight_screen_boundary_margins():
    from neuropest.fly import Fly, PlayArea, ScreenBox
    from neuropest.states import WALK

    s0 = ScreenBox(
        id=0, name="Main",
        raw_l=0.0, raw_t=0.0, raw_r=1920.0, raw_b=1080.0,
        phys_l=0.0, phys_t=0.0, phys_r=1920.0, phys_b=1080.0,
        dpr=1.0, margin=2.0,
    )
    area = PlayArea([s0])

    # 1. Fly can reach right up to margin=2.0 at left border
    fly = Fly(3.0, 500.0)
    fly.heading = math.pi
    fly.turn, fly.turn_t = 0.0, 10.0
    fly.update(0.05, WALK, (3.0, 500.0), area)
    assert fly.x >= 2.0 and fly.x < 2.5  # stopped right at 2px boundary

    # 2. Wall push is 0 when 30px away from the boundary (since margin is now 14px for WALK)
    wx, wy = area.wall_push(30.0, 500.0, 14.0)
    assert wx == 0.0 and wy == 0.0


def test_boundary_reflection_and_hit_wall_flag():
    from neuropest.fly import FLY_MARGIN_PX, FLY_RADIUS_PX, Fly, PlayArea, ScreenBox
    from neuropest.states import WALK

    # Verify safe margin constants: margin must exceed radius by at least 1-2px
    assert FLY_MARGIN_PX >= FLY_RADIUS_PX + 2.0

    s0 = ScreenBox(
        id=0, name="Main",
        raw_l=0.0, raw_t=0.0, raw_r=1920.0, raw_b=1080.0,
        phys_l=0.0, phys_t=0.0, phys_r=1920.0, phys_b=1080.0,
        dpr=1.0, margin=FLY_MARGIN_PX,
    )
    area = PlayArea([s0])

    # 1. Fly walking straight into right border
    fly = Fly(1920.0 - FLY_MARGIN_PX - 2.0, 500.0)
    fly.heading = 0.0  # pointing right (+x)
    fly.turn, fly.turn_t = 0.0, 10.0
    fly.update(0.1, WALK, (2000.0, 500.0), area)

    # Must have hit wall, stopped at usable_r, and reflected inward (-x)
    assert fly.hit_wall is True
    assert fly.x <= s0.usable_r
    assert math.cos(fly.heading) < -0.2  # heading pointing back into the room

    # 2. Fly walking straight into bottom border
    fly_b = Fly(500.0, 1080.0 - FLY_MARGIN_PX - 2.0)
    fly_b.heading = math.pi / 2  # pointing down (+y)
    fly_b.turn, fly_b.turn_t = 0.0, 10.0
    fly_b.update(0.1, WALK, (500.0, 1200.0), area)

    assert fly_b.hit_wall is True
    assert fly_b.y <= s0.usable_b
    assert math.sin(fly_b.heading) < -0.2  # heading pointing upward


def test_wall_nociception_retreat_reflex_and_learning():
    from neuropest.brain import Brain
    from neuropest.toy_circuit import build

    brain = Brain(build(146))
    # Baseline with no wall bump
    brain.set_stimulus(500.0, 0.0, walk_bias=0.65, wall_bump=0.0)
    assert brain._punish == 0.0

    # Impact with wall: triggers acute mechanosensory retreat and nociceptive punishment
    brain.set_stimulus(500.0, 0.0, walk_bias=0.65, phero_repel=0.7, wall_bump=1.0)
    assert brain.wall_bump == 1.0
    brain._learn(20.0)
    assert brain._punish >= 1.0  # PPL1 dopamine punishment fired
    brain.advance(40.0)


