import math
from unittest.mock import MagicMock
from PySide6.QtCore import QRect
from neuropest.fly import Fly, PlayArea, ScreenBox
from neuropest.states import STAND, WALK
from neuropest.pheromone import PheromoneField


def test_screen_available_geometry_bounds():
    # Mock a QScreen with different geometry (1920x1080) and availableGeometry (1920x1032, 48px taskbar)
    mock_screen = MagicMock()
    mock_screen.name.return_value = "Display1"
    mock_screen.geometry.return_value = QRect(0, 0, 1920, 1080)
    mock_screen.availableGeometry.return_value = QRect(0, 0, 1920, 1032)
    mock_screen.devicePixelRatio.return_value = 1.0

    area = PlayArea.from_screens([mock_screen], margin=24.0)
    box = area.screens[0]

    # Must use availableGeometry (height 1032) rather than full geometry (1080)
    assert box.raw_b == 1032.0, f"Expected raw_b=1032.0 (excluding taskbar), got {box.raw_b}"
    assert box.usable_b == 1032.0 - 24.0, f"Expected usable_b=1008.0, got {box.usable_b}"


def test_set_home_respawns_pheromones_in_active_screen():
    mock_screen = MagicMock()
    mock_screen.name.return_value = "DisplayPinned"
    mock_screen.geometry.return_value = QRect(0, 0, 1920, 1080)
    mock_screen.availableGeometry.return_value = QRect(0, 0, 1920, 1032)
    mock_screen.devicePixelRatio.return_value = 1.0

    phero = PheromoneField(border_margin=30.0, border_strength=0.4)
    area = PlayArea.from_screens([mock_screen], margin=24.0)
    phero.spawn_random_sources(area.screens, count=7)

    assert len(phero.sources) == 7
    for src in phero.sources:
        assert area.screens[0].usable_l <= src.x <= area.screens[0].usable_r
        assert area.screens[0].usable_t <= src.y <= area.screens[0].usable_b


def test_stand_orientation_reflex():
    s0 = ScreenBox(0, "Main", 0, 0, 1920, 1080, 0, 0, 1920, 1080, 1.0, 24.0)
    area = PlayArea([s0])

    fly = Fly(960.0, 540.0)
    fly.heading = 0.0  # facing +x
    # Provide positive steer (clockwise / turning right toward +y) in STAND state
    dt = 0.016
    for _ in range(30):
        fly.update(dt, STAND, (960.0, 700.0), area, steer=25.0)

    # Heading must have turned clockwise (> 0.0) even in STAND state
    assert fly.heading > 0.05, f"Expected standing fly to orient toward steer stimulus, got heading {fly.heading}"


def test_walk_wander_attenuation_during_active_steering():
    s0 = ScreenBox(0, "Main", 0, 0, 1920, 1080, 0, 0, 1920, 1080, 1.0, 24.0)
    area = PlayArea([s0])

    fly = Fly(960.0, 540.0)
    fly.heading = 0.0
    dt = 0.016
    # Steer strongly clockwise (e.g. steer=35.0 Hz from DNa02)
    for _ in range(60):
        fly.update(dt, WALK, (960.0, 800.0), area, steer=35.0)

    # Active steering must dominate random wander and turn clockwise
    assert fly.heading > 0.2, f"Expected fly to turn strongly clockwise under steer=35, got {fly.heading}"


def test_grazing_wall_bounce_inward_deflection():
    s0 = ScreenBox(0, "Main", 0, 0, 1920, 1080, 0, 0, 1920, 1080, 1.0, 24.0)
    area = PlayArea([s0])

    # 1. Fly approaching right wall with shallow angle (heading almost vertical, e.g. 0.05 rad)
    fly = Fly(1920.0 - 24.0 - 1.0, 500.0)
    fly.heading = 0.05
    nx, ny, nh = area.step(fly.x, fly.y, fly.heading, 70.0, 0.1)

    assert area.last_hit_wall is True
    # Inward deflection ensures cos(nh) <= -0.25 (heading points back into the room)
    assert math.cos(nh) <= -0.25, f"Expected inward reflection on right wall, got cos(nh)={math.cos(nh)}"

    # 2. Fly approaching bottom wall with shallow angle (heading almost horizontal)
    fly_b = Fly(500.0, 1080.0 - 24.0 - 1.0)
    fly_b.heading = math.pi / 2 - 0.05
    nx_b, ny_b, nh_b = area.step(fly_b.x, fly_b.y, fly_b.heading, 70.0, 0.1)

    assert area.last_hit_wall is True
    # Inward deflection ensures sin(nh_b) <= -0.25 (heading points upward back into the room)
    assert math.sin(nh_b) <= -0.25, f"Expected inward reflection on bottom wall, got sin(nh)={math.sin(nh_b)}"


def test_overlay_never_covers_a_monitor_edge_to_edge():
    # A borderless window equal to a monitor rect makes the Windows shell treat it as a fullscreen app
    # and drop the taskbar's topmost state, so other windows overlap the taskbar.
    from neuropest.app import overlay_rect

    r = overlay_rect(-1920, 0, 4480, 1440)   # two monitors: 1920x1080 left of a 2560x1440 primary
    assert (r.left(), r.top(), r.right() + 1, r.bottom() + 1) == (-1919, 1, 2559, 1439)

    r = overlay_rect(0, 0, 1920, 1080)
    assert r.width() == 1918 and r.height() == 1078

    assert overlay_rect(0, 0, 1, 1).width() >= 1   # degenerate desktop must not produce an empty rect
