"""Behavior states; kept free of heavy imports so the GUI process doesn't load numba."""
STAND, WALK, FLY, RETREAT, GROOM, FREEZE = "stand", "walk", "fly", "retreat", "groom", "freeze"
STATES = (STAND, WALK, FLY, RETREAT, GROOM, FREEZE)       # the worker passes the INDEX: append, never reorder
