"""Internal metabolic state and hunger drive.

Simulates biological energy depletion and hunger-driven foraging:
- Stand / Groom: Basal metabolic rate (slow digestion, ~3-4 min to empty).
- Walk: Moderate rate (~65 s of continuous walking).
- Fly: High expenditure (aerodynamic flight burns ~16x resting rate, ~12 s).
- Feeding: Increases energy (+0.35 per consumed source, up to 1.0).

Hunger gates olfactory perception, voluntary foraging locomotion (DNp09 tonic drive),
and Mushroom Body PAM dopamine reward.
"""
from __future__ import annotations

from dataclasses import dataclass

from .states import FLY, RETREAT, STAND, WALK


@dataclass
class MetabolicConfig:
    burn_stand: float = 0.005    # energy / second at rest (~200 s to empty from 100%)
    burn_walk: float = 0.015     # energy / second while walking (~65 s)
    burn_fly: float = 0.080      # energy / second during flight (~12.5 s, 16x rest)
    feed_amount: float = 0.35    # energy gained per food item consumed
    rate_mult: float = 1.0       # user speed multiplier (0.2x to 3.0x)


class MetabolicState:
    """Tracks fly's crop fullness / energy reserve and provides hunger drive."""

    def __init__(self, energy: float = 0.65, enabled: bool = True, cfg: MetabolicConfig | None = None):
        self.energy = float(energy)  # 0.0 (starved) .. 1.0 (satiated)
        self.enabled = enabled
        self.cfg = cfg or MetabolicConfig()
        self.last_state = STAND

    @property
    def hunger(self) -> float:
        """Hunger drive in [0.0, 1.0]. If disabled, returns 1.0 (legacy perpetual hunger)."""
        if not self.enabled:
            return 1.0
        return max(0.0, min(1.0, 1.0 - self.energy))

    @property
    def hunger_pct(self) -> int:
        return int(round(self.hunger * 100))

    @property
    def energy_pct(self) -> int:
        return int(round(self.energy * 100))

    def update(self, dt: float, state: str) -> None:
        if not self.enabled or dt <= 0:
            return
        self.last_state = state
        if state == FLY:
            rate = self.cfg.burn_fly
        elif state in (WALK, RETREAT):
            rate = self.cfg.burn_walk
        else:
            rate = self.cfg.burn_stand
        depletion = rate * self.cfg.rate_mult * dt
        self.energy = max(0.0, self.energy - depletion)

    def feed(self, amount: float | None = None) -> None:
        gain = self.cfg.feed_amount if amount is None else amount
        self.energy = min(1.0, self.energy + gain)

    def starve(self) -> None:
        self.energy = 0.0

    def satiate(self) -> None:
        self.energy = 1.0
