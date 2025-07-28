# evomof/optim/utils/p_scheduler.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Literal, Optional, Tuple


@dataclass
class PScheduler:
    # mode
    mode: Literal["fixed", "adaptive"] = "fixed"

    # p settings
    p0: float = 2.0
    p_mult: float = 1.5
    p_max: float = 1e9

    # fixed mode
    switch_every: int | None = 200  # None disables periodic ramp

    # adaptive mode
    window: int = 60
    min_wait: int = 30
    rel_progress_min: float = 1e-4
    abs_progress_min: float = 5e-7

    # guards (both modes)
    eps_near: float = 5e-7
    spike_ratio: float = 1.4
    cooldown_len: int = 25

    # state
    p: float = field(init=False)
    last_switch_step: int = 0
    _cooldown: int = 0
    _prev_sigma: Optional[float] = None
    _hist: Deque[Tuple[int, float]] = field(init=False)  # (step, global_best_coh)

    def __post_init__(self) -> None:
        self.p = float(self.p0)
        self._hist = deque(maxlen=max(2, self.window))

    def current_p(self) -> float:
        return self.p

    def update(
        self,
        *,
        step: int,
        cur_best_coh: float,
        global_best_coh: float,
        sigma: float | None = None,
    ) -> tuple[float, bool]:
        """Return (current p, switched_flag). Call once per gen/iter."""
        # σ-spike cooldown
        if sigma is not None and self._prev_sigma is not None and self._prev_sigma > 0:
            if (sigma / self._prev_sigma) > self.spike_ratio:
                self._cooldown = self.cooldown_len
        if sigma is not None:
            self._prev_sigma = sigma
        if self._cooldown > 0:
            self._cooldown -= 1

        # near-best guard
        near_best = (cur_best_coh - global_best_coh) <= self.eps_near
        if not near_best or self._cooldown > 0:
            # still record history for adaptive mode
            self._hist.append((step, float(global_best_coh)))
            return self.p, False

        if self.mode == "fixed":
            can_switch = (
                self.switch_every is not None
                and (step - self.last_switch_step) >= self.switch_every
            )
            if can_switch:
                old = self.p
                self.p = min(self.p * self.p_mult, self.p_max)
                self.last_switch_step = step
                return self.p, (self.p != old)
            self._hist.append((step, float(global_best_coh)))
            return self.p, False

        # adaptive mode
        self._hist.append((step, float(global_best_coh)))
        if (step - self.last_switch_step) < self.min_wait or len(self._hist) < 2:
            return self.p, False

        cutoff = step - self.window
        base = self._hist[0][1]
        for s, v in self._hist:
            if s <= cutoff:
                base = v
            else:
                break
        cur = self._hist[-1][1]
        abs_prog = max(0.0, base - cur)
        rel_prog = abs_prog / max(base, 1.0)
        plateau = (abs_prog < self.abs_progress_min) or (
            rel_prog < self.rel_progress_min
        )
        if plateau:
            old = self.p
            self.p = min(self.p * self.p_mult, self.p_max)
            self.last_switch_step = step
            return self.p, (self.p != old)
        return self.p, False
