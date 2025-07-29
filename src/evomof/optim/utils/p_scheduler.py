# evomof/optim/utils/p_scheduler.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Literal, Tuple


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
    improve_atol: float = 1e-8  # absolute tolerance for "improvement"
    improve_rtol: float = 1e-4  # relative tolerance (fraction of baseline)

    # state
    p: float = field(init=False)
    last_switch_step: int = 0
    _hist: Deque[Tuple[int, float]] = field(init=False)  # (step, global_best_coh)

    def __post_init__(self) -> None:
        self.p = float(self.p0)
        # keep at least window+1 points so we can compare current vs ~window steps ago
        self._hist = deque(maxlen=max(3, self.window + 1))

    def current_p(self) -> float:
        return self.p

    def update(
        self,
        *,
        step: int,
        global_best_coh: float,
    ) -> tuple[float, bool]:
        """Return (current p, switched_flag). Call once per gen/iter."""
        # --- FIXED MODE: pure periodic ramp, no guards ---
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
            return self.p, False

        # --- ADAPTIVE MODE: ramp based on plateau detection ---
        # record history for adaptive decision
        self._hist.append((step, float(global_best_coh)))

        # need at least window+1 samples to compare against value ~window steps ago
        if len(self._hist) < (self.window + 1):
            return self.p, False

        # baseline: oldest value in the buffer (approximately step - window)
        baseline = self._hist[0][1]
        cur = self._hist[-1][1]
        improvement = max(0.0, baseline - cur)
        tol = max(self.improve_atol, self.improve_rtol * max(abs(baseline), 1.0))

        # If no improvement beyond tolerance over the last `window` steps, ramp p
        if improvement <= tol and (step - self.last_switch_step) >= self.min_wait:
            old = self.p
            self.p = min(self.p * self.p_mult, self.p_max)
            self.last_switch_step = step
            return self.p, (self.p != old)

        return self.p, False
