"""
Lightweight, stateful p-schedulers used by CMA/CG drivers.

Design
------
We expose a small interface via a typing.Protocol:

    class PScheduler(Protocol):
        def current_p(self) -> float: ...
        def update(self, *, step: int, global_best_coh: float)
            -> tuple[float, bool]: ...

Two concrete implementations satisfy the protocol:

- FixedPScheduler
    Periodic schedule: multiply p by p_mult every `switch_every` steps,
    capped at p_max. Ignores coherence history.

- AdaptivePScheduler
    (Former “budgeted” scheduler) Event-based stall detector with a *constant* window.
    Ramp when (i) there has been no new global-best improvement for the last `window`
    steps, and (ii) at least `window` steps have elapsed since the previous ramp.
    On each ramp, choose a *budgeted* multiplier so that, assuming the same stall
    interval repeats, p reaches p_max by the end of total_steps. Requires total_steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

__all__ = [
    "PScheduler",
    "FixedPScheduler",
    "AdaptivePScheduler",
]


# ----------------------------- Public interface -----------------------------


class PScheduler(Protocol):
    """Minimal interface for a p-scheduler."""

    def current_p(self) -> float:
        """Return the current value of p without advancing state."""
        ...

    def update(self, *, step: int, global_best_coh: float) -> tuple[float, bool]:
        """Advance one step and (optionally) increase p.

        Parameters
        ----------
        step : int
            Monotone counter (e.g., generation/iteration index). Must be keyword-only.
        global_best_coh : float
            Best (lowest) coherence observed so far over the entire run.

        Returns
        -------
        (p, switched) : tuple[float, bool]
            Updated p to use for the *next* step and a flag indicating whether
            a ramp occurred.
        """
        ...


# --------------------------- Concrete schedulers ----------------------------


@dataclass
class FixedPScheduler:
    """Periodic p schedule.

    Every `switch_every` steps, multiply p by `p_mult`, cap at `p_max`.

    Parameters
    ----------
    p0 : float
        Initial p.
    p_mult : float
        Multiplicative factor (>1) when ramping.
    p_max : float
        Upper bound for p.
    switch_every : int | None
        Steps between ramps. If None, no ramps occur.

    State
    -----
    p : float
        Current p.
    last_switch_step : int
        Step index of the last ramp.
    """

    p0: float = 2.0
    p_mult: float = 1.5
    p_max: float = 1e9
    switch_every: int | None = 200

    # state
    p: float = field(init=False)
    # Intitialize to 1 so the first ramp occurs at step (1 + switch_every)
    # and the value returned by update() at that step is the new p
    last_switch_step: int = 1

    def __post_init__(self) -> None:
        self.p = float(self.p0)

    def current_p(self) -> float:
        return self.p

    def update(self, *, step: int, global_best_coh: float) -> tuple[float, bool]:
        if self.switch_every is None:
            return self.p, False
        if (step - self.last_switch_step) >= self.switch_every:
            old = self.p
            self.p = min(self.p * self.p_mult, self.p_max)
            self.last_switch_step = step
            return self.p, (self.p != old)
        return self.p, False


@dataclass
class AdaptivePScheduler:
    """Event-based, budgeted p schedule with a constant window.

    Ramp when:
      (i) there has been **no new global best** for the last `window` steps, and
     (ii) at least `window` steps have elapsed since the previous ramp.

    When ramping, choose a per-ramp multiplier so that, *if the observed stall
    interval repeats*, p will reach p_max by the end of total_steps. This gives
    predictable end-of-run targeting without hitting p_max too early.

    Parameters
    ----------
    p0 : float
        Initial p.
    p_max : float
        Upper bound for p.
    total_steps : int
        Total planned steps (e.g., generations). REQUIRED (> 0).
    window : int
        Constant patience window (in steps) for both stall and spacing tests.

    State
    -----
    p : float
        Current p.
    last_switch_step : int
        Step index of the last ramp event.
    _best_seen : float
        Best (lowest) coherence observed so far (strict improvement).
    _last_improve_step : int
        Step index when the last improvement occurred.
    """

    p0: float = 2.0
    p_max: float = 1e9
    total_steps: int = 0  # REQUIRED > 0
    window: int = 200

    # state
    p: float = field(init=False)
    last_switch_step: int = 0
    _best_seen: float = field(init=False, default=float("inf"))
    _last_improve_step: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("AdaptivePScheduler requires total_steps > 0.")
        if self.window <= 0:
            raise ValueError("AdaptivePScheduler requires window > 0.")
        self.p = float(self.p0)

    def current_p(self) -> float:
        return self.p

    def update(self, *, step: int, global_best_coh: float) -> tuple[float, bool]:
        # enforce driver bounds
        if step > self.total_steps:
            raise ValueError("AdaptivePScheduler.update called with step > total_steps")

        # Track strict improvements of the global best (any decrease counts)
        if global_best_coh < self._best_seen:
            self._best_seen = global_best_coh
            self._last_improve_step = step

        # Require a full window since the last ramp before considering another
        since_last_switch = step - self.last_switch_step
        if since_last_switch < self.window:
            return self.p, False

        # Stall criterion: no improvement within the last `window` steps
        stalled = (step - self._last_improve_step) >= self.window
        if not stalled:
            return self.p, False

        # Observed interval = generations since last ramp; assume it repeats
        interval = max(1, since_last_switch)
        steps_left = max(0, self.total_steps - step)
        ramps_possible = max(1, steps_left // interval)

        ratio = self.p_max / self.p
        if ratio <= 1.0:
            target_mult = 1.0
        else:
            target_mult = ratio ** (1.0 / ramps_possible)

        old = self.p
        self.p = min(self.p * max(1.0, target_mult), self.p_max)
        self.last_switch_step = step
        return self.p, (self.p != old)
