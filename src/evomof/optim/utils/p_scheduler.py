"""
Lightweight, stateful p-scheduler used by CMA/CG drivers.

The scheduler exposes three modes that control how the sharpness parameter `p`
increases over the course of an optimization run:

• "fixed": multiply `p` by `p_mult` every `switch_every` steps (pure periodic ramp).
• "adaptive": increase `p` only if the *global best coherence* has not
  improved (within tolerances) over the last `window` steps; also respect a
  minimum spacing of `min_wait` steps between two increases.
• "budgeted": event-based stalls (constant window) with per‑ramp multipliers chosen
  so that, assuming the observed stall interval repeats, `p` reaches `p_max`
  by the end of `total_steps` (REQUIRES `total_steps`).

This module is intentionally dependency-free so it can be reused from both
CMA and local (CG/L-BFGS) scripts.
"""

# evomof/optim/utils/p_scheduler.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Literal, Tuple, cast


@dataclass
class PScheduler:
    """Stateful scheduler for the frame potential parameter `p`.

    Modes
    -----
    - ``mode="fixed"``: periodic ramp. Increase ``p`` by ``p_mult`` every
      ``switch_every`` steps, capped at ``p_max``. No guards.
    - ``mode="adaptive"``: stall-triggered ramp. Track the *global best
      coherence* over time; if there has been no improvement beyond tolerances
      over the last ``window`` steps, and at least ``min_wait`` steps have passed
      since the previous increase, multiply ``p`` by ``p_mult`` (capped by
      ``p_max``).
    - ``mode="budgeted"``: event-based stall with constant window.
      Ramp when there has been **no global-best improvement** for the last ``window``
      steps **and** at least ``window`` steps have elapsed since the previous ramp.
      The per‑ramp multiplier is chosen so that, if the same stall interval repeats,
      ``p`` reaches ``p_max`` by the end of ``total_steps``
      (REQUIRES ``total_steps``).

    Parameters
    ----------
    mode : {"fixed", "adaptive", "budgeted"}
        Strategy for increasing ``p``.
    p0 : float
        Initial value of ``p``.
    p_mult : float
        Multiplicative factor (>1) applied when ramping ``p``.
    p_max : float
        Upper bound for ``p``.
    switch_every : int | None
        (Fixed mode) Number of steps between two ramps. ``None`` disables ramping.
    window : int
        (Adaptive mode) Lookback window (in steps) used to test for improvement.
    min_wait : int
        (Adaptive mode) Minimum number of steps between two ramps.
    improve_atol : float
        (Adaptive mode) Absolute tolerance: treat improvements smaller than this
        as "no improvement".
    improve_rtol : float
        (Adaptive mode) Relative tolerance w.r.t. the baseline value.
    total_steps : int | None
        (Budgeted mode) Total planned steps (e.g., generations).
        REQUIRED when mode="budgeted".

    State
    -----
    p : float
        Current value of ``p``.
    last_switch_step : int
        Step index of the last ramp event.
    _hist : deque[tuple[int, float]]
        History of ``(step, global_best_coh)``; sized to ``window+1`` so a
        baseline ~``step - window`` always exists.
    _best_seen : float
        Best global coherence seen so far (for budgeted mode).
    _last_improve_step : int
        Step index of the last improvement (for budgeted mode).
    """

    # mode
    mode: Literal["fixed", "adaptive", "budgeted"] = "fixed"

    # p settings
    p0: float = 2.0
    p_mult: float = 1.5
    p_max: float = 1e9

    # fixed mode
    switch_every: int | None = 200  # None disables periodic ramp

    # adaptive mode
    window: int = 50
    min_wait: int = 10
    improve_atol: float = 0  # absolute tolerance for "improvement"
    improve_rtol: float = 0  # relative tolerance (fraction of baseline)

    # budgeted mode
    total_steps: int | None = None  # total planned steps (e.g., generations)

    # state
    p: float = field(init=False)
    last_switch_step: int = 0
    _hist: Deque[Tuple[int, float]] = field(init=False)  # (step, global_best_coh)
    _best_seen: float = field(init=False, default=float("inf"))
    _last_improve_step: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Initialize state and allocate a history buffer of size ``window+1``.

        The buffer stores pairs ``(step, global_best_coh)`` for the adaptive mode,
        ensuring that a baseline at approximately ``step - window`` is available as
        soon as there are ``window+1`` samples.
        """
        self.p = float(self.p0)
        # keep at least window+1 points so we can compare current vs ~window steps ago
        self._hist = deque(maxlen=max(3, self.window + 1))
        self._best_seen = float("inf")
        self._last_improve_step = 0

        # Validate required arguments for budgeted mode
        if self.mode == "budgeted":
            if self.total_steps is None or self.total_steps <= 0:
                raise ValueError(
                    "PScheduler(mode='budgeted') requires total_steps > 0."
                )

    def current_p(self) -> float:
        """Return the scheduler's current value of ``p``.

        This is a convenience accessor; it does not advance the scheduler state.
        """
        return self.p

    def _update_fixed(self, step: int) -> tuple[float, bool]:
        """Fixed mode: periodic ramp every `switch_every` steps."""
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

    def _update_budgeted(self, step: int, global_best_coh: float) -> tuple[float, bool]:
        """Budgeted mode: constant-window stall + budgeted multiplier
        to hit p_max by total_steps."""
        ts = cast(int, self.total_steps)
        # Enforce step bounds: caller must not exceed total_steps
        if step > ts:
            raise ValueError("PScheduler.update called with step > total_steps.")

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
        steps_left = max(0, ts - step)
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

    def _update_adaptive(self, step: int, global_best_coh: float) -> tuple[float, bool]:
        """Adaptive mode: ramp when no improvement beyond tolerance over
        the last `window` steps."""
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

    def update(
        self,
        *,
        step: int,
        global_best_coh: float,
    ) -> tuple[float, bool]:
        """Advance the scheduler by one step and (optionally) increase ``p``.

        Parameters
        ----------
        step : int
            Monotone counter (e.g., generation or iteration index). Can start at 0 or 1
            as long as it increases by 1 per call.
        global_best_coh : float
            The best (lowest) coherence value seen so far over the *entire run*.
            Used in ``mode="adaptive"`` and ``mode="budgeted"``.

        Returns
        -------
        (p, switched) : tuple[float, bool]
            ``p`` is the (possibly updated) value to use for the *next* step.
            ``switched`` is ``True`` iff a ramp occurred at this call.

        Behavior
        --------
        - Fixed mode: increase ``p`` by ``p_mult`` every ``switch_every`` steps since
          ``last_switch_step``; ignore coherence history.
        - Budgeted mode: ramp when no improvement for last ``window`` steps and
          at least ``window`` steps since last ramp. The multiplier is chosen so that,
          assuming the stall interval repeats,
          ``p`` reaches ``p_max`` by ``total_steps``.
          Calls with step > total_steps raise a ValueError.
        - Adaptive mode: append ``(step, global_best_coh)`` to the history. When at
          least ``window+1`` samples are available, take the oldest value as the
          baseline (≈ value at ``step - window``). If there has been no improvement
          beyond ``max(improve_atol, improve_rtol*|baseline|)`` over the last
          ``window`` steps and at least ``min_wait`` steps have elapsed since the
          last ramp, increase ``p`` by ``p_mult`` (capped at ``p_max``).

        Notes
        -----
        - "Improvement" is measured as ``baseline - current`` (positive means better
          because coherence is minimized).
        - Set ``improve_atol=improve_rtol=0`` to require *strict* stalling; increase
          them slightly to ignore numerical flicker.
        - To allow at most one ramp per window, set ``min_wait == window``.
        """
        if self.mode == "fixed":
            return self._update_fixed(step)
        if self.mode == "budgeted":
            return self._update_budgeted(step, global_best_coh)
        # default to adaptive
        return self._update_adaptive(step, global_best_coh)
