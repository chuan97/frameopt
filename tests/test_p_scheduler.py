import math

import pytest

from frameopt.models.p_scheduler import AdaptivePScheduler, FixedPScheduler

# -------------------------- FixedPScheduler tests ---------------------------


def test_fixed_no_ramp_when_disabled():
    sched = FixedPScheduler(p0=2.0, p_mult=10.0, p_max=1e9, switch_every=None)
    p = sched.current_p()
    switched_any = False
    for step in range(1, 10):
        p_next, switched = sched.update(step=step, global_best_coh=1.0)
        switched_any |= switched
        assert p_next == p
    assert not switched_any


def test_fixed_ramps_on_schedule_and_caps():
    sched = FixedPScheduler(p0=2.0, p_mult=4.0, p_max=50.0, switch_every=3)
    expected = [2.0, 2.0, 2.0, 8.0, 8.0, 8.0, 32.0, 32.0, 32.0, 50.0]
    seen = []
    for step in range(1, 11):
        p, switched = sched.update(step=step, global_best_coh=1.0)
        seen.append(p)
    assert seen == expected
    # Last ramp should have capped p to p_max exactly and not switch afterwards
    p_last, switched = sched.update(step=11, global_best_coh=1.0)
    assert p_last == 50.0 and switched is False


# ------------------------ AdaptivePScheduler tests -------------------------


def test_adaptive_constructor_validation():
    with pytest.raises(ValueError):
        AdaptivePScheduler(p0=2.0, p_max=100.0, total_steps=0, window=5)
    with pytest.raises(ValueError):
        AdaptivePScheduler(p0=2.0, p_max=100.0, total_steps=10, window=0)
    # Valid
    AdaptivePScheduler(p0=2.0, p_max=100.0, total_steps=10, window=3)


def test_adaptive_error_on_step_overflow():
    sched = AdaptivePScheduler(p0=2.0, p_max=100.0, total_steps=5, window=2)
    # Steps 1..5 are OK
    for step in range(1, 6):
        sched.update(step=step, global_best_coh=1.0)
    # Step 6 must raise
    with pytest.raises(ValueError):
        sched.update(step=6, global_best_coh=1.0)


def test_adaptive_first_ramp_timing_and_budgeted_progression():
    """
    Constant coherence (no improvements after step 1) and window=2:
    - First improvement recorded at step=1 (inf -> value).
    - Need both: spacing since last_switch_step (>=2) and stall since
    last_improve (>=2). => First ramp at step=3.
    Then repeated stall maintains ramps every `window` steps.
    Verify that per-ramp multipliers budget p to reach p_max by total_steps.
    """
    p0, p_max, total_steps, window = 2.0, 512.0, 10, 2
    sched = AdaptivePScheduler(
        p0=p0, p_max=p_max, total_steps=total_steps, window=window
    )

    # Keep coherence flat so there are no improvements after step 1
    coh = 1.0
    ramp_steps = []
    p_values = []

    for step in range(1, total_steps + 1):
        p, switched = sched.update(step=step, global_best_coh=coh)
        p_values.append(p)
        if switched:
            ramp_steps.append(step)

    # Expected ramp steps with window=2 under constant stall: 3, 5, 7, 9
    assert ramp_steps[:3] == [3, 5, 7]  # first three are deterministic
    # p progression should monotonically increase and reach p_max
    # by or before final step
    assert all(p_values[i] <= p_values[i + 1] for i in range(len(p_values) - 1))
    assert math.isclose(p_values[-1], p_max, rel_tol=0, abs_tol=1e-12)


def test_adaptive_improvement_defers_ramp():
    """
    With window=3, without improvements the first possible ramp would be at step=4.
    Inject an improvement at step=3 to reset last_improve; verify the first ramp
    is deferred to step=6 (needs stall & spacing).
    """
    sched = AdaptivePScheduler(p0=2.0, p_max=256.0, total_steps=10, window=3)
    # Steps 1..2: flat coherence (first step records improvement from inf)
    ramp_steps = []
    # Step 1: set baseline improvement
    sched.update(step=1, global_best_coh=1.0)
    # Step 2: no improvement
    sched.update(step=2, global_best_coh=1.0)
    # Step 3: improvement (drops coherence) -> resets stall clock
    sched.update(step=3, global_best_coh=0.9)
    # Steps 4..10
    for step in range(4, 11):
        _, switched = sched.update(step=step, global_best_coh=0.9)
        if switched:
            ramp_steps.append(step)

    # Without the improvement at step=3, first ramp would have been at step=4.
    # With improvement, need spacing>=3 since last_switch (0)
    # and stall>=3 since last_improve (3):
    # First ramp occurs at step=6.
    assert ramp_steps and ramp_steps[0] == 6


def test_adaptive_stops_switching_after_pmax():
    sched = AdaptivePScheduler(p0=2.0, p_max=32.0, total_steps=20, window=2)
    # Keep constant stall; the scheduler should reach p_max and then stop switching
    switched_after_pmax = False
    seen_pmax = False
    for step in range(1, 21):
        p, switched = sched.update(step=step, global_best_coh=1.0)
        if math.isclose(p, 32.0, rel_tol=0, abs_tol=1e-12):
            seen_pmax = True
            # Once at p_max, switched should be False on subsequent steps
        elif seen_pmax and switched:
            switched_after_pmax = True
            break
    assert seen_pmax and not switched_after_pmax
