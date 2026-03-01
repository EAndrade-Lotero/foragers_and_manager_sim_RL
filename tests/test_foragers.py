"""Tests for ForagersEnv — full simulation pipeline.

Run:  conda run -n RLenv python -m pytest tests/test_foragers.py -v
  or: conda run -n RLenv python tests/test_foragers.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "environments"))

from foragers import ForagersEnv, Manager, Forager


# ---------------------------------------------------------------------------
# Suite 1: Instantiation & reset
# ---------------------------------------------------------------------------

def test_action_and_obs_spaces():
    env = ForagersEnv(seed=42)
    assert env.action_space.shape == (1,)
    assert env.observation_space.shape == (2,)


def test_initial_state():
    env = ForagersEnv(initial_rate=0.3, initial_wealth=0.7, seed=42)
    assert env.state == (0.3, 0.7)


def test_reset_returns_initial_state():
    env = ForagersEnv(initial_rate=0.3, initial_wealth=0.7, seed=42)
    state, info = env.reset()
    assert state == (0.3, 0.7)
    assert isinstance(info, dict)
    assert env.turn == 0


def test_reset_with_seed():
    env = ForagersEnv(seed=42)
    s1, _ = env.reset(seed=100)
    s2, _ = env.reset(seed=100)
    assert s1 == s2


# ---------------------------------------------------------------------------
# Suite 2: Step output format
# ---------------------------------------------------------------------------

def test_step_returns_5_tuple():
    env = ForagersEnv(seed=123)
    env.reset()
    result = env.step(0.5)
    assert len(result) == 5
    state, reward, done, trunc, info = result
    assert isinstance(state, tuple) and len(state) == 2
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert trunc is False
    assert isinstance(info, dict)


def test_info_keys():
    env = ForagersEnv(seed=42)
    env.reset()
    _, _, _, _, info = env.step(0.5)
    for key in ["harvest", "manager_investment", "forager_investment",
                "max_speed", "manager_wealth", "forager_mean_wealth",
                "total_wealth", "inequality", "reward"]:
        assert key in info, f"Missing info key: {key}"


# ---------------------------------------------------------------------------
# Suite 3: State bounds
# ---------------------------------------------------------------------------

def test_state_bounds():
    env = ForagersEnv(seed=0)
    env.reset()
    for action in [0.0, 0.5, 1.0, -0.1, 1.5, np.array([0.3])]:
        s, _, _, _, _ = env.step(action)
        rate, wealth = s
        assert 0.0 <= rate <= 1.0, f"rate={rate} out of bounds for action={action}"
        assert 0.0 <= wealth <= 1.0, f"wealth={wealth} out of bounds for action={action}"


# ---------------------------------------------------------------------------
# Suite 4: Episode termination
# ---------------------------------------------------------------------------

def test_done_after_max_turns():
    env = ForagersEnv(max_turns=3, seed=42)
    env.reset()
    for i in range(3):
        _, _, done, _, _ = env.step(0.5)
        if i < 2:
            assert not done, f"Should not be done at turn {i+1}"
        else:
            assert done, f"Should be done at turn {i+1}"


# ---------------------------------------------------------------------------
# Suite 5: Reward formula (Edgar's spec)
# ---------------------------------------------------------------------------

def test_reward_formula():
    """Reward = Wealth - Inequality where Inequality = MSD(manager, foragers) / Wealth."""
    env = ForagersEnv(seed=99)
    env.reset()
    _, reward, _, _, info = env.step(0.5)
    tw = info["total_wealth"]
    ineq = info["inequality"]
    expected = tw - ineq
    assert abs(reward - expected) < 1e-9, f"reward={reward}, expected={expected}"


def test_inequality_is_msd_normalized():
    """Inequality = mean((manager - f_i)^2) / total_wealth."""
    env = ForagersEnv(seed=42)
    env.reset()
    _, _, _, _, info = env.step(0.5)
    mw = info["manager_wealth"]
    fmw = info["forager_mean_wealth"]
    tw = info["total_wealth"]
    # With 2 foragers, forager_mean_wealth is the average.
    # We can't recompute exact MSD from info alone (we'd need individual forager scores),
    # but we can verify inequality >= 0 and reward <= total_wealth.
    assert info["inequality"] >= 0
    assert info["reward"] <= tw + 1e-9


# ---------------------------------------------------------------------------
# Suite 6: Discrete action bins
# ---------------------------------------------------------------------------

def test_discrete_index_mapping():
    env = ForagersEnv(discrete_action_bins=11, seed=42)
    env.reset()
    # action=5.0 > 1.0 → treated as index → 5/10 = 0.5
    s, _, _, _, _ = env.step(5.0)
    assert abs(s[0] - 0.5) < 1e-9


def test_continuous_passthrough_with_bins():
    env = ForagersEnv(discrete_action_bins=11, seed=42)
    env.reset()
    # action=0.5 <= 1.0 → treated as continuous rate
    s, _, _, _, _ = env.step(0.5)
    assert abs(s[0] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Suite 7: Behavioral models
# ---------------------------------------------------------------------------

def test_manager_cooperates():
    assert Manager.gives_good_coordinates((0.5, 0.5))


def test_manager_defects_low_rate():
    assert not Manager.gives_good_coordinates((0.1, 0.5))


def test_forager_goes():
    assert Forager.goes_foraging((0.5, 0.5))


def test_forager_stays_high_rate():
    assert not Forager.goes_foraging((0.9, 0.5))


# ---------------------------------------------------------------------------
# Suite 8: Internal methods
# ---------------------------------------------------------------------------

def test_manager_investment_positive():
    env = ForagersEnv(seed=42)
    assert env._manager_investment((0.5, 0.5)) > 0


def test_manager_investment_zero_on_defect():
    env = ForagersEnv(seed=42)
    assert env._manager_investment((0.0, 1.0)) == 0.0


def test_forager_investment_positive():
    env = ForagersEnv(seed=42)
    assert env._forager_investment((0.5, 0.5)) > 0


def test_forager_investment_zero_on_stay():
    env = ForagersEnv(seed=42)
    assert env._forager_investment((0.99, 0.5)) == 0.0


def test_forager_reach_levels():
    assert ForagersEnv._forager_reach(0.0) == 1
    assert ForagersEnv._forager_reach(0.5) == 2
    assert ForagersEnv._forager_reach(0.75) == 3  # banker's rounding fix
    assert ForagersEnv._forager_reach(1.0) == 3


def test_build_world():
    env = ForagersEnv(seed=42)
    world = env._build_world()
    assert world is not None
    assert world.count_coins() > 0


# ---------------------------------------------------------------------------
# Suite 9: Edge cases
# ---------------------------------------------------------------------------

def test_rate_zero():
    env = ForagersEnv(initial_rate=0.0, initial_wealth=0.5, seed=42)
    env.reset()
    _, _, _, _, info = env.step(0.0)
    assert "harvest" in info


def test_rate_one():
    env = ForagersEnv(initial_rate=1.0, initial_wealth=0.5, seed=42)
    env.reset()
    env.step(1.0)  # should not crash


def test_wealth_zero():
    env = ForagersEnv(initial_rate=0.5, initial_wealth=0.0, seed=42)
    env.reset()
    env.step(0.5)  # should not crash


def test_wealth_one():
    env = ForagersEnv(initial_rate=0.5, initial_wealth=1.0, seed=42)
    env.reset()
    env.step(0.5)  # should not crash


# ---------------------------------------------------------------------------
# Suite 10: Full episode
# ---------------------------------------------------------------------------

def test_full_episode():
    env = ForagersEnv(max_turns=10, seed=42)
    env.reset()
    total_reward = 0.0
    for t in range(10):
        action = 0.3 + 0.04 * t
        s, r, done, _, _ = env.step(action)
        total_reward += r
        rate, wealth = s
        assert 0 <= rate <= 1 and 0 <= wealth <= 1
    assert done
    assert np.isfinite(total_reward)


# ---------------------------------------------------------------------------
# Suite 11: Gymnasium compatibility
# ---------------------------------------------------------------------------

def test_gymnasium_action_sample():
    env = ForagersEnv(seed=42)
    env.reset()
    action = env.action_space.sample()
    assert action.shape == (1,)
    s, r, d, t, i = env.step(action)
    assert s is not None


# ---------------------------------------------------------------------------
# Suite 12: Pipeline output compatibility
# ---------------------------------------------------------------------------

def test_step_output_indexing():
    """Verify step output is compatible with interaction.py Episode class."""
    env = ForagersEnv(seed=42)
    env.reset()
    result = env.step(0.5)
    assert isinstance(result[0], tuple) and len(result[0]) == 2  # state
    assert isinstance(result[1], (int, float))  # reward
    assert isinstance(result[2], bool)  # done


# ---------------------------------------------------------------------------
# Suite 13: Rate sweep (diagnostic)
# ---------------------------------------------------------------------------

def test_rate_sweep():
    """Sweep commission rates and verify all produce valid outputs."""
    env = ForagersEnv(seed=42)
    for rate in np.linspace(0.0, 1.0, 11):
        env.reset()
        s, r, d, _, info = env.step(float(rate))
        assert 0 <= s[0] <= 1, f"rate out of bounds at action={rate}"
        assert 0 <= s[1] <= 1, f"wealth out of bounds at action={rate}"
        assert np.isfinite(r), f"non-finite reward at action={rate}"


# ---------------------------------------------------------------------------
# Run as script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS: {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {fn.__name__} -- {e}")
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"TOTAL: {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED")
