import math
import numpy as np

from gymnasium import Env
from gymnasium.spaces import Box
from typing import Tuple, Dict, Any, Optional, List, Sequence


# --- Imports  ---
def _import_deps():
    try:
        # Typical: PYTHONPATH=src
        from utils.helper_classes import World, WealthTracker
        from utils.game_parameters import (
            NUM_FORAGERS,
            NUM_COINS,
            NUM_CENTROIDS,
            DISPERSION,
            COORDINATOR_ENDOWMENT,
            STARTING_SLIDERS,
        )
        return World, WealthTracker, NUM_FORAGERS, NUM_COINS, NUM_CENTROIDS, DISPERSION, COORDINATOR_ENDOWMENT, STARTING_SLIDERS
    except ImportError:
        try:
            from ..utils.helper_classes import World, WealthTracker
            from ..utils.game_parameters import (
                NUM_FORAGERS,
                NUM_COINS,
                NUM_CENTROIDS,
                DISPERSION,
                COORDINATOR_ENDOWMENT,
                STARTING_SLIDERS,
            )
            return World, WealthTracker, NUM_FORAGERS, NUM_COINS, NUM_CENTROIDS, DISPERSION, COORDINATOR_ENDOWMENT, STARTING_SLIDERS
        except ImportError:
            from helper_classes import World, WealthTracker
            from game_parameters import (
                NUM_FORAGERS,
                NUM_COINS,
                NUM_CENTROIDS,
                DISPERSION,
                COORDINATOR_ENDOWMENT,
                STARTING_SLIDERS,
            )
            return World, WealthTracker, NUM_FORAGERS, NUM_COINS, NUM_CENTROIDS, DISPERSION, COORDINATOR_ENDOWMENT, STARTING_SLIDERS


World, WealthTracker, NUM_FORAGERS, NUM_COINS, NUM_CENTROIDS, DISPERSION, COORDINATOR_ENDOWMENT, STARTING_SLIDERS = _import_deps()


Info = Dict[str, Any]
State = Tuple[float, float]          # (commission_rate, normalized_system_wealth in [0,1])
State_Info = Tuple[State, Info]
Result = Tuple[State, float, bool, bool, Info]


class Manager:
    endowment = 0.1
    optimal_remaining = 0.07

    @staticmethod
    def invests(rate: float, wealth: float) -> bool:
        return (rate * wealth) > Manager.endowment


class Forager:
    endowment = 0.1
    optimal_remaining = 0.07

    @staticmethod
    def invests(rate: float, wealth: float) -> bool:
        return ((1.0 - rate) * wealth) > Forager.endowment


# --- Utility metrics ---
def gini(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(np.maximum(arr, 0.0))
    n = arr.size
    cum = np.cumsum(arr)
    total = cum[-1]
    # Gini = (n+1 - 2*sum(cum)/total)/n
    return float((n + 1 - 2.0 * np.sum(cum) / total) / n)


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


class ForagersEnv(Env):
    """
    Full dashed-box simulator:
      manager investment -> coordinator view -> assignment -> forager reach -> bot harvest -> redistribution
    Research upgrades:
      - optional persistent world across turns
      - softmax assignment policy + distance constraint
      - reward includes inequality + instability penalties (governance framing)
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        initial_rate: float = 0.5,
        initial_wealth: float = 0.5,
        num_foragers: int = NUM_FORAGERS,
        max_turns: int = 10,
        seed: Optional[int] = None,
        # World config
        world_num_coins: int = NUM_COINS,
        world_num_centroids: int = NUM_CENTROIDS,
        world_distribution: str = "circular",
        world_dispersion: float = DISPERSION,
        persistent_world: bool = True,
        # Assignment + movement
        assignment_min_distance: int = 8,
        assignment_temperature: float = 0.8,
        reach_min: int = 1,
        reach_max: int = 3,
        # Governance objective weights
        lambda_gini: float = 1.0,
        mu_instability: float = 0.10,
        # Optional discrete action interface
        discrete_action_bins: Optional[int] = None,
    ):
        if int(num_foragers) != NUM_FORAGERS:
            raise ValueError(
                f"num_foragers={num_foragers} incompatible with WealthTracker (NUM_FORAGERS={NUM_FORAGERS})."
            )

        self.action_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.name = "Foragers"

        self.initial_state: State = (clip01(initial_rate), clip01(initial_wealth))
        self.state: State = self.initial_state

        self.num_foragers = int(num_foragers)
        self.max_turns = int(max_turns)
        self.turn = 0

        self._rng = np.random.default_rng(seed)

        self.world_num_coins = int(world_num_coins)
        self.world_num_centroids = int(world_num_centroids)
        self.world_distribution = str(world_distribution)
        self.world_dispersion = float(world_dispersion)
        self.persistent_world = bool(persistent_world)

        self.assignment_min_distance = int(max(0, assignment_min_distance))
        self.assignment_temperature = float(max(1e-6, assignment_temperature))
        self.reach_min = int(reach_min)
        self.reach_max = int(reach_max)

        self.lambda_gini = float(max(0.0, lambda_gini))
        self.mu_instability = float(max(0.0, mu_instability))

        self.discrete_action_bins = discrete_action_bins

        self._world: Optional[World] = None
        self._prev_rate: float = self.initial_state[0]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> State_Info:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.state = self.initial_state
        self.turn = 0
        self._prev_rate = self.initial_state[0]

        self._world = self._make_world() if self.persistent_world else None
        return self.state, {}

    def step(self, action: Any) -> Result:
        rate, wealth = self.state
        new_rate = self._action_to_rate(action)

        new_wealth, reward, info = self._simulate_step(new_rate, wealth, prev_rate=rate)

        self.state = (new_rate, new_wealth)
        self.turn += 1
        done = self.turn >= self.max_turns

        return self.state, reward, done, False, info

    # ----------------------------
    # Core dashed-box simulation
    # ----------------------------
    def _simulate_step(self, rate: float, wealth: float, prev_rate: float) -> Tuple[float, float, Info]:
        manager_invest = self._manager_investment(rate, wealth)
        forager_invest = self._forager_investment(rate, wealth)

        world = self._world if (self.persistent_world and self._world is not None) else self._make_world()

        manager_view = world.coordinator_view(manager_invest)

        positions = self._assign_positions_softmax(manager_view, world)

        reach = self._reach_from_investment(forager_invest)
        coins_collected, tiles_visited = world.reward_from_bots(positions, max_speed=reach)

        coins_collected = [int(c) for c in coins_collected]
        harvest = int(sum(coins_collected))

        # Redistribution
        sliders = {
            "overhead": float(rate),
            "wages": float(STARTING_SLIDERS["wages"]),
            "prerogative": STARTING_SLIDERS["prerogative"],
        }
        tracker = WealthTracker(coins_collected)
        tracker.initialize(sliders, investment=manager_invest)

        manager_w = float(tracker.get_coordinator_reward())
        foragers_w = [float(tracker.get_forager_reward(i)) for i in range(self.num_foragers)]

        total_w = float(manager_w + sum(foragers_w))

        # Governance objective:
        # - Efficiency: total wealth
        # - Fairness: penalize inequality across ALL participants (manager + foragers)
        # - Stability: penalize abrupt commission changes (institutional legitimacy / predictability)
        wealth_vector = [manager_w] + foragers_w
        ineq = gini(wealth_vector)

        instability = abs(rate - prev_rate)

        reward = float(total_w - self.lambda_gini * ineq - self.mu_instability * instability)

        # Normalize system wealth for observation
        max_possible = float(max(1, world.count_coins()) + COORDINATOR_ENDOWMENT)
        new_wealth = clip01(total_w / max_possible)

        info: Info = {
            "harvest": float(harvest),
            "coins_collected": [float(x) for x in coins_collected],
            "manager_investment": float(manager_invest),
            "forager_investment": float(forager_invest),
            "reach": float(reach),
            "manager_wealth": float(manager_w),
            "foragers_wealth": [float(x) for x in foragers_w],
            "total_wealth": float(total_w),
            "gini": float(ineq),
            "instability": float(instability),
            "reward": float(reward),
            "visited_paths": float(sum(len(v) for v in tiles_visited)),
            "persistent_world": bool(self.persistent_world),
        }

        return new_wealth, reward, info

    # ----------------------------
    # Components
    # ----------------------------
    def _make_world(self) -> World:
        return World(
            num_coins=self.world_num_coins,
            num_centroids=self.world_num_centroids,
            distribution=self.world_distribution,
            dispersion=self.world_dispersion,
        )

    def _action_to_rate(self, action: Any) -> float:
        a = float(np.asarray(action, dtype=float).reshape(-1)[0])

        if self.discrete_action_bins is not None:
            bins = int(self.discrete_action_bins)
            if bins <= 1:
                raise ValueError("discrete_action_bins must be > 1")
            # If it looks like an index (e.g., 0..bins-1), map to [0,1].
            # Otherwise treat as already-normalized.
            if a > 1.0:
                a = round(a) / float(bins - 1)

        return clip01(a)

    def _manager_investment(self, rate: float, wealth: float) -> float:
        if Manager.invests(rate, wealth):
            return clip01(rate * wealth + Manager.optimal_remaining)
        return 0.0

    def _forager_investment(self, rate: float, wealth: float) -> float:
        if Forager.invests(rate, wealth):
            return clip01((1.0 - rate) * wealth + Forager.optimal_remaining)
        return 0.0

    def _reach_from_investment(self, inv: float) -> int:
        # Smooth mapping then clamp to discrete reach range
        # (different from the other file’s exact formula)
        x = self.reach_min + (self.reach_max - self.reach_min) * (1.0 / (1.0 + math.exp(-4.0 * (inv - 0.5))))
        return int(np.clip(int(round(x)), self.reach_min, self.reach_max))

    def _assign_positions_softmax(self, manager_view: List[List[float]], world: World) -> List[Tuple[int, int]]:
        terrain = np.asarray(manager_view, dtype=float)
        flat = terrain.reshape(-1)

        # Softmax over scores / temperature
        scores = flat / self.assignment_temperature
        scores = scores - np.max(scores)
        probs = np.exp(scores)
        s = probs.sum()
        if s <= 0:
            probs = np.full_like(probs, 1.0 / probs.size)
        else:
            probs = probs / s

        chosen: List[Tuple[int, int]] = []
        attempts = 0

        # Sample without hard sorting; enforce min-distance
        while len(chosen) < self.num_foragers and attempts < 5000:
            idx = int(self._rng.choice(flat.size, p=probs))
            y, x = np.unravel_index(idx, terrain.shape)
            cand = (int(x), int(y))

            if all(world.get_distance(cand, prev) >= self.assignment_min_distance for prev in chosen):
                chosen.append(cand)

            # discourage resampling the same tile again
            probs[idx] *= 0.2
            ps = probs.sum()
            probs = probs / ps if ps > 0 else np.full_like(probs, 1.0 / probs.size)
            attempts += 1

        # fallback if spacing makes it hard
        while len(chosen) < self.num_foragers:
            x = int(self._rng.integers(0, world.width))
            y = int(self._rng.integers(0, world.height))
            chosen.append((x, y))

        return chosen

    def render(self):
        print(f"Turn={self.turn} state={self.state}")