import numpy as np

from gymnasium import Env
from gymnasium.spaces import Box
from typing import Tuple, Dict, Any, Optional, List

try:
    from .helper_classes import World, WealthTracker
    from .game_parameters import (
        NUM_FORAGERS,
        NUM_COINS,
        NUM_CENTROIDS,
        DISPERSION,
        COORDINATOR_ENDOWMENT,
        STARTING_SLIDERS,
    )
except ImportError:  # pragma: no cover - fallback for direct execution
    from helper_classes import World, WealthTracker
    from game_parameters import (
        NUM_FORAGERS,
        NUM_COINS,
        NUM_CENTROIDS,
        DISPERSION,
        COORDINATOR_ENDOWMENT,
        STARTING_SLIDERS,
    )

Info = Dict[str, Any]
State = Tuple[float, float]  # (rate, normalized_wealth)
State_Info = Tuple[State, Info]
Result = Tuple[State, float, bool, bool, Info]


class Manager:

    endowment = 0.1
    optimal_remaining = 0.07

    @staticmethod
    def gives_good_coordinates(state: State) -> bool:
        rate, wealth = state
        estimated_reward = rate * wealth
        return estimated_reward > Manager.endowment

    def get_estimated_reward(self, state: State) -> float:
        rate, wealth = state
        if self.gives_good_coordinates(state):
            return rate * wealth + Manager.optimal_remaining
        else:
            return Manager.endowment


class Forager:

    endowment = 0.1
    optimal_remaining = 0.07

    @staticmethod
    def goes_foraging(state: State) -> bool:
        rate, wealth = state
        estimated_reward = (1 - rate) * wealth
        return estimated_reward > Forager.endowment

    @staticmethod
    def get_estimated_reward(state: State) -> float:
        rate, wealth = state
        if Forager.goes_foraging(state):
            if Manager.gives_good_coordinates(state):
                return (1 - rate) * wealth + Forager.optimal_remaining
            else:
                return Forager.optimal_remaining
        else:
            return Forager.endowment

    @staticmethod
    def get_estimated_harvest(state: State, num_foragers: int) -> float:
        _, wealth = state
        if Manager.gives_good_coordinates(state):
            return wealth
        else:
            return 0.0


class ForagersEnv(Env):

    def __init__(
        self,
        initial_rate: float = 0.5,
        initial_wealth: float = 0.5,
        num_foragers: int = NUM_FORAGERS,
        max_turns: int = 10,
        fairness_weight: float = 1.0,
        reward_mode: str = "harvest_plus_fairness",
        discrete_action_bins: Optional[int] = None,
        seed: Optional[int] = None,
        world_num_coins: int = NUM_COINS,
        world_num_centroids: int = NUM_CENTROIDS,
        world_distribution: str = "circular",
        world_dispersion: float = DISPERSION,
        assignment_min_distance: int = 8,
    ):
        if int(num_foragers) != NUM_FORAGERS:
            raise ValueError(
                f"num_foragers={num_foragers} is incompatible with WealthTracker (NUM_FORAGERS={NUM_FORAGERS})."
            )

        self.action_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.name = "Foragers"

        self.initial_state = (
            float(np.clip(initial_rate, 0.0, 1.0)),
            float(np.clip(initial_wealth, 0.0, 1.0)),
        )
        self.state = self.initial_state
        self.num_foragers = int(num_foragers)
        self.max_turns = int(max_turns)
        self.fairness_weight = float(fairness_weight)
        allowed_reward_modes = {"wealth_minus_inequality", "harvest_plus_fairness"}
        if reward_mode not in allowed_reward_modes:
            raise ValueError(
                f"reward_mode must be one of {sorted(allowed_reward_modes)}, got {reward_mode!r}"
            )
        self.reward_mode = reward_mode
        self.discrete_action_bins = discrete_action_bins
        self.turn = 0
        self._rng = np.random.default_rng(seed)

        self.world_num_coins = int(world_num_coins)
        self.world_num_centroids = int(world_num_centroids)
        self.world_distribution = str(world_distribution)
        self.world_dispersion = float(world_dispersion)
        self.assignment_min_distance = int(max(0, assignment_min_distance))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> State_Info:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.state = self.initial_state
        self.turn = 0
        return self.state, {}

    def step(self, action: Any) -> Result:
        _, wealth = self.state
        new_rate = self._normalize_action(action)
        new_wealth, reward, info = self._simulate_round(new_rate, wealth)
        new_state = (new_rate, new_wealth)

        self.state = new_state
        self.turn += 1
        done = self.get_done()
        return new_state, reward, done, False, info

    def get_new_wealth(self) -> float:
        _, wealth = self.state
        new_rate = self.state[0]
        new_wealth, _, _ = self._simulate_round(new_rate, wealth)
        return new_wealth

    def get_done(self) -> bool:
        return self.turn >= self.max_turns

    def _normalize_action(self, action: Any) -> float:
        arr = np.asarray(action, dtype=float).reshape(-1)
        value = float(arr[0])

        if self.discrete_action_bins is not None:
            bins = int(self.discrete_action_bins)
            if bins <= 1:
                raise ValueError("discrete_action_bins must be > 1")
            # NOTE (vs Andrade baseline):
            # Keep backward-compatible semantics where values in [0, 1] are treated
            # as already-normalized commission rates. Only values > 1 are interpreted
            # as discrete indices and mapped to [0, 1]. This makes action=1.0 map to
            # rate=1.0 (not index 1 -> 0.1 when bins=11), which is intentional here.
            if value > 1.0:
                value = round(value) / float(bins - 1)

        return float(np.clip(value, 0.0, 1.0))

    def _manager_investment(self, state: State) -> float:
        rate, wealth = state
        if Manager.gives_good_coordinates(state):
            return float(np.clip(rate * wealth + Manager.optimal_remaining, 0.0, 1.0))
        return 0.0

    def _forager_investment(self, state: State) -> float:
        rate, wealth = state
        if Forager.goes_foraging(state):
            return float(np.clip((1.0 - rate) * wealth + Forager.optimal_remaining, 0.0, 1.0))
        return 0.0

    @staticmethod
    def _forager_reach(investment: float) -> int:
        # Reach levels in this codebase are discrete: 1, 2, or 3.
        return int(np.clip(int(1.5 + 2 * investment), 1, 3))

    def _build_world(self) -> World:
        return World(
            num_coins=self.world_num_coins,
            num_centroids=self.world_num_centroids,
            distribution=self.world_distribution,
            dispersion=self.world_dispersion,
        )

    def _assign_foragers(self, manager_view: List[List[float]], world: World) -> List[Tuple[int, int]]:
        terrain = np.asarray(manager_view, dtype=float)
        order = np.argsort(terrain, axis=None)[::-1]

        positions: List[Tuple[int, int]] = []
        for idx in order:
            y, x = np.unravel_index(idx, terrain.shape)
            candidate = (int(x), int(y))

            too_close = any(
                world.get_distance(candidate, previous) < self.assignment_min_distance
                for previous in positions
            )
            if too_close:
                continue

            positions.append(candidate)
            if len(positions) >= self.num_foragers:
                break

        while len(positions) < self.num_foragers:
            x = int(self._rng.integers(0, world.width))
            y = int(self._rng.integers(0, world.height))
            positions.append((x, y))

        return positions

    def _simulate_round(self, rate: float, wealth: float) -> Tuple[float, float, Info]:
        decision_state = (rate, wealth)

        # 1) Manager investment in information.
        manager_investment = self._manager_investment(decision_state)

        # 2) Build world and manager's partial view.
        world = self._build_world()
        manager_view = world.coordinator_view(manager_investment)

        # 3) Assignment of foragers to likely coin-rich coordinates.
        positions = self._assign_foragers(manager_view, world)

        # 4) Forager investment determines movement reach.
        forager_investment = self._forager_investment(decision_state)
        max_speed = self._forager_reach(forager_investment)

        # 5) Foraging/harvest via bot simulation.
        coins_collected, tiles_visited = world.reward_from_bots(positions, max_speed=max_speed)
        coins_collected = [int(c) for c in coins_collected]
        harvest = int(sum(coins_collected))

        # 6) Redistribution into manager/foragers wealth.
        sliders = {
            "overhead": float(rate),
            "wages": float(STARTING_SLIDERS["wages"]),
            "prerogative": STARTING_SLIDERS["prerogative"],
        }
        wealth_tracker = WealthTracker(coins_collected)
        wealth_tracker.initialize(sliders, investment=manager_investment)

        manager_wealth = float(wealth_tracker.get_coordinator_reward())
        foragers_wealth = [
            float(wealth_tracker.get_forager_reward(i)) for i in range(self.num_foragers)
        ]

        total_wealth = float(manager_wealth + sum(foragers_wealth))

        # Between-role inequality: manager vs average forager wealth.
        forager_mean_wealth = float(np.mean(foragers_wealth))
        role_inequality = float(np.std([manager_wealth, forager_mean_wealth]))

        # Keep both formulations available; default is paper-safe for Edgar.
        fairness = float(-self.fairness_weight * role_inequality)
        if self.reward_mode == "wealth_minus_inequality":
            reward = float(total_wealth - self.fairness_weight * role_inequality)
        else:
            reward = float(harvest + fairness)

        max_possible_wealth = float(max(1, world.count_coins()) + COORDINATOR_ENDOWMENT)
        new_wealth = float(np.clip(total_wealth / max_possible_wealth, 0.0, 1.0))

        info: Info = {
            "harvest": float(harvest),
            "manager_investment": float(manager_investment),
            "forager_investment": float(forager_investment),
            "max_speed": float(max_speed),
            "manager_wealth": manager_wealth,
            "forager_mean_wealth": forager_mean_wealth,
            "total_wealth": total_wealth,
            "fairness": fairness,
            "role_inequality": role_inequality,
            "coins_collected_0": float(coins_collected[0]) if len(coins_collected) > 0 else 0.0,
            "coins_collected_1": float(coins_collected[1]) if len(coins_collected) > 1 else 0.0,
            "visited_paths": float(sum(len(v) for v in tiles_visited)),
        }
        return new_wealth, reward, info

    def render(self):
        print(f"State: {self.state}")
