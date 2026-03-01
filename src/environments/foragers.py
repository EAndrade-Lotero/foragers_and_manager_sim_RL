import numpy as np
import gymnasium as gym

from gymnasium import Env
from gymnasium.spaces import Box
from typing import Tuple, Dict, Optional, Any

Info = Dict[None, None]
State = Tuple[float, float]  # (commission_rate, system_wealth)
State_Info = Tuple[State, Info]
Result = Tuple[State, float, bool, bool, Info]


class Manager:
    budget: float = 0.1
    optimal_investment: float = 0.07

    def gives_good_coordinates(self, state: State) -> bool:
        if self.budget < self.optimal_investment:
            return False
        rate, wealth = state
        estimated_reward = rate * wealth
        return estimated_reward > self.budget

    def get_reward(self, state: State, harvest: float) -> float:
        rate = state[0]
        if self.gives_good_coordinates(state):
            remaining = self.budget - self.optimal_investment
            new_budget = remaining + rate * harvest
        else:
            new_budget = self.budget
        self.budget = new_budget
        return self.budget


class Forager:
    budget: float = 0.1
    optimal_investment: float = 0.07

    def __init__(self, manager: Manager, num_foragers: int) -> None:
        self.manager = manager
        self.num_foragers = num_foragers

    def goes_foraging(self, state: State) -> bool:
        rate, wealth = state
        estimated_reward = (1 - rate) * wealth
        return estimated_reward > self.budget

    def estimated_harvest(self, state: State) -> float:
        if self.manager.gives_good_coordinates(state):
            return np.clip(self.budget / 0.1, 0.0, 1.0) / self.num_foragers
        return 0.0

    def get_reward(self, state: State) -> float:
        rate = state[0]
        if self.goes_foraging(state):
            remaining = np.clip(self.budget - self.optimal_investment, 0.0, 1.0)
            print(f"\tForager invests: {self.optimal_investment}, Remaining budget: {remaining}")
            my_harvest = self.estimated_harvest(state)
            print(f"\tManager good?: {self.manager.gives_good_coordinates(state)}")
            print(f"\tEstimated harvest: {self.estimated_harvest(state)}")
            new_budget = (1 - rate) * my_harvest + remaining
            self.budget = new_budget
        return self.budget


class ForagersEnv(Env):
    """
    Action: new commission rate in [0, 1]
    State: (commission_rate, system_wealth)
    Reward: a continuous proxy for 'total harvest' (can be extended with fairness later)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        initial_rate: float = 0.5,
        initial_wealth: float = 0.5,
        num_foragers: int = 3,
        horizon: int = 10,
    ):
        self.action_space = Box(low=0.0, high=1.0, shape=(1,), dtype=float)
        self.observation_space = Box(low=0.0, high=1.0, shape=(2,), dtype=float)

        self.name = "Foragers"
        self.initial_state: State = (float(initial_rate), float(initial_wealth))
        self.state: State = self.initial_state

        self.num_foragers = int(num_foragers)
        self.horizon = int(horizon)
        self.turn = 0

        self.manager = Manager()
        self.forager = Forager(self.manager, self.num_foragers)

        self.render_mode: Optional[str] = None
        self.debug = True

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> State_Info:
        super().reset(seed=seed)
        self.state = self.initial_state
        self.turn = 0
        return self.state, {}

    def step(self, action: int) -> Result:
        new_rate = self._parse_action(action)

        _, old_wealth = self.state

        # Harvest proxy:
        # - If manager invests, each active forager can realize wealth as harvest
        # - If not, harvest collapses (0), matching your current placeholder behavior
        total_harvest = self.forager.estimated_harvest(self.state) * self.num_foragers
        if self.debug:
            print(f"Manager invests: {self.manager.gives_good_coordinates(self.state)}")
            print(f"Forager goes foraging: {self.forager.goes_foraging(self.state)}")
            print(f"Estimated total harvest: {total_harvest}")

        # Update "system wealth" (keep within [0,1] for your Box space)
        manager_budget = self.manager.get_reward((new_rate, old_wealth), total_harvest)
        foragers_budget = self.forager.get_reward((new_rate, old_wealth)) 
        self.forager.manager.budget = manager_budget  # Sync manager's budget with forager's knowledge
        if self.debug:
            print(f"Manager budget after update: {self.manager.budget}")
            print(f"Forager budget after update: {self.forager.budget}")

        # Simple smooth dynamics: carry-over + harvest, clipped.
        new_wealth = manager_budget + foragers_budget * self.num_foragers
        new_wealth = float(max(0.0, min(1.0, new_wealth)))
        if self.debug:
            print(f"New wealth before clipping: {manager_budget + foragers_budget}")
            print(f"New wealth after clipping: {new_wealth}")

        new_state: State = (new_rate, new_wealth)
        self.state = new_state

        # Reward 
        reward = self.get_reward(manager_budget, foragers_budget)

        self.turn += 1
        done = self.get_done()

        return new_state, reward, done, False, {}

    def get_done(self) -> bool:
        if self.manager.budget <= 0.0:
            return True
        if self.forager.budget <= 0.0:
            return True
        return self.turn >= self.horizon

    def get_reward(self, manager_budget: float, foragers_budget: float) -> float:
        wealth = manager_budget + foragers_budget
        inequality_penalty = (manager_budget - (foragers_budget / 3)) ** 2 / wealth if wealth > 0 else 0.0
        return manager_budget + foragers_budget - inequality_penalty     

    def _parse_action(self, action) -> float:
        # Gym Box actions usually arrive as array([x])
        try:
            if hasattr(action, "__len__") and not isinstance(action, (str, bytes)):
                a = float(action[0])
            else:
                a = float(action)
        except Exception:
            a = 0.0
        if a < 0.0:
            a = 0.0
        if a > 1.0:
            a = 1.0
        return a

    def render(self):
        if self.render_mode == "human" or self.render_mode is None:
            print(f"Turn={self.turn}  State={self.state} Manager budget={self.manager.budget:.2f} Forager budget={self.forager.budget:.2f}")
            return None
        # rgb_array rendering not implemented in this minimal version
        return None