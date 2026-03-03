import numpy as np
import gymnasium as gym

from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from typing import (Tuple, Dict, Optional, Any, Union)

Info = Dict[None, None]
State = np.ndarray[np.float32]  # (commission_rate, system_wealth)
State_Info = Tuple[State, Info]
Result = Tuple[State, float, bool, bool, Info]


class Manager:
    max_budget: float = 0.2
    optimal_investment: float = 0.07

    def __init__(self) -> None:
        self.budget: float = self.max_budget / 2

    def gives_good_coordinates(self, state: State) -> bool:
        if self.budget < self.optimal_investment:
            return False
        rate, wealth = state
        estimated_reward = rate * wealth
        # print(f"Manager estimated reward: {rate} * {wealth} = {estimated_reward}, Budget: {self.budget}")
        return estimated_reward > self.budget

    def get_reward(self, state: State, harvest: float) -> float:
        rate = state[0]
        if self.gives_good_coordinates(state):
            remaining = self.budget - self.optimal_investment
            new_budget = remaining + rate * harvest
        else:
            new_budget = self.budget
        self.budget = np.clip(new_budget, 0.0, self.max_budget)
        return self.budget


class Forager:
    max_budget: float = 0.2
    optimal_investment: float = 0.07

    def __init__(self, manager: Manager, num_foragers: int) -> None:
        self.manager = manager
        self.num_foragers = num_foragers
        self.budget: float = self.max_budget / 2

    def goes_foraging(self, state: State) -> bool:
        rate, wealth = state
        estimated_reward = (1 - rate) * wealth
        # print(f"\tForager estimated reward: {(1 - rate)} * {wealth} = {estimated_reward}, Budget: {self.budget}")
        return estimated_reward > self.budget

    def estimated_harvest(self, state: State) -> float:
        # print(f"=====>: Manager budget: {self.manager.budget}  Wealth * Rate: {np.prod(state)}")
        if self.goes_foraging(state) and self.manager.gives_good_coordinates(state):
            return np.clip(self.budget / 0.1, 0.0, 1.0) / self.num_foragers
        return 0.0

    def get_reward(self, state: State) -> float:
        rate = state[0]
        if self.goes_foraging(state):
            remaining = np.clip(self.budget - self.optimal_investment, 0.0, 1.0)
            # print(f"\tForager invests: {self.optimal_investment}, Remaining budget: {remaining}")
            my_harvest = self.estimated_harvest(state)
            # print(f"\tManager good?: {self.manager.gives_good_coordinates(state)}")
            # print(f"\tEstimated harvest: {self.estimated_harvest(state)}")
            new_budget = (1 - rate) * my_harvest + remaining
            self.budget = np.clip(new_budget, 0.0, self.max_budget)
        return self.budget


class ForagersEnv(Env):
    """
    Action: new commission rate in [0, 1]
    State: (commission_rate, system_wealth)
    Reward: a continuous proxy for wealth and fairness
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        initial_rate: float = 0.5,
        initial_wealth: float = 1.0,
        num_foragers: int = 3,
        num_discrete_actions: int = 11,
        _max_episode_steps: int = 10,
    ):
        self.action_space = Discrete(n=num_discrete_actions)  # 0 to num_discrete_actions-1 (representing 0.0 to 1.0 in steps of 1/num_discrete_actions)
        self.observation_space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.initial_state = np.array([np.float32(initial_rate), np.float32(initial_wealth)])
        self.state = self.initial_state

        self.name = "Foragers"
        self.num_foragers = int(num_foragers)
        self._max_episode_steps = int(_max_episode_steps)
        self.turn = 0
        self.render_mode: Optional[str] = None
        self.debug = False
        self.super_debug = False

        self.manager = Manager()
        self.forager = Forager(Manager(), self.num_foragers)


    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> State_Info:
        super().reset(seed=seed)
        # self.state = self.initial_state
        self.state = self._choose_random_initial_state()
        self.manager = Manager()
        self.forager = Forager(Manager(), self.num_foragers)
        self.turn = 0
        return self.state, {}
    
    def _choose_random_initial_state(self) -> State:
        # commision_rate = np.random.choice([0.0, 0.5, 1.0])  # Randomly choose from discrete rates
        # wealth = np.random.choice([0.0, 0.5, 1.0])  # Randomly choose from discrete wealth levels
        # return np.array([commision_rate, wealth], dtype=np.float32)
        return self.initial_state
    
    def step(self, action: float) -> Result:
        if self.super_debug:
            self.render()

        new_rate = self._parse_action(action)

        _, old_wealth = self.state
        transition_state = np.array([new_rate, old_wealth], dtype=np.float32)

        # Harvest proxy:
        # - If manager invests, each active forager can realize wealth as harvest
        # - If not, harvest collapses (0), matching your current placeholder behavior
        total_harvest = self.forager.estimated_harvest(transition_state) * self.num_foragers
        if self.debug:
            print(f"Manager invests: {self.manager.gives_good_coordinates(transition_state)}")
            print(f"Forager goes foraging: {self.forager.goes_foraging(transition_state)}")
            print(f"Total harvest: {total_harvest}")

        # Update "system wealth" (keep within [0,1] for your Box space)
        manager_budget = self.manager.get_reward(transition_state, total_harvest)
        foragers_budget = self.forager.get_reward(transition_state) 
        self.forager.manager.budget = manager_budget  # Sync manager's budget with forager's knowledge
        if self.debug:
            print(f"\tManager budget after update: {self.manager.budget}")
            print(f"\tForager budget after update: {self.forager.budget}")

        # Simple smooth dynamics: carry-over + harvest, clipped.
        new_wealth = manager_budget + foragers_budget * self.num_foragers
        new_wealth = np.clip(new_wealth, 0.0, 1.0)
        if self.debug:
            # print(f"New wealth before clipping: {manager_budget + foragers_budget * self.num_foragers}")
            print(f"New wealth clipping: {new_wealth}")

        new_state = np.array([new_rate, new_wealth], dtype=np.float32)
        self.state = new_state

        # Reward 
        reward = float(self.get_reward(manager_budget, foragers_budget))

        self.turn += 1
        done, truncated = self.get_done()

        if self.super_debug:
            self.render()

        return new_state, reward, done, truncated, {}

    def get_done(self) -> Tuple[bool, bool]:
        done = False
        truncated = False
        if self.manager.budget <= 0.0:
            done = True
        if self.forager.budget <= 0.0:
            done = True
        if self.turn >= self._max_episode_steps:
            truncated = True
        return done, truncated

    def get_reward(self, manager_budget: float, foragers_budget: float) -> float:
        numerator_wealth = manager_budget + foragers_budget  * self.num_foragers
        denominator_wealth = self.manager.max_budget + self.forager.max_budget * self.num_foragers
        wealth = numerator_wealth / denominator_wealth
        inequality_penalty = (manager_budget - (foragers_budget / 3)) ** 2 / wealth if wealth > 0 else 0.0
        reward = wealth - inequality_penalty
        if self.debug:
            print(f"Reward calculation: Wealth={wealth}, Inequality penalty={inequality_penalty}")
            print(f"Reward: {reward}")
        assert isinstance(reward, Union[float, np.float32]), f"Expected reward to be a float, got {type(reward)}"
        return reward     

    def render(self):
        if self.render_mode == "human" or self.render_mode is None:
            print(f"Turn={self.turn} --- State={self.state} --- Manager budget={self.manager.budget:.2f} --- Forager budget={self.forager.budget:.2f}")
            return None
        # rgb_array rendering not implemented in this minimal version
        return None
    
    @staticmethod
    def _normalize(value: float) -> float:
        return (value + 1.0) / 2.0
    
    @staticmethod
    def _denormalize(value: float) -> float:
        return value * 2.0 - 1.0
    
    def _parse_action(self, action: Union[int, np.ndarray]) -> float:
        # Gym Box actions usually arrive as array([x])
        try:
            if isinstance(action, np.ndarray):
                a = float(action[0])
            else:
                a = float(action)
        except Exception:
            a = 0.0
        return np.clip(a, 0.0, 1.0)


class DiscreteForagersEnv(ForagersEnv):
    """
    Action: new commission rate in {0, 0.1, 0.2, ..., 1.0}
    State: (commission_rate, system_wealth)
    Reward: a continuous proxy for wealth and fairness
    """
    
    def _parse_action(self, action: int) -> float:
        # Gym Box actions usually arrive as array([x])
        if isinstance(action, np.ndarray):
                a = action[0]
        else:
            a = action
        assert a in range(self.action_space.n), f"Action {a} is out of bounds for Discrete action space with n={self.action_space.n}"
        # Convert discrete action (0-num_actions) to continuous value (0.0-1.0)
        a = a / (self.action_space.n - 1)  # Assuming action_space.n is the number of discrete actions
        return np.clip(a, 0.0, 1.0)
