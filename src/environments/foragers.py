import gymnasium as gym

from gymnasium import Env
from gymnasium.spaces import Box
from typing import Tuple, Dict

from helper_classes import World

Info = Dict[None, None]
State = Tuple[float, float]  # (rate, wealth)
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
    
    def __init__(self, initial_rate: float = 0.5, initial_wealth: float = 0.5):
        self.action_space = Box(low=0, high=1, shape=(1,), dtype=float)
        self.observation_space = Box(low=0, high=1, shape=(2,), dtype=float)
        self.name = 'Foragers'
        self.initial_state = (initial_rate, initial_wealth)
        self.state = self.initial_state
        self.num_foragers = 3
        self.turn = 0
        
    def reset(self) -> State_Info:
        self.state = self.initial_state
        self.turn = 0
        return self.state, {}
    
    def step(self, action) -> Result:
        rate, wealth = self.state
        new_wealth = self.get_new_wealth()
        new_state = (action, new_wealth)
        self.state = new_state
        reward = new_wealth # Agent maximizes the wealth
        self.turn += 1
        done = self.get_done()
        return new_state, reward, done, False, {}

    def get_new_wealth(self) -> float:
        if Manager.gives_good_coordinates(self.state):
            return 1.0
        return 0.0  # Example calculation

    def render(self):
        print(f'State: {self.state}')
