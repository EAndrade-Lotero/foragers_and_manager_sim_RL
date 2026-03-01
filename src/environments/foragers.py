import gymnasium as gym

from gymnasium import Env
from gymnasium.spaces import Box
from typing import Tuple, Dict, Optional, Any

from helper_classes import World

Info = Dict[None, None]
State = Tuple[float, float]  # (commission_rate, system_wealth)
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
            return Forager.optimal_remaining
        return Forager.endowment

    @staticmethod
    def get_estimated_harvest(state: State, num_foragers: int) -> float:
        _, wealth = state
        if Manager.gives_good_coordinates(state):
            return wealth
        return 0.0


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
        self.forager = Forager()

        self.render_mode: Optional[str] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> State_Info:
        super().reset(seed=seed)
        self.state = self.initial_state
        self.turn = 0
        return self.state, {}

    def step(self, action) -> Result:
        new_rate = self._parse_action(action)

        old_rate, old_wealth = self.state

        # ----- Dashed-box dynamics (lightweight proxy) -----
        # Manager invests (good coordinates) depends on (rate, wealth)
        manager_invests = Manager.gives_good_coordinates((new_rate, old_wealth))

        # Foragers decide to forage depends on (rate, wealth)
        foragers_go = Forager.goes_foraging((new_rate, old_wealth))
        n_active = self.num_foragers if foragers_go else 0

        # Harvest proxy:
        # - If manager invests, each active forager can realize wealth as harvest
        # - If not, harvest collapses (0), matching your current placeholder behavior
        if manager_invests and n_active > 0:
            total_harvest = old_wealth
        else:
            total_harvest = 0.0

        # Update "system wealth" (keep within [0,1] for your Box space)
        # Simple smooth dynamics: carry-over + harvest, clipped.
        new_wealth = 0.5 * old_wealth + 0.5 * total_harvest
        new_wealth = float(max(0.0, min(1.0, new_wealth)))

        new_state: State = (new_rate, new_wealth)
        self.state = new_state

        # Reward (continuous, learnable): total harvest
        reward = float(total_harvest)

        self.turn += 1
        done = self.get_done()

        return new_state, reward, done, False, {}

    def get_done(self) -> bool:
        return self.turn >= self.horizon

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
            print(f"Turn={self.turn}  State(rate,wealth)={self.state}")
            return None
        # rgb_array rendering not implemented in this minimal version
        return None