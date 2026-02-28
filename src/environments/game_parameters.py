# Module with the game parameters
import numpy as np

NUM_FORAGERS = 2
NUM_ROUNDS = 2
NUMBER_OF_TRIALS = 3
MAX_NODES_PER_CHAIN = 12

WORLD_WIDTH = 80
WORLD_HEIGHT = 80
NUM_CENTROIDS = 20
NUM_COINS = 150
DISPERSION = 1

INITIAL_WEALTH = [0, 0, 0]
INITIAL_POSITIONS = [
    (int(WORLD_WIDTH / 2), int(WORLD_HEIGHT / 2)),
    (int(WORLD_WIDTH / 2), int(WORLD_HEIGHT / 2)),
    (int(WORLD_WIDTH / 2), int(WORLD_HEIGHT / 2)),
    (int(WORLD_WIDTH / 2), int(WORLD_HEIGHT / 2)),
    (int(WORLD_WIDTH / 2), int(WORLD_HEIGHT / 2)),
    (int(WORLD_WIDTH/2), WORLD_HEIGHT - 1),
    (WORLD_WIDTH - 1, int(WORLD_HEIGHT / 2)),
    (0, 0),
    (WORLD_WIDTH - 1, WORLD_HEIGHT - 1),
]
INITIAL_POSITIONS = INITIAL_POSITIONS[:NUM_FORAGERS]

COORDINATOR_ENDOWMENT = 10
STARTING_SLIDERS = {
    "overhead":0.05,
    "wages":0,
    "prerogative":2
}

# ----------------------------------------------
# These parameters determine the different seeds
# ----------------------------------------------
OVERHEADS = [0.05, 0.5, 0.95]
POWER_ROLES = ["coordinator", "forager"]
WORLD_PATHS = [
    "static/map1.json",
    "static/map2.json",
    # "static/map1.json",
    "static/map3.json",
    # "static/map2.json",
]
# OVERHEADS = [0.05]
# POWER_ROLES = ["forager"]
# WORLD_PATHS = [
#     "static/map1.json",
#     # "static/map2.json",
#     # "static/map1.json",
#     # "static/map3.json",
#     # "static/map2.json",
# ]
# ----------------------------------------------

FUEL_PER_MOVE = 1
MAX_STEPS = int(100 / FUEL_PER_MOVE)
MAX_MOVEMENT = lambda reach: int(reach * MAX_STEPS)
# COLLECTION_CHANCE = lambda reach: 0.4 + 1 / (2 * reach)
def coin_collection_chance(reach:int):
    if reach == 1:
        return 1.0
    elif reach == 2:
        return 0.5
    elif reach == 3:
        return 0.25
    else:
        return 0.0

COLLECTION_CHANCE = lambda reach: coin_collection_chance(reach)

ASSETS_PATHS = {
    "coin_url": "static/coin.png",
    "forager_url": "static/forager.png",
    "click_move_url": "static/click_move.gif",
    "gear_1_url": "static/gear_1_3x3_window.png",
    "gear_2_url": "static/gear_2_5x5_window.png",
    "gear_3_url": "static/gear_3_7x7_window.png",
    "gears_url": "static/gears.png",
    "foraging_url": "static/two_foragers.gif",
    "drag_drop_url": "static/drag_and_drop.gif",
    "investment_url": "static/slider_drag_assignment.gif",
    "minimap_url": "static/minimap.png",
    "slider_url": "static/slider.png",
    "coin_collected_url": "static/coin_collected.mp3",
    "reach_1_url": "static/reach_gear_1.png",
    "reach_2_url": "static/reach_gear_2.png",
    "reach_3_url": "static/reach_gear_3.png",
    "investment_test_url": "static/investment_test.json",
}

RNG = np.random.default_rng(42)

TOTAL_DURATION = 15
per_hour_rate = 10
PER_MINUTE_PAYMENT = per_hour_rate / 60
# TOTAL_DURATION = len(WORLD_PATHS) * 5
# per_hour_rate = 10
# PER_MINUTE_PAYMENT = per_hour_rate / 60

REWARD_SCALING_FACTOR = 0.01
MAX_BONUS_REWARD = 7
