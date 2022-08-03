import numpy as np

# ====================================== PREDICATE DEFINITIONS ======================================
def _agent_crouches(state):
    return state["objects"]["agent"]["is_crouching"]

def _agent_holds(state, obj):
    return state["objects"]["agent"]["holding"] == obj["name"]

def _in(state, obj1, obj2):
    return all([obj1["position"][i] == obj2["position"][i] for i in range(3)]) 

def _in_motion(state, obj):
    return np.linalg.norm(obj["velocity"]) > 0
# ===================================================================================================

# ====================================== FUNCTION DEFINITIONS =======================================

def _distance(obj1, obj2):
    return np.linalg.norm(np.array(obj1["position"]) - np.array(obj2["position"]))

# ===================================================================================================

OBJECTS_BY_TYPE = {"ball": ["blue-dodgeball-1", "red-dodgeball-1", "pink-dodgeball-1"],
                   "bin": ["hexagonal-bin-1", "hexagonal-bin-2"],
                   "wall": ["left-wall-1", "right-wall-1", "front-wall-1", "back-wall-1"],
                   "agent": ["agent"]}

PREDICATE_LIBRARY = {"agent_crouches": _agent_crouches,
                     "agent_holds": _agent_holds,
                     "in": _in,
                     "in_motion": _in_motion}

FUNCTION_LIBRARY = {"distance": _distance}

# ===================================================================================================

SAMPLE_TRAJECTORY = [# Starting state: nothing held
                     {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [4, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "red"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [10, 10, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "agent": {"name": "agent", "position": [0, 0, 0], "velocity": [0, 0, 0],
                                         "is_crouching": False, "holding": None, "objectType": "agent"},
                              },

                      "game_start": True,
                      "game_over": False},

                     # State 2: agent moves closer to red ball
                     {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [4, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "red"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [10, 10, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "agent": {"name": "agent", "position": [3, 3, 0], "velocity": [0, 0, 0],
                                         "is_crouching": False, "holding": None, "objectType": "agent"},
                              },

                      "game_start": False,
                      "game_over": False},

                     # State 3: red ball picked up
                     {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [4, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "red"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [10, 10, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "agent": {"name": "agent", "position": [3, 3, 0], "velocity": [0, 0, 0],
                                         "is_crouching": False, "holding": "red-dodgeball-1", "objectType": "agent"},
                              },

                      "game_start": False,
                      "game_over": False},

                     # State 4: no changes
                     {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [4, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "red"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [10, 10, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "agent": {"name": "agent", "position": [3, 3, 0], "velocity": [0, 0, 0],
                                         "is_crouching": False, "holding": "red-dodgeball-1", "objectType": "agent"},
                              },

                      "game_start": False,
                      "game_over": False},

                     # State 5: ball released towards bin-1
                     {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [4, 4, 0],
                                                    "velocity": [2, 2, 0], "objectType": "ball",
                                                    "color": "red"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [10, 10, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "agent": {"name": "agent", "position": [3, 3, 0], "velocity": [0, 0, 0],
                                         "is_crouching": False, "holding": None, "objectType": "agent"},
                              },

                      "game_start": False,
                      "game_over": False},

                     # State 6: ball travels towards bin-1
                     {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [6, 6, 0],
                                                    "velocity": [2, 2, 0], "objectType": "ball",
                                                    "color": "red"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [10, 10, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "agent": {"name": "agent", "position": [3, 3, 0], "velocity": [0, 0, 0],
                                         "is_crouching": False, "holding": None, "objectType": "agent"},
                              },

                      "game_start": False,
                      "game_over": False},

                     # State 7: ball travels towards bin-1, agent crouches and picks up bin-1
                     {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [8, 8, 0],
                                                    "velocity": [2, 2, 0], "objectType": "ball",
                                                    "color": "red"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [10, 10, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "agent": {"name": "agent", "position": [3, 3, 0], "velocity": [0, 0, 0],
                                         "is_crouching": True, "holding": "hexagonal-bin-1", "objectType": "agent"},
                              },

                      "game_start": False,
                      "game_over": False},

                     # State 8: ball reaches bin-1 but has velocity still, agent stands up and picks up pink ball
                     {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [10, 10, 0],
                                                    "velocity": [0.2, 0.2, 0], "objectType": "ball",
                                                    "color": "red"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [10, 10, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "agent": {"name": "agent", "position": [3, 3, 0], "velocity": [0, 0, 0],
                                         "is_crouching": False, "holding": "pink-dodgeball-1", "objectType": "agent"},
                              },

                      "game_start": False,
                      "game_over": False},

                     # State 9: ball stops moving, game ends
                     {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [10, 10, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "red"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [10, 10, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "agent": {"name": "agent", "position": [3, 3, 0], "velocity": [0, 0, 0],
                                         "is_crouching": False, "holding": None, "objectType": "agent"},
                              },

                      "game_start": False,
                      "game_over": True},

                    ]