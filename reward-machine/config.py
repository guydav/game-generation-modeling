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
