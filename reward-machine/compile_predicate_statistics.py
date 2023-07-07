import itertools
import pandas as pd
import pathlib
from tqdm import tqdm

from utils import FullState, get_project_dir
from manual_run import _load_trace
from predicate_handler import PREDICATE_LIBRARY_RAW

COMMON_SENSE_PREDICATES_AND_FUNCTIONS = (
    ("adjacent", 2),
    ("agent_holds", 1),
    ("in", 2),
    ("in_motion", 1),
    ("on", 2),
    ("touch", 2),
    # ("between", 3),
)

data = []

trace_path = pathlib.Path(get_project_dir() + '/reward-machine/traces/throw_all_dodgeballs.json')
trace = list(_load_trace(trace_path))

most_recent_agent_state = None
most_recent_object_states = {}

# HACK: we extract the full set of objects in the room by looking for the first "full state update"
#       in the trace. We do this by looking for the first state that has more than 20 objects updated

for idx, (state, is_final) in enumerate(tqdm(trace, total=len(trace), desc="Processing trace")):
    state = FullState.from_state_dict(state)

    if state.agent_state_changed:
        most_recent_agent_state = state.agent_state

    # Update the most recent state of each object and record the objects that changed
    changed_this_step = []
    if state.n_objects_changed > 0:
        for obj in state.objects:
            most_recent_object_states[obj.object_id] = obj
            changed_this_step.append(obj.object_id)

    # Only perform predicate checks if we've received at least one full state update, which we 
    # hackily detect by ensuring that at least 20 objects have been updated
    if len(most_recent_object_states) > 20 and len(changed_this_step) > 0:

        for predicate, n_args in COMMON_SENSE_PREDICATES_AND_FUNCTIONS:
            if n_args == 0:
                evaluation = PREDICATE_LIBRARY_RAW[predicate](most_recent_agent_state, [])

                if evaluation:
                    info = {"predicate": predicate, "arg_ids": (), "arg_types": (), "step": idx, "trace": trace_path.stem}
                    data.append(info)

            # Perform the check for each updated object
            elif n_args == 1:
                for object_id in changed_this_step:
                    evaluation = PREDICATE_LIBRARY_RAW[predicate](most_recent_agent_state, [most_recent_object_states[object_id]])

                    if evaluation:
                        info = {"predicate": predicate, "arg_ids": (object_id,), "arg_types": (most_recent_object_states[object_id].object_type,), 
                                "step": idx, "trace": trace_path.stem}
                        data.append(info)
                
            # Perform the check for every pair of objects where at least one is updated
            elif n_args == 2:
                for object_id_1, object_id_2 in itertools.product(changed_this_step, most_recent_object_states.keys()):
                    objects = [most_recent_object_states[object_id_1], most_recent_object_states[object_id_2]]

                    for permutation in itertools.permutations(objects):
                        evaluation = PREDICATE_LIBRARY_RAW[predicate](most_recent_agent_state, permutation)

                        if evaluation:
                            arg_ids = tuple([obj.object_id for obj in permutation])
                            arg_types = tuple([obj.object_type for obj in permutation])
                            info = {"predicate": predicate, "arg_ids": arg_ids, "arg_types": arg_types, 
                                    "step": idx, "trace": trace_path.stem}
                            data.append(info)


            # Perform the check for every triple of objects where at least one is updated
            elif n_args == 3:
                for object_id_1, object_id_2, object_id_3 in itertools.product(changed_this_step, 
                                                                                most_recent_object_states.keys(),
                                                                                most_recent_object_states.keys()):
                    
                    objects = [most_recent_object_states[object_id_1], most_recent_object_states[object_id_2], most_recent_object_states[object_id_3]]

                    for permutation in itertools.permutations(objects):
                        evaluation = PREDICATE_LIBRARY_RAW[predicate](most_recent_agent_state, permutation)

                        if evaluation:
                            arg_ids = tuple([obj.object_id for obj in permutation])
                            arg_types = tuple([obj.object_type for obj in permutation])
                            info = {"predicate": predicate, "arg_ids": arg_ids, "arg_types": arg_types, 
                                    "step": idx, "trace": trace_path.stem}
                            data.append(info)

df = pd.DataFrame(data)
print(df.loc[(df["predicate"] == "in_motion") & (df["arg_types"] == ("Dodgeball",))])
        
