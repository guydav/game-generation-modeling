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

trace_path = pathlib.Path(get_project_dir() + '/reward-machine/traces/throw_all_dodgeballs.json')
trace = list(_load_trace(trace_path))

most_recent_agent_state = None
most_recent_object_states = {}

# HACK: we extract the full set of objects in the room by looking for the first "full state update"
#       in the trace. We do this by looking for the first state that has more than 20 objects updated

# IDEA: ultimately the plan is to construct a dataframe with the following keys: predicate | arg_ids | arg_types | start step | end step | trace
#       where the start and end step are intervals where the predicate is true with the given arguments. A given predicate / set of arguments
#       might have multiple intervals in a single trace.
#       
#       This will require storing partial intervals as we work through the trace, and terminating them when the predicate becomes false.

# Will map from a key that uniquely identifies a predicate and its arguments to a list of intervals
intervals = {}

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

                # If the predicate is true, then check to see if the last interval is closed. If it is, then
                # create a new interval
                key = f"{predicate}-()"
                if evaluation:
                    if key not in intervals or intervals[key][-1]["end_step"] is not None:
                        intervals[key] = [{"predicate": predicate, "arg_ids": (), "arg_types": (), "start_step": idx, "end_step": None, "trace": trace_path.stem}]

                # If the predicate is false, then check to see if the last interval is open. If it is, then
                # close it
                else:
                    if key in intervals and intervals[key][-1]["end_step"] is None:
                        intervals[key][-1]["end_step"] = idx

            # Perform the check for each updated object
            elif n_args == 1:
                for object_id in changed_this_step:
                    evaluation = PREDICATE_LIBRARY_RAW[predicate](most_recent_agent_state, [most_recent_object_states[object_id]])

                    arg_ids = (object_id,)
                    arg_types = (most_recent_object_states[object_id].object_type,)

                    key = f"{predicate}-{arg_ids}"
                    if evaluation:
                        if key not in intervals or intervals[key][-1]["end_step"] is not None:  
                            intervals[key] = [{"predicate": predicate, "arg_ids": arg_ids, "arg_types": arg_types, 
                                               "start_step": idx, "end_step": None, "trace": trace_path.stem}]

                    else:
                        if key in intervals and intervals[key][-1]["end_step"] is None:
                            intervals[key][-1]["end_step"] = idx

                
            # Perform the check for every pair of objects where at least one is updated
            elif n_args == 2:
                for object_id_1, object_id_2 in itertools.product(changed_this_step, most_recent_object_states.keys()):
                    objects = [most_recent_object_states[object_id_1], most_recent_object_states[object_id_2]]

                    for permutation in itertools.permutations(objects):
                        evaluation = PREDICATE_LIBRARY_RAW[predicate](most_recent_agent_state, permutation)

                        arg_ids = tuple([obj.object_id for obj in permutation])
                        arg_types = tuple([obj.object_type for obj in permutation])

                        key = f"{predicate}-{arg_ids}"
                        if evaluation:
                            if key not in intervals or intervals[key][-1]["end_step"] is not None:  
                                intervals[key] = [{"predicate": predicate, "arg_ids": arg_ids, "arg_types": arg_types, 
                                                   "start_step": idx, "end_step": None, "trace": trace_path.stem}]

                        else:
                            if key in intervals and intervals[key][-1]["end_step"] is None:
                                intervals[key][-1]["end_step"] = idx

            # Perform the check for every triple of objects where at least one is updated
            elif n_args == 3:
                for object_id_1, object_id_2, object_id_3 in itertools.product(changed_this_step, 
                                                                                most_recent_object_states.keys(),
                                                                                most_recent_object_states.keys()):
                    
                    objects = [most_recent_object_states[object_id_1], most_recent_object_states[object_id_2], most_recent_object_states[object_id_3]]

                    for permutation in itertools.permutations(objects):
                        evaluation = PREDICATE_LIBRARY_RAW[predicate](most_recent_agent_state, permutation)

                        arg_ids = tuple([obj.object_id for obj in permutation])
                        arg_types = tuple([obj.object_type for obj in permutation])

                        key = f"{predicate}-{arg_ids}"
                        if evaluation:
                            if key not in intervals or intervals[key][-1]["end_step"] is not None:  
                                intervals[key] = [{"predicate": predicate, "arg_ids": arg_ids, "arg_types": arg_types, 
                                                   "start_step": idx, "end_step": None, "trace": trace_path.stem}]

                        else:
                            if key in intervals and intervals[key][-1]["end_step"] is None:
                                intervals[key][-1]["end_step"] = idx


# Set the end step of any intervals that are still open
for key in intervals:
    if intervals[key][-1]["end_step"] is None:
        intervals[key][-1]["end_step"] = len(trace)

data = sum(intervals.values(), [])

df = pd.DataFrame(data)




target_types = ["Dodgeball"]
filter_fn = lambda x: all([target_type in x for target_type in target_types])

print(df.loc[df["arg_types"].apply(filter_fn) & df["predicate"].isin(["in_motion", "touch", "agent_holds"])].sort_values("start_step"))