from itertools import permutations, product, repeat
import itertools
import json
import os
import pandas as pd
import pathlib
from tqdm import tqdm
import typing

from config import OBJECTS_BY_ROOM_AND_TYPE, UNITY_PSEUDO_OBJECTS, COLORS
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

class CommonSensePredicateStatistics():
    def __init__(self,
                 cache_dir: str,
                 trace_paths: typing.Sequence[str]):
        
        self.cache_dir = cache_dir
        self.data = pd.DataFrame(columns=['predicate', 'arg_ids', 'arg_types', 'start_step', 'end_step', 'id'])


    def _predicate_key(self, predicate: str, args: typing.Sequence[str]) -> str:
        return f"{predicate}-({','.join(args)})"
    
    def _get_room_objects(self, trace) -> set:
        '''
        Returns the set of objects in the room type of the given trace, excluding pseudo-objects,
        colors, and the agent
        '''

        if trace['scene'].endswith('few_new_objects'):
            room_type = 'few'
        elif trace['scene'].endswith('medium_new_objects'):
            room_type = 'medium'
        elif trace['scene'].endswith('many_new_objects'):
            room_type = 'many'
        else:
            raise ValueError(f"Unrecognized scene: {trace['scene']}")
        
        room_objects = set(sum([list(OBJECTS_BY_ROOM_AND_TYPE[room_type][obj_type]) for obj_type in OBJECTS_BY_ROOM_AND_TYPE[room_type]], []))
        room_objects -= set(UNITY_PSEUDO_OBJECTS.keys())
        room_objects -= set(COLORS)
        room_objects -= set(['agent'])

        return room_objects
    
    def process_trace(self, trace):
        '''
        Process a trace, collecting the intervals in which each predicate is true (for
        every possible set of its arguments). Adds the information to the overall dataframe
        '''
     
        room_objects = self._get_room_objects(trace)
        replay = trace['replay']
        
        # Maps from the predicate-arg key to a list of intervals in which the predicate is true
        predicate_intervals = {}

        # Stores the most recent state of the agent and of each object
        most_recent_agent_state = None
        most_recent_object_states = {}

        received_full_update = False

        for idx, state in tqdm(enumerate(replay), total=len(replay), desc="Processing replay", leave=False):
            is_final = idx == len(replay) - 1
            state = FullState.from_state_dict(state)

            # Track changes to the agent
            if state.agent_state_changed:
                most_recent_agent_state = state.agent_state

            # And to objects
            for obj in state.objects:
                most_recent_object_states[obj.object_id] = obj
            
            # Check if we've received a full state update, which we detect by seeing if the most_recent_object_states
            # includes every object in the room
            if not received_full_update:
                difference = room_objects.difference(set(most_recent_object_states.keys()))
                received_full_update = (len(difference) == 0)
            
            # Only perform predicate checks if we've received at least one full state update
            if received_full_update and state.n_objects_changed > 0:
                for predicate, n_args in COMMON_SENSE_PREDICATES_AND_FUNCTIONS:

                    # Some predicates take only an empty list for arguments
                    if n_args == 0:
                        possible_args = [[]]

                    # Collect all possible sets of arguments in which at least one has been updated this step
                    else:
                        changed_this_step = [obj.object_id for obj in state.objects]
                        possible_args = list(product(*([changed_this_step] + list(repeat(room_objects, n_args - 1)))))

                    for arg_set in possible_args:
                        for arg_ids in permutations(arg_set):
                            args = [most_recent_object_states[obj_id] for obj_id in arg_ids]
                            arg_types = tuple([obj.object_type for obj in args])

                            key = self._predicate_key(predicate, arg_ids)
                            predicate_fn = PREDICATE_LIBRARY_RAW[predicate]

                            evaluation = predicate_fn(most_recent_agent_state, args)

                            # If the predicate is true, then check to see if the last interval is closed. If it is, then
                            # create a new interval
                            if evaluation:
                                if key not in predicate_intervals or predicate_intervals[key][-1]["end_step"] is not None:
                                    predicate_intervals[key] = [{"predicate": predicate, "arg_ids": arg_ids, 
                                                                 "arg_types": arg_types, "start_step": idx, 
                                                                 "end_step": None, "id": trace['id']}]

                            # If the predicate is false, then check to see if the last interval is open. If it is, then
                            # close it
                            else:
                                if key in predicate_intervals and predicate_intervals[key][-1]["end_step"] is None:
                                    predicate_intervals[key][-1]["end_step"] = idx


        # Close any intervals that are still open
        for key in predicate_intervals:
            if predicate_intervals[key][-1]["end_step"] is None:
                predicate_intervals[key][-1]["end_step"] = len(replay)

        # Collapse the intervals into a single dataframe
        game_df = pd.DataFrame(sum(predicate_intervals.values(), []))
        self.data = pd.concat([self.data, game_df], ignore_index=True)
        


trace_path = pathlib.Path(get_project_dir() + '/reward-machine/traces/new_replay_format_test.json')
# trace = list(_load_trace(trace_path))
trace = json.load(open(trace_path, 'r'))
a = CommonSensePredicateStatistics('', '')
a.process_trace(trace)

# target_types = []
# filter_fn = lambda x: all([target_type in x for target_type in target_types])

# print(df.loc[df["arg_types"].apply(filter_fn) & df["predicate"].isin(["in_motion", "touch", "agent_holds"])].sort_values("start_step"))


