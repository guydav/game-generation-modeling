from collections import defaultdict
from itertools import permutations, product, repeat
import itertools
import json
import os
import pandas as pd
import pathlib
import tatsu, tatsu.ast, tatsu.grammars
from tqdm import tqdm
import typing

from config import COLORS, META_TYPES, OBJECTS_BY_ROOM_AND_TYPE, UNITY_PSEUDO_OBJECTS
from utils import (extract_predicate_function_name, 
                   extract_variables, 
                   extract_variable_type_mapping,
                   get_project_dir,
                   get_object_assignments,
                   FullState)
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

TYPE_REMAP = {"hexagonal_bin": "garbagecan"}

class CommonSensePredicateStatistics():
    def __init__(self,
                 cache_dir: str,
                 trace_paths: typing.Sequence[str],
                 overwrite=False):
        
        self.cache_dir = cache_dir

        cache_filename = os.path.join(cache_dir, 'predicate_statistics.pkl')

        if os.path.exists(cache_filename) and not overwrite:
            self.data = pd.read_pickle(cache_filename)

        else:
            self.data = pd.DataFrame(columns=['predicate', 'arg_ids', 'arg_types', 'start_step', 'end_step', 'id'])
            for trace_path in tqdm(trace_paths, desc="Processing traces"):
                trace = json.load(open(trace_path, 'r'))
                self.process_trace(trace)
            self.data.to_pickle(cache_filename)

    def _predicate_key(self, predicate: str, args: typing.Sequence[str]) -> str:
        return f"{predicate}-({','.join(args)})"
    
    def _domain_key(self, domain: str):
        if domain.endswith('few_new_objects'):
            return 'few'
        elif domain.endswith('medium_new_objects'):
            return 'medium'
        elif domain.endswith('many_new_objects'):
            return 'many'
        else:
            raise ValueError(f"Unrecognized domain: {domain}")
    
    def _get_room_objects(self, trace) -> set:
        '''
        Returns the set of objects in the room type of the given trace, excluding pseudo-objects,
        colors, and the agent
        '''

        room_type = self._domain_key(trace['scene'])          
        room_objects = set(sum([list(OBJECTS_BY_ROOM_AND_TYPE[room_type][obj_type]) for obj_type in OBJECTS_BY_ROOM_AND_TYPE[room_type]], []))
        room_objects -= set(UNITY_PSEUDO_OBJECTS.keys())
        room_objects -= set(COLORS)
        room_objects -= set(['agent'])

        return room_objects
    
    def _intervals_to_indices(self, intervals):
        '''
        Converts a list of (start, end) intervals to a list of indices (i.e. timesteps) covered
        by the intervals
        '''
        if intervals == []:
            return set()
        
        return set.union(*[set(range(*interval)) for interval in intervals if interval != []])
    
    def _indices_to_intervals(self, indices):
        '''
        Converts a list of indices (i.e. timesteps) to a list of (start, end) intervals
        '''
        intervals = []
        for idx in sorted(indices):
            # If this is the first interval, or the current index is not adjacent to the last index, then create a new interval
            if len(intervals) == 0 or idx != intervals[-1][-1]:
                intervals.append((idx, idx + 1))

            # Otherwise, extend the last interval
            else:
                intervals[-1] = (intervals[-1][0], idx + 1)

        return intervals
    
    def process_trace(self, trace):
        '''
        Process a trace, collecting the intervals in which each predicate is true (for
        every possible set of its arguments). Adds the information to the overall dataframe
        '''
     
        room_objects = self._get_room_objects(trace)
        replay = trace['replay']
        
        # Maps from the predicate-arg key to a list of intervals in which the predicate is true
        predicate_intervals = defaultdict(list)

        # Stores the most recent state of the agent and of each object
        most_recent_agent_state = None
        most_recent_object_states = {}

        received_full_update = False

        for idx, state in tqdm(enumerate(replay), total=len(replay), desc=f"Processing replay {trace['id']}", leave=False):
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

                        # Filter out any sets of arguments with duplicates
                        possible_args = [arg_set for arg_set in possible_args if len(set(arg_set)) == len(arg_set)]

                    for arg_set in possible_args:
                        for arg_ids in permutations(arg_set):
                            args = [most_recent_object_states[obj_id] for obj_id in arg_ids]
                            arg_types = tuple([obj.object_type.lower() for obj in args])

                            key = self._predicate_key(predicate, arg_ids)
                            predicate_fn = PREDICATE_LIBRARY_RAW[predicate]

                            evaluation = predicate_fn(most_recent_agent_state, args)

                            # If the predicate is true, then check to see if the last interval is closed. If it is, then
                            # create a new interval
                            if evaluation:
                                if key not in predicate_intervals or predicate_intervals[key][-1]["end_step"] is not None:
                                    predicate_intervals[key].append({"predicate": predicate, "arg_ids": arg_ids, 
                                                                     "arg_types": arg_types, "start_step": idx, 
                                                                     "end_step": None, "id": trace['id'],
                                                                     "domain": trace['scene']})

                            # If the predicate is false, then check to see if the last interval is open. If it is, then
                            # close it
                            else:
                                if key in predicate_intervals and predicate_intervals[key][-1]["end_step"] is None:
                                    predicate_intervals[key][-1]["end_step"] = idx


        # Close any intervals that are still open
        for key in predicate_intervals:
            if predicate_intervals[key][-1]["end_step"] is None:
                predicate_intervals[key][-1]["end_step"] = len(replay)

        # Collapse the intervals into a single dataframe and add it to the overall dataframe
        game_df = pd.DataFrame(sum(predicate_intervals.values(), []))
        self.data = pd.concat([self.data, game_df], ignore_index=True)

    def _tuple_containment(self, tuple_1, tuple_2):
        '''
        Returns true if the elements of tuple 1 appear, in order, in tuple 2
        '''
        if len(tuple_1) > len(tuple_2):
            return False
        
        for idx in range(len(tuple_2) - len(tuple_1) + 1):
            if tuple_1 == tuple_2[idx:idx + len(tuple_1)]:
                return True
            
        return False

    def _group_contained_mappings(self, argument_mapping_lists, operator="and"):
        '''
        Given a list of [N] argument mapping lists, each of which is a list of mappings the form (?v1-obj1, ?v2-obj2, ...),
        returns a list of list of [N] argument mappings where each mapping is contained by the largest in the list
        '''
        grouped_mappings = []

        for combination in product(*argument_mapping_lists):
            sorted_combination = sorted(combination, key=lambda x: len(x[0]), reverse=True)
            largest_mapping = sorted_combination[0]
            smaller_mappings = sorted_combination[1:]

            if operator == "and":
                condition = all([self._tuple_containment(smaller_mapping, largest_mapping) for smaller_mapping in smaller_mappings])

            # TODO: finish
            elif operator == "or":
                condition = any([self._tuple_containment(smaller_mapping, largest_mapping) for smaller_mapping in smaller_mappings])

            if condition:
                grouped_mappings.append(combination)

        return grouped_mappings


    def filter(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, str]):
        '''
        Filters the data by the given predicate and mapping, returning a list of intervals in which the predicate is true
        for each processed trace

        Returns a dictionary mapping from the trace ID to a dictionary that maps from the set of arguments to a list of
        intervals in which the predicate is true for that set of arguments
        '''

        predicate_rule = predicate.parseinfo.rule  # type: ignore

        if predicate_rule == "predicate":
            predicate_name = extract_predicate_function_name(predicate)  # type: ignore
            variables = extract_variables(predicate)  # type: ignore

            # Restrict the mapping to just the referenced variables and expand meta-types
            relevant_arg_mapping = {var: sum([META_TYPES.get(arg_type, [arg_type]) for arg_type in mapping[var]], []) 
                                    for var in variables if var in mapping}
            
            # Apply the type remapping, when needed. TODO: this is probably pretty slow -- best would be to just be consistent about
            # type naming
            for key, val in relevant_arg_mapping.items():
                relevant_arg_mapping[key] = [TYPE_REMAP.get(arg_type, arg_type) for arg_type in val]
            
            # For each predicate, we imagine that it is satisfied for every possible assignment of its unused arguments
            # (e.g., in the context of (?b - ball, ?h - bin), if the predicate (in_motion ?b) is satisfied for ?b = Dodgeball1,
            #  then we store its satisfaction for all possible assignments of ?h). The downside of this is that we wind up
            # storing an interval for every entry in the cartesian product of the unused variables' types -- possible TODO?
            unused_variables = [var for var in mapping.keys() if var not in variables]
            unused_variable_types = [mapping[var] for var in unused_variables]

            # In cases where a variable can have multiple types, we consider all possible combinations of types
            possible_arg_types = list(product(*[types for types in relevant_arg_mapping.values()]))

            # Merge the intervals for each possible assignment of argument types
            predicate_df = self.data[(self.data["predicate"] == predicate_name) & (self.data["arg_types"].isin(possible_arg_types))]

            # Construct the interval mapping, which maps from the trace ID to the argument mapping to a list of intervals
            interval_mapping = defaultdict(lambda: defaultdict(list))
            for row in predicate_df.itertuples():

                domain = self._domain_key(row.domain)
                other_object_assignments = get_object_assignments(domain, unused_variable_types, used_objects=row.arg_ids)

                for assignment in other_object_assignments:
                    full_assignment = tuple(sorted([f"{var}-{id}" for var, id in zip(variables, row.arg_ids)] + 
                                                   [f"{var}-{id}" for var, id in zip(unused_variables, assignment)]))

                    interval_mapping[row.id][full_assignment].append((row.start_step, row.end_step))

            return interval_mapping

        elif predicate_rule == "super_predicate":
            return self.filter(predicate['pred'], mapping)
        
        elif predicate_rule == "super_predicate_and":
            and_args = predicate["and_args"]
            if isinstance(and_args, tatsu.ast.AST):
                and_args = [and_args]

            interval_mapping = defaultdict(lambda: defaultdict(list))
            sub_interval_mappings = [self.filter(and_arg, mapping) for and_arg in and_args]

            # For each trace ID in which each sub-predicate is true for at least one interval...
            for id in set.intersection(*[set(sub_interval_mapping.keys()) for sub_interval_mapping in sub_interval_mappings]):
                
                for arg_ids in set.intersection(*[set(sub_interval_mapping[id].keys()) for sub_interval_mapping in sub_interval_mappings]):

                    # Compute, for each sub-predicate, the full set of indices in which it is true
                    truth_idxs = [self._intervals_to_indices(sub_interval_mapping[id][arg_ids]) for sub_interval_mapping in sub_interval_mappings]

                    # Compute the intersection of those indices
                    union = set.intersection(*truth_idxs)

                    # Reduce the set of indices back to a list of intervals
                    if len(union) > 0:
                        interval_mapping[id][arg_ids] = self._indices_to_intervals(union)

            return interval_mapping


        elif predicate_rule == "super_predicate_or":
            or_args = predicate["or_args"]
            if isinstance(or_args, tatsu.ast.AST):
                or_args = [or_args]

            interval_mapping = defaultdict(lambda: defaultdict(list))
            sub_interval_mappings = [self.filter(or_arg, mapping) for or_arg in or_args]

            # For each trace ID in which at least one sub-predicate is true for at least interval...
            for id in set.union(*[set(sub_interval_mapping.keys()) for sub_interval_mapping in sub_interval_mappings]):
                for arg_ids in set.union(*[set(sub_interval_mapping[id].keys()) for sub_interval_mapping in sub_interval_mappings]):

                    # Compute, for each sub-predicate, the full set of indices in which it is true
                    truth_idxs = [self._intervals_to_indices(sub_interval_mapping[id][arg_ids]) for sub_interval_mapping in sub_interval_mappings]

                    # Compute the union of those indices (i.e. the indices in which at least one sub-predicate is true)
                    union = set.union(*truth_idxs)

                    # Reduce the set of indices back to a list of intervals
                    if len(union) > 0:
                        interval_mapping[id][arg_ids] = self._indices_to_intervals(union)

            return interval_mapping
        
        elif predicate_rule == "super_predicate_not":
            sub_intervals = self.filter(predicate["not_args"], mapping)
            
            interval_mapping = defaultdict(lambda: defaultdict(list))

            # We need to check every trace ID in the dataset in case there are some traces in which the sub-predicate is never true
            for id in self.data["id"].unique():
                
                domain = self._domain_key(self.data[self.data["id"] == id]["domain"].unique()[0])
                possible_arg_assignments = get_object_assignments(domain, mapping.values())

                # Similarly, we need to check every possible set of arguments in case there are some sets in which the sub-predicate is never true
                for arg_ids in possible_arg_assignments:
              
                    argument_mapping = tuple(sorted(f"{var}-{id}" for var, id in zip(mapping.keys(), arg_ids)))

                    # TODO: there's a very small chance that these values could be undefined
                    earliest_start = min(self.data[self.data["id"] == id]["start_step"])
                    latest_end = max(self.data[self.data["id"] == id]["end_step"])

                    if sub_intervals[id][argument_mapping] == []:
                        interval_mapping[id][argument_mapping] = [(earliest_start, latest_end)]

                    else:
                        truth_idxs = self._intervals_to_indices(sub_intervals[id][argument_mapping])
                        inverted_idxs = set(range(earliest_start, latest_end)) - truth_idxs

                        if len(inverted_idxs) > 0:
                            interval_mapping[id][argument_mapping] = self._indices_to_intervals(inverted_idxs)

            return interval_mapping

        elif predicate_rule == "super_predicate_exists":
            variable_type_mapping = extract_variable_type_mapping(predicate["exists_vars"]["variables"])  # type: ignore  
            # will return something like {?c2 - cube_block}
            print(f"Exists variable type mapping: {variable_type_mapping}")

            sub_intervals = self.filter(predicate["exists_args"], {**mapping, **variable_type_mapping})
            # this will return, for each game, the set of intervals based on the argument *types* (not the argument *names*)
            # is it sufficient to just directly return this?

            return sub_intervals
        
        elif predicate_rule == "super_predicate_forall":
            variable_type_mapping = extract_variable_type_mapping(predicate["forall_vars"]["variables"])  # type: ignore
            sub_intervals = self.filter(predicate["forall_args"], {**mapping, **variable_type_mapping})

            interval_mapping = {}
            for id in sub_intervals:
                pass

        else:
            raise ValueError(f"Error: Unknown rule '{predicate_rule}'")   

if __name__ == '__main__':

    DEFAULT_GRAMMAR_PATH = "./dsl/dsl.ebnf"
    grammar = open(DEFAULT_GRAMMAR_PATH).read()
    grammar_parser = typing.cast(tatsu.grammars.Grammar, tatsu.compile(grammar))

    game = open(get_project_dir() + '/reward-machine/games/ball_to_bin_from_bed.txt').read()
    game_ast = grammar_parser.parse(game)  # type: ignore

    # should be: (and (not (in_motion ?b)) (in ?h ?b)))
    test_pred = game_ast[4][1]['preferences'][0]['definition']['forall_pref']['preferences']['pref_body']['body']['exists_args']['then_funcs'][2]['seq_func']['once_pred']
    
    # should be: (and (in_motion ?b) (not (agent_holds ?b)))
    test_pred_2 = game_ast[4][1]['preferences'][0]['definition']['forall_pref']['preferences']['pref_body']['body']['exists_args']['then_funcs'][1]['seq_func']['hold_pred'] 

    # should be: (exists (?c2 - cube_block) (touch ?c2 ?b))
    # test_pred_3 = game_ast[4][1]['preferences'][0]['definition']['forall_pref']['preferences']['pref_body']['body']['exists_args']['then_funcs'][3]['seq_func']['once_pred'] 


    test_mapping = {"?b": ["ball"], "?h": ["hexagonal_bin"]}

    # trace_path = pathlib.Path(get_project_dir() + '/reward-machine/traces/new_replay_format_test.json')
    trace_path = pathlib.Path(get_project_dir() + '/reward-machine/traces/otcaCEGfUhzEfGy72Qm8-preCreateGame.json')
    trace_path = pathlib.Path(get_project_dir() + '/reward-machine/traces/three_wall_to_bin_bounces-RErerecorded.json')

    # trace_path = pathlib.Path(get_project_dir() + '/reward-machine/traces/otcaCEGfUhzEfGy72Qm8-preCreateGame-rerecorded.json')
    cache_dir = pathlib.Path(get_project_dir() + '/reward-machine/caches')


    TEST_TRACE_NAMES = ["throw_ball_to_bin_unique_positions", "setup_test_trace", "building_castle",
                        "throw_all_dodgeballs", "stack_3_cube_blocks", "three_wall_to_bin_bounces",
                        "complex_stacking_trace"]

    # stats = CommonSensePredicateStatistics(cache_dir, [f"{get_project_dir()}/reward-machine/traces/{trace}-rerecorded.json" for trace in TEST_TRACE_NAMES], overwrite=True)
    # print(stats.data)

    stats = CommonSensePredicateStatistics(cache_dir, [trace_path], overwrite=True)
    print(stats.data[stats.data["predicate"] == "in"])
    print(stats.filter(test_pred, test_mapping))

    # target_types = []
    # filter_fn = lambda x: all([target_type in x for target_type in target_types])

    # print(stats.data.loc[stats.data["arg_types"].apply(filter_fn) & stats.data["predicate"].isin(["in_motion", "touch", "agent_holds"])].sort_values("start_step"))


