from collections import defaultdict
from functools import reduce
import glob
import gzip
import hashlib
from itertools import groupby, permutations, product, repeat, starmap
import json
import os
import pandas as pd
import pathlib
import pickle
import polars as pl
import re
import tatsu, tatsu.ast, tatsu.grammars
import time
from tqdm import tqdm
import typing
from viztracer import VizTracer


from config import COLORS, META_TYPES, OBJECTS_BY_ROOM_AND_TYPE, ORIENTATIONS, SIDES, UNITY_PSEUDO_OBJECTS
from utils import (extract_predicate_function_name,
                   extract_variables,
                   extract_variable_type_mapping,
                   get_project_dir,
                   get_object_assignments,
                   FullState,
                   AgentState)
from ast_parser import PREFERENCES
from ast_printer import ast_section_to_string
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

INTERVALS_LIST_POLARS_TYPE = pl.List(pl.List(pl.Int64))


# Maps from types returned by unity to the types used in the DSL
TYPE_REMAP = {
    "garbagecan": "hexagonal_bin", "bridgeblock": "bridge_block", "cubeblock": "cube_block",
    "cylinderblock": "cylindrical_block", "flatrectblock": "flat_block",
    "pyramidblock": "pyramid_block", "longcylinderblock": "tall_cylindrical_block",
    "tallrectblock": "tall_rectangular_block", "triangleblock": "triangle_block"
}

DEBUG = False
PROFILE = False
DEFAULT_CACHE_DIR = pathlib.Path(get_project_dir() + '/reward-machine/caches')
DEFAULT_CACHE_FILE_NAME_FORMAT = 'predicate_statistics_{traces_hash}.pkl.gz'
DEFAULT_TRACE_LENGTHS_FILE_NAME_FORMAT = 'trace_lengths_{traces_hash}.pkl'
DEFAULT_BASE_TRACE_PATH = "reward-machine/traces/participant-traces/"


class PredicateNotImplementedException(Exception):
    pass


def stable_hash(str_data: str):
    return hashlib.md5(bytearray(str_data, 'utf-8')).hexdigest()


def stable_hash_list(list_data: typing.Sequence[str]):
    return stable_hash('\n'.join(sorted(list_data)))


class CommonSensePredicateStatisticsSplitArgs():
    data: pl.DataFrame
    domains: typing.List[str]
    predicates: typing.List[str]
    trace_lengths_and_domains: typing.Dict[str, typing.Tuple[int, str]]
    trace_lengths_and_domains_df: pl.DataFrame

    def __init__(self,
                 cache_dir: typing.Union[str, pathlib.Path],
                 trace_names: typing.Sequence[str],
                 cache_rules: typing.Optional[typing.Sequence[str]] = None,
                 base_trace_path: typing.Union[str, pathlib.Path] = DEFAULT_BASE_TRACE_PATH,
                 cache_filename_format: str = DEFAULT_CACHE_FILE_NAME_FORMAT,
                 trace_lengths_filename_format: str = DEFAULT_TRACE_LENGTHS_FILE_NAME_FORMAT,
                 overwrite: bool = False, trace_hash_n_characters: int = 8):

        self.cache_dir = cache_dir

        # Compute hash of trace names
        trace_names_hash = stable_hash_list([os.path.basename(trace_name) for trace_name in trace_names])[:trace_hash_n_characters]

        stats_filename = os.path.join(cache_dir, cache_filename_format.format(traces_hash=trace_names_hash))
        trace_lengths_and_domains_filename = os.path.join(cache_dir, trace_lengths_filename_format.format(traces_hash=trace_names_hash))
        open_method = gzip.open if stats_filename.endswith('.gz') else open

        if os.path.exists(stats_filename) and not overwrite:
            self.data = pd.read_pickle(stats_filename)  # type: ignore
            print(f'Loaded data with shape {self.data.shape} from {stats_filename}')
            with open_method(trace_lengths_and_domains_filename, 'rb') as f:
                self.trace_lengths_and_domains = pickle.load(f)

        else:
            if base_trace_path is None:
                raise ValueError("Must specify base_trace_path if cache file does not exist")

            print(f"No cache file found at {stats_filename}. Building from scratch...")

            trace_paths = [os.path.join(base_trace_path, f"{trace_name}.json") for trace_name in trace_names]

            # TODO (gd1279): if we ever decide to support 3- or 4- argument predicates, we'll need to
            # add additional columns here
            self.data = pd.DataFrame(columns=['predicate', 'arg_1_id', 'arg_1_type', 'arg_2_id', 'arg_2_type', 'trace_id', 'domain', 'intervals'])  # type: ignore
            self.trace_lengths_and_domains = {}
            for trace_path in tqdm(trace_paths, desc="Processing traces"):
                trace = json.load(open(trace_path, 'r'))
                self.process_trace(trace)

            self.data.to_pickle(stats_filename)
            with open_method(trace_lengths_and_domains_filename, 'wb') as f:
                pickle.dump(self.trace_lengths_and_domains, f)

        self.domains = list(self.data['domain'].unique())  # type: ignore
        self.predicates = list(self.data['predicate'].unique())  # type: ignore
        self._trace_lengths_and_domains_to_df()

        # Convert to polars
        # breakpoint()
        self.data = pl.from_pandas(self.data)

        # Cache calls to get_object_assignments
        self.cache_rules = cache_rules
        self.object_assignment_cache = {}
        self.predicate_interval_cache = {}

    def _trace_lengths_and_domains_to_df(self):
        trace_ids = []
        trace_lengths = []
        domains = []

        for trace_id, (length, domain) in self.trace_lengths_and_domains.items():
            trace_ids.append(trace_id)
            trace_lengths.append(length)
            domains.append(domain)

        self.trace_lengths_and_domains_df = pl.DataFrame({
            'trace_id': trace_ids,
            'trace_length': trace_lengths,
            'domain': domains
        })

    def _predicate_key(self, predicate: str, args: typing.Sequence[str]) -> str:
        return f"{predicate}-({','.join(args)})"

    def _domain_key(self, domain: str):
        if domain.endswith('few_new_objects'):
            return 'few'
        elif domain.endswith('semi_sparse_new_objects'):
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
        # room_objects -= set(UNITY_PSEUDO_OBJECTS.keys())
        room_objects -= set(COLORS)
        room_objects -= set(SIDES)
        room_objects -= set(ORIENTATIONS)
        room_objects -= set(['agent'])

        return room_objects

    def _object_assignments(self, domain, variable_types, used_objects=[]):
        '''
        Wrapper around get_object_assignments in order to cache outputs
        '''

        key = (domain, tuple(variable_types), tuple(used_objects))
        if key not in self.object_assignment_cache:
            object_assignments = get_object_assignments(domain, variable_types, used_objects=used_objects)
            self.object_assignment_cache[key] = object_assignments

        return self.object_assignment_cache[key]

    def _intersect_intervals(self, intervals_1: typing.List[typing.List[int]], intervals_2: typing.List[typing.List[int]]):
        '''
        Given two lists of [start, end] intervals, returns the list of intervals in which they intersect

        Code from: https://stackoverflow.com/questions/69997547/intersections-of-intervals
        '''
        intersections = []
        i = j = 0

        while i < len(intervals_1) and j < len(intervals_2):
            low = max(intervals_1[i][0], intervals_2[j][0])
            high = min(intervals_1[i][1], intervals_2[j][1])
            if low <= high:
                intersections.append([low, high])

            # Remove the interval with the smallest endpoint
            if intervals_1[i][1] < intervals_2[j][1]:
                i += 1
            else:
                j += 1

        return intersections


    def _intersect_intervals_tuple(self, intervals: typing.Tuple[typing.List[typing.List[int]], typing.List[typing.List[int]]]):
        if not intervals[0] or not intervals[1]:
            return ([],)

        return (self._intersect_intervals(intervals[0], intervals[1]),)


    def _union_intervals(self, intervals_1: typing.List[typing.List[int]], intervals_2: typing.List[typing.List[int]]):
        '''
        Given two lists of [start, end] intervals, returns the list of intervals in which either is true
        '''
        all_intervals = sorted(intervals_1 + intervals_2)
        unions = []

        for start, end in all_intervals:
            if unions != [] and unions[-1][1] >= start - 1:
                unions[-1][1] = max(unions[-1][1], end)
            else:
                unions.append([start, end])

        return unions

    def _union_intervals_tuple(self, intervals: typing.Tuple[typing.List[typing.List[int]], typing.List[typing.List[int]]]):
        i0, i1 = intervals
        retval = None
        if not i0:
            if not i1:
                retval = []
            else:
                retval = i1
        elif not i1:
            retval = i0

        if retval is None:
            retval = self._union_intervals(intervals[0], intervals[1])

        return (retval, )

    def _invert_intervals(self, intervals: typing.List[typing.List[int]], length: int):
        if not intervals:
            return [[0, length]]

        inverted = []
        cur = 0

        for interval in intervals:
            if cur < interval[0]:
                inverted.append([cur, interval[0]])
            cur = interval[1]

        if cur < length:
            inverted.append([cur, length])

        return inverted

    def _invert_intervals_tuple_apply(self, intervals_tuple: typing.Tuple[typing.List[typing.List[int]], int]):
        return (self._invert_intervals(*intervals_tuple), )

    def process_trace(self, trace):
        '''
        Process a trace, collecting the intervals in which each predicate is true (for
        every possible set of its arguments). Adds the information to the overall dataframe
        '''

        room_objects = self._get_room_objects(trace)
        replay = trace['replay']
        replay_len = int(len(replay))

        # Maps from the predicate-arg key to a list of intervals in which the predicate is true
        predicate_satisfaction_mapping = {}

        # Stores the most recent state of the agent and of each object
        most_recent_agent_state = None
        most_recent_object_states = {}

        received_full_update = False

        full_trace_id = f"{trace['id']}-{trace['replayKey']}"

        for idx, state in tqdm(enumerate(replay), total=replay_len, desc=f"Processing replay {full_trace_id}", leave=False):
            is_final = idx == replay_len - 1
            state = FullState.from_state_dict(state)

            # Track changes to the agent
            if state.agent_state_changed:
                most_recent_agent_state = state.agent_state


            # And to objects
            for obj in state.objects:
                most_recent_object_states[obj.object_id] = obj

            # Check if we've received a full state update, which we detect by seeing if the most_recent_object_states
            # includes every object in the room (aside from PseudoObjects, which never receive updates)
            if not received_full_update:
                difference = room_objects.difference(set(most_recent_object_states.keys()))
                received_full_update = (difference == set(UNITY_PSEUDO_OBJECTS.keys()))

            # Only perform predicate checks if we've received at least one full state update
            if received_full_update and (state.n_objects_changed > 0 or state.agent_state_changed):
                for predicate, n_args in COMMON_SENSE_PREDICATES_AND_FUNCTIONS:

                    # Some predicates take only an empty list for arguments
                    if n_args == 0:
                        possible_args = [[]]

                    # Collect all possible sets of arguments in which at least one has been updated this step
                    else:
                        changed_this_step = [obj.object_id for obj in state.objects]

                        if state.agent_state_changed:
                            changed_this_step.append("agent")

                        possible_args = list(product(*([changed_this_step] + list(repeat(room_objects, n_args - 1)))))

                        # Filter out any sets of arguments with duplicates
                        possible_args = [arg_set for arg_set in possible_args if len(set(arg_set)) == len(arg_set)]

                    for arg_set in possible_args:
                        for arg_ids in permutations(arg_set):

                            args, arg_types = [], []
                            for obj_id in arg_ids:
                                if obj_id == "agent":
                                    args.append(most_recent_agent_state)
                                    arg_types.append("agent")

                                elif obj_id in UNITY_PSEUDO_OBJECTS:
                                    args.append(UNITY_PSEUDO_OBJECTS[obj_id])
                                    arg_types.append(UNITY_PSEUDO_OBJECTS[obj_id].object_type.lower())

                                else:
                                    args.append(most_recent_object_states[obj_id])
                                    arg_types.append(most_recent_object_states[obj_id].object_type.lower())

                            key = self._predicate_key(predicate, arg_ids)
                            predicate_fn = PREDICATE_LIBRARY_RAW[predicate]

                            evaluation = predicate_fn(most_recent_agent_state, args)

                            # If the predicate is true, then check to see if the last interval is closed. If it is, then
                            # create a new interval
                            if evaluation:
                                if key not in predicate_satisfaction_mapping:
                                    info = {"predicate": predicate,"trace_id": full_trace_id,
                                            "domain": self._domain_key(trace['scene']),
                                            "intervals": [[idx, None]]}

                                    for i, (arg_id, arg_type) in enumerate(zip(arg_ids, arg_types)):
                                        info[f"arg_{i + 1}_id"] = arg_id
                                        info[f"arg_{i + 1}_type"] = TYPE_REMAP.get(arg_type, arg_type)
                                    predicate_satisfaction_mapping[key] = info

                                elif predicate_satisfaction_mapping[key]['intervals'][-1][1] is not None:
                                    predicate_satisfaction_mapping[key]['intervals'].append([idx, None])

                            # If the predicate is false, then check to see if the last interval is open. If it is, then
                            # close it
                            else:
                                if key in predicate_satisfaction_mapping and predicate_satisfaction_mapping[key]["intervals"][-1][1] is None:
                                    predicate_satisfaction_mapping[key]["intervals"][-1][1] = idx


        # Close any intervals that are still open
        for key in predicate_satisfaction_mapping:
            if predicate_satisfaction_mapping[key]["intervals"][-1][1] is None:
                predicate_satisfaction_mapping[key]["intervals"][-1][1] = replay_len

        # Record the trace's length
        self.trace_lengths_and_domains[full_trace_id] = (replay_len, self._domain_key(trace['scene']))

        # Collapse the intervals into a single dataframe and add it to the overall dataframe
        game_df = pd.DataFrame(predicate_satisfaction_mapping.values())
        self.data = pd.concat([self.data, game_df], ignore_index=True)  # type: ignore

    def filter(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]]):
        try:
            used_variables = set()
            result = self._inner_filter(predicate, mapping, used_variables)
            sorted_variables = sorted(used_variables)
            return {(row_dict['trace_id'], tuple([f'{k}->{row_dict[k]}' for k in sorted_variables])): row_dict['intervals']
                    for row_dict in result.to_dicts()}
        except PredicateNotImplementedException as e:
            # TODO: decide what we return in this case, or if we pass it through and let the feature handle it
            raise e


    def _inner_filter(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], used_variables: typing.Set[str]) -> pl.DataFrame:
        '''
        Filters the data by the given predicate and mapping, returning a list of intervals in which the predicate is true
        for each processed trace

        Returns a dictionary mapping from the trace ID to a dictionary that maps from the set of arguments to a list of
        intervals in which the predicate is true for that set of arguments
        '''

        predicate_rule = predicate.parseinfo.rule  # type: ignore

        # Temporarily disable caching to profile without it
        # if predicate_rule in self.cache_rules:
        #     ast_str = ast_section_to_string(predicate, PREFERENCES)
        #     ast_str = re.sub(r"\s+", " ", ast_str)
        #     for key, val in mapping.items():
        #         ast_str = ast_str.replace(key, str(val))

        #     if ast_str in self.predicate_interval_cache:
        #         return self.predicate_interval_cache[ast_str]

        if predicate_rule == "predicate":
            predicate_name = extract_predicate_function_name(predicate)  # type: ignore

            if predicate_name not in self.predicates:
                raise PredicateNotImplementedException(predicate_name)

            variables = extract_variables(predicate)  # type: ignore
            used_variables.update(variables)

            if DEBUG: start = time.perf_counter()

            # Restrict the mapping to just the referenced variables and expand meta-types
            relevant_arg_mapping = {}
            for var in variables:
                if var in mapping:
                    relevant_arg_mapping[var] = sum([META_TYPES.get(arg_type, [arg_type]) for arg_type in mapping[var]], [])
                elif not var.startswith("?"):
                    relevant_arg_mapping[var] = [var]

            filter_expr = pl.col("predicate") == predicate_name
            rename_mapping = {}
            for i, (arg_var, arg_types) in enumerate(relevant_arg_mapping.items()):
                # TODO: think about what to do about directly quantified variables here
                filter_expr &= pl.col(f"arg_{i + 1}_type").is_in(arg_types)
                rename_mapping[f"arg_{i + 1}_id"] = arg_var

            # Returns a dataframe in which the arg_id columns are renamed to the variable names they map to
            predicate_df = self.data.filter(filter_expr).rename(rename_mapping)

            # We drop the arg_type columns and any un-renamed arg_id columns, since they're no longer needed
            predicate_df = predicate_df.drop([c for c in predicate_df.columns if c.startswith("arg_")])

            if DEBUG:
                end = time.perf_counter()
                print(f"Time per collect '{predicate_name}': {'%.5f' % (end - start)}s")  # type: ignore

        elif predicate_rule == "super_predicate":
            predicate_df = self._inner_filter(predicate["pred"], mapping, used_variables)  # type: ignore

        elif predicate_rule == "super_predicate_and":
            and_args = predicate["and_args"]
            if isinstance(and_args, tatsu.ast.AST):
                and_args = [and_args]

            sub_predicate_dfs = [self._inner_filter(and_arg, mapping, used_variables) for and_arg in and_args]  # type: ignore

            predicate_df = sub_predicate_dfs[0]

            if DEBUG: start = time.perf_counter()
            for i, sub_predicate_df in enumerate(sub_predicate_dfs[1:]):
                # Collect all variables which appear in both the current predicate (which will be expanded) and the sub-predicate
                shared_var_columns = [c for c in (set(predicate_df.columns) & set(sub_predicate_df.columns) & used_variables)]

                # Join the two dataframes based on the trace identifier, domain, and shared variable columns
                predicate_df = predicate_df.join(sub_predicate_df, how="inner", on=["trace_id", "domain"] + shared_var_columns)

                # Replace the intervals column with the intersection of the current intervals and the new ones from the sub-predicate
                predicate_df.replace("intervals", predicate_df.select("intervals", "intervals_right").apply(
                    self._intersect_intervals_tuple, INTERVALS_LIST_POLARS_TYPE)['column_0'])

                # Remove all the 'right-hand' columns added by the join
                predicate_df = predicate_df.drop([c for c in predicate_df.columns if c.endswith("_right")])

                # Remove any rows with empty intervals
                predicate_df = predicate_df.filter(pl.col("intervals").list.lengths() > 0)

            if DEBUG:
                end = time.perf_counter()
                print(f"Time to AND: {'%.5f' % (end - start)}s")  # type: ignore

        elif predicate_rule == "super_predicate_or":
            or_args = predicate["or_args"]
            if isinstance(or_args, tatsu.ast.AST):
                or_args = [or_args]

            sub_predicate_dfs = [self._inner_filter(or_arg, mapping, used_variables) for or_arg in or_args]  # type: ignore

            predicate_df = sub_predicate_dfs[0]

            if DEBUG: start = time.perf_counter()
            for i, sub_predicate_df in enumerate(sub_predicate_dfs[1:]):
                # Same procedure as with 'and', above, except a union instead of an intersection for the intervals
                shared_var_columns = [c for c in (set(predicate_df.columns) & set(sub_predicate_df.columns) & used_variables)]
                predicate_df = predicate_df.join(sub_predicate_df, how="outer", on=["trace_id", "domain"] + shared_var_columns)
                predicate_df.replace("intervals", predicate_df.select("intervals", "intervals_right").apply(self._union_intervals_tuple, INTERVALS_LIST_POLARS_TYPE)['column_0'])
                predicate_df = predicate_df.drop([c for c in predicate_df.columns if c.endswith("_right")])
                predicate_df = predicate_df.filter(pl.col("intervals").list.lengths() > 0)

            if DEBUG:
                end = time.perf_counter()
                print(f"Time to OR: {'%.5f' % (end - start)}s")  # type: ignore

        elif predicate_rule == "super_predicate_not":
            predicate_df = self._inner_filter(predicate["not_args"], mapping, used_variables)  # type: ignore
            if DEBUG: start = time.perf_counter()

            # For each trace ID, and each assignment of the vars that exist in the sub_predicate_df so far:
            relevant_vars = [c for c in predicate_df.columns if c in used_variables]
            relevant_var_mapping = {var: mapping[var] for var in relevant_vars}
            variable_types = tuple(tuple(relevant_var_mapping[var]) for var in relevant_var_mapping.keys())

            # For each cartesian product of the valid assignments for those vars given the domain
            possible_arg_assignments = [self._object_assignments(domain, variable_types) for domain in self.domains]

            possible_assignments_df = pl.DataFrame(dict(domain=self.domains, assignments=possible_arg_assignments, intervals=[[]] * len(self.domains)),
                                                   schema=dict(domain=None, assignments=None, intervals=pl.List(pl.List(pl.Int64))))  # type: ignore

            potential_missing_values_df = self.trace_lengths_and_domains_df.join(possible_assignments_df, how="left", on="domain")
            potential_missing_values_df = potential_missing_values_df.explode('assignments').select(
                'domain', 'trace_id', 'trace_length',
                pl.col("assignments").list.to_struct(fields=relevant_vars), 'intervals').unnest('assignments')

            # trace_ids = []
            # domains = []
            # assignment_columns = {var: [] for var in relevant_vars}
            # for trace_id, (_, domain) in self.trace_lengths_and_domains.items():

            #     possible_arg_assignments = self._object_assignments(domain, variable_types)
            #     trace_ids.extend([trace_id] * len(possible_arg_assignments))
            #     domains.extend([domain] * len(possible_arg_assignments))
            #     for arg_ids in possible_arg_assignments:
            #         for var, id in zip(relevant_vars, arg_ids):
            #             assignment_columns[var].append(id)

            # intervals = [[] for _ in range(len(trace_ids))]

            # # If they're missing in the sub_predicate_df, add them, with an empty interval
            # potential_missing_values_df = pl.DataFrame(dict(trace_id=trace_ids, domain=domains, intervals=intervals, **assignment_columns))

            # Now, for each possible combination of args on each trace / domain, 'intervals' will contain the truth intervals if
            # they exist and null otherwise, and 'intervals_right' will contain the empty interval
            predicate_df = predicate_df.join(potential_missing_values_df, how="outer", on=["trace_id", "domain"] + relevant_vars)

            # Union with the empty intervals will do nothing when they exist, and leave an empty interval when they don't
            predicate_df.replace("intervals", predicate_df.select("intervals", "intervals_right").apply(self._union_intervals_tuple, INTERVALS_LIST_POLARS_TYPE)["column_0"])

            # Invert intervals will then flip them to be the entire length of the trace
            predicate_df.replace("intervals", predicate_df.select("intervals", "trace_length").apply(self._invert_intervals_tuple_apply, INTERVALS_LIST_POLARS_TYPE)["column_0"])
            predicate_df = predicate_df.drop([c for c in predicate_df.columns if c.endswith("_right")] + ['trace_length'])

            if DEBUG:
                end = time.perf_counter()
                print(f"Time to NOT: {'%.5f' % (end - start)}s")  # type: ignore

        # elif predicate_rule == "super_predicate_exists":
        #     variable_type_mapping = extract_variable_type_mapping(predicate["exists_vars"]["variables"])  # type: ignore

        #     variables = extract_variables(predicate)
        #     unused_variables = [var for var in mapping.keys() if var not in variables]
        #     unused_variable_types = [mapping[var] for var in unused_variables]

        #     interval_mapping = defaultdict(lambda: defaultdict(list))
        #     sub_intervals = self._inner_filter(predicate["exists_args"], {**mapping, **variable_type_mapping}, used_variables)

        #     # Groups the intervals by the part of the mapping that *isn't* within the (exists)
        #     def keyfunc(element):
        #         key = tuple(sorted(elem for elem in element if elem.split('->')[0] not in variable_type_mapping.keys()))
        #         return key

        #     for id in sub_intervals:
        #         sorted_arg_ids = sorted(sub_intervals[id].keys(), key=keyfunc)
        #         for key, group in groupby(sorted_arg_ids, keyfunc):

        #             used_variables = tuple(elem.split('->')[0] for elem in key)
        #             used_objects = tuple(elem.split('->')[1] for elem in key)

        #             # As with [or], above, we need to compute the union of the indices in which the sub-predicate is true
        #             truth_idxs = [self._intervals_to_indices(sub_intervals[id][arg_ids]) for arg_ids in group]
        #             union = set.union(*truth_idxs)

        #             if len(union) > 0:

        #                 domain = self._domain_key(self.data[self.data["id"] == id]["domain"].unique()[0])
        #                 other_object_assignments = get_object_assignments(domain, unused_variable_types, used_objects=used_objects)
        #                 if len(other_object_assignments) == 0:
        #                     other_object_assignments = [()]

        #                 for assignment in other_object_assignments:
        #                     full_assignment = tuple(sorted([f"{var}->{id}" for var, id in zip(used_variables, used_objects)] +
        #                                                    [f"{var}->{id}" for var, id in zip(unused_variables, assignment)]))


        #                     interval_mapping[id][full_assignment] = self._indices_to_intervals(union)

        #     return interval_mapping

        # elif predicate_rule == "super_predicate_forall":
        #     variable_type_mapping = extract_variable_type_mapping(predicate["forall_vars"]["variables"])  # type: ignore

        #     variables = extract_variables(predicate)
        #     unused_variables = [var for var in mapping.keys() if var not in variables]
        #     unused_variable_types = [mapping[var] for var in unused_variables]

        #     interval_mapping = defaultdict(lambda: defaultdict(list))
        #     sub_intervals = self._inner_filter(predicate["forall_args"], {**mapping, **variable_type_mapping}, used_variables)

        #     # Groups the intervals by the part of the mapping that *isn't* within the (forall)
        #     def keyfunc(element):
        #         key = tuple(sorted(elem for elem in element if elem.split('->')[0] not in variable_type_mapping.keys()))
        #         return key

        #     for id in sub_intervals:
        #         sorted_arg_ids = sorted(sub_intervals[id].keys(), key=keyfunc)
        #         for key, group in groupby(sorted_arg_ids, keyfunc):

        #             used_variables = tuple(elem.split('->')[0] for elem in key)
        #             used_objects = tuple(elem.split('->')[1] for elem in key)

        #             # TODO

        else:
            raise ValueError(f"Error: Unknown rule '{predicate_rule}'")

        # Temporarily disable caching
        # if predicate_rule in self.cache_rules:
        #     self.predicate_interval_cache[ast_str] = interval_mapping

        return predicate_df



CURRENT_TEST_TRACE_NAMES = [
    '1HOTuIZpRqk2u1nZI1v1-gameplay-attempt-1-rerecorded',
    'IvoZWi01FO2uiNpNHyci-createGame-rerecorded',
    '4WUtnD8W6PGVy0WBtVm4-gameplay-attempt-1-rerecorded',
    'LTZh4k4THamxI5QJfVrk-gameplay-attempt-1-rerecorded',
    'WtZpe3LQFZiztmh7pBBC-gameplay-attempt-1-rerecorded',
    'FyGQn1qJCLTLU1hfQfZ2-preCreateGame-rerecorded',
    '6ZjBeRCvHxG05ORmhInj-gameplay-attempt-1-rerecorded',
    'Tcfpwc8v8HuKRyZr5Dyc-gameplay-attempt-2-rerecorded',
    '4WUtnD8W6PGVy0WBtVm4-createGame-rerecorded',
    '39PytL3fAMFkYXNoB5l6-gameplay-attempt-1-rerecorded',
    '5lTRHBueXsaOu9yhvOQo-gameplay-attempt-1-rerecorded',
    'SQErBa5s5TPVxmm8R6ks-freePlay-rerecorded',
    '9C0wMm4lzrJ5JeP0irIu-preCreateGame-rerecorded',
    'f2WUeVzu41E9Lmqmr2FJ-preCreateGame-rerecorded',
    '6XD5S6MnfzAPQlsP7k30-gameplay-attempt-2-rerecorded',
    'xMUrxzK3fXjgitdzPKsm-freePlay-rerecorded',
    'IhOkh1l3TBY9JJVubzHx-gameplay-attempt-1-rerecorded',
    'WtZpe3LQFZiztmh7pBBC-createGame-rerecorded',
    'vfh1MTEQorWXKy8jOP1x-gameplay-attempt-2-rerecorded',
    'LTZh4k4THamxI5QJfVrk-preCreateGame-rerecorded',
    '79X7tsrbEIu5ffDGnY8q-gameplay-attempt-1-rerecorded',
    'jCc0kkmGUg3xUmUSXg5w-gameplay-attempt-1-rerecorded',
    'ktwB7wT09sh4ivNme3Dw-createGame-rerecorded',
    'ktwB7wT09sh4ivNme3Dw-preCreateGame-rerecorded',
    'ktwB7wT09sh4ivNme3Dw-gameplay-attempt-1-rerecorded',
    'vfh1MTEQorWXKy8jOP1x-createGame-rerecorded',
    'QclKeEZEVr7j0klPuanE-gameplay-attempt-1-rerecorded',
    'jCc0kkmGUg3xUmUSXg5w-preCreateGame-rerecorded',
    'FyGQn1qJCLTLU1hfQfZ2-freePlay-rerecorded',
    'SQErBa5s5TPVxmm8R6ks-editGame-rerecorded',
    'IvoZWi01FO2uiNpNHyci-freePlay-rerecorded',
    'IvoZWi01FO2uiNpNHyci-preCreateGame-rerecorded',
    'SQErBa5s5TPVxmm8R6ks-preCreateGame-rerecorded',
    '9dQSLmtxxIBy0Rsnc8uu-freePlay-rerecorded',
    'FyGQn1qJCLTLU1hfQfZ2-createGame-rerecorded',
    'IhOkh1l3TBY9JJVubzHx-freePlay-rerecorded',
    '7r4cgxJHzLJooFaMG1Rd-gameplay-attempt-1-rerecorded',
    '79X7tsrbEIu5ffDGnY8q-preCreateGame-rerecorded',
    '6XD5S6MnfzAPQlsP7k30-freePlay-rerecorded',
    '9C0wMm4lzrJ5JeP0irIu-gameplay-attempt-1-rerecorded',
    'vfh1MTEQorWXKy8jOP1x-preCreateGame-rerecorded',
    'QyX7AlBzBW8hZHsJeDWI-gameplay-attempt-2-rerecorded',
    'SQErBa5s5TPVxmm8R6ks-gameplay-attempt-1-rerecorded',
    'NJUY0YT1Pq6dZXsmw0wE-gameplay-attempt-2-rerecorded',
    '9dQSLmtxxIBy0Rsnc8uu-createGame-rerecorded',
    'IvoZWi01FO2uiNpNHyci-gameplay-attempt-2-rerecorded',
    'f2WUeVzu41E9Lmqmr2FJ-gameplay-attempt-1-rerecorded',
    'QclKeEZEVr7j0klPuanE-gameplay-attempt-3-rerecorded',
    '4WUtnD8W6PGVy0WBtVm4-freePlay-rerecorded',
    '9dQSLmtxxIBy0Rsnc8uu-gameplay-attempt-1-rerecorded',
    '6XD5S6MnfzAPQlsP7k30-preCreateGame-rerecorded',
    'Tcfpwc8v8HuKRyZr5Dyc-createGame-rerecorded',
    'Tcfpwc8v8HuKRyZr5Dyc-gameplay-attempt-1-rerecorded',
    '5lTRHBueXsaOu9yhvOQo-preCreateGame-rerecorded',
    'IhOkh1l3TBY9JJVubzHx-createGame-rerecorded',
    'QclKeEZEVr7j0klPuanE-gameplay-attempt-2-rerecorded',
    'Tcfpwc8v8HuKRyZr5Dyc-preCreateGame-rerecorded',
    'R9nZAvDq7um7Sg49yf8T-preCreateGame-rerecorded',
    '7r4cgxJHzLJooFaMG1Rd-createGame-rerecorded',
    'QyX7AlBzBW8hZHsJeDWI-preCreateGame-rerecorded',
    'R9nZAvDq7um7Sg49yf8T-gameplay-attempt-1-rerecorded',
    '1HOTuIZpRqk2u1nZI1v1-preCreateGame-rerecorded',
    'xMUrxzK3fXjgitdzPKsm-gameplay-attempt-1-rerecorded',
    'FyGQn1qJCLTLU1hfQfZ2-gameplay-attempt-1-rerecorded',
    '9C0wMm4lzrJ5JeP0irIu-createGame-rerecorded',
    '4WUtnD8W6PGVy0WBtVm4-editGame-rerecorded',
    'NJUY0YT1Pq6dZXsmw0wE-preCreateGame-rerecorded',
    '4WUtnD8W6PGVy0WBtVm4-preCreateGame-rerecorded',
    'xMUrxzK3fXjgitdzPKsm-preCreateGame-rerecorded',
    'NJUY0YT1Pq6dZXsmw0wE-createGame-rerecorded',
    '6ZjBeRCvHxG05ORmhInj-preCreateGame-rerecorded',
    '39PytL3fAMFkYXNoB5l6-createGame-rerecorded',
    'QyX7AlBzBW8hZHsJeDWI-gameplay-attempt-3-rerecorded',
    'f2WUeVzu41E9Lmqmr2FJ-createGame-rerecorded',
    '79X7tsrbEIu5ffDGnY8q-createGame-rerecorded',
    'jCc0kkmGUg3xUmUSXg5w-gameplay-attempt-2-rerecorded',
    '7r4cgxJHzLJooFaMG1Rd-preCreateGame-rerecorded'
]


def _print_results_as_expected_intervals(filter_results):
    print(' ' * 8 + 'expected_intervals={')
    for key, intervals in filter_results.items():
        print(f'{" " * 12}{key}: {intervals},')
    print(' ' * 8 + '}')


if __name__ == '__main__':

    DEFAULT_GRAMMAR_PATH = "./dsl/dsl.ebnf"
    grammar = open(DEFAULT_GRAMMAR_PATH).read()
    grammar_parser = typing.cast(tatsu.grammars.Grammar, tatsu.compile(grammar))

    game = open(get_project_dir() + '/reward-machine/games/ball_to_bin_from_bed.txt').read()
    game_ast = grammar_parser.parse(game)  # type: ignore

    # should be: (and (in_motion ?b) (not (agent_holds ?b)))
    test_pred_1 = game_ast[4][1]['preferences'][0]['definition']['forall_pref']['preferences']['pref_body']['body']['exists_args']['then_funcs'][1]['seq_func']['hold_pred']

    # should be: (and (not (in_motion ?b)) (in ?h ?b)))
    test_pred_2 = game_ast[4][1]['preferences'][0]['definition']['forall_pref']['preferences']['pref_body']['body']['exists_args']['then_funcs'][2]['seq_func']['once_pred']

    # should be: (once (and (not (in_motion ?b) (exists (?c - hexagonal_bin) (in ?c ?b)))))
    # test_pred_3 = game_ast[4][1]['preferences'][0]['definition']['forall_pref']['preferences']['pref_body']['body']['exists_args']['then_funcs'][3]['seq_func']['once_pred']

    block_stacking_game = open(get_project_dir() + '/reward-machine/games/block_stacking.txt').read()
    block_stacking_game_ast = grammar_parser.parse(block_stacking_game)  # type: ignore

    test_pred_or = block_stacking_game_ast[3][1]['preferences'][0]['definition']['pref_body']['body']['exists_args']['then_funcs'][2]['seq_func']['hold_pred']
    test_pred_desk_or = block_stacking_game_ast[3][1]['preferences'][1]['definition']['pref_body']['body']['exists_args']['at_end_pred']

    BALL_TO_BIN_FROM_BED_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/FhhBELRaBFiGGvX0YG7W-preCreateGame-rerecorded.json')
    agent_adj_game = open(get_project_dir() + '/reward-machine/games/test_agent_door_adjacent.txt').read()
    agent_adj_game_ast = grammar_parser.parse(agent_adj_game)  # type: ignore

    agent_adj_predicate = agent_adj_game_ast[3][1]['preferences'][0]['definition']['pref_body']['body']['exists_args']['then_funcs'][0]['seq_func']['once_pred']
    agent_adj_mapping = {"?d": ["ball"]}


    test_mapping = {"?b": ["ball"], "?h": ["hexagonal_bin"]}
    block_test_mapping = {"?b1": ['cube_block'], "?b2": ["cube_block"]}
    block_desk_test_mapping = {"?b": ["block"]}
    all_block_test_mapping = {"?b1": ["block"], "?b2": ["block"]}

    stats = CommonSensePredicateStatisticsSplitArgs(cache_dir=DEFAULT_CACHE_DIR,
                                                    trace_names=CURRENT_TEST_TRACE_NAMES,
                                                    cache_rules=[],
                                                    base_trace_path=DEFAULT_BASE_TRACE_PATH,
                                                    overwrite=False)
    out = stats.filter(test_pred_2, test_mapping)
    _print_results_as_expected_intervals(out)

    exit()


    trace_paths = glob.glob(f"{get_project_dir()}/reward-machine/traces/participant-traces/*.json")

    stats = CommonSensePredicateStatisticsSplitArgs(cache_dir, trace_paths, cache_rules=["predicate", "super_predicate_and",
                                           "super_predicate_or", "super_predicate_not"], overwrite=False)
    out = stats.filter(agent_adj_predicate, agent_adj_mapping)
    _print_results_as_expected_intervals(out)

    exit()

    tracer = None
    if PROFILE:
        tracer = VizTracer(10000000)
        tracer.start()

    test_out = stats.filter(test_pred_desk_or, block_desk_test_mapping)
    _print_results_as_expected_intervals(test_out)
    start = time.perf_counter()
    N_ITER = 100
    for i in range(N_ITER):
        # print(f"\n====================")
        inner_start = time.perf_counter()
        stats.filter(test_pred_2, test_mapping)
        # inner_end = time.perf_counter()
        # print(f"Time per iteration: {'%.5f' % (inner_end - inner_start)}s")
    end = time.perf_counter()
    print(f"Time per iteration: {'%.5f' % ((end - start) / N_ITER)}s")

    if tracer is not None:
        tracer.stop()
        profile_output_path = os.path.join(get_project_dir(), 'reward-machine/temp/viztracer_split_args.json')
        print(f'Saving profile to {profile_output_path}')
        tracer.save(profile_output_path)

    exit()

    # Satisfactions of in_motion
    in_motion_sats = stats.data[stats.data["predicate"] == "in_motion"]

    print("All 'in_motion' satisfactions:")
    print(in_motion_sats)

    # Satisfactions of in
    in_sats = stats.data[stats.data["predicate"] == "in"]
    long_in_sats = in_sats[(in_sats["end_step"] - in_sats["start_step"]) / in_sats["replay_len"] >= 0.9]

    print("All 'in' satisfactions:")
    print(in_sats[["predicate", "arg_ids", "start_step", "end_step", "replay_len"]])

    print("\n\nLong 'in' satisfactions (>90pct of trace):")
    print(long_in_sats[["predicate", "arg_ids", "start_step", "end_step", "replay_len"]])

    # Satisfactions of agent_holds
    print(stats.data[stats.data["predicate"] == "agent_holds"])

    # stats = CommonSensePredicateStatistics(cache_dir, [f"{get_project_dir()}/reward-machine/traces/{trace}-rerecorded.json" for trace in TEST_TRACE_NAMES], overwrite=True)
    # print(stats.data)
