import cachetools
import cProfile
import duckdb
from functools import reduce
import glob
import gzip
import hashlib
import heapq
import io
from itertools import chain, groupby, permutations, product, repeat, starmap
import json
import logging
import os
import operator
import pandas as pd
import pathlib
import pickle
import polars as pl
pl.enable_string_cache(True)
import pstats
import signal
import tatsu, tatsu.ast, tatsu.grammars
import time
from tqdm import tqdm
import typing
from viztracer import VizTracer


from config import ROOMS, META_TYPES, TYPES_TO_META_TYPE, OBJECTS_BY_ROOM_AND_TYPE, ORIENTATIONS, SIDES, UNITY_PSEUDO_OBJECTS, NAMED_WALLS, SPECIFIC_NAMED_OBJECTS_BY_ROOM, OBJECT_ID_TO_SPECIFIC_NAME_BY_ROOM, GAME_OBJECT, GAME_OBJECT_EXCLUDED_TYPES
from utils import (extract_predicate_function_name,
                   extract_variables,
                   extract_variable_type_mapping,
                   get_project_dir,
                   get_object_assignments,
                   FullState,
                   AgentState)
from ast_parser import PREFERENCES
from ast_printer import ast_section_to_string
from building_handler import BuildingHandler
from predicate_handler import PREDICATE_LIBRARY_RAW


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


COMMON_SENSE_PREDICATES_AND_FUNCTIONS = (
    ("above", 2),
    ("adjacent", 2),
    ("agent_crouches", 0),
    ("agent_holds", 1),
    ("broken", 1),
    ("equal_x_position", 2),
    ("equal_z_position", 2),
    ("game_start", 0),
    ("game_over", 0),
    ("in", 2),
    ("in_motion", 1),
    ("object_orientation", 1),  # as it takes 1 object and an orientation we'll hard-code
    ("on", 2),
    ("open", 1),
    ("toggled_on", 1),
    ("touch", 2),
    # ("between", 3),
)

INTERVALS_LIST_POLARS_TYPE = pl.List(pl.List(pl.Int64))


# Maps from types returned by unity to the types used in the DSL
TYPE_REMAP = {
    "alarmclock": "alarm_clock",
    "bridgeblock": "bridge_block",
    "creditcard": "credit_card",
    "cubeblock": "cube_block",
    "curvedramp": "curved_wooden_ramp",
    "cylinderblock": "cylindrical_block",
    "desklamp": "lamp",
    "dogbed": "doggie_bed",
    "flatrectblock": "flat_block",
    "garbagecan": "hexagonal_bin",
    "keychain": "key_chain",
    "longcylinderblock": "tall_cylindrical_block",
    "lightswitch": "main_light_switch",
    "pyramidblock": "pyramid_block",
    "sidetable": "side_table",
    "smallslide": "triangular_ramp",
    "tallrectblock": "tall_rectangular_block",
    "teddybear": "teddy_bear",
    "triangleblock": "triangle_block"
}

DEBUG = False
PROFILE = True
DEFAULT_CACHE_DIR = pathlib.Path(get_project_dir() + '/reward-machine/caches')
DEFAULT_CACHE_FILE_NAME_FORMAT = 'predicate_statistics_bitstring_intervals_{traces_hash}.pkl.gz'
NO_INTERVALS_CACHE_FILE_NAME_FORMAT = 'predicate_statistics_no_intervals_{traces_hash}.pkl.gz'
DEFAULT_TRACE_LENGTHS_FILE_NAME_FORMAT = 'trace_lengths_{traces_hash}.pkl'
DEFAULT_IN_PROCESS_TRACES_FILE_NAME_FORMAT = 'in_progress_traces_{traces_hash}.pkl'
DEFAULT_BASE_TRACE_PATH = os.path.join(os.path.dirname(__file__), "traces/participant-traces/")
CLUSTER_BASE_TRACE_PATH = '/misc/vlgscratch4/LakeGroup/guy/participant-traces'


DEFAULT_COLUMNS = ['predicate', 'arg_1_id', 'arg_1_type', 'arg_2_id', 'arg_2_type', 'trace_id', 'domain', 'intervals']
FULL_PARTICIPANT_TRACE_SET = [os.path.splitext(os.path.basename(t))[0] for t in glob.glob(os.path.join(DEFAULT_BASE_TRACE_PATH, '*.json'))]


class PredicateNotImplementedException(Exception):
    pass


class MissingVariableException(Exception):
    pass


class QueryTimeoutException(Exception):
    pass


class Timeout:
    def __init__(self, seconds=1, message="Timed out"):
        self._seconds = seconds
        self._message = message

    @property
    def seconds(self):
        return self._seconds

    @property
    def message(self):
        return self._message

    @property
    def handler(self):
        return self._handler

    @handler.setter
    def handler(self, handler):
        self._handler = handler

    def handle_timeout(self, *_):
        raise QueryTimeoutException(self.message)

    def __enter__(self):
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *_):
        signal.alarm(0)


def raise_query_timeout(*args, **kwargs):
    raise QueryTimeoutException("Query timed out")


alarm_handler = signal.getsignal(signal.SIGALRM)
if alarm_handler is not None and alarm_handler is not signal.SIG_DFL and alarm_handler is not signal.SIG_IGN:
    signal.signal(signal.SIGALRM, raise_query_timeout)



def stable_hash(str_data: str):
    return hashlib.md5(bytearray(str_data, 'utf-8')).hexdigest()


def stable_hash_list(list_data: typing.Sequence[str]):
    return stable_hash('\n'.join(sorted(list_data)))


DEBUG = False
MAX_CACHE_SIZE = 2 ** 12
MAX_CHILD_ARGS = 4  # 6
DEFAULT_QUERY_TIMEOUT = 5  # seconds

class CommonSensePredicateStatisticsFullDatabase():
    def __init__(self,
                 cache_dir: typing.Union[str, pathlib.Path] = DEFAULT_CACHE_DIR,
                 cache_filename_format: str = DEFAULT_CACHE_FILE_NAME_FORMAT,
                 trace_lengths_filename_format: str = DEFAULT_TRACE_LENGTHS_FILE_NAME_FORMAT,
                 force_trace_names_hash: typing.Optional[str] = None,
                 trace_folder: typing.Optional[str] = None,
                 max_cache_size: int = MAX_CACHE_SIZE,
                 max_child_args: int = MAX_CHILD_ARGS,
                 query_timeout: int = DEFAULT_QUERY_TIMEOUT,
                 ):

        self.cache = cachetools.LRUCache(maxsize=max_cache_size)
        self.temp_table_index = -1
        self.temp_table_prefix = 't'
        self.max_child_args = max_child_args
        self.query_timeout = query_timeout

        if trace_folder is None:
            if os.path.exists(CLUSTER_BASE_TRACE_PATH):
                trace_folder = CLUSTER_BASE_TRACE_PATH
            else:
                trace_folder = DEFAULT_BASE_TRACE_PATH

        self.all_trace_ids = [os.path.splitext(os.path.basename(t))[0] for t in glob.glob(os.path.join(trace_folder, '*.json'))]
        self.all_predicates = set([t[0] for t in COMMON_SENSE_PREDICATES_AND_FUNCTIONS])
        self.all_types = set(reduce(lambda x, y: x + y, [list(x.keys()) for x in chain(OBJECTS_BY_ROOM_AND_TYPE.values(), SPECIFIC_NAMED_OBJECTS_BY_ROOM.values())]))
        self.all_types.remove(GAME_OBJECT)
        self.all_arg_ids = set(reduce(lambda x, y: x + y, [object_types for room_types in OBJECTS_BY_ROOM_AND_TYPE.values() for object_types in room_types.values()]))
        self.all_arg_ids.update(UNITY_PSEUDO_OBJECTS.keys())

        self.trace_names_hash = force_trace_names_hash
        self.stats_filename = os.path.join(cache_dir, cache_filename_format.format(traces_hash=self.trace_names_hash))
        self.trace_lengths_and_domains_filename = os.path.join(cache_dir, trace_lengths_filename_format.format(traces_hash=self.trace_names_hash))

        self._create_databases()

    # def __getstate__(self) -> typing.Dict[str, typing.Any]:
    #     state = self.__dict__.copy()
    #     return state

    def __setstate__(self, state: typing.Dict[str, typing.Any]) -> None:
        self.__dict__.update(state)
        self._create_databases()

    def _create_databases(self):
        table_query = duckdb.sql("SHOW TABLES")
        if table_query is not None:
            all_tables = set(t.lower() for t in chain.from_iterable(table_query.fetchall()))
            if 'data' in all_tables:
                # logger.info('Skipping creating tables because they already exist')
                return

        logger.info("Loading data from files")
        open_method = gzip.open if self.stats_filename.endswith('.gz') else open

        if os.path.exists(self.stats_filename):
            data_df = pd.read_pickle(self.stats_filename)

        else:
            raise ValueError(f"Could not find file {self.stats_filename}")

        if os.path.exists(self.trace_lengths_and_domains_filename):
            with open_method(self.trace_lengths_and_domains_filename, 'rb') as f:
                trace_lengths_and_domains = pickle.load(f)

        else:
            raise ValueError(f"Could not find file {self.trace_lengths_and_domains_filename}")

        logger.info("Creating DuckDB table...")
        duckdb.sql(f"set temp_directory='/tmp/duckdb/{os.getpid()}';")
        duckdb.sql(f"set enable_progress_bar=false;")
        duckdb.sql(f"set enable_progress_bar_print=false;")

        duckdb.sql(f"CREATE TYPE domain AS ENUM {tuple(ROOMS)};")
        duckdb.sql(f"CREATE TYPE trace_id AS ENUM {tuple(self.all_trace_ids)};")
        duckdb.sql(f"CREATE TYPE predicate AS ENUM {tuple(sorted(self.all_predicates))};")
        duckdb.sql(f"CREATE TYPE arg_type AS ENUM {tuple(sorted(self.all_types))};")
        duckdb.sql(f"CREATE TYPE arg_id AS ENUM {tuple(sorted(self.all_arg_ids))};")

        trace_length_and_domain_rows = [(trace_id, domain, length) for (trace_id, (length, domain)) in trace_lengths_and_domains.items()]
        duckdb.sql('CREATE TABLE trace_length_and_domains(trace_id trace_id PRIMARY KEY, domain domain NOT NULL, length INTEGER NOT NULL);')
        duckdb.sql(f'INSERT INTO trace_length_and_domains VALUES {str(tuple(trace_length_and_domain_rows))[1:-1]}')
        duckdb.sql('CREATE INDEX idx_tld_domain ON trace_length_and_domains (domain)')

        duckdb.sql("CREATE TABLE empty_bitstrings(trace_id trace_id PRIMARY KEY, intervals BITSTRING NOT NULL);")
        duckdb.sql("INSERT INTO empty_bitstrings SELECT trace_id, BITSTRING('0', length) as intervals FROM trace_length_and_domains")

        data_rows = []
        for domain in ROOMS:
            for object_dict in (OBJECTS_BY_ROOM_AND_TYPE[domain], SPECIFIC_NAMED_OBJECTS_BY_ROOM[domain]):
                for object_type, object_ids in object_dict.items():
                    if object_type in self.all_types:
                        for object_id in object_ids:
                            data_rows.append((domain, object_type, object_id))

        duckdb.sql('CREATE TABLE object_type_to_id(domain domain NOT NULL, type arg_type NOT NULL, object_id arg_id NOT NULL);')
        duckdb.sql(f'INSERT INTO object_type_to_id VALUES {str(tuple(data_rows))[1:-1]}')
        duckdb.sql('CREATE INDEX idx_obj_id_domain ON object_type_to_id (domain)')
        duckdb.sql('CREATE INDEX idx_obj_id_type ON object_type_to_id (type)')
        duckdb.sql('CREATE INDEX idx_obj_id_id ON object_type_to_id (object_id)')

        duckdb.sql("CREATE TABLE data(predicate predicate NOT NULL, arg_1_id arg_id, arg_1_type arg_type, arg_2_id arg_id, arg_2_type arg_type, trace_id trace_id NOT NULL, domain domain NOT NULL, intervals BITSTRING NOT NULL);")
        duckdb.sql("INSERT INTO data SELECT * FROM data_df")

        duckdb.sql("INSERT INTO data (predicate, trace_id, domain, intervals) SELECT 'game_start' as predicate, trace_id, domain, bitstring('1', length) as intervals FROM trace_length_and_domains")
        duckdb.sql("INSERT INTO data (predicate, trace_id, domain, intervals) SELECT 'game_over' as predicate, trace_id, domain, set_bit(bitstring('0', length), 0, 1) as intervals FROM trace_length_and_domains")

        duckdb.sql('CREATE INDEX idx_data_trace_id ON data (trace_id)')
        duckdb.sql('CREATE INDEX idx_data_arg_1_id ON data (arg_1_id)')
        duckdb.sql('CREATE INDEX idx_data_arg_2_id ON data (arg_2_id)')
        data_rows = duckdb.sql("SELECT count(*) FROM data").fetchone()[0]  # type: ignore
        logger.info(f"Loaded data, found {data_rows} rows")
        del data_df
        del trace_lengths_and_domains

    def _table_name(self, index: int):
        return f"{self.temp_table_prefix}{index}"

    def _next_temp_table_index(self):
        self.temp_table_index += 1
        return self.temp_table_index

    def _next_temp_table_name(self):
        return self._table_name(self._next_temp_table_index())

    def _predicate_and_mapping_cache_key(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], *args, **kwargs) -> str:
        '''
        Returns a string that uniquely identifies the predicate and mapping
        '''
        return ast_section_to_string(predicate, PREFERENCES) + "_" + str(mapping)

    @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def filter(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], **kwargs):

        try:
            if self.temp_table_index > 2 ** 31:
                self.temp_table_index = -1

            result_query, _ = self._inner_filter(predicate, mapping, **kwargs)

            with Timeout(seconds=self.query_timeout):
                if 'return_full_result' in kwargs and kwargs['return_full_result']:
                    output_query = f"SELECT * FROM ({result_query})"
                    return duckdb.sql(output_query).fetchdf()
                else:
                    output_query = f"SELECT count(*) FROM ({result_query})"
                    return duckdb.sql(output_query).fetchone()[0]  # type: ignore

        except PredicateNotImplementedException as e:
            # Pass the exception through and let the caller handle it
            raise e

    # @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def _handle_predicate(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], return_trace_ids: bool = False, **kwargs) -> typing.Tuple[str, typing.Set[str]]:
        predicate_name = extract_predicate_function_name(predicate)  # type: ignore

        if predicate_name not in self.all_predicates:
            raise PredicateNotImplementedException(predicate_name)

        variables = extract_variables(predicate)  # type: ignore
        used_variables = set(variables)

        # Restrict the mapping to just the referenced variables and expand meta-types
        relevant_arg_mapping = {}
        for var in variables:
            if var in mapping:
                relevant_arg_mapping[var] = sum([META_TYPES.get(arg_type, [arg_type]) for arg_type in mapping[var]], [])

            # This handles variables which are referenced directly, like the desk and bed
            elif not var.startswith("?"):
                relevant_arg_mapping[var] = [var]

            else:
                raise MissingVariableException(f"Variable {var} is not in the mapping")

        select_items = ["trace_id", "domain", "intervals"]
        where_items = [f"predicate='{predicate_name}'"]

        for i, (arg_var, arg_types) in enumerate(relevant_arg_mapping.items()):
            # if it can be the generic object type, we filter for it specifically
            if GAME_OBJECT in arg_types:
                exclude_types = set(GAME_OBJECT_EXCLUDED_TYPES)
                for type in arg_types:
                    exclude_types.discard(type)

                where_items.append(f"arg_{i + 1}_type NOT IN {self._types_to_arg_casts(exclude_types)}")

            else:
                if len(arg_types) == 1:
                    where_items.append(f"(arg_{i + 1}_type='{arg_types[0]}')")
                else:
                    where_items.append(f"(arg_{i + 1}_type IN {self._types_to_arg_casts(arg_types)})")

            select_items.append(f'arg_{i + 1}_id AS "{arg_var}"')

        query = f"SELECT {', '.join(select_items)} FROM data WHERE {' AND '.join(where_items)}"
        return query, used_variables

    # @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def _handle_and(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], **kwargs) -> typing.Tuple[str, typing.Set[str]]:
        and_args = predicate["and_args"]
        if not isinstance(and_args, list):
            and_args = [and_args]

        if len(and_args) > self.max_child_args:
            raise PredicateNotImplementedException("Too many and args")

        sub_queries = []
        used_variables_by_child = []
        all_used_variables = set()

        for and_arg in and_args:  # type: ignore
            try:
                sub_query, sub_used_variables = self._inner_filter(and_arg, mapping)  # type: ignore
                sub_queries.append(sub_query)
                used_variables_by_child.append(sub_used_variables)
                all_used_variables |= sub_used_variables

            except PredicateNotImplementedException as e:
                continue

        if len(sub_queries) == 0:
            raise PredicateNotImplementedException("All sub-predicates of the and were not implemented")

        if len(sub_queries) == 1:
            return sub_queries[0], used_variables_by_child[0]

        subquery_table_names = [self._next_temp_table_name() for _ in range(len(sub_queries))]
        with_items = [f"{table_name} AS ({subquery})" for table_name, subquery in zip(subquery_table_names, sub_queries)]

        select_items = [f"{subquery_table_names[0]}.trace_id", f"{subquery_table_names[0]}.domain"]
        selected_variables = set()
        intervals = []
        join_clauses = []

        for i, (table_name, sub_used_variables) in enumerate(zip(subquery_table_names, used_variables_by_child)):
            intervals.append(f"{table_name}.intervals")

            for variable in sub_used_variables:
                if variable not in selected_variables:
                    select_items.append(f'{table_name}."{variable}"')
                    selected_variables.add(variable)

            if i > 0:
                join_parts = [f"INNER JOIN {table_name} ON ({subquery_table_names[0]}.trace_id={table_name}.trace_id)"]
                joined_variables = set()

                for prev_table_name, prev_used_variables in zip(subquery_table_names[:i], used_variables_by_child[:i]):
                    shared_variables = sub_used_variables & prev_used_variables - joined_variables
                    join_parts.extend([f'({table_name}."{v}"={prev_table_name}."{v}")' for v in shared_variables])
                    joined_variables |= shared_variables

                join_clauses.append(" AND ".join(join_parts))


        select_items.append(f'({" & ".join(intervals)}) AS intervals')

        inner_query = f"WITH {', '.join(with_items)} SELECT {', '.join(select_items)} FROM {subquery_table_names[0]} {' '.join(join_clauses)}"

        table_name = self._next_temp_table_name()
        query = f"WITH {table_name} AS ({inner_query}) SELECT * FROM {table_name} WHERE bit_count(intervals) != 0"
        if DEBUG: print(query)
        return query, all_used_variables

    # @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def _handle_and_de_morgans(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], **kwargs) -> typing.Tuple[str, typing.Set[str]]:
        and_args = predicate["and_args"]
        if not isinstance(and_args, list):
            and_args = [and_args]

        if len(and_args) > self.max_child_args:
            raise PredicateNotImplementedException("Too many and args")

        sub_queries = []
        used_variables_by_child = []
        all_used_variables = set()

        for and_arg in and_args:  # type: ignore
            try:
                subquery, sub_used_variables = self._inner_filter(and_arg, mapping)  # type: ignore
                sub_queries.append(subquery)
                used_variables_by_child.append(sub_used_variables)
                all_used_variables |= sub_used_variables

            except PredicateNotImplementedException as e:
                continue

        if len(sub_queries) == 0:
            raise PredicateNotImplementedException("All sub-predicates of the or were not implemented")

        if len(sub_queries) == 1:
            return sub_queries[0], used_variables_by_child[0]

        sub_queries.insert(0, self._build_potential_missing_values_query(mapping, list(all_used_variables)))
        used_variables_by_child.insert(0, all_used_variables)

        subquery_table_names = [self._next_temp_table_name() for _ in range(len(sub_queries))]

        with_items = [f"{table_name} AS ({subquery})" for table_name, subquery in zip(subquery_table_names, sub_queries)]

        select_items = [f"{subquery_table_names[0]}.trace_id", f"{subquery_table_names[0]}.domain"]
        selected_variables = set()
        intervals = []
        join_clauses = []

        for i, (sub_table_name, sub_used_variables) in enumerate(zip(subquery_table_names, used_variables_by_child)):
            intervals.append(f"{sub_table_name}.intervals")

            for variable in sub_used_variables:
                if variable not in selected_variables:
                    select_items.append(f'{sub_table_name}."{variable}"')
                    selected_variables.add(variable)

            if i > 0:
                join_parts = [f"LEFT JOIN {sub_table_name} ON ({subquery_table_names[0]}.trace_id={sub_table_name}.trace_id)"]

                shared_variables = sub_used_variables & all_used_variables
                join_parts.extend([f'({subquery_table_names[0]}."{v}"={sub_table_name}."{v}")' for v in shared_variables])

                join_clauses.append(" AND ".join(join_parts))

        intervals_coalesce = [f"~{intervals_select}" if i > 0 else intervals_select for i, intervals_select in enumerate(intervals)]
        select_items.append(f'~({" | ".join(intervals_coalesce)}) AS intervals')

        inner_query = f"WITH {', '.join(with_items)} SELECT {', '.join(select_items)} FROM {subquery_table_names[0]} {' '.join(join_clauses)}"

        table_name = self._next_temp_table_name()
        query = f"WITH {table_name} AS ({inner_query}) SELECT * FROM {table_name} WHERE intervals IS NOT NULL AND (bit_count(intervals) != 0)"
        if DEBUG: print(query)
        return query, all_used_variables


    # @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def _handle_or(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], **kwargs) -> typing.Tuple[str, typing.Set[str]]:
        or_args = predicate["or_args"]
        if not isinstance(or_args, list):
            or_args = [or_args]

        if len(or_args) > self.max_child_args:
            raise PredicateNotImplementedException("Too many and args")

        sub_queries = []
        used_variables_by_child = []
        all_used_variables = set()

        for or_arg in or_args:  # type: ignore
            try:
                subquery, sub_used_variables = self._inner_filter(or_arg, mapping)  # type: ignore
                sub_queries.append(subquery)
                used_variables_by_child.append(sub_used_variables)
                all_used_variables |= sub_used_variables

            except PredicateNotImplementedException as e:
                continue

        if len(sub_queries) == 0:
            raise PredicateNotImplementedException("All sub-predicates of the or were not implemented")

        if len(sub_queries) == 1:
            return sub_queries[0], used_variables_by_child[0]

        sub_queries.insert(0, self._build_potential_missing_values_query(mapping, list(all_used_variables)))
        used_variables_by_child.insert(0, all_used_variables)

        subquery_table_names = [self._next_temp_table_name() for _ in range(len(sub_queries))]

        with_items = [f"{table_name} AS ({subquery})" for table_name, subquery in zip(subquery_table_names, sub_queries)]

        select_items = [f"{subquery_table_names[0]}.trace_id", f"{subquery_table_names[0]}.domain"]
        selected_variables = set()
        intervals = []
        join_clauses = []

        for i, (sub_table_name, sub_used_variables) in enumerate(zip(subquery_table_names, used_variables_by_child)):
            intervals.append(f"{sub_table_name}.intervals")

            for variable in sub_used_variables:
                if variable not in selected_variables:
                    select_items.append(f'{sub_table_name}."{variable}"')
                    selected_variables.add(variable)

            if i > 0:
                join_parts = [f"LEFT JOIN {sub_table_name} ON ({subquery_table_names[0]}.trace_id={sub_table_name}.trace_id)"]

                shared_variables = sub_used_variables & all_used_variables
                join_parts.extend([f'({subquery_table_names[0]}."{v}"={sub_table_name}."{v}")' for v in shared_variables])

                join_clauses.append(" AND ".join(join_parts))

        intervals_coalesce = [f"COALESCE({intervals_select}, {intervals[0]})" if i > 0 else intervals_select for i, intervals_select in enumerate(intervals)]
        select_items.append(f'({" | ".join(intervals_coalesce)}) AS intervals')

        inner_query = f"WITH {', '.join(with_items)} SELECT {', '.join(select_items)} FROM {subquery_table_names[0]} {' '.join(join_clauses)}"

        table_name = self._next_temp_table_name()
        query = f"WITH {table_name} AS ({inner_query}) SELECT * FROM {table_name} WHERE bit_count(intervals) != 0"
        if DEBUG: print(query)
        return query, all_used_variables

    def _types_to_arg_casts(self, types: typing.Collection[str]):
        return '(' + ', '.join(f"'{t}'::arg_type" for t in types) + ')'

    def _build_object_assignment_cte(self, var: str, object_types: typing.Union[str, typing.List[str]]):
        if isinstance(object_types, str) or len(object_types) == 1:
            if isinstance(object_types, list):
                object_types = object_types[0]

            if object_types == GAME_OBJECT:
                where_clause = f"type NOT IN {self._types_to_arg_casts(GAME_OBJECT_EXCLUDED_TYPES)}"
            else:
                where_clause = f"type = '{object_types}'"
        else:
            if GAME_OBJECT in object_types:
                exclude_types = set(GAME_OBJECT_EXCLUDED_TYPES)
                for type in object_types:
                    exclude_types.discard(type)

                where_clause = f"type NOT IN {self._types_to_arg_casts(exclude_types)}"
            else:
                where_clause = f"type IN {self._types_to_arg_casts(object_types)}"

        return f'SELECT domain, object_id AS "{var}" FROM object_type_to_id WHERE {where_clause}'

    def object_assignments_query(self, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]]):
        if len(mapping) == 0:
            return None

        if len(mapping) == 1:
            first_key = list(mapping.keys())[0]
            query = self._build_object_assignment_cte(first_key, mapping[first_key])

        else:
            object_id_selects = []
            ctes = []
            join_statements = []
            variables = list(mapping.keys())
            table_names = [self._next_temp_table_name() for _ in range(len(variables))]
            for i, (var, table_name) in enumerate(zip(variables, table_names)):
                var_types = mapping[var]
                ctes.append(f"{table_name} AS ({self._build_object_assignment_cte(var, var_types)})")
                object_id_selects.append(f'{table_name}."{var}" AS "{var}"')
                if i > 0:
                    join_clauses = []
                    join_clauses.append(f"({table_names[0]}.domain = {table_name}.domain)")
                    for j in range(i):
                        join_clauses.append(f'({table_names[j]}."{variables[j]}" != {table_name}."{var}")')

                    join_statements.append(f"JOIN {table_name} ON {' AND '.join(join_clauses)}")

            query = f"""WITH {', '.join(ctes)}
SELECT {table_names[0]}.domain, {', '.join(object_id_selects)} FROM {table_names[0]}
{' '.join(join_statements)}
"""

        return query

    def _build_potential_missing_values_query(self, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], relevant_vars: typing.List[str]):
        # For each trace ID, and each assignment of the vars that exist in the sub_predicate_df so far:
        relevant_var_mapping = {var: mapping[var] if var.startswith("?") else [var] for var in relevant_vars}

        object_assignments_query = self.object_assignments_query(relevant_var_mapping)

        select_variables = ', '.join(f'object_assignments."{var}" as "{var}"' for var in relevant_vars)
        query = f"SELECT trace_length_and_domains.trace_id as trace_id, trace_length_and_domains.domain as domain, empty_bitstrings.intervals as intervals, {select_variables} FROM trace_length_and_domains"

        if object_assignments_query is not None:
            query += f" JOIN ({object_assignments_query}) AS object_assignments ON (trace_length_and_domains.domain = object_assignments.domain)"

        query += " JOIN empty_bitstrings ON (trace_length_and_domains.trace_id = empty_bitstrings.trace_id)"

        if DEBUG: print(query)
        return query

    # @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def _handle_not(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], **kwargs) -> typing.Tuple[str, typing.Set[str]]:
        try:
            inner_query, used_variables = self._inner_filter(predicate["not_args"], mapping)  # type: ignore
        except PredicateNotImplementedException as e:
            raise PredicateNotImplementedException(f"Sub-predicate of the not ({e.args}) was not implemented")


        relevant_vars = list(used_variables)
        potential_missing_values_query = self._build_potential_missing_values_query(mapping, relevant_vars)
        potential_missing_values_table_name = self._next_temp_table_name()
        inner_table_name = self._next_temp_table_name()

        # Now, for each possible combination of args on each trace / domain, 'intervals' will contain the truth intervals if
        # they exist and null otherwise, and 'intervals_right' will contain the empty interval'
        join_columns = ["trace_id"] + relevant_vars

        select_items = [f"{potential_missing_values_table_name}.trace_id as trace_id", f"{potential_missing_values_table_name}.domain as domain"]
        select_items.extend(f'{potential_missing_values_table_name}."{var}" as "{var}"' for var in relevant_vars)
        select_items.append(f"(~( {potential_missing_values_table_name}.intervals | COALESCE({inner_table_name}.intervals, {potential_missing_values_table_name}.intervals) )) AS intervals")

        join_items = [f'{potential_missing_values_table_name}."{column}"={inner_table_name}."{column}"'  for column in join_columns]

        not_query = f"""WITH {potential_missing_values_table_name} AS ({potential_missing_values_query}), {inner_table_name} AS ({inner_query})
        SELECT {', '.join(select_items)} FROM {potential_missing_values_table_name} LEFT JOIN {inner_table_name} ON {' AND '.join(join_items)}
        """
        table_name = self._next_temp_table_name()
        query = f"WITH {table_name} AS ({not_query}) SELECT * FROM {table_name} WHERE bit_count(intervals) != 0"
        if DEBUG: print(query)
        return query, used_variables

    def _inner_filter(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], **kwargs) -> typing.Tuple[str, typing.Set[str]]:
        '''
        Filters the data by the given predicate and mapping, returning a list of intervals in which the predicate is true
        for each processed trace

        Returns a dictionary mapping from the trace ID to a dictionary that maps from the set of arguments to a list of
        intervals in which the predicate is true for that set of arguments
        '''

        predicate_rule = predicate.parseinfo.rule  # type: ignore

        if predicate_rule == "predicate":
            return self._handle_predicate(predicate, mapping, **kwargs)

        elif predicate_rule == "super_predicate":
            return self._inner_filter(predicate["pred"], mapping, **kwargs)  # type: ignore

        elif predicate_rule == "super_predicate_and":
            if 'use_de_morgans' in kwargs and kwargs['use_de_morgans']:
                return self._handle_and_de_morgans(predicate, mapping, **kwargs)

            return self._handle_and(predicate, mapping, **kwargs)

        elif predicate_rule == "super_predicate_or":
            return self._handle_or(predicate, mapping, **kwargs)

        elif predicate_rule == "super_predicate_not":
            return self._handle_not(predicate, mapping, **kwargs)

        elif predicate_rule in ["super_predicate_exists", "super_predicate_forall", "function_comparison"]:
            raise PredicateNotImplementedException(predicate_rule)

        else:
            raise ValueError(f"Error: Unknown rule '{predicate_rule}'")



if __name__ == '__main__':
    DEFAULT_GRAMMAR_PATH = "./dsl/dsl.ebnf"
    grammar = open(DEFAULT_GRAMMAR_PATH).read()
    grammar_parser = typing.cast(tatsu.grammars.Grammar, tatsu.compile(grammar))

    game = open(get_project_dir() + '/reward-machine/games/ball_to_bin_from_bed.txt').read()
    game_ast = grammar_parser.parse(game)  # type: ignore

    test_pred_orientation = game_ast[3][1]['setup']['and_args'][0]['setup']['exists_args']['setup']['statement']['conserved_pred']['pred']['and_args'][0]['pred']

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
    test_pred_agent_as_arg = block_stacking_game_ast[3][1]['preferences'][2]['definition']['pref_body']['body']['exists_args']['at_end_pred']
    # test these with ?g - game_object
    test_pred_object_in_top_drawer = block_stacking_game_ast[3][1]['preferences'][3]['definition']['pref_body']['body']['exists_args']['at_end_pred']
    test_pred_agent_adjacent = block_stacking_game_ast[3][1]['preferences'][4]['definition']['pref_body']['body']['exists_args']['at_end_pred']
    # test with ?s - sliding_door
    test_pred_agent_adjacent = block_stacking_game_ast[3][1]['preferences'][5]['definition']['pref_body']['body']['exists_args']['at_end_pred']

    BALL_TO_BIN_FROM_BED_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/FhhBELRaBFiGGvX0YG7W-preCreateGame-rerecorded.json')
    agent_adj_game = open(get_project_dir() + '/reward-machine/games/test_agent_door_adjacent.txt').read()
    agent_adj_game_ast = grammar_parser.parse(agent_adj_game)  # type: ignore

    agent_adj_predicate = agent_adj_game_ast[3][1]['preferences'][0]['definition']['pref_body']['body']['exists_args']['then_funcs'][0]['seq_func']['once_pred']
    agent_adj_mapping = {"?d": ["ball"]}


    test_mapping = {"?b": ["ball"], "?h": ["hexagonal_bin"]}
    block_test_mapping = {"?b1": ['cube_block'], "?b2": ["cube_block"]}
    block_desk_test_mapping = {"?b": ["block"]}
    all_block_test_mapping = {"?b1": ["block"], "?b2": ["block"]}

    test_predicates_and_mappings = [
        (test_pred_1, test_mapping),
        (test_pred_1, block_desk_test_mapping),
        (test_pred_2, test_mapping),
        (test_pred_or, block_test_mapping),
        (test_pred_or, all_block_test_mapping),
        (test_pred_desk_or, test_mapping),
        (test_pred_desk_or, block_desk_test_mapping),
        (agent_adj_predicate, agent_adj_mapping),
    ]

    stats = CommonSensePredicateStatisticsFullDatabse(cache_dir=DEFAULT_CACHE_DIR,
                                                    # trace_names=CURRENT_TEST_TRACE_NAMES,
                                                    trace_names=FULL_PARTICIPANT_TRACE_SET,
                                                    cache_rules=[],
                                                    base_trace_path=DEFAULT_BASE_TRACE_PATH,
                                                    force_trace_names_hash='028b3733',
                                                    # overwrite=True
                                                    )

    variable_type_usage = json.loads(open(f"{get_project_dir()}/reward-machine/caches/variable_type_usage.json", "r").read())
    for var_type in variable_type_usage:
        if var_type in META_TYPES:
            continue

        n_intervals = duckdb.sql(f"SELECT count(*) FROM data WHERE (arg_1_type='{var_type}' OR arg_2_type='{var_type}')").fetchone()[0]  # type: ignore

        prefix = "[+]" if n_intervals > 0 else "[-]"
        print(f"{prefix} {var_type} has {n_intervals} appearances")

    exit()

    # out = stats.filter(test_pred_object_in_top_drawer, {"?g": ["game_object"]})
    # print(out)
    # _print_results_as_expected_intervals(out)

    exit()

    tracer = None
    profile = None
    if PROFILE:
        # tracer = VizTracer(10000000, ignore_c_function=True, ignore_frozen=True)
        # tracer.start()
        profile = cProfile.Profile()
        profile.enable()

    N_ITER = 100
    for i in range(N_ITER):
        # print(f"\n====================")

        for test_pred, test_mapping in test_predicates_and_mappings:
            # print(f"Testing {test_pred} with mapping {test_mapping}")
            out = stats.filter(test_pred, test_mapping)
            # _print_results_as_expected_intervals(out)
        # inner_end = time.perf_counter()
        # print(f"Time per iteration: {'%.5f' % (inner_end - inner_start)}s")

    if profile is not None:
        profile.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(profile, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    # if tracer is not None:
        # tracer.stop()
        # profile_output_path = os.path.join(get_project_dir(), 'reward-machine/temp/viztracer_split_args.json')
        # print(f'Saving profile to {profile_output_path}')
        # tracer.save(profile_output_path)

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
