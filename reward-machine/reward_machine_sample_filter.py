import argparse
from collections import defaultdict, Counter
import cachetools
from datetime import datetime
import duckdb
import json
import logging
import multiprocessing
import operator
import os
import shutil
import sys
import tatsu.ast
from tqdm import tqdm
import typing
import pathlib
import polars as pl


import compile_predicate_statistics_full_database
from game_handler import GameHandler
from manual_run import _load_trace
from utils import FullState


sys.path.append((pathlib.Path(__file__).parents[1].resolve() / 'src').as_posix())
import ast_printer
import ast_parser
from ast_utils import simplified_context_deepcopy
from fitness_energy_utils import load_data, save_data
from ast_parser import SECTION_CONTEXT_KEY, VARIABLES_CONTEXT_KEY
from evolutionary_sampler import *


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.getLogger('wandb.docker.auth').setLevel(logging.WARNING)


parser = argparse.ArgumentParser()
parser.add_argument('--trace-names-hash', type=str, default=FULL_DATASET_TRACES_HASH)
parser.add_argument('--map-elites-model-name', type=str, default=None)
parser.add_argument('--map-elites-model-date-id', type=str, default=None)
parser.add_argument('--run-from-real-games', action='store_true')
parser.add_argument('--map-elites-model-folder', type=str, default='samples')
parser.add_argument('--relative-path', type=str, default='.')
parser.add_argument('--base-trace-path', type=str, default=compile_predicate_statistics_database.DEFAULT_BASE_TRACE_PATH)
parser.add_argument('--single-key', type=str, default=None)
DEFAULT_STOP_AFTER_COUNT = 5
parser.add_argument('--stop-after-count', type=int, default=DEFAULT_STOP_AFTER_COUNT)
parser.add_argument('--max-traces-per-game', type=int, default=None)
parser.add_argument('--max-keys', type=int, default=None)
parser.add_argument('--tqdm', action='store_true')
parser.add_argument('--use-trace-intersection', action='store_true')
parser.add_argument('--n-workers', type=int, default=1)
parser.add_argument('--chunksize', type=int, default=1)
parser.add_argument('--maxtasksperchild', default=None, type=int)
parser.add_argument('--copy-traces-to-tmp', action='store_true')
DEFAULT_START_METHOD = 'spawn'
parser.add_argument('--parallel-start-method', type=str, default=DEFAULT_START_METHOD)
DEFAULT_GRAMMAR_FILE = './dsl/dsl.ebnf'
parser.add_argument('-g', '--grammar-file', default=DEFAULT_GRAMMAR_FILE)
parser.add_argument('-t', '--test-file', default='./dsl/interactive-beta.pddl')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--force-trace-ids', type=str, nargs='*', default=[], required=False)

FULL_DATASET_TRACES_HASH = '028b3733'


class TraceFinderASTParser(ast_parser.ASTParser):
    expected_keys: typing.Set[str]
    not_implemented_predicate_counts: typing.DefaultDict[str, int]
    predicate_data_estimator: compile_predicate_statistics_full_database.CommonSensePredicateStatisticsFullDatabase
    trace_names_hash: str
    traces_by_preference_or_section: typing.Dict[str, typing.Set[str]]
    preferences_or_sections_with_implemented_predicates: typing.Set[str]

    def __init__(self, trace_names_hash: str = FULL_DATASET_TRACES_HASH):
        self.not_implemented_predicate_counts = defaultdict(int)
        self.trace_names_hash = trace_names_hash
        self.predicate_data_estimator = compile_predicate_statistics_full_database.CommonSensePredicateStatisticsFullDatabase(
            force_trace_names_hash=self.trace_names_hash
        )  # type: ignore

    def __call__(self, ast, **kwargs):
        initial_call = 'inner_call' not in kwargs or not kwargs['inner_call']
        if initial_call:
            kwargs['inner_call'] = True
            kwargs['local_context'] = {'mapping': {VARIABLES_CONTEXT_KEY: {}}}
            kwargs['global_context'] = {}
            self.expected_keys = set()
            self.traces_by_preference_or_section = {}
            self.preferences_or_sections_with_implemented_predicates = set()
            self.predicate_strings_by_preference_or_section = defaultdict(set)
            self.not_implemented_predicate_counts = defaultdict(int)

        retval = super().__call__(ast, **kwargs)

        if initial_call:
            return self.traces_by_preference_or_section, self.expected_keys
        else:
            return retval

    def _current_ast_to_contexts_hook(self, ast: tatsu.ast.AST, kwargs: typing.Dict[str, typing.Any]):
        rule = typing.cast(str, ast.parseinfo.rule)  # type: ignore

        if rule == 'preference':
            kwargs['local_context']['current_preference_name'] = ast.pref_name

    def _handle_ast(self, ast: tatsu.ast.AST, **kwargs):
        self._current_ast_to_contexts(ast, **kwargs)
        kwargs['local_context']['mapping'] = ast_parser.update_context_variables(ast, kwargs['local_context']['mapping'])

        if SECTION_CONTEXT_KEY in kwargs and kwargs[SECTION_CONTEXT_KEY] == ast_parser.SETUP:
            self.expected_keys.add(kwargs[SECTION_CONTEXT_KEY])
        elif 'current_preference_name' in kwargs['local_context']:
            self.expected_keys.add(kwargs['local_context']['current_preference_name'])

        if ast.parseinfo.rule == 'predicate':  # type: ignore
            context_variables = kwargs['local_context']['mapping'][VARIABLES_CONTEXT_KEY]
            secion_or_preference_key = kwargs['local_context']['current_preference_name'] if 'current_preference_name' in kwargs['local_context'] else kwargs[SECTION_CONTEXT_KEY]
            predicate_string = ast_printer.ast_section_to_string(ast, kwargs[SECTION_CONTEXT_KEY])
            self.predicate_strings_by_preference_or_section[secion_or_preference_key].add(predicate_string)

            try:
                mapping = {k: v.var_types for k, v in context_variables.items()} if context_variables is not None else {}  # type: ignore
                trace_ids = self.predicate_data_estimator.filter(ast, mapping, return_trace_ids=True)

                if secion_or_preference_key not in self.traces_by_preference_or_section:
                    self.traces_by_preference_or_section[secion_or_preference_key] = set(trace_ids)
                else:
                    self.traces_by_preference_or_section[secion_or_preference_key].intersection_update(trace_ids)

                self.preferences_or_sections_with_implemented_predicates.add(secion_or_preference_key)

            except compile_predicate_statistics_full_database.PredicateNotImplementedException:
                self.not_implemented_predicate_counts[ast.pred.parseinfo.rule.replace('predicate_', '')] += 1
                # pass

        else:
            for key in ast:
                if key != 'parseinfo':
                    child_kwargs = simplified_context_deepcopy(kwargs)
                    retval = self(ast[key], **child_kwargs)
                    self._update_contexts_from_retval(kwargs, retval)


MAX_TRACE_CACHE_SIZE = 512


class TraceGameEvaluator:
    base_trace_path: str
    chunksize: int
    force_trace_ids: typing.Optional[typing.List[str]]
    max_traces_per_game: typing.Optional[int]
    n_workers: int
    population: typing.Dict[typing.Union[KeyTypeAnnotation, int], ASTType]
    stop_after_count: int
    traces_by_population_key: typing.Dict[typing.Union[KeyTypeAnnotation, int], typing.Tuple[typing.List[str], typing.Set[str]]]
    trace_finder: TraceFinderASTParser
    use_trace_intersection: bool
    verbose: bool

    def __init__(self, trace_finder: TraceFinderASTParser, population: typing.Union[typing.Dict[KeyTypeAnnotation, ASTType], typing.List[ASTType]],
                 use_trace_intersection: bool = False, stop_after_count: int = DEFAULT_STOP_AFTER_COUNT,
                 max_traces_per_game: typing.Optional[int] = None, force_trace_ids: typing.Optional[typing.List[str]] = None,
                 base_trace_path: str = compile_predicate_statistics_database.DEFAULT_BASE_TRACE_PATH,
                 max_cache_size: int = MAX_TRACE_CACHE_SIZE, tqdm: bool = False,
                 n_workers: int = 1, chunksize: int = 1,
                 maxtasksperchild: typing.Optional[int] = None, verbose: bool = False):
        self.trace_finder = trace_finder
        if isinstance(population, list):
            population = {idx: sample for idx, sample in enumerate(population)}

        self.population = population
        self.stop_after_count = stop_after_count
        self.use_trace_intersection = use_trace_intersection
        self.max_traces_per_game = max_traces_per_game
        self.force_trace_ids = force_trace_ids
        self.base_trace_path = base_trace_path
        self.tqdm = tqdm
        self.n_workers = n_workers
        self.chunksize = chunksize
        self.maxtasksperchild = maxtasksperchild
        self.verbose = verbose

        self.traces_by_population_key = {}

        self.cache = cachetools.LRUCache(maxsize=max_cache_size)

    @cachetools.cachedmethod(operator.attrgetter('cache'))
    def _load_trace(self, trace_id: str):
        trace_path = os.path.join(self.base_trace_path, trace_id + '.json')

        if isinstance(trace_path, pathlib.Path):
            trace_path = trace_path.resolve().as_posix()

        with open(trace_path, 'r') as f:
            trace = json.load(f)
            if not isinstance(trace, list) and 'replay' in trace:
                trace = trace['replay']

        return trace

    def _iterate_trace(self, trace_id: str) -> typing.Iterator[typing.Tuple[dict, bool]]:
        trace = self._load_trace(trace_id)
        for idx, event in enumerate(trace):
            yield (event, idx == len(trace) - 1)

    def _find_key_traces(self, key: typing.Union[KeyTypeAnnotation, int]) -> typing.Tuple[typing.List[str], typing.Set[str]]:
        if key not in self.traces_by_population_key:
            sample = self.population[key]

            traces_by_key, expected_keys = self.trace_finder(sample)  # type: ignore

            all_traces = set()
            traces = []
            initial_traces = True

            if self.use_trace_intersection:
                for trace_set in traces_by_key.values():
                    if initial_traces:
                        all_traces.update(trace_set)
                        initial_traces = False
                    else:
                        all_traces.intersection_update(trace_set)

                traces = list(all_traces)

            else:
                trace_sets_by_length = sorted(traces_by_key.values(), key=len)
                for trace_set in trace_sets_by_length:
                    new_traces = trace_set - all_traces
                    traces.extend(new_traces)
                    all_traces.update(new_traces)

            self.traces_by_population_key[key] = traces, expected_keys

        return self.traces_by_population_key[key]

    def handle_single_game(self, key: typing.Union[KeyTypeAnnotation, int], print_results: bool = False, return_key: bool = True):
        all_traces, expected_keys = self._find_key_traces(key)
        if self.force_trace_ids:
            all_traces = self.force_trace_ids

        if print_results: logger.info(f'For key {key} found {len(all_traces)} traces')
        if len(all_traces) == 0:
            if print_results: logger.info('No traces found')
            return key, -1, {}

        if self.max_traces_per_game is not None:
            all_traces = list(all_traces)[:self.max_traces_per_game]

        counts_by_trace_and_key = {key: {} for key in expected_keys}
        scores_by_trace = {}
        stop_count_by_key = {key: 0 for key in expected_keys}
        total_count_by_key = {key: 0 for key in expected_keys}

        trace_iter = all_traces
        if self.tqdm and self.n_workers <= 1:
            trace_iter = tqdm(trace_iter, desc=f'Traces for key {key}')

        sample = self.population[key]

        for trace in trace_iter:
            if isinstance(self.trace_finder.predicate_data_estimator, compile_predicate_statistics_full_database.CommonSensePredicateStatisticsFullDatabase):
                domain = duckdb.sql(f"SELECT domain FROM trace_length_and_domains WHERE trace_id='{trace}'").fetchone()[0]  # type: ignore
                if domain is None:
                    raise ValueError(f'No domain found for trace {trace}')
            else:
                domain = self.trace_finder.predicate_data_estimator.trace_lengths_and_domains_df.filter(pl.col('trace_id') == trace).select('domain').item()
            # TODO: figure out what needs to be reset between games to instantiate a game handler only once
            game_handler = GameHandler(sample, force_domain=domain)  # type: ignore

            for state, is_final in self._iterate_trace(trace):
                state = FullState.from_state_dict(state)
                score = game_handler.process(state, is_final, ignore_terminals=True)
                if score is not None:
                    break

            score = game_handler.score(game_handler.scoring)
            scores_by_trace[trace] = score

            if ast_parser.SETUP in expected_keys:
                counts_by_trace_and_key[ast_parser.SETUP][trace] = game_handler.setup_met
                stop_count_by_key[ast_parser.SETUP] += int(game_handler.setup_met)

            for preference_name in expected_keys:
                if preference_name in game_handler.preference_satisfactions:
                    n_preference_satisfcations = len(game_handler.preference_satisfactions[preference_name])
                    counts_by_trace_and_key[preference_name][trace] = n_preference_satisfcations
                    stop_count_by_key[preference_name] += int(n_preference_satisfcations > 0)
                    total_count_by_key[preference_name] += n_preference_satisfcations

            if self.verbose:
                n_satisfactions_by_pref = " ".join(f'{k}: {len(v)}/{total_count_by_key[k]}' for k, v in game_handler.preference_satisfactions.items())
                logger.info(f'For trace {trace} | setup met: {game_handler.setup_met} | satisfaction count: {n_satisfactions_by_pref}')

            if all(stop_count >= self.stop_after_count for stop_count in stop_count_by_key.values()):
                return key, self.stop_after_count, counts_by_trace_and_key

        if print_results:
            for preference_name in expected_keys:
                non_zero_count_traces = {trace: count for trace, count in counts_by_trace_and_key[preference_name].items() if count > 0 and count is not False}
                print(f'For preference {preference_name}, {len(non_zero_count_traces)} traces have non-zero counts:')
                for trace, count in non_zero_count_traces.items():
                    print(f'    - {trace}: {count}')
            print()
            non_zero_score_traces = {trace: score for trace, score in scores_by_trace.items() if score != 0}
            print(f'For key {key}, {len(non_zero_score_traces)} traces have non-zero scores, while {len(scores_by_trace) - len(non_zero_score_traces)} traces have score zero:')
            for trace, score in non_zero_score_traces.items():
                print(f'    - {trace}: {score}')

        min_count = min(stop_count_by_key.values())
        return key, min_count, counts_by_trace_and_key

    def __call__(self, key: typing.Optional[KeyTypeAnnotation],
                 max_keys: typing.Optional[int] = None,
                 sort_keys_by_traces: bool = True):
        if key is not None:
            if key not in self.population:
                logger.info(f'Key {key} not found')
                return

            logger.info(f'Filtering {key}')
            self.handle_single_game(key, print_results=True)
        else:
            result_summary_by_key = {}
            full_result_by_key = {}

            if sort_keys_by_traces:
                key_iter = sorted(self.population.keys(), key=lambda key: len(self._find_key_traces(key)[0]), reverse=True)
                print([len(self._find_key_traces(key)[0]) for key in key_iter])

            else:
                key_iter = self.population.keys()

            if max_keys is not None:
                key_iter = list(key_iter)[:max_keys]

            if self.tqdm and self.n_workers <= 1:
                key_iter = tqdm(key_iter)

            if self.n_workers > 1:
                with multiprocessing.Pool(self.n_workers, maxtasksperchild=self.maxtasksperchild) as p:
                    logger.info('Pool started')
                    pbar = tqdm(desc='MAP-Elites Population', total=len(self.population))

                    for key, min_count, counts_by_trace_and_key in p.imap_unordered(self.handle_single_game, key_iter, chunksize=self.chunksize):
                        result_summary_by_key[key] = min_count
                        full_result_by_key[key] = counts_by_trace_and_key
                        pbar.update(1)
                        pbar.set_postfix(dict(timestamp=datetime.now().strftime('%H:%M:%S')))

            else:
                for key in key_iter:
                    key, min_count, counts_by_trace_and_key = self.handle_single_game(key)
                    result_summary_by_key[key] = min_count
                    full_result_by_key[key] = counts_by_trace_and_key

            print('=' * 80)
            print(result_summary_by_key)
            print('=' * 80)

            print('Summary of results:')
            result_counts = Counter(result_summary_by_key.values())
            for count in sorted(result_counts.keys()):
                print(f'    - {count}: {result_counts[count]}')

            return result_summary_by_key, full_result_by_key


def main(args: argparse.Namespace):
    multiprocessing.set_start_method(args.parallel_start_method, force=True)

    try:
        if args.copy_traces_to_tmp:
            shutil.rmtree('/tmp/participant_traces', ignore_errors=True)
            shutil.copytree(args.base_trace_path, '/tmp/participant_traces')
            logger.info('Copied traces to /tmp/participant_traces')
            args.base_trace_path = '/tmp/participant_traces'

        if args.run_from_real_games:
            logger.info('Running from real games')
            grammar = open(args.grammar_file).read()
            grammar_parser = tatsu.compile(grammar)
            population = list(cached_load_and_parse_games_from_file(args.test_file, grammar_parser, False, relative_path='.'))  # type: ignore

        else:
            if args.map_elites_model_date_id is None or args.map_elites_model_name is None:
                raise ValueError('Must provide map elites model date id and name if not running from real games')
            logger.info(f'Running from MAP-Elites model | {args.map_elites_model_date_id} | {args.map_elites_model_name}')
            model = typing.cast(MAPElitesSampler, load_data(args.map_elites_model_date_id, args.map_elites_model_folder, args.map_elites_model_name, relative_path=args.relative_path))
            population = model.population

        trace_finder = TraceFinderASTParser(args.trace_names_hash)

        trace_evaluator = TraceGameEvaluator(trace_finder, population, # type: ignore
                                             use_trace_intersection=args.use_trace_intersection,
                                            stop_after_count=args.stop_after_count, max_traces_per_game=args.max_traces_per_game,
                                            force_trace_ids=args.force_trace_ids,
                                            base_trace_path=args.base_trace_path, tqdm=args.tqdm,
                                            n_workers=args.n_workers, chunksize=args.chunksize,
                                            maxtasksperchild=args.maxtasksperchild, verbose=args.verbose)

        key = None
        if args.single_key is not None:
            key = tuple(map(int, args.single_key.replace('(', '').replace(')', '').split(',')))

        results = trace_evaluator(key, args.max_keys)

        if results is not None:
            result_summary_by_key, full_result_by_key = results
            output = dict(summary=result_summary_by_key, full=full_result_by_key)
            name = 'real_games' if args.run_from_real_games else f'{args.map_elites_model_name}_{args.map_elites_model_date_id}'
            save_data(output, folder=args.map_elites_model_folder, name=f'trace_filter_results_{name}', relative_path=args.relative_path)

    finally:
        if args.copy_traces_to_tmp:
            shutil.rmtree('/tmp/participant_traces', ignore_errors=True)



if __name__ == '__main__':
    main(parser.parse_args())
