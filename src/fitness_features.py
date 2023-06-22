from abc import ABC, abstractmethod
import argparse
from datetime import datetime
from collections import namedtuple, defaultdict
import csv
from enum import Enum
import itertools
import glob
import gzip
import logging
import multiprocessing
import os
import pickle
import re
import sys
import typing

import boolean
import numpy as np
import pandas as pd
import tatsu
import tatsu.ast
import tatsu.buffering
import tatsu.infos
from tqdm import tqdm

from ast_to_latex_doc import TYPE_RULES, extract_n_args, extract_predicate_function_args, extract_predicate_function_name
from ast_parser import VariableDefinition, extract_variables_from_ast, update_context_variables, predicate_function_term_to_type_categories,\
    VARIABLES_CONTEXT_KEY, SECTION_CONTEXT_KEY, QUANTIFICATIONS_CONTEXT_KEY, MODAL_CONTEXT_KEY
import ast_parser
import ast_printer
from ast_utils import cached_load_and_parse_games_from_file, simplified_context_deepcopy
from fitness_features_preprocessing import FitnessFeaturesPreprocessor, DEFAULT_MERGE_THRESHOLD_PROPORTION, BinarizeFitnessFeatures, MergeFitnessFeatures, NGRAM_SCORE_PATTERN
from fitness_ngram_models import NGramTrieNode, NGramTrieModel, ASTNGramTrieModel, NGramASTParser, DEFAULT_N_BY_SECTION
from latest_model_paths import LATEST_AST_N_GRAM_MODEL_PATH, LATEST_SPECIFIC_OBJECTS_AST_N_GRAM_MODEL_PATH
import room_and_object_types


parser = argparse.ArgumentParser()
DEFAULT_GRAMMAR_FILE = './dsl/dsl.ebnf'
parser.add_argument('-g', '--grammar-file', default=DEFAULT_GRAMMAR_FILE)
DEFAULT_TEST_FILES = (
    './dsl/interactive-beta.pddl',
    './dsl/ast-real-regrowth-samples-1024.pddl.gz',
    # './dsl/ast-real-regrowth-samples.pddl',
    # './dsl/ast-mle-samples-large.pddl',

    # './dsl/ast-codex-combine-samples.pddl',
    # './dsl/ast-codex-regrowth-samples.pddl',

    # './dsl/ast-mle-samples.pddl',
    # './dsl/ast-mle-regrowth-samples.pddl',
    # './dsl/ast-mle-samples-large.pddl',
    # './dsl/ast-mle-samples-large-best.pddl',
    # './dsl/ast-best-mle-regrowth-samples.pddl',
    # './dsl/ast-mle-samples-medium.pddl',
    # './dsl/ast-medium-mle-regrowth-samples.pddl',
)
parser.add_argument('-t', '--test-files', action='append', default=[])
parser.add_argument('-q', '--dont-tqdm', action='store_true')
# DEFAULT_OUTPUT_PATH ='./data/fitness_features.csv.gz'
DEFAULT_OUTPUT_PATH ='./data/fitness_features_1024_regrowths.csv.gz'
parser.add_argument('-o', '--output-path', default=DEFAULT_OUTPUT_PATH)
DEFAULT_FEATURIZER_OUTPUT_PATH_PATTERN = './models/fitness_featurizer_{today}.pkl.gz'
parser.add_argument('-f', '--featurizer-output-path', default=None)
DEFAULT_RECURSION_LIMIT = 2000
parser.add_argument('--recursion-limit', type=int, default=DEFAULT_RECURSION_LIMIT)
parser.add_argument('--no-binarize', action='store_true')
parser.add_argument('--no-merge', action='store_true')
parser.add_argument('--merge-threshold', type=float, default=DEFAULT_MERGE_THRESHOLD_PROPORTION)
parser.add_argument('--use-specific-objects-ngram-model', action='store_true')
parser.add_argument('--existing-featurizer-path', default=None)
parser.add_argument('--n-workers', type=int, default=1)
parser.add_argument('--chunksize', type=int, default=1024)
parser.add_argument('--maxtasksperchild', default=None)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ContextDict = typing.Dict[str, typing.Union[str, int, VariableDefinition]]
Number = typing.Union[int, float]


TEMP_DIR = '/tmp/gd1279/fitness_features'


class FitnessTerm(ABC):
    rules: typing.Sequence[typing.Union[str, re.Pattern]]
    header: str

    def __init__(self, rule_or_rules: typing.Union[str, re.Pattern, typing.Sequence[typing.Union[str, re.Pattern]]], header: str):
        if isinstance(rule_or_rules, str) or isinstance(rule_or_rules, re.Pattern):
            rule_or_rules = [rule_or_rules]

        self.rules = rule_or_rules
        self.header = header

    def game_start(self) -> None:
        pass

    @abstractmethod
    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict) -> None:
        pass

    def parse_full_text(self, full_text: str) -> None:
        pass

    @abstractmethod
    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        pass

    def get_all_keys(self) -> typing.List[str]:
        inner_keys = self._get_all_inner_keys()
        if inner_keys is None:
            return [self.header]

        return [f'{self.header}_{inner_key}' for inner_key in inner_keys]

    def _get_all_inner_keys(self) -> typing.Optional[typing.List[str]]:
        return None


REAL_GAMES_SRC_FILE = 'interactive-beta.pddl'
DEFAULT_HEADERS = ('src_file', 'game_name', 'domain_name', 'real')

DEPTH_CONTEXT_KEY = 'depth'
EXTERNAL_FORALL_CONTEXT_KEY = 'external_forall'

COUNT_RULE_PATTERN = re.compile('count.*')
COMPUTE_REAL_MIN_MAX_FEATURE_PATTERNS = [NGRAM_SCORE_PATTERN]

class ASTFitnessFeaturizer:
    columns_to_terms: typing.Dict[str, FitnessTerm]
    full_text_registry: typing.List[FitnessTerm]
    headers: typing.List[str]
    header_registry: typing.Dict[str, FitnessTerm]
    list_reduce: typing.Callable[[typing.Sequence[Number]], Number]
    preprocessors: typing.Optional[typing.Iterable[FitnessFeaturesPreprocessor]]
    regex_rules: typing.List[typing.Tuple[re.Pattern, FitnessTerm]]
    rows: typing.List
    rows_df: typing.Optional[pd.DataFrame]
    rule_registry: typing.Dict[str, typing.List[FitnessTerm]]
    section_keys: typing.List[str]
    section_registry: typing.Dict[str, typing.List[FitnessTerm]]
    tuple_registry: typing.Dict[str, typing.List[FitnessTerm]]

    def __init__(self, preprocessors: typing.Optional[typing.Iterable[FitnessFeaturesPreprocessor]] = None,
        default_headers: typing.Sequence[str] = DEFAULT_HEADERS,
        list_reduce: typing.Callable[[typing.Sequence[Number]], Number] = np.sum,
        section_keys: typing.Sequence[str] = ast_parser.SECTION_KEYS,
        real_games_src_file: str = REAL_GAMES_SRC_FILE,
        compute_real_min_max_feature_patterns: typing.Sequence[typing.Union[str, re.Pattern]] = tuple()):

        self.preprocessors = preprocessors
        self.default_headers = default_headers
        self.headers = list(default_headers)
        self.list_reduce = list_reduce
        self.section_keys = list(section_keys)
        self.real_games_src_file = real_games_src_file
        self.compute_real_min_max_feature_patterns = compute_real_min_max_feature_patterns

        self.rule_registry = defaultdict(list)
        self.tuple_registry = defaultdict(list)
        self.section_registry = defaultdict(list)
        self.full_ast_registry = []
        self.full_text_registry = []
        self.regex_rules = []
        self.header_registry = dict()

        self.rows = []
        self.rows_df = None
        self.df_keys_set = set()
        self.df_keys = []
        self.all_column_keys = None
        self.columns_to_terms = None

    def __getstate__(self) -> typing.Dict[str, typing.Any]:
        # Prevents the rows from being dumped to file when this is pickled
        state = self.__dict__.copy()
        del state['rows']
        del state['rows_df']
        return state

    def __setstate__(self, state: typing.Dict[str, typing.Any]) -> None:
        self.__dict__.update(state)
        self.rows = []
        self.rows_df = None

    def _register(self, term: FitnessTerm, rule: str, tuple_rule: bool = False) -> None:
        if tuple_rule:
            self.tuple_registry[rule].append(term)
        else:
            self.rule_registry[rule].append(term)

    def register(self, term: FitnessTerm, tuple_rule: bool = False, section_rule: bool = False,
                 full_ast_rule: bool = False, full_text_rule: bool = False) -> None:

        if full_ast_rule:
            self.full_ast_registry.append(term)

        elif full_text_rule:
            self.full_text_registry.append(term)

        else:
            if section_rule:
                for section in term.rules:
                    section = typing.cast(str, section)
                    if section not in self.section_keys:
                        raise ValueError(f'Invalid section key: {section}')

                    self.section_registry[section].append(term)

            for rule in term.rules:
                if isinstance(rule, re.Pattern):
                    self.regex_rules.append((rule, term))
                else:
                    self._register(term, rule, tuple_rule)

        self.header_registry[term.header] = term
        self.headers.append(term.header)

    def register_multiple(self, terms: typing.Sequence[FitnessTerm], tuple_rule: bool = False, section_rule: bool = False) -> None:
        for term in terms:
            self.register(term, tuple_rule, section_rule)

    def combine_featurizers(self, *others: 'ASTFitnessFeaturizer') -> None:
        self.rows.extend(itertools.chain.from_iterable(other.rows for other in others))

    def to_df(self, use_prior_values: typing.Union[bool, typing.List[bool]] = False, preprocess: bool = True) -> pd.DataFrame:
        if self.rows_df is not None:
            df = self.rows_df
        else:
            df = pd.DataFrame.from_records(self.rows, columns=list(self.default_headers) + self.df_keys)

        real_min_max_values = {}
        if len(self.compute_real_min_max_feature_patterns) > 0:
            for pattern in self.compute_real_min_max_feature_patterns:
                if isinstance(pattern, str):
                    if pattern in df.columns:
                        real_min_max_values[pattern] = (df.loc[df.real, pattern].min(), df.loc[df.real, pattern].max())
                else:
                    for column in df.columns:
                        if pattern.match(column):
                            real_min_max_values[column] = (df.loc[df.real, column].min(), df.loc[df.real, column].max())

        if preprocess and self.preprocessors is not None:
            if isinstance(use_prior_values, bool):
                use_prior_values = [use_prior_values] * len(self.preprocessors)  # type: ignore

            for i, preprocessor in enumerate(self.preprocessors):
                df = preprocessor.preprocess_df(df, use_prior_values=use_prior_values[i], min_max_values=real_min_max_values)

        return df

    def _add_df_key(self, key: str, previous_key: typing.Optional[str]):
        if key in self.df_keys_set:
            return

        self.df_keys_set.add(key)
        if previous_key is None:
            self.df_keys.append(key)

        else:
            previous_index = self.df_keys.index(previous_key)
            self.df_keys.insert(previous_index + 1, key)

    def get_column_to_term_mapping(self):
        if self.columns_to_terms is not None:
            return self.columns_to_terms

        self.get_all_column_keys()
        return self.columns_to_terms

    def get_all_column_keys(self):
        if self.all_column_keys is not None:
            return self.all_column_keys

        self.all_column_keys = list(self.default_headers)
        self.columns_to_terms = {}
        for _, term in self.header_registry.items():
            term_keys = term.get_all_keys()
            self.all_column_keys.extend(term_keys)
            for key in term_keys:
                self.columns_to_terms[key] = term

        return self.all_column_keys

    def parse(self, full_ast: typing.Tuple[tatsu.ast.AST, tatsu.ast.AST, tatsu.ast.AST, tatsu.ast.AST], src_file: str = '', return_row: bool = False, preprocess_row: bool = True):
        row = {}
        row['src_file'] = os.path.basename(src_file)
        game_name = full_ast[1]['game_name']
        domain_name = full_ast[2]['domain_name']
        row['game_name'] = game_name  # type: ignore
        row['domain_name'] = domain_name  # type: ignore
        row['real'] = row['src_file'] == self.real_games_src_file
        ast = full_ast[3:]  # type: ignore

        for term in self.header_registry.values():
            term.game_start()

        context = dict(game_name=game_name, domain_name=domain_name)

        self._parse(ast, context=context)  # type: ignore

        for term in self.full_ast_registry:
            term.update(full_ast, 'full_ast', context=context.copy())

        ast_full_text = ast_printer.ast_to_string(full_ast, ' ')  # type: ignore
        for term in self.full_text_registry:
            term.parse_full_text(ast_full_text)

        previous_key = None

        for header, term in self.header_registry.items():
            term_result = term.game_end()
            if isinstance(term_result, bool):
                term_result = int(term_result)

            if isinstance(term_result, (int, float)):
                row[header] = term_result
                self._add_df_key(header, previous_key)
                previous_key = header
            elif isinstance(term_result, dict):
                for key, val in term_result.items():
                    header_key = f'{header}_{key}'
                    row[header_key] = val
                    self._add_df_key(header_key, previous_key)
                    previous_key = header_key
            else:
                row[header] = self.list_reduce(row[header])
                self._add_df_key(header, previous_key)
                previous_key = header

        if return_row:
            if preprocess_row:
                if self.preprocessors is not None:
                    for preprocessor in self.preprocessors:
                        row = preprocessor.preprocess_row(row)

            return row

        else:
            self.rows.append(row)

    def _parse(self, ast: typing.Union[str, int, tatsu.buffering.Buffer, tuple, list, tatsu.ast.AST],
        context: typing.Optional[ContextDict] = None):
        if context is None:
            context = {}

        if DEPTH_CONTEXT_KEY not in context:
            context[DEPTH_CONTEXT_KEY] = 0

        if VARIABLES_CONTEXT_KEY not in context:
            context[VARIABLES_CONTEXT_KEY] = {}  # type: ignore

        if not ast or isinstance(ast, (str, int, np.int32, np.int64, tatsu.buffering.Buffer)):  # type: ignore
            return

        elif isinstance(ast, (tuple, list)):
            if len(ast) > 0 and isinstance(ast[0], str):
                # check if should update the section key
                if ast[0] in self.section_keys:
                    context[SECTION_CONTEXT_KEY] = ast[0]

                for tuple_stat in self.tuple_registry[ast[0]]:
                    tuple_stat.update(ast, '', context)

            [self._parse(element, context) for element in ast]

        elif isinstance(ast, tatsu.ast.AST):
            # Look for variable definitions
            context = update_context_variables(ast, context)

            if 'parseinfo' not in ast or not ast.parseinfo:
                raise ValueError('No parseinfo found', ast)

            # Check for any handlers of the current node
            rule = ast.parseinfo.rule

            if rule == 'seq_func':
                seq_func = ast.seq_func
                if isinstance(seq_func, str):
                    context[MODAL_CONTEXT_KEY] = seq_func
                else:
                    context[MODAL_CONTEXT_KEY] = seq_func.parseinfo.rule  # type: ignore
            elif rule == 'at_end':
                context[MODAL_CONTEXT_KEY] = 'at_end'

            stat_parsers = self.rule_registry[rule]
            for stat in stat_parsers:
                stat.update(ast, rule, context)

            if SECTION_CONTEXT_KEY in context:
                section = typing.cast(str, context[SECTION_CONTEXT_KEY])
                for term in self.section_registry[section]:
                    term.update(ast, rule, context)

            for regex_pattern, regex_term in self.regex_rules:
                if regex_pattern.match(rule):
                    regex_term.update(ast, rule, context)

            # Perform other context updates for children
            child_context = simplified_context_deepcopy(context)  # .copy()
            child_context[DEPTH_CONTEXT_KEY] += 1  # type: ignore

            if ast.parseinfo.rule in ('scoring_external_maximize', 'scoring_external_minimize'):
                child_context[EXTERNAL_FORALL_CONTEXT_KEY] = ast.parseinfo

            for child_key in ast:
                if child_key != 'parseinfo':
                    self._parse(ast[child_key], child_context)  # type: ignore

        else:
            logger.warn(f'Encountered AST element with unrecognized type: {ast} of type {type(ast)}')


PREDICATE_OR_FUNCTION_TERM_PATTERN = re.compile(r'predicate_or_function[\w\d_]+term')


class VariableBasedFitnessTerm(FitnessTerm):
    def __init__(self, header: str):
        super().__init__(('setup_statement', 'predicate', 'function_eval', PREDICATE_OR_FUNCTION_TERM_PATTERN), header)
        self.variables = set()

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        context_variables = context.get(VARIABLES_CONTEXT_KEY, {})

        if 'term' in ast:
            if isinstance(ast.term, str) and ast.term.startswith('?'):  # type: ignore
                self._inner_update(ast.term, context_variables)  # type: ignore

        else:
            self._inner_update(None, context_variables)  # type: ignore

    @abstractmethod
    def _inner_update(self, term: str, variables: typing.Dict[str, VariableDefinition]):
        pass


class AllVariablesDefined(VariableBasedFitnessTerm):
    defined_count: int = 0
    undefined_count: int = 0

    def __init__(self):
        super().__init__('all_variables_defined')

    def game_start(self) -> None:
        self.defined_count = 0
        self.undefined_count = 0

    def _inner_update(self, term: str, variables: typing.Dict[str, typing.Union[VariableDefinition, typing.List[VariableDefinition]]]):
        if term is None:
            return
        elif term in variables:
            self.defined_count += 1
        else:
            self.undefined_count += 1

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        # print(self.defined_count, self.undefined_count)
        if self.defined_count == 0:
            return 0

        return self.undefined_count == 0

        # return self.defined_count / (self.defined_count + self.undefined_count)


class AllVariablesUsed(VariableBasedFitnessTerm):
    defined_variables: typing.Set[typing.Tuple[str, str, int]]
    used_variables: typing.Set[typing.Tuple[str, str, int]]
    variable_definition_repeated: bool

    def __init__(self):
        super().__init__('all_variables_used')

    def game_start(self) -> None:
        self.defined_variables = set()
        self.used_variables = set()
        self.variable_definition_repeated = False

    def _inner_update(self, term: str, variables: typing.Dict[str, typing.Union[VariableDefinition, typing.List[VariableDefinition]]]):
        for v, var_def_or_list in variables.items():
            if isinstance(var_def_or_list, list):
                self.variable_definition_repeated = True
                return
            else:
                self.defined_variables.add((v, var_def_or_list.parseinfo.rule, var_def_or_list.parseinfo.pos))

        if term is not None and term in variables:
            var_def = variables[term]
            if isinstance(var_def, list):
                self.variable_definition_repeated = True
                return

            self.used_variables.add((term, var_def.parseinfo.rule, var_def.parseinfo.pos))

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        # print(self.defined_variables, self.used_variables, self.variable_definition_repeated)
        if len(self.defined_variables) == 0 or self.variable_definition_repeated:
            return 0

        return len(self.defined_variables.intersection(self.used_variables)) == len(self.defined_variables)


class AllPreferencesUsed(FitnessTerm):
    defined_preferences: typing.Set[str] = set()
    used_preferences: typing.Set[str] = set()

    def __init__(self):
        super().__init__(('preference', COUNT_RULE_PATTERN), 'all_preferences_used')

    def game_start(self) -> None:
        self.defined_preferences = set()
        self.used_preferences = set()

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if rule == 'preference':
            self.defined_preferences.add(ast.pref_name)  # type: ignore

        else:
            self.used_preferences.add(ast.name_and_types.pref_name)  # type: ignore

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if len(self.defined_preferences) == 0:
            return 0

        return len(self.defined_preferences.intersection(self.used_preferences)) == len(self.defined_preferences.union(self.used_preferences))


class NumPreferencesDefined(FitnessTerm):
    defined_preferences: typing.Set[str] = set()
    max_count: int
    min_count: int

    def __init__(self, max_count: int = 6, min_count: int = 1):
        super().__init__('preference', 'num_preferences_defined')
        self.max_count = max_count
        self.min_count = min_count

    def game_start(self) -> None:
        self.defined_preferences = set()

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        self.defined_preferences.add(ast.pref_name)  # type: ignore

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        num_preferences = np.clip(len(self.defined_preferences), self.min_count, self.max_count)
        return {str(d): d == num_preferences for d in range(self.min_count, self.max_count + 1)}

    def _get_all_inner_keys(self):
        return [str(d) for d in range(self.min_count, self.max_count + 1)]


SETUP_OBJECTS_SKIP_CATEGORIES = [room_and_object_types.AGENT, room_and_object_types.COLORS, room_and_object_types.ORIENTATIONS, room_and_object_types.SIDES]
SETUP_OBJECTS_SKIP_OBJECTS = set(sum([list(room_and_object_types.CATEGORIES_TO_TYPES[category]) for category in SETUP_OBJECTS_SKIP_CATEGORIES], []))


class SetupObjectsUsed(FitnessTerm):
    setup_objects: typing.Set[str] = set()
    used_objects: typing.Set[str] = set()

    def __init__(self, skip_objects: typing.Set[str] = SETUP_OBJECTS_SKIP_OBJECTS, rules: typing.Optional[typing.Sequence[typing.Union[str, re.Pattern]]] = None,
                 header: str = 'setup_objects_used'):
        if rules is None:
            rules = list(TYPE_RULES.keys())
        else:
            rules = list(rules)

        if 'pref_name_and_types' in rules:  # to avoid counting scoring references
            rules.remove('pref_name_and_types')

        super().__init__(rules, header)
        self.skip_objects = skip_objects

    def game_start(self) -> None:
        self.setup_objects = set()
        self.used_objects = set()

    def _add_result(self, result_set: typing.Set[str], result: typing.Union[None, str, typing.List[str], tuple]):
        if isinstance(result, (list, tuple)):
            result_set.update([r for r in result if r not in self.skip_objects])
        elif result is not None and result not in self.skip_objects:
            result_set.add(result)

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if SECTION_CONTEXT_KEY not in context:
            raise ValueError('Section not found in context', context)

        self._add_result(self.setup_objects if context[SECTION_CONTEXT_KEY] == ast_parser.SETUP else self.used_objects, TYPE_RULES[rule][0](ast))

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if len(self.setup_objects) == 0:
            return 0

        return len(self.setup_objects.intersection(self.used_objects)) / len(self.setup_objects)


VARIABLE_TYPE_DEF_RULES_PATTERN = re.compile(r'[\w_]*variable_type_def')
EITHER_TYPES_RULES_PATTERN = re.compile(r'either[\w_]+types')

class SetupQuantifiedObjectsUsed(SetupObjectsUsed):
    def __init__(self, skip_objects: typing.Set[str] = SETUP_OBJECTS_SKIP_OBJECTS):
        super().__init__(skip_objects,
                         (VARIABLE_TYPE_DEF_RULES_PATTERN, EITHER_TYPES_RULES_PATTERN, 'pref_name_and_types'),
                         'setup_quantified_objects_used')


class NoAdjacentOnce(FitnessTerm):
    total_prefs: int = 0
    prefs_with_adjacent_once: int = 0

    def __init__(self):
        super().__init__('then', 'adjacent_once_found')

    def game_start(self) -> None:
        self.total_prefs = 0
        self.prefs_with_adjacent_once = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        self.total_prefs += 1

        if isinstance(ast.then_funcs, list):  # type: ignore
            func_rules = [sf.seq_func.parseinfo.rule if isinstance(sf.seq_func, tatsu.ast.AST) else sf.seq_func for sf in ast.then_funcs]  # type: ignore
            for i in range(len(func_rules) - 1):
                if func_rules[i] == func_rules[i + 1] == 'once':
                    self.prefs_with_adjacent_once += 1
                    break

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if self.total_prefs == 0:
            return 0

        return 1 if self.prefs_with_adjacent_once > 0 else 0


class NoAdjacentSameModal(FitnessTerm):
    then_found: bool = False
    adjacent_identical_modals_found: bool = False

    def __init__(self):
        super().__init__('then', 'no_adjacent_same_modal')

    def game_start(self) -> None:
        self.then_found = False
        self.adjacent_identical_modals_found = False

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast.then_funcs, list):  # type: ignore
            if isinstance(ast.then_funcs, list) and len(ast.then_funcs) > 1:  # type: ignore
                self.then_found = True

            func_rules = [sf.seq_func.parseinfo.rule if isinstance(sf.seq_func, tatsu.ast.AST) else sf.seq_func for sf in ast.then_funcs]  # type: ignore
            consecutive_counts = [len(list(group)) for _, group in itertools.groupby(func_rules)]
            if max(consecutive_counts) > 1:
                self.adjacent_identical_modals_found = True

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if not self.then_found:
            return 0

        return -1 if self.adjacent_identical_modals_found else 1


class PrefStartsAndEndsWithOnce(FitnessTerm):
    total_prefs: int = 0
    prefs_start_and_end_with_once: int = 0

    def __init__(self):
        super().__init__('then', 'starts_and_ends_once')

    def game_start(self) -> None:
        self.total_prefs = 0
        self.prefs_start_and_end_with_once = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        self.total_prefs += 1
        if isinstance(ast.then_funcs, list):  # type: ignore
            func_rules = [sf.seq_func.parseinfo.rule if isinstance(sf.seq_func, tatsu.ast.AST) else sf.seq_func for sf in ast.then_funcs]  # type: ignore
            if len(func_rules) >= 2 and func_rules[0] == func_rules[-1] == 'once':
                self.prefs_start_and_end_with_once += 1

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if self.total_prefs == 0:
            return 0

        return self.prefs_start_and_end_with_once / self.total_prefs


class OnceInMiddleOfPref(FitnessTerm):
    prefs_with_once_in_middle: int = 0

    def __init__(self):
        super().__init__('then', 'once_in_middle_of_pref_found')

    def game_start(self) -> None:
        self.prefs_with_once_in_middle = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast.then_funcs, list):  # type: ignore
            func_rules = [sf.seq_func.parseinfo.rule if isinstance(sf.seq_func, tatsu.ast.AST) else sf.seq_func for sf in ast.then_funcs]  # type: ignore
            if any(rule == 'once' for rule in func_rules[1:-1]):
                self.prefs_with_once_in_middle += 1

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return self.prefs_with_once_in_middle > 0


class PrefWithoutHold(FitnessTerm):
    prefs_without_hold: int = 0

    def __init__(self):
        super().__init__('then', 'pref_without_hold_found')

    def game_start(self) -> None:
        self.prefs_without_hold = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast.then_funcs, list):  # type: ignore
            func_rules = [sf.seq_func.parseinfo.rule if isinstance(sf.seq_func, tatsu.ast.AST) else sf.seq_func for sf in ast.then_funcs]  # type: ignore
            if not any(rule == 'hold' or rule == 'while_hold' for rule in func_rules):
                self.prefs_without_hold += 1

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return self.prefs_without_hold > 0


DEFAULT_LENGTH_OF_THEN_MIN_LENGTH = 1
DEFAULT_LENGTH_OF_THEN_MAX_LENGTH = 7


class LengthOfThenModals(FitnessTerm):
    then_lengths_found: typing.Dict[int, bool]

    def __init__(self, max_length: int = DEFAULT_LENGTH_OF_THEN_MAX_LENGTH, min_length: int = DEFAULT_LENGTH_OF_THEN_MIN_LENGTH):
        super().__init__('then', 'length_of_then_modals')
        self.max_length = max_length
        self.min_length = min_length

    def game_start(self) -> None:
        self.then_lengths_found = {i: False for i in range(self.min_length, self.max_length + 1)}

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast.then_funcs, list):  # type: ignore
            l = len(ast.then_funcs)  # type: ignore
            if l in self.then_lengths_found:
                self.then_lengths_found[l] = True
            else:
                self.then_lengths_found[self.max_length] = True

        else:
            self.then_lengths_found[1] = True

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return {k: int(v) for k, v in self.then_lengths_found.items()}

    def _get_all_inner_keys(self):
        return self.then_lengths_found.keys()


class AtEndFound(FitnessTerm):
    at_end_found: bool

    def __init__(self):
        super().__init__('at_end', 'at_end_found')

    def game_start(self) -> None:
        self.at_end_found = False

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        self.at_end_found = True

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return int(self.at_end_found)



DEFAULT_MAX_QUANTIFICATION_COUNT = 5


class MaxQuantificationCount(FitnessTerm):
    min_count: int = 0
    max_count: int = DEFAULT_MAX_QUANTIFICATION_COUNT
    max_count_by_section: typing.Dict[str, int] = {}

    def __init__(self, max_count: int = DEFAULT_MAX_QUANTIFICATION_COUNT, min_count: int = 0):
        super().__init__(re.compile(r'[\w_]+_(exists|forall)'), 'max_quantification_count')
        self.max_count = max_count
        self.min_count = min_count

    def game_start(self) -> None:
        self.max_count_by_section = {ast_parser.SETUP: 0, ast_parser.PREFERENCES: 0}

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            section = typing.cast(str, context[SECTION_CONTEXT_KEY])
            depth = typing.cast(int, context[QUANTIFICATIONS_CONTEXT_KEY] if QUANTIFICATIONS_CONTEXT_KEY in context else 0)
            if rule not in self.max_count_by_section:
                self.max_count_by_section[section] = depth
            else:
                self.max_count_by_section[section] = max(self.max_count_by_section[rule], depth)

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        output = {f'{section.replace("(:", "")}_{d}': 0 for section in self.max_count_by_section for d in range(self.min_count, self.max_count + 1)}
        for section, depth in self.max_count_by_section.items():
            output[f'{section.replace("(:", "")}_{np.clip(depth, self.min_count, self.max_count)}'] = 1

        return output  # type: ignore

    def _get_all_inner_keys(self):
        return [f'{section.replace("(:", "")}_{d}' for section in self.max_count_by_section for d in range(self.min_count, self.max_count + 1)]


DEFAULT_MAX_VARIABLES_TYPES = 8


class MaxNumberVariablesTypesQuantified(FitnessTerm):
    max_count: int = DEFAULT_MAX_VARIABLES_TYPES
    max_types_quantified: int = 0
    max_variables_quantified: int = 0
    min_count: int = 1

    def __init__(self, max_count: int = DEFAULT_MAX_VARIABLES_TYPES, min_count: int = 1):
        super().__init__(VARIABLE_TYPE_DEF_RULES_PATTERN, 'max_number_variables_types_quantified')
        self.max_count = max_count
        self.min_count = min_count

    def game_start(self) -> None:
        self.max_types_quantified = 0
        self.max_variables_quantified = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            var_type = ast.var_type.type  # type: ignore
            if isinstance(var_type, str):
                n_types = 1
            else:
                n_types = len(var_type.type_names) if isinstance(var_type.type_names, list) else 1
            self.max_types_quantified = max(self.max_types_quantified, n_types)

            variables = ast.var_names
            if isinstance(variables, str):
                n_variables = 1
            else:
                n_variables = len(variables)  # type: ignore
            self.max_variables_quantified = max(self.max_variables_quantified, n_variables)  # type: ignore

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        output = {f'types_{d}': 0 for d in range(self.min_count, self.max_count + 1)}
        output[f'types_{np.clip(self.max_types_quantified, self.min_count, self.max_count)}'] = 1

        output.update({f'variables_{d}': 0 for d in range(self.min_count, self.max_count + 1)})
        output[f'variables_{np.clip(self.max_variables_quantified, self.min_count, self.max_count)}'] = 1

        return output  # type: ignore

    def _get_all_inner_keys(self):
        return [f'{inner_key}_{d}' for inner_key in ['types', 'variables'] for d in range(self.min_count, self.max_count + 1)]


PREDICATE_AND_FUNCTION_RULES = ('predicate', 'function_eval')


class NoVariablesRepeated(FitnessTerm):
    count_with_repeats: int = 0
    total_count: int = 0
    variable_definition_repeated: bool = False

    def __init__(self):
        super().__init__(PREDICATE_AND_FUNCTION_RULES + ('variable_list',), 'repeated_variables_found')

    def game_start(self) -> None:
        self.total_count = 0
        self.count_with_repeats = 0
        self.variable_definition_repeated = False

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            if rule == 'variable_list':
                variables = ast.variables
                if not isinstance(variables, list):
                    variables = [variables]

                if VARIABLES_CONTEXT_KEY in context:
                    context_vars = context[VARIABLES_CONTEXT_KEY]
                    for var in variables:  # type: ignore
                        var_names = var.var_names  # type: ignore
                        if not isinstance(var_names, list):
                            var_names = [var_names]

                        if any(var_name in context_vars and isinstance(context_vars[var_name], list) for var_name in var_names):  # type: ignore
                            self.variable_definition_repeated = True
                            return

                all_variable_names = list(itertools.chain.from_iterable(var.var_names if isinstance(var.var_names, list) else [var.var_names] for var in variables))  # type: ignore
                if len(set(all_variable_names)) != len(all_variable_names):
                    self.variable_definition_repeated = True
            else:
                self.total_count += 1
                args = list(extract_predicate_function_args(ast))
                self.count_with_repeats += 1 if len(args) != len(set(args)) else 0

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if self.total_count == 0 or self.variable_definition_repeated:
            return 1

        return self.count_with_repeats != 0


ALL_BOOLEAN_RULE_PATTERN = re.compile(r'[\w_]+(_and|_or|_not)$')
MULTI_BOOLEAN_RULE_PATTERN = re.compile(r'[\w_]+(_and|_or)$')
LOGICALS_CHILD_KEY_OPTIONS = ['pred', 'setup', 'terminal']

class NoNestedLogicals(FitnessTerm):
    child_key_options: typing.Sequence[str]
    nested_logicals: int = 0
    total_logicals: int = 0


    def __init__(self, child_key_options: typing.Sequence[str] = LOGICALS_CHILD_KEY_OPTIONS):
        super().__init__(ALL_BOOLEAN_RULE_PATTERN, 'nested_logicals_found')
        self.child_key_options = child_key_options

    def game_start(self) -> None:
        self.total_logicals = 0
        self.nested_logicals = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            self.total_logicals += 1

            if rule.endswith('_not'):
                if isinstance(ast.not_args, tatsu.ast.AST) and isinstance(ast.not_args.pred, tatsu.ast.AST) and ast.not_args.pred.parseinfo.rule == rule:  # type: ignore
                    self.nested_logicals += 1

            else:
                rule_name = rule.split('_')[-1]
                children = ast[f'{rule_name}_args']
                if isinstance(children, tatsu.ast.AST):
                    children = [children]

                child_key = None
                for child in children:  # type: ignore
                    if isinstance(child, tatsu.ast.AST):
                        for key in self.child_key_options:
                            if key in child:
                                child_key = key
                                break

                if child_key is None:
                    raise ValueError(f'No logical child key found for rule {rule} in {ast}')

                if any(isinstance(child, tatsu.ast.AST) and (isinstance(child[child_key], tatsu.ast.AST) and child[child_key].parseinfo.rule == rule) for child in children):  # type: ignore
                    self.nested_logicals += 1

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if self.total_logicals == 0:
            # TODO: should this return a NaN? If so, we should think about how to handle them
            # currently assuming that games without logicals are rare enough that this is fine
            return 0

        return self.nested_logicals != 0


class NoIdenticalChildrenInLogicals(FitnessTerm):
    total_logicals: int = 0
    identical_children: int = 0

    def __init__(self):
        self.rule_to_section = {
            'setup_and': ast_parser.SETUP,
            'setup_or': ast_parser.SETUP,
            'super_predicate_and': ast_parser.PREFERENCES,
            'super_predicate_or': ast_parser.PREFERENCES,
            'terminal_and': ast_parser.TERMINAL,
            'terminal_or': ast_parser.TERMINAL,
            'scoring_and': ast_parser.SCORING,
            'scoring_or': ast_parser.SCORING,
        }
        super().__init__(tuple(self.rule_to_section.keys()), 'identical_logical_children_found')

    def game_start(self) -> None:
        self.total_logicals = 0
        self.identical_children = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            rule_name = rule.split('_')[-1]
            children = ast[f'{rule_name}_args']
            if isinstance(children, tatsu.ast.AST) or len(children) < 2:  # type: ignore
                return

            self.total_logicals += 1

            children_strs = [ast_printer.ast_section_to_string(child, self.rule_to_section[rule]) for child in children]  # type: ignore
            if len(set(children_strs)) != len(children_strs):
                self.identical_children += 1

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if self.total_logicals == 0:
            return 0

        return self.identical_children != 0


class NoIdenticalChildrenInScoringExpressions(FitnessTerm):
    total_scoring_multi_expressions: int = 0
    identical_children: int = 0

    def __init__(self):
        super().__init__(('scoring_equals_comp', 'scoring_multi_expr'), 'identical_scoring_children_found')

    def game_start(self) -> None:
        self.total_scoring_multi_expressions = 0
        self.identical_children = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            children = ast.expr
            if isinstance(children, tatsu.ast.AST) or len(children) < 2:  # type: ignore
                return

            self.total_scoring_multi_expressions += 1

            children_strs = [ast_printer.ast_section_to_string(child, ast_parser.SCORING) for child in children]  # type: ignore
            if len(set(children_strs)) != len(children_strs):
                self.identical_children += 1

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if self.total_scoring_multi_expressions == 0:
            return 0

        return self.identical_children != 0


class ScoringCountExpressionRepetitions(FitnessTerm):
    scoring_expression_to_count: typing.Dict[str, int] = {}

    def __init__(self):
        super().__init__(COUNT_RULE_PATTERN, 'scoring_count_expression_repetitions_max')

    def game_start(self) -> None:
        self.scoring_expression_to_count = defaultdict(int)

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            self.scoring_expression_to_count[ast_printer.ast_section_to_string(ast, ast_parser.SCORING)] += 1

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if len(self.scoring_expression_to_count) == 0:
            return 0

        values = [v - 1 for v in self.scoring_expression_to_count.values()]
        return max(values)  # type: ignore



BOOLEAN_PARSER = ast_parser.ASTBooleanParser()
BOOLEAN_LOGIC_IGNORE_PREFIXES = ('terminal_', 'scoring_')


class BooleanLogicTerm(FitnessTerm):
    boolean_parser: ast_parser.ASTBooleanParser
    def __init__(self, name: str, ignore_prefixes: typing.Sequence[str] = BOOLEAN_LOGIC_IGNORE_PREFIXES,
                 rules_or_pattern: typing.Union[typing.Sequence[str], typing.Pattern] = MULTI_BOOLEAN_RULE_PATTERN):
        super().__init__(rules_or_pattern, name)
        self.boolean_parser = BOOLEAN_PARSER
        self.ignore_prefixes = ignore_prefixes

    def game_start(self) -> None:
        self.boolean_parser.game_start()

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if any(rule.startswith(prefix) for prefix in self.ignore_prefixes):
            return

        if isinstance(ast, tatsu.ast.AST):
            expr = self.boolean_parser(ast, **simplified_context_deepcopy(context))
            self._inner_update(expr, rule, context)  # type: ignore

    @abstractmethod
    def _inner_update(self, expr: boolean.Expression, rule: str, context: ContextDict):
        pass


class TautologicalBooleanExpression(BooleanLogicTerm):
    def __init__(self):
        super().__init__('tautological_expression_found')
        self.tautalogy_found = False

    def game_start(self) -> None:
        super().game_start()
        self.tautalogy_found = False

    def _inner_update(self, expr: boolean.Expression, rule: str, context: ContextDict):
        if self.boolean_parser.evaluate_tautology(expr):  # type: ignore
            self.tautalogy_found = True

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return self.tautalogy_found


class RedundantBooleanExpression(BooleanLogicTerm):
    def __init__(self):
        super().__init__('redundant_expression_found')
        self.redundancy_found = False

    def game_start(self) -> None:
        super().game_start()
        self.redundancy_found = False

    def _inner_update(self, expr: boolean.Expression, rule: str, context: ContextDict):
        if self.boolean_parser.evaluate_redundancy(expr):  # type: ignore
            self.redundancy_found = True

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return self.redundancy_found


class RedundantBooleanScoringTerminalExpression(BooleanLogicTerm):
    def __init__(self):
        super().__init__('redundant_scoring_terminal_expression_found', rules_or_pattern=('terminal_and', 'terminal_or', 'terminal_comp', 'scoring_comparison'),
                         ignore_prefixes=[])
        self.redundancy_found = False

    def game_start(self) -> None:
        super().game_start()
        self.redundancy_found = False

    def _inner_update(self, expr: boolean.Expression, rule: str, context: ContextDict):
        if self.boolean_parser.evaluate_redundancy(expr):  # type: ignore
            self.redundancy_found = True

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return self.redundancy_found


class IdenticalConsecutiveSeqFuncPredicates(FitnessTerm):
    def __init__(self):
        super().__init__(('then',), 'identical_consecutive_seq_func_predicates_found')
        self.boolean_parser = BOOLEAN_PARSER

    def game_start(self) -> None:
        self.boolean_parser.game_start()
        self.identical_predicates_found = False

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            preds = []
            for seq_func in ast.then_funcs:  # type: ignore
                if isinstance(seq_func, str):
                    preds.append(seq_func)
                    continue

                s = seq_func.seq_func
                pred_key = [k for k in s.keys() if k.endswith('_pred')][0]
                preds.append(str(self.boolean_parser(s[pred_key], **simplified_context_deepcopy(context))))  # type: ignore

            if any(preds[i] == preds[i + 1] for i in range(len(preds) - 1)):
                self.identical_predicates_found = True

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return self.identical_predicates_found


class ASTPredicateTermTracker(ast_parser.ASTParser):
    def __call__(self, ast, **kwargs) -> typing.Set[str]:
        self._default_kwarg(kwargs, 'terms', set())
        super().__call__(ast, **kwargs)
        return kwargs['terms']

    def _handle_ast(self, ast: tatsu.ast.AST, **kwargs):
        if ast.parseinfo.rule in ('predicate', 'function_eval'):  # type: ignore
            kwargs['terms'].update(extract_predicate_function_args(ast))

        return super()._handle_ast(ast, **kwargs)


class DisjointSeqFuncPredicateTerm(FitnessTerm):
    disjoint_found: bool
    term_tracker: ASTPredicateTermTracker

    def __init__(self):
        super().__init__(('then',), 'disjoint_seq_funcs_found')
        self.disjoint_found = False
        self.term_tracker = ASTPredicateTermTracker()

    def game_start(self) -> None:
        self.disjoint_found = False

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        seq_func_terms = [self.term_tracker(seq_func) for seq_func in ast.then_funcs if not isinstance(seq_func, str)]  # type: ignore
        seq_func_terms = [terms for terms in seq_func_terms if len(terms) > 0]

        if len(seq_func_terms) > 1:
            for i, j in itertools.combinations(range(len(seq_func_terms)), 2):
                if len(seq_func_terms[i].intersection(seq_func_terms[j])) == 0:
                    self.disjoint_found = True
                    return

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return self.disjoint_found


class PrefForallTerm(FitnessTerm):
    keys: typing.List[str]
    pref_forall_found: bool = False
    pref_forall_prefs: typing.Set[str] = set()
    pref_forall_prefs_by_position: typing.Dict[int, typing.Set[str]] = {}
    return_incorrect_count: bool = False

    def __init__(self, name: str, return_incorrect_count: bool = False):
        super().__init__(('scoring_external_maximize', 'scoring_external_minimize', 'pref_forall', COUNT_RULE_PATTERN), f'pref_forall_{name}')
        self.return_incorrect_count = return_incorrect_count
        self.keys = ['correct', 'incorrect']
        if self.return_incorrect_count:
            self.keys.append('incorrect_count')

    def game_start(self) -> None:
        self.pref_forall_found = False
        self.pref_forall_prefs = set()
        self.pref_forall_prefs_by_position = defaultdict(set)
        self._inner_game_start()

    @abstractmethod
    def _inner_game_start(self) -> None:
        pass

    def _update_pref_forall_def(self, ast: tatsu.ast.AST, context: ContextDict):
        preferences = ast.forall_pref.preferences  # type: ignore
        if isinstance(preferences, tatsu.ast.AST):
            preferences = [preferences]

        self.pref_forall_prefs.update([pref.pref_name for pref in preferences])  # type: ignore
        self.pref_forall_prefs_by_position[ast.parseinfo.pos].update([pref.pref_name for pref in preferences])  # type: ignore

    # Not abstract as is optional
    def _update_external_forall(self, ast: tatsu.ast.AST, context: ContextDict):
        pass

    @abstractmethod
    def _update_count(self, pref_name: str, object_types: typing.Optional[typing.List[tatsu.ast.AST]],
        rule: str, context: ContextDict):
        pass

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            if rule == 'pref_forall':
                self.pref_forall_found = True
                self._update_pref_forall_def(ast, context)

            elif rule in ('scoring_external_maximize', 'scoring_external_minimize'):
                self._update_external_forall(ast, context)

            else:   # count*
                pref_name = ast.name_and_types['pref_name']  # type: ignore
                object_types = ast.name_and_types['object_types']  # type: ignore
                self._update_count(pref_name, object_types, rule, context)

    @abstractmethod
    def _inner_game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        pass

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return_value = {k: 0 for k in self.keys}

        inner_value = float(self._inner_game_end())  # type: ignore
        if inner_value > 0:
            return_value['correct'] = 1
        elif inner_value < 0:
            return_value['incorrect'] = 1  # abs(inner_value)
            if self.return_incorrect_count:
                return_value['incorrect_count'] = abs(inner_value)  # type: ignore

        return return_value  # type: ignore

    def _get_all_inner_keys(self):
        return self.keys


class CountOncePerExternalObjectsUsedCorrectly(PrefForallTerm):
    pref_forall_prefs: typing.Set[str] = set()
    count_once_per_external_objects_prefs: typing.Dict[str, int]

    def __init__(self):
        super().__init__('count_once_per_external_objects_used', return_incorrect_count=True)

    def _inner_game_start(self) -> None:
        self.count_once_per_external_objects_prefs = defaultdict(int)

    def _update_count(self, pref_name: str, object_types: typing.Optional[typing.List[tatsu.ast.AST]],
        rule: str, context: ContextDict):

        if rule == 'count_once_per_external_objects':
            self.count_once_per_external_objects_prefs[pref_name] += 1

    def _inner_game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if len(self.count_once_per_external_objects_prefs) == 0:
            return 0

        count_once_per_external_objects_pref_names = set(self.count_once_per_external_objects_prefs.keys())

        if len(count_once_per_external_objects_pref_names.intersection(self.pref_forall_prefs)) == len(self.count_once_per_external_objects_prefs):
            return 1

        return - sum([
            self.count_once_per_external_objects_prefs[pref_name] if pref_name in self.count_once_per_external_objects_prefs else 1
            for pref_name in count_once_per_external_objects_pref_names.symmetric_difference(self.pref_forall_prefs)
        ])


class ExternalForallUsedCorrectly(PrefForallTerm):
    pref_forall_prefs: typing.Set[str] = set()
    external_forall_positions: typing.Set[int] = set()
    external_forall_used_with_forall_pref_positions: typing.Set[int] = set()

    def __init__(self):
        super().__init__('external_forall_used', return_incorrect_count=True)

    def _inner_game_start(self) -> None:
        self.external_forall_positions = set()
        self.external_forall_used_with_forall_pref_positions = set()

    def _update_external_forall(self, ast: tatsu.ast.AST, context: ContextDict):
        self.external_forall_positions.add(ast.parseinfo.pos)  # type: ignore

    def _update_count(self, pref_name: str, object_types: typing.Optional[typing.List[tatsu.ast.AST]],
        rule: str, context: ContextDict):
        if pref_name in self.pref_forall_prefs and EXTERNAL_FORALL_CONTEXT_KEY in context:
            self.external_forall_used_with_forall_pref_positions.add(context[EXTERNAL_FORALL_CONTEXT_KEY].pos)  # type: ignore

    def _inner_game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if len(self.external_forall_positions) == 0:
            return 0

        if len(self.external_forall_positions.intersection(self.external_forall_used_with_forall_pref_positions)) == len(self.external_forall_positions):
            return 1

        return - len(self.external_forall_positions.symmetric_difference(self.external_forall_used_with_forall_pref_positions))


class PrefForallUsed(PrefForallTerm):
    prefs_used_as_pref_forall_prefs: typing.Dict[str, int]

    def __init__(self):
        super().__init__('used', return_incorrect_count=True)

    def _inner_game_start(self) -> None:
        self.prefs_used_as_pref_forall_prefs = defaultdict(int)

    def _update_count(self, pref_name: str, object_types: typing.Optional[typing.List[tatsu.ast.AST]],
        rule: str, context: ContextDict):
        if object_types is not None or EXTERNAL_FORALL_CONTEXT_KEY in context or rule == 'count_once_per_external_objects':
            self.prefs_used_as_pref_forall_prefs[pref_name] += 1

    def _inner_game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if len(self.pref_forall_prefs) == 0 and len(self.prefs_used_as_pref_forall_prefs) == 0:
            return 0

        n_unused_prefs = 0

        prefs_used_as_pref_forall_prefs_names = set(self.prefs_used_as_pref_forall_prefs.keys())

        # for set of preferences defined as a forall pref
        for pos_prefs in self.pref_forall_prefs_by_position.values():
            # if none of them were used as a forall pref, return -1
            n_pos_prefs_used_as_forall = len(pos_prefs.intersection(prefs_used_as_pref_forall_prefs_names))
            if n_pos_prefs_used_as_forall == 0:
                n_unused_prefs += len(pos_prefs)

            # remove the preferences that were used as a forall pref
            prefs_used_as_pref_forall_prefs_names.difference_update(pos_prefs)

        # if there are any preferences left that were used as a forall pref, return -1, since they were not defined as forall prefs
        if len(prefs_used_as_pref_forall_prefs_names) > 0:
            n_unused_prefs += sum(self.prefs_used_as_pref_forall_prefs[pref_name] for pref_name in prefs_used_as_pref_forall_prefs_names)

        if n_unused_prefs == 0:
            return 1

        return -n_unused_prefs

class PrefForallCorrectArity(PrefForallTerm):
    correct_usage_count: int = 0
    incorrect_usage_count: int = 0
    pref_forall_prefs_to_counts: typing.Dict[str, int] = dict()

    def __init__(self):
        super().__init__('pref_forall_correct_arity', return_incorrect_count=True)

    def _inner_game_start(self) -> None:
        self.pref_forall_prefs_to_counts = dict()
        self.correct_usage_count = 0
        self.incorrect_usage_count = 0

    def _update_pref_forall_def(self, ast: tatsu.ast.AST, context: ContextDict):
        preferences = ast.forall_pref.preferences  # type: ignore
        if isinstance(preferences, tatsu.ast.AST):
            preferences = [preferences]

        vars = ast.forall_vars.variables  # type: ignore

        if isinstance(vars, tatsu.ast.AST):
            n_vars = 1
        else:
            n_vars = len(vars)

        for pref in preferences:
            self.pref_forall_prefs_to_counts[pref.pref_name] = n_vars  # type: ignore

    def _update_count(self, pref_name: str, object_types: typing.Optional[typing.List[tatsu.ast.AST]],
        rule: str, context: ContextDict):
        if object_types is None:
            n_vars = 0
        elif isinstance(object_types, tatsu.ast.AST):
            n_vars = 1
        else:
            n_vars = len(object_types)

        if pref_name in self.pref_forall_prefs_to_counts and n_vars <= self.pref_forall_prefs_to_counts[pref_name]:
            self.correct_usage_count += 1
        elif n_vars > 0:
            self.incorrect_usage_count += 1

    def _inner_game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if len(self.pref_forall_prefs_to_counts) == 0:
            return 0

        total_usage_count = self.correct_usage_count + self.incorrect_usage_count
        if total_usage_count == 0:
            return 0

        return 1 if self.incorrect_usage_count == 0 else -self.incorrect_usage_count


class PrefForallCorrectTypes(PrefForallTerm):
    pref_forall_prefs_to_types: typing.Dict[str, typing.Dict[str, VariableDefinition]] = defaultdict(dict)
    n_incorrect_types_per_pref: typing.List[float] = list()

    def __init__(self):
        super().__init__('pref_forall_correct_types', return_incorrect_count=True)

    def _inner_game_start(self) -> None:
        self.pref_forall_prefs_to_types = defaultdict(dict)
        self.n_incorrect_types_per_pref = list()

    def _update_pref_forall_def(self, ast: tatsu.ast.AST, context: ContextDict):
        preferences = ast.forall_pref.preferences  # type: ignore
        if isinstance(preferences, tatsu.ast.AST):
            preferences = [preferences]

        var_dict = {}
        extract_variables_from_ast(ast, 'forall_vars', var_dict)

        for pref in preferences:
            self.pref_forall_prefs_to_types[pref.pref_name] = var_dict  # type: ignore

    def _update_count(self, pref_name: str, object_types: typing.Optional[typing.List[tatsu.ast.AST]],
        rule: str, context: ContextDict):
        if object_types is None:
            return

        if pref_name not in self.pref_forall_prefs_to_types:
            return

        elif isinstance(object_types, tatsu.ast.AST):
            object_types = [object_types]

        if len(object_types) > len(self.pref_forall_prefs_to_types[pref_name]):
            return

        count_correct = 0
        for obj_type, (_, var_def) in zip(object_types, self.pref_forall_prefs_to_types[pref_name].items()):
            obj = obj_type.type_name
            var_types = var_def.var_types
            if obj in var_types or (obj in room_and_object_types.TYPE_TO_META_TYPE and room_and_object_types.TYPE_TO_META_TYPE[obj] in var_types):  # type: ignore
                count_correct += 1

        self.n_incorrect_types_per_pref.append(len(object_types) - count_correct)

    def _inner_game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if len(self.pref_forall_prefs_to_types) == 0 or len(self.n_incorrect_types_per_pref) == 0:
            return 0

        total_incorrect_types = sum(self.n_incorrect_types_per_pref)
        return 1 if total_incorrect_types == 0 else -total_incorrect_types


TOTAL_TERMINALS = ('(total-time)', '(total-score)')


class SectionWithoutPrefOrTotalCounts(FitnessTerm):
    preference_count_found: bool
    section: str
    section_found: bool
    section_name: str
    total_found: bool

    def __init__(self, section: str):
        self.section = section
        self.section_name = section.replace("(:", "")
        super().__init__(section, f'section_without_pref_or_total_count_{self.section_name}')

    def game_start(self) -> None:
        self.preference_count_found = False
        self.section_found = False
        self.total_found = False

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if SECTION_CONTEXT_KEY in context and context[SECTION_CONTEXT_KEY] == self.section:
            self.section_found = True

        if isinstance(ast, tatsu.ast.AST) and COUNT_RULE_PATTERN.match(rule):
            self.preference_count_found = True

        if rule in ('terminal_expr', 'scoring_expr') and isinstance(ast.expr, str) and ast.expr in TOTAL_TERMINALS:  # type:ignore
            self.total_found = True

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        # Retrun 0 if either the section isn't in the game, or there's a preference count, or it's a terminal with a total-count
        if not self.section_found or self.preference_count_found or (self.section_name == 'terminal' and self.total_found):
            return 0

        # Otherwise return 1
        return 1


PREDICATE_FUNCTION_ARITY_MAP = {
    'above': 2, 'adjacent': 2, 'adjacent_side_3': 3, 'adjacent_side_4': 4,
    'agent_crouches': 0, 'agent_holds': 1, 'between': 3, 'broken': 1,
    'building_size': 1, 'distance': 2, 'distance_side_3': 3, 'distance_side_4': 4,
    'equal_x_position': 2, 'equal_z_position': 2, 'faces': 2,
    'game_over': 0, 'game_start': 0, 'in': 2, 'in_motion': 1, 'is_setup_object': 1,
    'object_orientation': 2, 'on': 2, 'open': 1, 'opposite': 2, 'rug_color_under': 2,
    'same_color': 2, 'same_object': 2, 'same_type': 2, 'toggled_on': 1, 'touch': 2,
    'x_position': 1,
}


class PredicateArgumentSymmetryType(Enum):
    ALL_ARGUMENTS = 0
    FIRST_AND_THIRD_ARGUMENTS = 1


SYMMETRIC_PREDICATE_ARG_INDICES = {
    'adjacent': PredicateArgumentSymmetryType.ALL_ARGUMENTS,
    'adjacent_side_4': PredicateArgumentSymmetryType.FIRST_AND_THIRD_ARGUMENTS,
    'between': PredicateArgumentSymmetryType.FIRST_AND_THIRD_ARGUMENTS,
    'distance': PredicateArgumentSymmetryType.ALL_ARGUMENTS,
    'distance_side_4': PredicateArgumentSymmetryType.FIRST_AND_THIRD_ARGUMENTS,
    'equal_x_position': PredicateArgumentSymmetryType.ALL_ARGUMENTS,
    'equal_z_position': PredicateArgumentSymmetryType.ALL_ARGUMENTS,
    'faces': PredicateArgumentSymmetryType.ALL_ARGUMENTS,
    'opposite': PredicateArgumentSymmetryType.ALL_ARGUMENTS,
    'same_color': PredicateArgumentSymmetryType.ALL_ARGUMENTS,
    'same_object': PredicateArgumentSymmetryType.ALL_ARGUMENTS,
    'same_type': PredicateArgumentSymmetryType.ALL_ARGUMENTS,
    'touch': PredicateArgumentSymmetryType.ALL_ARGUMENTS,
}


# class CorrectPredicateFunctionArity(FitnessTerm):
#     total_count: int = 0
#     name_to_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]] = {}
#     count_with_wrong_arity: int = 0

#     def __init__(self, name_to_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]] = PREDICATE_FUNCTION_ARITY_MAP):  # type: ignore
#         super().__init__(PREDICATE_AND_FUNCTION_RULES, 'correct_predicate_function_arity')
#         self.name_to_arity_map = name_to_arity_map

#     def game_start(self) -> None:
#         self.total_count = 0
#         self.count_with_wrong_arity = 0

#     def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
#         if isinstance(ast, tatsu.ast.AST):
#             self.total_count += 1

#             name = extract_predicate_function_name(ast, remove_digits=False)

#             if name not in self.name_to_arity_map:
#                 raise ValueError(f'Predicate {name} not in predicate arity map')

#             n_args = extract_n_args(ast)
#             arity = self.name_to_arity_map[name]  # type: ignore

#             if isinstance(arity, int):
#                 if n_args != arity:
#                     self.count_with_wrong_arity += 1

#             elif n_args not in arity:
#                 self.count_with_wrong_arity += 1

#     def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
#         if self.total_count == 0:
#             return 0

#         return 1 - (self.count_with_wrong_arity / self.total_count)


TWO_ARG_COMPARISON_RULE = 'two_arg_comparison'
MULTIPLE_ARG_COMPARISON_RULE = 'multiple_args_equal_comparison'
TERMINAL_COMP = 'terminal_comp'
SCORING_COMP = 'scoring_comp'
SCORING_EQUALS_COMP = 'scoring_equals_comp'
SCORING_BINARY_EXPR = 'scoring_binary_expr'
SCORING_MULTI_EXPR = 'scoring_multi_expr'

TWO_NUMBER_RULES = (
    TWO_ARG_COMPARISON_RULE, MULTIPLE_ARG_COMPARISON_RULE,
    TERMINAL_COMP, SCORING_COMP, SCORING_EQUALS_COMP,
    SCORING_BINARY_EXPR, SCORING_MULTI_EXPR)


def _is_number(s: typing.Any) -> bool:
    if isinstance(s, str):
        return s.replace('.', '', 1).isdigit()

    elif isinstance(s, tatsu.ast.AST):
        if s.parseinfo.rule in ('scoring_expr', 'scoring_neg_expr'):  # type: ignore
            return _is_number(s.expr)

    elif isinstance(s, (int, float)):
        return True

    return False


class NoTwoNumberOperations(FitnessTerm):
    total_operations: int = 0
    two_number_operations: int = 0

    def __init__(self):
        super().__init__(TWO_NUMBER_RULES, 'two_number_operation_found')

    def game_start(self) -> None:
        self.total_operations = 0
        self.two_number_operations = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            self.total_operations += 1

            if rule == TWO_ARG_COMPARISON_RULE:
                if _is_number(ast.arg_1.arg) and _is_number(ast.arg_2.arg):  # type: ignore
                    self.two_number_operations += 1

            elif rule == MULTIPLE_ARG_COMPARISON_RULE:
                args = ast.equal_comp_args
                if isinstance(args, tatsu.ast.AST):
                    return

                if all(_is_number(arg.arg) for arg in args) or len(args) <= 1:  # type: ignore
                    self.two_number_operations += 1

            elif rule == TERMINAL_COMP:
                first_number = 'expr' in ast.expr_1.expr and _is_number(ast.expr_1.expr.expr)  # type: ignore
                second_number = 'expr' in ast.expr_2.expr and _is_number(ast.expr_2.expr.expr)  # type: ignore
                if first_number and second_number:
                    self.two_number_operations += 1

            elif rule == SCORING_COMP or rule == SCORING_BINARY_EXPR:
                if _is_number(ast.expr_1.expr) and _is_number(ast.expr_2.expr):  # type: ignore
                    self.two_number_operations += 1

            elif rule == SCORING_EQUALS_COMP or rule == SCORING_MULTI_EXPR:
                args = ast.expr
                if isinstance(args, tatsu.ast.AST):
                    return

                if all(_is_number(arg.expr) for arg in args) or len(args) <= 1:  # type: ignore
                    self.two_number_operations += 1

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if self.total_operations == 0:
            return 0

        return self.two_number_operations > 0


class NoSingleArgumentMultiOperations(FitnessTerm):
    total_operations: int = 0
    single_argument_multi_operations: int = 0

    def __init__(self):
        super().__init__((MULTIPLE_ARG_COMPARISON_RULE, SCORING_EQUALS_COMP, SCORING_MULTI_EXPR), 'single_argument_multi_operation_found')

    def game_start(self) -> None:
        self.total_operations = 0
        self.single_argument_multi_operations = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            self.total_operations += 1

            if rule == MULTIPLE_ARG_COMPARISON_RULE:
                args = ast.equal_comp_args
                if isinstance(args, tatsu.ast.AST) or len(args) == 1:  # type: ignore
                    self.single_argument_multi_operations += 1

            elif rule == SCORING_EQUALS_COMP or rule == SCORING_MULTI_EXPR:
                args = ast.expr
                if isinstance(args, tatsu.ast.AST) or len(args) == 1:  # type: ignore
                    self.single_argument_multi_operations += 1

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if self.total_operations == 0:
            return 0

        return self.single_argument_multi_operations > 0




# COMMON_SENSE_PREDICATES_FUNCTIONS = ('adjacent', 'agent_holds', 'distance', 'in', 'in_motion', 'on', 'touch')
COMMON_SENSE_PREDICATES_FUNCTIONS = ('adjacent', 'adjacent_side_3', 'agent_holds', 'between', 'distance', 'in', 'in_motion', 'object_orientation', 'on', 'touch')
COMMON_SENSE_TYPE_CATEGORIES = list(room_and_object_types.CATEGORIES_TO_TYPES.keys())
COMMON_SENSE_TYPE_CATEGORIES.remove(room_and_object_types.EMPTY_OBJECT)
KNOWN_MISSING_TYPES = []


MODALS = ['once', 'once_measure', 'hold', 'while_hold', 'at_end']  # , 'hold_for']


class PredicateUnderModal(FitnessTerm):
    modals: typing.Set[str]
    predicates_or_functions: typing.Set[str]
    modal_to_predicate_map: typing.Dict[str, typing.Dict[str, bool]]

    def __init__(self, modals: typing.Sequence[str], predicates_or_functions: typing.Sequence[str],):
        super().__init__(PREDICATE_AND_FUNCTION_RULES, 'predicate_under_modal')
        self.modals = set(modals)
        self.predicates_or_functions = set(predicates_or_functions)
        self.modal_to_predicate_map = {modal: {predicate_or_function: False for predicate_or_function in predicates_or_functions} for modal in modals}

    def game_start(self) -> None:
        self.modal_to_predicate_map = {modal: {predicate_or_function: False for predicate_or_function in self.predicates_or_functions} for modal in self.modals}

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            if rule == 'predicate':
                pred = ast.pred.parseinfo.rule.replace('predicate_', '')  # type: ignore

            else:
                pred = ast.func.parseinfo.rule.replace('function_', '')  # type: ignore

            if MODAL_CONTEXT_KEY not in context:
                return

            modal = context[MODAL_CONTEXT_KEY]

            if modal in self.modals and pred in self.predicates_or_functions:
                self.modal_to_predicate_map[modal][pred] = True

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return {f'{modal}_{pred}': 1 if self.modal_to_predicate_map[modal][pred] else 0  for modal in self.modals for pred in self.predicates_or_functions}

    def _get_all_inner_keys(self):
        return [f'{modal}_{pred}' for modal in self.modals for pred in self.predicates_or_functions]


class PredicateFunctionArgumentTypes(FitnessTerm):
    # argument_type_categories: typing.Sequence[str]
    # matching_argument_types_count: int = 0
    argument_types_to_count: typing.Dict[typing.Tuple[str, ...], int]
    argument_types_to_count_by_section: typing.Dict[str, typing.Dict[typing.Tuple[str, ...], int]]
    name_to_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]]
    predicate_or_function: str
    symmetric_predicate_symmetry_types: typing.Dict[str, PredicateArgumentSymmetryType]
    type_categories: typing.Sequence[str]

    def __init__(self, predicate_or_function: str, # argument_type_categories: typing.Sequence[str],
        name_to_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]] = PREDICATE_FUNCTION_ARITY_MAP,  # type: ignore
        known_missing_types: typing.Sequence[str] = KNOWN_MISSING_TYPES,
        type_categories: typing.Sequence[str] = COMMON_SENSE_TYPE_CATEGORIES,
        symmetric_predicate_symmetry_types: typing.Dict[str, PredicateArgumentSymmetryType] = SYMMETRIC_PREDICATE_ARG_INDICES,
        ):

        super().__init__((f'predicate_{predicate_or_function}', f'function_{predicate_or_function}'),
            # f'{predicate_or_function}_arg_types_{"_".join(argument_type_categories)}'
            f'{predicate_or_function}_arg_types'
            )
        self.predicate_or_function = predicate_or_function
        # self.argument_type_categories = argument_type_categories
        self.name_to_arity_map = name_to_arity_map
        self.known_missing_types = known_missing_types
        self.type_categories = list(sorted(type_categories))
        self.symmetric_predicate_symmetry_types = symmetric_predicate_symmetry_types

        self.argument_types_to_count = defaultdict(int)
        self.argument_types_to_count_by_section = {ast_parser.SETUP: defaultdict(int), ast_parser.PREFERENCES: defaultdict(int)}

    def game_start(self) -> None:
        # self.matching_argument_types_count = 0
        self.argument_types_to_count = defaultdict(int)
        self.argument_types_to_count_by_section = {ast_parser.SETUP: defaultdict(int), ast_parser.PREFERENCES: defaultdict(int)}

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            name = extract_predicate_function_name(ast, remove_digits=False)

            if name != self.predicate_or_function:
                return

            # if name not in self.name_to_arity_map:
            #     raise ValueError(f'Predicate {ast.name} not in predicate arity map')

            # n_args = extract_n_args(ast)
            # arity = self.name_to_arity_map[name]  # type: ignore

            # if isinstance(arity, int):
            #     if n_args != arity:
            #         return

            # elif n_args not in arity:
            #     return

            terms = extract_predicate_function_args(ast)
            context_variables = typing.cast(typing.Dict[str, typing.Union[VariableDefinition, typing.List[VariableDefinition]]], context[VARIABLES_CONTEXT_KEY]) if VARIABLES_CONTEXT_KEY in context else {}
            term_categories = [predicate_function_term_to_type_categories(term, context_variables, self.known_missing_types) for term in terms]
            if any(term_category is None for term_category in term_categories):
                return

            for category_product in itertools.product(*term_categories):  # type: ignore
                if name in self.symmetric_predicate_symmetry_types:
                    symmetry_type = self.symmetric_predicate_symmetry_types[name]
                    if symmetry_type == PredicateArgumentSymmetryType.ALL_ARGUMENTS:
                        category_product = tuple(sorted(category_product))

                    elif symmetry_type == PredicateArgumentSymmetryType.FIRST_AND_THIRD_ARGUMENTS:
                        if category_product[0] > category_product[2]:
                            category_product = (category_product[2], category_product[1], category_product[0], *category_product[3:])

                    else:
                        raise ValueError(f'Unknown symmetry type {symmetry_type}')

                self.argument_types_to_count[category_product] += 1
                self.argument_types_to_count_by_section[context[SECTION_CONTEXT_KEY]][category_product] += 1  # type: ignore

    #         if all(term_categories[i] is not None and self.argument_type_categories[i] in term_categories[i] for i in range(len(term_categories))):  # type: ignore
    #             self._count(context)

    # def _count(self, context: ContextDict):
    #     self.matching_argument_types_count += 1

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        # output = {'_'.join(argument_types): count for argument_types, count in self.argument_types_to_count.items()}
        output = {}
        for section, section_results in self.argument_types_to_count_by_section.items():
            s = section.replace('(:', '')
            output.update({f'{"_".join(argument_types)}_{s}': count for argument_types, count in section_results.items()})

        return output  # type: ignore
        # return self.matching_argument_types_count

    def _get_all_inner_keys(self):
        if self.predicate_or_function not in self.name_to_arity_map:
            raise ValueError(f'Predicate {self.predicate_or_function} not in predicate arity map')

        arity = self.name_to_arity_map[self.predicate_or_function]  # type: ignore

        return [f'{"_".join(type_combinations)}_{section.replace("(:", "")}'
                for section in self.argument_types_to_count_by_section
                for type_combinations in itertools.product(*([self.type_categories] * arity))]  # type: ignore


PREDICATE_SECTIONS = [ast_parser.SETUP, ast_parser.PREFERENCES]


# class PredicateFunctionArgumentTypesBySection(PredicateFunctionArgumentTypes):
#     matching_argument_types_count_by_section: typing.Dict[str, Number]

#     def __init__(self, predicate_or_function: str, # argument_type_categories: typing.Sequence[str],
#         name_to_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]] = PREDICATE_FUNCTION_ARITY_MAP,  # type: ignore
#         known_missing_types: typing.Sequence[str] = KNOWN_MISSING_TYPES):

#         super().__init__(predicate_or_function, argument_type_categories, name_to_arity_map, known_missing_types)

#     def game_start(self) -> None:
#         self.matching_argument_types_count_by_section = {section: 0 for section in PREDICATE_SECTIONS}

#     def _count(self, context: ContextDict):
#         self.matching_argument_types_count_by_section[context[SECTION_CONTEXT_KEY]] += 1  # type: ignore

#     def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
#         return {k.replace('(:', ''): v for k, v in self.matching_argument_types_count_by_section.items()}


def build_argument_types_fitness_terms(
    predicates: typing.Sequence[str] = COMMON_SENSE_PREDICATES_FUNCTIONS,
    # type_categories: typing.Sequence[str] = COMMON_SENSE_TYPE_CATEGORIES,
    # predicate_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]] = PREDICATE_FUNCTION_ARITY_MAP
    ) -> typing.Sequence[FitnessTerm]:
    fitness_terms = []

    # sorted_type_categories = list(sorted(type_categories))

    for predicate in predicates:
        fitness_terms.append(PredicateFunctionArgumentTypes(predicate))
        # for type_combinations in itertools.product(*([sorted_type_categories] * predicate_arity_map[predicate])):  # type: ignore
            # fitness_terms.append(PredicateFunctionArgumentTypesBySection(predicate, type_combinations, predicate_arity_map))
            # fitness_terms.append(PredicateFunctionArgumentTypes(predicate, type_combinations, predicate_arity_map))

    return fitness_terms


COMPOSITIONALITY_STRUCTURES = (
    ['(hold (and (not (agent_holds ?x) ) (in_motion ?x) ) )', '(hold (and (in_motion ?x) (not (agent_holds ?x) ) ) )'],
    ['(once (and (not (in_motion ?x) ) (in ?x ?x) ) )', '(once (and (in ?x ?x) (not (in_motion ?x) ) ) )'],
    ['(once (agent_holds ?x) )',],
    ['(once (not (in_motion ?x) ) )',],
    ['(once (and (agent_holds ?x) (adjacent ?x ?x) ) )', '(once (and (adjacent ?x ?x) (agent_holds ?x) ) )',],
    ['(hold-while (and (not (agent_holds ?x) ) (in_motion ?x) ) (touch ?x ?x) )',  '(hold-while (and (in_motion ?x) (not (agent_holds ?x) ) ) (touch ?x ?x) )'],
    ['(once (and (not (in_motion ?x) ) (on ?x ?x) ) )', '(once (and (on ?x ?x) (not (in_motion ?x) ) ) )'],
    # ['(hold-while (and (in_motion ?x) (not (agent_holds ?x) ) ) (touch ?x ?x) )',  '(hold-while (and (not (agent_holds ?x) ) (in_motion ?x) ) (touch ?x ?x) )'],  # removed because equivalent to another after consider both orders
    # ['(once (and (adjacent ?x ?x) (agent_holds ?x) ) )',  '(once (and (agent_holds ?x) (adjacent ?x ?x) ) )'],  # removed because equivalent to another after consider both orders
    # '(hold-while (and (not (agent_holds ?x) ) (in ?x ?x) (or (agent_holds ?x) (in_motion ?x) ) ) (touch ?x ?x) )',  # removed because rare and conjunction of three + disjunction of two is pain to enumerate
    ['(once (and (agent_holds ?x) (on ?x ?x) ) )', '(once (and (on ?x ?x) (agent_holds ?x) ) )'],
    ['(hold-while (and (in_motion ?x) (not (agent_holds ?x) ) ) (touch ?x ?x) (in_motion ?x) )',  '(hold-while (and (not (agent_holds ?x) ) (in_motion ?x) ) (touch ?x ?x) (in_motion ?x) )',],
    ['(hold-while (and (not (agent_holds ?x) ) (in_motion ?x) ) (on ?x ?x) )',  '(hold-while (and (in_motion ?x) (not (agent_holds ?x) ) ) (on ?x ?x) )',],

 )


COMPOSITIONALITY_STRUCTURES_TO_INDEX = {s: i for i, structures in enumerate(COMPOSITIONALITY_STRUCTURES) for s in structures}


PREDICATE_ARGS_PATTERN = r'\(\s*(?:[\w-]+)\s+((?:\??\w+\s*)+)\)'
COMPOSITIONALITY_VARIABLE_REPLACEMENT = '?x'


class CompositionalityStructureCounter(FitnessTerm):
    structure_counts: typing.Dict[int, int]
    structure_to_index: typing.Dict[str, int]
    args_pattern: re.Pattern
    variable_replacement: str

    def __init__(self, structure_to_index: typing.Dict[str, int] = COMPOSITIONALITY_STRUCTURES_TO_INDEX,
        variable_replacement: str = COMPOSITIONALITY_VARIABLE_REPLACEMENT,
        predicate_arg_pattern: str = PREDICATE_ARGS_PATTERN):

        super().__init__(MODALS, f'compositionality_structure')
        self.structure_to_index = structure_to_index
        self.variable_replacement = variable_replacement
        self.args_pattern = re.compile(predicate_arg_pattern)
        self.min_structure_index = min(self.structure_to_index.values())
        self.max_structure_index = max(self.structure_to_index.values())

    def game_start(self) -> None:
        self.structure_counts = {i: 0 for i in range(self.min_structure_index, self.max_structure_index + 1)}

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            ast_str = ast_printer.ast_section_to_string(ast, ast_parser.PREFERENCES)
            for args in self.args_pattern.findall(ast_str):
                ast_str = ast_str.replace(args, ' '.join(map(lambda x: self.variable_replacement, args.split(" "))), 1)

            if ast_str in self.structure_to_index:
                self.structure_counts[self.structure_to_index[ast_str]] += 1

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return {str(i): c for i, c in self.structure_counts.items()}

    def _get_all_inner_keys(self):
        return [str(i) for i in range(self.min_structure_index, self.max_structure_index + 1)]


class SectionCountTerm(FitnessTerm):
    def __init__(self, section: str, header: str, thresholds: typing.Optional[typing.Sequence[float]],
                 right: bool = True):
        super().__init__(section, header.replace('(:', ''))
        if thresholds is not None:
            thresholds = list(thresholds)

        self.thresholds = thresholds
        self.right = right

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        result = self._inner_game_end()
        if self.thresholds is None:
            return result

        output = {i: 0 for i in range(len(self.thresholds) + 1)}
        output[np.digitize(result, self.thresholds, self.right)] = 1
        return output  # type: ignore

    @abstractmethod
    def _inner_game_end(self) -> Number:
        pass

    def _get_all_inner_keys(self):
        return range(len(self.thresholds) + 1) if self.thresholds is not None else None


class SectionMaxDepth(SectionCountTerm):
    max_depth: int = 0

    def __init__(self, section: str, thresholds: typing.Optional[typing.Sequence[float]] = None):
        super().__init__(section, f'max_depth_{section}', thresholds)
        self.max_depth = 0

    def game_start(self) -> None:
        self.max_depth = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if DEPTH_CONTEXT_KEY in context:
            self.max_depth = max(self.max_depth, context[DEPTH_CONTEXT_KEY])  # type: ignore

    def _inner_game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return self.max_depth


class SectionMeanDepth(SectionCountTerm):
    depths: typing.List[int] = []

    def __init__(self, section: str, thresholds: typing.Optional[typing.Sequence[float]] = None):
        super().__init__(section, f'mean_depth_{section}', thresholds)
        self.depths = []

    def game_start(self) -> None:
        self.depths = []

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if DEPTH_CONTEXT_KEY in context:
            self.depths.append(context[DEPTH_CONTEXT_KEY])  # type: ignore

    def _inner_game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if len(self.depths) == 0:
            return 0

        return sum(self.depths) / len(self.depths)


class SectionNodeCount(SectionCountTerm):
    node_count: int = 0

    def __init__(self, section: str, thresholds: typing.Optional[typing.Sequence[float]] = None):
        super().__init__(section, f'node_count_{section}', thresholds)
        self.node_count = 0

    def game_start(self) -> None:
        self.node_count = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        self.node_count += 1

    def _inner_game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return self.node_count


class SectionMaxWidth(SectionCountTerm):
    max_width: int = 0

    def __init__(self, section: str, thresholds: typing.Optional[typing.Sequence[float]] = None):
        super().__init__(section, f'max_width_{section}', thresholds)
        self.max_width = 0

    def game_start(self) -> None:
        self.max_width = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            for child_key in ast:
                if isinstance(ast[child_key], list):
                    self.max_width = max(self.max_width, len(ast[child_key]))  # type: ignore

    def _inner_game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return self.max_width


SECTION_COUNT_THRESHOLDS = {
    SectionMaxDepth: {
        ast_parser.SETUP: [0.0, 12.0, 13.0, 26.0],
        ast_parser.PREFERENCES: [9.0, 15.0, 17.0, 23.0],
        ast_parser.TERMINAL: [0.0, 0.6, 9.0, 12.0],  # [0.0, 0.0, 7.0, 12.0]
        ast_parser.SCORING: [3.0, 7.5, 10.0, 17.0],  # [3.0, 3.0, 10.0, 17.0]
    },
    SectionMeanDepth: {
        ast_parser.SETUP: [0.0, 6.3, 7.5, 14.7],
        ast_parser.PREFERENCES: [5.1, 9.2, 10.3, 14.0],
        ast_parser.TERMINAL: [0.0, 1.6, 3.3, 6.0],
        ast_parser.SCORING: [1.5, 4.9, 5.6, 8.4],  # [1.5, 1.5, 5.2, 8.4]
    },
    SectionNodeCount: {
        ast_parser.SETUP: [0.0, 19.0, 36.5, 131.0],
        ast_parser.PREFERENCES: [14.0, 64.0, 110.0, 565.0],
        ast_parser.TERMINAL: [0.0, 9.0, 11.0, 50.0],  # [0.0, 0.0, 10.0, 50.0]
        ast_parser.SCORING: [4.0, 16.0, 32.0, 134.0],  # [4.0, 4.0, 28.0, 134.0]
    },
    SectionMaxWidth: {
        ast_parser.SETUP: [0.0, 1.0, 3.0, 10.0],
        ast_parser.PREFERENCES: [1.0, 3.0, 4.0, 10],  # [1.0, 3.0, 3.0, 10.0]
        ast_parser.TERMINAL: [0.0, 2.0, 3.0, 4.0],  # [0.0, 0.0, 0.0, 4.0]
        ast_parser.SCORING: [0.0, 2.0, 3.0, 12.0],
    }
}


def build_section_count_fitness_terms(sections: typing.Sequence[str] = ast_parser.SECTION_KEYS,
    term_classes: typing.Sequence[typing.Callable] = (SectionMaxDepth, SectionMeanDepth, SectionNodeCount, SectionMaxWidth),
    thresholds: typing.Optional[typing.Dict[typing.Callable, typing.Dict[str, typing.Sequence[float]]]] = SECTION_COUNT_THRESHOLDS,
    ) -> typing.Sequence[FitnessTerm]:
    if thresholds is None:
        thresholds = {}

    return [term_class(section, thresholds[term_class][section] if term_class in thresholds else None) for term_class in term_classes for section in sections]

SECTION_EXISTS_SECTIONS = (ast_parser.SETUP, ast_parser.TERMINAL)


class SectionExistsFitnessTerm(FitnessTerm):
    relevant_sections: typing.Sequence[str]
    sections_found: typing.Dict[str, bool] = {}

    def __init__(self, relevant_sections: typing.Sequence[str] = SECTION_EXISTS_SECTIONS):
        super().__init__(relevant_sections, 'section_doesnt_exist')
        self.relevant_sections = relevant_sections
        self.sections_found = {section: False for section in self.relevant_sections}

    def game_start(self) -> None:
        self.sections_found = {section: False for section in self.relevant_sections}

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            section = typing.cast(str, context[SECTION_CONTEXT_KEY])
            if section in self.relevant_sections:
                self.sections_found[section] = True

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return {f'{section.replace("(:", "")}': not found for section, found in self.sections_found.items()}

    def _get_all_inner_keys(self):
        return [f'{section.replace("(:", "")}' for section in self.relevant_sections]


# TEXT_N_GRAM_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/text_7_ngram_model_2023_02_16.pkl')
# class TextNGramTerm(FitnessTerm):
#     game_output: typing.Optional[dict] = None
#     n_gram_model: NGramTrieModel
#     n_gram_model_path: str
#     stupid_backoff: bool
#     top_k_max_n: typing.Optional[int]
#     top_k_min_n: typing.Optional[int]
#     top_k_ngrams: int

#     def __init__(self, top_k_ngrams: int = DEFAULT_TOP_K_NGRAMS,
#                  stupid_backoff: bool = True, log: bool = True,
#                  filter_padding_top_k: bool = False, top_k_min_n: typing.Optional[int] = None,
#                  top_k_max_n: typing.Optional[int] = None, score_all: bool = False,
#                  n_gram_model_path: str = TEXT_N_GRAM_MODEL_PATH):
#         super().__init__('', 'text_ngram')
#         self.top_k_ngrams = top_k_ngrams
#         self.stupid_backoff = stupid_backoff
#         self.log = log
#         self.filter_padding_top_k = filter_padding_top_k
#         self.top_k_min_n = top_k_min_n
#         self.top_k_max_n = top_k_max_n
#         self.score_all = score_all
#         self.n_gram_model_path = n_gram_model_path

#         with open(self.n_gram_model_path, 'rb') as f:
#             self.n_gram_model = pickle.load(f)

#     def game_start(self) -> None:
#         self.game_output = None

#     def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
#         pass

#     def parse_full_text(self, full_text: str) -> None:
#         self.game_output = self.n_gram_model.score(
#             full_text, k=self.top_k_ngrams, stupid_backoff=self.stupid_backoff,  # type: ignore
#             log=self.log, filter_padding_top_k=self.filter_padding_top_k,
#             top_k_min_n=self.top_k_min_n, top_k_max_n=self.top_k_max_n,
#             score_all=self.score_all
#         )

#     def game_end(self):
#         return self.game_output



DEFAULT_TOP_K_NGRAMS = 10
DEFAULT_TOP_K_NGRAMS_FOR_SECTIONS = 5
DEFAULT_TOP_K_MIN_N = 2
DEFAULT_TOP_K_MAX_N = 7


class ASTNGramTerm(FitnessTerm):
    filter_padding_top_k: bool
    game_output: typing.Optional[dict] = None
    log: bool
    model_n: int
    n_by_section: typing.Dict[str, int]
    n_gram_model: ASTNGramTrieModel
    n_gram_model_path: str
    stupid_backoff: bool
    top_k_max_n: typing.Optional[int]
    top_k_min_n: typing.Optional[int]
    top_k_ngrams: typing.Optional[int]
    top_k_ngrams_for_sections: typing.Optional[int]

    def __init__(self, top_k_ngrams: typing.Optional[int] = DEFAULT_TOP_K_NGRAMS,
                 stupid_backoff: bool = True, log: bool = True,
                 filter_padding_top_k: bool = False, top_k_min_n: typing.Optional[int] = DEFAULT_TOP_K_MIN_N,
                 top_k_max_n: typing.Optional[int] = DEFAULT_TOP_K_MAX_N, score_all: bool = False,
                 top_k_ngrams_for_sections: typing.Optional[int] = DEFAULT_TOP_K_NGRAMS_FOR_SECTIONS,
                 n_by_section: typing.Dict[str, int] = DEFAULT_N_BY_SECTION,
                 n_gram_model_path: str = LATEST_AST_N_GRAM_MODEL_PATH):
        super().__init__('', 'ast_ngram')
        self.top_k_ngrams = top_k_ngrams
        self.stupid_backoff = stupid_backoff
        self.log = log
        self.filter_padding_top_k = filter_padding_top_k
        self.top_k_min_n = top_k_min_n
        self.top_k_max_n = top_k_max_n
        self.score_all = score_all
        self.top_k_ngrams_for_sections = top_k_ngrams_for_sections
        self.n_by_section = n_by_section
        self.n_gram_model_path = n_gram_model_path

        with open(self.n_gram_model_path, 'rb') as f:
            self.n_gram_model = pickle.load(f)

        path_split = os.path.basename(self.n_gram_model_path).split('_')
        path_digits = [s for s in path_split if s.isdigit()]
        if len(path_digits) == 0:
            raise ValueError(f'Could not find model_n in {self.n_gram_model_path}')
        self.model_n = int(path_digits[0])

    def game_start(self) -> None:
        self.game_output = None

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        self.game_output = self.n_gram_model.score(
            ast, k=self.top_k_ngrams, stupid_backoff=self.stupid_backoff,  # type: ignore
            log=self.log, filter_padding_top_k=self.filter_padding_top_k,
            top_k_min_n=self.top_k_min_n, top_k_max_n=self.top_k_max_n,
            score_all=self.score_all, k_for_sections=self.top_k_ngrams_for_sections,
        )

    def game_end(self):
        return self.game_output

    def _get_all_inner_keys(self):
        inner_keys = []
        n_values = []
        min_n, max_n = self.top_k_min_n, self.top_k_max_n

        if min_n is None and max_n is None:
            n_values = [self.model_n]

        else:
            if min_n is None:
                min_n = 2

            if max_n is None:
                max_n = self.model_n

            n_values = list(range(min_n, max_n + 1))

        if self.score_all:
            inner_keys.extend([f'full_n_{n}_score' for n in n_values])
            if self.top_k_ngrams is not None:
                inner_keys.extend([f'full_n_{n}_{k}' for n in n_values for k in range(self.top_k_ngrams)])

        else:
            inner_keys.append(f'full_n_{self.model_n}_score')
            if self.top_k_ngrams is not None:
                inner_keys.extend([f'full_n_{self.model_n}_{k}' for k in range(self.top_k_ngrams)])

        for section, section_max_n in self.n_by_section.items():
            section = section.replace('(:', '')
            if min_n is None and max_n is None:
                section_n_values = [section_max_n]
            else:
                section_n_values = list(range(min_n, min(max_n, section_max_n) + 1))  # type: ignore

            if self.score_all:
                inner_keys.extend([f'{section}_n_{n}_score' for n in section_n_values])
                if self.top_k_ngrams_for_sections is not None:
                    inner_keys.extend([f'{section}_n_{n}_{k}' for n in section_n_values for k in range(self.top_k_ngrams_for_sections)])

            else:
                inner_keys.append(f'{section}_n_{self.model_n}_score')
                if self.top_k_ngrams_for_sections is not None:
                    inner_keys.extend([f'{section}_n_{self.model_n}_{k}' for k in range(self.top_k_ngrams_for_sections)])

        return inner_keys



def build_fitness_featurizer(args) -> ASTFitnessFeaturizer:
    preprocessors = []

    if not args.no_binarize:
        preprocessors.append(BinarizeFitnessFeatures())

    if not args.no_merge:
        preprocessors.append(MergeFitnessFeatures(COMMON_SENSE_PREDICATES_FUNCTIONS))

    fitness = ASTFitnessFeaturizer(preprocessors=preprocessors)

    all_variables_defined = AllVariablesDefined()
    fitness.register(all_variables_defined)

    all_variables_used = AllVariablesUsed()
    fitness.register(all_variables_used)

    all_preferences_used = AllPreferencesUsed()
    fitness.register(all_preferences_used)

    num_preferences_defined = NumPreferencesDefined()
    fitness.register(num_preferences_defined)

    all_setup_objects_used = SetupObjectsUsed()
    fitness.register(all_setup_objects_used)

    setup_quantified_objects_used = SetupQuantifiedObjectsUsed()
    fitness.register(setup_quantified_objects_used)

    no_adjacent_once = NoAdjacentOnce()
    fitness.register(no_adjacent_once)

    no_adjacent_same_modal = NoAdjacentSameModal()
    fitness.register(no_adjacent_same_modal)

    pref_starts_and_ends_with_once = PrefStartsAndEndsWithOnce()
    fitness.register(pref_starts_and_ends_with_once)

    once_in_middle_of_pref = OnceInMiddleOfPref()
    fitness.register(once_in_middle_of_pref)

    pref_without_hold = PrefWithoutHold()
    fitness.register(pref_without_hold)

    length_of_then_modals = LengthOfThenModals()
    fitness.register(length_of_then_modals)

    at_end_found = AtEndFound()
    fitness.register(at_end_found)

    max_quantification_count = MaxQuantificationCount()
    fitness.register(max_quantification_count)

    max_number_variables_types = MaxNumberVariablesTypesQuantified()
    fitness.register(max_number_variables_types)

    no_repeated_variables_in_predicate = NoVariablesRepeated()
    fitness.register(no_repeated_variables_in_predicate)

    no_nested_logicals = NoNestedLogicals()
    fitness.register(no_nested_logicals)

    no_identical_logical_children = NoIdenticalChildrenInLogicals()
    fitness.register(no_identical_logical_children)

    no_identical_scoring_expression_children = NoIdenticalChildrenInScoringExpressions()
    fitness.register(no_identical_scoring_expression_children)

    scoring_expression_repetitions = ScoringCountExpressionRepetitions()
    fitness.register(scoring_expression_repetitions)

    tautological_boolean_expression = TautologicalBooleanExpression()
    fitness.register(tautological_boolean_expression)

    redundant_boolean_expression = RedundantBooleanExpression()
    fitness.register(redundant_boolean_expression)

    redundant_boolean_scoring_terminal_expression = RedundantBooleanScoringTerminalExpression()
    fitness.register(redundant_boolean_scoring_terminal_expression)

    identical_consecutive_predicates = IdenticalConsecutiveSeqFuncPredicates()
    fitness.register(identical_consecutive_predicates)

    disjoint_seq_funcs = DisjointSeqFuncPredicateTerm()
    fitness.register(disjoint_seq_funcs)

    count_once_per_external_objects_used = CountOncePerExternalObjectsUsedCorrectly()
    fitness.register(count_once_per_external_objects_used)

    external_forall_used = ExternalForallUsedCorrectly()
    fitness.register(external_forall_used)

    pref_forall_used = PrefForallUsed()
    fitness.register(pref_forall_used)

    pref_forall_correct_arity = PrefForallCorrectArity()
    fitness.register(pref_forall_correct_arity)

    pref_forall_correct_types = PrefForallCorrectTypes()
    fitness.register(pref_forall_correct_types)

    # Changed in the grammar to enforce correct arity
    # correct_predicate_arity = CorrectPredicateFunctionArity()
    # fitness.register(correct_predicate_arity)

    no_two_number_operations = NoTwoNumberOperations()
    fitness.register(no_two_number_operations)

    no_single_argument_multi_operations = NoSingleArgumentMultiOperations()
    fitness.register(no_single_argument_multi_operations)

    no_count_in_terminal = SectionWithoutPrefOrTotalCounts(ast_parser.TERMINAL)
    fitness.register(no_count_in_terminal, section_rule=True)

    no_count_in_scoring = SectionWithoutPrefOrTotalCounts(ast_parser.SCORING)
    fitness.register(no_count_in_scoring, section_rule=True)

    predicate_under_modal = PredicateUnderModal(MODALS, COMMON_SENSE_PREDICATES_FUNCTIONS)
    fitness.register(predicate_under_modal)

    argument_types_fitness_terms = build_argument_types_fitness_terms()
    fitness.register_multiple(argument_types_fitness_terms)

    compositionality_fitness_term = CompositionalityStructureCounter()
    fitness.register(compositionality_fitness_term)

    section_count_fitness_terms = build_section_count_fitness_terms()
    fitness.register_multiple(section_count_fitness_terms, section_rule=True)

    section_exists_fitness_term = SectionExistsFitnessTerm()
    fitness.register(section_exists_fitness_term, section_rule=True)

    # text_ngram_term = TextNGramTerm(top_k_min_n=2, score_all=True)
    # fitness.register(text_ngram_term, full_text_rule=True)

    if args.use_specific_objects_ngram_model:
        ngram_path = LATEST_SPECIFIC_OBJECTS_AST_N_GRAM_MODEL_PATH
    else:
        ngram_path = LATEST_AST_N_GRAM_MODEL_PATH

    ast_ngram_term = ASTNGramTerm(top_k_min_n=2, score_all=True, top_k_ngrams=None,
                                  top_k_ngrams_for_sections=None, n_gram_model_path=ngram_path)
    fitness.register(ast_ngram_term, full_ast_rule=True)

    return fitness


def parse_single_game(game_and_src_file: typing.Tuple[tuple, str]) -> None:
    process_index = multiprocessing.current_process()._identity[0] - 1 % args.n_workers
    row = featurizers[process_index].parse(*game_and_src_file, return_row=True, preprocess_row=False)  # type: ignore
    temp_csv_writers[process_index].writerow(row)


def game_iterator():
    for src_file in args.test_files:
        for game in cached_load_and_parse_games_from_file(src_file, grammar_parser, False, log_every_change=False):  # type: ignore
            yield game, src_file


def get_headers(args: argparse.Namespace, featurizer: ASTFitnessFeaturizer) -> typing.List[str]:
    src_file = args.test_files[0]
    game = next(cached_load_and_parse_games_from_file(src_file, grammar_parser, False))  # type: ignore
    featurizer.parse(game, src_file, return_row=True, preprocess_row=False)  # type: ignore
    return featurizer.get_all_column_keys()


def extract_game_index(game_name: str):
    first_dash = game_name.find('-')
    second_dash = game_name.find('-', first_dash + 1)
    index = game_name[first_dash + 1:second_dash] if second_dash != -1 else game_name[first_dash + 1:]
    return int(index)


def extract_negative_index(game_name: str):
    first_dash = game_name.find('-')
    second_dash = game_name.find('-', first_dash + 1)
    if second_dash == -1:
        return -1

    third_dash = game_name.find('-', second_dash + 1)
    index = game_name[second_dash + 1:third_dash]
    return int(index)


def build_or_load_featurizer(args: argparse.Namespace) -> ASTFitnessFeaturizer:
    if args.existing_featurizer_path is not None:
        logger.info(f'Loading featurizer from {args.existing_featurizer_path}')
        with gzip.open(args.existing_featurizer_path, 'rb') as f:
            featurizer = pickle.load(f)  # type: ignore

    else:
        featurizer = build_fitness_featurizer(args)

    return featurizer

if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)

    for test_file in args.test_files:
        if not os.path.exists(test_file):
            raise ValueError(f'File {test_file} does not exist')

    if not args.output_path.endswith('.gz'):
        args.output_path += '.gz'

    if args.featurizer_output_path is None:
        args.featurizer_output_path = DEFAULT_FEATURIZER_OUTPUT_PATH_PATTERN.format(today=datetime.now().strftime('%Y_%m_%d'))

    args_str = '\n'.join([f'{" " * 26}{k}: {v}' for k, v in vars(args).items()])
    logger.debug(f'Shell arguments:\n{args_str}')

    original_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(args.recursion_limit)

    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    if args.n_workers > 1:
        featurizers = [build_or_load_featurizer(args) for _ in range(args.n_workers)]
        headers = get_headers(args, featurizers[0])
        # headers = featurizers[0].get_all_column_keys()

        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR, exist_ok=True)

        # rows = []
        for file in glob.glob(os.path.join(TEMP_DIR, '*.temp.csv')):
            os.remove(file)

        temp_output_paths = [os.path.join(TEMP_DIR, os.path.basename(args.output_path) + f'_{i}.temp.csv') for i in range(args.n_workers)]
        temp_files = [open(temp_output_path, 'w', newline='') for temp_output_path in temp_output_paths]
        temp_csv_writers = [csv.DictWriter(temp_file, headers) for temp_file in temp_files]

        # TODO: consider rewriting with asyncio instead of multiprocessing

        logger.info(f'About to start pool with {args.n_workers} workers')
        with multiprocessing.Pool(args.n_workers) as p:
            logger.info('Pool started')
            for row in tqdm(p.imap_unordered(parse_single_game, game_iterator(), chunksize=args.chunksize)):  # type: ignore
                continue
                # if headers is None:
                #     headers = list(row.keys())
                # rows.append(row)

        for temp_file in temp_files:
            temp_file.close()

        featurizer = featurizers[0]
        # featurizer.rows = rows
        logger.info('About to parse rows from temp files into dataframe')
        rows_dfs = [pd.read_csv(temp_output_path, header=None, names=headers) for temp_output_path in temp_output_paths]
        rows_df = pd.concat(rows_dfs, sort=False)

        rows_df = rows_df.assign(real=(rows_df.src_file == 'interactive-beta.pddl').astype(int))
        rows_df = rows_df.assign(game_index=rows_df.game_name.apply(extract_game_index),
                                 negative_index= rows_df.game_name.apply(extract_negative_index), fake=~rows_df.real)
        rows_df = rows_df.sort_values(by=['fake', 'game_index', 'negative_index'], ignore_index=True).reset_index(drop=True)
        rows_df.drop(columns=['fake', 'game_index', 'negative_index'], inplace=True)

        featurizer.rows_df = rows_df.reset_index(drop=True)
        # for file in glob.glob(os.path.join(TEMP_DIR, '*.temp.csv')):
        #     os.remove(file)

    else:
        featurizer = build_or_load_featurizer(args)
        for test_file in args.test_files:
            for ast in cached_load_and_parse_games_from_file(test_file, grammar_parser, not args.dont_tqdm):  # type: ignore
                featurizer.parse(ast, test_file)  # type: ignore

    logger.info('Done parsing games, about to convert to dataframe')
    df = featurizer.to_df(use_prior_values=args.existing_featurizer_path is not None)

    logger.debug(df.groupby('src_file').agg([np.mean, np.std]))

    for src_file in df.src_file.unique():
        zero_std = df[df.src_file == src_file].std(numeric_only=True) == 0
        zero_std_columns = [c for c in zero_std.index if zero_std[c] and not 'arg_types' in c]  # type: ignore
        logger.debug(f'For src_file {src_file}, the following columns have zero std (excluding arg_types columns): {zero_std_columns}')

    global_zero_std = df.std(numeric_only=True) == 0
    global_zero_std_columns = [c for c in global_zero_std.index if global_zero_std[c] and not 'arg_types' in c]  # type: ignore
    logger.debug(f'For all src_files, the following columns have zero std (excluding arg_types columns): {global_zero_std_columns}')


    zero_mean_features = []
    positive_mean_features = []
    one_mean_features = []

    for feature in df.columns:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            continue

        if any(x in feature for x in ('arg_types', 'predicate_under_modal', 'max_number', 'max_quantification',
                                      'compositionality_structure', 'depth', 'node_count', 'ast_ngram')):
            continue

        real_game_feature_mean = df.loc[df.real == True, feature].mean()
        if real_game_feature_mean == 0:
            zero_mean_features.append(feature)
        elif real_game_feature_mean == 1:
            one_mean_features.append(feature)
        else:
            positive_mean_features.append(feature)

    zero_mean_features_str = '\n'.join([f'    - {feature}' for feature in zero_mean_features])
    logger.debug(f'The following features have a mean of zero over the real games:\n{zero_mean_features_str}\n')

    positive_mean_features_str = '\n'.join([f'    - {feature}' for feature in positive_mean_features])
    logger.debug(f'The following features have a positive mean over the real games:\n{positive_mean_features_str}\n')

    one_mean_features_str = '\n'.join([f'    - {feature}' for feature in one_mean_features])
    logger.debug(f'The following features have a mean of 1 over the real games:\n{one_mean_features_str}\n')

    logger.info(f'Writing to {args.output_path}')
    df.to_csv(args.output_path, index_label='Index', compression='gzip')

    if args.existing_featurizer_path is None and args.featurizer_output_path is not None:
        with gzip.open(args.featurizer_output_path, 'wb') as f:
            pickle.dump(featurizer, f)  # type: ignore

    sys.setrecursionlimit(original_recursion_limit)
