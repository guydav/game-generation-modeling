from abc import ABC, abstractmethod
import argparse
from datetime import datetime
from collections import namedtuple, defaultdict
import itertools
import gzip
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

from ast_to_latex_doc import TYPE_RULES, extract_n_args, extract_predicate_function_args, extract_predicate_function_name
from ast_parser import VariableDefinition, extract_variables_from_ast, update_context_variables, predicate_function_term_to_type_category
import ast_parser
import ast_printer
from ast_utils import cached_load_and_parse_games_from_file
from fitness_features_preprocessing import FitnessFeaturesPreprocessor, DEFAULT_MERGE_THRESHOLD, BinarizeFitnessFeatures, MergeFitnessFeatures
from fitness_ngram_models import NGramTrieNode, NGramTrieModel, ASTNGramTrieModel, NGramASTParser
import room_and_object_types


parser = argparse.ArgumentParser()
DEFAULT_GRAMMAR_FILE = './dsl/dsl.ebnf'
parser.add_argument('-g', '--grammar-file', default=DEFAULT_GRAMMAR_FILE)
DEFAULT_TEST_FILES = (
    './dsl/interactive-beta.pddl',
    './dsl/ast-real-regrowth-samples.pddl',
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
DEFAULT_OUTPUT_PATH ='./data/fitness_scores.csv.gz'
parser.add_argument('-o', '--output-path', default=DEFAULT_OUTPUT_PATH)
DEFAULT_FEATURIZER_OUTPUT_PATH_PATTERN = './models/fitness_featurizer_{today}.pkl.gz'
parser.add_argument('-f', '--featurizer-output-path', default=None)
DEFAULT_RECURSION_LIMIT = 2000
parser.add_argument('--recursion-limit', type=int, default=DEFAULT_RECURSION_LIMIT)
parser.add_argument('--no-binarize', action='store_true')
parser.add_argument('--no-merge', action='store_true')
parser.add_argument('--merge-threshold', type=float, default=DEFAULT_MERGE_THRESHOLD)


ContextDict = typing.Dict[str, typing.Union[str, int, VariableDefinition]]
Number = typing.Union[int, float]


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


DEFAULT_HEADERS = ('src_file', 'game_name', 'domain_name')

VARIABLES_CONTEXT_KEY = 'variables'
SECTION_CONTEXT_KEY = 'section'
DEPTH_CONTEXT_KEY = 'depth'
EXTERNAL_FORALL_CONTEXT_KEY = 'external_forall'

COUNT_RULE_PATTERN = re.compile('count.*')


class ASTFitnessFeaturizer:
    full_text_registry: typing.List[FitnessTerm]
    headers: typing.List[str]
    header_registry: typing.Dict[str, FitnessTerm]
    list_reduce: typing.Callable[[typing.Sequence[Number]], Number]
    preprocessors: typing.Optional[typing.Iterable[FitnessFeaturesPreprocessor]]
    regex_rules: typing.List[typing.Tuple[re.Pattern, FitnessTerm]]
    rows: typing.List
    rule_registry: typing.Dict[str, typing.List[FitnessTerm]]
    section_keys: typing.List[str]
    section_registry: typing.Dict[str, typing.List[FitnessTerm]]
    tuple_registry: typing.Dict[str, typing.List[FitnessTerm]]


    def __init__(self, preprocessors: typing.Optional[typing.Iterable[FitnessFeaturesPreprocessor]] = None,
        headers: typing.Sequence[str] = DEFAULT_HEADERS,
        list_reduce: typing.Callable[[typing.Sequence[Number]], Number] = np.sum,
        section_keys: typing.Sequence[str] = ast_parser.SECTION_KEYS):

        self.preprocessors = preprocessors
        self.headers = list(headers)
        self.list_reduce = list_reduce
        self.section_keys = list(section_keys)

        self.rule_registry = defaultdict(list)
        self.tuple_registry = defaultdict(list)
        self.section_registry = defaultdict(list)
        self.full_ast_registry = []
        self.full_text_registry = []
        self.regex_rules = []
        self.header_registry = dict()

        self.rows = []

    def __getstate__(self) -> typing.Dict[str, typing.Any]:
        # Prevents the rows from being dumped to file when this is pickled
        state = self.__dict__.copy()
        del state['rows']
        return state

    def __setstate__(self, state: typing.Dict[str, typing.Any]) -> None:
        self.__dict__.update(state)
        self.rows = []

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
                section = typing.cast(str, term.rules[0])
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

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame.from_records(self.rows, columns=list(self.rows[0].keys()))
        if self.preprocessors is not None:
            for preprocessor in self.preprocessors:
                df = preprocessor.preprocess_df(df)

        return df

    def parse(self, full_ast: typing.Tuple[tatsu.ast.AST, tatsu.ast.AST, tatsu.ast.AST, tatsu.ast.AST], src_file: str, return_row: bool = False):
        row = {}
        row['src_file'] = os.path.basename(src_file)
        row['game_name'] = full_ast[1]["game_name"]  # type: ignore
        row['domain_name'] = full_ast[2]["domain_name"]  # type: ignore
        ast = full_ast[3:]  # type: ignore

        for term in self.header_registry.values():
            term.game_start()

        self._parse(ast)

        for term in self.full_ast_registry:
            term.update(full_ast, 'full_ast', {})

        ast_full_text = ast_printer.ast_to_string(full_ast, ' ')  # type: ignore
        for term in self.full_text_registry:
            term.parse_full_text(ast_full_text)

        for header, term in self.header_registry.items():
            term_result = term.game_end()
            if isinstance(term_result, (int, float)):
                row[header] = term_result
            elif isinstance(term_result, dict):
                for key, val in term_result.items():
                    row[f'{header}_{key}'] = val
            else:
                row[header] = self.list_reduce(row[header])


        if return_row:
            if self.preprocessors is not None:
                for preprocessor in self.preprocessors:
                    row = preprocessor.preprocess_row(row)

            return row

        else:
            self.rows.append(row)

    def _parse(self, ast: typing.Union[str, int, tatsu.buffering.Buffer, tuple, list, tatsu.ast.AST],
        context: typing.Optional[ContextDict] = None):
        if context is None:
            context = {DEPTH_CONTEXT_KEY: 0}

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
            child_context = context.copy()
            child_context[DEPTH_CONTEXT_KEY] += 1  # type: ignore

            if ast.parseinfo.rule in ('scoring_external_maximize', 'scoring_external_minimize'):
                child_context[EXTERNAL_FORALL_CONTEXT_KEY] = ast.parseinfo

            for child_key in ast:
                if child_key != 'parseinfo':
                    self._parse(ast[child_key], child_context)  # type: ignore

        else:
            print(f'Encountered AST element with unrecognized type: {ast} of type {type(ast)}')


class ASTNodeCounter(ast_parser.ASTParser):
    def __init__(self):
        self.count = 0

    def __call__(self, ast, **kwargs):
        if 'zero_count' in kwargs:
            self.count = 0
            del kwargs['zero_count']
        super().__call__(ast, **kwargs)
        return self.count

    def _handle_ast(self, ast, **kwargs):
        self.count += 1
        super()._handle_ast(ast, **kwargs)


class VariableBasedFitnessTerm(FitnessTerm):
    def __init__(self, header: str):
        super().__init__(('setup_statement', 'predicate', 'function', 'predicate_term', 'function_term', 'predicate_or_function_term'), header)
        self.variables = set()

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if VARIABLES_CONTEXT_KEY not in context:
            return

        if 'term' in ast:
            if isinstance(ast.term, str) and ast.term.startswith('?'):  # type: ignore
                self._inner_update(ast.term, context[VARIABLES_CONTEXT_KEY])  # type: ignore

        else:
            self._inner_update(None, context[VARIABLES_CONTEXT_KEY])  # type: ignore

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

    def _inner_update(self, term: str, variables: typing.Dict[str, VariableDefinition]):
        if term is None:
            return
        elif term in variables:
            self.defined_count += 1
        else:
            self.undefined_count += 1

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if self.defined_count == 0:
            return 0

        return self.defined_count / (self.defined_count + self.undefined_count)


class AllVariablesUsed(VariableBasedFitnessTerm):
    defined_variables: typing.Set[typing.Tuple[str, str, int]]
    used_variables: typing.Set[typing.Tuple[str, str, int]]

    def __init__(self):
        super().__init__('all_variables_used')

    def game_start(self) -> None:
        self.defined_variables = set()
        self.used_variables = set()

    def _inner_update(self, term: str, variables: typing.Dict[str, VariableDefinition]):
        self.defined_variables.update([(v, var_def.parseinfo.rule, var_def.parseinfo.pos) for v, var_def in variables.items()])
        if term is not None and term in variables:
            self.used_variables.add((term, variables[term].parseinfo.rule, variables[term].parseinfo.pos))

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if len(self.defined_variables) == 0:
            return 0

        return len(self.defined_variables.intersection(self.used_variables)) / len(self.defined_variables)


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

        return len(self.defined_preferences.intersection(self.used_preferences)) / len(self.defined_preferences.union(self.used_preferences))


class SetupObjectsUsed(FitnessTerm):
    setup_objects: typing.Set[str] = set()
    used_objects: typing.Set[str] = set()

    def __init__(self):
        super().__init__(list(TYPE_RULES.keys()), 'setup_objects_used')

    def game_start(self) -> None:
        self.setup_objects = set()
        self.used_objects = set()

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if SECTION_CONTEXT_KEY not in context:
            raise ValueError('Section not found in context', context)

        if context[SECTION_CONTEXT_KEY] == ast_parser.SETUP:
            result = TYPE_RULES[rule][0](ast)
            if isinstance(result, (list, tuple)):
                self.setup_objects.update(result)
            elif result is not None:
                self.setup_objects.add(result)

        else:
            result = TYPE_RULES[rule][0](ast)
            if isinstance(result, (list, tuple)):
                self.used_objects.update(result)
            elif result is not None:
                self.used_objects.add(result)

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if len(self.setup_objects) == 0:
            return 0

        return len(self.setup_objects.intersection(self.used_objects)) / len(self.setup_objects)


class NoAdjacentOnce(FitnessTerm):
    total_prefs: int = 0
    prefs_with_adjacent_once: int = 0

    def __init__(self):
        super().__init__('then', 'no_adjacent_once')

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
            if len(func_rules) >= 3 and func_rules[0] == func_rules[-1] == 'once':
                self.prefs_start_and_end_with_once += 1

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if self.total_prefs == 0:
            return 0

        return self.prefs_start_and_end_with_once / self.total_prefs


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



PREDICATE_AND_FUNCTION_RULES = ('predicate', 'function_eval')


class VariableNotRepeatedInPredicateFunction(FitnessTerm):
    total_count: int = 0
    count_with_repeats: int = 0

    def __init__(self):
        super().__init__(PREDICATE_AND_FUNCTION_RULES, 'variable_not_repeated')

    def game_start(self) -> None:
        self.total_count = 0
        self.count_with_repeats = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            self.total_count += 1
            args = list(extract_predicate_function_args(ast))
            self.count_with_repeats += 1 if len(args) != len(set(args)) else 0

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if self.total_count == 0:
            return 0

        return 1 - (self.count_with_repeats / self.total_count)


ALL_BOOLEAN_RULE_PATTERN = re.compile(r'[\w_]+(_and|_or|_not)$')
MULTI_BOOLEAN_RULE_PATTERN = re.compile(r'[\w_]+(_and|_or)$')


class NoNestedLogicals(FitnessTerm):
    total_logicals: int = 0
    nested_logicals: int = 0

    def __init__(self):
        super().__init__(ALL_BOOLEAN_RULE_PATTERN, 'no_nested_logicals')

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

                if any((isinstance(child, tatsu.ast.AST) and isinstance(child.pred, tatsu.ast.AST) and child.pred.parseinfo.rule == rule) for child in children):  # type: ignore
                    self.nested_logicals += 1

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if self.total_logicals == 0:
            # TODO: should this return a NaN? If so, we should thin kabout how to handle them
            return 1

        return 1 - (self.nested_logicals / self.total_logicals)


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
        super().__init__(tuple(self.rule_to_section.keys()), 'no_identical_logical_children')

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
            # TODO: should this return a NaN? If so, we should think about how to handle them
            return 1

        return 1 - (self.identical_children / self.total_logicals)


BOOLEAN_PARSER = ast_parser.ASTBooleanParser()
BOOLEAN_LOGIC_IGNORE_PREFIXES = ('terminal_', 'scoring_')


class BooleanLogicTerm(FitnessTerm):
    boolean_parser: ast_parser.ASTBooleanParser
    def __init__(self, name: str, ignore_prefixes: typing.Sequence[str] = BOOLEAN_LOGIC_IGNORE_PREFIXES):
        super().__init__(MULTI_BOOLEAN_RULE_PATTERN, name)
        self.boolean_parser = BOOLEAN_PARSER
        self.ignore_prefixes = ignore_prefixes

    def game_start(self) -> None:
        self.boolean_parser.game_start()

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if any(rule.startswith(prefix) for prefix in self.ignore_prefixes):
            return

        if isinstance(ast, tatsu.ast.AST):
            expr = self.boolean_parser(ast, **context)
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
        return int(self.tautalogy_found)


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
        return int(self.redundancy_found)


class PrefForallTerm(FitnessTerm):
    pref_forall_found: bool = False
    pref_forall_prefs: typing.Set[str] = set()
    pref_forall_prefs_by_position: typing.Dict[int, typing.Set[str]] = {}

    def __init__(self, name: str):
        super().__init__(('scoring_external_maximize', 'scoring_external_minimize', 'pref_forall', COUNT_RULE_PATTERN), f'pref_forall_{name}')

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
        return_value = dict(correct=0, incorrect=0)

        inner_value = self._inner_game_end()
        if inner_value == 1:
            return_value['correct'] = 1
        elif inner_value == -1:
            return_value['incorrect'] = 1

        return return_value  # type: ignore


class CountOncePerExternalObjectsUsedCorrectly(PrefForallTerm):
    pref_forall_prefs: typing.Set[str] = set()
    count_once_per_external_objects_prefs: typing.Set[str] = set()

    def __init__(self):
        super().__init__('count_once_per_external_objects_used')

    def _inner_game_start(self) -> None:
        self.count_once_per_external_objects_prefs = set()

    def _update_count(self, pref_name: str, object_types: typing.Optional[typing.List[tatsu.ast.AST]],
        rule: str, context: ContextDict):

        if rule == 'count_once_per_external_objects':
            self.count_once_per_external_objects_prefs.add(pref_name)

    def _inner_game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if len(self.count_once_per_external_objects_prefs) == 0:
            return 0

        if len(self.count_once_per_external_objects_prefs.intersection(self.pref_forall_prefs)) == len(self.count_once_per_external_objects_prefs):
            return 1

        return -1


class ExternalForallUsedCorrectly(PrefForallTerm):
    pref_forall_prefs: typing.Set[str] = set()
    external_forall_positions: typing.Set[int] = set()
    external_forall_used_with_forall_pref_positions: typing.Set[int] = set()

    def __init__(self):
        super().__init__('external_forall_used')

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

        return -1


class PrefForallUsed(PrefForallTerm):
    pref_forall_prefs: typing.Set[str] = set()
    prefs_used_as_pref_forall_prefs: typing.Set[str] = set()

    def __init__(self):
        super().__init__('used')

    def _inner_game_start(self) -> None:
        self.prefs_used_as_pref_forall_prefs = set()

    def _update_count(self, pref_name: str, object_types: typing.Optional[typing.List[tatsu.ast.AST]],
        rule: str, context: ContextDict):
        if object_types is not None or EXTERNAL_FORALL_CONTEXT_KEY in context or rule == 'count_once_per_external_objects':
            self.prefs_used_as_pref_forall_prefs.add(pref_name)

    def _inner_game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if len(self.pref_forall_prefs) == 0 and len(self.prefs_used_as_pref_forall_prefs) == 0:
            return 0

        # return 1 if for each pref forall, at least one pref was used in a manner that requires a forall
        if all(len(pos_prefs.intersection(self.prefs_used_as_pref_forall_prefs)) > 0 for pos_prefs in self.pref_forall_prefs_by_position.values()):
            return 1

        return -1

class PrefForallCorrectArity(PrefForallTerm):
    correct_usage_count: int = 0
    incorrect_usage_count: int = 0
    pref_forall_prefs_to_counts: typing.Dict[str, int] = dict()


    def __init__(self):
        super().__init__('pref_forall_correct_arity')

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

        return 1 if  self.incorrect_usage_count == 0 else -1


class PrefForallCorrectTypes(PrefForallTerm):
    pref_forall_prefs_to_types: typing.Dict[str, typing.Dict[str, VariableDefinition]] = defaultdict(dict)
    prefs_with_correct_types: typing.List[float] = list()

    def __init__(self):
        super().__init__('pref_forall_correct_types')

    def _inner_game_start(self) -> None:
        self.pref_forall_prefs_to_types = defaultdict(dict)
        self.prefs_with_correct_types = list()

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

        self.prefs_with_correct_types.append(count_correct / len(object_types))

    def _inner_game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if len(self.pref_forall_prefs_to_types) == 0 or len(self.prefs_with_correct_types) == 0:
            return 0

        return 1 if np.isclose(np.mean(self.prefs_with_correct_types), 1) else -1  # type: ignore


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

        if rule == 'scoring_expr' and isinstance(ast.expr, str) and ast.expr in TOTAL_TERMINALS:  # type:ignore
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
    'game_over': 0, 'game_start': 0, 'in':2, 'in_motion': 1, 'is_setup_object': 1,
    'object_orientation': 2, 'on': 2, 'open': 1, 'opposite': 2, 'rug_color_under': 2,
    'same_color': 2, 'same_object': 2, 'same_type': 2, 'toggled_on': 1, 'touch': 2,
    'x_position': 1,
}


class CorrectPredicateFunctionArity(FitnessTerm):
    total_count: int = 0
    name_to_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]] = {}
    count_with_wrong_arity: int = 0

    def __init__(self, name_to_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]] = PREDICATE_FUNCTION_ARITY_MAP):  # type: ignore
        super().__init__(PREDICATE_AND_FUNCTION_RULES, 'correct_predicate_function_arity')
        self.name_to_arity_map = name_to_arity_map

    def game_start(self) -> None:
        self.total_count = 0
        self.count_with_wrong_arity = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            self.total_count += 1

            name = extract_predicate_function_name(ast, remove_digits=False)

            if name not in self.name_to_arity_map:
                raise ValueError(f'Predicate {name} not in predicate arity map')

            n_args = extract_n_args(ast)
            arity = self.name_to_arity_map[name]  # type: ignore

            if isinstance(arity, int):
                if n_args != arity:
                    self.count_with_wrong_arity += 1

            elif n_args not in arity:
                self.count_with_wrong_arity += 1

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if self.total_count == 0:
            return 0

        return 1 - (self.count_with_wrong_arity / self.total_count)


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
    return isinstance(s, str) and s.replace('.', '', 1).isdigit()


class NoTwoNumberOperations(FitnessTerm):
    total_operations: int = 0
    two_number_operations: int = 0

    def __init__(self):
        super().__init__(TWO_NUMBER_RULES, 'no_two_number_operations')

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
            return 1

        return 1 - (self.two_number_operations / self.total_operations)


# COMMON_SENSE_PREDICATES_FUNCTIONS = ('adjacent', 'agent_holds', 'distance', 'in', 'in_motion', 'on', 'touch')
COMMON_SENSE_PREDICATES_FUNCTIONS = ('adjacent', 'adjacent_side_3', 'agent_holds', 'between', 'distance', 'in', 'in_motion', 'object_orientation', 'on', 'touch')
COMMON_SENSE_TYPE_CATEGORIES = list(room_and_object_types.CATEGORIES_TO_TYPES.keys())
COMMON_SENSE_TYPE_CATEGORIES.remove(room_and_object_types.EMPTY_OBJECT)
KNOWN_MISSING_TYPES = []


class PredicateFunctionArgumentTypes(FitnessTerm):
    argument_type_categories: typing.Sequence[str]
    matching_argument_types_count: int = 0
    predicate_or_function: str
    name_to_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]]

    def __init__(self, predicate_or_function: str, argument_type_categories: typing.Sequence[str],
        name_to_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]] = PREDICATE_FUNCTION_ARITY_MAP,  # type: ignore
        known_missing_types: typing.Sequence[str] = KNOWN_MISSING_TYPES):

        super().__init__((f'predicate_{predicate_or_function}', f'function_{predicate_or_function}'),
            f'{predicate_or_function}_arg_types_{"_".join(argument_type_categories)}')
        self.predicate_or_function = predicate_or_function
        self.argument_type_categories = argument_type_categories
        self.name_to_arity_map = name_to_arity_map
        self.known_missing_types = known_missing_types

        if len(argument_type_categories) != self.name_to_arity_map[predicate_or_function]:
            raise ValueError(f'Predicate {predicate_or_function} has arity {self.name_to_arity_map[predicate_or_function]} but {len(argument_type_categories)} argument types were provided')

    def game_start(self) -> None:
        self.matching_argument_types_count = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            name = extract_predicate_function_name(ast, remove_digits=False)

            if name not in self.name_to_arity_map:
                raise ValueError(f'Predicate {ast.name} not in predicate arity map')

            if name != self.predicate_or_function:
                return

            n_args = extract_n_args(ast)
            arity = self.name_to_arity_map[name]  # type: ignore

            if isinstance(arity, int):
                if n_args != arity:
                    return

            elif n_args not in arity:
                return

            terms = extract_predicate_function_args(ast)
            term_type_lists = []

            context_variables = typing.cast(typing.Dict[str, VariableDefinition], context[VARIABLES_CONTEXT_KEY]) if VARIABLES_CONTEXT_KEY in context else {}
            term_categories = [predicate_function_term_to_type_category(term, context_variables, self.known_missing_types) for term in terms]

            if all(term_categories[i] is not None and self.argument_type_categories[i] in term_categories[i] for i in range(len(term_categories))):
                self._count(context)

    def _count(self, context: ContextDict):
        self.matching_argument_types_count += 1

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return self.matching_argument_types_count


PREDICATE_SECTIONS = [ast_parser.SETUP, ast_parser.PREFERENCES]


class PredicateFunctionArgumentTypesBySection(PredicateFunctionArgumentTypes):
    matching_argument_types_count_by_section: typing.Dict[str, Number]

    def __init__(self, predicate_or_function: str, argument_type_categories: typing.Sequence[str],
        name_to_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]] = PREDICATE_FUNCTION_ARITY_MAP,  # type: ignore
        known_missing_types: typing.Sequence[str] = KNOWN_MISSING_TYPES):

        super().__init__(predicate_or_function, argument_type_categories, name_to_arity_map, known_missing_types)

    def game_start(self) -> None:
        self.matching_argument_types_count_by_section = {section: 0 for section in PREDICATE_SECTIONS}

    def _count(self, context: ContextDict):
        self.matching_argument_types_count_by_section[context[SECTION_CONTEXT_KEY]] += 1  # type: ignore

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return {k.replace('(:', ''): v for k, v in self.matching_argument_types_count_by_section.items()}


def build_argument_types_fitness_terms(
    predicates: typing.Sequence[str] = COMMON_SENSE_PREDICATES_FUNCTIONS,
    type_categories: typing.Sequence[str] = COMMON_SENSE_TYPE_CATEGORIES,
    predicate_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]] = PREDICATE_FUNCTION_ARITY_MAP) -> typing.Sequence[FitnessTerm]:  # type: ignore
    fitness_terms = []

    sorted_type_categories = list(sorted(type_categories))

    for predicate in predicates:
        for type_combinations in itertools.product(*([sorted_type_categories] * predicate_arity_map[predicate])):  # type: ignore
            fitness_terms.append(PredicateFunctionArgumentTypesBySection(predicate, type_combinations, predicate_arity_map))
            # fitness_terms.append(PredicateFunctionArgumentTypes(predicate, type_combinations, predicate_arity_map))

    return fitness_terms


COMPOSITIONALITY_STRUCTURES = (
    '(hold (and (not (agent_holds ?x) ) (in_motion ?x) ) )',
    '(once (and (not (in_motion ?x) ) (in ?x ?x) ) )',
    '(once (agent_holds ?x) )',
    '(once (not (in_motion ?x) ) )',
    '(once (and (agent_holds ?x) (adjacent ?x ?x) ) )',
    '(hold-while (and (not (agent_holds ?x) ) (in_motion ?x) ) (touch ?x ?x) )',
    '(once (and (not (in_motion ?x) ) (on ?x ?x) ) )',
    '(hold-while (and (in_motion ?x) (not (agent_holds ?x) ) ) (touch ?x ?x) )',
    '(once (and (adjacent ?x ?x) (agent_holds ?x) ) )',
    '(hold-while (and (not (agent_holds ?x) ) (in ?x ?x) (or (agent_holds ?x) (and (not (agent_holds ?x) ) (in_motion ?x) ) ) ) (touch ?x ?x) )',
    '(hold-while (and (in_motion ?x) (not (agent_holds ?x) ) ) (touch ?x ?x) (in_motion ?x) )',
    '(hold-while (and (not (agent_holds ?x) ) (in_motion ?x) ) (on ?x ?x) )',
    '(once (and (agent_holds ?x) (on ?x ?x) ) )'
 )


PREDICATE_ARGS_PATTERN = r'\(\s*(?:[\w-]+)\s+((?:\??\w+\s*)+)\)'
COMPOSITIONALITY_VARIABLE_REPLACEMENT = '?x'


class CompositionalityStructureCounter(FitnessTerm):
    structure_str: str
    structure_count: int = 0
    structure_index: int
    args_pattern: re.Pattern
    variable_replacement: str

    def __init__(self, structure: str, structure_index: int,
        variable_replacement: str = COMPOSITIONALITY_VARIABLE_REPLACEMENT,
        predicate_arg_pattern: str = PREDICATE_ARGS_PATTERN):
        rule = structure.split(' ')[0][1:].replace('-', '_')
        if rule == 'hold_while':
            rule = 'while_hold'
        super().__init__(rule, f'compositionality_structure_{structure_index}')
        self.structure_str = structure
        self.structure_index = structure_index
        self.variable_replacement = variable_replacement
        self.args_pattern = re.compile(predicate_arg_pattern)

    def game_start(self) -> None:
        self.structure_count = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            ast_str = ast_printer.ast_section_to_string(ast, ast_parser.PREFERENCES)
            for args in self.args_pattern.findall(ast_str):
                ast_str = ast_str.replace(args, ' '.join(map(lambda x: self.variable_replacement, args.split(" "))), 1)

            self.structure_count += ast_str == self.structure_str

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return self.structure_count


def build_compositionality_fitness_terms(
    compositionality_structures: typing.Sequence[str] = COMPOSITIONALITY_STRUCTURES,
    variable_replacement: str = COMPOSITIONALITY_VARIABLE_REPLACEMENT) -> typing.Sequence[FitnessTerm]:

    return [CompositionalityStructureCounter(structure, i, variable_replacement) for i, structure in enumerate(compositionality_structures)]


class SectionCountTerm(FitnessTerm):
    def __init__(self, section: str, header: str, thresholds: typing.Optional[typing.Sequence[float]]):
        super().__init__(section, header.replace('(:', ''))
        if thresholds is not None:
            thresholds = list(thresholds)
            thresholds.insert(0, float('-inf'))
            thresholds.append(float('inf'))

        self.thresholds = thresholds

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        result = self._inner_game_end()
        if self.thresholds is None:
            return result

        return {i: 1 if self.thresholds[i] <= result < self.thresholds[i + 1] else 0 for i in range(len(self.thresholds) - 1)}

    @abstractmethod
    def _inner_game_end(self) -> Number:
        pass


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


SECTION_COUNT_THRESHOLDS = {
    (SectionMaxDepth, ast_parser.SETUP): [1, 8.5, 17.5, 25.5],
    (SectionMaxDepth, ast_parser.PREFERENCES): [8.5, 15.5, 19.5, 23.5],
    (SectionMaxDepth, ast_parser.TERMINAL): [1, 4.5, 8.5, 11.5],
    (SectionMaxDepth, ast_parser.SCORING): [2.5, 4.5, 8.5, 12.5],

    (SectionMeanDepth, ast_parser.SETUP): [1, 4, 9, 12],
    (SectionMeanDepth, ast_parser.PREFERENCES): [6.5, 8, 10, 12],
    (SectionMeanDepth, ast_parser.TERMINAL): [1, 2, 3.8, 5.5],
    (SectionMeanDepth, ast_parser.SCORING): [1.5, 2.75, 4, 6],

    (SectionNodeCount, ast_parser.SETUP): [1, 10, 30, 100],
    (SectionNodeCount, ast_parser.PREFERENCES): [30, 70, 135, 200],
    (SectionNodeCount, ast_parser.TERMINAL): [1, 5, 25, 55],
    (SectionNodeCount, ast_parser.SCORING): [6, 18, 33, 60],

}


def build_section_count_fitness_terms(sections: typing.Sequence[str] = ast_parser.SECTION_KEYS,
    term_classes: typing.Sequence[typing.Callable] = (SectionMaxDepth, SectionMeanDepth, SectionNodeCount),
    thresholds: typing.Optional[typing.Dict[typing.Tuple[typing.Callable, str], typing.Sequence[float]]] = SECTION_COUNT_THRESHOLDS,
    ) -> typing.Sequence[FitnessTerm]:

    if thresholds is not None:
        return [term_class(section, thresholds[(term_class, section)]) for term_class in term_classes for section in sections]
    else:
        return [term_class(section) for term_class in term_classes for section in sections]


DEFAULT_TOP_K_NGRAMS = 10
TEXT_N_GRAM_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/text_2_3_4_5_ngram_model_2023_02_05.pkl')


class TextNGramTerm(FitnessTerm):
    game_output: typing.Optional[dict] = None
    n_gram_model: NGramTrieModel
    n_gram_model_path: str
    top_k_ngrams: int

    def __init__(self, top_k_ngrams: int = DEFAULT_TOP_K_NGRAMS, n_gram_model_path: str = TEXT_N_GRAM_MODEL_PATH):
        super().__init__('', 'text_ngram')
        self.top_k_ngrams = top_k_ngrams
        self.n_gram_model_path = n_gram_model_path
        with open(self.n_gram_model_path, 'rb') as f:
            self.n_gram_model = pickle.load(f)

    def game_start(self) -> None:
        self.game_output = None

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        pass

    def parse_full_text(self, full_text: str) -> None:
        self.game_output = self.n_gram_model.score(full_text, k=self.top_k_ngrams)

    def game_end(self):
        return self.game_output


AST_N_GRAM_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/ast_7_ngram_model_2023_02_13.pkl')


class ASTNGramTerm(FitnessTerm):
    filter_padding_top_k: bool
    game_output: typing.Optional[dict] = None
    log: bool
    n_gram_model: ASTNGramTrieModel
    n_gram_model_path: str
    stupid_backoff: bool
    top_k_max_n: typing.Optional[int]
    top_k_min_n: typing.Optional[int]
    top_k_ngrams: int

    def __init__(self, top_k_ngrams: int = DEFAULT_TOP_K_NGRAMS,
                 stupid_backoff: bool = True, log: bool = True,
                 filter_padding_top_k: bool = False, top_k_min_n: typing.Optional[int] = None,
                 top_k_max_n: typing.Optional[int] = None,
                 n_gram_model_path: str = AST_N_GRAM_MODEL_PATH):
        super().__init__('', 'ast_ngram')
        self.top_k_ngrams = top_k_ngrams
        self.stupid_backoff = stupid_backoff
        self.log = log
        self.filter_padding_top_k = filter_padding_top_k
        self.top_k_min_n = top_k_min_n
        self.top_k_max_n = top_k_max_n
        self.n_gram_model_path = n_gram_model_path

        with open(self.n_gram_model_path, 'rb') as f:
            self.n_gram_model = pickle.load(f)

    def game_start(self) -> None:
        self.game_output = None

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        self.game_output = self.n_gram_model.score(
            ast, k=self.top_k_ngrams, stupid_backoff=self.stupid_backoff,  # type: ignore
            log=self.log, filter_padding_top_k=self.filter_padding_top_k,
            top_k_min_n=self.top_k_min_n, top_k_max_n=self.top_k_max_n
        )

    def game_end(self):
        return self.game_output


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

    all_setup_objects_used = SetupObjectsUsed()
    fitness.register(all_setup_objects_used)

    no_adjacent_once = NoAdjacentOnce()
    fitness.register(no_adjacent_once)

    no_adjacent_same_modal = NoAdjacentSameModal()
    fitness.register(no_adjacent_same_modal)

    pref_starts_and_ends_with_once = PrefStartsAndEndsWithOnce()
    fitness.register(pref_starts_and_ends_with_once)

    length_of_then_modals = LengthOfThenModals()
    fitness.register(length_of_then_modals)

    no_repeated_variables_in_predicate = VariableNotRepeatedInPredicateFunction()
    fitness.register(no_repeated_variables_in_predicate)

    no_nested_logicals = NoNestedLogicals()
    fitness.register(no_nested_logicals)

    no_identical_logical_children = NoIdenticalChildrenInLogicals()
    fitness.register(no_identical_logical_children)

    tautological_boolean_expression = TautologicalBooleanExpression()
    fitness.register(tautological_boolean_expression)

    redundant_boolean_expression = RedundantBooleanExpression()
    fitness.register(redundant_boolean_expression)

    count_once_per_external_objects_used = CountOncePerExternalObjectsUsedCorrectly()
    fitness.register(count_once_per_external_objects_used)

    external_forall_used = ExternalForallUsedCorrectly()
    fitness.register(external_forall_used)

    # This feature is just subsumed by the rest of the pref forall features at this point
    # pref_forall_used = PrefForallUsed()
    # fitness.register(pref_forall_used)

    pref_forall_correct_arity = PrefForallCorrectArity()
    fitness.register(pref_forall_correct_arity)

    pref_forall_correct_types = PrefForallCorrectTypes()
    fitness.register(pref_forall_correct_types)

    # Changed in the grammar to enforce correct arity
    # correct_predicate_arity = CorrectPredicateFunctionArity()
    # fitness.register(correct_predicate_arity)

    no_two_number_comparisons = NoTwoNumberOperations()
    fitness.register(no_two_number_comparisons)

    no_count_in_terminal = SectionWithoutPrefOrTotalCounts(ast_parser.TERMINAL)
    fitness.register(no_count_in_terminal, section_rule=True)

    no_count_in_scoring = SectionWithoutPrefOrTotalCounts(ast_parser.SCORING)
    fitness.register(no_count_in_scoring, section_rule=True)

    argument_types_fitness_terms = build_argument_types_fitness_terms()
    fitness.register_multiple(argument_types_fitness_terms)

    compositionality_fitness_terms = build_compositionality_fitness_terms()
    fitness.register_multiple(compositionality_fitness_terms)

    section_count_fitness_terms = build_section_count_fitness_terms()
    fitness.register_multiple(section_count_fitness_terms, section_rule=True)

    # text_ngram_term = TextNGramTerm()
    # fitness.register(text_ngram_term, full_text_rule=True)

    ast_ngram_term = ASTNGramTerm(top_k_min_n=2)
    fitness.register(ast_ngram_term, full_ast_rule=True)

    return fitness


def main(args):
    original_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(args.recursion_limit)

    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    featurizer = build_fitness_featurizer(args)

    for test_file in args.test_files:
        for ast in cached_load_and_parse_games_from_file(test_file, grammar_parser, not args.dont_tqdm):
            featurizer.parse(ast, test_file)

    df = featurizer.to_df()

    print(df.groupby('src_file').agg([np.mean, np.std]))

    for src_file in df.src_file.unique():
        zero_std = df[df.src_file == src_file].std(numeric_only=True) == 0
        zero_std_columns = [c for c in zero_std.index if zero_std[c] and not 'arg_types' in c]  # type: ignore
        print(f'For src_file {src_file}, the following columns have zero std (excluding arg_types columns): {zero_std_columns}')

    global_zero_std = df.std(numeric_only=True) == 0
    global_zero_std_columns = [c for c in global_zero_std.index if global_zero_std[c] and not 'arg_types' in c]  # type: ignore
    print(f'For all src_files, the following columns have zero std (excluding arg_types columns): {global_zero_std_columns}')

    if not args.output_path.endswith('.gz'):
        args.output_path += '.gz'

    df.to_csv(args.output_path, index_label='Index', compression='gzip')

    if args.featurizer_output_path is not None:
        with gzip.open(args.featurizer_output_path, 'wb') as f:
            pickle.dump(featurizer, f)  # type: ignore

    sys.setrecursionlimit(original_recursion_limit)


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)

    for test_file in args.test_files:
        if not os.path.exists(test_file):
            raise ValueError(f'File {test_file} does not exist')

    if args.featurizer_output_path is None:
        args.featurizer_output_path = DEFAULT_FEATURIZER_OUTPUT_PATH_PATTERN.format(today=datetime.now().strftime('%Y_%m_%d'))

    main(args)
