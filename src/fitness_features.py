from abc import ABC, abstractmethod
import argparse
from collections import namedtuple, defaultdict
import itertools
import tatsu
import tatsu.ast
import tatsu.buffering
import tatsu.infos
import tqdm
import pandas as pd
import numpy as np
import os
import re
import sys
import typing

from ast_utils import cached_load_and_parse_games_from_file
from ast_parser import ASTParser
import ast_printer
from ast_to_latex_doc import TYPE_RULES, extract_n_args, extract_predicate_function_args, extract_predicate_function_name
from room_and_object_types import *



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
DEFAULT_OUTPUT_PATH ='./data/fitness_scores.csv'
parser.add_argument('-o', '--output-path', default=DEFAULT_OUTPUT_PATH)
DEFAULT_RECURSION_LIMIT = 2000
parser.add_argument('--recursion-limit', type=int, default=DEFAULT_RECURSION_LIMIT)


VariableDefinition = namedtuple('VariableDefinition', ('var_names', 'var_types', 'parseinfo'))
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

    @abstractmethod
    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        pass


DEFAULT_HEADERS = ('src_file', 'game_name', 'domain_name')
SETUP = 'setup'
PREFERENCES = 'constraints'
TERMINAL = 'terminal'
SCORING = 'scoring'
SECTION_KEYS = (SETUP, PREFERENCES, TERMINAL, SCORING)

VARIABLES_CONTEXT_KEY = 'variables'
SECTION_CONTEXT_KEY = 'section'
DEPTH_CONTEXT_KEY = 'depth'
EXTERNAL_FORALL_CONTEXT_KEY = 'external_forall'

COUNT_RULE_PATTERN = re.compile('count.*')


def _extract_variables_from_ast(ast: tatsu.ast.AST, vars_key: str, context_vars: typing.Dict[str, VariableDefinition]) -> None:
    variables = ast[vars_key].variables  # type: ignore
    if isinstance(variables, tatsu.ast.AST):
        variables = [variables]
    
    for var_def in variables:  # type: ignore
        var_names = var_def.var_names
        if isinstance(var_names, str): 
            var_names = [var_names]
        
        var_type = var_def.var_type.type  # type: ignore
        if isinstance(var_type, tatsu.ast.AST):
            var_type = var_type.type_names

        if isinstance(var_type, str): 
            var_type = [var_type]

        for var_name in var_names:  # type: ignore
            context_vars[var_name] = VariableDefinition(var_names, var_type, var_def.parseinfo)


class ASTFitnessFeaturizer:
    headers: typing.List[str]
    header_registry: typing.Dict[str, FitnessTerm]
    list_reduce: typing.Callable[[typing.Sequence[Number]], Number]
    regex_rules: typing.List[typing.Tuple[re.Pattern, FitnessTerm]]
    rows: typing.List
    rule_registry: typing.Dict[str, typing.List[FitnessTerm]]
    section_registry: typing.Dict[str, typing.List[FitnessTerm]]
    tuple_registry: typing.Dict[str, typing.List[FitnessTerm]] 
    section_keys: typing.List[str]

    def __init__(self, headers: typing.Sequence[str] = DEFAULT_HEADERS, 
        list_reduce: typing.Callable[[typing.Sequence[Number]], Number] = np.sum,
        section_keys: typing.Sequence[str] = SECTION_KEYS):
        self.headers = list(headers)
        self.list_reduce = list_reduce
        self.section_keys = list(section_keys)


        self.rule_registry = defaultdict(list)
        self.tuple_registry = defaultdict(list)
        self.section_registry = defaultdict(list)    
        self.regex_rules = []
        self.header_registry = dict()

        self.rows = []

    def _register(self, term: FitnessTerm, rule: str, tuple_rule: bool = False) -> None:
        if tuple_rule:
            self.tuple_registry[rule].append(term)
        else:
            self.rule_registry[rule].append(term)

    def register(self, term: FitnessTerm, tuple_rule: bool = False, section_rule: bool = False) -> None:
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
        return pd.DataFrame.from_records(self.rows, columns=list(self.rows[0].keys()))

    def parse(self, ast: typing.Tuple[tatsu.ast.AST, tatsu.ast.AST, tatsu.ast.AST, tatsu.ast.AST], src_file: str, return_row: bool = False):
        row = {}
        row['src_file'] = os.path.basename(src_file)
        row['game_name'] = ast[1]["game_name"]  # type: ignore
        row['domain_name'] = ast[2]["domain_name"]  # type: ignore
        ast = ast[3:]  # type: ignore

        for term in self.header_registry.values():
            term.game_start()

        self._parse(ast)

        for header, term in self.header_registry.items():
            term_result = term.game_end()
            if isinstance(term_result, (int, float)):
                row[header] = term_result
            elif isinstance(term_result, dict):
                for key, val in term_result.items():
                    row[f'{header}_{key}'] = val
            else:
                row[header] = self.list_reduce(row[header])

        self.rows.append(row)
        if return_row:
            return row

    def _parse(self, ast: typing.Union[str, int, tatsu.buffering.Buffer, tuple, list, tatsu.ast.AST],
        context: typing.Optional[ContextDict] = None):
        if context is None:
            context = {DEPTH_CONTEXT_KEY: 0}

        if not ast or isinstance(ast, (str, int, np.int32, np.int64, tatsu.buffering.Buffer)):  # type: ignore
            return

        elif isinstance(ast, (tuple, list)):
            if len(ast) > 0 and isinstance(ast[0], str):
                # check if should update the section key
                if ast[0][2:] in self.section_keys:
                    context[SECTION_CONTEXT_KEY] = ast[0][2:]

                for tuple_stat in self.tuple_registry[ast[0]]:
                    tuple_stat.update(ast, '', context)

            [self._parse(element, context) for element in ast]

        elif isinstance(ast, tatsu.ast.AST):
            # Look for variable definitions
            vars_keys = [key for key in ast.keys() if key.endswith('_vars')]
            if len(vars_keys) > 1:
                raise ValueError(f'Found multiple variables keys: {vars_keys}', ast)

            elif len(vars_keys) > 0:
                vars_key = vars_keys[0]
                context_vars = typing.cast(dict, context[VARIABLES_CONTEXT_KEY]) if VARIABLES_CONTEXT_KEY in context else {}
                _extract_variables_from_ast(ast, vars_key, context_vars) 
                context = context.copy()
                context[VARIABLES_CONTEXT_KEY] = context_vars  # type: ignore

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
                child_context[EXTERNAL_FORALL_CONTEXT_KEY] = ast.parseinfo.rule  

            for child_key in ast:
                if child_key != 'parseinfo':
                    self._parse(ast[child_key], child_context)  # type: ignore

        else:
            print(f'Encountered AST element with unrecognized type: {ast} of type {type(ast)}')


class ASTNodeCounter(ASTParser):
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

        if context[SECTION_CONTEXT_KEY] == SETUP:
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
        super().__init__(('then', 'at_end'), 'no_adjacent_once')

    def game_start(self) -> None:
        self.total_prefs = 0
        self.prefs_with_adjacent_once = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        self.total_prefs += 1
        
        if rule == 'at_end':
            return

        if isinstance(ast.then_funcs, list):  # type: ignore
            func_rules = [sf.seq_func.parseinfo.rule if isinstance(sf.seq_func, tatsu.ast.AST) else sf.seq_func for sf in ast.then_funcs]  # type: ignore
            for i in range(len(func_rules) - 1):
                if func_rules[i] == func_rules[i + 1] == 'once':
                    self.prefs_with_adjacent_once += 1
                    break

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if self.total_prefs == 0:
            return 0

        return 1 - (self.prefs_with_adjacent_once / self.total_prefs)


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


class NoNestedLogicals(FitnessTerm):
    total_logicals: int = 0
    nested_logicals: int = 0

    def __init__(self):
        super().__init__(('setup_and', 'setup_or', 'setup_not', 'super_predicate_and',
        'super_predicate_or', 'super_predicate_not', 'terminal_and',
        'terminal_or', 'terminal_not', 'scoring_and', 'scoring_or', 'scoring_not'), 'no_nested_logicals')

    def game_start(self) -> None:
        self.total_logicals = 0
        self.nested_logicals = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            self.total_logicals += 1
            
            if rule.endswith('_not'):
                if isinstance(ast.not_args, tatsu.ast.AST) and ast.not_args.parseinfo.rule == rule:  # type: ignore
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
            'setup_and': ast_printer.SETUP_KEY,
            'setup_or': ast_printer.SETUP_KEY,
            'super_predicate_and': ast_printer.PREFERENCES_KEY,
            'super_predicate_or': ast_printer.PREFERENCES_KEY,
            'terminal_and': ast_printer.TERMINAL_KEY,
            'terminal_or': ast_printer.TERMINAL_KEY,
            'scoring_and': ast_printer.SCORING_KEY,
            'scoring_or': ast_printer.SCORING_KEY,
        }
        super().__init__(tuple(self.rule_to_section.keys()), 'no_identical_logical_children')

    def game_start(self) -> None:
        self.total_logicals = 0
        self.identical_children = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            rule_name = rule.split('_')[-1]
            children = ast[f'{rule_name}_args']
            if isinstance(children, tatsu.ast.AST) or len(children) < 2:
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


# ast_str = ast_printer.ast_section_to_string(ast, ast_printer.PREFERENCES_KEY)

class PrefForallTerm(FitnessTerm):
    def __init__(self, name: str):
        super().__init__(('scoring_external_maximize', 'scoring_external_minimize', 'pref_forall', COUNT_RULE_PATTERN), name)

    def _update_pref_forall_def(self, ast: tatsu.ast.AST, context: ContextDict):
        preferences = ast.forall_pref.preferences  # type: ignore
        if isinstance(preferences, tatsu.ast.AST):
            preferences = [preferences]

        self.pref_forall_prefs.update([pref.pref_name for pref in preferences])  # type: ignore

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
                self._update_pref_forall_def(ast, context)

            elif rule in ('scoring_external_maximize', 'scoring_external_minimize'):
                self._update_external_forall(ast, context)

            else:   # count*
                pref_name = ast.name_and_types['pref_name']  # type: ignore
                object_types = ast.name_and_types['object_types']  # type: ignore
                self._update_count(pref_name, object_types, rule, context)


class CountOncePerExternalObjectsUsedCorrectly(PrefForallTerm):
    pref_forall_prefs: typing.Set[str] = set()
    count_once_per_external_objects_prefs: typing.Set[str] = set()

    def __init__(self):
        super().__init__('count_once_per_external_objects_used_correctly')

    def game_start(self) -> None:
        self.pref_forall_prefs = set()
        self.count_once_per_external_objects_prefs = set()

    def _update_count(self, pref_name: str, object_types: typing.Optional[typing.List[tatsu.ast.AST]], 
        rule: str, context: ContextDict):

        if rule == 'count_once_per_external_objects':
            self.count_once_per_external_objects_prefs.add(pref_name)

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if len(self.count_once_per_external_objects_prefs) == 0:
            return 1

        return len(self.count_once_per_external_objects_prefs.intersection(self.pref_forall_prefs)) / len(self.count_once_per_external_objects_prefs)


class ExternalForallUsedCorrectly(PrefForallTerm):
    pref_forall_prefs: typing.Set[str] = set()
    external_forall_used: int = 0
    external_forall_used_with_forall_pref: int = 0

    def __init__(self):
        super().__init__('external_forall_used_correctly')

    def game_start(self) -> None:
        self.pref_forall_prefs = set()
        self.external_forall_used = 0
        self.external_forall_used_with_forall_pref = 0

    def _update_external_forall(self, ast: tatsu.ast.AST, context: ContextDict):
        self.external_forall_used += 1

    def _update_count(self, pref_name: str, object_types: typing.Optional[typing.List[tatsu.ast.AST]], 
        rule: str, context: ContextDict):
        if pref_name in self.pref_forall_prefs:
            self.external_forall_used_with_forall_pref += 1

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if self.external_forall_used == 0:
            return 1

        return min(self.external_forall_used_with_forall_pref / self.external_forall_used, 1)


class PrefForallUsed(PrefForallTerm):
    pref_forall_prefs: typing.Set[str] = set()
    prefs_used_as_pref_forall_prefs: typing.Set[str] = set()

    def __init__(self):
        super().__init__('pref_forall_used')

    def game_start(self) -> None:
        self.pref_forall_prefs = set()
        self.prefs_used_as_pref_forall_prefs = set()

    def _update_count(self, pref_name: str, object_types: typing.Optional[typing.List[tatsu.ast.AST]], 
        rule: str, context: ContextDict):
        if object_types is not None or EXTERNAL_FORALL_CONTEXT_KEY in context:
            self.prefs_used_as_pref_forall_prefs.add(pref_name)

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if len(self.pref_forall_prefs) == 0 and len(self.prefs_used_as_pref_forall_prefs) == 0:
            return 1

        return len(self.pref_forall_prefs.intersection(self.prefs_used_as_pref_forall_prefs)) / len(self.pref_forall_prefs.union(self.prefs_used_as_pref_forall_prefs))


class PrefForallCorrectArity(PrefForallTerm):
    correct_usage_count: int = 0 
    incorrect_usage_count: int = 0
    pref_forall_prefs_to_counts: typing.Dict[str, int] = dict()
    

    def __init__(self):
        super().__init__('pref_forall_correct_arity')

    def game_start(self) -> None:
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

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if len(self.pref_forall_prefs_to_counts) == 0:
            return 1

        total_usage_count = self.correct_usage_count + self.incorrect_usage_count
        if total_usage_count == 0:
            return 0

        return self.correct_usage_count / total_usage_count


# Copied from `reward_machines/config.py`
META_TYPES = {BALL: [BEACHBALL, BASKETBALL, DODGEBALL, GOLFBALL],
              BLOCK: [BRIDGE_BLOCK, CUBE_BLOCK, CYLINDRICAL_BLOCK, FLAT_BLOCK, 
                      PYRAMID_BLOCK, TALL_CYLINDRICAL_BLOCK, TALL_RECTANGULAR_BLOCK, TRIANGLE_BLOCK],
              COLOR: CATEGORIES_TO_TYPES[COLORS],
              CUBE_BLOCK: [BLUE_CUBE_BLOCK, TAN_CUBE_BLOCK, YELLOW_CUBE_BLOCK],
              DODGEBALL: [BLUE_DODGEBALL, PINK_DODGEBALL, RED_DODGEBALL],
              PYRAMID_BLOCK: [ BLUE_PYRAMID_BLOCK, RED_PYRAMID_BLOCK, YELLOW_PYRAMID_BLOCK]}

TYPE_TO_META_TYPE = {t: m for m, ts in META_TYPES.items() for t in ts}



class PrefForallCorrectTypes(PrefForallTerm):
    pref_forall_prefs_to_types: typing.Dict[str, typing.Dict[str, VariableDefinition]] = defaultdict(dict)
    prefs_with_correct_types: typing.List[float] = list()

    def __init__(self):
        super().__init__('pref_forall_correct_types')

    def game_start(self) -> None:
        self.pref_forall_prefs_to_types = defaultdict(dict)
        self.prefs_with_correct_types = list()

    def _update_pref_forall_def(self, ast: tatsu.ast.AST, context: ContextDict):
        preferences = ast.forall_pref.preferences  # type: ignore
        if isinstance(preferences, tatsu.ast.AST):
            preferences = [preferences]

        var_dict = {}
        _extract_variables_from_ast(ast, 'forall_vars', var_dict) 

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
            if obj in var_types or (obj in TYPE_TO_META_TYPE and TYPE_TO_META_TYPE[obj] in var_types):  # type: ignore
                count_correct += 1

        self.prefs_with_correct_types.append(count_correct / len(object_types))

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        if len(self.pref_forall_prefs_to_types) == 0 or len(self.prefs_with_correct_types) == 0:
            return 1

        return np.mean(self.prefs_with_correct_types)  # type: ignore


class SectionWithoutPrefCounts(FitnessTerm):
    section_found: bool
    count_rule_found: bool
    def __init__(self, section: str):
        super().__init__(section, f'section_without_pref_count_{section}')
        self.section = section

    def game_start(self) -> None:
        self.section_found = False
        self.count_rule_found = False

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if SECTION_CONTEXT_KEY in context and context[SECTION_CONTEXT_KEY] == self.section:
            self.section_found = True
        
        if isinstance(ast, tatsu.ast.AST) and COUNT_RULE_PATTERN.match(rule):
            self.found_count = True

    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return 1 if self.section_found == self.count_rule_found else 0


PREDICATE_FUNCTION_ARITY_MAP = {
    'above': 2, 'adjacent': 2, 'adjacent_side': (3, 4), 'agent_crouches': 0, 'agent_holds': 1,
    'between': 3, 'broken': 1, 'building_size': 1, 'distance': 2, 'distance_side': (3, 4),
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

    def __init__(self, name_to_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]] = PREDICATE_FUNCTION_ARITY_MAP):
        super().__init__(PREDICATE_AND_FUNCTION_RULES, 'correct_predicate_function_arity')
        self.name_to_arity_map = name_to_arity_map

    def game_start(self) -> None:
        self.total_count = 0
        self.count_with_wrong_arity = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            self.total_count += 1

            name = extract_predicate_function_name(ast)

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
    




COMMON_SENSE_PREDICATES_FUNCTIONS = ('adjacent', 'agent_holds', 'distance', 'in', 'in_motion', 'on', 'touch')
COMMON_SENSE_TYPE_CATEGORIES = list(CATEGORIES_TO_TYPES.keys())
COMMON_SENSE_TYPE_CATEGORIES.remove(EMPTY_OBJECT)
KNOWN_MISSING_TYPES = ('back', 'front', 'front_left_corner', 'left', 'right', 'sideways', 'upright', 'upside_down')


class PredicateFunctionArgumentTypes(FitnessTerm):
    argument_type_categories: typing.Sequence[str]
    matching_argument_types_count: int = 0
    predicate_or_function: str
    name_to_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]]

    def __init__(self, predicate: str, argument_type_categories: typing.Sequence[str], 
        name_to_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]] = PREDICATE_FUNCTION_ARITY_MAP,
        known_missing_types: typing.Sequence[str] = KNOWN_MISSING_TYPES):

        super().__init__(PREDICATE_AND_FUNCTION_RULES, f'arg_types_{predicate}_{"_".join(argument_type_categories)}')
        self.predicate_or_function = predicate
        self.argument_type_categories = argument_type_categories
        self.name_to_arity_map = name_to_arity_map
        self.known_missing_types = known_missing_types

        if len(argument_type_categories) != self.name_to_arity_map[predicate]:
            raise ValueError(f'Predicate {predicate} has arity {self.name_to_arity_map[predicate]} but {len(argument_type_categories)} argument types were provided')

    def game_start(self) -> None:
        self.matching_argument_types_count = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            name = extract_predicate_function_name(ast)

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
            for term in terms:
                if term.startswith('?'):
                    if term in context_variables:  
                        term_type_lists.append(context_variables[term].var_types)
                    else:
                        return
                else:
                    term_type_lists.append([term])

            term_categories = []
            for term_type_list in term_type_lists:
                term_type_categories = set()
                for term_type in term_type_list:
                    if term_type not in TYPES_TO_CATEGORIES:
                        if term_type not in self.known_missing_types and not term_type.isnumeric():
                            continue
                            # print(f'Unknown type {term_type_list} not in the types to categories map')
                    else:
                        term_type_categories.add(TYPES_TO_CATEGORIES[term_type])

                term_categories.append(term_type_categories)

            self.matching_argument_types_count += all(self.argument_type_categories[i] in term_categories[i] for i in range(len(term_categories)))


    def game_end(self) -> typing.Union[Number, typing.Sequence[Number], typing.Dict[typing.Any, Number]]:
        return self.matching_argument_types_count


def build_argument_types_fitness_terms(
    predicates: typing.Sequence[str] = COMMON_SENSE_PREDICATES_FUNCTIONS,
    type_categories: typing.Sequence[str] = COMMON_SENSE_TYPE_CATEGORIES,
    predicate_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]] = PREDICATE_FUNCTION_ARITY_MAP) -> typing.Sequence[FitnessTerm]:
    fitness_terms = []

    for predicate in predicates:
        for type_combinations in itertools.product(*([type_categories] * predicate_arity_map[predicate])):  # type: ignore
            fitness_terms.append(PredicateFunctionArgumentTypes(predicate, type_combinations, predicate_arity_map))

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
            ast_str = ast_printer.ast_section_to_string(ast, ast_printer.PREFERENCES_KEY)
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
        super().__init__(section, header)
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
    (SectionMaxDepth, 'setup'): [1, 8.5, 17.5, 25.5],
    (SectionMaxDepth, 'constraints'): [8.5, 15.5, 19.5, 23.5],
    (SectionMaxDepth, 'terminal'): [1, 4.5, 8.5, 11.5],
    (SectionMaxDepth, 'scoring'): [2.5, 4.5, 8.5, 12.5],
    
    (SectionMeanDepth, 'setup'): [1, 4, 9, 12],
    (SectionMeanDepth, 'constraints'): [6.5, 8, 10, 12],
    (SectionMeanDepth, 'terminal'): [1, 2, 3.8, 5.5],
    (SectionMeanDepth, 'scoring'): [1.5, 2.75, 4, 6],

    (SectionNodeCount, 'setup'): [1, 10, 30, 100],
    (SectionNodeCount, 'constraints'): [30, 70, 135, 200],
    (SectionNodeCount, 'terminal'): [1, 5, 25, 55],
    (SectionNodeCount, 'scoring'): [6, 18, 33, 60],
    
}


def build_section_count_fitness_terms(sections: typing.Sequence[str] = SECTION_KEYS,
    term_classes: typing.Sequence[typing.Callable] = (SectionMaxDepth, SectionMeanDepth, SectionNodeCount),
    thresholds: typing.Optional[typing.Dict[typing.Tuple[typing.Callable, str], typing.Sequence[float]]] = SECTION_COUNT_THRESHOLDS,
    ) -> typing.Sequence[FitnessTerm]:

    if thresholds is not None:
        return [term_class(section, thresholds[(term_class, section)]) for term_class in term_classes for section in sections]
    else:
        return [term_class(section) for term_class in term_classes for section in sections]


def build_fitness_featurizer(args) -> ASTFitnessFeaturizer:
    fitness = ASTFitnessFeaturizer()

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

    pref_starts_and_ends_with_once = PrefStartsAndEndsWithOnce()
    fitness.register(pref_starts_and_ends_with_once)

    no_repeated_variables_in_predicate = VariableNotRepeatedInPredicateFunction()
    fitness.register(no_repeated_variables_in_predicate)

    no_nested_logicals = NoNestedLogicals()
    fitness.register(no_nested_logicals)

    no_identical_logical_children = NoIdenticalChildrenInLogicals()
    fitness.register(no_identical_logical_children)

    count_once_per_external_objects_used_correctly = CountOncePerExternalObjectsUsedCorrectly()
    fitness.register(count_once_per_external_objects_used_correctly)
    
    external_forall_used_correctly = ExternalForallUsedCorrectly()
    fitness.register(external_forall_used_correctly)

    pref_forall_used_correctly = PrefForallUsed()
    fitness.register(pref_forall_used_correctly)

    pref_forall_correct_arity = PrefForallCorrectArity()
    fitness.register(pref_forall_correct_arity)

    pref_forall_correct_types = PrefForallCorrectTypes()
    fitness.register(pref_forall_correct_types)

    correct_predicate_arity = CorrectPredicateFunctionArity()
    fitness.register(correct_predicate_arity)

    no_two_number_comparisons = NoTwoNumberOperations()
    fitness.register(no_two_number_comparisons)

    no_count_in_terminal = SectionWithoutPrefCounts(TERMINAL)
    fitness.register(no_count_in_terminal, section_rule=True)

    no_count_in_scoring = SectionWithoutPrefCounts(SCORING)
    fitness.register(no_count_in_scoring, section_rule=True)

    argument_types_fitness_terms = build_argument_types_fitness_terms()
    fitness.register_multiple(argument_types_fitness_terms)

    compositionality_fitness_terms = build_compositionality_fitness_terms()
    fitness.register_multiple(compositionality_fitness_terms)

    section_count_fitness_terms = build_section_count_fitness_terms()
    fitness.register_multiple(section_count_fitness_terms, section_rule=True)

    return fitness
            

def main(args):
    original_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(args.recursion_limit)

    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    aggregator = build_fitness_featurizer(args)

    for test_file in args.test_files:
        for ast in cached_load_and_parse_games_from_file(test_file, grammar_parser, not args.dont_tqdm):
            aggregator.parse(ast, test_file)

    df = aggregator.to_df()
    print(df.groupby('src_file').agg([np.mean, np.std]))

    for src_file in df.src_file.unique():
        zero_std = df[df.src_file == src_file].std(numeric_only=True) == 0
        zero_std_columns = [c for c in zero_std.index if zero_std[c] and not c.startswith('arg_types')]  # type: ignore
        print(f'For src_file {src_file}, the following columns have zero std (excluding arg_types columns): {zero_std_columns}')

    global_zero_std = df.std(numeric_only=True) == 0
    global_zero_std_columns = [c for c in global_zero_std.index if global_zero_std[c] and not c.startswith('arg_types')]  # type: ignore
    print(f'For all src_files, the following columns have zero std (excluding arg_types columns): {global_zero_std_columns}')

    df.to_csv(args.output_path, index_label='Index')    

    sys.setrecursionlimit(original_recursion_limit)


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)

    for test_file in args.test_files:
        if not os.path.exists(test_file):
            raise ValueError(f'File {test_file} does not exist')
    
    main(args)
