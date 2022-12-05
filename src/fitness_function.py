from abc import ABC, abstractmethod
import argparse
from collections import namedtuple, defaultdict
import itertools
import tatsu
import tatsu.ast
import tatsu.buffering
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
from ast_to_latex_doc import TYPE_RULES
from room_and_object_types import *


parser = argparse.ArgumentParser()
DEFAULT_GRAMMAR_FILE = './dsl/dsl.ebnf'
parser.add_argument('-g', '--grammar-file', default=DEFAULT_GRAMMAR_FILE)
DEFAULT_TEST_FILES = (
    './dsl/interactive-beta.pddl',
    './dsl/ast-real-regrowth-samples.pddl',
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


ContextDict = typing.Dict[str, typing.Union[str, int, typing.Dict[str, typing.Any]]]
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
    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
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


class ASTFitnessFunction:
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
        return pd.DataFrame.from_records(self.rows, columns=self.headers)

    def parse(self, ast: typing.Tuple[tatsu.ast.AST, tatsu.ast.AST, tatsu.ast.AST, tatsu.ast.AST], src_file: str):
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
            if term_result is not None:
                if isinstance(term_result, (int, float)):
                    row[header] = term_result
                else:
                    row[header] = self.list_reduce(row[header])

        self.rows.append(row)

    def _extract_variables(self, ast: tatsu.ast.AST, vars_key: str, context_vars: typing.Dict[str, typing.List[str]]) -> None:
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
                context_vars[var_name] = var_type  # type: ignore

    def _parse(self, ast: typing.Union[str, int, tatsu.buffering.Buffer, tuple, list, tatsu.ast.AST],
        context: typing.Optional[ContextDict] = None):
        if context is None:
            context = {DEPTH_CONTEXT_KEY: 0}

        if not ast or isinstance(ast, (str, int, tatsu.buffering.Buffer)):
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
                self._extract_variables(ast, vars_key, context_vars) 
                context = context.copy()
                context[VARIABLES_CONTEXT_KEY] = context_vars

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
                child_context['external_forall'] = ast.parseinfo.rule  

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


def _extract_predicate_terms(ast: tatsu.ast.AST) -> typing.List[str]:
    args = ast.pred_args

    if args is None:
        return []

    if isinstance(args, tatsu.ast.AST):
        args = [args]

    return [str(arg.term) for arg in args]


class VariableBasedFitnessTerm(FitnessTerm):
    def __init__(self, header: str):
        super().__init__(('predicate_term', 'function_term'), header)
        self.variables = set()

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if VARIABLES_CONTEXT_KEY not in context:
            return

        if isinstance(ast.term, str) and ast.term.startswith('?'):  # type: ignore
            self._inner_update(ast.term, context[VARIABLES_CONTEXT_KEY])  # type: ignore

    @abstractmethod
    def _inner_update(self, term: str, variables: typing.Dict[str, typing.List[str]]):
        pass
        

class AllVariablesDefined(VariableBasedFitnessTerm):
    defined_count: int = 0
    undefined_count: int = 0

    def __init__(self):
        super().__init__('all_variables_defined')

    def game_start(self) -> None:
        self.defined_count = 0
        self.undefined_count = 0

    def _inner_update(self, term: str, variables: typing.Dict[str, typing.List[str]]):
        if term in variables: 
            self.defined_count += 1
        else:
            self.undefined_count += 1

    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
        if self.defined_count == 0:
            return 0

        return self.defined_count / (self.defined_count + self.undefined_count)


class AllVariablesUsed(VariableBasedFitnessTerm):
    defined_variables: typing.Set[str]
    used_variables: typing.Set[str]

    def __init__(self):
        super().__init__('all_variables_used')

    def game_start(self) -> None:
        self.defined_variables = set()
        self.used_variables = set()

    def _inner_update(self, term: str, variables: typing.Dict[str, typing.List[str]]):
        # TODO: this is incomplete, and can fail in the case of the same variable being
        # defined in multiple contexts, but only used in one or some of them
        # If this appears to happen, we can redefine this to be a bit smarter

        self.defined_variables.update(variables)  # type: ignore
        if term in self.defined_variables:
            self.used_variables.add(term)

    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
        if len(self.defined_variables) == 0:
            return 0

        return len(self.used_variables) / len(self.defined_variables)
        

class AllPreferencesUsed(FitnessTerm):
    defined_preferences: typing.Set[str] = set()
    used_preferences: typing.Set[str] = set()

    def __init__(self):
        super().__init__(('preference', re.compile('count.*')), 'all_preferences_used')

    def game_start(self) -> None:
        self.defined_preferences = set()
        self.used_preferences = set()

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if rule == 'preference':
            self.defined_preferences.add(ast.pref_name)  # type: ignore

        else:
            self.used_preferences.add(ast.name_and_types.pref_name)  # type: ignore 

    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
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

    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
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

    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
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

    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
        if self.total_prefs == 0:
            return 0

        return self.prefs_start_and_end_with_once / self.total_prefs


class VariableNotRepeatedInPredicate(FitnessTerm):
    total_predicates: int = 0
    predicates_with_repeated_variables: int = 0

    def __init__(self):
        super().__init__('predicate', 'variable_not_repeated')

    def game_start(self) -> None:
        self.total_predicates = 0
        self.predicates_with_repeated_variables = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            self.total_predicates += 1
            predicate_vars = [t for t in _extract_predicate_terms(ast) if isinstance(t, str) and t.startswith('?')]
            self.predicates_with_repeated_variables += 1 if len(predicate_vars) != len(set(predicate_vars)) else 0

    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
        if self.total_predicates == 0:
            return 0

        return 1 - (self.predicates_with_repeated_variables / self.total_predicates)


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

    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
        if self.total_logicals == 0:
            # TODO: should this return a NaN? If so, we should thin kabout how to handle them
            return 1

        return 1 - (self.nested_logicals / self.total_logicals)


class PrefForallUsedCorrectly(FitnessTerm):
    pref_forall_prefs: typing.Set[str] = set()
    prefs_used_as_pref_forall_prefs: typing.Set[str] = set()

    def __init__(self):
        super().__init__(('pref_forall', re.compile('count.*')), 'pref_forall_correct')

    def game_start(self) -> None:
        self.pref_forall_prefs = set()
        self.prefs_used_as_pref_forall_prefs = set()

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            if rule == 'pref_forall':
                preferences = ast.forall_pref.preferences  # type: ignore
                if isinstance(preferences, tatsu.ast.AST):
                    preferences = [preferences]

                self.pref_forall_prefs.update([pref.pref_name for pref in preferences])  # type: ignore

            else:   # count*
                pref_name = ast.name_and_types['pref_name']  # type: ignore
                object_types = ast.name_and_types['object_types']  # type: ignore
                if object_types is not None or 'external_forall' in context:
                    self.prefs_used_as_pref_forall_prefs.add(pref_name)

    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
        if len(self.pref_forall_prefs) == 0 and len(self.prefs_used_as_pref_forall_prefs) == 0:
            return 1

        return len(self.pref_forall_prefs.intersection(self.prefs_used_as_pref_forall_prefs)) / len(self.pref_forall_prefs.union(self.prefs_used_as_pref_forall_prefs))


PREDICATE_ARITY_MAP = {
    'above': 2, 'adjacent': 2, 'adjacent_side': (3, 4), 'agent_crouches': 0, 'agent_holds': 1,
    'between': 3, 'broken': 1, 'equal_x_position': 2, 'equal_z_position': 2, 'faces': 2,
    'game_over': 0, 'game_start': 0, 'in':2, 'in_motion': 1, 'is_setup_object': 1, 
    'object_orientation': 2, 'on': 2, 'open': 1, 'opposite': 2, 'rug_color_under': 2, 
    'same_color': 2, 'same_object': 2, 'same_type': 2, 'toggled_on': 1, 'touch': 2
}


def _extract_n_args(ast: tatsu.ast.AST):
    if 'pred_args' in ast:
        if isinstance(ast.pred_args, tatsu.ast.AST):
            return 1
        else:
            return len(ast.pred_args)  # type: ignore

    return 0


class CorrectPredicateArity(FitnessTerm):
    total_predicates: int = 0
    predicate_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]] = {}
    predicates_with_wrong_arity: int = 0

    def __init__(self, predicate_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]] = PREDICATE_ARITY_MAP):
        super().__init__('predicate', 'correct_predicate_arity')
        self.predicate_arity_map = predicate_arity_map

    def game_start(self) -> None:
        self.total_predicates = 0
        self.predicates_with_wrong_arity = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            self.total_predicates += 1

            if ast.pred_name not in self.predicate_arity_map:
                raise ValueError(f'Predicate {ast.pred_name} not in predicate arity map')

            n_args = _extract_n_args(ast)
            arity = self.predicate_arity_map[ast.pred_name]  # type: ignore

            if isinstance(arity, int):
                if n_args != arity:
                    self.predicates_with_wrong_arity += 1
            
            elif n_args not in arity:
                self.predicates_with_wrong_arity += 1

    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
        if self.total_predicates == 0:
            return 0

        return 1 - (self.predicates_with_wrong_arity / self.total_predicates)


TWO_ARG_COMPARISON_RULE = 'two_arg_comparison'
MULTIPLE_ARG_COMPARISON_RULE = 'multiple_args_equal_comparison'


class NoTwoNumberComparisons(FitnessTerm):
    total_comparisons: int = 0
    two_number_comparisons: int = 0
    
    def __init__(self):
        super().__init__((TWO_ARG_COMPARISON_RULE, MULTIPLE_ARG_COMPARISON_RULE), 'no_two_number_comparisons')

    def game_start(self) -> None:
        self.total_comparisons = 0
        self.two_number_comparisons = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            self.total_comparisons += 1

            if rule == TWO_ARG_COMPARISON_RULE:
                if isinstance(ast.arg_1, str) and isinstance(ast.arg_2, str):
                    self.two_number_comparisons += 1

            elif rule == MULTIPLE_ARG_COMPARISON_RULE:
                args = ast.equal_comp_args  
                if all(isinstance(arg, str) for arg in args) or len(args) <= 1:  # type: ignore
                    self.two_number_comparisons += 1

    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
        if self.total_comparisons == 0:
            return 1

        return 1 - (self.two_number_comparisons / self.total_comparisons)
    

class NoVariableTwiceInPredicate(FitnessTerm):
    total_predicates: int = 0
    predicates_with_variable_twice: int = 0

    def __init__(self):
        super().__init__('predicate', 'no_variable_twice_in_predicate')

    def game_start(self) -> None:
        self.total_predicates = 0
        self.predicates_with_variable_twice = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            self.total_predicates += 1

            if 'pred_args' in ast:
                if isinstance(ast.pred_args, tatsu.ast.AST):
                    return

                terms = _extract_predicate_terms(ast)
                if len(terms) != len(set(terms)):
                    self.predicates_with_variable_twice += 1


    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
        if self.total_predicates == 0:
            return 0

        return 1 - (self.predicates_with_variable_twice / self.total_predicates)


COMMON_SENSE_PREDICATES = ('adjacent', 'agent_holds', 'in', 'in_motion', 'on', 'touch')
COMMON_SENSE_TYPE_CATEGORIES = list(CATEGORIES_TO_TYPES.keys())
COMMON_SENSE_TYPE_CATEGORIES.remove(EMPTY_OBJECT)
KNOWN_MISSING_TYPES = ('back', 'front', 'front_left_corner', 'left', 'right', 'sideways', 'upright', 'upside_down')


class PredicateArgumentTypes(FitnessTerm):
    argument_type_categories: typing.Sequence[str]
    matching_argument_types_count: int = 0
    predicate: str
    predicate_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]]

    def __init__(self, predicate: str, argument_type_categories: typing.Sequence[str], 
        predicate_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]] = PREDICATE_ARITY_MAP,
        known_missing_types: typing.Sequence[str] = KNOWN_MISSING_TYPES):

        super().__init__('predicate', f'pred_arg_types_{predicate}_{"_".join(argument_type_categories)}')
        self.predicate = predicate
        self.argument_type_categories = argument_type_categories
        self.predicate_arity_map = predicate_arity_map
        self.known_missing_types = known_missing_types

        if len(argument_type_categories) != self.predicate_arity_map[predicate]:
            raise ValueError(f'Predicate {predicate} has arity {self.predicate_arity_map[predicate]} but {len(argument_type_categories)} argument types were provided')

    def game_start(self) -> None:
        self.matching_argument_types_count = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            if ast.pred_name not in self.predicate_arity_map:
                raise ValueError(f'Predicate {ast.pred_name} not in predicate arity map')

            if ast.pred_name != self.predicate:
                return

            n_args = _extract_n_args(ast)
            arity = self.predicate_arity_map[ast.pred_name]  # type: ignore

            if isinstance(arity, int):
                if n_args != arity:
                    return
            
            elif n_args not in arity:
                return

            terms = _extract_predicate_terms(ast)
            term_type_lists = []

            context_variables = typing.cast(typing.Dict[str, typing.Dict[str, typing.Any]], context[VARIABLES_CONTEXT_KEY]) if VARIABLES_CONTEXT_KEY in context else {}
            for term in terms:
                if term.startswith('?'):
                    if term in context_variables:  
                        term_type_lists.append(context_variables[term])
                    else:
                        return
                else:
                    term_type_lists.append([term])

            term_categories = []
            for term_type_list in term_type_lists:
                term_type_categories = set()
                for term_type in term_type_list:
                    if term_type not in TYPES_TO_CATEGORIES:
                        if term_type not in self.known_missing_types:
                            print(f'Unknown type {term_type_list} not in the types to categories map')
                            # raise ValueError(f'Unknown type {term_type_list} not in the types to categories map')
                    else:
                        term_type_categories.add(TYPES_TO_CATEGORIES[term_type])

                term_categories.append(term_type_categories)

            self.matching_argument_types_count += all(self.argument_type_categories[i] in term_categories[i] for i in range(len(term_categories)))


    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
        return self.matching_argument_types_count


def build_predicate_argument_types_fitness_terms(
    predicates: typing.Sequence[str] = COMMON_SENSE_PREDICATES,
    type_categories: typing.Sequence[str] = COMMON_SENSE_TYPE_CATEGORIES,
    predicate_arity_map: typing.Dict[str, typing.Union[int, typing.Tuple[int, ...]]] = PREDICATE_ARITY_MAP) -> typing.Sequence[FitnessTerm]:
    fitness_terms = []

    for predicate in predicates:
        for type_combinations in itertools.product(*([type_categories] * predicate_arity_map[predicate])):  # type: ignore
            fitness_terms.append(PredicateArgumentTypes(predicate, type_combinations, predicate_arity_map))

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
        self.args_pattern = re.compile(PREDICATE_ARGS_PATTERN)

    def game_start(self) -> None:
        self.structure_count = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            ast_str = ast_printer.ast_section_to_string(ast, ast_printer.PREFERENCES_KEY)
            for pred_args in self.args_pattern.findall(ast_str):
                ast_str = ast_str.replace(pred_args, ' '.join(map(lambda x: self.variable_replacement, pred_args.split(" "))), 1)
                
            self.structure_count += ast_str == self.structure_str

    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
        return self.structure_count


def build_compositionality_fitness_terms(
    compositionality_structures: typing.Sequence[str] = COMPOSITIONALITY_STRUCTURES, 
    variable_replacement: str = COMPOSITIONALITY_VARIABLE_REPLACEMENT) -> typing.Sequence[FitnessTerm]:

    return [CompositionalityStructureCounter(structure, i, variable_replacement) for i, structure in enumerate(compositionality_structures)]


class SectionMaxDepth(FitnessTerm):
    max_depth: int = 0

    def __init__(self, section: str):
        super().__init__(section, f'max_depth_{section}')
        self.max_depth = 0

    def game_start(self) -> None:
        self.max_depth = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if DEPTH_CONTEXT_KEY in context:
            self.max_depth = max(self.max_depth, context[DEPTH_CONTEXT_KEY])  # type: ignore

    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
        return self.max_depth


class SectionMeanDepth(FitnessTerm):
    depths: typing.List[int] = []

    def __init__(self, section: str):
        super().__init__(section, f'mean_depth_{section}')
        self.depths = []

    def game_start(self) -> None:
        self.depths = []

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if DEPTH_CONTEXT_KEY in context:
            self.depths.append(context[DEPTH_CONTEXT_KEY])  # type: ignore

    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
        if len(self.depths) == 0:
            return 0

        return sum(self.depths) / len(self.depths)


class SectionNodeCount(FitnessTerm):
    node_count: int = 0

    def __init__(self, section: str):
        super().__init__(section, f'node_count_{section}')
        self.node_count = 0

    def game_start(self) -> None:
        self.node_count = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        self.node_count += 1

    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
        return self.node_count


def build_section_count_fitness_terms(sections: typing.Sequence[str] = SECTION_KEYS,
    term_classes: typing.Sequence[typing.Callable] = (SectionMaxDepth, SectionMeanDepth, SectionNodeCount)
    ) -> typing.Sequence[FitnessTerm]:

    return [term_class(section) for term_class in term_classes for section in sections]


def build_aggregator(args):
    fitness = ASTFitnessFunction()

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

    no_repeated_variables_in_predicate = VariableNotRepeatedInPredicate()
    fitness.register(no_repeated_variables_in_predicate)

    no_nested_logicals = NoNestedLogicals()
    fitness.register(no_nested_logicals)

    pref_forall_used_correctly = PrefForallUsedCorrectly()
    fitness.register(pref_forall_used_correctly)

    correct_predicate_arity = CorrectPredicateArity()
    fitness.register(correct_predicate_arity)

    no_two_number_comparisons = NoTwoNumberComparisons()
    fitness.register(no_two_number_comparisons)

    no_variable_twice_in_predicate = NoVariableTwiceInPredicate()
    fitness.register(no_variable_twice_in_predicate)

    predicate_argument_types_fitness_terms = build_predicate_argument_types_fitness_terms()
    fitness.register_multiple(predicate_argument_types_fitness_terms)

    compositionality_fitness_terms = build_compositionality_fitness_terms()
    fitness.register_multiple(compositionality_fitness_terms)

    # TODO: break these down by features for different lengths?
    section_count_fitness_terms = build_section_count_fitness_terms()
    fitness.register_multiple(section_count_fitness_terms, section_rule=True)

    return fitness
            

def main(args):
    original_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(args.recursion_limit)

    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    aggregator = build_aggregator(args)

    for test_file in args.test_files:
        for ast in cached_load_and_parse_games_from_file(test_file, grammar_parser, not args.dont_tqdm):
            aggregator.parse(ast, test_file)

    df = aggregator.to_df()
    print(df.groupby('src_file').agg([np.mean, np.std]))
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
