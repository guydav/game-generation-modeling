from abc import ABC, abstractmethod
import argparse
from collections import namedtuple, defaultdict
import tatsu
import tatsu.ast
import tatsu.buffering
import tqdm
import pandas as pd
import numpy as np
import os
import re
import typing

from ast_utils import cached_load_and_parse_games_from_file
from ast_parser import ASTParser
from ast_to_latex_doc import TYPE_RULES


parser = argparse.ArgumentParser()
DEFAULT_GRAMMAR_FILE = './dsl/dsl.ebnf'
parser.add_argument('-g', '--grammar-file', default=DEFAULT_GRAMMAR_FILE)
DEFAULT_TEST_FILES = (
    # './dsl/problems-few-objects.pddl',
    # './dsl/problems-medium-objects.pddl',
    # './dsl/problems-many-objects.pddl',
    './dsl/interactive-beta.pddl',
    './dsl/ast-mle-samples.pddl',
    './dsl/ast-regrwoth-samples.pddl',
)
parser.add_argument('-t', '--test-files', action='append', default=[])
parser.add_argument('-q', '--dont-tqdm', action='store_true')
DEFAULT_OUTPUT_PATH ='./data/fitness_scores.csv'
parser.add_argument('-o', '--output-path', default=DEFAULT_OUTPUT_PATH)


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


class ASTFitnessFunction:
    headers: typing.List[str]
    header_registry: typing.Dict[str, FitnessTerm]
    list_reduce: typing.Callable[[typing.Sequence[Number]], Number]
    regex_rules: typing.List[typing.Tuple[re.Pattern, FitnessTerm]]
    rows: typing.List
    rule_registry: typing.Dict[str, typing.List[FitnessTerm]]
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
        self.regex_rules = []
        self.header_registry = dict()

        self.rows = []

    def _register(self, term: FitnessTerm, rule: str, tuple_rule: bool = False) -> None:
        if tuple_rule:
            self.tuple_registry[rule].append(term)
        else:
            self.rule_registry[rule].append(term)

    def register(self, term: FitnessTerm, tuple_rule: bool = False) -> None:
        for rule in term.rules:
            if isinstance(rule, re.Pattern):
                self.regex_rules.append((rule, term))
            else:
                self._register(term, rule, tuple_rule)

        self.header_registry[term.header] = term
        self.headers.append(term.header)

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
            context = dict(depth=0)

        if not ast or isinstance(ast, (str, int, tatsu.buffering.Buffer)):
            return

        elif isinstance(ast, (tuple, list)):
            if len(ast) > 0 and isinstance(ast[0], str):
                # check if should update the section key
                if ast[0][2:] in self.section_keys:
                    context['section'] = ast[0][2:]

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
                context_vars = typing.cast(dict, context['variables']) if 'variables' in context else {}
                self._extract_variables(ast, vars_key, context_vars) 
                context = context.copy()
                context['variables'] = context_vars

            if ast.parseinfo is not None:
                rule = ast.parseinfo.rule
                stat_parsers = self.rule_registry[rule]
                for stat in stat_parsers:
                    stat.update(ast, rule, context)
                
                for regex_pattern, regex_term in self.regex_rules:
                    if regex_pattern.match(rule):
                        regex_term.update(ast, rule, context)

            child_context = context.copy()
            child_context['depth'] += 1  # type: ignore

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


class VariablesDefinedFitnessTerm(FitnessTerm):
    defined_count: int = 0
    undefined_count: int = 0

    def __init__(self):
        super().__init__('predicate', 'variables_defined')

    def game_start(self) -> None:
        self.defined_count = 0
        self.undefined_count = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if 'variables' not in context:
            return

        if isinstance(ast, tatsu.ast.AST):
            for term in _extract_predicate_terms(ast):  # type: ignore
                if isinstance(term, str) and term.startswith('?'):
                    if term in context['variables']:  # type: ignore
                        self.defined_count += 1
                    else:
                        self.undefined_count += 1

    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
        if self.defined_count == 0:
            return 0

        return self.defined_count / (self.defined_count + self.undefined_count)
        

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
        if 'section' not in context:
            raise ValueError('Section not found in context', context)

        if context['section'] == SETUP:
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

                if any([isinstance(child, tatsu.ast.AST) and child.parseinfo.rule == rule for child in children]):  # type: ignore
                    self.nested_logicals += 1

    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
        if self.total_logicals == 0:
            return 0

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
                if object_types is not None:
                    self.prefs_used_as_pref_forall_prefs.add(pref_name)

    def game_end(self) -> typing.Optional[typing.Union[Number, typing.Sequence[Number]]]:
        if len(self.pref_forall_prefs) == 0 and len(self.prefs_used_as_pref_forall_prefs) == 0:
            return 1

        return len(self.pref_forall_prefs.intersection(self.prefs_used_as_pref_forall_prefs)) / len(self.pref_forall_prefs.union(self.prefs_used_as_pref_forall_prefs))


def build_aggregator(args):
    fitness = ASTFitnessFunction()

    variables_defined = VariablesDefinedFitnessTerm()
    fitness.register(variables_defined)
    
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

    # TODO: common sense? predicate role-filler pairs

    # TODO: recurring structures -- generate features for top-k from previous analysis

    # TODO: length/depth/width features, perhaps by section, perhaps discretized by length

    return fitness
            

def main(args):
    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    aggregator = build_aggregator(args)

    for test_file in args.test_files:
        for ast in cached_load_and_parse_games_from_file(test_file, grammar_parser, not args.dont_tqdm):
            aggregator.parse(ast, test_file)

    df = aggregator.to_df()
    print(df.groupby('src_file').agg([np.mean, np.std]))
    df.to_csv(args.output_path, index_label='Index')    


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)
    
    main(args)
