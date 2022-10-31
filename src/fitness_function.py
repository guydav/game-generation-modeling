import argparse
from collections import namedtuple, defaultdict
import tatsu
import tatsu.ast
import tatsu.buffering
import tqdm
import pandas as pd
import numpy as np
import os
from re import Pattern
import typing

from parse_dsl import load_tests_from_file
from ast_parser import ASTParser


parser = argparse.ArgumentParser()
DEFAULT_GRAMMAR_FILE = './dsl/dsl.ebnf'
parser.add_argument('-g', '--grammar-file', default=DEFAULT_GRAMMAR_FILE)
DEFAULT_TEST_FILES = (
    # './dsl/problems-few-objects.pddl',
    # './dsl/problems-medium-objects.pddl',
    # './dsl/problems-many-objects.pddl',
    './dsl/interactive-beta.pddl',
)
parser.add_argument('-t', '--test-files', action='append', default=[])
parser.add_argument('-q', '--dont-tqdm', action='store_true')
DEFAULT_OUTPUT_PATH ='./data/fitness_score.csv'
parser.add_argument('-o', '--output-path', default=DEFAULT_OUTPUT_PATH)


ContextDict = typing.Dict[str, typing.Union[str, int, typing.Dict[str, typing.Any]]]


class FitnessTerm:
    rules: typing.Sequence[typing.Union[str, Pattern]]
    header: str

    def __init__(self, rule_or_rules: typing.Union[str, Pattern, typing.Sequence[typing.Union[str, Pattern]]], header: str):
        if isinstance(rule_or_rules, str) or isinstance(rule_or_rules, Pattern):
            rule_or_rules = [rule_or_rules]

        self.rules = rule_or_rules
        self.header = header
    
    def __call__(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], context: ContextDict) -> typing.Optional[float]:
        raise NotImplementedError


DEFAULT_HEADERS = ('src_file', 'game_name', 'domain_name')


class ASTFitnessFunction:
    headers: typing.List[str]
    header_registry: typing.Dict[str, FitnessTerm]
    list_reduce: typing.Callable[[typing.Sequence[float]], float]
    regex_rules: typing.List[typing.Tuple[Pattern, FitnessTerm]]
    rows: typing.List
    rule_registry: typing.Dict[str, typing.List[FitnessTerm]]
    tuple_registry: typing.Dict[str, typing.List[FitnessTerm]] 

    def __init__(self, headers: typing.Sequence[str] = DEFAULT_HEADERS, list_reduce: typing.Callable[[typing.Sequence[float]], float] = np.sum):
        self.headers = list(headers)
        self.list_reduce = list_reduce

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
            if isinstance(rule, Pattern):
                self.regex_rules.append((rule, term))
            else:
                self._register(term, rule, tuple_rule)

        self.header_registry[term.header] = term
        self.headers.append(term.header)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.rows, columns=self.headers)

    def parse(self, ast: typing.Tuple[tatsu.ast.AST, tatsu.ast.AST, tatsu.ast.AST, tatsu.ast.AST], src_file: str):
        row = defaultdict(list)
        row['src_file'].append(os.path.basename(src_file))
        row['game_name'].append(ast[1]["game_name"])  # type: ignore
        row['domain_name'].append(ast[2]["domain_name"])  # type: ignore
        ast = ast[3:]  # type: ignore
        self._parse(ast, row)

        for header in row:
            if row[header]:
                if len(row[header]) == 1:
                    row[header] = row[header][0]
                else:
                    row[header] = self.list_reduce(row[header])  # type: ignore

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

    def _parse(self, ast: typing.Union[str, int, tatsu.buffering.Buffer, tuple, list, tatsu.ast.AST], row: typing.Dict[str, typing.List[float]], 
        context: typing.Optional[ContextDict] = None):
        if context is None:
            context = dict(depth=0)

        if not ast or isinstance(ast, (str, int, tatsu.buffering.Buffer)):
            return

        elif isinstance(ast, (tuple, list)):
            if len(ast) > 0 and isinstance(ast[0], str):
                for tuple_stat in self.tuple_registry[ast[0]]:
                    result = tuple_stat(ast, context)
                    if result:
                        row[tuple_stat.header].append(result)

            [self._parse(element, row, context) for element in ast]

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
                stat_parsers = self.rule_registry[ast.parseinfo.rule]
                for stat in stat_parsers:
                    result = stat(ast, context)
                    if result:
                        row[stat.header].append(result)
                
                for regex_pattern, regex_term in self.regex_rules:
                    if regex_pattern.match(ast.parseinfo.rule):
                        result = regex_term(ast, context)
                        if result:
                            row[regex_term.header].append(result)

            child_context = context.copy()
            child_context['depth'] += 1  # type: ignore

            for child_key in ast:
                if child_key != 'parseinfo':
                    self._parse(ast[child_key], row, child_context)  # type: ignore

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


class VariablesDefinedFitnessTerm(FitnessTerm):
    def __init__(self):
        super().__init__('predicate', 'variables_defined')

    def __call__(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], context: ContextDict):
        if 'variables' not in context:
            return

        fitness = 0

        if isinstance(ast, tatsu.ast.AST):
            args = ast.pred_args

            if args is None:
                return

            if isinstance(args, tatsu.ast.AST):
                args = [args]

            for arg in args:  # type: ignore
                term = arg.term
                if isinstance(term, str) and term.startswith('?') and term in context['variables']:  # type: ignore
                    fitness += 1

        return fitness
        

def build_aggregator(args):
    fitness = ASTFitnessFunction()

    variables_defined = VariablesDefinedFitnessTerm()
    fitness.register(variables_defined)
    

    # length_of_then = StatExtractor('then', 'length_of_then', lambda ast, context: len(ast.then_funcs), lambda x: x)
    # agg.register(length_of_then)

    # num_preferences = StatExtractor('preference', 'num_preferences', lambda ast, context: 1, np.sum)
    # agg.register(num_preferences)

    # def objects_quantified(ast, context=None):
    #     key = 'exists_vars'
    #     if 'forall_vars' in ast:
    #         key = 'forall_vars'

    #     return len(ast[key]['variables'])

    # num_setup_objects_quantified = StatExtractor(
    #     ('setup_exists', 'setup_forall'), 'setup_objects_quantified', 
    #     objects_quantified, lambda x: x)
    # agg.register(num_setup_objects_quantified)

    # num_preference_objects_quantified = StatExtractor(
    #     ('pref_body_exists', 'pref_body_forall', 'pref_forall'), 'preference_objects_quantified', 
    #     objects_quantified, lambda x: x)
    # agg.register(num_preference_objects_quantified)

    # terminal_clause_exists = StatExtractor(
    #     'terminal_comp', 'terminal_exists', lambda ast, context: True, all
    # )
    # agg.register(terminal_clause_exists)

    # def objects_referenced(ast, context=None):
    #     results = defaultdict(lambda: 0)

    #     if ast.parseinfo.rule == 'predicate':
    #         if 'pred_args' in ast:
    #             # single pred arg
    #             if isinstance(ast.pred_args, str) and not ast.pred_args.startswith('?'):
    #                 results[ast.pred_args] += 1

    #             # multiple pred args
    #             filtered_args = [arg for arg in ast.pred_args if isinstance(arg, str) and not arg.startswith('?')]
    #             for arg in filtered_args:
    #                 results[arg] += 1

    #     elif ast.parseinfo.rule == 'pref_name_and_types':
    #         if 'object_types' in ast:
    #             # single object type
    #             if isinstance(ast.object_types, tatsu.ast.AST):
    #                 results[ast.object_types.type_name] += 1
                
    #             # multiple object types
    #             else:
    #                 for type_name in [t.type_name for t in ast.object_types if 'type_name' in t]:
    #                     results[type_name] += 1
        
    #     else:
    #         key = 'exists_vars'
    #         if 'forall_vars' in ast:
    #             key = 'forall_vars'
            
    #         for quantification in ast[key]['variables']:
    #             # single type
    #             if isinstance(quantification['var_type'], str):
    #                 results[quantification['var_type']] += 1
                
    #             # either types
    #             else:
    #                 for name in quantification['var_type'].type_names:
    #                     results[name] += 1

    #     return results

    # def aggregate_count_dicts(count_dicts):
    #     results = defaultdict(lambda: 0)
    #     for cd in count_dicts:
    #         for key in cd:
    #             results[key] += cd[key]
    #     return dict(results)

    # object_types_referenced = StatExtractor(
    #     ('setup_exists', 'setup_forall', 'setup_exists_predicate', 'setup_forall_predicate',
    #     'pref_body_exists', 'pref_body_forall', 'pref_forall',
    #     'pref_predicate_exists', 'pref_predicate_forall', 
    #     'pref_name_and_types', 'predicate'),
    #     'object_types_referenced', objects_referenced, aggregate_count_dicts
    # )
    # agg.register(object_types_referenced)

    # predicates_referenced = StatExtractor(
    #     'predicate', 'predicates_referenced', lambda ast, context: {ast.pred_name: 1}, aggregate_count_dicts
    # )
    # agg.register(predicates_referenced)

    # max_depth = StatExtractor('predicate', 'max_depth', lambda ast, context: context['depth'], max)
    # agg.register(max_depth)

    # total_ast_nodes = StatExtractor(re.compile('.*'), 'ast_nodes', lambda ast, context: 1, np.sum)
    # agg.register(total_ast_nodes)

    # ast_node_counter = ASTNodeCounter()

    # def count_setup_nodes(ast, context=None):
    #     if isinstance(ast[1], tatsu.ast.AST):
    #         return ast_node_counter(ast[1], zero_count=True)

    #     return 0

    # setup_nodes = StatExtractor('(:setup', 'setup_nodes', count_setup_nodes, np.sum)
    # agg.register(setup_nodes, tuple_rule=True)

    # def map_types_to_predicates(ast, context):
    #     if context is None or not isinstance(context, dict):
    #         raise ValueError(f'Expected a context dict, received: {context}')
    #     type_to_pred_counts = defaultdict(lambda: defaultdict(lambda: 0))
    #     inner_counts = []

    #     variables = context['variables'] if 'variables' in context else {}
    #     pred_name = ast.pred_name
    #     pred_args = ast.pred_args
    #     if isinstance(pred_args, str): pred_args = [pred_args]

    #     for arg in pred_args:
    #         if isinstance(arg, tatsu.ast.AST) and arg.parseinfo.rule == 'predicate':
    #             inner_counts.append(map_types_to_predicates(arg, context))

    #         elif arg.startswith('?'):
    #             if arg not in variables:
    #                 raise ValueError(f'Encountered undefined argument {arg} in AST: {ast}')

    #             for type_name in variables[arg]:
    #                 type_to_pred_counts[type_name][pred_name] += 1

    #         else:
    #             type_to_pred_counts[arg][pred_name] += 1

    #     if len(inner_counts) > 0:
    #         inner_counts.append(type_to_pred_counts)
    #         return aggregate_nested_count_dicts(inner_counts)

    #     return type_to_pred_counts

    # def aggregate_nested_count_dicts(nested_count_dicts):
    #     results = defaultdict(lambda: defaultdict(lambda: 0))
    #     for nested_count_dict in nested_count_dicts:
    #         for outer_key, inner_dict in nested_count_dict.items():
    #             for inner_key, count in inner_dict.items():
    #                 results[outer_key][inner_key] += count

    #     return {outer_key: dict(inner_dict) for outer_key, inner_dict in results.items()}
    

    # type_to_pred_counts = StatExtractor('predicate', 'type_to_pred_counts', map_types_to_predicates, aggregate_nested_count_dicts)
    # agg.register(type_to_pred_counts)

    return fitness
            

def main(args):
    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    aggregator = build_aggregator(args)

    for test_file in args.test_files:
        test_cases = load_tests_from_file(test_file)

        if not args.dont_tqdm:
            test_cases = tqdm.tqdm(test_cases)

        for test_case in test_cases:
            ast = grammar_parser.parse(test_case)
            aggregator.parse(ast, test_file)

    df = aggregator.to_df()
    print(df.head)
    # df.to_csv(args.output_path, index_label='Index')    


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)
    
    main(args)
