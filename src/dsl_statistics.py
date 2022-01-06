import argparse
from collections import namedtuple, defaultdict
import tatsu
import tqdm
import pandas as pd
import numpy as np
import os
import re

from parse_dsl import load_tests_from_file


parser = argparse.ArgumentParser()
DEFAULT_GRAMMAR_FILE = './dsl/dsl.ebnf'
parser.add_argument('-g', '--grammar-file', default=DEFAULT_GRAMMAR_FILE)
DEFAULT_TEST_FILES = (
    './dsl/problems-few-objects.pddl',
    './dsl/problems-medium-objects.pddl',
    './dsl/problems-many-objects.pddl',
    './dsl/interactive-beta.pddl',
)
parser.add_argument('-t', '--test-files', action='append', default=[])
parser.add_argument('-q', '--dont-tqdm', action='store_true')
DEFAULT_OUTPUT_PATH ='./data/dsl_statistics_interactive.csv'
parser.add_argument('-o', '--output-path', default=DEFAULT_OUTPUT_PATH)


class StatExtractor:
    def __init__(self, rule_or_rules, header, extract, aggregate=None):
        self.rule_or_rules = rule_or_rules
        self.header = header
        self.extract = extract
        self.aggregate = aggregate

class ASTStatisticsAggregator:
    def __init__(self):
        self.rule_registry = defaultdict(list)
        self.regex_rules = []
        self.header_registry = dict()
        self.headers = ['src_file', 'game_name', 'domain_name']
        self.rows = []

    def _register(self, stat, rule):
        self.rule_registry[rule].append(stat)

    def register(self, stat):
        if isinstance(stat.rule_or_rules, re.Pattern):
            self.regex_rules.append(stat)

        else:
            if isinstance(stat.rule_or_rules, str):
                self._register(stat, stat.rule_or_rules)
            else:
                for rule in stat.rule_or_rules:
                    self._register(stat, rule)

        self.header_registry[stat.header] = stat
        self.headers.append(stat.header)

    def to_df(self):
        return pd.DataFrame.from_records(self.rows, columns=self.headers)

    def parse(self, ast, src_file):
        row = defaultdict(list)
        row['src_file'].append(os.path.basename(src_file))
        row['game_name'].append(ast[1]["game_name"])
        row['domain_name'].append(ast[2]["domain_name"])
        ast = ast[3:]
        self._parse(ast, row)

        for header in row:
            if row[header]:
                if header in self.header_registry and self.header_registry[header].aggregate is not None:
                    row[header] = self.header_registry[header].aggregate(row[header])
                elif len(row[header]) == 1:
                    row[header] = row[header][0]

        self.rows.append(row)

    def _parse(self, ast, row, depth=0):
        if not ast or isinstance(ast, (str, int, tatsu.buffering.Buffer)):
            return

        elif isinstance(ast, (tuple, list)):
            [self._parse(element, row, depth) for element in ast]

        elif isinstance(ast, tatsu.ast.AST):
            if ast.parseinfo is not None:
                stat_parsers = self.rule_registry[ast.parseinfo.rule]
                for stat in stat_parsers:
                    result = stat.extract(ast, depth)
                    if result:
                        row[stat.header].append(result)
                
                for regex_stat in self.regex_rules:
                    if regex_stat.rule_or_rules.match(ast.parseinfo.rule):
                        result = regex_stat.extract(ast, depth)
                        if result:
                            row[regex_stat.header].append(result)

            [self._parse(element, row, depth + 1) for element in ast.values()]

        else:
            print(f'Encountered AST element with unrecognized type: {ast} of type {type(ast)}')


def build_aggregator(args):
    agg = ASTStatisticsAggregator()

    length_of_then = StatExtractor('then', 'length_of_then', lambda ast, depth: len(ast.then_funcs), lambda x: x)
    agg.register(length_of_then)

    num_preferences = StatExtractor('preference', 'num_preferences', lambda ast, depth: 1, np.sum)
    agg.register(num_preferences)

    def objects_quantified(ast, depth=0):
        key = 'exists_vars'
        if 'forall_vars' in ast:
            key = 'forall_vars'

        return len(ast[key]['variables'])

    num_setup_objects_quantified = StatExtractor(
        ('setup_exists', 'setup_forall'), 'setup_objects_quantified', 
        objects_quantified, lambda x: x)
    agg.register(num_setup_objects_quantified)

    num_preference_objects_quantified = StatExtractor(
        ('pref_body_exists', 'pref_body_forall', 'pref_forall'), 'preference_objects_quantified', 
        objects_quantified, lambda x: x)
    agg.register(num_preference_objects_quantified)

    terminal_clause_exists = StatExtractor(
        'terminal_comp', 'terminal_exists', lambda ast, depth: True, all
    )
    agg.register(terminal_clause_exists)

    def objects_referenced(ast, depth=None):
        key = 'exists_vars'
        if 'forall_vars' in ast:
            key = 'forall_vars'

        results = defaultdict(lambda: 0)
        for quantification in ast[key]['variables']:
            if isinstance(quantification['var_type'], str):
                results[quantification['var_type']] += 1
            else:
                for name in quantification['var_type'].type_names:
                    results[name] += 1

        return results

    def aggregate_count_dicts(count_dicts):
        results = count_dicts[0]
        for cd in count_dicts[1:]:
            for key in cd:
                results[key] += cd[key]
        return dict(results)

    object_types_quantified = StatExtractor(
        ('setup_exists', 'setup_forall', 'pref_body_exists', 'pref_body_forall', 'preference_forall'),
        'object_types_quantified', objects_referenced, aggregate_count_dicts
    )
    agg.register(object_types_quantified)

    max_depth = StatExtractor('predicate', 'max_depth', lambda ast, depth: depth, max)
    agg.register(max_depth)

    total_ast_nodes = StatExtractor(re.compile('.*'), 'ast_nodes', lambda ast, depth: 1, np.sum)
    agg.register(total_ast_nodes)

    return agg
            

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
    df.to_csv(args.output_path, index_label='Index')    


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)
    
    main(args)
