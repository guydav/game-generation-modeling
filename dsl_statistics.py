import argparse
from collections import namedtuple, defaultdict
import tatsu
import tqdm
import pandas as pd
import numpy as np

from parse_dsl import load_tests_from_file


parser = argparse.ArgumentParser()
DEFAULT_GRAMMAR_FILE = './dsl.ebnf'
parser.add_argument('-g', '--grammar-file', default=DEFAULT_GRAMMAR_FILE)
DEFAULT_TEST_FILES = (
    './problems-few-objects.pddl',
    './problems-medium-objects.pddl',
    './problems-many-objects.pddl'
)
parser.add_argument('-t', '--test-files', action='append', default=[])
parser.add_argument('-q', '--dont-tqdm', action='store_true')
DEFAULT_OUTPUT_PATH ='./dsl_statistics.csv'
parser.add_argument('-o', '--output-path', default=DEFAULT_OUTPUT_PATH)


ASTStat = namedtuple('ASTStat', ('rule_or_rules', 'header', 'method', ))

class StatExtractor:
    def __init__(self, rule_or_rules, header, extract, aggregate=None):
        self.rule_or_rules = rule_or_rules
        self.header = header
        self.extract = extract
        self.aggregate = aggregate

class ASTStatisticsAggregator:
    def __init__(self):
        self.rule_registry = defaultdict(list)
        self.header_registry = dict()
        self.headers = ['game_name', 'domain_name']
        self.rows = []

    def _register(self, stat, rule):
        self.rule_registry[rule].append(stat)

    def register(self, stat):
        if isinstance(stat.rule_or_rules, str):
            self._register(stat, stat.rule_or_rules)
        else:
            for rule in stat.rule_or_rules:
                self._register(stat, rule)

        self.header_registry[stat.header] = stat
        self.headers.append(stat.header)

    def to_df(self):
        return pd.DataFrame.from_records(self.rows, columns=self.headers)

    def parse(self, ast):
        row = defaultdict(list)
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

    def _parse(self, ast, row):
        if not ast or isinstance(ast, (str, int, tatsu.buffering.Buffer)):
            return

        elif isinstance(ast, (tuple, list)):
            [self._parse(element, row) for element in ast]

        elif isinstance(ast, tatsu.ast.AST):
            if ast.parseinfo is not None:
                stat_parsers = self.rule_registry[ast.parseinfo.rule]
                for stat in stat_parsers:
                    result = stat.extract(ast)
                    if result:
                        row[stat.header].append(result)

            [self._parse(element, row) for element in ast.values()]

        else:
            print(f'Encountered AST element with unrecognized type: {ast} of type {type(ast)}')


def build_aggregator(args):
    agg = ASTStatisticsAggregator()

    length_of_then = StatExtractor('then', 'length_of_then', lambda ast: len(ast.then_funcs), np.mean)
    agg.register(length_of_then)

    num_preferences = StatExtractor('preference', 'num_preferences', lambda ast: 1, np.sum)
    agg.register(num_preferences)

    def objects_quantified(ast):
        key = 'exists_vars'
        if 'forall_vars' in ast:
            key = 'forall_vars'
        return len(ast[key][1])

    num_setup_objects_quantified = StatExtractor(
        ('setup_exists', 'setup_forall'), 'setup_objects_quantified', 
        objects_quantified, np.mean)
    agg.register(num_setup_objects_quantified)

    num_preference_objects_quantified = StatExtractor(
        ('pref_body_exists', 'pref_body_forall'), 'preference_objects_quantified', 
        objects_quantified, np.mean)
    agg.register(num_preference_objects_quantified)

    terminal_clause_exists = StatExtractor(
        'terminal_comp', 'terminal_exists', lambda ast: True, all
    )
    agg.register(terminal_clause_exists)

    def objects_referenced(ast):
        key = 'exists_vars'
        if 'forall_vars' in ast:
            key = 'forall_vars'

        results = defaultdict(lambda: 0)
        for quantification in ast[key][1]:
            if isinstance(quantification[2], str):
                results[quantification[2]] += 1
            else:
                for name in quantification[2].type_names:
                    results[name] += 1

        return results

    def aggregate_count_dicts(count_dicts):
        results = count_dicts[0]
        for cd in count_dicts[1:]:
            for key in cd:
                results[key] += cd[key]
        return dict(results)

    object_types_quantified = StatExtractor(
        ('setup_exists', 'setup_forall', 'pref_body_exists', 'pref_body_forall'),
        'object_types_quantified', objects_referenced, aggregate_count_dicts
    )
    agg.register(object_types_quantified)
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
            aggregator.parse(ast)

    df = aggregator.to_df()
    df.to_csv(args.output_path, index_label='Index')    


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)
    
    main(args)
