import argparse
from collections import namedtuple, defaultdict
import tatsu
import tqdm
import pandas as pd
import numpy as np
import pickle
import os
import re
import typing

from parse_dsl import load_tests_from_file
from ast_parser import ASTParser
from ast_utils import update_ast
import ast_printer

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
DEFAULT_OUTPUT_PATH ='./data/ast_counter.pickle'
parser.add_argument('-o', '--output-path', default=DEFAULT_OUTPUT_PATH)


class RuleKeyValueCounter:
    def __init__(self, rule: str, key:str):
        self.rule = rule
        self.key = key
        self.rule_counts = defaultdict(int)
        self.value_counts = defaultdict(int)
        self.length_counts = []

    def __call__(self, value: typing.Union[str, typing.Sequence, tatsu.ast.AST]):
        if isinstance(value, str):
            self.value_counts[value] += 1

        elif isinstance(value, list):
            self.length_counts.append(len(value))
            for item in value:
                self(item)

        elif isinstance(value, tatsu.ast.AST):
            self.rule_counts[value.parseinfo.rule] += 1

    def __repr__(self):
        return f'<Counter for {self.rule}.{self.key}: {sum(self.rule_counts.values())} rules counted | {sum(self.value_counts.values())} values counted | {len(self.length_counts)} lengths counted>'
        

def empty_defaultdict_dict():
    return defaultdict(dict)


class ASTRuleValueCounter(ASTParser):
    """
    This class counts the values appearing for each rule in the AST.
    For the current node, we iterate through all children:
    - If the child is an AST, we count the rule of the AST
    - If the child is a list, we note the length of the list, and count each child
    - If the child is a tuple, TBD, we probably only count some of the elements, 
        but need to figure out which ones, which might vary by rule
    """
    def __init__(self):
        self.counters = defaultdict(dict)
        self.counters_by_section = defaultdict(empty_defaultdict_dict)

    def __call__(self, ast, **kwargs):
        if 'is_root' not in kwargs:
            kwargs['is_root'] = False
            kwargs['section'] = 'preamble'

        return super().__call__(ast, **kwargs)

    def _handle_value(self, ast, **kwargs):
        if 'parent_rule' in kwargs and kwargs['parent_rule'] is not None:
            self.counters[kwargs['parent_rule']][kwargs['rule_key']](ast)
            self.counters_by_section[kwargs['section']][kwargs['parent_rule']][kwargs['rule_key']](ast)

    def _handle_ast(self, ast, **kwargs):
        self._handle_value(ast, **kwargs)
        rule = ast.parseinfo.rule

        for key in ast:
            if key != 'parseinfo':
                if key not in self.counters[rule]:
                    self.counters[rule][key] = RuleKeyValueCounter(rule, key)

                if key not in self.counters_by_section[kwargs['section']][rule]:
                    self.counters_by_section[kwargs['section']][rule][key] = RuleKeyValueCounter(rule, key)
                
                new_kwargs = kwargs.copy()
                new_kwargs['parent_rule'] = rule
                new_kwargs['rule_key'] = key
                self(ast[key], **new_kwargs)

    def _handle_str(self, ast, **kwargs):
        self._handle_value(ast, **kwargs)

    def _handle_int(self, ast, **kwargs):
        self._handle_value(ast, **kwargs)

    def _handle_list(self, ast, **kwargs):
        self._handle_value(ast, **kwargs)
        super()._handle_list(ast, **kwargs)

    def _handle_tuple(self, ast, **kwargs):
        if ast[0].startswith('(:'):
            kwargs['section'] = ast[0][2:]

        return super()._handle_tuple(ast, **kwargs)


class ASTSampler:
    """
    Other than a counter, right now it would have to take in:
    - The root fields to generate, and what they can be resolved by 
    (unless I rewrite the EBNF for each section to return a AST, not a tuple,
    but that requires rewriting the section handling logic in the `ASTCounter`
    and `ASTPrinter` classes as well)
    - Which objects/types are in a room, to filter?
    - A prior? Presumably Dirichlet over the valid tokens? 
    - That means we need lists of valid tokens for some of the rules:
        - Predicates, function calls
        - Named objects (desk, bed, etc.)
        - Object types
    - Actually, it means we need a list of valid expansions for each rule, right?


    """
    def __init__(self):
        pass


def main(args):
    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    counter = ASTRuleValueCounter()

    for test_file in args.test_files:
        test_cases = load_tests_from_file(test_file)

        if not args.dont_tqdm:
            test_cases = tqdm.tqdm(test_cases)

        for test_case in test_cases:
            ast = grammar_parser.parse(test_case)
            counter(ast)

    with open(args.output_path, 'wb') as out_file:
        pickle.dump(counter, out_file)


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)
    
    main(args)
