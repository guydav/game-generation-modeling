import argparse
from collections import namedtuple, defaultdict, Counter
from functools import reduce
from multiprocessing.sharedctypes import Value
import tatsu
from tatsu import grammars
import tqdm
import pandas as pd
import numpy as np
import pickle
import os
import re
import typing
from uritemplate import variables

from zmq import TYPE

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
parser.add_argument('-c', '--parse-counter', action='store_true')


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
        self.section_counts = defaultdict(int)

    def __getitem__(self, key):
        return self.counters[key]

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
            section = ast[0][2:]
            kwargs['section'] = section
            self.section_counts[section] += 1

        return super()._handle_tuple(ast, **kwargs)

# TODO: populate the below with actual values, and consider splitting type names by rooms
BINARY_OPERATORS = ['-', '/']
MULTI_OPERATORS = ['+', '*']
COMPARISON_OPERATORS = ['<=', '<', '=', '>=', '>']
TYPE_NAMES = []
FUNCTION_NAMES = []
PREDICATE_NAMES = []


def generate_game_id(**kwargs):
    return 'game-id'


def generate_domain_name(**kwargs):
    return 'domain-name'
    

def sample_new_preference_name(**kwargs):
    # TODO: return preference{i} where i is the current number of preferences
    raise NotImplemented()


def sample_existing_preference_name(**kwargs):
    # TODO: return a random preference name from the current game
    raise NotImplemented()


def generate_number(**kwargs):
    # TODO: decide how to handle this
    raise NotImplemented()


def sample_new_variable(**kwargs):
    # TODO: return ?{l} where l is the next letter
    raise NotImplemented()


def sample_existing_variable(**kwargs):
    # TODO: return a random variable name from the current context
    raise NotImplemented()


VARIABLE_DEFAULTS = defaultdict(lambda: sample_existing_variable)
VARIABLE_DEFAULTS[('variable_type_def', 'var_names')] = sample_new_variable


DEFAULT_PATTERN_RULE_OPTIONS_BY_RULE = dict(
    binary_comp=defaultdict(lambda: COMPARISON_OPERATORS),
    binary_op=defaultdict(lambda: BINARY_OPERATORS),
    multi_op=defaultdict(lambda: MULTI_OPERATORS),
    func_name=defaultdict(lambda: FUNCTION_NAMES),
    type_name=defaultdict(lambda: TYPE_NAMES),
    predicate_name=defaultdict(lambda: PREDICATE_NAMES),
    preference_name={
        ('preference', 'pref_name'): sample_new_preference_name,
        ('pref_name_and_types', 'pref_name'): sample_existing_preference_name,
    },
    id={
        ('game_def', 'game_name'): generate_game_id, 
        ('domain_def', 'domain_name'): generate_domain_name, 
    },
    number=defaultdict(lambda: generate_number),  # TODO: decide how we want to handle number sampling
    variable=VARIABLE_DEFAULTS,
    total_time=defaultdict(lambda: ['(total-time)']),
    total_score=defaultdict(lambda: ['(total-score)'])
)

SPECIAL_RULE_FIELD_VALUE_TYPES = {
    ('type_definition', 'type'): 'type_name',
    ('comparison_arg', 'arg'): ('number', 'type_name', 'variable'),
    ('predicate_term', 'term'): ('type_name', 'variable'),
    ('scoring_expr', 'expr'): ('number', 'total_time', 'total_score'),
}

PATTERN_TYPE_MAPPINGS = {
    'type_name': 'name',
    'func_name': 'name',
    'preference_name': 'name',   
}

PRIOR_COUNT = 5
LENGTH_PRIOR = {i: PRIOR_COUNT for i in range(4)}

class ASTSampler:
    """
    I have a prior parser into a dict format of sort. 
    If I start sampling from the `start` token, that should get me somewhere.
    What I need to figure out is how to combine the information from the 
    `ASTCounter` into the prior -- which of these formats is more conducive to sampling from?
    Can I transform the `ASTCounter` into a format more like the dict in which I have the prior in?

    There are a few pieces of information I want to get from the Counter:
    - For optional fields, how likely they are to exist
    - For list/variable length fields, a distribution over lengths (+ some prior?)
    - For mandatory fields, how likely the different productions are to exist
    (agin, + a prior over the rest of the productions)

    I have some thoughts in the comment above and the slack messages about how to 
    handle this for the terminal productions. I need to figure out how to combine
    the information for the rule-based productions, though. 
    """
    def __init__(self, grammar_parser, ast_counter, 
                 pattern_rule_options=DEFAULT_PATTERN_RULE_OPTIONS_BY_RULE,
                 rule_field_value_types=SPECIAL_RULE_FIELD_VALUE_TYPES,
                 pattern_type_mappings=PATTERN_TYPE_MAPPINGS,
                 prior_rule_count=PRIOR_COUNT, 
                 prior_value_count=PRIOR_COUNT, 
                 length_prior=LENGTH_PRIOR, ):
        
        self.grammar_parser = grammar_parser
        self.ast_counter = ast_counter
        
        self.pattern_rule_options = pattern_rule_options
        self.rule_field_value_types = rule_field_value_types
        self.pattern_type_mappings = pattern_type_mappings

        self.prior_rule_count = prior_rule_count
        self.prior_value_count = prior_value_count
        self.length_prior = length_prior

        self.rule_probabilities = {}
        self.value_patterns = dict(
            any=re.compile(re.escape('(any)')),
            total_time=re.compile(re.escape('(total-time)')),
            total_score=re.compile(re.escape('(total-score)'))
        )
        self.parse_prior_to_posterior()
        print(self.rule_probabilities)

    def _update_prior_to_posterior(self, rule_name, field_name, field_prior):
        rule_counter = self.ast_counter.counters[rule_name]

        if 'options' not in field_prior:
            if field_name in rule_counter:
                print(f'No options for {rule_name}.{field_name} with counted data: {rule_counter.field_name}')
            else:
                print(f'No options for {rule_name}.{field_name} with counted data: {rule_counter}')
            
            return

        options = field_prior['options']
        field_counter = rule_counter[field_name] if field_name in rule_counter else None

        if '_min_length' in field_prior:
            self._create_length_posterior(rule_name, field_name, field_prior, field_counter)

        if isinstance(options, str):
            # If it's a string, check if it's a rule that expands a token
            if options in self.pattern_rule_options:
                self._create_value_posterior(rule_name, field_name, field_prior, options, field_counter)

            # Otherwise, it's a singular rule, and we fall through to the next case
            else:
                options = [options]

        # It's a list, which at this point means it's a list of optional expansion rules
        if isinstance(options, list):
            self._create_rule_posterior(rule_name, field_name, field_prior, options, field_counter)

        elif not isinstance(options, str):
            print(f'Unrecognized options type for {rule_name}.{field_name}: {options}')

        # TODO: should these values come from the prior, likelihood, or posterior? Or be fixed?
        rule_counts = sum(field_prior['rule_posterior'].values()) if 'rule_posterior' in field_prior else 0
        value_counts = sum(field_prior['value_posterior'].values()) if 'value_posterior' in field_prior else 0
        total_counts = rule_counts + value_counts

        if total_counts == 0:
            print(f'No counts for {rule_name}.{field_name}')
            total_counts = 1
        
        field_prior['type_posterior'] = dict(p_rule=rule_counts / total_counts, p_value=value_counts / total_counts)
        

    def _create_length_posterior(self, rule_name, field_name, field_prior, field_counter):
        min_length = field_prior['_min_length']
        length_posterior = Counter({k: v for k, v in self.length_prior.items() if k >= min_length})

        if field_counter is not None:
            if len(field_counter.length_counts) == 0:
                raise ValueError(f'No length counts for {rule_name}.{field_name} which has a min length')

            length_posterior.update(field_counter.length_counts)
            total_lengths = sum(field_counter.length_counts)
            total_obs = sum(field_counter.rule_counts.values()) + sum(field_counter.value_counts.values())
                # TODO: is there a prior over lengths?
            if total_lengths > total_obs:
                raise ValueError(f'Length counts for {rule_name}.{field_name} are too high: {total_lengths} > {total_obs}')

            elif total_lengths < total_obs:
                length_posterior[1] += total_obs - total_lengths

        field_prior['length_posterior'] = length_posterior

    def _create_value_posterior(self, rule_name, field_name, field_prior, value_type, field_counter, rule_hybrird=False):
        field_default = self.pattern_rule_options[value_type][(rule_name, field_name)]
        if 'token_posterior' not in field_prior:
            field_prior['token_posterior'] = defaultdict(int)

        pattern_type = value_type
        if pattern_type in self.pattern_type_mappings:
            pattern_type = self.pattern_type_mappings[pattern_type]

        value_pattern = self.value_patterns[pattern_type] if pattern_type in self.value_patterns else None

        if isinstance(field_default, list):
            field_prior['token_posterior'].update({value: self.prior_value_count for value in field_default})

            if field_counter is not None:
                if len(field_counter.rule_counts) > 0 and not rule_hybrird:
                    raise ValueError(f'{rule_name}.{field_name} has counted rules, which should not exist')

                for value, count in field_counter.value_counts.items():
                    if value_pattern is None or value_pattern.match(value) is not None:
                        field_prior['token_posterior'][value] += count

        elif hasattr(field_default, '__call__'):
            count = self.prior_value_count

            if value_pattern is not None and field_counter is not None:
                valid_values = filter(lambda v: value_pattern.match(v) is not None, field_counter.value_counts)
                count += sum(field_counter.value_counts[value] for value in valid_values)

            field_prior['token_posterior'][value_type] = count
            if 'samplers' not in field_prior:
                field_prior['samplers'] = {}
            field_prior['samplers'][value_type] = field_default
        else:
            raise ValueError(f'Unknown field_default type: {field_default}')

    def _create_rule_posterior(self, rule_name, field_name, field_prior, options, field_counter):
        field_prior['rule_posterior'] = {value: self.prior_value_count for value in options}

        if field_counter is not None:
            if len(field_counter.value_counts) > 0:
                if (rule_name, field_name) in self.rule_field_value_types:
                    value_type = self.rule_field_value_types[(rule_name, field_name)]
                    if isinstance(value_type, str):
                        self._create_value_posterior(rule_name, field_name, field_prior, value_type, field_counter, rule_hybrird=True)
                    else:
                        for vt in value_type:
                            self._create_value_posterior(rule_name, field_name, field_prior, vt, field_counter, rule_hybrird=True)

                else:
                    raise ValueError(f'{rule_name}.{field_name} has counted values but should only have rules or have a special type')

            for counted_rule, count in field_counter.rule_counts.items():
                if counted_rule not in field_prior['rule_posterior']:
                    raise ValueError(f'{rule_name}.{field_name} has counted rule {counted_rule} which is not in the prior {options}')

                field_prior['rule_posterior'][counted_rule] += count
        else:
            print(f'No counted data for {rule_name}.{field_name}')
        
    def parse_prior_to_posterior(self):
        all_rules = set([rule.name for rule in self.grammar_parser.rules])

        for rule in self.grammar_parser.rules:
            children = rule.children()
            if len(children) > 1:
                print(f'Encountered rule with multiple children: {rule.name}')
                continue

            child = children[0]
            rule_name = rule.name
            rule_prior = self.parse_rule_child(child)

            # Special cases
            if rule_name in ('preferences', 'pref_forall_prefs'):
                rule_prior = rule_prior[0]

            if rule_name == 'start':
                continue

            elif isinstance(rule_prior, dict):
                for field_name, field_prior in rule_prior.items():
                    if field_name == 'pattern' and isinstance(field_prior, str):
                        self.value_patterns[rule_name] = re.compile(field_prior)
                        continue

                    if not isinstance(field_prior, dict): 
                        print(f'Encountered non-dict prior for {rule_name}.{field_name}: {field_prior}')
                        continue
                    self._update_prior_to_posterior(rule_name, field_name, field_prior)

            elif isinstance(rule_prior, list):
                # The sections that optionally exist
                if None in rule_prior:
                    section_name = rule_name.replace('_def', '')
                    if section_name not in self.ast_counter.section_counts:
                        raise ValueError(f'{rule_name} has no section counts')

                    section_prob = self.ast_counter.section_counts[section_name] / max(self.ast_counter.section_counts.values())
                    child_rule = list(filter(lambda x: x is not None, rule_prior))[0]
                    rule_prior = dict(rule_posterior={child_rule: self.prior_rule_count}, production_prob=section_prob)

                else:
                    rule_prior = dict(rule_posterior={r: self.prior_rule_count for r in rule_prior})

            elif isinstance(rule_prior, str):
                # This is a rule that expands only to a single other rule
                if rule_prior in all_rules:
                    rule_prior = dict(rule_posterior={rule_prior: 1})
                else:
                    token = rule_prior
                    rule_prior = dict(token_posterior={token: 1})
                    self.value_patterns[rule_name] = re.compile(re.escape(token))
                    print(f'String token rule for {rule_name}: {rule_prior}')

            else:
                print(f'Encountered rule with unknown prior or no special case: {rule.name}\n{rule_prior}')

            self.rule_probabilities[rule.name] = rule_prior

    def parse_rule_child(self, child, parse_children=True):
        if isinstance(child, grammars.EOF):
            return
        
        if isinstance(child, grammars.Token):
            return child.token

        if isinstance(child, grammars.Pattern):
            return dict(pattern=child.pattern)

        if isinstance(child, grammars.Sequence):
            filtered_children = list(filter(lambda x: not isinstance(x, grammars.Token), child.children()))
            if len(filtered_children) == 0:
                return
            
            if parse_children:
                if len(filtered_children) == 1:
                    return self.parse_rule_child(filtered_children[0])

                else:
                    parsed_children = [self.parse_rule_child(x) for x in filtered_children]
                    children_are_dicts = [isinstance(pc, dict) for pc in parsed_children]
                    if any(children_are_dicts):
                        if all(children_are_dicts):
                            return reduce(lambda x, y: {**x, **y}, parsed_children)
                        else:
                            print(f'WARNING: {child} has parsed children that are not dicts: {parsed_children}')
                    
                    return parsed_children

        if isinstance(child, grammars.Named):
            return {child.name: dict(options=self.parse_rule_child(child_child)) for child_child in child.children()}
        
        if isinstance(child, grammars.Group):
            if len(child.children()) == 1:
                return self.parse_rule_child(child.children()[0])

            else:
                return [self.parse_rule_child(child_child) for child_child in child.children()]

        if isinstance(child, grammars.Choice):
            return [self.parse_rule_child(child_child) for child_child in child.children()]

        if isinstance(child, (grammars.PositiveClosure, grammars.Closure)):
            d = {'_min_length': 1 if isinstance(child, grammars.PositiveClosure) else 0}
            if len(child.children()) == 1:
                child_value = self.parse_rule_child(child.children()[0])
                if not isinstance(child_value, dict):
                    print(f'Encoutered positive closure with unexpected value type: {child_value}')    

                child_value[next(iter(child_value))].update(d)
                d = child_value
            else:
                print(f'Encoutered positive closure with multiple children: {child}')

            return d

        if isinstance(child, grammars.RuleRef):
            return child.name

        if isinstance(child, grammars.Void):
            return None

        else:
            print(f'Unhandled child type: {type(child)}: {child}')

 
    def sample(self):
        # TODO: think about how to write this
        if isinstance(child, grammars.Sequence):
            return [self.parse_rule_child(rule, child_child) for child_child in child]

        elif isinstance(child, grammars.Token):
            return child.token


def main(args):
    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    if args.parse_counter:
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

    else:
        with open(args.output_path, 'rb') as pickle_file:
            counter = pickle.load(pickle_file)

    sampler = ASTSampler(grammar_parser, counter)


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)
    
    main(args)
