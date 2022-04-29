import argparse
from collections import namedtuple, defaultdict, Counter
from functools import reduce
from multiprocessing.sharedctypes import Value
from random import sample
import tatsu
from tatsu import grammars
import tqdm
import pandas as pd
import numpy as np
import pickle
import copy
import re
import typing
import string
from uritemplate import variables

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
parser.add_argument('-n', '--num-samples', type=int, default=10)
parser.add_argument('-p', '--print-samples', action='store_true')
parser.add_argument('-v', '--validate-samples', action='store_true')


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


def generate_game_id(global_context=None, local_context=None):
    return 'game-id'


def generate_domain_name(global_context=None, local_context=None):
    return 'domain-name'
    

def _preference_name(i):
    return f'preference{chr(ord("A") + i - 1)}'


def sample_new_preference_name(global_context=None, local_context=None):
    if 'preference_count' not in global_context:
        global_context['preference_count'] = 0

    global_context['preference_count'] += 1
    return _preference_name(global_context['preference_count'])


def sample_existing_preference_name(global_context=None, local_context=None):
    if 'preference_count' not in global_context:
        # TODO: do we want to fail here, or return a nonsensical value
        return 'no-preferences-exist'

    if 'rng' not in global_context:
        rng = np.random.default_rng()
    else:
        rng = global_context['rng']

    pref_count = global_context['preference_count']
    return _preference_name(rng.integers(1, pref_count + 1))


def generate_number(global_context=None, local_context=None):
    # TODO: decide how to handle this -- do we want to look at the existing values?
    # TODO: we should return 1 
    return str(1)


def sample_new_variable(global_context=None, local_context=None):
    if 'variables' not in local_context:
        local_context['variables'] = set()

    if 'rng' not in global_context:
        rng = np.random.default_rng()
    else:
        rng = global_context['rng']

    valid_vars = set(string.ascii_lowercase) - local_context['variables']
    new_var = rng.choice(list(valid_vars))
    local_context['variables'].add(new_var)
    return f'?{new_var}'


def sample_existing_variable(global_context=None, local_context=None):
    if 'variables' not in local_context:
        # TODO: decide if we should return a nonsensical value here or an error
        return '?xxx'

    if 'rng' not in global_context:
        rng = np.random.default_rng()
    else:
        rng = global_context['rng']

    return f'?{rng.choice(list(local_context["variables"]))}'


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

# TODO: consider if we want to try to remove some of these extra steps
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

LOCAL_CONTEXT_PROPAGATING_RULES = set(['variable_type_def', 'variable_list'])

PRIOR_COUNT = 5
LENGTH_PRIOR = {i: PRIOR_COUNT for i in range(4)}

PRODUCTION = 'production'
OPTIONS = 'options'
SAMPLERS = 'samplers'
TYPE_POSTERIOR = 'type_posterior'
RULE_POSTERIOR = 'rule_posterior'
TOKEN_POSTERIOR = 'token_posterior'
LENGTH_POSTERIOR = 'length_posterior'
PRODUCTION_PROBABILITY = 'production_probability'
START = 'start'
EOF = 'EOF'
SAMPLE = 'SAMPLE'
RULE = 'rule'
TOKEN = 'token'
NAMED = 'named'
PATTERN = 'pattern'
DEFAULT_RANDOM_SEED = 33

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
                 local_context_propagating_rules=LOCAL_CONTEXT_PROPAGATING_RULES,
                 prior_rule_count=PRIOR_COUNT, 
                 prior_token_count=PRIOR_COUNT, 
                 length_prior=LENGTH_PRIOR, rng=None, seed=DEFAULT_RANDOM_SEED):
        
        self.grammar_parser = grammar_parser
        self.ast_counter = ast_counter
        
        self.pattern_rule_options = pattern_rule_options
        self.rule_field_value_types = rule_field_value_types
        self.pattern_type_mappings = pattern_type_mappings
        self.local_context_propagating_rules = local_context_propagating_rules

        self.prior_rule_count = prior_rule_count
        self.prior_token_count = prior_token_count
        self.length_prior = length_prior

        if rng is None:
            rng = np.random.default_rng(seed)
        self.rng = rng

        self.rules = {}
        self.value_patterns = dict(
            any=re.compile(re.escape('(any)')),
            total_time=re.compile(re.escape('(total-time)')),
            total_score=re.compile(re.escape('(total-score)'))
        )
        self.parse_prior_to_posterior()

    def _update_prior_to_posterior(self, rule_name: str, field_name: str, field_prior: RuleKeyValueCounter):
        rule_counter = self.ast_counter.counters[rule_name]

        if OPTIONS not in field_prior:
            if field_name in rule_counter:
                print(f'No options for {rule_name}.{field_name} with counted data: {rule_counter.field_name}')
            else:
                print(f'No options for {rule_name}.{field_name} with counted data: {rule_counter}')
            
            return

        options = field_prior[OPTIONS]
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
        rule_counts = sum(field_prior[RULE_POSTERIOR].values()) if RULE_POSTERIOR in field_prior else 0
        token_counts = sum(field_prior[TOKEN_POSTERIOR].values()) if TOKEN_POSTERIOR in field_prior else 0
        type_posterior = {RULE: rule_counts, TOKEN: token_counts}
        field_prior[TYPE_POSTERIOR] = self._normalize_posterior_dict(type_posterior)
        
        # Normalize the rule and token posteriors
        if RULE_POSTERIOR in field_prior:
            field_prior[RULE_POSTERIOR] = self._normalize_posterior_dict(field_prior[RULE_POSTERIOR])
        if TOKEN_POSTERIOR in field_prior:
            field_prior[TOKEN_POSTERIOR] = self._normalize_posterior_dict(field_prior[TOKEN_POSTERIOR])

    def _normalize_posterior_dict(self, posterior_dict):
        total_counts = sum(posterior_dict.values())
        if total_counts == 0:
            total_counts = 1
        
        return {key: count / total_counts for key, count in posterior_dict.items()}

    def _create_length_posterior(self, rule_name: str, field_name: str, 
        field_prior: typing.Dict[str, typing.Union[str, typing.Sequence[str], typing.Dict[str, float]]], 
        field_counter: RuleKeyValueCounter):

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

        field_prior[LENGTH_POSTERIOR] = self._normalize_posterior_dict(length_posterior)

    def _create_value_posterior(self, rule_name: str, field_name: str, 
        field_prior: typing.Dict[str, typing.Union[str, typing.Sequence[str], typing.Dict[str, float]]], 
        value_type: str, field_counter: RuleKeyValueCounter, rule_hybrird: bool = False):
        
        field_default = self.pattern_rule_options[value_type][(rule_name, field_name)]
        if TOKEN_POSTERIOR not in field_prior:
            field_prior[TOKEN_POSTERIOR] = defaultdict(int)

        pattern_type = value_type
        if pattern_type in self.pattern_type_mappings:
            pattern_type = self.pattern_type_mappings[pattern_type]

        value_pattern = self.value_patterns[pattern_type] if pattern_type in self.value_patterns else None

        if isinstance(field_default, list):
            field_prior[TOKEN_POSTERIOR].update({value: self.prior_token_count for value in field_default})

            if field_counter is not None:
                if len(field_counter.rule_counts) > 0 and not rule_hybrird:
                    raise ValueError(f'{rule_name}.{field_name} has counted rules, which should not exist')

                for value, count in field_counter.value_counts.items():
                    if value_pattern is None or value_pattern.match(value) is not None:
                        field_prior[TOKEN_POSTERIOR][value] += count

        elif hasattr(field_default, '__call__'):
            count = self.prior_token_count

            if value_pattern is not None and field_counter is not None:
                valid_values = filter(lambda v: value_pattern.match(v) is not None, field_counter.value_counts)
                count += sum(field_counter.value_counts[value] for value in valid_values)

            field_prior[TOKEN_POSTERIOR][value_type] = count
            if SAMPLERS not in field_prior:
                field_prior[SAMPLERS] = {}
            field_prior[SAMPLERS][value_type] = field_default
        else:
            raise ValueError(f'Unknown field_default type: {field_default}')

    def _create_rule_posterior(self, rule_name: str, field_name: str, 
        field_prior: typing.Dict[str, typing.Union[str, typing.Sequence[str], typing.Dict[str, float]]], 
        options: typing.Sequence[str], field_counter: RuleKeyValueCounter):

        field_prior[RULE_POSTERIOR] = {value: self.prior_token_count for value in options}

        if field_counter is not None:
            if len(field_counter.value_counts) > 0:
                if (rule_name, field_name) in self.rule_field_value_types:
                    value_type = self.rule_field_value_types[(rule_name, field_name)]
                    if isinstance(value_type, str):
                        if value_type in field_prior[RULE_POSTERIOR]:
                            del field_prior[RULE_POSTERIOR][value_type]
                        self._create_value_posterior(rule_name, field_name, field_prior, value_type, field_counter, rule_hybrird=True)
                    else:
                        for vt in value_type:
                            if vt in field_prior[RULE_POSTERIOR]:
                                del field_prior[RULE_POSTERIOR][vt]
                            self._create_value_posterior(rule_name, field_name, field_prior, vt, field_counter, rule_hybrird=True)

                else:
                    raise ValueError(f'{rule_name}.{field_name} has counted values but should only have rules or have a special type')

            for counted_rule, count in field_counter.rule_counts.items():
                if counted_rule not in field_prior[RULE_POSTERIOR]:
                    raise ValueError(f'{rule_name}.{field_name} has counted rule {counted_rule} which is not in the prior {options}')

                field_prior[RULE_POSTERIOR][counted_rule] += count
        else:
            print(f'No counted data for {rule_name}.{field_name}')

        if len(field_prior[RULE_POSTERIOR]) == 0:
            del field_prior[RULE_POSTERIOR]
        
    def parse_prior_to_posterior(self):
        all_rules = set([rule.name for rule in self.grammar_parser.rules])

        for rule in self.grammar_parser.rules:
            children = rule.children()
            if len(children) > 1:
                print(f'Encountered rule with multiple children: {rule.name}')
                continue

            child = children[0]
            rule_name = rule.name
            rule_prior = self.parse_rule_prior(child)

            # Special cases
            if rule_name in ('preferences', 'pref_forall_prefs'):
                rule_prior = rule_prior[0]

            if rule_name == START:
                pass

            elif isinstance(rule_prior, dict):
                for field_name, field_prior in rule_prior.items():
                    production = None
                    if field_name == PRODUCTION:
                        continue

                    if field_name == PATTERN and isinstance(field_prior, str):
                        production = [(PATTERN, field_prior)]
                        self.value_patterns[rule_name] = re.compile(field_prior)
                        continue

                    if not isinstance(field_prior, dict): 
                        print(f'Encountered non-dict prior for {rule_name}.{field_name}: {field_prior}')
                        continue
                    self._update_prior_to_posterior(rule_name, field_name, field_prior)

                if PRODUCTION not in rule_prior:
                    if production is None:
                        production = [(NAMED, key) for key in rule_prior.keys()]
                    
                    rule_prior[PRODUCTION] = production

            elif isinstance(rule_prior, list):
                # The sections that optionally exist
                if None in rule_prior:
                    section_name = rule_name.replace('_def', '')
                    if section_name not in self.ast_counter.section_counts:
                        raise ValueError(f'{rule_name} has no section counts')

                    section_prob = self.ast_counter.section_counts[section_name] / max(self.ast_counter.section_counts.values())
                    child = list(filter(lambda x: x is not None, rule_prior))[0]
                    child[PRODUCTION_PROBABILITY] = section_prob
                    rule_prior = child

                else:
                    if rule_name == 'predicate_name':
                        rule_prior = dict(rule_posterior={r: 1.0 / len(rule_prior) for r in rule_prior}, production=[('rule', SAMPLE)])
                    else:
                        rule_prior = dict(token_posterior={r: 1.0 / len(rule_prior) for r in rule_prior}, production=[('token', SAMPLE)])
                        

            elif isinstance(rule_prior, str):
                # This is a rule that expands only to a single other rule
                if rule_prior in all_rules:
                    rule_prior = dict(rule_posterior={rule_prior: 1}, production=[(RULE, rule_prior)])

                # This is a rule that expands directly to a token
                else:
                    token = rule_prior
                    rule_prior = dict(token_posterior={token: 1}, production=[(TOKEN, token)])
                    self.value_patterns[rule_name] = re.compile(re.escape(token))
                    print(f'String token rule for {rule_name}: {rule_prior}')

            else:
                print(f'Encountered rule with unknown prior or no special case: {rule.name}\n{rule_prior}')

            self.rules[rule.name] = rule_prior

    def parse_rule_prior(self, rule: tatsu.grammars.Node):
        if isinstance(rule, grammars.EOF):
            return EOF
        
        if isinstance(rule, grammars.Token):
            return rule.token

        if isinstance(rule, grammars.Pattern):
            return dict(pattern=rule.pattern)

        if isinstance(rule, grammars.Sequence):
            rule_dict = {}
            sequence = []
            for child in rule.children():
                parsed_child = self.parse_rule_prior(child)
                if isinstance(parsed_child, str):
                    if isinstance(child, grammars.RuleRef):
                        sequence.append((RULE, parsed_child))
                    else:
                        sequence.append((TOKEN, parsed_child))

                elif isinstance(parsed_child, dict):
                    rule_dict.update(parsed_child)
                    if len(parsed_child.keys()) > 1:
                        print(f'Encountered child rule parsing to dict with multiple keys: {rule.name}: {parsed_child}')
                    else:
                        sequence.append((NAMED, list(parsed_child.keys())[0]))

                else:
                    print(f'Encountered child rule parsing to unknown type: {rule.name}: {parsed_child}')

            rule_dict[PRODUCTION] = sequence
            return rule_dict

        if isinstance(rule, grammars.Named):
            return {rule.name: dict(options=self.parse_rule_prior(child)) for child in rule.children()}
        
        if isinstance(rule, grammars.Group):
            if len(rule.children()) == 1:
                return self.parse_rule_prior(rule.children()[0])

            else:
                return [self.parse_rule_prior(child) for child in rule.children()]

        if isinstance(rule, grammars.Choice):
            return [self.parse_rule_prior(child) for child in rule.children()]

        if isinstance(rule, (grammars.PositiveClosure, grammars.Closure)):
            d = {'_min_length': 1 if isinstance(rule, grammars.PositiveClosure) else 0}
            if len(rule.children()) == 1:
                child_value = self.parse_rule_prior(rule.children()[0])
                if not isinstance(child_value, dict):
                    print(f'Encoutered positive closure with unexpected value type: {child_value}')    

                child_value[next(iter(child_value))].update(d)
                d = child_value
            else:
                print(f'Encoutered positive closure with multiple children: {rule}')

            return d

        if isinstance(rule, grammars.RuleRef):
            return rule.name

        if isinstance(rule, grammars.Void):
            return None

        else:
            print(f'Unhandled child type: {type(rule)}: {rule}')

    def _split_posterior(self, posterior_dict: typing.Dict[typing.Any, typing.Any]):
        return zip(*posterior_dict.items())

    def _posterior_dict_sample(self, posterior_dict: typing.Dict[str, float], size: typing.Union[int, typing.Tuple[int], None] = None):
        values, probs = self._split_posterior(posterior_dict)
        return self.rng.choice(values, size=size, p=probs)

    def _sample_named(self, 
        sample_dict: typing.Dict[str, typing.Union[str, typing.Sequence[str], typing.Dict[str, float]]],
        global_context: typing.Dict[str, typing.Any] = None,
        local_context: typing.Dict[str, typing.Any] = None):

        if TYPE_POSTERIOR not in sample_dict:
            raise ValueError(f'Missing type_posterior in sample: {sample_dict}')

        if LENGTH_POSTERIOR in sample_dict:
            length = self._posterior_dict_sample(sample_dict[LENGTH_POSTERIOR])
            if length == 0:
                return None, None
            values, context_updates = zip(*[self._sample_single_named_value(sample_dict, global_context, local_context) for _ in range(length)])
            context_update = reduce(lambda x, y: {**x, **y}, filter(lambda d: d is not None, context_updates), {})
            return list(values), context_update

        else:
            return self._sample_single_named_value(sample_dict, global_context, local_context)

    def _sample_single_named_value(self, sample_dict: typing.Dict[str, typing.Union[str, typing.Sequence[str], typing.Dict[str, float]]],
        global_context: typing.Dict[str, typing.Any] = None,
        local_context: typing.Dict[str, typing.Any] = None):

        sample_type = self._posterior_dict_sample(sample_dict[TYPE_POSTERIOR])

        if sample_type == RULE:
            if RULE_POSTERIOR not in sample_dict:
                raise ValueError(f'Missing rule_posterior in sample: {sample_dict}')

            rule = self._posterior_dict_sample(sample_dict[RULE_POSTERIOR])
            return self.sample(rule, global_context, local_context)

        elif sample_type == TOKEN:
            if TOKEN_POSTERIOR not in sample_dict:
                raise ValueError(f'Missing token_posterior in sample: {sample_dict}')

            token = self._posterior_dict_sample(sample_dict[TOKEN_POSTERIOR])
            if SAMPLERS in sample_dict and token in sample_dict[SAMPLERS]:
                token = sample_dict[SAMPLERS][token](global_context, local_context)

            return token, None

        else:
            raise ValueError(f'Unknown sample type: {sample_type} in sample: {sample_dict}')

    def sample(self, rule: str = START, global_context: typing.Dict[str, typing.Any] = None, 
        local_context: typing.Dict[str, typing.Any] = None):

        if global_context is None:
            global_context = dict(rng=self.rng)
        
        if local_context is None:
            local_context = dict()
        else:
            local_context = copy.deepcopy(local_context)

        rule_dict = self.rules[rule]
        production = rule_dict[PRODUCTION]
        output = []
        return_ast = False
        # TODO: check production_probability

        for prod_type, prod_value in production:
            if prod_type == TOKEN:
                if prod_value == EOF:
                    pass
                elif prod_value == SAMPLE:
                    output.append(self._posterior_dict_sample(rule_dict[TOKEN_POSTERIOR]))
                else:
                    output.append(prod_value)

            elif prod_type == RULE:
                value, context_update = self.sample(prod_value, global_context, local_context)
                if context_update is not None:
                    local_context.update(context_update)
                output.append(value)

            elif prod_type == NAMED:
                return_ast = True
                value, context_update = self._sample_named(rule_dict[prod_value], global_context, local_context)
                if context_update is not None:
                    local_context.update(context_update)
                output.append({prod_value: value})

        if len(output) == 0:
            print(f'Encountered empty production for {rule}: {production}')
            return None, None

        if return_ast:
            out_dict = reduce(lambda x, y: {**x, **y}, filter(lambda x: isinstance(x, dict), output))
            out_dict['parseinfo'] = tatsu.infos.ParseInfo(None, rule, 0, 0, 0, 0)
            output = tatsu.ast.AST(out_dict)

        elif len(output) == 1:
            output = output[0]

        else:
            output = tuple(output)

        if rule in self.local_context_propagating_rules:
            return output, local_context

        elif rule == START:
            return output

        return output, None


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
    samples = []

    for _ in range(args.num_samples):
        try:     
            ast = sampler.sample()
            samples.append(ast)

            if args.print_samples:
                ast_printer.BUFFER = None
                ast_printer.pretty_print_ast(ast)
                print()

            if args.validate_samples:
                ast_printer.BUFFER = []
                ast_printer.pretty_print_ast(ast)
                first_print_out = ''.join(ast_printer.BUFFER)

                second_ast = grammar_parser.parse(first_print_out)
                ast_printer.BUFFER = []
                ast_printer.pretty_print_ast(second_ast)
                second_print_out = ''.join(ast_printer.BUFFER)

                if first_print_out != second_print_out:
                    print('Mismatch found')

        except (tatsu.exceptions.FailedToken, tatsu.exceptions.FailedParse) as e:
            print(f'Parse failed: at position {e.pos} expected {e.item}:')
            print(first_print_out[e.pos:])

    # TODO: conceptual issue: in places where the expansion is recursive 
    # (e.g. `setup`` expands to rules that all contain setup, or 
    # `preference`` expand to exists/forall that contain preferences) 
    # the inner rules (`setup_game_conserved` / `setup_game_optional`), 
    # or `then` for preferences) are overrepreesnted, and more likely
    # to be sampled at the root of this expression than they are in the corpus

    return


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)
    
    main(args)
