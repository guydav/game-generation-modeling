import argparse
from collections import namedtuple, defaultdict, Counter
from functools import reduce
import tatsu
import tatsu.ast
import tatsu.exceptions
import tatsu.infos
import tatsu.grammars
from tatsu import grammars
import tqdm
import pandas as pd
import numpy as np
import pickle
import copy
import re
import typing
import string
import sys

from parse_dsl import load_games_from_file
from ast_parser import ASTParser, ASTParentMapper, ASTParseinfoSearcher, ASTDepthParser
from ast_utils import cached_load_and_parse_games_from_file, replace_child, fixed_hash
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
DEFAULT_COUNTER_OUTPUT_PATH ='./data/ast_counter.pickle'
parser.add_argument('--counter-output-path', default=DEFAULT_COUNTER_OUTPUT_PATH)
DEFAULT_SAMPLES_OUTPUT_PATH = './dsl/ast-mle-samples.pddl'
parser.add_argument('--samples-output-path', default=DEFAULT_SAMPLES_OUTPUT_PATH)
parser.add_argument('-s', '--save-samples', action='store_true')
parser.add_argument('-c', '--parse-counter', action='store_true')
parser.add_argument('-n', '--num-samples', type=int, default=10)
parser.add_argument('-p', '--print-samples', action='store_true')
parser.add_argument('-v', '--validate-samples', action='store_true')
parser.add_argument('--sample-tqdm', action='store_true')
DEFAULT_RANDOM_SEED = 33
parser.add_argument('--random-seed', type=int, default=DEFAULT_RANDOM_SEED)
DEFAULT_RECURSION_LIMIT = 2000
parser.add_argument('--recursion-limit', type=int, default=DEFAULT_RECURSION_LIMIT)
parser.add_argument('--verbose', action='store_true')

MLE_SAMPLING = 'mle'
REGROWTH_SAMPLING = 'regrowth'
MCMC_REGRWOTH = 'mcmc-regrowth'
parser.add_argument('--sampling-method', choices=[MLE_SAMPLING, REGROWTH_SAMPLING, MCMC_REGRWOTH], required=True)


ContextDict = typing.Dict[str, typing.Any]


class SamplingException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


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
            self.rule_counts[value.parseinfo.rule] += 1  # type: ignore

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
        rule = ast.parseinfo.rule  # type: ignore

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
NUMBER_DEFAULTS = list(range(11))
TYPE_NAMES = []
FUNCTION_NAMES = []
PREDICATE_NAMES = []


def _split_posterior(posterior_dict: typing.Dict[typing.Any, typing.Any]):
    return zip(*posterior_dict.items())


def posterior_dict_sample(rng: np.random.Generator, posterior_dict: typing.Dict[str, float], size: typing.Union[int, typing.Tuple[int], None] = None):
    values, probs = _split_posterior(posterior_dict)
    return rng.choice(values, size=size, p=probs)


def generate_game_id(global_context: ContextDict, local_context: typing.Optional[ContextDict]=None):
    game_id = global_context['original_game_id'] if 'original_game_id' in global_context else 'game-id'
    if 'sample_id' in global_context:
        game_id = f'{game_id}-{global_context["sample_id"]}'
    return game_id


DOMAINS = ('few-objects-room-v1', 'medium-objects-room-v1', 'many-objects-room-v1',)


def generate_domain_name(global_context: ContextDict, local_context: typing.Optional[ContextDict]=None):
    if 'rng' not in global_context:
        rng = np.random.default_rng()
    else:
        rng = global_context['rng']
    return rng.choice(DOMAINS)


def sample_new_preference_name_factory(field_counter: RuleKeyValueCounter, prior_token_count: int=0):
    total_count = sum(field_counter.value_counts.values()) + (prior_token_count * len(field_counter.value_counts))
    value_posterior = {k: (v + prior_token_count) / total_count for k, v in field_counter.value_counts.items()}

    def sample_new_preference_name(global_context: ContextDict, local_context: typing.Optional[ContextDict]=None):
        if 'preference_names' not in global_context:
            global_context['preference_names'] = set()

        pref_name = None
        while (pref_name is None) or (pref_name in global_context['preference_names']):
            pref_name = posterior_dict_sample(global_context['rng'], value_posterior)

        global_context['preference_names'].add(pref_name)
        return pref_name

    return sample_new_preference_name


sample_new_preference_name_factory.factory = True



def sample_existing_preference_name(global_context: ContextDict, local_context: typing.Optional[ContextDict]=None):
    if 'preference_names' not in global_context:
        raise SamplingException('Attempted to sample a preference name with no sampled preference')

    if 'rng' not in global_context:
        rng = np.random.default_rng()
    else:
        rng = global_context['rng']

    pref_names = list(global_context['preference_names'])
    return rng.choice(pref_names)


def sample_new_variable_factory(field_counter: RuleKeyValueCounter, prior_token_count: int=0):
    total_count = sum(field_counter.value_counts.values()) + (prior_token_count * len(field_counter.value_counts))
    value_posterior = {k: (v + prior_token_count) / total_count for k, v in field_counter.value_counts.items()}

    def sample_new_variable(global_context: ContextDict, local_context: typing.Optional[ContextDict]=None):
        if local_context is None:
            local_context = {}

        if 'variables' not in local_context:
            local_context['variables'] = dict()

        if 'rng' not in global_context:
            rng = np.random.default_rng()
        else:
            rng = global_context['rng']

        valid_vars = set(value_posterior.keys()) - set(local_context['variables'].keys())

        if len(valid_vars) == 0:
            raise SamplingException('No valid variables left to sample')

        filtered_posterior_normalization = sum(value_posterior[k] for k in valid_vars)
        filtered_posterior = {k: v / filtered_posterior_normalization for k, v in value_posterior.items() if k in valid_vars}

        new_var = posterior_dict_sample(rng, filtered_posterior)[1:]
        local_context['variables'][new_var] = None
        return f'?{new_var}'

    return sample_new_variable


sample_new_variable_factory.factory = True


def sample_existing_variable(global_context: ContextDict, local_context: typing.Optional[ContextDict]=None):
    if local_context is None:
        local_context = {}

    if 'variables' not in local_context:
        raise SamplingException('Attempted to sample an existing variable with no variables in scope')

    if 'rng' not in global_context:
        rng = np.random.default_rng()
    else:
        rng = global_context['rng']

    return f'?{rng.choice(list(local_context["variables"].keys()))}'


def sample_empty_list(global_context: ContextDict, local_context: typing.Optional[ContextDict]=None):
    return list()


VARIABLE_DEFAULTS = defaultdict(lambda: sample_existing_variable)
VARIABLE_DEFAULTS[('variable_type_def', 'var_names')] = sample_new_variable_factory   # type: ignore


DEFAULT_PATTERN_RULE_OPTIONS_BY_RULE = dict(
    binary_comp=defaultdict(lambda: COMPARISON_OPERATORS),
    binary_op=defaultdict(lambda: BINARY_OPERATORS),
    multi_op=defaultdict(lambda: MULTI_OPERATORS),
    func_name=defaultdict(lambda: FUNCTION_NAMES),
    type_name=defaultdict(lambda: TYPE_NAMES),
    predicate_name=defaultdict(lambda: PREDICATE_NAMES),
    preference_name={
        ('preference', 'pref_name'): sample_new_preference_name_factory,
        ('pref_name_and_types', 'pref_name'): sample_existing_preference_name,
    },
    id={
        ('game_def', 'game_name'): generate_game_id,
        ('domain_def', 'domain_name'): generate_domain_name,
    },
    number=defaultdict(lambda: NUMBER_DEFAULTS),
    variable=VARIABLE_DEFAULTS,
    total_time=defaultdict(lambda: ['(total-time)']),
    total_score=defaultdict(lambda: ['(total-score)']),
)

# TODO: consider if we want to try to remove some of these extra steps
SPECIAL_RULE_FIELD_VALUE_TYPES = {
    ('type_definition', 'type'): 'type_name',
    ('comparison_arg', 'arg'): 'number',
    ('predicate_or_function_term', 'term'): ('type_name', 'variable',),
    # ('function_term', 'term'): ('type_name', 'variable',),
    # ('predicate_term', 'term'): ('type_name', 'variable'),
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
MIN_LENGTH = '_min_length'
OPTIONAL_VOID = 'void'
EMPTY_CLOSURE = 'empty_closure'
EMPTY_LIST = 'empty_list'

HARDCODED_RULES = {
    EMPTY_CLOSURE: {
        TYPE_POSTERIOR: {RULE: 0.0, TOKEN: 1.0},
        TOKEN_POSTERIOR: {EMPTY_LIST: 1.0},
        SAMPLERS: {EMPTY_LIST: sample_empty_list},
        PRODUCTION: ((TOKEN, []),)
    }
}

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
                 prior_rule_count=PRIOR_COUNT, prior_token_count=PRIOR_COUNT,
                 length_prior=LENGTH_PRIOR, hardcoded_rules=HARDCODED_RULES,
                 verbose=False, rng=None, seed=DEFAULT_RANDOM_SEED):

        self.grammar_parser = grammar_parser
        self.ast_counter = ast_counter

        self.pattern_rule_options = pattern_rule_options
        self.rule_field_value_types = rule_field_value_types
        self.pattern_type_mappings = pattern_type_mappings
        self.local_context_propagating_rules = local_context_propagating_rules

        self.prior_rule_count = prior_rule_count
        self.prior_token_count = prior_token_count
        self.length_prior = length_prior
        self.verbose = verbose

        if rng is None:
            rng = np.random.default_rng(seed)  # type: ignore
        self.rng = rng

        self.rules = {k: v for k, v in hardcoded_rules.items()}
        self.value_patterns = dict(
            any=re.compile(re.escape('(any)')),
            total_time=re.compile(re.escape('(total-time)')),
            total_score=re.compile(re.escape('(total-score)'))
        )
        self.all_rules = set([rule.name for rule in self.grammar_parser.rules])
        self.all_rules.update(self.rules.keys())
        self.parse_prior_to_posterior()

        self.sample_parseinfo_index = 0

    def _update_prior_to_posterior(self, rule_name: str, field_name: str, field_prior: ContextDict):
        rule_counter = self.ast_counter.counters[rule_name]

        if OPTIONS not in field_prior:
            if field_name in rule_counter:
                print(f'No options for {rule_name}.{field_name} with counted data: {rule_counter.field_name}')
            else:
                print(f'No options for {rule_name}.{field_name} with counted data: {rule_counter}')

            return

        options = field_prior[OPTIONS]
        field_counter = rule_counter[field_name] if field_name in rule_counter else None

        if MIN_LENGTH in field_prior:
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
            if self.verbose: print(f'Unrecognized options type for {rule_name}.{field_name}: {options}')

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
        field_counter: typing.Optional[RuleKeyValueCounter]):

        min_length = field_prior[MIN_LENGTH]
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
        value_type: str, field_counter: typing.Optional[RuleKeyValueCounter], rule_hybrird: bool = False):

        field_default = self.pattern_rule_options[value_type][(rule_name, field_name)]
        if TOKEN_POSTERIOR not in field_prior:
            field_prior[TOKEN_POSTERIOR] = defaultdict(int)

        pattern_type = value_type
        if pattern_type in self.pattern_type_mappings:
            pattern_type = self.pattern_type_mappings[pattern_type]

        value_pattern = self.value_patterns[pattern_type] if pattern_type in self.value_patterns else None

        if isinstance(field_default, list):
            if not isinstance(field_prior[TOKEN_POSTERIOR], dict):
                raise ValueError(f'Prior for {rule_name}.{field_name} is not a dict')

            field_prior[TOKEN_POSTERIOR].update({value: self.prior_token_count for value in field_default})  # type: ignore

            if field_counter is not None:
                if len(field_counter.rule_counts) > 0 and not rule_hybrird:
                    raise ValueError(f'{rule_name}.{field_name} has counted rules, which should not exist')

                for value, count in field_counter.value_counts.items():
                    if value_pattern is None or value_pattern.match(value) is not None:
                        field_prior[TOKEN_POSTERIOR][value] += count  # type: ignore

        elif hasattr(field_default, '__call__'):
            count = self.prior_token_count

            if value_pattern is not None and field_counter is not None:
                valid_values = filter(lambda v: value_pattern.match(v) is not None, field_counter.value_counts)
                count += sum(field_counter.value_counts[value] for value in valid_values)

            field_prior[TOKEN_POSTERIOR][value_type] = count  # type: ignore
            if SAMPLERS not in field_prior:
                field_prior[SAMPLERS] = {}

            field_sampler = field_default

            if hasattr(field_default, 'factory') and field_default.factory:
                field_sampler = field_default(field_counter, self.prior_token_count)

            field_prior[SAMPLERS][value_type] = field_sampler  # type: ignore

        else:
            raise ValueError(f'Unknown field_default type: {field_default}')

    def _create_rule_posterior(self, rule_name: str, field_name: str,
        field_prior: typing.Dict[str, typing.Union[str, typing.Sequence[str], typing.Dict[str, float]]],
        options: typing.Sequence[str], field_counter: typing.Optional[RuleKeyValueCounter]):

        field_prior[RULE_POSTERIOR] = {value: self.prior_token_count for value in options}

        if field_counter is not None:
            if len(field_counter.value_counts) > 0:
                if (rule_name, field_name) in self.rule_field_value_types:
                    value_type = self.rule_field_value_types[(rule_name, field_name)]
                    if isinstance(value_type, str):
                        if value_type in field_prior[RULE_POSTERIOR]:
                            del field_prior[RULE_POSTERIOR][value_type]  # type: ignore
                        self._create_value_posterior(rule_name, field_name, field_prior, value_type, field_counter, rule_hybrird=True)
                    else:
                        for vt in value_type:
                            if vt in field_prior[RULE_POSTERIOR]:
                                del field_prior[RULE_POSTERIOR][vt]  # type: ignore
                            self._create_value_posterior(rule_name, field_name, field_prior, vt, field_counter, rule_hybrird=True)

                else:
                    raise ValueError(f'{rule_name}.{field_name} has counted values but should only have rules or have a special type')

            for counted_rule, count in field_counter.rule_counts.items():
                if counted_rule not in field_prior[RULE_POSTERIOR]:
                    raise ValueError(f'{rule_name}.{field_name} has counted rule {counted_rule} which is not in the prior {options}')

                field_prior[RULE_POSTERIOR][counted_rule] += count  # type: ignore
        else:
            if self.verbose: print(f'No counted data for {rule_name}.{field_name}')

        if len(field_prior[RULE_POSTERIOR]) == 0:
            del field_prior[RULE_POSTERIOR]

    def parse_prior_to_posterior(self):
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
                if not isinstance(rule_prior, (list, tuple)):
                    raise ValueError(f'Prior for {rule_name} is not a list or tuple')

                rule_prior = rule_prior[0]  # type: ignore

            if rule_name == START:
                pass

            elif isinstance(rule_prior, dict):
                production = None

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
                if OPTIONAL_VOID in rule_prior:
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
                if rule_prior in self.all_rules:
                    rule_prior = dict(rule_posterior={rule_prior: 1}, production=[(RULE, rule_prior)])

                # This is a rule that expands directly to a token
                else:
                    token = rule_prior
                    rule_prior = dict(token_posterior={token: 1}, production=[(TOKEN, token)])
                    self.value_patterns[rule_name] = re.compile(re.escape(token))
                    if self.verbose: print(f'String token rule for {rule_name}: {rule_prior}')

            else:
                raise ValueError(f'Encountered rule with unknown prior or no special case: {rule.name}\n{rule_prior}')

            self.rules[rule.name] = rule_prior

    def parse_rule_prior(self, rule: tatsu.grammars.Node) -> typing.Union[None, str, ContextDict, typing.List]:
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
                        print(f'Encountered child rule parsing to dict with multiple keys: {rule.name}: {parsed_child}')  # type: ignore
                    else:
                        sequence.append((NAMED, list(parsed_child.keys())[0]))

                else:
                    print(f'Encountered child rule parsing to unknown type: {rule.name}: {parsed_child}')  # type: ignore

            rule_dict[PRODUCTION] = sequence
            return rule_dict

        if isinstance(rule, grammars.Named):
            children = rule.children()
            if len(children) > 1:
                raise ValueError(f'Named rule has more than one child: {rule}')

            child_prior = self.parse_rule_prior(children[0])
            # if isinstance(child_prior, str) and child_prior not in self.all_rules:
            #     return child_prior

            return {rule.name: dict(options=child_prior)}

        if isinstance(rule, grammars.Group):
            if len(rule.children()) == 1:
                return self.parse_rule_prior(rule.children()[0])
            else:
                return [self.parse_rule_prior(child) for child in rule.children()]

        if isinstance(rule, grammars.Choice):
            return [self.parse_rule_prior(child) for child in rule.children()]

        if isinstance(rule, grammars.Option):
            children = rule.children()
            if len(children) > 1 :
                raise ValueError(f'Option rule has more than one child: {rule}')

            return self.parse_rule_prior(children[0])

        if isinstance(rule, (grammars.PositiveClosure, grammars.Closure)):
            d = {MIN_LENGTH: 1 if isinstance(rule, grammars.PositiveClosure) else 0}
            if len(rule.children()) == 1:
                child_value = self.parse_rule_prior(rule.children()[0])
                if not isinstance(child_value, dict):
                    print(f'Encoutered positive closure with unexpected value type: {child_value}')

                child_value[next(iter(child_value))].update(d)  # type: ignore
                d = child_value
            else:
                print(f'Encoutered positive closure with multiple children: {rule}')

            return d

        if isinstance(rule, grammars.RuleRef):
            return rule.name

        if isinstance(rule, grammars.Void):
            return OPTIONAL_VOID

        if isinstance(rule, grammars.EmptyClosure):
            return EMPTY_CLOSURE

        raise ValueError(f'Encountered unknown rule type: {type(rule)}: {rule}')

    def _sample_named(self,
        sample_dict: typing.Dict[str, typing.Union[str, typing.Sequence[str], typing.Dict[str, float]]],
        global_context: ContextDict,
        local_context: ContextDict):

        if TYPE_POSTERIOR not in sample_dict:
            raise ValueError(f'Missing type_posterior in sample: {sample_dict}')

        if LENGTH_POSTERIOR in sample_dict:
            length = posterior_dict_sample(self.rng, sample_dict[LENGTH_POSTERIOR])  # type: ignore
            if length == 0:
                return None, None
            values, context_updates = zip(*[self._sample_single_named_value(sample_dict, global_context, local_context) for _ in range(length)])
            context_update = reduce(lambda x, y: {**x, **y}, filter(lambda d: d is not None, context_updates), {})
            return list(values), context_update

        else:
            return self._sample_single_named_value(sample_dict, global_context, local_context)

    def _sample_single_named_value(self, sample_dict: typing.Dict[str, typing.Union[str, typing.Sequence[str], typing.Dict[str, float]]],
        global_context: ContextDict,
        local_context: ContextDict):

        sample_type = posterior_dict_sample(self.rng, sample_dict[TYPE_POSTERIOR])  # type: ignore

        if sample_type == RULE:
            if RULE_POSTERIOR not in sample_dict:
                raise ValueError(f'Missing rule_posterior in sample: {sample_dict}')

            rule = typing.cast(str, posterior_dict_sample(self.rng, sample_dict[RULE_POSTERIOR]))  # type: ignore
            return self.sample(rule, global_context, local_context)

        elif sample_type == TOKEN:
            if TOKEN_POSTERIOR not in sample_dict:
                raise ValueError(f'Missing token_posterior in sample: {sample_dict}')

            token = posterior_dict_sample(self.rng, sample_dict[TOKEN_POSTERIOR])  # type: ignore
            if SAMPLERS in sample_dict and token in sample_dict[SAMPLERS]:  # type: ignore
                token = sample_dict[SAMPLERS][token](global_context, local_context)     # type: ignore

            return token, None

        else:
            raise ValueError(f'Unknown sample type: {sample_type} in sample: {sample_dict}')

    def sample(self, rule: str = START,
        global_context: typing.Optional[ContextDict] = None,
        local_context: typing.Optional[ContextDict] = None):

        if rule == START:
            self.sample_parseinfo_index = 0

        if global_context is None:
            global_context = dict(rng=self.rng)
        elif 'rng' not in global_context:
            global_context['rng'] = self.rng

        if local_context is None:
            local_context = dict()
        else:
            local_context = simplified_context_deepcopy(local_context)

        rule_dict = self.rules[rule]
        production = rule_dict[PRODUCTION]
        output = []
        return_ast = False

        for prod_type, prod_value in production:
            if prod_type == TOKEN:
                if prod_value == EOF:
                    pass
                elif prod_value == SAMPLE:
                    output.append(posterior_dict_sample(self.rng, rule_dict[TOKEN_POSTERIOR]))
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
            out_dict['parseinfo'] = tatsu.infos.ParseInfo(None, rule, self.sample_parseinfo_index, self.sample_parseinfo_index, self.sample_parseinfo_index, self.sample_parseinfo_index)
            self.sample_parseinfo_index += 1
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


def simplified_context_deepcopy(context: dict) -> typing.Dict[str, typing.Union[typing.Dict, typing.Set, int]]:
    context_new = {}

    for k, v in context.items():
        if isinstance(v, dict):
            context_new[k] = dict(v)
        elif isinstance(v, set):
            context_new[k] = set(v)
        elif isinstance(v, int):
            context_new[k] = v
        else:
            raise ValueError('Unexpected value')

    return context_new


ASTNodeInfo = typing.Tuple[tatsu.ast.AST, tatsu.ast.AST, typing.List, ContextDict, ContextDict]
ASTParentMapping = typing.Dict[tatsu.infos.ParseInfo, ASTNodeInfo]

# TODO: move this class to a separate module?
class RegrowthSampler(ASTParentMapper):
    depth_parser: ASTDepthParser
    node_keys: typing.List[tatsu.infos.ParseInfo]
    original_game_id: str
    parent_mapping: ASTParentMapping
    rng: np.random.Generator
    sampler: ASTSampler
    seed: int
    source_ast: typing.Union[tuple, tatsu.ast.AST]

    def __init__(self, sampler: ASTSampler, seed: int = 0):
        super().__init__()
        self.sampler = sampler
        self.seed = seed
        self.rng = np.random.RandomState(seed)  # type: ignore
        self.parent_mapping = dict()
        self.depth_parser = ASTDepthParser()
        self.searcher = ASTParseinfoSearcher()
        self.source_ast = None  # type: ignore

    def set_source_ast(self, source_ast: typing.Union[tuple, tatsu.ast.AST]):
        self.source_ast = source_ast
        self.parent_mapping = {}
        self(source_ast)
        self.node_keys = list(
            key for key, parent_tuple
            in self.parent_mapping.items()
            if isinstance(parent_tuple[1], tatsu.ast.AST)
        )

    def _update_contexts(self, kwargs: typing.Dict[str, typing.Any], retval: typing.Any):
        if retval is not None and isinstance(retval, tuple) and len(retval) == 2:
            global_context_update, local_context_update = retval
            if global_context_update is not None:
                kwargs['global_context'].update(global_context_update)
            if local_context_update is not None:
                kwargs['local_context'].update(local_context_update)

    def __call__(self, ast, **kwargs):
        self._default_kwarg(kwargs, 'global_context', dict())
        self._default_kwarg(kwargs, 'local_context', dict())
        kwargs['local_context'] = simplified_context_deepcopy(kwargs['local_context'])
        retval = super().__call__(ast, **kwargs)
        self._update_contexts(kwargs, retval)
        return retval

    def _build_mapping_value(self, ast, **kwargs):
        return(*super()._build_mapping_value(ast, **kwargs),
            simplified_context_deepcopy(kwargs['global_context']),
            simplified_context_deepcopy(kwargs['local_context']))

    def _handle_iterable(self, ast, **kwargs):
        base_selector = kwargs['selector']
        for i, element in enumerate(ast):
            kwargs['selector'] = base_selector + [i]
            retval = self(element, **kwargs)
            self._update_contexts(kwargs, retval)

        return kwargs['global_context'], kwargs['local_context']

    def _extract_variable_def_variables(self, ast):
        var_names = ast.var_names
        if isinstance(var_names, str):
            var_names = [var_names]

        return {v[1:]: ast.parseinfo.pos for v in var_names}

    def _handle_ast(self, ast, **kwargs):
        rule = ast.parseinfo.rule

        if rule == 'game_def':
            self.original_game_id = ast.game_name

        elif rule == 'preference':
            if 'preference_names' not in kwargs['global_context']:
                kwargs['global_context']['preference_names'] = set()

            kwargs['global_context']['preference_names'].add(ast.pref_name)

        elif rule == 'variable_list':
            if 'variables' not in kwargs['local_context']:
                kwargs['local_context']['variables'] = dict()

            if isinstance(ast.variables, tatsu.ast.AST):
                kwargs['local_context']['variables'].update(self._extract_variable_def_variables(ast.variables))

            else:
                for var_def in ast.variables:
                    kwargs['local_context']['variables'].update(self._extract_variable_def_variables(var_def))

        elif rule == 'variable_type_def':
            if 'variables' not in kwargs['local_context']:
                kwargs['local_context']['variables'] = dict()

            kwargs['local_context']['variables'].update(self._extract_variable_def_variables(ast))

        self._add_ast_to_mapping(ast, **kwargs)

        for key in ast:
            if key != 'parseinfo':
                kwargs['parent'] = ast
                kwargs['selector'] = [key]
                retval = self(ast[key], **kwargs)
                self._update_contexts(kwargs, retval)

        if rule in self.sampler.local_context_propagating_rules:
            return kwargs['global_context'], kwargs['local_context']

        return kwargs['global_context'], None

    def _update_game_id(self, ast: typing.Union[tuple, tatsu.ast.AST], sample_index: int, suffix: typing.Optional[typing.Any] = None):
        new_game_name = f'{self.original_game_id}-{sample_index}{"-" + str(suffix) if suffix else ""}'
        game_key = next(filter(lambda p: p.rule == 'game_def', self.parent_mapping.keys()))
        game_node, _, game_selector, _, _ = self.parent_mapping[game_key]

        new_game_node = tatsu.ast.AST(dict(game_name=new_game_name, parseinfo=game_node.parseinfo))
        return replace_child(ast, game_selector, new_game_node)

    def _sample_node_to_update(self):
        node_index = self.rng.choice(len(self.node_keys))
        node_key = self.node_keys[node_index]
        return self.parent_mapping[node_key]

    def _find_node_depth(self, node: tatsu.ast.AST):
        node_info = self.parent_mapping[self._ast_key(node)]
        depth = 1
        while not (isinstance(node_info[1], tuple) and node_info[1][0] == '(define'):
            node_info = self.parent_mapping[self._ast_key(node_info[1])]
            depth += 1

        return depth

    def sample(self, sample_index: int, external_global_context: typing.Optional[ContextDict] = None,
        external_local_context: typing.Optional[ContextDict] = None, update_game_id: bool = True):

        node, parent, selector, global_context, local_context = self._sample_node_to_update()
        if external_global_context is not None:
            global_context.update(external_global_context)

        if external_local_context is not None:
            local_context.update(external_local_context)

        global_context['original_game_id'] = self.original_game_id

        new_node = self.sampler.sample(node.parseinfo.rule, global_context, local_context)[0]  # type: ignore
        new_source = copy.deepcopy(self.source_ast)
        node_depth = self._find_node_depth(node)
        if update_game_id:
            regrwoth_depth = self.depth_parser(node)
            new_source = self._update_game_id(new_source, sample_index, f'nd-{node_depth}-rd{regrwoth_depth}')
        new_parent = self.searcher(new_source, parseinfo=parent.parseinfo)  # type: ignore
        replace_child(new_parent, selector, new_node)  # type: ignore

        return new_source





def parse_or_load_counter(args: argparse.Namespace, grammar_parser: typing.Optional[tatsu.grammars.Grammar] = None):
    if args.parse_counter:
        if grammar_parser is None:
            raise ValueError('Grammar parser must be provided if parsing counter')

        counter = ASTRuleValueCounter()

        for test_file in args.test_files:
            for ast in cached_load_and_parse_games_from_file(test_file, grammar_parser, not args.dont_tqdm):
                counter(ast)

        with open(args.counter_output_path, 'wb') as out_file:
            pickle.dump(counter, out_file)

    else:
        with open(args.counter_output_path, 'rb') as pickle_file:
            counter = pickle.load(pickle_file)

    return counter


def test_ast_sample(ast, args: argparse.Namespace, grammar_parser: tatsu.grammars.Grammar):
    first_print_out = ''

    try:
        if args.print_samples:
            ast_printer.BUFFER = None
            ast_printer.pretty_print_ast(ast)
            print()

        if args.validate_samples or args.save_samples:
            first_print_out = ast_printer.ast_to_string(ast, line_delimiter='\n')

        if args.validate_samples:
            second_ast = grammar_parser.parse(first_print_out)
            second_print_out = ast_printer.ast_to_string(second_ast, line_delimiter='\n')

            if first_print_out != second_print_out:
                print('Mismatch found')

    except (tatsu.exceptions.FailedToken, tatsu.exceptions.FailedParse) as e:
        print(f'Parse failed: at position {e.pos} expected "{e.item}" :')
        if len(first_print_out) > e.pos:
            print(first_print_out[e.pos:])

    return first_print_out


def _generate_mle_samples(args: argparse.Namespace, sampler: ASTSampler, grammar_parser: tatsu.grammars.Grammar):
    samples = []
    text_samples = []

    sample_iter = range(args.num_samples)
    if args.sample_tqdm:
        sample_iter = tqdm.tqdm(sample_iter, desc='Samples')

    for sample_id in sample_iter:
        generated_sample = False
        while not generated_sample:
            try:
                ast = sampler.sample(global_context=dict(sample_id=sample_id))
                sample_text = test_ast_sample(ast, args, grammar_parser)
                samples.append(ast)
                text_samples.append(sample_text)
                generated_sample = True
            except ValueError as e:
                print(f'ValueError while sampling, repeating: {e}')
            except SamplingException as e:
                print(f'SamplingException while sampling, repeating: {e}')

    return samples, text_samples


def _generate_regrowth_samples(args: argparse.Namespace, sampler: ASTSampler, grammar_parser: tatsu.grammars.Grammar):
    samples = []
    text_samples = []

    regrowth_sampler = RegrowthSampler(sampler, args.random_seed)

    real_games = []

    for test_file in args.test_files:
        for sample_ast in cached_load_and_parse_games_from_file(test_file, grammar_parser, not args.dont_tqdm):
            real_games.append(sample_ast)

    game_iter = enumerate(real_games)
    if args.sample_tqdm:
        game_iter = tqdm.tqdm(game_iter, desc=f'Game #', total=len(real_games))

    for game_index, real_game in game_iter:
        regrowth_sampler.set_source_ast(real_game)
        real_game_str = ast_printer.ast_to_string(real_game, line_delimiter='\n')
        sample_hashes = set([fixed_hash(real_game_str[real_game_str.find('(:domain'):])])

        for sample_index in range(args.num_samples):
            sample_generated = False

            while not sample_generated:
                try:
                    sample_ast = regrowth_sampler.sample(sample_index)
                    sample_str = test_ast_sample(sample_ast, args, grammar_parser)
                    sample_hash = fixed_hash(sample_str[sample_str.find('(:domain'):])

                    if sample_hash in sample_hashes:
                        if args.verbose: print('Regrowth generated identical games, repeating')
                    else:
                        sample_generated = True
                        samples.append(sample_ast)
                        text_samples.append(sample_str + '\n\n')
                        sample_hashes.add(sample_hash)

                except RecursionError:
                    if args.verbose: print('Recursion error, skipping sample')

                except SamplingException:
                    if args.verbose: print('Sampling exception, skipping sample')

    return samples, text_samples


def main(args):
    original_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(args.recursion_limit)

    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)
    counter = parse_or_load_counter(args, grammar_parser)

    sampler = ASTSampler(grammar_parser, counter, seed=args.random_seed)
    samples, text_samples = None, None

    if args.sampling_method == MLE_SAMPLING:
        samples, text_samples = _generate_mle_samples(args, sampler, grammar_parser)
    elif args.sampling_method == REGROWTH_SAMPLING:
        samples, text_samples = _generate_regrowth_samples(args, sampler, grammar_parser)
    else:
        raise ValueError(f'Unknown sampling method: {args.sampling_method}')

    # TODO: conceptual issue: in places where the expansion is recursive
    # (e.g. `setup`` expands to rules that all contain setup, or
    # `preference`` expand to exists/forall that contain preferences)
    # the inner rules (`setup_game_conserved` / `setup_game_optional`),
    # or `then` for preferences) are overrepreesnted, and more likely
    # to be sampled at the root of this expression than they are in the corpus

    sys.setrecursionlimit(original_recursion_limit)

    if args.save_samples:
        if text_samples is None:
            raise ValueError('No samples to save')

        with open(args.samples_output_path, 'w') as out_file:
            text_samples = [sample + '\n\n' for sample in text_samples]
            out_file.writelines(text_samples)

    return


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)

    main(args)
