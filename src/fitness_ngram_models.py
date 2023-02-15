import argparse
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
import itertools
import os
import pickle
import re
import sys
import typing

from nltk.util import ngrams as nltk_ngrams
import numpy as np
import tatsu
import tatsu.ast

import ast_printer
import ast_parser
from ast_utils import cached_load_and_parse_games_from_file


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
DEFAULT_N = 5
parser.add_argument('-n', '--n', type=int, default=DEFAULT_N)
DEFAULT_OUTPUT_PATH_PATTERN = './models/{model_type}_{n}_ngram_model_{today}.pkl'
parser.add_argument('-o', '--output-path', default=None)
DEFAULT_STUPID_BACKOFF_DISCOUNT = 0.4
parser.add_argument('--stupid-backoff-discount', type=float, default=DEFAULT_STUPID_BACKOFF_DISCOUNT)
DEFAULT_ZERO_LOG_PROB = -7
parser.add_argument('--zero-log-prob', type=float, default=DEFAULT_ZERO_LOG_PROB)
parser.add_argument('--from-asts', action='store_true')



WHITESPACE_PATTERN = re.compile(r'\s+')
VARIABLE_PATTERN = re.compile(r'\?[A-Za-z0-9_]+')
PREFERENCE_NAME_PATTERN = re.compile(r'\(preference\s+([A-Za-z0-9_]+)\s+')
NUMBER_AND_DECIMAL_PATTERN = re.compile(r'\-?[0-9]+(\.[0-9]+)?')
NON_TOKEN_CHARACTERS_PATTERN = re.compile(r'\(|\)|\:|\-')


def ngram_preprocess(game_text: str):
    # remove game preamble
    domain_start = game_text.find('(:domain')
    domain_end = game_text.find(')', domain_start)
    game_text = game_text[domain_end + 1:]

    # TODO: convert variables to category types
    # remove variables
    game_text = VARIABLE_PATTERN.sub('', game_text)
    # replace preference names
    for preference_name in PREFERENCE_NAME_PATTERN.findall(game_text):
        game_text = game_text.replace(preference_name, 'preferenceName')
    # remove numbers and decimals
    game_text = NUMBER_AND_DECIMAL_PATTERN.sub('number', game_text)
    # remove non-token characters
    game_text = NON_TOKEN_CHARACTERS_PATTERN.sub('', game_text)
    # standardize whitespace
    game_text = WHITESPACE_PATTERN.sub(' ', game_text)

    return game_text.strip()


START_PAD = '<start>'
END_PAD = '<end>'
UNKNOWN_CATEGORY = '<unknown>'

def _ngrams(text: str, n: int, pad: int = 0, start_pad: str = START_PAD, end_pad: str = END_PAD) -> typing.Iterable[typing.Tuple[str, ...]]:
    tokens = ngram_preprocess(text).split()
    if pad > 0:
        tokens = [start_pad] * pad + tokens + [end_pad] * pad
    return nltk_ngrams(tokens, n)


@dataclass
class NGramTrieNode:
    children: typing.Dict[str, 'NGramTrieNode'] = field(default_factory=dict)
    count: int = 0

    def __getitem__(self, key):
        return self.children[key]

    def __setitem__(self, key, value):
        self.children[key] = value

    def __contains__(self, key):
        return key in self.children


class NGramTrieModel:
    def __init__(self, n: int, stupid_backoff_discount: float = DEFAULT_STUPID_BACKOFF_DISCOUNT,
                 zero_log_prob: float = DEFAULT_ZERO_LOG_PROB, should_pad: bool = True):
        self.n = n
        self.stupid_backoff_discount = stupid_backoff_discount
        self.zero_log_prob = zero_log_prob
        self.should_pad = should_pad

        self.k = None
        self.root = NGramTrieNode(children={}, count=0)
        self.total_token_count = 0

    def _add(self, ngram: typing.Sequence[str], count: int = 1):
        node = self.root
        for i in range(self.n):
            if ngram[i] not in node:
                node[ngram[i]] = NGramTrieNode()

            node = node[ngram[i]]
            node.count += count

    def _add_all(self, ngrams: typing.Iterable[typing.Sequence[str]], count: int = 1):
        ngrams = list(ngrams)
        self.total_token_count += (len(ngrams) - self.n + 1) * count
        for ngram in ngrams:
            self._add(ngram, count)

    def _text_to_ngrams(self, text: str) -> typing.Iterable[typing.Tuple[str, ...]]:
        return _ngrams(text, self.n, pad=self.n - 1 if self.should_pad else 0)

    def fit(self, game_texts: typing.Optional[typing.Sequence[str]] = None,
            ngram_counts: typing.Optional[typing.Dict[typing.Tuple[str, ...], int]] = None,
            n_games: typing.Optional[int] = None):

        self.root = NGramTrieNode(children={}, count=0)
        self.total_token_count = 0

        if game_texts is None and ngram_counts is None:
            raise ValueError('Must provide either game_texts or ngram_counts')

        if game_texts is not None and ngram_counts is not None:
            raise ValueError('Must provide either game_texts or ngram_counts, not both')

        if game_texts is not None:
            for text in game_texts:
                self._add_all(self._text_to_ngrams(text))

        if ngram_counts is not None:
            if n_games is None:
                raise ValueError('Must provide n_games if ngram_counts is provided')

            for ngram, count in ngram_counts.items():
                self._add(ngram, count)

            self.total_token_count = sum([v.count for v in self.root.children.values()]) + ((self.n - 1) * n_games)

    def get(self, ngram: typing.Sequence[str]):
        node = self.root
        for i in range(min(self.n, len(ngram))):
            if ngram[i] not in node:
                return 0
            node = node[ngram[i]]

        return node.count

    def _score_ngram(self, ngram: typing.Sequence[str], stupid_backoff: bool = True, log: bool = False):
        if stupid_backoff:
            discount_factor = 1.0
            start_index = 0
            n = min(self.n, len(ngram))
            ret_val = 0

            while start_index < n:
                if self.get(ngram[start_index:]) > 0:
                    break
                start_index += 1
                discount_factor *= self.stupid_backoff_discount

            if start_index == n:
                ret_val = 0

            elif start_index == n - 1:
                ret_val =  discount_factor * self.get(ngram[start_index:]) / self.total_token_count

            else:
                ret_val = discount_factor * self.get(ngram[start_index:]) / self.get(ngram[start_index:-1])

            if log:
                return np.log(ret_val) if ret_val > 0 else self.zero_log_prob

            return ret_val

        return self.get(ngram) / self.get(ngram[:-1])

    def _transform_ngrams(self, ngrams: typing.Iterable[typing.Sequence[str]],
                  stupid_backoff: bool = True, log: bool = False,
                  reduction: str = 'mean'):
        scores = [self._score_ngram(ngram, stupid_backoff, log) for ngram in ngrams]
        if reduction == 'mean':
            return np.mean(scores)

        return scores

    def transform(self, game_texts: typing.Sequence[str], stupid_backoff: bool = True, log: bool = False,):
        return np.array([self._transform_ngrams(self._text_to_ngrams(text), stupid_backoff, log)
                         for text in game_texts])

    def fit_transform(self, game_texts: typing.Sequence[str]):
        self.fit(game_texts)
        return self.transform(game_texts)

    def find_ngram_counts(self, filter_padding: bool = False, min_length: int = 2, max_length: typing.Optional[int] = None):
        if max_length is None:
            max_length = self.n

        counts = {n: dict() for n in range(min_length, max_length + 1)}
        self._find_ngram_counts(self.root, [], counts, min_length=min_length, max_length=max_length)

        if filter_padding:
            counts = {n: {k: v for k, v in n_counts.items() if START_PAD not in k and END_PAD not in k}
                      for n, n_counts in counts.items()}

        return counts

    def _find_ngram_counts(self, node: NGramTrieNode, ngram: typing.List[str],
                           ngram_counts_by_length: typing.Dict[int, typing.Dict[typing.Tuple[str, ...], int]],
                           min_length: int = 2, max_length: typing.Optional[int] = None):

        if max_length is None:
            max_length = self.n

        current_length = len(ngram)
        if min_length <= current_length <= max_length:
            ngram_counts_by_length[current_length][tuple(ngram)] = node.count

        if current_length < max_length:
            for child, child_node in node.children.items():
                self._find_ngram_counts(child_node, ngram + [child], ngram_counts_by_length)

    def _get_dict_item_value(self, item: typing.Tuple[typing.Tuple[str, ...], int]):
        return item[1]

    def score(self, input_text: typing.Optional[str] = None,
              input_ngrams: typing.Optional[typing.Dict[int, typing.Iterable[typing.Tuple[str, ...]]]] = None,
              k: typing.Optional[int] = None, stupid_backoff: bool = True, log: bool = False,
              filter_padding_top_k: bool = True, top_k_min_n: typing.Optional[int] = None,
              top_k_max_n: typing.Optional[int] = None):

        if input_text is None and input_ngrams is None:
            raise ValueError('Must provide either text or ngrams')

        if input_text is not None and input_ngrams is not None:
            raise ValueError('Must provide either text or ngrams, not both')

        if input_text is not None:
            input_ngrams = {self.n: list(self._text_to_ngrams(input_text))}

        input_ngrams = typing.cast(typing.Dict[int, typing.Iterable[typing.Tuple[str, ...]]], input_ngrams)
        output = dict(score=self._transform_ngrams(input_ngrams[self.n], stupid_backoff, log, reduction='mean'))

        if k is not None:
            if k != self.k:
                self.k = k
                self.ngram_counts = self.find_ngram_counts(filter_padding=filter_padding_top_k)
                self.top_k_ngrams = {n: sorted(n_counts.items(), key=self._get_dict_item_value, reverse=True)[:k]
                                     for n, n_counts in self.ngram_counts.items()}

            if top_k_min_n is None and top_k_max_n is None:
                text_ngram_counts = Counter(input_ngrams[self.n])
                for i, (ngram, _) in enumerate(self.top_k_ngrams[self.n]):
                    output[i] = text_ngram_counts[ngram]  # type: ignore

            else:
                if top_k_min_n is None:
                    top_k_min_n = 2

                if top_k_max_n is None:
                    top_k_max_n = self.n

                for n in range(top_k_min_n, top_k_max_n + 1):
                    text_ngram_counts = Counter(input_ngrams[n])
                    for i, (ngram, _) in enumerate(self.top_k_ngrams[n]):
                        output[f'n_{n}_{i}'] = text_ngram_counts[ngram]  # type: ignore

        return output


# class BaseNGramModel:
#     def __init__(self, n: int, default_logprob: typing.Optional[float] = None):
#         self.n = n
#         self.default_logprob = default_logprob
#         self.k = None
#         self.top_k_ngrams = None

#     def _default_logprob(self):
#         return self.default_logprob

#     def _compute_ngram_counts(self, game_texts: typing.Sequence[str]):
#         return Counter(itertools.chain.from_iterable(_ngrams(text, self.n) for text in game_texts))

#     def fit(self, game_texts: typing.Optional[typing.Sequence[str]] = None,
#             ngram_counts: typing.Optional[typing.Dict[typing.Tuple[str, ...], int]] = None):

#         if game_texts is None and ngram_counts is None:
#             raise ValueError('Must provide either game_texts or ngram_counts')

#         if game_texts is not None and ngram_counts is not None:
#             raise ValueError('Must provide either game_texts or ngram_counts, not both')

#         if game_texts is not None:
#             ngram_counts = self._compute_ngram_counts(game_texts)

#         self.ngram_counts = typing.cast(typing.Dict[typing.Tuple[str, ...], int], ngram_counts)
#         self.total_ngram_counts = sum(self.ngram_counts.values())
#         if self.default_logprob is None:
#             self.default_logprob = np.log(1 / self.total_ngram_counts)
#         self.ngram_logprobs = defaultdict(self._default_logprob, {ngram: np.log(count / self.total_ngram_counts) for ngram, count in self.ngram_counts.items()})

#     def _text_to_ngrams(self, text: str) -> typing.Iterable[typing.Tuple[str, ...]]:
#         return nltk_ngrams(ngram_preprocess(text).split(), self.n)

#     def _transform_ngrams(self, ngrams: typing.Iterable[typing.Tuple[str, ...]], exp: bool = False):
#         mean_logprob = np.mean([self.ngram_logprobs[ngram] for ngram in ngrams])
#         if exp:
#             return np.exp(mean_logprob)
#         return mean_logprob

#     def transform(self, game_texts: typing.Sequence[str], exp: bool = False):
#         return np.array([self._transform_ngrams(self._text_to_ngrams(text), exp) for text in game_texts])

#     def fit_transform(self, game_texts: typing.Sequence[str]):
#         self.fit(game_texts)
#         return self.transform(game_texts)

#     def _get_dict_item_value(self, item: typing.Tuple[typing.Tuple[str, ...], int]):
#         return item[1]

#     def score(self, input_text: typing.Optional[str] = None,
#               input_ngrams: typing.Optional[typing.Iterable[typing.Tuple[str, ...]]] = None,
#               k: typing.Optional[int] = None, exp: bool = False):

#         if input_text is None and input_ngrams is None:
#             raise ValueError('Must provide either text or ngrams')

#         if input_text is not None and input_ngrams is not None:
#             raise ValueError('Must provide either text or ngrams, not both')

#         if input_text is not None:
#             input_ngrams = self._text_to_ngrams(input_text)

#         input_ngrams = list(input_ngrams)  # type: ignore
#         output = dict(score=self._transform_ngrams(input_ngrams, exp=exp))
#         if k is not None:
#             text_ngram_counts = Counter(input_ngrams)
#             if k != self.k:
#                 self.k = k
#                 self.top_k_ngrams = sorted(self.ngram_counts.items(), key=self._get_dict_item_value, reverse=True)[:k]

#             for i, (ngram, _) in enumerate(self.top_k_ngrams):  # type: ignore
#                 output[i] = text_ngram_counts[ngram]  # type: ignore

#         return output


# class TextMultiNGramModel:
#     def __init__(self, n_values: typing.Sequence[int], default_logprob: typing.Union[None, float, typing.Sequence[typing.Union[float, None]]] = None):
#         self.n_values = n_values
#         self.default_logprob = default_logprob

#         if default_logprob is None:
#             default_logprob = [None] * len(n_values)
#         elif isinstance(default_logprob, float):
#             default_logprob = [default_logprob] * len(n_values)
#         else:
#             assert len(default_logprob) == len(n_values)

#         self.models = [BaseNGramModel(n, dlp) for n, dlp in zip(n_values, default_logprob)]

#     def fit(self, game_texts: typing.Sequence[str]):
#         for model in self.models:
#             model.fit(game_texts)

#     def transform(self, game_texts: typing.Sequence[str]):
#         return np.array([model.transform(game_texts) for model in self.models])

#     def fit_transform(self, game_texts: typing.Sequence[str]):
#         self.fit(game_texts)
#         return self.transform(game_texts)

#     def score(self, text: str, k: typing.Optional[int] = None):
#         output_dict = {}
#         for model in self.models:
#             model_output = model.score(input_text=text, k=k)
#             output_dict.update({f'n_{model.n}_{key}': value for key, value in model_output.items()})

#         return output_dict


IGNORE_RULES = [
    'setup', 'setup_statement',
    'type_definition', 'either_types',
    'super_predicate', 'predicate', 'function_eval',
    'pref_def', 'pref_body', 'seq_func',
    'terminal', 'scoring_expr', 'preference_eval',
]


class NGramASTParser(ast_parser.ASTParser):
    def __init__(self, n: int, ignore_rules: typing.Sequence[str] = IGNORE_RULES,
                preorder_traversal: bool = True, pad: int = 0):
        self.n = n
        self.ignore_rules = set(ignore_rules)
        self.preorder_traversal = preorder_traversal
        self.pad = pad

        self.ngram_counts = defaultdict(int)
        self.current_input_ngrams = {}
        self.preorder_ast_tokens = []

    def parse_test_input(self, test_ast: typing.Union[tatsu.ast.AST, tuple],
                         n_values: typing.Optional[typing.Sequence[int]] = None, **kwargs):
        if n_values is not None:
            self.current_input_ngrams = {n: [] for n in n_values}
            self(test_ast, update_model_counts=False, n_values=n_values, **kwargs)
        else:
            self.current_input_ngrams = {self.n: []}
            self(test_ast, update_model_counts=False, **kwargs)
        return self.current_input_ngrams

    def __call__(self, ast, **kwargs):
        self._default_kwarg(kwargs, 'ast_tokens', [])
        self._default_kwarg(kwargs, 'update_model_counts', False)
        initial_call = 'inner_call' not in kwargs or not kwargs['inner_call']
        if initial_call:
            kwargs['inner_call'] = True
            self.preorder_ast_tokens = []

        ast_tokens = kwargs['ast_tokens'] if not self.preorder_traversal else self.preorder_ast_tokens

        if self.pad > 0 and initial_call:
            for _ in range(self.pad):
                ast_tokens.append(START_PAD)

        super().__call__(ast, **kwargs)

        if initial_call:
            if self.pad > 0:
                for _ in range(self.pad):
                    ast_tokens.append(END_PAD)

            if 'n_values' in kwargs:
                for n in kwargs['n_values']:
                    for start_index in range(len(ast_tokens) - n + 1):
                        ngram = tuple(ast_tokens[start_index:start_index + n])
                        self.current_input_ngrams[n].append(ngram)

            else:
                for start_index in range(len(ast_tokens) - self.n + 1):
                    ngram = tuple(ast_tokens[start_index:start_index + self.n])
                    if kwargs['update_model_counts']:
                        self.ngram_counts[ngram] += 1
                    else:
                        self.current_input_ngrams[self.n].append(ngram)

    def _tokenize_ast_node(self, ast: tatsu.ast.AST, **kwargs) -> typing.Union[str, typing.List[str]]:
        rule = ast.parseinfo.rule  # type: ignore
        if rule == 'predicate_or_function_term':
            term = typing.cast(str, ast.term)
            categories = ast_parser.predicate_function_term_to_type_category(
                term, kwargs[ast_parser.VARIABLES_CONTEXT_KEY] if ast_parser.VARIABLES_CONTEXT_KEY in kwargs else {},  # type: ignore
                {}
                )

            if categories is None or len(categories) == 0:
                return UNKNOWN_CATEGORY

            return list(categories)[0]

        if rule == 'variable_type_def':
            var_type = ast.var_type.type  # type: ignore
            if isinstance(var_type, str):
                category_set = ast_parser.predicate_function_term_to_type_category(var_type, {}, {})
                if category_set is None or len(category_set) == 0:
                    return UNKNOWN_CATEGORY
                return list(category_set)[0]

            elif isinstance(var_type, tatsu.ast.AST):
                type_names = var_type.type_names  # type: ignore
                if isinstance(type_names, str):
                    type_names = [type_names]

                categories = []
                for type_name in type_names:  # type: ignore
                    type_categories = ast_parser.predicate_function_term_to_type_category(type_name, {}, {})
                    if type_categories is None or len(type_categories) == 0:
                        categories.append(UNKNOWN_CATEGORY)
                    else:
                        categories.extend(type_categories)

                return ['either_types'] + categories

        return rule

    def _handle_ast(self, ast: tatsu.ast.AST, **kwargs):
        rule = ast.parseinfo.rule  # type: ignore
        ast_tokens = ast_tokens = kwargs['ast_tokens'] if not self.preorder_traversal else self.preorder_ast_tokens

        kwargs = ast_parser.update_context_variables(ast, kwargs)

        if rule not in self.ignore_rules:
            token = self._tokenize_ast_node(ast, **kwargs)
            if isinstance(token, list):
                ast_tokens.extend(token)
            else:
                ast_tokens.append(token)

        kwargs['ast_tokens'] = ast_tokens

        for child_key in ast:
            if child_key != 'parseinfo':
                self(ast[child_key], **kwargs)


class ASTNGramTrieModel:
    def __init__(self, n: int, ignore_rules: typing.Sequence[str] = IGNORE_RULES,
                 stupid_backoff_discount: float = DEFAULT_STUPID_BACKOFF_DISCOUNT,
                 zero_log_prob: float = DEFAULT_ZERO_LOG_PROB, preorder_traversal: bool = True, pad: int = 0):

        self.n = n
        self.ignore_rules = ignore_rules

        self.ngram_ast_parser = NGramASTParser(n, ignore_rules, preorder_traversal, pad)
        self.model = NGramTrieModel(n, stupid_backoff_discount=stupid_backoff_discount, zero_log_prob=zero_log_prob, should_pad=False)

    def fit(self, asts: typing.Sequence[typing.Union[tuple,tatsu.ast.AST]]):
        for ast in asts:
            self.ngram_ast_parser(ast, update_model_counts=True)

        self.model.fit(ngram_counts=self.ngram_ast_parser.ngram_counts, n_games=len(asts))

    def score(self, ast: typing.Union[tuple,tatsu.ast.AST], k: typing.Optional[int] = None,
              stupid_backoff: bool = True, log: bool = False,
              filter_padding_top_k: bool = True, top_k_min_n: typing.Optional[int] = None,
              top_k_max_n: typing.Optional[int] = None):

        n_values = None
        if top_k_min_n is not None:
            if top_k_max_n is None:
                top_k_max_n = self.n

            n_values = list(range(top_k_min_n, top_k_max_n + 1))

        current_input_ngrams = self.ngram_ast_parser.parse_test_input(ast, n_values=n_values)
        return self.model.score(input_ngrams=current_input_ngrams, k=k,
                                stupid_backoff=stupid_backoff, log=log,
                                filter_padding_top_k=filter_padding_top_k,
                                top_k_min_n=top_k_min_n, top_k_max_n=top_k_max_n)


def main(args: argparse.Namespace):
    if args.from_asts:
        model = ASTNGramTrieModel(n=args.n, stupid_backoff_discount=args.stupid_backoff_discount, zero_log_prob=args.zero_log_prob)
    else:
        model = NGramTrieModel(n=args.n, stupid_backoff_discount=args.stupid_backoff_discount, zero_log_prob=args.zero_log_prob)

    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    game_inputs = []
    for test_file in args.test_files:
        if args.from_asts:
            game_inputs.extend(cached_load_and_parse_games_from_file(test_file, grammar_parser, True))
        else:
            game_inputs.extend(ast_printer.ast_to_string(ast, '\n') for ast in cached_load_and_parse_games_from_file(test_file, grammar_parser, False))

    model.fit(game_inputs)
    with open(args.output_path, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    args = parser.parse_args()

    # if not args.n:
    #     args.n = [DEFAULT_N]

    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)

    if args.output_path is None:
        args.output_path = DEFAULT_OUTPUT_PATH_PATTERN.format(model_type='ast' if args.from_asts else 'text',
            n=args.n, today=datetime.now().strftime('%Y_%m_%d'))

    main(args)
