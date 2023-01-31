import argparse
from collections import defaultdict, Counter
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
parser.add_argument('-n', '--n', type=int, action='append', default=[])
DEFAULT_OUTPUT_PATH_PATTERN = './models/{model_type}_{n}_ngram_model_{today}.pkl'
parser.add_argument('-o', '--output-path', default=None)
DEFAULT_LOGPROB = 1e-5
parser.add_argument('--default-logprob', type=float, default=DEFAULT_LOGPROB)
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


def _ngrams(text: str, n: int) -> typing.Iterable[typing.Tuple[str, ...]]:
    return nltk_ngrams(ngram_preprocess(text).split(), n)


class TextNGramModel:
    def __init__(self, n: int, default_logprob: float = DEFAULT_LOGPROB):
        self.n = n
        self.default_logprob = default_logprob
        self.k = None
        self.top_k_ngrams = None

    def _default_logprob(self):
        return self.default_logprob

    def _compute_ngram_counts(self, game_texts: typing.Sequence[str]):
        return Counter(itertools.chain.from_iterable(_ngrams(text, self.n) for text in game_texts))

    def fit(self, game_texts: typing.Optional[typing.Sequence[str]] = None,
            ngram_counts: typing.Optional[typing.Dict[typing.Tuple[str, ...], int]] = None):

        if game_texts is None and ngram_counts is None:
            raise ValueError('Must provide either game_texts or ngram_counts')

        if game_texts is not None and ngram_counts is not None:
            raise ValueError('Must provide either game_texts or ngram_counts, not both')

        if game_texts is not None:
            ngram_counts = self._compute_ngram_counts(game_texts)

        self.ngram_counts = typing.cast(typing.Dict[typing.Tuple[str, ...], int], ngram_counts)
        self.total_ngram_counts = sum(self.ngram_counts.values())
        self.ngram_logprobs = defaultdict(self._default_logprob, {ngram: np.log(count / self.total_ngram_counts) for ngram, count in self.ngram_counts.items()})

    def _text_to_ngrams(self, text: str) -> typing.Iterable[typing.Tuple[str, ...]]:
        return nltk_ngrams(ngram_preprocess(text).split(), self.n)

    def _transform_ngrams(self, ngrams: typing.Iterable[typing.Tuple[str, ...]], exp: bool = False):
        mean_logprob = np.mean([self.ngram_logprobs[ngram] for ngram in ngrams])
        if exp:
            return np.exp(mean_logprob)
        return mean_logprob

    def transform(self, game_texts: typing.Sequence[str], exp: bool = False):
        return np.array([self._transform_ngrams(self._text_to_ngrams(text), exp) for text in game_texts])

    def fit_transform(self, game_texts: typing.Sequence[str]):
        self.fit(game_texts)
        return self.transform(game_texts)

    def _get_dict_item_value(self, item: typing.Tuple[typing.Tuple[str, ...], int]):
        return item[1]

    def score(self, input_text: typing.Optional[str] = None,
              input_ngrams: typing.Optional[typing.Iterable[typing.Tuple[str, ...]]] = None,
              k: typing.Optional[int] = None):

        if input_text is None and input_ngrams is None:
            raise ValueError('Must provide either text or ngrams')

        if input_text is not None and input_ngrams is not None:
            raise ValueError('Must provide either text or ngrams, not both')

        if input_text is not None:
            input_ngrams = self._text_to_ngrams(input_text)

        input_ngrams = list(input_ngrams)  # type: ignore
        output = dict(score=self._transform_ngrams(input_ngrams))
        if k is not None:
            text_ngram_counts = Counter(input_ngrams)
            if k != self.k:
                self.k = k
                self.top_k_ngrams = sorted(self.ngram_counts.items(), key=self._get_dict_item_value, reverse=True)[:k]

            for i, (ngram, _) in enumerate(self.top_k_ngrams):  # type: ignore
                output[i] = text_ngram_counts[ngram]  # type: ignore

        return output


class TextMultiNGramModel:
    def __init__(self, n_values: typing.Sequence[int], default_logprob: typing.Union[float, typing.Sequence[float]] = DEFAULT_LOGPROB):
        self.n_values = n_values
        self.default_logprob = default_logprob

        if isinstance(default_logprob, float):
            default_logprob = [default_logprob] * len(n_values)
        else:
            assert len(default_logprob) == len(n_values)

        self.models = [TextNGramModel(n, dlp) for n, dlp in zip(n_values, default_logprob)]

    def fit(self, game_texts: typing.Sequence[str]):
        for model in self.models:
            model.fit(game_texts)

    def transform(self, game_texts: typing.Sequence[str]):
        return np.array([model.transform(game_texts) for model in self.models])

    def fit_transform(self, game_texts: typing.Sequence[str]):
        self.fit(game_texts)
        return self.transform(game_texts)

    def score(self, text: str, k: typing.Optional[int] = None):
        output_dict = {}
        for model in self.models:
            model_output = model.score(input_text=text, k=k)
            output_dict.update({f'n_{model.n}_{key}': value for key, value in model_output.items()})

        return output_dict


IGNORE_RULES = [
    'setup', 'setup_statement',
    'super_predicate', 'predicate', 'function_eval',
    'pref_def', 'pref_body', 'seq_func',
    'terminal', 'scoring_expr', 'preference_eval',
]


class NGramASTParser(ast_parser.ASTParser):
    def __init__(self, n_values: typing.Sequence[int], default_logprob: typing.Union[float, typing.Sequence[float]] = DEFAULT_LOGPROB,
                 ignore_rules: typing.Sequence[str] = IGNORE_RULES):
        self.n_values = n_values
        if isinstance(default_logprob, float):
            default_logprob = [default_logprob] * len(n_values)
        self.default_logprob = default_logprob
        self.ignore_rules = ignore_rules

        self.ngram_counts = {n: defaultdict(int) for n in n_values}
        self.current_input_ngrams = {n: list() for n in n_values}
        self.min_n = min(n_values)
        self.max_n = max(n_values)

        self.models = {}

        if isinstance(default_logprob, float):
            default_logprob = [default_logprob] * len(n_values)
        else:
            assert len(default_logprob) == len(n_values)

    def parse_test_input(self, test_ast: typing.Union[tatsu.ast.AST, tuple]):
        self.current_input_ngrams = {n: list() for n in self.n_values}
        self(test_ast, update_model_counts=False)
        return self.current_input_ngrams

    def __call__(self, ast, **kwargs):
        self._default_kwarg(kwargs, 'rule_chain', [])
        self._default_kwarg(kwargs, 'update_model_counts', False)
        super().__call__(ast, **kwargs)

    def _handle_ast(self, ast: tatsu.ast.AST, **kwargs):
        rule = ast.parseinfo.rule  # type: ignore
        rule_chain = kwargs['rule_chain']

        if rule not in self.ignore_rules:
            rule_chain = rule_chain[:]
            rule_chain.append(rule)
            if len(rule_chain) > self.max_n:
                rule_chain = rule_chain[-self.max_n:]

            if len(rule_chain) >= self.min_n:
                for n in self.n_values:
                    if len(rule_chain) >= n:
                        ngram = tuple(rule_chain[-n:])
                        if kwargs['update_model_counts']:
                            self.ngram_counts[n][ngram] += 1
                        else:
                            self.current_input_ngrams[n].append(ngram)

        kwargs['rule_chain'] = rule_chain

        for child_key in ast:
            if child_key != 'parseinfo':
                self(ast[child_key], **kwargs)


class ASTMultiNGramModel:
    def __init__(self, n_values: typing.Sequence[int], default_logprob: typing.Union[float, typing.Sequence[float]] = DEFAULT_LOGPROB,
                 ignore_rules: typing.Sequence[str] = IGNORE_RULES):

        self.n_values = n_values
        if isinstance(default_logprob, float):
            default_logprob = [default_logprob] * len(n_values)

        self.default_logprob = default_logprob
        self.ignore_rules = ignore_rules

        self.ngram_ast_parser = NGramASTParser(n_values, default_logprob, ignore_rules)
        self.models = {n: TextNGramModel(n, dlp) for n, dlp in zip(n_values, default_logprob)}

    def fit(self, asts: typing.Sequence[typing.Union[tuple,tatsu.ast.AST]]):
        for ast in asts:
            self.ngram_ast_parser(ast, update_model_counts=True)

        for n, model in self.models.items():
            model.fit(ngram_counts=self.ngram_ast_parser.ngram_counts[n])

    def score(self, ast: typing.Union[tuple,tatsu.ast.AST], k: typing.Optional[int] = None):
        current_input_ngrams = self.ngram_ast_parser.parse_test_input(ast)

        output_dict = {}
        for n, model in self.models.items():
            model_output = model.score(input_ngrams=current_input_ngrams[n], k=k)
            output_dict.update({f'n_{n}_{key}': value for key, value in model_output.items()})

        return output_dict


def main(args: argparse.Namespace):
    if args.from_asts:
        model = ASTMultiNGramModel(n_values=args.n, default_logprob=args.default_logprob)
    else:
        if len(args.n) == 1:
            model = TextNGramModel(n=args.n[0], default_logprob=args.default_logprob)
        else:
            model = TextMultiNGramModel(n_values=args.n, default_logprob=args.default_logprob)

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

    if not args.n:
        args.n = [DEFAULT_N]

    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)

    if args.output_path is None:
        args.output_path = DEFAULT_OUTPUT_PATH_PATTERN.format(model_type='ast' if args.from_asts else 'text',
            n='_'.join([str(n) for n in args.n]), today=datetime.now().strftime('%Y_%m_%d'))

    main(args)
