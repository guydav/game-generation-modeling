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

import ast_printer
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
DEFAULT_OUTPUT_PATH_PATTERN = './models/text_{n}_ngram_model_{today}.pkl'
parser.add_argument('-o', '--output-path', default=None)
DEFAULT_LOGPROB = 1e-5
parser.add_argument('--default-logprob', type=float, default=DEFAULT_LOGPROB)



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

    def fit(self, game_texts: typing.Sequence[str]):
        self.ngram_counts = self._compute_ngram_counts(game_texts)
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

    def score(self, text: str, k: typing.Optional[int] = None):
        text_ngrams = list(self._text_to_ngrams(text))
        output = dict(score=self._transform_ngrams(text_ngrams))
        if k is not None:
            text_ngram_counts = Counter(text_ngrams)
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
            model_output = model.score(text, k)
            output_dict.update({f'n_{model.n}_{key}': value for key, value in model_output.items()})

        return output_dict


def main(args: argparse.Namespace):
    if len(args.n) == 1:
        model = TextNGramModel(n=args.n, default_logprob=args.default_logprob)
    else:
        model = TextMultiNGramModel(n_values=args.n, default_logprob=args.default_logprob)

    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    game_texts = []
    for test_file in args.test_files:
        game_texts.extend(ast_printer.ast_to_string(ast, '\n') for ast in cached_load_and_parse_games_from_file(test_file, grammar_parser, False))

    model.fit(game_texts)
    with open(args.output_path, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    args = parser.parse_args()

    if not args.n:
        args.n = [DEFAULT_N]

    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)

    if args.output_path is None:
        args.output_path = DEFAULT_OUTPUT_PATH_PATTERN.format(n="_".join([str(n) for n in args.n]), today=datetime.now().strftime('%Y_%m_%d'))

    main(args)
