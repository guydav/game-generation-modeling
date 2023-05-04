import argparse
from collections import namedtuple
import gzip
import hashlib
import logging
import numpy as np
import os
import pickle
import tatsu
import tatsu.ast
import tatsu.infos
import tatsu.grammars
import tempfile
import tqdm
import typing

import ast_printer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DEFAULT_TEST_FILES = (
    './problems-few-objects.pddl',
    './problems-medium-objects.pddl',
    './problems-many-objects.pddl'
)


def load_asts(args: argparse.Namespace, grammar_parser: tatsu.grammars.Grammar,
    should_print: bool = False):

    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)

    if should_print:
        results = []
        for test_file in args.test_files:
            for game in load_games_from_file(test_file):
                print(game)
                results.append(grammar_parser.parse(game))
        return results

    else:
        return [grammar_parser.parse(game)
            for test_file in args.test_files
            for game in load_games_from_file(test_file)]


DEFAULT_STOP_TOKENS = ('(define', )  # ('(:constraints', )


def load_games_from_file(path: str, start_token: str='(define',
    stop_tokens: typing.Optional[typing.Sequence[str]] = None,
    remove_comments: bool = True, comment_prefixes_to_keep: typing.Optional[typing.Sequence[str]] = None):

    if stop_tokens is None or not stop_tokens:
        stop_tokens = DEFAULT_STOP_TOKENS

    with open(path) as f:
        lines = f.readlines()
        # new_lines = []
        # for l in lines:
        #     if not l.strip()[0] == ';':
        #         print(l)
        #         new_lines.append(l[:l.find(';')])

        if remove_comments:
            new_lines = [l[:l.find(';')] for l in lines
                if len(l.strip()) > 0 and not l.strip()[0] == ';']

        else:
            new_lines = []
            for l in lines:
                l_s = l.strip()
                if l_s.startswith(';') and (comment_prefixes_to_keep is None or any(l_s.startswith(prefix) for prefix in comment_prefixes_to_keep)):
                    new_lines.append(l.rstrip())
                elif not l_s.startswith(';'):
                    new_lines.append(l[:l.find(';')])

        text = '\n'.join(new_lines)
        start = text.find(start_token)

        while start != -1:
            end_matches = [text.find(stop_token, start + 1) for stop_token in stop_tokens]  # type: ignore
            end_matches = [match != -1 and match or len(text) for match in end_matches]
            end = min(end_matches)
            next_start = text.find(start_token, start + 1)
            if end <= next_start or end == len(text):  # we have a match
                test_case = text[start:end]
                if end < next_start:
                    test_case += ')'

                yield test_case.strip()

            start = next_start




CACHE_FOLDER = os.path.abspath(os.environ.get('GAME_GENERATION_CACHE', os.path.join(tempfile.gettempdir(), 'game_generation_cache')))
logger.debug(f'Using cache folder: {CACHE_FOLDER}')
CACHE_FILE_PATTERN = '{name}-cache.pkl.gz'
CACHE_HASHES_KEY = 'hashes'
CACHE_ASTS_KEY = 'asts'
CACHE_DSL_HASH_KEY = 'dsl'


def _generate_cache_file_name(file_path: str, relative_path: typing.Optional[str] = None):
    if not os.path.exists(CACHE_FOLDER):
        logger.debug(f'Creating cache folder: {CACHE_FOLDER}')
        os.makedirs(CACHE_FOLDER, exist_ok=True)

    name, _ = os.path.splitext(os.path.basename(file_path))
    # if relative_path is not None:
    #     return os.path.join(relative_path, CACHE_FOLDER, CACHE_FILE_PATTERN.format(name=name))
    # else:
    return os.path.join(CACHE_FOLDER, CACHE_FILE_PATTERN.format(name=name))


def _extract_game_id(game_str: str):
    start = game_str.find('(game') + 6
    end = game_str.find(')', start)
    return game_str[start:end]


def fixed_hash(str_data: str):
    return hashlib.md5(bytearray(str_data, 'utf-8')).hexdigest()


def cached_load_and_parse_games_from_file(games_file_path: str, grammar_parser: tatsu.grammars.Grammar,
    use_tqdm: bool, relative_path: typing.Optional[str] = None,
    save_updates_every: int = -1, log_every_change: bool = True):

    cache_path = _generate_cache_file_name(games_file_path, relative_path)
    grammar_hash = fixed_hash(grammar_parser._to_str())

    game_iter = load_games_from_file(games_file_path)
    if use_tqdm:
        game_iter = tqdm.tqdm(game_iter)

    if os.path.exists(cache_path):
        with gzip.open(cache_path, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {CACHE_HASHES_KEY: {}, CACHE_ASTS_KEY: {},
            CACHE_DSL_HASH_KEY: grammar_hash}

    cache_updates = {CACHE_HASHES_KEY: {}, CACHE_ASTS_KEY: {},
            CACHE_DSL_HASH_KEY: grammar_hash}
    n_cache_updates = 0

    cache_updated = False
    grammar_changed = CACHE_DSL_HASH_KEY not in cache or cache[CACHE_DSL_HASH_KEY] != grammar_hash
    if grammar_changed:
        if CACHE_DSL_HASH_KEY not in cache:
            logger.debug('No cached DSL hash found')
        else:
            logger.debug('Grammar changed, clearing cache')

        cache[CACHE_DSL_HASH_KEY] = grammar_hash
        cache_updated = True

    for game in game_iter:
        game_id = _extract_game_id(game)
        game_hash = fixed_hash(game)

        if grammar_changed or game_id not in cache[CACHE_HASHES_KEY] or cache[CACHE_HASHES_KEY][game_id] != game_hash:
            if not grammar_changed and log_every_change:
                if game_id not in cache[CACHE_HASHES_KEY]:
                    logger.debug(f'Game not found in cache: {game_id}')
                else:
                    logger.debug(f'Game changed: {game_id}')
            cache_updated = True
            ast = grammar_parser.parse(game)
            cache_updates[CACHE_HASHES_KEY][game_id] = game_hash
            cache_updates[CACHE_ASTS_KEY][game_id] = ast
            n_cache_updates += 1

        else:
            ast = cache[CACHE_ASTS_KEY][game_id]

        yield ast

        if save_updates_every > 0 and n_cache_updates >= save_updates_every:
            logger.debug(f'Updating cache with {n_cache_updates} new games')
            cache[CACHE_HASHES_KEY].update(cache_updates[CACHE_HASHES_KEY])
            cache[CACHE_ASTS_KEY].update(cache_updates[CACHE_ASTS_KEY])
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)
            cache_updates = {CACHE_HASHES_KEY: {}, CACHE_ASTS_KEY: {},
                CACHE_DSL_HASH_KEY: grammar_hash}
            n_cache_updates = 0
            logger.debug(f'Done updating cache, returning to parsing')

    if n_cache_updates > 0:
        logger.debug(f'Updating cache with {n_cache_updates} new games')
        cache[CACHE_HASHES_KEY].update(cache_updates[CACHE_HASHES_KEY])
        cache[CACHE_ASTS_KEY].update(cache_updates[CACHE_ASTS_KEY])
        cache_updated = True

    if cache_updated:
        logger.debug(f'About to finally update the cache')
        with gzip.open(cache_path, 'wb') as f:
            pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)


def copy_ast(grammar_parser: tatsu.grammars.Grammar, ast: tatsu.ast.AST):
    ast_printer.reset_buffers(True)
    ast_printer.pretty_print_ast(ast)
    ast_str = ''.join(ast_printer.BUFFER)  # type: ignore
    return grammar_parser.parse(ast_str)


def update_ast(ast: tatsu.ast.AST, key: str, value: typing.Any):
    if isinstance(ast, tatsu.ast.AST):
        super(tatsu.ast.AST, ast).__setitem__(key, value)


def apply_selector_list(parent: tatsu.ast.AST, selector: typing.Sequence[typing.Union[str, int]],
    max_index: typing.Optional[int] = None):

    if max_index is None:
        max_index = len(selector)
    for s in selector[:max_index]:
        parent = parent[s]  # type: ignore
    return parent


def replace_child(parent: typing.Union[tuple, tatsu.ast.AST], selector: typing.Union[str, typing.Sequence[typing.Union[str, int]]],
    new_value: typing.Any):

    if isinstance(selector, str):
        selector = (selector,)

    if isinstance(parent, tuple):
        if len(selector) != 1 or not isinstance(selector[0], int):
            raise ValueError('Invalid selector for tuple: {}'.format(selector))

        child_index = selector[0]
        return (*parent[:child_index], new_value, *parent[child_index + 1:])

    last_parent = apply_selector_list(parent, selector, -1)
    last_selector = selector[-1]

    if isinstance(last_selector, str):
        update_ast(last_parent, last_selector, new_value)

    elif isinstance(last_selector, int):
        last_parent[last_selector] = new_value

    else:
        raise ValueError(f'replace_child received last selector of unknown type: {last_selector} ({type(last_selector)})', parent, selector)


def find_all_parents(parent_mapping: typing.Dict[tatsu.infos.ParseInfo, tuple], ast: tatsu.ast.AST):
    parents = []
    parent = parent_mapping[ast.parseinfo][1]  # type: ignore
    while parent is not None and parent != 'root':
        parents.append(parent)
        if isinstance(parent, tuple):
            parent = None
        else:
            parent = parent_mapping[parent.parseinfo][1]

    return parents


def find_selectors_from_root(parent_mapping: typing.Dict[tatsu.infos.ParseInfo, tuple],
    ast: tatsu.ast.AST, root_node: typing.Union[str, tatsu.ast.AST] = 'root'):
    selectors = []
    parent = ast
    while parent != root_node:
        _, parent, selector = parent_mapping[parent.parseinfo]  # type: ignore
        selectors = selector + selectors

    return selectors


def simplified_context_deepcopy(context: dict) -> typing.Dict[str, typing.Union[typing.Dict, typing.Set, int]]:
    context_new = {}

    for k, v in context.items():
        if isinstance(v, dict):
            context_new[k] = dict(v)
        elif isinstance(v, set):
            context_new[k] = set(v)
        elif isinstance(v, (int, float, str, tuple, np.random.Generator)):
            context_new[k] = v
        else:
            raise ValueError(f'Unexpected value type: {v}, {type(v)}')

    return context_new
