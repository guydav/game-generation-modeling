from argparse import Namespace
import io
import os
import pstats
import tatsu
import time
import tqdm
import sys

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./src'))

from src import fitness_energy_utils as utils
from src.fitness_energy_utils import NON_FEATURE_COLUMNS
from src.ast_counter_sampler import *
from src.ast_utils import cached_load_and_parse_games_from_file, load_games_from_file, _extract_game_id
from src import ast_printer
from src.fitness_features import *


PROFILE = False


if __name__ == '__main__':
    grammar = open('./dsl/dsl.ebnf').read()
    grammar_parser = tatsu.compile(grammar)
    game_asts = list(cached_load_and_parse_games_from_file('./dsl/interactive-beta.pddl', grammar_parser, False, relative_path='.'))

    args = Namespace(no_binarize=False, no_merge=False, use_specific_objects_ngram_model=False)
    featurizer = build_fitness_featurizer(args)
    np.seterr(all='raise')

    profile = None
    if PROFILE:
        import cProfile
        profile = cProfile.Profile()
        profile.enable()

    start = time.perf_counter()
    for i in range(0, len(game_asts)):
    # for i in tqdm(range(0, len(game_asts))):
        print(f'Parsing game #{i}: {game_asts[i][1].game_name}')
        _ = featurizer.parse(game_asts[i], 'interactive-beta.pddl', return_row=False)

    end = time.perf_counter()
    print(f'\nParsing took {end - start:.4f} seconds\n')

    if profile is not None:
        profile.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.TIME
        ps = pstats.Stats(profile, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    d = featurizer.to_df()
    print(d[[c for c in d.columns if 'predicate_found_in_data_' in c]].describe())
