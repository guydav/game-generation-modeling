import argparse
import json
import logging
import os
from pprint import pformat
import sys
import typing

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from src import fitness_energy_utils as utils

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')


parser = argparse.ArgumentParser()
parser.add_argument('--fitness-features-file', type=str, default='./data/fitness_features_1024_regrowths.csv.gz')
parser.add_argument('--output-name', type=str, required=True)
parser.add_argument('--output-folder', type=str, default='./data/fitness_cv')
parser.add_argument('--output-relative-path', type=str, default='.')
parser.add_argument('--feature-score-threshold', type=float, required=True)
parser.add_argument('--device', type=str, required=False)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--random-seed', type=int, default=utils.DEFAULT_RANDOM_SEED)
LOSS_FUNCTIONS = [x for x in dir(utils) if 'loss' in x]
parser.add_argument('--default-loss-function', type=str, choices=LOSS_FUNCTIONS, default='fitness_softmin_loss')
parser.add_argument('--cv-settings-json', type=str, default=os.path.join(os.path.dirname(__file__), 'fitness_cv_settings.json'))


def get_features_by_abs_diff_threshold(diffs: pd.Series, score_threshold: float) -> typing.List[str]:
    feature_columns = list(diffs[diffs >= score_threshold].index)

    remove_all_ngram_scores = []
    for score_type in ('full', 'setup', 'constraints', 'terminal', 'scoring'):
        col_names = sorted([c for c in feature_columns if c.startswith(f'ast_ngram_{score_type}') and c.endswith('_score')])

        if score_type not in remove_all_ngram_scores:
            col_names = col_names[:-1]

        for col in col_names:
            feature_columns.remove(col)

    return feature_columns


def get_feature_columns(df: pd.DataFrame, score_threshold: float) -> typing.List[str]:
    mean_features_by_real = df[['real'] + [c for c in df.columns if c not in utils.NON_FEATURE_COLUMNS]].groupby('real').mean()
    feature_diffs = mean_features_by_real.loc[1] - mean_features_by_real.loc[0]
    abs_diffs = feature_diffs.abs()
    return get_features_by_abs_diff_threshold(abs_diffs, score_threshold)


def main(args: argparse.Namespace):
    fitness_df = utils.load_fitness_data('./data/fitness_features_1024_regrowths.csv.gz')
    logging.info(f'Unique source files: {fitness_df.src_file.unique()}')
    logging.info(f'Dataframe shape: {fitness_df.shape}')
    original_game_counts = fitness_df.groupby('original_game_name').src_file.count().value_counts()
    if len(original_game_counts) == 1:
        logging.debug(f'All original games have {original_game_counts.index[0] - 1} regrowths')  # type: ignore
    else:
        raise ValueError('Some original games have different numbers of regrowths: {original_game_counts}')

    feature_columns = get_feature_columns(fitness_df, args.feature_score_threshold)
    logging.debug(f'Fitting models with {len(feature_columns)} features')

    with open(args.cv_settings_json, 'r') as f:
        cv_settings = json.load(f)

    logging.debug(f'CV settings:\n{pformat(cv_settings)}')

    param_grid = cv_settings['param_grid']
    cv_kwargs = cv_settings['cv_kwargs']
    train_kwargs = cv_settings['train_kwargs']

    if 'beta' not in train_kwargs and 'fitness__beta' not in param_grid:
        train_kwargs['beta'] = args.beta

    if 'device' in train_kwargs:
        train_kwargs['device'] = torch.device(train_kwargs['device'])
    else:
        train_kwargs['device'] = args.device

    if 'fitness__loss_function' in param_grid:
        param_grid['fitness__loss_function'] = [getattr(utils, x) for x in param_grid['fitness__loss_function']]
    elif 'loss_function' not in train_kwargs:
        train_kwargs['loss_function'] = getattr(utils, args.default_loss_function)

    scaler_kwargs = dict(passthrough=True)
    model_kwargs = dict(output_activation=nn.Identity())

    # scoring = utils.build_multiple_scoring_function(
    #     [utils.wrap_loss_function_to_metric(utils.fitness_sofmin_loss_positive_negative_split, dict(beta=args.beta), True),  # type: ignore
    #     utils.evaluate_fitness_overall_ecdf, utils.evaluate_fitness_single_game_rank, utils.evaluate_fitness_single_game_min_rank,
    #     utils.wrap_loss_function_to_metric(utils.energy_of_negative_at_quantile, dict(quantile=0.01), True),  # type: ignore
    #     utils.wrap_loss_function_to_metric(utils.energy_of_negative_at_quantile, dict(quantile=0.05), True),  # type: ignore
    #     ],
    #     ['loss', 'overall_ecdf', 'single_game_rank', 'single_game_min_rank', 'energy_of_negative@1%', 'energy_of_negative@5%'],
    # )

    cv, (train_tensor, test_tensor), results = utils.model_fitting_experiment(
        fitness_df,
        param_grid, feature_columns=feature_columns,
        scoring_function=utils.default_multiple_scoring,
        verbose=1, scaler_kwargs=scaler_kwargs,
        model_kwargs=model_kwargs, train_kwargs=train_kwargs, cv_kwargs=cv_kwargs,
        random_seed=args.random_seed,
        )

    utils.print_results_dict(results, notebook=False)
    cv.scorer_ = None
    cv.scoring = None

    output_data = dict(cv=cv, train_tensor=train_tensor, test_tensor=test_tensor, results=results, feature_columns=feature_columns)
    utils.save_data(output_data, folder=args.output_folder, name=args.output_name, relative_path=args.output_relative_path)


if __name__ == '__main__':
    args = parser.parse_args()

    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)

    args_str = '\n'.join([f'{" " * 26}{k}: {v}' for k, v in vars(args).items()])
    logging.debug(f'Shell arguments:\n{args_str}')

    main(args)
