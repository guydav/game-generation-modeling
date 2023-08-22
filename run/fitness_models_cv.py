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

import ast_printer  # for logging
import ast_parser  # for logging
from src import fitness_energy_utils as utils
from src import latest_model_paths

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--fitness-features-file', type=str, default=latest_model_paths.LATEST_FITNESS_FEATURES)
parser.add_argument('--output-name', type=str, required=True)
parser.add_argument('--output-folder', type=str, default='./data/fitness_cv')
parser.add_argument('--output-relative-path', type=str, default='.')
parser.add_argument('--feature-score-threshold', type=float, required=True)
parser.add_argument('--device', type=str, required=False)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--random-seed', type=int, default=utils.DEFAULT_RANDOM_SEED)
parser.add_argument('--ngram-scores-to-remove', type=str, nargs='+', default=[])
LOSS_FUNCTIONS = [x for x in dir(utils) if 'loss' in x]
parser.add_argument('--ignore-features', type=str, nargs='+', default=[])
parser.add_argument('--default-loss-function', type=str, choices=LOSS_FUNCTIONS, default='fitness_softmin_loss')
parser.add_argument('--output-activation', type=str, default=None)
parser.add_argument('--output-scaling', type=float, default=1.0)
parser.add_argument('--cv-settings-json', type=str, default=os.path.join(os.path.dirname(__file__), 'fitness_cv_settings.json'))
parser.add_argument('--no-save-full-model', action='store_true')
parser.add_argument('--full-model-without-test', action='store_true')


def get_features_by_abs_diff_threshold(diffs: pd.Series, score_threshold: float,
                                       ngram_scores_to_remove: typing.Optional[typing.List[str]] = None) -> typing.List[str]:
    if ngram_scores_to_remove is None:
        ngram_scores_to_remove = []

    feature_columns = list(diffs[diffs >= score_threshold].index)

    for score_type in ('full', 'setup', 'constraints', 'terminal', 'scoring'):
        col_names = sorted([c for c in feature_columns if c.startswith(f'ast_ngram_{score_type}') and c.endswith('_score')])

        if score_type not in ngram_scores_to_remove:
            col_names = col_names[:-1]

        for col in col_names:
            feature_columns.remove(col)

    return feature_columns


def get_feature_columns(df: pd.DataFrame, score_threshold: float,
                        ngram_scores_to_remove: typing.Optional[typing.List[str]] = None) -> typing.List[str]:
    mean_features_by_real = df[['real'] + [c for c in df.columns if c not in utils.NON_FEATURE_COLUMNS]].groupby('real').mean()
    feature_diffs = mean_features_by_real.loc[1] - mean_features_by_real.loc[0]
    abs_diffs = feature_diffs.abs()
    return get_features_by_abs_diff_threshold(abs_diffs, score_threshold, ngram_scores_to_remove)  # type: ignore


def main(args: argparse.Namespace):
    logger.info(f'Loading fitness data from {args.fitness_features_file}')
    fitness_df = utils.load_fitness_data(args.fitness_features_file)
    logger.info(f'Unique source files: {fitness_df.src_file.unique()}')
    logger.info(f'Dataframe shape: {fitness_df.shape}')
    original_game_counts = fitness_df.groupby('original_game_name').src_file.count().value_counts()
    if len(original_game_counts) == 1:
        logger.debug(f'All original games have {original_game_counts.index[0] - 1} regrowths')  # type: ignore
    else:
        raise ValueError('Some original games have different numbers of regrowths: {original_game_counts}')

    feature_columns = get_feature_columns(fitness_df, args.feature_score_threshold, args.ngram_scores_to_remove)

    if args.ignore_features:
        logger.debug(f'Ignoring features: {args.ignore_features}')
        feature_columns = [c for c in feature_columns if c not in args.ignore_features]

    logger.debug(f'Fitting models with {len(feature_columns)} features')

    with open(args.cv_settings_json, 'r') as f:
        cv_settings = json.load(f)

    logger.debug(f'CV settings:\n{pformat(cv_settings)}')

    param_grid = cv_settings['param_grid']
    cv_kwargs = cv_settings['cv_kwargs']
    train_kwargs = cv_settings['train_kwargs']

    if 'beta' not in train_kwargs and 'fitness__beta' not in param_grid:
        train_kwargs['beta'] = args.beta

    if 'device' in train_kwargs:
        train_kwargs['device'] = torch.device(train_kwargs['device'])
    else:
        train_kwargs['device'] = args.device

    if 'regularizer' in train_kwargs:
        if 'regularization_weight' not in train_kwargs:
            raise ValueError('regularizer is specified but regularization_weight is not')

        threshold = None
        if 'regularization_threshold' in train_kwargs:
            threshold = train_kwargs.pop('regularization_threshold')

        train_kwargs['regularizer'] = utils.ModelRegularizer(train_kwargs['regularizer'], threshold)

    if 'fitness__loss_function' in param_grid:
        param_grid['fitness__loss_function'] = [getattr(utils, x) for x in param_grid['fitness__loss_function']]
    elif 'loss_function' not in train_kwargs:
        train_kwargs['loss_function'] = getattr(utils, args.default_loss_function)

    scaler_kwargs = dict(passthrough=True)

    output_activation = nn.Identity()
    if args.output_activation is not None:
        if args.output_activation == 'sigmoid':
            output_activation = nn.Sigmoid()

        elif args.output_activation == 'tanh':
            output_activation = nn.Tanh()

        else:
            raise ValueError(f'Unknown output activation: {args.output_activation}')

    model_kwargs = dict(output_activation=output_activation, output_scaling=args.output_scaling)

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

    logger.info(f'Best params: {cv.best_params_}')

    utils.visualize_cv_outputs(cv, train_tensor, test_tensor, results, notebook=False)
    cv.scorer_ = None  # type: ignore
    cv.scoring = None  # type: ignore

    output_data = dict(cv=cv, train_tensor=train_tensor, test_tensor=test_tensor, results=results, feature_columns=feature_columns)
    utils.save_data(output_data, folder=args.output_folder, name=args.output_name, relative_path=args.output_relative_path)

    if not args.no_save_full_model:
        logger.debug('Saving full model')
        if not args.full_model_without_test:
            logger.debug('Fitting full model with entire dataset (including test data)')
            full_tensor = utils.df_to_tensor(fitness_df, feature_columns)
            cv.best_estimator_['fitness'].train_kwargs['split_validation_from_train'] = False  # type: ignore
            cv.best_estimator_.fit(full_tensor)  # type: ignore
            print(utils.evaluate_trained_model(cv.best_estimator_, full_tensor, utils.default_multiple_scoring))  # type: ignore

            full_tensor_scores = cv.best_estimator_.transform(full_tensor).detach()  # type: ignore
            real_game_scores = full_tensor_scores[:, 0]
            print(f'Real game scores: {real_game_scores.mean():.4f} ± {real_game_scores.std():.4f}, min = {real_game_scores.min():.4f}, max = {real_game_scores.max():.4f}')

            negatives_scores = full_tensor_scores[:, 1:].ravel()
            print(torch.quantile(negatives_scores, torch.linspace(0, 1, 11)))
            print(torch.quantile(negatives_scores, 0.2))

        model_name = args.output_name
        if 'fitness_sweep_' in model_name:
            model_name = model_name.replace('fitness_sweep_', '')

        model_name = f'{utils.DEFAULT_SAVE_MODEL_NAME}_{model_name}'
        utils.save_model_and_feature_columns(cv, feature_columns, name=model_name, relative_path=args.output_relative_path)


if __name__ == '__main__':
    args = parser.parse_args()

    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)

    args_str = '\n'.join([f'{" " * 26}{k}: {v}' for k, v in vars(args).items()])
    logger.debug(f'Shell arguments:\n{args_str}')

    main(args)
