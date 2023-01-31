from bisect import bisect
from functools import reduce
import re
import typing

import numpy as np
import pandas as pd


## -- Binarization and other preprocessing -- ##


NON_FEATURE_COLUMNS = set(['Index', 'src_file', 'game_name', 'domain_name', 'real', 'original_game_name'])

BINARIZE_IGNORE_FEATURES = [
    'setup_objects_used', 'starts_and_ends_once', 'correct_predicate_function_arity',
    'section_without_pref_or_total_count_terminal', 'section_without_pref_or_total_count_scoring'
]

BINARIZE_IGNORE_PATTERNS = [
    re.compile(r'max_depth_[\w\d_]+'),
    re.compile(r'mean_depth_[\w\d_]+'),
    re.compile(r'node_count_[\w\d_]+')
]

BINARIZE_NON_ONE = [
    'all_variables_defined', 'all_variables_used',
    'all_preferences_used', 'no_adjacent_once', 'variable_not_repeated',
    'no_nested_logicals', 'no_identical_logical_children',
    'count_once_per_external_objects_used_correctly',
    'external_forall_used_correctly', 'pref_forall_used',
    'pref_forall_correct_arity', 'pref_forall_correct_types', 'no_two_number_operations',
    'tautological_expression_found', 'redundant_expression_found',
]

SCALE_ZERO_ONE_PATTERNS = [
    re.compile(r'(ast|text)_ngram_n_\d+_score'),
]

BINRARIZE_NONZERO_PATTERNS = [
    re.compile(r'[\w\d+_]+_arg_types_[\w_]+'),
    re.compile(r'compositionality_structure_\d+'),
    re.compile(r'(ast|text)_ngram_n_\d+_\d+')
]


def _binarize_series(series: pd.Series, ignore_columns: typing.Iterable[str] = NON_FEATURE_COLUMNS):
    c = str(series.name)
    if c in ignore_columns:
        return series

    if c in BINARIZE_IGNORE_FEATURES:
        return series

    if any([p.match(c) for p in BINARIZE_IGNORE_PATTERNS]):
        return series

    if c in BINARIZE_NON_ONE:
        return (series == 1).astype(int)

    if any([p.match(c) for p in SCALE_ZERO_ONE_PATTERNS]):
        min_val, max_val = series.min(), series.max()
        return (series - min_val) / (max_val - min_val)

    if any([p.match(c) for p in BINRARIZE_NONZERO_PATTERNS]):
        return (series != 0).astype(int)

    raise ValueError(f'No binarization rule for column {c}')


def binarize_features(df: pd.DataFrame, ignore_columns: typing.Iterable[str] = NON_FEATURE_COLUMNS) -> pd.DataFrame:
    binarized_df = df.apply(_binarize_series, axis=0, ignore_columns=ignore_columns)
    return binarized_df


DEFAULT_MERGE_THRESHOLD = 10
DEFAULT_FEATURE_SUFFIXES = ('setup', 'constraints')
DEFAULT_MERGE_COLUMN_SUFFIX = 'other'


def _merge_single_prefix(df: pd.DataFrame, feature_prefix: str, threshold: int = DEFAULT_MERGE_THRESHOLD,
    merge_function: typing.Callable = np.logical_or, merged_column_suffix: str = DEFAULT_MERGE_COLUMN_SUFFIX,
    feature_suffix: str = '') -> None:

    merged_column_key = f'{feature_prefix}_{merged_column_suffix}{"_" + feature_suffix if feature_suffix else ""}'
    prefix_feature_names = [c for c in df.columns if c.startswith(feature_prefix) and c.endswith(feature_suffix)]
    if len(prefix_feature_names) == 0:
        raise ValueError(f'No features found for prefix {feature_prefix} and suffix {feature_suffix}')
    merged_column_insert_index = bisect(prefix_feature_names, merged_column_key)
    first_prefix_feature_index = list(df.columns).index(prefix_feature_names[0])
    insert_index = first_prefix_feature_index + merged_column_insert_index

    counts = df[[c for c in df.columns if c.startswith(feature_prefix) and c.endswith(feature_suffix)]].sum()
    keys_to_merge = counts.index[counts < threshold]  # type: ignore
    if len(keys_to_merge) == 0:
        print(feature_prefix)
        return
    new_series_values = reduce(merge_function, [df[k] for k in keys_to_merge[1:]], df[keys_to_merge[0]]).astype(int)

    df.insert(insert_index, merged_column_key, new_series_values)
    df.drop(keys_to_merge, axis=1, inplace=True)


def merge_sparse_features(df: pd.DataFrame, predicates: typing.Sequence[str],
    threshold: int = DEFAULT_MERGE_THRESHOLD, merge_function: typing.Callable = np.logical_or,
    merged_column_suffix: str = 'other', feature_suffixes: typing.Sequence[str] = DEFAULT_FEATURE_SUFFIXES
    ) -> pd.DataFrame:

    df = df.copy(deep=True)

    for feature_suffix in feature_suffixes:
        for p in predicates:
            feature_prefix = f'{p}_arg_types'
            _merge_single_prefix(df, feature_prefix, threshold, merge_function, merged_column_suffix, feature_suffix)

            # if p not in PREDICATE_FUNCTION_ARITY_MAP:
            #     raise ValueError(f'Predicate {p} not in arity map')

            # arity = PREDICATE_FUNCTION_ARITY_MAP[p]
            # if arity == 1:
            #     feature_prefix = f'arg_types_{p}'
            #     _merge_single_prefix(df, feature_prefix, threshold, merge_function, merged_column_suffix, feature_suffix)

            # else:  # arity = 2/3
            #     for c in CATEGORIES_TO_TYPES.keys():
            #         if c == EMPTY_OBJECT:
            #             continue
            #         feature_prefix = f'arg_types_{p}_{c}'
            #         _merge_single_prefix(df, feature_prefix, threshold, merge_function, merged_column_suffix, feature_suffix)

    return df
