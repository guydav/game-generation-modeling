import re

from fitness_features import COMMON_SENSE_PREDICATES_FUNCTIONS
from fitness_features_preprocessing import NGRAM_SCORE_PATTERN, ARG_TYPES_PATTERN

GRAMMAR_CONTEXT_FEATURES = [
    'variables_defined_all',
    'variables_defined_prop',
    'variables_used_all',
    'variables_used_prop',
    'preferences_used_all',
    'preferences_used_prop',

    'repeated_variables_found',
    'repeated_variable_type_in_either',

    'section_without_pref_or_total_count_terminal',
    'section_without_pref_or_total_count_scoring',
    'section_doesnt_exist_setup',
    'section_doesnt_exist_terminal',
]

NGRAM_AND_DATA_BASED_FEATURES = [
    NGRAM_SCORE_PATTERN,
    re.compile(r'predicate_found_in_data_[\w\d_]+'),
]

COUNTING_FEATURES = [
    # How many preferences are defined
    re.compile(r'num_preferences_defined_[\d_]+'),
    # How many modals are under a then
    re.compile(r'length_of_then_modals_[\w\d_]+'),
    # Various features related to variable quantifications
    re.compile(r'max_quantification_count_[\w\d_]+'),
    re.compile(r'max_number_variables_types_quantified_[\w\d_]+'),
    # Man and max depth and node count
    re.compile(r'max_depth_[\w\d_]+'),
    re.compile(r'mean_depth_[\w\d_]+'),
    re.compile(r'node_count_[\w\d_]+'),
    re.compile(r'max_width_[\w\d_]+'),
]

# A Smaller set of counting features to exclude
COUNTING_LESS_IMPORTANT_FEATURES = [
    # How many preferences are defined
    # re.compile(r'num_preferences_defined_[\d_]+'),
    # How many modals are under a then
    re.compile(r'length_of_then_modals_[\w\d_]+'),
    # Various features related to variable quantifications
    re.compile(r'max_quantification_count_[\w\d_]+'),
    re.compile(r'max_number_variables_types_quantified_[\w\d_]+'),
    # Man and max depth and node count
    re.compile(r'max_depth_[\w\d_]+'),
    # re.compile(r'mean_depth_[\w\d_]+'),
    re.compile(r'node_count_[\w\d_]+'),
    # re.compile(r'max_width_[\w\d_]+'),
]

COUNTING_FEATURES_PATTERN_DICT = {
    # How many preferences are defined
    'num_preferences_defined': re.compile(r'num_preferences_defined_[\d_]+'),
    # How many modals are under a then
    'modals under `then`': re.compile(r'length_of_then_modals_[\w\d_]+'),
    # Various features related to variable quantifications
    'variables quantified': re.compile(r'max_quantification_count_[\w\d_]+'),
    'variable types quantified': re.compile(r'max_number_variables_types_quantified_[\w\d_]+'),
    # Man and max depth and node count
    'max depth by section': re.compile(r'max_depth_[\w\d_]+'),
    'mean depth by section': re.compile(r'mean_depth_[\w\d_]+'),
    'node count by section': re.compile(r'node_count_[\w\d_]+'),
    'max width by section': re.compile(r'max_width_[\w\d_]+'),
}

FORALL_FEATURES = [
    re.compile(r'pref_forall_[\w\d_]+_correct$'),
    re.compile(r'pref_forall_[\w\d_]+_incorrect$'),
    re.compile(r'[\w\d_]+_incorrect_count$'),
]


FORALL_LESS_IMPORTANT_FEATURES = [
    # re.compile(r'pref_forall_[\w\d_]+_correct$'),
    # re.compile(r'pref_forall_[\w\d_]+_incorrect$'),
    re.compile(r'[\w\d_]+_incorrect_count$'),
]


PREDICATE_UNDER_MODAL_FEATURES = [
    re.compile(r'predicate_under_modal_[\w\d_]+'),
]

PREDICATE_ROLE_FILLER_FEATURES = [
    ARG_TYPES_PATTERN
]

PREDICATE_ROLE_FILLER_PATTERN_DICT = {
    pred: re.compile(f'{pred}_arg_types_[\\w_]+')
    for pred in COMMON_SENSE_PREDICATES_FUNCTIONS
}

COMPOSITIONALITY_FEATURES = [
    re.compile(r'compositionality_structure_\d+'),
]

GRAMMAR_USE_FEATURES = [
    'setup_objects_used',
    'setup_quantified_objects_used',
    'any_setup_objects_used',
    'adjacent_once_found',
    'no_adjacent_same_modal',
    'starts_and_ends_once',
    'once_in_middle_of_pref_found',
    'pref_without_hold_found',
    'at_end_found',

    'nested_logicals_found',
    'identical_logical_children_found',
    'identical_scoring_children_found',
    'scoring_count_expression_repetitions_exist',
    'tautological_expression_found',
    'redundant_expression_found',
    'redundant_scoring_terminal_expression_found',
    'unnecessary_expression_found',
    'identical_consecutive_seq_func_predicates_found',
    'disjoint_preferences_found',
    'disjoint_preferences_prop',
    'disjoint_preferences_scoring_terminal_types',
    'disjoint_preferences_scoring_terminal_predicates',
    'disjoint_preferences_same_predicates_only',
    'disjoint_seq_funcs_found',
    'disjoint_at_end_found',
    'disjoint_modal_predicates_found',
    'disjoint_modal_predicates_prop',

    'two_number_operation_found',
    'single_argument_multi_operation_found',
]

# The ones that are kept are the ones commented out
GRAMMAR_USE_LESS_IMPORTANT_FEATURES = [
    # 'setup_objects_used',
    # 2023-09:21: trying without the specific feature to the quantified objects
    'setup_quantified_objects_used',
    # 'any_setup_objects_used,
    'adjacent_once_found',
    # 'adjacent_same_modal_found',
    'starts_and_ends_once',
    'once_in_middle_of_pref_found',
    'pref_without_hold_found',
    # 2023-09:21: trying without an explicit marking of at_end
    'at_end_found',

    'nested_logicals_found',
    'identical_logical_children_found',
    'identical_scoring_children_found',
    'scoring_count_expression_repetitions_exist',
    'tautological_expression_found',
    # 'redundant_expression_found',
    # 'redundant_scoring_terminal_expression_found',
    # 'unnecessary_expression_found',
    # 'identical_consecutive_seq_func_predicates_found',
    # 'disjoint_preferences_found',
    # 'disjoint_preferences_prop',
    # 'disjoint_preferences_scoring_terminal_types',
    # 'disjoint_preferences_scoring_terminal_predicates',
    # 'disjoint_preferences_same_predicates_only',
    # 'disjoint_seq_funcs_found',
    # 'disjoint_at_end_found',
    # 'disjoint_modal_predicates_found',
    # 'disjoint_modal_predicates_prop',
    # 'two_number_operation_found',
    'single_argument_multi_operation_found',
]

FEATURE_CATEGORIES = {
    'grammar_context': GRAMMAR_CONTEXT_FEATURES,
    'ngram_and_data_based': NGRAM_AND_DATA_BASED_FEATURES,
    'counting': COUNTING_FEATURES,
    'counting_less_important': COUNTING_LESS_IMPORTANT_FEATURES,
    'forall': FORALL_FEATURES,
    'forall_less_important': FORALL_LESS_IMPORTANT_FEATURES,
    'predicate_under_modal': PREDICATE_UNDER_MODAL_FEATURES,
    'predicate_role_filler': PREDICATE_ROLE_FILLER_FEATURES,
    'compositionality': COMPOSITIONALITY_FEATURES,
    'grammar_use': GRAMMAR_USE_FEATURES,
    'grammar_use_less_important': GRAMMAR_USE_LESS_IMPORTANT_FEATURES,
}
