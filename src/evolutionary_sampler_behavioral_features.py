import abc
import argparse
from collections import Counter
import logging
import numpy as np
import typing
import re

import tatsu
import tatsu.ast
import tatsu.grammars
from sklearn.decomposition import PCA

import ast_parser
import ast_printer
from ast_utils import cached_load_and_parse_games_from_file
from fitness_features import ASTFitnessFeaturizer, FitnessTerm, SetupObjectsUsed, ContextDict, SETUP_OBJECTS_SKIP_OBJECTS, PREDICATE_AND_FUNCTION_RULES, DEPTH_CONTEXT_KEY, SectionExistsFitnessTerm
import room_and_object_types

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _bin_number(number: int, bins: np.ndarray) -> int:
    return int(np.digitize(number, bins, right=True))


NODE_COUNT_BINS = [55, 85, 103, 117, 129.5, 160, 190, 220, 300]
NODE_COUNT_BINS_8 = [61, 93, 113, 129, 168, 205, 263]


class NodeCount(FitnessTerm):
    bins: np.ndarray
    count: int = 0
    def __init__(self, bins: typing.List[int] = NODE_COUNT_BINS):
        super().__init__(re.compile('.*'), 'node_count')
        if bins is None:
            self.bins = None  # type: ignore
        else:
            self.bins = np.array(bins)

    def game_start(self) -> None:
        self.count = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        self.count += 1

    def game_end(self) -> int:
        if self.bins is not None:
            return _bin_number(self.count, self.bins)
        else:
            return self.count


UNIQUE_OBJECT_REFERENCES_BINS = [2, 3, 4, 5, 6, 7, 9, 11, 13]


class UniqueObjectsReferenced(SetupObjectsUsed):
    bins: np.ndarray
    def __init__(self, bins: typing.List[int] = UNIQUE_OBJECT_REFERENCES_BINS, skip_objects: typing.Set[str] = SETUP_OBJECTS_SKIP_OBJECTS):
        super().__init__(skip_objects=skip_objects, header='unique_objects_referenced')
        self.bins = np.array(bins)

    def game_end(self) -> int:
        return _bin_number(len(self.setup_objects.union(self.used_objects)), self.bins)


UNIQUE_PREDICATE_REFERENCES_BINS = [2, 3, 4, 5, 6, 7, 8, 9]


class UniquePredicatesReferenced(FitnessTerm):
    bins: np.ndarray
    predicates_referenced: typing.Set[str] = set()

    def __init__(self, bins: typing.List[int] = UNIQUE_PREDICATE_REFERENCES_BINS):
        super().__init__(PREDICATE_AND_FUNCTION_RULES, 'unique_predicates_referenced')
        self.bins = np.array(bins)

    def game_start(self) -> None:
        self.predicates_referenced = set()

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            if rule == 'predicate':
                pred = ast.pred.parseinfo.rule.replace('predicate_', '')  # type: ignore

            else:
                pred = ast.func.parseinfo.rule.replace('function_', '')  # type: ignore

            self.predicates_referenced.add(pred)

    def game_end(self) -> int:
        return _bin_number(len(self.predicates_referenced), self.bins)


class MeanNodeDepth(FitnessTerm):
    node_count: int = 0
    total_depth: int = 0

    def __init__(self):
        super().__init__(re.compile('.*'), 'mean_node_depth')
        self.node_count = 0

    def game_start(self) -> None:
        self.node_count = 0
        self.total_depth = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if DEPTH_CONTEXT_KEY in context:
            self.total_depth += context[DEPTH_CONTEXT_KEY]  # type: ignore
            self.node_count += 1

    def game_end(self) -> float:
        return self.total_depth / self.node_count if self.node_count > 0 else 0


SPECIFIC_PREDICATES =  ['adjacent', 'agent_holds', 'between', 'in', 'in_motion', 'on', 'touch']


class PredicateUsed(FitnessTerm):
    predicates_used: typing.Set[str]

    def __init__(self, predicates: typing.List[str] = SPECIFIC_PREDICATES):
        super().__init__(PREDICATE_AND_FUNCTION_RULES, 'predicate_used')
        self.predicates = predicates

    def game_start(self) -> None:
        self.predicates_used = set()

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            if rule == 'predicate':
                pred = ast.pred.parseinfo.rule.replace('predicate_', '')  # type: ignore

            else:
                pred = ast.func.parseinfo.rule.replace('function_', '')  # type: ignore

            self.predicates_used.add(pred)

    def game_end(self):
        return {pred: int(pred in self.predicates_used) for pred in self.predicates}

    def _get_all_inner_keys(self):
        return self.predicates


SPECIFIC_CATEGORIES = [room_and_object_types.BALLS, room_and_object_types.BLOCKS,
                       room_and_object_types.FURNITURE, room_and_object_types.LARGE_OBJECTS,
                       room_and_object_types.RAMPS, room_and_object_types.RECEPTACLES,
                       room_and_object_types.SMALL_OBJECTS,
                       ]

class ObjectCategoryUsed(SetupObjectsUsed):
    def __init__(self, categories: typing.List[str] = SPECIFIC_CATEGORIES, skip_objects: typing.Set[str] = SETUP_OBJECTS_SKIP_OBJECTS):
        super().__init__(skip_objects=skip_objects, header='object_category_used')
        self.categories = categories

    def game_end(self):
        categories_used = set()
        for object_set in [self.setup_objects, self.used_objects]:
            for obj in object_set:
                if obj in room_and_object_types.TYPES_TO_CATEGORIES:
                    categories_used.add(room_and_object_types.TYPES_TO_CATEGORIES[obj])

        return {cat: int(cat in categories_used) for cat in self.categories}

    def _get_all_inner_keys(self):
        return self.categories


BASIC_BINNED = 'basic_binned'
BASIC_WITH_NODE_DEPTH = 'basic_with_node_depth'
NODE_COUNT_OBJECTS = 'node_count_objects'
NODE_COUNT_PREDICATES = 'node_count_predicates'
NODE_COUNT_OBJECTS_SETUP = 'node_count_objects_setup'
NODE_COUNT_PREDICATES_SETUP = 'node_count_predicates_setup'
SPECIFIC_PREDICATES_SETUP = 'specific_predicates_setup'
SPECIFIC_CATEGORIES_SETUP = 'specific_categories_setup'
NODE_COUNT_SPECIFIC_PREDICATES = 'node_count_specific_predicates'


FEATURE_SETS = [
    BASIC_BINNED,
    BASIC_WITH_NODE_DEPTH,
    NODE_COUNT_OBJECTS,
    NODE_COUNT_PREDICATES,
    NODE_COUNT_OBJECTS_SETUP,
    NODE_COUNT_PREDICATES_SETUP,
    SPECIFIC_PREDICATES_SETUP,
    SPECIFIC_CATEGORIES_SETUP,
    NODE_COUNT_SPECIFIC_PREDICATES
]


class BehavioralFeaturizer(abc.ABC):
    @abc.abstractmethod
    def get_feature_names(self) -> typing.List[str]:
        pass

    @abc.abstractmethod
    def get_feature_value_counts(self) -> typing.Dict[str, int]:
        pass

    @abc.abstractmethod
    def get_game_features(self) -> typing.Dict[str, typing.Any]:
        pass


class FitnessFeaturesBehavioralFeaturizer(ASTFitnessFeaturizer, BehavioralFeaturizer):
    def get_feature_names(self) -> typing.List[str]:
        #  [4:] to remove the first few automatically-added columns
        return self.get_all_column_keys()[4:]

    def get_feature_value_counts(self) -> typing.Dict[str, int]:
        feature_to_term_mapping = self.get_column_to_term_mapping()
        n_values_by_feature = {}
        for feature_name in self.get_feature_names():
            feature_term = feature_to_term_mapping[feature_name]
            if hasattr(feature_term, 'bins'):
                n_values = (len(feature_term.bins) + 1)  # type: ignore

            else:
                n_values = 2

            n_values_by_feature[feature_name] = n_values

        return n_values_by_feature

    def get_game_features(self, game) -> typing.Dict[str, typing.Any]:
        return self.parse(game, return_row=True)  # type: ignore


DEFAULT_N_COMPONENTS = 20
DEFAULT_RANDOM_SEED = 33


class PCABehavioralFeaturizer(BehavioralFeaturizer):
    def __init__(self, feature_indices: typing.List[int], bins_per_feature: int,
                 ast_file_path: str, grammar_parser: tatsu.grammars.Grammar,  # type: ignore
                 fitness_featurizer: ASTFitnessFeaturizer, feature_names: typing.List[str],
                 n_components: int = DEFAULT_N_COMPONENTS, random_seed: int = DEFAULT_RANDOM_SEED):
        self.feature_indices = feature_indices
        self.bins_per_feature = bins_per_feature
        self.ast_file_path = ast_file_path
        self.grammar_parser = grammar_parser
        self.fitness_featurizer = fitness_featurizer
        self.feature_names = feature_names
        self.n_components = n_components
        self.random_seed = random_seed

        self.output_feature_names = [self._feature_name(i) for i in self.feature_indices]

        self._init_pca()

    def _game_to_feature_vector(self, game) -> np.ndarray:
        game_features = self.fitness_featurizer.parse(game, return_row=True)  # type: ignore
        return np.array([game_features[name] for name in self.feature_names])

    def _init_pca(self):
        game_asts = list(cached_load_and_parse_games_from_file(self.ast_file_path, self.grammar_parser, False))
        game_features = []
        for game in game_asts:
            game_features.append(self._game_to_feature_vector(game))

        features_array = np.stack(game_features)

        self.pca = PCA(n_components=self.n_components, random_state=self.random_seed)
        projections = self.pca.fit_transform(features_array)

        self.bins_by_feature_index = {}
        for feature_index in self.feature_indices:
            feature_values = projections[:, feature_index]
            step = 1 / self.bins_per_feature
            quantiles = np.quantile(feature_values, np.linspace(step, 1 - step, self.bins_per_feature - 1))
            self.bins_by_feature_index[feature_index] = quantiles

            digits = np.digitize(feature_values, quantiles)
            counts = Counter(digits)
            logger.debug(f'On feature #{feature_index}, the real games have counts: {counts}')

        all_game_features = [self.get_game_features(game) for game in game_asts]
        all_game_feature_tuples = [tuple(game_features[name] for name in self.output_feature_names) for game_features in all_game_features]
        all_game_feature_tuples = set(all_game_feature_tuples)
        logger.debug(f'The real games have {len(all_game_feature_tuples)} unique feature tuples')

    def _feature_name(self, feature_index: int):
        return f'pca_{feature_index}'

    def get_feature_names(self) -> typing.List[str]:
        return self.output_feature_names

    def get_feature_value_counts(self) -> typing.Dict[str, int]:
        return {self._feature_name(i): self.bins_per_feature for i in self.feature_indices}

    def get_game_features(self, game) -> typing.Dict[str, typing.Any]:
        game_vector = self._game_to_feature_vector(game)
        game_projection = self.pca.transform(game_vector.reshape(1, -1))[0]
        return {self._feature_name(i): np.digitize(game_projection[i], self.bins_by_feature_index[i]) for i in self.feature_indices}


def build_behavioral_features_featurizer(
        args: argparse.Namespace,
        grammar_parser: tatsu.grammars.Grammar,  # type: ignore
        fitness_featurizer: ASTFitnessFeaturizer, feature_names: typing.List[str]
        ) -> BehavioralFeaturizer:

    feature_set = args.map_elites_custom_behavioral_features_key

    if feature_set is not None:
        if feature_set not in FEATURE_SETS:
            raise ValueError(f'Invalid feature set: {feature_set}')

        featurizer = FitnessFeaturesBehavioralFeaturizer()

        if feature_set == BASIC_BINNED:
            featurizer.register(NodeCount())
            featurizer.register(UniqueObjectsReferenced())
            featurizer.register(UniquePredicatesReferenced())

        elif feature_set == BASIC_WITH_NODE_DEPTH:
            featurizer.register(NodeCount())
            featurizer.register(UniqueObjectsReferenced())
            featurizer.register(UniquePredicatesReferenced())
            featurizer.register(MeanNodeDepth())

        elif feature_set == NODE_COUNT_OBJECTS:
            featurizer.register(NodeCount())
            featurizer.register(UniqueObjectsReferenced())

        elif feature_set == NODE_COUNT_PREDICATES:
            featurizer.register(NodeCount())
            featurizer.register(UniquePredicatesReferenced())

        elif feature_set == NODE_COUNT_OBJECTS_SETUP:
            featurizer.register(NodeCount())
            featurizer.register(UniqueObjectsReferenced())
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)

        elif feature_set == NODE_COUNT_PREDICATES_SETUP:
            featurizer.register(NodeCount())
            featurizer.register(UniquePredicatesReferenced())
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)

        elif feature_set == SPECIFIC_PREDICATES_SETUP:
            featurizer.register(PredicateUsed())
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)

        elif feature_set == SPECIFIC_CATEGORIES_SETUP:
            featurizer.register(ObjectCategoryUsed())
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)

        elif feature_set == NODE_COUNT_SPECIFIC_PREDICATES:
            featurizer.register(NodeCount(NODE_COUNT_BINS_8))
            featurizer.register(PredicateUsed())

        else:
            raise ValueError(f'Unimplemented feature set: {feature_set}')

        return featurizer

    indices = args.map_elites_pca_behavioral_features_indices
    bins_per_feature = args.map_elites_pca_behavioral_features_bins_per_feature

    if bins_per_feature is None:
        raise ValueError('Must specify bins per feature for PCA featurizer')

    ast_file_path = args.map_elites_pca_behavioral_features_ast_file_path
    n_components = args.map_elites_pca_behavioral_features_n_components if args.map_elites_pca_behavioral_features_n_components is not None else max(indices) + 1
    random_seed = args.random_seed

    pca_featurizer = PCABehavioralFeaturizer(
        feature_indices=indices,
        bins_per_feature=bins_per_feature,
        ast_file_path=ast_file_path,
        grammar_parser=grammar_parser,
        fitness_featurizer=fitness_featurizer,
        feature_names=feature_names,
        n_components=n_components,
        random_seed=random_seed
    )

    return pca_featurizer
