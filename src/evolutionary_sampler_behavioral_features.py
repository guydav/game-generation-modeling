import logging
import numpy as np
import typing
import re

import tatsu
import tatsu.ast

import ast_parser
import ast_printer
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


def build_behavioral_features_featurizer(feature_set: str) -> ASTFitnessFeaturizer:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f'Invalid feature set: {feature_set}')

    featurizer = ASTFitnessFeaturizer()

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
