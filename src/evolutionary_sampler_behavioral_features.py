import logging
import numpy as np
import typing
import re

import tatsu
import tatsu.ast

import ast_parser
import ast_printer
from fitness_features import ASTFitnessFeaturizer, FitnessTerm, SetupObjectsUsed, ContextDict, SETUP_OBJECTS_SKIP_OBJECTS, PREDICATE_AND_FUNCTION_RULES, DEPTH_CONTEXT_KEY

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _bin_number(number: int, bins: np.ndarray) -> int:
    return int(np.digitize(number, bins, right=True))


NODE_COUNT_BINS = [55, 85, 103, 117, 129.5, 160, 190, 220, 300]


class NodeCount(FitnessTerm):
    count: int = 0
    def __init__(self, bins: typing.List[int] = NODE_COUNT_BINS):
        super().__init__(re.compile('.*'), 'node_count')
        self.bins = np.array(bins)

    def game_start(self) -> None:
        self.count = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        self.count += 1

    def game_end(self) -> int:
        return _bin_number(self.count, self.bins)


UNIQUE_OBJECT_REFERENCES_BINS = [2, 3, 4, 5, 6, 7, 9, 11, 13]


class UniqueObjectsReferenced(SetupObjectsUsed):
    def __init__(self, bins: typing.List[int] = UNIQUE_OBJECT_REFERENCES_BINS, skip_objects: typing.Set[str] = SETUP_OBJECTS_SKIP_OBJECTS):
        super().__init__(skip_objects=skip_objects, header='unique_objects_referenced')
        self.bins = np.array(bins)

    def game_end(self) -> int:
        return _bin_number(len(self.setup_objects.union(self.used_objects)), self.bins)


UNIQUE_PREDICATE_REFERENCES_BINS = [2, 3, 4, 5, 6, 7, 8, 9]


class UniquePredicatesReferenced(FitnessTerm):
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


BASIC_BINNED = 'basic_binned'
BASIC_WITH_NODE_DEPTH = 'basic_with_node_depth'
FEATURE_SETS = [
    BASIC_BINNED,
    BASIC_WITH_NODE_DEPTH,
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

    else:
        raise ValueError(f'Unimplemented feature set: {feature_set}')

    return featurizer
