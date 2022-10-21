from importlib.resources import path
import pathlib
from cv2 import trace
import pytest
import typing

from utils import FullState
from manual_run import _load_trace
from game_handler import GameHandler
from preference_handler import PreferenceSatisfaction


BLOCK_STACKING_TRACE = pathlib.Path('./reward-machine/traces/block_stacking_test_trace.json')
TEST_ON_BUG_GAME = """
(define (game building-test) (:domain many-objects-room-v1)
(:constraints (and 
    (preference chairOnTallRectBlock (exists (?c - chair ?l - tall_rect_block)
        (at-end
            (and 
                (on ?l ?c)
            )
        )
    ))
))
(:scoring maximize (+
    (* (count-nonoverlapping chairOnTallRectBlock) 1)
))
)
"""

TEST_BUILDING_GAME = """
(define (game building-test) (:domain many-objects-room-v1)
(:constraints (and 
    (forall (?b - building) 
        (preference blockInBuildingAtEnd (exists (?l - block)
            (at-end
                (and 
                    (in ?b ?l)
                )
            )
        ))
    )
))
(:scoring maximize (+
    (* (count-nonoverlapping blockInBuildingAtEnd) 1)
))
)
"""


TEST_GAME_LIBRARY = {
    'on-chair-bug': TEST_ON_BUG_GAME,
    'test-building': TEST_BUILDING_GAME,
}

TEST_CASES = [
    ('on-chair-bug', BLOCK_STACKING_TRACE, 1.0, {
        'chairOnTallRectBlock': [
            PreferenceSatisfaction(mapping={'?l': 'TallRectBlock|-02.95|+02.05|-02.31', '?c': 'Chair|+02.73|00.00|-01.21'}, start=2140, end=2140, measures={})
            ], 
    },),
    ('test-building', BLOCK_STACKING_TRACE, 10.0, {}),
]


@pytest.mark.parametrize("game_key, trace_path, expected_score, expected_satisfactions", TEST_CASES)
def test_single_game(game_key: str, trace_path: typing.Union[str, pathlib.Path],
    expected_score: float, 
    expected_satisfactions: typing.Optional[typing.Dict[str, typing.List[PreferenceSatisfaction]]],
    debug: bool = False, debug_building_handler: bool = False, debug_preference_handlers: bool = False):

    game_def = TEST_GAME_LIBRARY[game_key]

    game_handler = GameHandler(game_def)
    score = None

    if isinstance(trace_path, pathlib.Path):
        trace_path = trace_path.resolve().as_posix()

    for state, is_final in _load_trace(trace_path):
        state = FullState.from_state_dict(state)
        score = game_handler.process(state, is_final, debug=debug, 
            debug_building_handler=debug_building_handler, 
            debug_preference_handlers=debug_preference_handlers)
        if score is not None:
            break

    score = game_handler.score(game_handler.scoring)

    assert score == expected_score

    if expected_satisfactions is not None:
        for pref_name, pref_satisfactions in expected_satisfactions.items():
            assert pref_name in game_handler.preference_satisfactions

            for pref_satisfaction in pref_satisfactions:
                assert pref_satisfaction in game_handler.preference_satisfactions[pref_name]
