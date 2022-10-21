import numpy as np
import pathlib
import pytest
import typing
import sys

from utils import FullState, get_project_dir
from manual_run import _load_trace
from game_handler import GameHandler
from preference_handler import PreferenceSatisfaction




BLOCK_STACKING_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/block_stacking_test_trace.json')
BALL_TO_WALL_TO_BIN_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/three_wall_to_bin_bounces.json')


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

TEST_THROWING_GAME = """
(define (game 61267978e96853d3b974ca53-23) (:domain many-objects-room-v1)

(:constraints (and 
    (forall (?b - (either dodgeball golfball)) 
        (preference throwBallToBin (exists (?h - hexagonal_bin)
            (then
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
    )

    (preference throwAttempt
        (exists (?d - ball)
            (then 
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (not (in_motion ?d)))
            )
        )
    )

))

(:scoring maximize (+
    (count-nonoverlapping throwBallToBin)
    (- (/ (count-nonoverlapping throwAttempt) 5))
)))
"""

TEST_THROW_BALL_AT_WALL_GAME = """
    (define (game 61267978e96853d3b974ca53-23) (:domain many-objects-room-v1)

    (:constraints (and 

        (preference throwToWall
            (exists (?w - wall ?b - ball) 
                (then 
                    (once (agent_holds ?b))
                    (hold-while  
                        (and (not (agent_holds ?b)) (in_motion ?b))
                        (touch ?b ?w)
                    ) 
                    (once (not (in_motion ?b)))
                )
            )
        )
    ))

    (:scoring maximize (+
        (* (count-nonoverlapping throwToWall) 1)
    )))
"""


TEST_GAME_LIBRARY = {
    'on-chair-bug': TEST_ON_BUG_GAME,
    'test-building': TEST_BUILDING_GAME,
    'test-throwing': TEST_THROWING_GAME,
    'test-throw-to-wall': TEST_THROW_BALL_AT_WALL_GAME,
}

TEST_CASES = [
    ('on-chair-bug', BLOCK_STACKING_TRACE, 0.0, {},),
    ('test-building', BLOCK_STACKING_TRACE, 10.0, {
        'blockInBuildingAtEnd': [
            PreferenceSatisfaction(mapping={'?b': 'building_0', '?l': 'CubeBlock|-02.96|+01.26|-01.72'}, start=2140, end=2140, measures={}), 
            PreferenceSatisfaction(mapping={'?b': 'building_0', '?l': 'CubeBlock|-02.97|+01.26|-01.94'}, start=2140, end=2140, measures={}), 
            PreferenceSatisfaction(mapping={'?b': 'building_0', '?l': 'CylinderBlock|-02.95|+01.62|-01.95'}, start=2140, end=2140, measures={}), 
            PreferenceSatisfaction(mapping={'?b': 'building_0', '?l': 'CylinderBlock|-03.02|+01.62|-01.73'}, start=2140, end=2140, measures={}), 
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'CubeBlock|-02.99|+01.26|-01.49'}, start=2140, end=2140, measures={}), 
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'CylinderBlock|-02.97|+01.62|-01.50'}, start=2140, end=2140, measures={}), 
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'PyramidBlock|-02.95|+01.61|-02.66'}, start=2140, end=2140, measures={}), 
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'TallRectBlock|-02.95|+02.05|-02.31'}, start=2140, end=2140, measures={}), 
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'TallRectBlock|-02.95|+02.05|-02.52'}, start=2140, end=2140, measures={}), 
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'TallRectBlock|-02.95|+02.05|-02.72'}, start=2140, end=2140, measures={})
            ]
        }
    ),
    ('test-throwing', BALL_TO_WALL_TO_BIN_TRACE, -1.2, {
        'throwBallToBin': [
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.75|-00.03|-02.74', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1958, end=2015, measures={}), 
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.75|-00.03|-02.74', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2040, end=2151, measures={}), 
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.75|-00.03|-02.74', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2374, end=2410, measures={})
        ], 
        'throwAttempt': [
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=343, end=456, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=457, end=590, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=769, end=880, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=881, end=947, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=948, end=1019, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1020, end=1120, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1121, end=1209, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1210, end=1279, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1280, end=1370, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1371, end=1439, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1440, end=1502, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1503, end=1555, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1556, end=1656, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1699, end=1782, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1783, end=1868, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1869, end=1957, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1958, end=2015, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2040, end=2151, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2198, end=2293, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2294, end=2373, measures={}), 
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2374, end=2410, measures={})
        ]
    }),
    ('test-throw-to-wall', BALL_TO_WALL_TO_BIN_TRACE, 10.0, {
        'throwToWall': [
            PreferenceSatisfaction(mapping={'?w': 'east_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=343, end=456, measures={}), 
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=343, end=456, measures={}), 
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=457, end=590, measures={}), 
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1699, end=1782, measures={}), 
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1783, end=1868, measures={}), 
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1869, end=1957, measures={}), 
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1958, end=2015, measures={}), 
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2040, end=2151, measures={}), 
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2294, end=2373, measures={}), 
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2374, end=2410, measures={})
            ]
        },
    ),
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

    assert np.allclose(score, expected_score)

    if expected_satisfactions is not None:
        for pref_name, pref_satisfactions in expected_satisfactions.items():
            assert pref_name in game_handler.preference_satisfactions

            for pref_satisfaction in pref_satisfactions:
                assert pref_satisfaction in game_handler.preference_satisfactions[pref_name]

if __name__ == '__main__':
    print(__file__)
    sys.exit(pytest.main([__file__]))
