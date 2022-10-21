import json
import pathlib
import typing

from game_handler import GameHandler
from utils import FullState


BLOCK_STACKING_TRACE = pathlib.Path('./reward-machine/traces/block_stacking_test_trace.json')
THROWING_BALLS_AT_WALL_TRACE = pathlib.Path('./reward-machine/traces/throwing_balls_at_wall.json')
WALL_BALL_TRACE = pathlib.Path('/Users/guydavidson/Downloads/m9yCPYToAPeSSYKh7WuL-preCreateGame.json')
SECOND_WALL_BALL_TRACE = pathlib.Path('/Users/guydavidson/Downloads/HuezY8vhxETSFyQL6BZK-preCreateGame.json')
SIMPLE_STACKING_TRACE = pathlib.Path('./reward-machine/traces/simple_stacking_trace.json')
TEST_TRACE = pathlib.Path('./reward-machine/traces/three_wall_to_bin_bounces.json')
SETUP_TEST_TRACE = pathlib.Path('./reward-machine/traces/setup_test_trace.json')
CASTLE_TEST_TRACE = pathlib.Path('./reward-machine/traces/building_castle.json')
BUILDING_IN_TOUCH_TEST_TRACE = pathlib.Path('./reward-machine/traces/weZ1UVzKNaiTjaqu0DGI-preCreateGame-buildings-in-touching.json')

REPLAY_NESTING_KEYS = (
    'participants-v2-develop', 
    '17tSEDmCvGp1uKVEh5iq',
    'subCollection', 
    'participants-v2-develop/17tSEDmCvGp1uKVEh5iq/replay-preCreateGame'
)

def _load_trace(path: str, replay_nesting_keys: typing.Optional[typing.Sequence[str]] = REPLAY_NESTING_KEYS):
    with open(path, 'r') as f:
        trace = json.load(f)

    simple = isinstance(trace, list)

    if not simple and replay_nesting_keys is None:
        raise ValueError('Must provide replay_nesting_keys when not using simple mode')

    if simple:
        for idx, event in enumerate(trace):
            yield (event, idx == len(trace) - 1)

    else:
        replay_nesting_keys = typing.cast(typing.Sequence[str], replay_nesting_keys)
        for key in replay_nesting_keys:
            trace = trace[key]

        assert(all([key.startswith('batch-') for key in trace.keys()]))

        for batch_idx in range(len(trace)):
            batch = trace[f'batch-{batch_idx}']
            for idx, event in enumerate(batch['events']):
                yield (event, (idx == len(batch['events']) - 1) and (batch_idx == len(trace) - 1)) # make sure we're in the last batch and the last event

# (once (and (agent_holds ?d) (< (distance agent ?d) 5)))
# (hold-while (and (not (agent_holds ?d)) (in_motion ?d)) (agent_crouches) (agent_holds ?h))
# (:terminal (or (>= (count-nonoverlapping throwAttempt) 5) (not (= (count-nonoverlapping throwBallToBin) 2) )))
# (:terminal (>= (count-nonoverlapping throwAttempt) 5))
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

TEST_THROW_BOUNCE_GAME = """
    (define (game 61267978e96853d3b974ca53-23) (:domain many-objects-room-v1)

    (:constraints (and 

        (preference throwToWallToBin
            (exists (?w - wall ?b - ball ?h - hexagonal_bin) 
                (then 
                    (once (agent_holds ?b))
                    (hold-while (and (not (agent_holds ?b)) (in_motion ?b)) (touch ?w ?b))
                    (once  (and (in ?h ?b) (not (in_motion ?b))))
                )
            )
        )
    ))

    (:scoring maximize (+
        (* (count-nonoverlapping throwToWallToBin) 10)
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

# (game-optional (not (exists (?g - game_object) (on desk ?g))))
# (forall (?g - game_object) (game-optional (not (on desk ?g))))

TEST_SETUP_GAME = """
(define (game 60d432ce6e413e7509dd4b78-22) (:domain many-objects-room-v1)  ; 22
(:setup (and 
    (exists (?h - hexagonal_bin) (game-conserved (on bed ?h)))
    (forall (?b - ball) (game-optional (on desk ?b)))
))
(:constraints (and 
    (preference throwToBin
        (exists (?b - ball ?h - hexagonal_bin) 
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once  (and (in ?h ?b) (not (in_motion ?b))))
            )
        )
    )

    (preference throwAttempt
        (exists (?b - ball)
            (then 
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                (once (not (in_motion ?b)))
            )
        )
    )
))
(:scoring maximize (+ 
    (* (count-nonoverlapping throwToBin) 1)
)))
"""

TEST_AT_END_GAME = """
(define (game 613e4bf960ca68f8de00e5e7-17) (:domain many-objects-room-v1)  ; 17/18

(:constraints (and 
    (preference castleBuilt (exists (?b - bridge_block ?f - flat_block ?t - tall_cylindrical_block ?c - cube_block ?p - pyramid_block)
        (at-end
            (and 
                (on ?b ?f)
                (on ?f ?t)
                (on ?t ?c)
                (on ?c ?p)
            )
        )
    ))
))
(:scoring maximize (+ 
    (* 10 (count-once-per-objects castleBuilt))
)))
"""

TEST_BLOCK_STACK_GAME = """
(define (game block-test) (:domain many-objects-room-v1)
(:constraints (and 
        (preference blockOnBlock
            (exists (?b1 ?b2 - block)
                (then 
                    (once (agent_holds ?b1))
                    (hold (or (in_motion ?b1) (on ?b2 ?b1)))
                    (hold (on ?b2 ?b1))
                    (once (or (in_motion ?b1) (in_motion ?b2)))
                )
            )
        )
))
(:scoring maximize (+
    (* (count-nonoverlapping blockOnBlock) 1)
)))
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


TEST_BUILDING_ON_CHAIR_GAME = """
(define (game building-test) (:domain many-objects-room-v1)
(:constraints (and 
    (forall (?b - building) 
        (preference blockInBuildingOnChairAtEnd (exists (?c - chair ?l - block)
            (at-end
                (and 
                    (in ?b ?l)
                    (on ?c ?b)
                )
            )
        ))
    )
))
(:scoring maximize (+
    (* (count-nonoverlapping blockInBuildingOnChairAtEnd) 2)
))
)
"""


TEST_BUILDING_IN_BIN_GAME = """
(define (game building-test) (:domain medium-objects-room-v1)
(:constraints (and 
    (forall (?b - building) 
        (preference blockInBuildingInBinAtEnd (exists (?h - hexagonal_bin ?l - block)
            (at-end
                (and 
                    (in ?b ?l)
                    (in ?h ?b)
                )
            )
        ))
    )
))
(:scoring maximize (+
    (* (count-nonoverlapping blockInBuildingInBinAtEnd) 2)
))
)
"""


TEST_BUILDING_TOUCHES_WALL_GAME = """
(define (game building-test) (:domain medium-objects-room-v1)
(:constraints (and 
    (forall (?b - building) 
        (preference blockInBuildingTouchingWallAtEnd (exists (?w - wall ?l - block)
            (at-end
                (and 
                    (in ?b ?l)
                    (touch ?w ?b)
                )
            )
        ))
    )
))
(:scoring maximize (+
    (* (count-nonoverlapping blockInBuildingTouchingWallAtEnd) 2)
))
)
"""


if __name__ == "__main__":
    game_handler = GameHandler(TEST_THROW_BALL_AT_WALL_GAME)
    score = None

    trace_path = TEST_TRACE.resolve().as_posix()

    for idx, (state, is_final) in enumerate(_load_trace(trace_path)):
        print(f"\n\n================================PROCESSING STATE {idx} ================================")
        state = FullState.from_state_dict(state)
        score = game_handler.process(state, is_final)
        if score is not None:
            break

    score = game_handler.score(game_handler.scoring)

    if score is not None:
        print("\n\nSCORE ACHIEVED:", score)
