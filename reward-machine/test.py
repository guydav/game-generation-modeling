import json
import pathlib
import typing

from game_handler import GameHandler


TEST_TRACE = pathlib.Path('./reward-machine/traces/throwing_balls_test_trace.json').resolve().as_posix()
REPLAY_NESTING_KEYS = (
    'participants-v2-develop', 
    '17tSEDmCvGp1uKVEh5iq',
    'subCollection', 
    'participants-v2-develop/17tSEDmCvGp1uKVEh5iq/replay-preCreateGame'
)

def _load_trace(path: str, replay_nesting_keys: typing.Sequence[str]):
    with open(path, 'r') as f:
        trace = json.load(f)

    for key in replay_nesting_keys:
        trace = trace[key]

    assert(all([key.startswith('batch-') for key in trace.keys()]))

    for batch_idx in range(len(trace)):
        batch = trace[f'batch-{batch_idx}']
        for event in batch['events']:
            yield event

# (once (and (agent_holds ?d) (< (distance agent ?d) 5)))
# (hold-while (and (not (agent_holds ?d)) (in_motion ?d)) (agent_crouches) (agent_holds ?h))
TEST_THROWING_GAME = """
    (define (game 61267978e96853d3b974ca53-23) (:domain few-objects-room-v1)

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
        (count-nonoverlapping throwBallToBin:golfball)
        (- (/ (count-nonoverlapping throwAttempt) 5))
    )))
    """


if __name__ == "__main__":
    game_handler = GameHandler(TEST_THROWING_GAME)
    score = None

    for idx, state in enumerate(_load_trace(TEST_TRACE, REPLAY_NESTING_KEYS)):
        print(f"\n\n================================PROCESSING STATE {idx} ================================")
        score = game_handler.process(state)
        if score is not None:
            break

    score = game_handler.score(game_handler.scoring)

    if score is not None:
        print("\n\nSCORE ACHIEVED:", score)
