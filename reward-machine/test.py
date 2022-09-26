import json
import pathlib
import typing

from game_handler import GameHandler


SIMPLE_STACKING_TRACE = pathlib.Path('./reward-machine/traces/simple_stacking_trace.json')
TEST_TRACE = pathlib.Path('./reward-machine/traces/throwing_balls_test_trace.json')
REPLAY_NESTING_KEYS = (
    'participants-v2-develop', 
    '17tSEDmCvGp1uKVEh5iq',
    'subCollection', 
    'participants-v2-develop/17tSEDmCvGp1uKVEh5iq/replay-preCreateGame'
)

def _load_trace(path: str, replay_nesting_keys: typing.Optional[typing.Sequence[str]] = None):
    with open(path, 'r') as f:
        trace = json.load(f)

    simple = isinstance(trace, list)

    if not simple and replay_nesting_keys is None:
        raise ValueError('Must provide replay_nesting_keys when not using simple mode')

    if simple:
        for event in trace:
            yield event

    else:
        replay_nesting_keys = typing.cast(typing.Sequence[str], replay_nesting_keys)
        for key in replay_nesting_keys:
            trace = trace[key]

        assert(all([key.startswith('batch-') for key in trace.keys()]))

        for batch_idx in range(len(trace)):
            batch = trace[f'batch-{batch_idx}']
            for event in batch['events']:
                yield event

# (once (and (agent_holds ?d) (< (distance agent ?d) 5)))
# (hold-while (and (not (agent_holds ?d)) (in_motion ?d)) (agent_crouches) (agent_holds ?h))
# (:terminal (or (>= (count-nonoverlapping throwAttempt) 5) (not (= (count-nonoverlapping throwBallToBin) 2) )))
# (:terminal (>= (count-nonoverlapping throwAttempt) 5))
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

    (:terminal (>= (count-nonoverlapping throwAttempt) 10))

    (:scoring maximize (+
        (count-nonoverlapping throwBallToBin:dodgeball)
        (- (/ (count-nonoverlapping throwAttempt) 5))
    )))
    """


if __name__ == "__main__":
    game_handler = GameHandler(TEST_THROWING_GAME)
    score = None

    trace_path = SIMPLE_STACKING_TRACE.resolve().as_posix()

    for idx, state in enumerate(_load_trace(trace_path, REPLAY_NESTING_KEYS)):
        print(f"\n\n================================PROCESSING STATE {idx} ================================")
        score = game_handler.process(state)
        if score is not None:
            break

    score = game_handler.score(game_handler.scoring)

    if score is not None:
        print("\n\nSCORE ACHIEVED:", score)
