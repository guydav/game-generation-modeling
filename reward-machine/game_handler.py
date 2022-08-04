import os
import sys
import tatsu

from preference_handler import PreferenceHandler

class GameHandler():
    def __init__(self, game, grammar_path="../dsl/dsl.ebnf"):
        grammar = open(grammar_path).read()
        self.grammar_parser = tatsu.compile(grammar)

        self.game_ast = self.grammar_parser.parse(game)

        self.game_name = None
        self.domain_name = None
        self.setup = None
        self.preferences = None
        self.terminal = None
        self.scoring = None

        self._extract_game_info(self.game_ast)

        # Maps from each preference name to the PreferenceHandler (or list of PreferenceHandlers) that will 
        # evaluate that preference
        self.preference_handlers = {}

        for preference in self.preferences:
            rule = preference["definition"]["parseinfo"].rule

            # A preference definition expands into either (forall <variable_list> <preference>) or <preference>
            if rule == "preference":
                name = preference["definition"]["pref_name"]
                try:
                    pref_handler = PreferenceHandler(preference["definition"])
                    self.preference_handlers[name] = pref_handler

                except Exception as exc:
                    print(f"Unable to construct PreferenceHandler for '{name}' due to following exception: {repr(exc)}")

            elif rule == "pref_forall":
                pass


    def _extract_game_info(self, ast):
        '''
        Recursively extract the game's name, domain, setup, preferences, terminal conditions, and
        scoring (if they exist)
        '''
        if isinstance(ast, tuple) or isinstance(ast, list):
            for item in ast:
                self._extract_game_info(item)

        elif isinstance(ast, tatsu.ast.AST):
            rule = ast["parseinfo"].rule
            if rule == "game_def":
                self.game_name = ast["game_name"]

            elif rule == "domain_def":
                self.domain_name = ast["domain_name"]

            elif rule == "setup":
                self.setup = ast["setup"]

            elif rule == "preferences":
                self.preferences = ast["preferences"]

            elif rule == "terminal":
                self.terminal = ast["terminal"]

            elif rule == "scoring":
                self.scoring = ast["scoring"]


if __name__ == "__main__":
    test_game = """
    (define (game 61267978e96853d3b974ca53-23) (:domain few-objects-room-v1)

    (:constraints (and 
        (preference throwBallToBin
            (exists (?d - ball ?h - bin)
                (then
                    (once (and (agent_holds ?d) (< (distance agent ?d) 5)))
                    (hold-while (and (not (agent_holds ?d)) (in_motion ?d)) (agent_crouches) (agent_holds ?h))
                    (once (and (not (in_motion ?d)) (in ?h ?d)))
                )
            )
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

    test_game_2 = """
    (define (game 56cb8858edf8da000b6df354-32) (:domain many-objects-room-v1)  ; 32
    (:setup (and 
        (exists (?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - (either cube_block cylindrical_block pyramid_block)) (game-optional (and ; specifying the pyramidal structure
            (on desk ?b1)
            (on desk ?b2)
            (on desk ?b3)
            (on ?b1 ?b4)
            (on ?b2 ?b5)
            (on ?b4 ?b6) 
        )))
        (exists (?w1 ?w2 - wall ?h - hexagonal_bin) 
            (game-conserved (and
                (adjacent ?h ?w1)
                (adjacent ?h ?w2)   
            ))
        )
    ))
    (:constraints (and 
        (forall (?b - (either dodgeball golfball)) 
            (preference ballThrownToBin (exists (?h - hexagonal_bin)
                (then
                    (once (agent_holds ?b))
                    (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                    (once (and (not (in_motion ?b)) (in ?h ?b)))
                )
            ))
        )
        (preference blockInTowerKnocked (exists (?b - building ?c - (either cube_block cylindrical_block pyramid_block)
            ?d - (either dodgeball golfball))
            (then
                (once (and 
                    (agent_holds ?d)
                    (on desk ?b)
                    (in ?b ?c) 
                ))
                (hold-while 
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (or 
                        (touch ?c ?d)
                        (exists (?c2 - (either cube_block cylindrical_block pyramid_block)) (touch ?c2 ?c))
                    )
                    (in_motion ?c)
                )
                (once (not (in_motion ?c)))
            )
        ))
        (forall (?d - (either dodgeball golfball))
            (preference throwAttempt
                (then 
                    (once (agent_holds ?d))
                    (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                    (once (not (in_motion ?d)))
                )
            )
        )
        (forall (?d - (either dodgeball golfball))
            (preference ballNeverThrown
                (then
                    (once (game_start))
                    (hold (not (agent_holds ?d)))
                    (hold (game_over))
                )
            )
        )
    ))
    (:terminal (or 
        (> (count-maximal-nonoverlapping throwAttempt) 2)
        (>= (count-nonoverlapping throwAttempt) 12)
    ))
    (:scoring maximize (* 
        (>=     
            (+
                (count-nonoverlapping ballThrownToBin:dodgeball)
                (* 2 (count-nonoverlapping ballThrownToBin:golfball))
            ) 
            2
        )
        (+
            (count-once-per-objects blockInTowerKnocked)
            (count-once-per-objects ballNeverThrown:golfball)
            (* 2 (count-once-per-objects ballNeverThrown:dodgeball))
        )
    )))
    """

    game_handler = GameHandler(test_game_2)
