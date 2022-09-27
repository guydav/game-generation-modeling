import itertools
import os
import sys
import tatsu
import tatsu.ast
import typing

from math import prod

from config import SAMPLE_TRAJECTORY, OBJECTS_BY_ROOM_AND_TYPE
from preference_handler import PreferenceHandler
from predicate_handler import PredicateHandler
from utils import PreferenceDescriber, extract_variable_type_mapping


DEFAULT_GRAMMAR_PATH = "./dsl/dsl.ebnf"


class GameHandler():
    def __init__(self, game: str, grammar_path: str = DEFAULT_GRAMMAR_PATH):
        grammar = open(grammar_path).read()
        self.grammar_parser = tatsu.compile(grammar)

        self.game_name = None
        self.domain_name = None
        self.setup = None
        self.preferences = []
        self.terminal = None
        self.scoring = None

        self.game_ast = self.grammar_parser.parse(game)
        self._extract_game_info(self.game_ast)

        if self.domain_name is None:
            raise ValueError("Error: Failed to extract domain from game specification")

        self.predicate_handler = PredicateHandler(self.domain_name)

        # Maps from each preference name to the PreferenceHandler (or list of PreferenceHandlers) that will 
        # evaluate that preference
        self.preference_handlers = {}

        # Maps from each preference name to a list of satisfaction data. Each entry in the list is a tuple
        # of the following form: (variable_mapping, start_state_num, end_state_num)
        self.preference_satisfactions = {}

        for preference in self.preferences:
            rule = preference["definition"]["parseinfo"].rule

            # A preference definition expands into either (forall <variable_list> <preference>) or <preference>
            if rule == "preference":
                name = preference["definition"]["pref_name"]
                
                pref_handler = PreferenceHandler(preference["definition"], self.predicate_handler, self.domain_name)
                self.preference_handlers[name] = pref_handler
                self.preference_satisfactions[name] = []
                print(f"Successfully constructed PreferenceHandler for '{name}'")


            # TODO: forall can cover multiple preferences
            elif rule == "pref_forall":
                # The outer "forall" works to track the times the inner preference is satisfied for each type
                # in the outer variables (e.g. )
                forall_vars = preference["definition"]["forall_vars"]
                forall_pref = preference["definition"]["forall_pref"]
                
                variable_type_mapping = extract_variable_type_mapping(forall_vars["variables"])
                sub_preference = forall_pref["preferences"]
                name = sub_preference["pref_name"]

                pref_handler = PreferenceHandler(sub_preference, self.predicate_handler, self.domain_name, 
                    additional_variable_mapping=variable_type_mapping)
                self.preference_handlers[name] = pref_handler
                self.preference_satisfactions[name] = []
                print(f"Successfully constructed PreferenceHandler for '{name}'")

    def _extract_game_info(self, ast: typing.Union[list, tuple, tatsu.ast.AST]):
        '''
        Recursively extract the game's name, domain, setup, preferences, terminal conditions, and
        scoring (if they exist)
        '''
        if isinstance(ast, tuple) or isinstance(ast, list):
            for item in ast:
                self._extract_game_info(item)

        elif isinstance(ast, tatsu.ast.AST):
            rule = ast["parseinfo"].rule  # type: ignore
            if rule == "game_def":
                self.game_name = ast["game_name"]

            elif rule == "domain_def":

                if ast["domain_name"] == "few-objects-room-v1":
                    self.domain_name = "few"
                elif ast["domain_name"] == "medium-objects-room-v1":
                    self.domain_name = "medium"
                elif ast["domain_name"] == "many-objects-room-v1":
                    self.domain_name = "many"

            elif rule == "setup":
                self.setup = ast["setup"]

            elif rule == "preferences":
                # Handle games with single preference
                if isinstance(ast["preferences"], tatsu.ast.AST):
                    self.preferences = [ast["preferences"]]
                else:
                    self.preferences = ast["preferences"]

            elif rule == "terminal":
                self.terminal = ast["terminal"]

            elif rule == "scoring":
                self.scoring = ast["scoring"]

    def process(self, state: typing.Dict[str, typing.Any]) -> typing.Optional[float]:  
        '''
        Process a state in a game trajectory by passing it to each of the relevant PreferenceHandlers. If the state is
        the last one in the trajectory or the terminal conditions are met, then we also do scoring
        '''
        for preference_name, handlers in self.preference_handlers.items():
            if isinstance(handlers, PreferenceHandler):
                satisfactions = handlers.process(state)
                self.preference_satisfactions[preference_name] += satisfactions

            elif isinstance(handlers, list):
                pass

        if self.terminal is not None:
            terminate = self.evaluate_terminals(self.terminal) # TODO
        else:
            terminate = False

        if terminate:
            score = self.score(self.scoring) 

        else:
            score = None

        return score

    def evaluate_terminals(self, terminal_expression: typing.Optional[tatsu.ast.AST]) -> bool:
        '''
        Determine whether the terminal conditions of the game have been met
        '''
        if terminal_expression is None:
            return False


        rule = terminal_expression["parseinfo"].rule  # type: ignore

        if rule == "terminal":
            return self.evaluate_terminals(terminal_expression["terminal"])  

        elif rule == "terminal_not":
            inner_value = self.evaluate_terminals(terminal_expression["not_args"])  

            return not inner_value

        elif rule == "terminal_and":
            inner_values = [self.evaluate_terminals(sub) for sub in terminal_expression["and_args"]]  # type: ignore

            return all(inner_values)

        elif rule == "terminal_or":
            inner_values = [self.evaluate_terminals(sub) for sub in terminal_expression["or_args"]]   # type: ignore

            return any(inner_values)

        # Interestingly, in the grammar a terminal comparison can only over have 2 arguments (i.e. there can be no
        # (= arg1 arg2 arg3)), so this makes extracting the expressions a bit more straightforward
        elif rule == "terminal_comp":
            comparison_operator = terminal_expression["op"]

            expr_1 = self.score(terminal_expression["expr_1"]["expr"]) # type: ignore
            expr_2 = self.score(terminal_expression["expr_2"]["expr"]) # type: ignore

            if comparison_operator == "=":
                return expr_1 == expr_2
            elif comparison_operator == "<":
                return expr_1 < expr_2
            elif comparison_operator == "<=":
                return expr_1 <= expr_2
            elif comparison_operator == ">":
                return expr_1 > expr_2
            elif comparison_operator == ">=":
                return expr_1 >= expr_2
            else:
                raise ValueError(f"Error: Unknown comparison operator '{comparison_operator}'")

        else:
            raise ValueError(f"Error: Unknown terminal rule '{rule}'")

    def _extract_name_and_types(self, scoring_expression: tatsu.ast.AST) -> typing.Tuple[str, typing.Optional[typing.Sequence[str]]]:
        '''
        Helper function to extract the name of the preference being scored, as well as any of the object types that have been
        passed to it using the ":" syntax
        '''
        name_and_types = typing.cast(tatsu.ast.AST, scoring_expression["name_and_types"])
        preference_name = name_and_types["pref_name"]

        if isinstance(name_and_types["object_types"], tatsu.ast.AST):
            object_types = [name_and_types["object_types"]["type_name"]]  # type: ignore

        elif isinstance(name_and_types["object_types"], list):
            object_types = [object_type["type_name"] for object_type in name_and_types["object_types"]]  # type: ignore

        else:
            object_types = None
        
        return str(preference_name), object_types


    def _filter_satisfactions(self, preference_name: str, object_types: typing.Optional[typing.Sequence[str]]):
        '''
        Given the name of a preference and a list of object types, return the set of all satisfactions
        of the given preference that abide by the given variable mapping. In the case where object_types
        is None, this amounts to returning all the satisfactions of the given preference. When object_types
        is not None (as in the case of an external forall and the use of the ":" syntax), then we filter
        accordingly.
        '''

        if object_types is None:
            return self.preference_satisfactions[preference_name]

        pref_handler = self.preference_handlers[preference_name]
        satisfactions = []

        # Check to see if the mapping lines up with the specified object type
        for potential_sat in self.preference_satisfactions[preference_name]:
            mapping = potential_sat.mapping
            
            # A PreferenceHandler's additional_variable_mapping is an OrderedDict, so the types in 
            # object_types need to match the order of varaibles in the external forall
            acceptable_sat = True
            for variable, object_type in zip(pref_handler.additional_variable_mapping.keys(), object_types):
                specififed_object = mapping[variable]
                self.domain_name = typing.cast(str, self.domain_name)
                if specififed_object not in OBJECTS_BY_ROOM_AND_TYPE[self.domain_name][object_type]:
                    acceptable_sat = False

            if acceptable_sat:
                satisfactions.append(potential_sat)

        return satisfactions

    def score(self, scoring_expression: typing.Union[str, tatsu.ast.AST, None]) -> float:
        '''
        Determine the score of the current trajectory using the given scoring expression
        '''
        if scoring_expression is None:
            return 0.0

        # TODO: is the only situation in which we'll directly score a string?
        if isinstance(scoring_expression, str):
            return float(scoring_expression)
        
        rule = scoring_expression["parseinfo"].rule  # type: ignore

        # TODO: clearly there needs to be some logic here for handling maximize vs. minimize. Maybe we should
        # pass an argument down the recursive calls?
        if rule == "scoring_maximize":
            return self.score(scoring_expression["expr"])

        elif rule == "scoring_minimize":
            return self.score(scoring_expression["expr"])

        elif rule == "scoring_expr":
            return self.score(scoring_expression["expr"])

        elif rule == "scoring_multi_expr":
            # Multi-expression operators are either addition (+) or multiplication (*)
            operator = scoring_expression["op"]

            # Despite being multi-expression operators, they can still accept one or more arguments
            expressions = scoring_expression["expr"]

            if isinstance(expressions, tatsu.ast.AST):
                return self.score(expressions)

            elif isinstance(expressions, list):
                if operator == "+":
                    return sum([self.score(expression) for expression in expressions])

                elif operator == "*":
                    return prod([self.score(expression) for expression in expressions])

        elif rule == "scoring_binary_expr":
            # Binary expression operators are either subtraction (-) or division (/)
            operator = scoring_expression["op"]

            expr_1 = scoring_expression["expr_1"]
            expr_2 = scoring_expression["expr_2"]

            if operator == "-":
                return self.score(expr_1) - self.score(expr_2)
            elif operator == "/":
                return self.score(expr_1) / self.score(expr_2)

        elif rule == "scoring_neg_expr":
            return - self.score(scoring_expression["expr"])

        elif rule == "scoring_comparison":
            comp_expr = typing.cast(tatsu.ast.AST, scoring_expression["comp"])
            comparison_operator = comp_expr["op"]  

            # In this case, we know that the operator is = and that we have more than 2 comparison arguments,
            # so we just determine whether all arguments evaluate to the same value
            if comparison_operator is None:
                expressions = comp_expr["expr"]
                evaluations = [self.score(expr) for expr in expressions]  # type: ignore

                return float(evaluations.count(evaluations[0]) == len(evaluations))

            # Otherwise, there will be exactly two comparison arguments and we can compare them normally
            else:
                expr_1 = comp_expr["expr_1"]
                expr_2 = comp_expr["expr_2"]

                if comparison_operator == "=":
                    return self.score(expr_1) == self.score(expr_2)
                elif comparison_operator == "<":
                    return self.score(expr_1) < self.score(expr_2)
                elif comparison_operator == "<=":
                    return self.score(expr_1) <= self.score(expr_2)
                elif comparison_operator == ">":
                    return self.score(expr_1) > self.score(expr_2)
                elif comparison_operator == ">=":
                    return self.score(expr_1) >= self.score(expr_2)
                else:
                    raise ValueError(f"Error: Unknown comparison operator '{comparison_operator}'")

        elif rule == "preference_eval":
            return self.score(scoring_expression["count_method"])

        # Count the number of satisfactions of the given preference that don't overlap in both
        # (a) the mapping of variables to objects
        # (b) the temporal states involved
        elif rule == "count_nonoverlapping":
            preference_name, object_types = self._extract_name_and_types(scoring_expression)

            satisfactions = self._filter_satisfactions(preference_name, object_types)

            # Group the satisfactions by their mappings. Within each group, ensure there are no state overlaps and
            # count the total number of satisfactions that satisfy those criteria
            count = 0

            keyfunc = lambda satisfaction: "_".join(satisfaction[0].values())
            for key, group in itertools.groupby(sorted(satisfactions, key=keyfunc), keyfunc):
                group = list(sorted(group, key=lambda satisfaction: satisfaction[2]))

                prev_end = -1
                for mapping, start, end, measures in group:
                    if start >= prev_end:
                        prev_end = end
                        count += 1

            return count

        # Count whether the preference has been satisfied at all
        elif rule == "count_once":
            preference_name, object_types = self._extract_name_and_types(scoring_expression)

            satisfactions = self._filter_satisfactions(preference_name, object_types)

            return 1 if len(satisfactions) > 0 else 0

        # Count the number of satisfactions of the given preference that use distinct variable mappings
        elif rule == "count_once_per_objects":
            preference_name, object_types = self._extract_name_and_types(scoring_expression)

            satisfactions = self._filter_satisfactions(preference_name, object_types)

            count = 0

            keyfunc = lambda satisfaction: "_".join(satisfaction[0].values())
            for key, group in itertools.groupby(sorted(satisfactions, key=keyfunc), keyfunc):
                count += 1

            return count

        # For each nonoverlapping satisfaction (see count_nonoverlapping above), sum the value of the measurement
        elif rule == "count_nonoverlapping_measure":
            preference_name, object_types = self._extract_name_and_types(scoring_expression)

            satisfactions = self._filter_satisfactions(preference_name, object_types)

            count = 0

            keyfunc = lambda satisfaction: "_".join(satisfaction[0].values())
            for key, group in itertools.groupby(sorted(satisfactions, key=keyfunc), keyfunc):
                group = list(sorted(group, key=lambda satisfaction: satisfaction[2]))

                prev_end = -1
                for mapping, start, end, measures in group:
                    if start >= prev_end:
                        prev_end = end
                        count += list(measures.values())[0] # TODO: will we only ever have one measurement per preference?

            return count

        elif rule == "count_unique_positions":
            pass # TODO

        elif rule == "count_same_positions":
            pass # TODO

        elif rule == "count_maximal_nonoverlapping":
            pass # TODO

        elif rule == "count_maximal_overlapping":
            pass # TODO

        elif rule == "count_maximal_once_per_objects":
            pass # TODO

        elif rule == "count_maximal_once":
            pass # TODO

        elif rule == "count_once_per_external_objects":
            pass # TODO

        else:
            raise ValueError(f"Error: Unknown rule '{rule}' in scoring expression")

        return 0.0


if __name__ == "__main__":
    test_game_1 = """
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

    test_game_forall = """
    (define (game 61267978e96853d3b974ca53-23) (:domain few-objects-room-v1)

    (:constraints (and 
        (forall (?b - (either ball golfball)) 
            (preference ballThrownToBin (exists (?h - bin)
                (then
                    (once (agent_holds ?b))
                    (hold (and (not (agent_holds ?b)) (in_motion ?b))) 
                    (once (and (not (in_motion ?b)) (in ?h ?b)))
                )
            ))
        )
    ))
    (:scoring maximize (+
        (count-nonoverlapping ballThrownToBin:golfball)
        (* 5 (= (count-nonoverlapping ballThrownToBin:ball) 1))
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

    test_game_3 = """
    (define (game 616da508e4014f74f43c8433-77) (:domain many-objects-room-v1)

    (:constraints (and 
        (preference throwToBinFromDistance (exists (?d - ball ?h - bin)
            (then 
                (once-measure (agent_holds ?d) (distance agent ?h))
                (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        ))
    )) 
    (:scoring maximize (count-nonoverlapping-measure throwToBinFromDistance)
    ))
    """

    test_game_4 = """
    (define (game 61015f63f9a351d3171a0f98-105) (:domain few-objects-room-v1)  ; 105
    (:setup (and 
        (forall (?c - cube_block) (game-optional (on rug ?c)))
        (game-optional (not (exists (?o - game_object) (above ?o desk))))
        (forall (?d - dodgeball) (game-conserved (not (exists (?s - shelf) (on ?s ?d)))))
    ))
    (:constraints (and 
        (preference woodenBlockMovedFromRugToDesk (exists (?b - tan_cube_block)
            (then 
                (once (and 
                    (forall (?c - (either blue_cube_block yellow_cube_block)) (on rug ?c))
                    (on rug ?b)
                ))
                (hold (forall (?c - (either blue_cube_block yellow_cube_block)) (or
                    (on rug ?c) 
                    (agent_holds ?c)
                    (in_motion ?c)
                    (exists (?c2 - (either blue_cube_block yellow_cube_block)) (and 
                        (not (= ?c ?c2))
                        (< (distance ?c ?c2) 0.5)
                        (on floor ?c)
                        (on floor ?c2) 
                    ))
                )))
                (hold (forall (?c - (either blue_cube_block yellow_cube_block))
                    (< (distance desk ?c) 1)
                ))
                (once (above ?b desk)) 
            )  
        ))
    ))
    (:scoring maximize
        (count-once-per-objects woodenBlockMovedFromRugToDesk)
    ))
    """

    game_handler = GameHandler(test_game_forall)
    score = None

    for idx, state in enumerate(SAMPLE_TRAJECTORY):
        print(f"\n\n================================PROCESSING STATE {idx+1}================================")
        score = game_handler.process(state)
        if score is not None:
            break

    if score is not None:
        print("\n\nSCORE ACHIEVED:", score)


    # satisfactions = [({"?a": "ball"}, 1, 10),
    #                  ({"?a": "ball-2"}, 3, 7),
    #                  ({"?a": "ball"}, 2, 4),
    #                  ({"?a": "ball"}, 3, 7),
    #                  ({"?a": "ball"}, 4, 8),
    #                  ({"?a": "ball"}, 1, 3),
    #                  ({"?a": "ball-3"}, 1, 10)
    #                 ]

    # count = 0

    # keyfunc = lambda satisfaction: "_".join(satisfaction[0].values())
    # for key, group in itertools.groupby(sorted(satisfactions, key=keyfunc), keyfunc):
    #     # print("New group! Key =", key)
    #     group = list(sorted(group, key=lambda satisfaction: satisfaction[2]))

    #     prev_end = -1
    #     for mapping, start, end in group:
    #         if start >= prev_end:
    #             prev_end = end
    #             count += 1


    # print("\nCount:", count)