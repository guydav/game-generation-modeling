import ast as pyast
import itertools
import os
import re
import sys
import tatsu
import typing
import tatsu.ast
import enum
import copy

import numpy as np

from utils import extract_variable_type_mapping, extract_variables, get_object_assignments
from predicate_handler import PredicateHandler, FUNCTION_LIBRARY

from config import NAMED_OBJECTS, SAMPLE_TRAJECTORY


class PartialPreferenceSatisfcation(typing.NamedTuple):
    mapping: typing.Dict[str, str]
    current_predicate: typing.Optional[tatsu.ast.AST]
    next_predicate: typing.Optional[tatsu.ast.AST]
    while_sat: int
    start: int
    measures: typing.Optional[dict]


class PreferenceSatisfcation(typing.NamedTuple):
    mapping: typing.Dict[str, str]
    start: int
    end: int
    measures: typing.Optional[dict]


class PredicateType(enum.Enum):
    ONCE = 1
    ONCE_MEASURE = 2
    HOLD = 3
    HOLD_WHILE = 4


class PreferenceHandler():
    def __init__(self, preference: tatsu.ast.AST, predicate_handler: PredicateHandler, domain: str,
                 additional_variable_mapping: typing.Dict[str, str] = {}):
        '''
        Construct a handler object for the provided preference, responsible for tracking when and how the
        the preference has been satisfied by the various objects in the state using its process() method

        preference: the preference to monitor (tatsu.ast.AST)
        predicate_handler: object passed from the GameHandler which actually evaluates (and caches) the predicates
        additional_variable_mapping: variable to type mapping beyond what is specified inside the preference's own
            quantification. This is used to handle external an "forall", which quantifies additional variables 
        '''
        # Validity check
        assert isinstance(preference, tatsu.ast.AST) and preference["parseinfo"].rule == "preference"  # type: ignore

        self.preference = preference
        self.predicate_handler = predicate_handler
        self.domain = domain

        self.preference_name = preference["pref_name"]
        body = preference["pref_body"]["body"]  # type: ignore

        # Extract the mapping of variable names to types (e.g. {?d : dodgeball})
        self.variable_type_mapping = extract_variable_type_mapping(body["exists_vars"]["variables"])

        # Add additional variable mapping, and store it as well (it's needed for scoring)
        self.additional_variable_mapping = additional_variable_mapping
        self.variable_type_mapping.update(self.additional_variable_mapping)

        # Add all of the explicitly named variables. This includes "agent", but also things like "desk" that
        # can just be referred to explicitly within predicates without quantification beforehand
        self.variable_type_mapping.update({obj: [obj] for obj in NAMED_OBJECTS})

        # Extract the ordered list of temporal predicates
        self.temporal_predicates = [func["seq_func"] for func in body["exists_args"]["body"]["then_funcs"]]

        # A list of tuples, containing the state of the preference evaluated on partial sets of arguments.
        # Each tuple includes a partial mapping from variables to specific objects, the current predicate,
        # the next predicate, and the number of while conditions satisfied at the current state, which is
        # only important for hold-while predicates.
        #
        # State info: (current_predicate: None or ast.AST, next_predicate: None or ast.AST, while_sat: int,
        #              start: int, measures: dict)
        #
        #
        # EXAMPLE:
        #      [({?d : blue-dodgeball-1}, None, _once, 0, -1, {}),
        #       ({?d : red-dodgeball-1}, None, _once, 0, -1, {}),
        #       ({?d : pink-dodgeball-1, ?w: left-wall}, _once, _hold_while, 0, 2, {}),
        #       ({?d : pink-dodgeball-1, ?w: right-wall}, _once", _hold_while, 0, 2, {})
        #      ]
        self.partial_preference_satisfactions = []

        initial_variables = extract_variables(self.temporal_predicates[0])
        initial_var_types = [self.variable_type_mapping[var] for var in initial_variables]
        object_assignments = get_object_assignments(self.domain, initial_var_types)

        for object_assignment in object_assignments:
            mapping = dict(zip(initial_variables, object_assignment))
            self.partial_preference_satisfactions.append((mapping, None, self.temporal_predicates[0], 0, -1, {}))

        # A list of all versions of the predicate satisfied at a particular step, updated during self.process()
        self.satisfied_this_step = []

        self.cur_step = 1

    def _predicate_type(self, predicate: tatsu.ast.AST) -> PredicateType:
        '''
        Returns the temporal logic type of a given predicate
        '''
        if "once_pred" in predicate.keys():
            return PredicateType.ONCE

        elif "once_measure_pred" in predicate.keys():
            return PredicateType.ONCE_MEASURE

        elif "hold_pred" in predicate.keys():

            if "while_preds" in predicate.keys():
                return PredicateType.HOLD_WHILE

            return PredicateType.HOLD

        else:
            raise ValueError("Error: predicate does not have a temporal logic type")

    def advance_preference(self, partial_preference_satisfcation: PartialPreferenceSatisfcation, new_partial_preference_satisfactions: typing.List[PartialPreferenceSatisfcation]):
        '''
        Called when a predicate inside a (then) operator has been fully satisfied and we are moving to the
        next predicate. This function adds new partial object mappings and predicates to the provided list
        called "new_partial_preference_satisfactions"

        TODO: should only add non-duplicates to new_partial_preference_satisfactions (a duplicate is possible in the
        case where the initial mapping was split, before one of the later branches was reverted back to the
        initial state. If the first predicate is satisfied again, the initial mapping will again split and
        we need to make sure not to add duplicate branches back in)

        '''

        next_pred_idx = self.temporal_predicates.index(partial_preference_satisfcation.next_predicate)
        new_cur_predicate = partial_preference_satisfcation.next_predicate

        # Check to see whether we've just satisfied the last predicate of the (then) operator, in which case
        # the entire preference has been satisfied! Add the current mapping to the list satisfied at this step
        # and add the reverted mapping back to new_partial_preference_satisfactions
        if next_pred_idx + 1 == len(self.temporal_predicates):
            print("\n\tPREFERENCE SATISFIED!")
            self.satisfied_this_step.append(PreferenceSatisfcation(partial_preference_satisfcation.mapping, partial_preference_satisfcation.start, 
                self.cur_step, partial_preference_satisfcation.measures))
            self.revert_preference(partial_preference_satisfcation.mapping, new_partial_preference_satisfactions)
            return

        else:
            new_next_predicate = self.temporal_predicates[next_pred_idx + 1]
            # Determine all of the variables referenced in the new predicate that aren't referenced already
            new_variables = [var for var in extract_variables(new_next_predicate) if var not in partial_preference_satisfcation.mapping]

        print("\n\tNew variables required by the next predicate:", new_variables)

        # If there are new variables, then we iterate overall all possible assignments for them, add them to the
        # existing mapping, and add it to our list of partial preference satisfactions while advancing the predicates
        if len(new_variables) > 0:
            new_var_types = [self.variable_type_mapping[var] for var in new_variables]
            object_assignments = get_object_assignments(self.domain, new_var_types)

            for object_assignment in object_assignments:
                new_mapping = dict(zip(new_variables, object_assignment))
                new_mapping.update(partial_preference_satisfcation.mapping)

                new_partial_preference_satisfactions.append(partial_preference_satisfcation._replace(mapping=new_mapping, current_predicate=new_cur_predicate, 
                                                                                                     next_predicate=new_next_predicate, while_sat=0))

        # Otherwise, just advance the predicates but keep the mapping the same
        else:
            new_partial_preference_satisfactions.append(partial_preference_satisfcation._replace(current_predicate=new_cur_predicate, next_predicate=new_next_predicate,
                                                                                                 while_sat=0))

    def revert_preference(self, mapping: typing.Dict[str, str], new_partial_preference_satisfactions: typing.List[PartialPreferenceSatisfcation]) -> None:
        '''
        Called when a predicate inside a (then) operator is no longer satisfied and we have to return to
        the start state. This function will add at most one tuple to new_partial_preference_satisfactions that
        represents the "initial component" of the current mapping: the portion of the mapping that consists
        of variables required by the first predicate.
        '''
        initial_variables = extract_variables(self.temporal_predicates[0])
        new_mapping = {key: val for key, val in mapping.items() if key in initial_variables}
        
        new_partial_preference_satisfactions.append(PartialPreferenceSatisfcation(new_mapping, None, self.temporal_predicates[0], 0, -1, {}))

    def _evaluate_next_predicate(self, next_predicate_type: typing.Optional[PredicateType], next_predicate: tatsu.ast.AST, mapping: typing.Dict[str, str], traj_state: typing.Dict[str, typing.Any]) -> bool:
        next_predicate_value = None
        
        if next_predicate_type is None:
            return True
        elif next_predicate_type == PredicateType.ONCE:
            next_predicate_value = self.predicate_handler(next_predicate["once_pred"], traj_state, mapping)
        elif next_predicate_type == PredicateType.ONCE_MEASURE:
            next_predicate_value = self.predicate_handler(next_predicate["once_measure_pred"], traj_state, mapping)
        elif next_predicate_type in [PredicateType.HOLD, PredicateType.HOLD_WHILE]:
            next_predicate_value = self.predicate_handler(next_predicate["hold_pred"], traj_state, mapping)

        if next_predicate_value is None:
            next_predicate_value = False

        return next_predicate_value

    def process(self, traj_state: typing.Dict[str, typing.Any]) -> typing.List[PreferenceSatisfcation]:
        '''
        Take a state from an active trajectory and update each of the internal states based on the
        satisfcation of predicates and the rules of the temporal logic operators
        '''

        self.satisfied_this_step = []

        new_partial_preference_satisfactions = []

        for mapping, current_predicate, next_predicate, while_sat, start, measures in self.partial_preference_satisfactions:
            cur_predicate_type = None if current_predicate is None else self._predicate_type(current_predicate)
            next_predicate_type = None if next_predicate is None else self._predicate_type(next_predicate)

            print(f"\nEvaluating a new partial satisfaction for {self.preference_name}")
            print("\tMapping:", mapping)
            print("\tCurrent predicate type:", cur_predicate_type)
            print("\tNext predicate type:", next_predicate_type)
            print("\tWhile-conditions satisfied:", while_sat)
            print("\tPreference satisfcation start:", start)
            print("\tMeasures:", measures)

            # The "Start" state: transition forward if the basic condition of the next predicate is met

            pred_eval = None
            next_pred_eval = None

            if cur_predicate_type is None:
                pred_eval = self._evaluate_next_predicate(next_predicate_type, next_predicate, mapping, traj_state)
                print("\n\tEvaluation of next predicate:", pred_eval)

                # If the basic condition of the next predicate is met, we'll advance the predicates through the (then) operator.
                # We also record the current step as the "start" of the predicate being satisfied
                if pred_eval:
                    self.advance_preference(PartialPreferenceSatisfcation(mapping, current_predicate, next_predicate, 0, self.cur_step, measures),
                                            new_partial_preference_satisfactions)

                # If not, then just add the same predicates back to the list
                else:
                    new_partial_preference_satisfactions.append(PartialPreferenceSatisfcation(mapping, 
                        current_predicate, next_predicate, 0, start, measures))

            elif cur_predicate_type == PredicateType.ONCE:
                cur_pred_eval = self.predicate_handler(current_predicate["once_pred"], traj_state, mapping)
                next_pred_eval = self._evaluate_next_predicate(next_predicate_type, next_predicate, mapping, traj_state)
                
                print("\n\tEvaluation of next predicate:", next_pred_eval)
                print("\tEvaluation of current predicate:", cur_pred_eval)

                # If the next predicate is satisfied, then we advance regardless of the state of the current predicate
                if next_pred_eval:
                    self.advance_preference(PartialPreferenceSatisfcation(mapping, current_predicate, next_predicate, 0, start, measures),
                                            new_partial_preference_satisfactions)

                # If the next predicate *isn't* satisfied, but the current one *is* then we stay in our current state 
                elif cur_pred_eval:
                    new_partial_preference_satisfactions.append(PartialPreferenceSatisfcation(mapping, 
                        current_predicate, next_predicate, 0, start, measures))

                # If neither are satisfied, we return to the start
                else:
                    self.revert_preference(mapping, new_partial_preference_satisfactions)

            elif cur_predicate_type == PredicateType.ONCE_MEASURE:
                cur_pred_eval = self.predicate_handler(current_predicate["once_measure_pred"], traj_state, mapping)
                next_pred_eval = self._evaluate_next_predicate(next_predicate_type, next_predicate, mapping, traj_state)

                print("\n\tEvaluation of next predicate:", next_pred_eval)
                print("\tEvaluation of current predicate:", cur_pred_eval)

                measurement = current_predicate["measurement"]
                measurement_fn = FUNCTION_LIBRARY[measurement["func_name"]]

                # Map the variables names to the object names, and extract them from the state
                func_args = [traj_state["objects"][mapping[var]] for var in extract_variables(measurement)]

                evaluation = measurement_fn(*func_args)

                measures_copy = measures.copy()
                measures_copy[measurement["func_name"]] = evaluation

                # TODO: when we advance out of a once-measure, should we update the measurement on that frame? Or does
                # that frame technically not count as satisfying the once condition, meaning it should be excluded?

                # If the next predicate is satisfied, then we advance regardless of the state of the current predicate
                if next_pred_eval:
                    self.advance_preference(PartialPreferenceSatisfcation(mapping, current_predicate, next_predicate, 0, start, measures_copy),
                                            new_partial_preference_satisfactions)

                # If the next predicate *isn't* satisfied, but the current one *is* then we stay in our current state 
                elif cur_pred_eval:
                    new_partial_preference_satisfactions.append(PartialPreferenceSatisfcation(mapping, 
                        current_predicate, next_predicate, 0, start, measures_copy))

                # If neither are satisfied, we return to the start
                else:
                    self.revert_preference(mapping, new_partial_preference_satisfactions)

            elif cur_predicate_type == PredicateType.HOLD:
                cur_pred_eval = self.predicate_handler(current_predicate["hold_pred"], traj_state, mapping)
                next_pred_eval = self._evaluate_next_predicate(next_predicate_type, next_predicate, mapping, traj_state)

                print("\n\tEvaluation of next predicate:", next_pred_eval)
                print("\tEvaluation of current predicate:", cur_pred_eval)

                # If the next predicate is satisfied, then we advance regardless of the state of the current predicate
                if next_pred_eval:
                    self.advance_preference(PartialPreferenceSatisfcation(mapping, 
                        current_predicate, next_predicate, 0, start, measures), 
                        new_partial_preference_satisfactions)

                # If the next predicate *isn't* satisfied, but the current one *is* then we stay in our current state 
                elif cur_pred_eval:
                    new_partial_preference_satisfactions.append(PartialPreferenceSatisfcation(mapping, 
                        current_predicate, next_predicate, 0, start, measures))

                # If neither are satisfied, we return to the start
                else:
                    self.revert_preference(mapping, new_partial_preference_satisfactions)

            elif cur_predicate_type == PredicateType.HOLD_WHILE:
                num_while_conditions = 1 if isinstance(current_predicate["while_preds"], tatsu.ast.AST) else len(current_predicate["while_preds"])

                # If all of the while condition has already been met, then we can treat this exactly like a normal hold
                if while_sat == num_while_conditions:
                    cur_pred_eval = self.predicate_handler(current_predicate["hold_pred"], traj_state, mapping)
                    next_pred_eval = self._evaluate_next_predicate(next_predicate_type, next_predicate, mapping, traj_state)

                    print("\n\tEvaluation of next predicate:", next_pred_eval)
                    print("\tEvaluation of current predicate:", cur_pred_eval)

                    # If the next predicate is satisfied, then we advance regardless of the state of the current predicate
                    if next_pred_eval:
                        self.advance_preference(PartialPreferenceSatisfcation(mapping, 
                            current_predicate, next_predicate, 0, start, measures),
                            new_partial_preference_satisfactions)

                    # If the next predicate *isn't* satisfied, but the current one *is* then we stay in our current state 
                    elif cur_pred_eval:
                        new_partial_preference_satisfactions.append(PartialPreferenceSatisfcation(mapping, 
                            current_predicate, next_predicate, while_sat, start, measures))

                    # If neither are satisfied, we return to the start
                    else:
                        self.revert_preference(mapping, new_partial_preference_satisfactions)

                # If not, then we only care about the while condition and the current hold
                else:
                    cur_pred_eval = self.predicate_handler(current_predicate["hold_pred"], traj_state, mapping)

                    # Determine whether the next while condition is satisfied in the current state
                    if num_while_conditions == 1:
                        cur_while_eval = self.predicate_handler(current_predicate["while_preds"], traj_state, mapping)
                    else:
                        cur_while_eval = self.predicate_handler(current_predicate["while_preds"][while_sat], traj_state, mapping)

                    print("\n\tEvaluation of current predicate:", cur_pred_eval)
                    print("\tEvaluation of current while pred:", cur_while_eval)

                    if cur_pred_eval:
                        if cur_while_eval:
                            new_partial_preference_satisfactions.append(PartialPreferenceSatisfcation(mapping, 
                                current_predicate, next_predicate, while_sat + 1, start, measures))

                        else:
                            new_partial_preference_satisfactions.append(PartialPreferenceSatisfcation(mapping, 
                                current_predicate, next_predicate, while_sat, start, measures))

                    else:
                        self.revert_preference(mapping, new_partial_preference_satisfactions)

        # Janky way to remove duplicates: group by the concatenation of every value in the mapping, so each
        # specific assignment is represented by a different string.
        # TODO: figure out if this will ever break down
        keyfunc = lambda pref_sat: "_".join(pref_sat.mapping.values())
        new_partial_preference_satisfactions = [list(g)[0] for k, g in itertools.groupby(sorted(
            new_partial_preference_satisfactions, key=keyfunc), keyfunc)]

        self.partial_preference_satisfactions = new_partial_preference_satisfactions

        self.cur_step += 1

        return self.satisfied_this_step

    



if __name__ == "__main__":
    grammar_path= "../dsl/dsl.ebnf"
    grammar = open(grammar_path).read()
    grammar_parser = tatsu.compile(grammar)

    # test_game = """
    # (define (game 61267978e96853d3b974ca53-23) (:domain few-objects-room-v1)

    # (:constraints (and 
    #     (preference throwBallToBin
    #         (exists (?d - ball ?h - bin)
    #             (then
    #                 (once (or (agent_holds ?d) (and (in_motion ?h) (< (distance agent ?d) 5)) ))
    #                 (hold-while (and (not (agent_holds ?d)) (in_motion ?d)) (in_motion ?h) (agent_crouches))
    #                 (once-measure (and (not (in_motion ?d)) (in ?h ?d)) (color ?h))
    #             )
    #         )
    #     )
    #     (preference throwAttempt
    #         (exists (?d - ball)
    #             (then 
    #                 (once (agent_holds ?d))
    #                 (hold (and (not (agent_holds ?d)) (in_motion ?d))) 
    #                 (once (not (in_motion ?d)))
    #             )
    #         )
    #     )
    # ))
    # (:scoring maximize (+
    #     (count-nonoverlapping throwBallToBin)
    #     (- (/ (count-nonoverlapping throwAttempt) 5))
    # )))
    # """

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

    # Extra predicates
    # (once (and (agent_holds ?d) (in_motion ?d)))
    # (once (or (in ?d ?h) (and (in_motion ?d) (not (agent_holds ?d))) (agent_crouches)))
    # 
    # (hold (and (not (agent_holds ?d)) (in_motion ?d) (not (forall (?w - wall ?x - bin) (touch ?w ?x)))))

    DUMMY_STATE = {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [4, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "red"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [15, 10, 0],
                                                    "velocity": [1.0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "agent": {"name": "agent", "position": [0, 0, 0], "velocity": [0, 0, 0],
                                         "is_crouching": False, "holding": "red-dodgeball-1", "objectType": "agent"},
                              },

                   "game_start": True,
                   "game_over": False

                  }

    ast = grammar_parser.parse(test_game)

    preferences = ast[3][1]["preferences"]
    
    pref1 = preferences[1]["definition"]
    predicate_handler = PredicateHandler()
    handler1 = PreferenceHandler(pref1, predicate_handler)
    
    for idx, state in enumerate(SAMPLE_TRAJECTORY):
        print(f"\n\n================================PROCESSING STATE {idx+1}================================")
        handler1.process(state)
