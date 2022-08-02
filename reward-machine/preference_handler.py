import ast as pyast
import itertools
import os
import re
import sys
import tatsu

import numpy as np

from config import OBJECTS_BY_TYPE, PREDICATE_LIBRARY, FUNCTION_LIBRARY, SAMPLE_TRAJECTORY

class PreferenceHandler():
    def __init__(self, preference):
        # Validity check
        assert isinstance(preference, tatsu.ast.AST) and "definition" in preference

        self.preference_name = preference["definition"]["pref_name"]
        body = preference["definition"]["pref_body"]["body"]

        # Extract the mapping of variable names to types (e.g. {?d : dodgeball})
        self.variable_mapping = self._extract_variable_mapping(body["exists_vars"]["variables"])
        self.variable_mapping["agent"] = "agent"

        # Extract the ordered list of temporal predicates
        self.temporal_predicates = [func["seq_func"] for func in body["exists_args"]["body"]["then_funcs"]]

        # A list of tuples, containing the state of the preference evaluated on partial sets of arguments. 
        # To begin, the list contains an entry for every possible assignment of objects to the arguments 
        # in the first temporal predicate. In addition, each entry includes the current state, and the index
        # of current predicate (these are potentially distinct because a hold-while generates 2 states)
        #
        # State info: (current_predicate: None or ast.AST, next_predicate: None or ast.AST, while_sat: Boolean)
        #
        # EXAMPLE:
        #      [({?d : blue-dodgeball-1}, None, _once, False),
        #       ({?d : red-dodgeball-1}, None, _once, False),
        #       ({?d : pink-dodgeball-1, ?w: left-wall}, _once, _hold_while, False),
        #       ({?d : pink-dodgeball-1, ?w: right-wall}, _once", _hold_while, False)
        #      ]
        self.partial_preference_satisfactions = []

        initial_variables = self._extract_variables(self.temporal_predicates[0])
        initial_var_types = [self.variable_mapping[var] for var in initial_variables]
        object_assignments = list(itertools.product(*[OBJECTS_BY_TYPE[var_type] for var_type in initial_var_types]))

        for object_assignment in object_assignments:
            mapping = dict(zip(initial_variables, object_assignment))
            self.partial_preference_satisfactions.append((mapping, None, self.temporal_predicates[0], False))

    def _extract_variable_mapping(self, variable_list):
        if isinstance(variable_list, tatsu.ast.AST):
            variable_list = [variable_list]

        variables = {}
        for var_info in variable_list:
            variables[var_info["var_names"]] = var_info["var_type"]["type"]

        return variables

    def _extract_variables(self, predicate):
        '''
        Recursively extract every variable referenced in the predicate (including inside functions 
        used within the predicate)
        '''

        if isinstance(predicate, list) or isinstance(predicate, tuple):
            pred_vars = []
            for sub_predicate in predicate:
                pred_vars += self._extract_variables(sub_predicate)

            return list(set(pred_vars))

        elif isinstance(predicate, tatsu.ast.AST):
            pred_vars = []
            for key in predicate:
                if key == "term":

                    # Different structure for predicate args vs. function args
                    if isinstance(predicate["term"], tatsu.ast.AST):
                        pred_vars += [predicate["term"]["arg"]]
                    else:
                        pred_vars += [predicate["term"]]

                elif key != "parseinfo":
                    pred_vars += self._extract_variables(predicate[key])

            return list(set(pred_vars))

        else:
            return []

    def _type(self, predicate):
        '''
        Returns the temporal logic type of a given predicate
        '''
        if "once_pred" in predicate.keys():
            return "once"

        elif "once_measure_pred" in predicate.keys():
            return "once-measure"

        elif "hold_pred" in predicate.keys():

            if "while_preds" in predicate.keys():
                return "hold-while"

            return "hold"

        else:
            exit("Error: predicate does not have a temporal logic type")

    def advance_preference(self, mapping, current_predicate, next_predicate, new_preference_satisfactions):
        '''
        Called when a predicate inside a (then) operator has been fully satisfied and we are moving to the
        next predicate. This function adds new partial object mappings and predicates to the provided list
        called "new_preference_satisfactions"

        TODO: should only add non-duplicates to new_preference_satisfactions (a duplicate is possible in the
        case where the initial mapping was split, before one of the later branches was reverted back to the
        initial state. If the first predicate is satisfied again, the initial mapping will again split and
        we need to make sure not to add duplicate branches back in)
        '''

        next_pred_idx = self.temporal_predicates.index(next_predicate)
        new_cur_predicate = next_predicate

        # Check to see whether we've just satisfied the last predicate of the (then) operator, in which case
        # the entire preference has been satisfied!
        if next_pred_idx+1 == len(self.temporal_predicates):
            print("\n\tPREFERENCE SATISFIED!")
            return

        else:
            new_next_predicate = self.temporal_predicates[next_pred_idx+1] # TODO: make sure this doesn't go off the end of the list

            # Determine all of the variables referenced in the new predicate that aren't referenced already
            new_variables = [var for var in self._extract_variables(new_next_predicate) if var not in mapping]

        print("\n\tNew variables required by the next predicate:", new_variables)

        # If there are new variables, then we iterate overall all possible assignments for them, add them to the
        # existing mapping, and add it to our list of partial preference satisfactions while advancing the predicates
        if len(new_variables) > 0:
            new_var_types = [self.variable_mapping[var] for var in new_variables]
            object_assignments = list(itertools.product(*[OBJECTS_BY_TYPE[var_type] for var_type in new_var_types]))

            for object_assignment in object_assignments:
                new_mapping = dict(zip(new_variables, object_assignment))
                new_mapping.update(mapping)

                new_preference_satisfactions.append((new_mapping, new_cur_predicate, new_next_predicate, False))

        # Otherwise, just advance the predicates but keep the mapping the same
        else:
            new_preference_satisfactions.append((mapping, new_cur_predicate, new_next_predicate, False))

    def revert_preference(self, mapping, new_preference_satisfactions):
        '''
        Called when a predicate inside a (then) operator is no longer satisfied and we have to return to
        the start state. This function will add at most one tuple to new_preference_satisfactions that
        represents the "initial component" of the current mapping: the portion of the mapping that consists
        of variables required by the first predicate. Importantly, a tuple is only added if it is a non-duplicate
        '''
        initial_variables = self._extract_variables(self.temporal_predicates[0])
        new_mapping = {key: val for key, val in mapping.items() if key in initial_variables}
        
        new_preference_satisfactions.append((new_mapping, None, self.temporal_predicates[0], False))

    def process(self, traj_state):
        '''
        Take a state from an active trajectory and update each of the internal states based on the
        satisfcation of predicates and the rules of the temporal logic operators
        '''

        new_preference_satisfactions = []

        for mapping, current_predicate, next_predicate, while_sat in self.partial_preference_satisfactions:
            cur_predicate_type = None if current_predicate is None else self._type(current_predicate)
            next_predicate_type = None if next_predicate is None else self._type(next_predicate)

            print("\nEvaluating a new partial satisfaction:")
            print("\tMapping:", mapping)
            print("\tCurrent predicate type:", cur_predicate_type)
            print("\tNext predicate type:", next_predicate_type)
            print("\tWhile-condition satisfied?", while_sat)

            # The "Start" state: transition forward if the basic condition of the next predicate is met
            if cur_predicate_type is None:
                if next_predicate_type == "once":
                    pred_eval = self.evaluate_predicate(next_predicate["once_pred"]["pred"], traj_state, mapping)
                elif next_predicate_type in ["hold", "hold-while"]:
                    pred_eval = self.evaluate_predicate(next_predicate["hold_pred"]["pred"], traj_state, mapping)

                print("\n\tEvaluation of next predicate:", pred_eval)

                # If the basic condition of the next predicate is met, we'll advance the predicates through the (then) operator
                if pred_eval:
                    self.advance_preference(mapping, current_predicate, next_predicate, new_preference_satisfactions)

                # If not, then just add the same predicates back to the list
                else:
                    new_preference_satisfactions.append((mapping, current_predicate, next_predicate, False))


            elif cur_predicate_type == "once":
                if next_predicate_type == "once":
                    next_pred_eval = self.evaluate_predicate(next_predicate["once_pred"]["pred"], traj_state, mapping)
                elif next_predicate_type in ["hold", "hold-while"]:
                    next_pred_eval = self.evaluate_predicate(next_predicate["hold_pred"]["pred"], traj_state, mapping)

                cur_pred_eval = self.evaluate_predicate(current_predicate["once_pred"]["pred"], traj_state, mapping)

                print("\n\tEvaluation of next predicate:", next_pred_eval)
                print("\tEvaluation of current predicate:", cur_pred_eval)

                # If the next predicate is satisfied, then we advance regardless of the state of the current predicate
                if next_pred_eval:
                    self.advance_preference(mapping, current_predicate, next_predicate, new_preference_satisfactions)

                # If the next predicate *isn't* satisfied, but the current one *is* then we stay in our current state 
                elif cur_pred_eval:
                    new_preference_satisfactions.append((mapping, current_predicate, next_predicate, False))

                # If neither are satisfied, we return to the start
                else:
                    self.revert_preference(mapping, new_preference_satisfactions)

            elif cur_predicate_type == "hold":
                if next_predicate_type == "once":
                    next_pred_eval = self.evaluate_predicate(next_predicate["once_pred"]["pred"], traj_state, mapping)
                elif next_predicate_type in ["hold", "hold-while"]:
                    next_pred_eval = self.evaluate_predicate(next_predicate["hold_pred"]["pred"], traj_state, mapping)

                cur_pred_eval = self.evaluate_predicate(current_predicate["hold_pred"]["pred"], traj_state, mapping)

                print("\n\tEvaluation of next predicate:", next_pred_eval)
                print("\tEvaluation of current predicate:", cur_pred_eval)

                # If the next predicate is satisfied, then we advance regardless of the state of the current predicate
                if next_pred_eval:
                    self.advance_preference(mapping, current_predicate, next_predicate, new_preference_satisfactions)

                # If the next predicate *isn't* satisfied, but the current one *is* then we stay in our current state 
                elif cur_pred_eval:
                    new_preference_satisfactions.append((mapping, current_predicate, next_predicate, False))

                # If neither are satisfied, we return to the start
                else:
                    self.revert_preference(mapping, new_preference_satisfactions)

            elif cur_predicate_type == "hold-while":
                # If the while condition has already been met, then we can treat this exactly like a normal hold
                if while_sat:
                    if next_predicate_type == "once":
                        next_pred_eval = self.evaluate_predicate(next_predicate["once_pred"]["pred"], traj_state, mapping)
                    elif next_predicate_type in ["hold", "hold-while"]:
                        next_pred_eval = self.evaluate_predicate(next_predicate["hold_pred"]["pred"], traj_state, mapping)

                    cur_pred_eval = self.evaluate_predicate(current_predicate["hold_pred"]["pred"], traj_state, mapping)

                    print("\n\tEvaluation of next predicate:", next_pred_eval)
                    print("\tEvaluation of current predicate:", cur_pred_eval)

                    # If the next predicate is satisfied, then we advance regardless of the state of the current predicate
                    if next_pred_eval:
                        self.advance_preference(mapping, current_predicate, next_predicate, new_preference_satisfactions)

                    # If the next predicate *isn't* satisfied, but the current one *is* then we stay in our current state 
                    elif cur_pred_eval:
                        new_preference_satisfactions.append((mapping, current_predicate, next_predicate, True))

                    # If neither are satisfied, we return to the start
                    else:
                        self.revert_preference(mapping, new_preference_satisfactions)

                # If not, then we only care about the while condition and the current hold
                else:
                    cur_pred_eval = self.evaluate_predicate(current_predicate["hold_pred"]["pred"], traj_state, mapping)

                    # TODO: there can be more than one while predicate, and they all need to be evaluated in sequence :(
                    cur_while_eval = self.evaluate_predicate(current_predicate["while_preds"]["pred"], traj_state, mapping)

                    print("\n\tEvaluation of current predicate:", cur_pred_eval)
                    print("\tEvaluation of current while pred:", cur_while_eval)

                    if cur_pred_eval:
                        if cur_while_eval:
                            new_preference_satisfactions.append((mapping, current_predicate, next_predicate, True))

                        else:
                            new_preference_satisfactions.append((mapping, current_predicate, next_predicate, False))

                    else:
                        self.revert_preference(mapping, new_preference_satisfactions)

        # Janky way to remove duplicates: group by the concatenation of every value in the mapping, so each
        # specific assignment is represented by a different string.
        # TODO: figure out if this will ever break down
        keyfunc = lambda pref_sat: "_".join(pref_sat[0].values())
        new_preference_satisfactions = [list(g)[0] for k, g in itertools.groupby(
                                        sorted(new_preference_satisfactions, key=keyfunc), keyfunc)]

        self.partial_preference_satisfactions = new_preference_satisfactions

    def evaluate_predicate(self, predicate, state, mapping):
        '''
        Given a predicate, a trajectory state, and an assignment of each of the predicate's
        arguments to specific objects in the state, returns the evaluation of the predicate

        TODO: predicates always include a key called "parseinfo", which in turn includes a child
              called "rule". So we can do the if / elif / elif switching on that instead, which
              is more robust / clean.
        '''
        for key in predicate:
            if key == "pred_name":
                # Obtain the functional representation of the base predicate
                predicate_fn = PREDICATE_LIBRARY[predicate["pred_name"]]

                # Map the variables names to the object names, and extract them from the state
                predicate_args = [state["objects"][mapping[var]] for var in self._extract_variables(predicate)]
                
                # Evaluate the predicate
                evaluation = predicate_fn(state, *predicate_args)

                return evaluation

            elif key == "not_args":
                return not self.evaluate_predicate(predicate[key]["pred"], state, mapping)

            elif key == "and_args":
                return all([self.evaluate_predicate(sub["pred"], state, mapping) for sub in predicate[key]])

            elif key == "or_args":
                return any([self.evaluate_predicate(sub["pred"], state, mapping) for sub in predicate[key]])

            elif key == "comp":
                comparison_operator = predicate[key]["comp_op"]

                # For each comparison argument, evaluate it if it's a function or convert to an int if not
                comp_arg_1 = predicate[key]["arg_1"]["arg"]
                if isinstance(comp_arg_1, tatsu.ast.AST):
                    function = FUNCTION_LIBRARY[comp_arg_1["func_name"]]
                    function_args = [state["objects"][mapping[var]] for var in self._extract_variables(comp_arg_1)]

                    comp_arg_1 = function(*function_args)

                else:
                    comp_arg_1 = int(comp_arg_1)

                comp_arg_2 = predicate[key]["arg_2"]["arg"]
                if isinstance(comp_arg_1, tatsu.ast.AST):
                    function = FUNCTION_LIBRARY[comp_arg_2["func_name"]]
                    function_args = [state["objects"][mapping[var]] for var in self._extract_variables(comp_arg_2)]

                    comp_arg_2 = function(*function_args)

                else:
                    comp_arg_2 = int(comp_arg_2)

                if comparison_operator == "=":
                    return comp_arg_1 == comp_arg_2
                elif comparison_operator == "<":
                    return comp_arg_1 < comp_arg_2
                elif comparison_operator == "<=":
                    return comp_arg_1 <= comp_arg_2
                elif comparison_operator == ">":
                    return comp_arg_1 > comp_arg_2
                elif comparison_operator == ">=":
                    return comp_arg_1 >= comp_arg_2
                else:
                    exit("Error: unknown comparison operator")



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
                    (once (agent_holds ?d))
                    (hold-while (and (not (agent_holds ?d)) (in_motion ?d)) (agent_holds ?h))
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
    
    pref1 = preferences[0]
    handler1 = PreferenceHandler(pref1)
    
    for idx, state in enumerate(SAMPLE_TRAJECTORY):
        print(f"\n\n================================PROCESSING STATE {idx+1}================================")
        handler1.process(state)
