import itertools
import os
import re
import sys
import tatsu

import numpy as np

from config import OBJECTS_BY_TYPE, PREDICATE_LIBRARY

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
        # in the first temporal predicate, each set to the START state (0).
        # E.X. [({?d : blue-dodgeball-1}, 0),
        #       ({?d : red-dodgeball-1}, 0),
        #       ({?d : pink-dodgeball-1, ?w: left-wall}, 1),
        #       ({?d : pink-dodgeball-1, ?w: right-wall}, 1)
        #      ]
        self.predicate_states = []

        initial_variables = self._extract_variables(self.temporal_predicates[0])
        initial_arg_types = [self.variable_mapping[arg] for arg in initial_variables]
        object_assignments = list(itertools.product(*[OBJECTS_BY_TYPE[arg_type] for arg_type in initial_arg_types]))

        for object_assignment in object_assignments:
            mapping = dict(zip(initial_variables, object_assignment))
            self.predicate_states.append((mapping, 0))


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

    def process(self, traj_state):
        '''
        Take a state from an active trajectory and update each of the internal states based on the
        satisfcation of predicates and the rules of the temporal logic operators
        '''

        for mapping, internal_state in self.predicate_states:
            current_predicate = self.temporal_predicates[internal_state]
            pred_type = self._type(current_predicate)
            
            print(f"Running a predicate of type '{pred_type}' with mapping: {mapping}")

            if pred_type == "once":
                # Evaluate the predicate on the current state. By construction, "mapping" will 
                # include the assignment for all arguments needed by this particular predicate
                pred_eval = self.evaluate_predicate(current_predicate["once_pred"]["pred"], traj_state, mapping)
                print("Overall evaluation:", pred_eval)



    def evaluate_predicate(self, predicate, state, mapping):
        '''
        Given a predicate, a trajectory state, and an assignment of each of the predicate's
        arguments to specific objects in the state, returns the evaluation of the predicate
        '''
        for key in predicate:
            if key == "pred_name":
                # Obtain the functional representation of the base predicate
                predicate_fn = PREDICATE_LIBRARY[predicate["pred_name"]]

                # Map the variables names to the object names, and extract them from the state
                predicate_args = [state["objects"][mapping[var]] for var in self._extract_variables(predicate)]
                
                # Evaluate the predicate
                evaluation = predicate_fn(state, *predicate_args)

                # print(f"Evaluation of predicate '{predicate['pred_name']}' with args = {predicate_args}: {evaluation}")
                return evaluation

            elif key == "not_args":
                return not self.evaluate_predicate(predicate[key]["pred"], state, mapping)

            elif key == "and_args":
                return all([self.evaluate_predicate(sub["pred"], state, mapping) for sub in predicate[key]])

            elif key == "or_args":
                return any([self.evaluate_predicate(sub["pred"], state, mapping) for sub in predicate[key]])


if __name__ == "__main__":
    grammar_path= "../dsl/dsl.ebnf"
    grammar = open(grammar_path).read()
    grammar_parser = tatsu.compile(grammar)

    test_game = """
    (define (game 61267978e96853d3b974ca53-23) (:domain few-objects-room-v1)

    (:constraints (and 
        (preference throwBallToBin
            (exists (?d - ball ?h - bin)
                (then
                    (once (and (agent_holds ?d) (in_motion ?d)))
                    (once (or (in ?d ?h) (and (in_motion ?d) (not (agent_holds ?d))) (agent_crouches)))
                    (once (or (agent_holds ?d) (and (in_motion ?h) (< (distance agent ?d) 5)) ))
                    (hold-while (and (not (agent_holds ?d)) (in_motion ?d)) (in_motion ?h) (agent_crouches))
                    (once-measure (and (not (in_motion ?d)) (in ?h ?d)) (color ?h))
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

    DUMMY_STATE = {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 1, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [4, 4, 0],
                                                    "velocity": [1, 0, 0], "objectType": "ball",
                                                    "color": "red"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [15, 10, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

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
    handler1.process(DUMMY_STATE)