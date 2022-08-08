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
        assert isinstance(preference, tatsu.ast.AST) and preference["parseinfo"].rule == "preference"

        self.preference_name = preference["pref_name"]
        body = preference["pref_body"]["body"]

        # BIG TODO: currently we are only handling (exists) quantifications at the top level, but we need to
        # be able to do (forall) as well.
        #
        # I think the way to do this is to have each PreferenceHandler() optionally expect an "initial_mapping"
        # argument. In the case of a (forall ?d - dodgeball) around a preference, we would construct a different
        # PreferenceHandler() object for each dodgeball, and pass each of them a variant of {"?d": "dodgeball-1"}
        # This partial mapping gets added to all other partial mappings monitored by the PreferenceHandler

        # Extract the mapping of variable names to types (e.g. {?d : dodgeball})
        self.variable_type_mapping = self._extract_variable_type_mapping(body["exists_vars"]["variables"])
        self.variable_type_mapping["agent"] = ["agent"]

        # Extract the ordered list of temporal predicates
        self.temporal_predicates = [func["seq_func"] for func in body["exists_args"]["body"]["then_funcs"]]

        # A list of tuples, containing the state of the preference evaluated on partial sets of arguments.
        # Each tuple includes a partial mapping from variables to specific objects, the current predicate,
        # the next predicate, and the number of while conditions satisfied at the current state, which is
        # only important for hold-while predicates.
        #
        # State info: (current_predicate: None or ast.AST, next_predicate: None or ast.AST, while_sat: int,
        #              start: int)
        #
        # TODO: should add data for the first and last state number used in satisfying the preference, to allow
        # for computing things like count-nonoverlapping
        #
        # EXAMPLE:
        #      [({?d : blue-dodgeball-1}, None, _once, 0, -1),
        #       ({?d : red-dodgeball-1}, None, _once, 0, -1),
        #       ({?d : pink-dodgeball-1, ?w: left-wall}, _once, _hold_while, 0, 2),
        #       ({?d : pink-dodgeball-1, ?w: right-wall}, _once", _hold_while, 0, 2)
        #      ]
        self.partial_preference_satisfactions = []

        initial_variables = self._extract_variables(self.temporal_predicates[0])
        initial_var_types = [self.variable_type_mapping[var] for var in initial_variables]
        object_assignments = list(itertools.product(*[sum([OBJECTS_BY_TYPE[var_type] for var_type in var_types], []) 
                                  for var_types in initial_var_types]))

        for object_assignment in object_assignments:
            mapping = dict(zip(initial_variables, object_assignment))
            self.partial_preference_satisfactions.append((mapping, None, self.temporal_predicates[0], 0, -1))

        # A list of all versions of the predicate satisfied at a particular step, updated during self.process()
        self.satisfied_this_step = []

        self.cur_step = 1

    def _extract_variable_type_mapping(self, variable_list):
        '''
        Given a list of variables, extract the mapping from variable names to variable types. Variable types are
        stored in lists, even in cases where there is only one possible for the variable in order to handle cases
        where multiple types are linked together with an (either) clause
        '''
        if isinstance(variable_list, tatsu.ast.AST):
            variable_list = [variable_list]

        variables = {}
        for var_info in variable_list:
            if isinstance(var_info["var_type"]["type"], tatsu.ast.AST):
                variables[var_info["var_names"]] = var_info["var_type"]["type"]["type_names"]
            else:
                variables[var_info["var_names"]] = [var_info["var_type"]["type"]]

        return variables

    def _extract_variables(self, predicate):
        '''
        Recursively extract every variable referenced in the predicate (including inside functions 
        used within the predicate)

        BIG TODO: some objects (like 'desk') are just referred to directly inside of predicates, and
        are never quantified over (i.e. we never see (exists ?d - desk)). We need to be able to detect
        and handle these kinds of variables
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

                # We don't want to capture any variables within an (exists) or (forall) that's inside 
                # the preference, since those are not globally required -- see evaluate_predicate()
                elif key == "exists_args":
                    continue

                elif key == "forall_args":
                    continue

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

    def advance_preference(self, mapping, current_predicate, next_predicate, start, new_partial_preference_satisfactions):
        '''
        Called when a predicate inside a (then) operator has been fully satisfied and we are moving to the
        next predicate. This function adds new partial object mappings and predicates to the provided list
        called "new_partial_preference_satisfactions"

        TODO: should only add non-duplicates to new_partial_preference_satisfactions (a duplicate is possible in the
        case where the initial mapping was split, before one of the later branches was reverted back to the
        initial state. If the first predicate is satisfied again, the initial mapping will again split and
        we need to make sure not to add duplicate branches back in)
        '''

        next_pred_idx = self.temporal_predicates.index(next_predicate)
        new_cur_predicate = next_predicate

        # Check to see whether we've just satisfied the last predicate of the (then) operator, in which case
        # the entire preference has been satisfied! Add the current mapping to the list satisfied at this step
        # and add the reverted mapping back to new_partial_preference_satisfactions
        if next_pred_idx+1 == len(self.temporal_predicates):
            print("\n\tPREFERENCE SATISFIED!")
            self.satisfied_this_step.append((mapping, start, self.cur_step))
            self.revert_preference(mapping, new_partial_preference_satisfactions)
            return

        else:
            new_next_predicate = self.temporal_predicates[next_pred_idx+1]

            # Determine all of the variables referenced in the new predicate that aren't referenced already
            new_variables = [var for var in self._extract_variables(new_next_predicate) if var not in mapping]

        print("\n\tNew variables required by the next predicate:", new_variables)

        # If there are new variables, then we iterate overall all possible assignments for them, add them to the
        # existing mapping, and add it to our list of partial preference satisfactions while advancing the predicates
        if len(new_variables) > 0:
            new_var_types = [self.variable_type_mapping[var] for var in new_variables]
            object_assignments = list(itertools.product(*[sum([OBJECTS_BY_TYPE[var_type] for var_type in var_types], []) 
                                  for var_types in new_var_types]))

            for object_assignment in object_assignments:
                new_mapping = dict(zip(new_variables, object_assignment))
                new_mapping.update(mapping)

                new_partial_preference_satisfactions.append((new_mapping, new_cur_predicate, new_next_predicate, 0, start))

        # Otherwise, just advance the predicates but keep the mapping the same
        else:
            new_partial_preference_satisfactions.append((mapping, new_cur_predicate, new_next_predicate, 0, start))

    def revert_preference(self, mapping, new_partial_preference_satisfactions):
        '''
        Called when a predicate inside a (then) operator is no longer satisfied and we have to return to
        the start state. This function will add at most one tuple to new_partial_preference_satisfactions that
        represents the "initial component" of the current mapping: the portion of the mapping that consists
        of variables required by the first predicate. Importantly, a tuple is only added if it is a non-duplicate
        '''
        initial_variables = self._extract_variables(self.temporal_predicates[0])
        new_mapping = {key: val for key, val in mapping.items() if key in initial_variables}
        
        new_partial_preference_satisfactions.append((new_mapping, None, self.temporal_predicates[0], 0, -1))

    def process(self, traj_state):
        '''
        Take a state from an active trajectory and update each of the internal states based on the
        satisfcation of predicates and the rules of the temporal logic operators

        BIG TODO: handle once-measure!
        '''

        self.satisfied_this_step = []

        new_partial_preference_satisfactions = []

        for mapping, current_predicate, next_predicate, while_sat, start in self.partial_preference_satisfactions:
            cur_predicate_type = None if current_predicate is None else self._type(current_predicate)
            next_predicate_type = None if next_predicate is None else self._type(next_predicate)

            print(f"\nEvaluating a new partial satisfaction for {self.preference_name}")
            print("\tMapping:", mapping)
            print("\tCurrent predicate type:", cur_predicate_type)
            print("\tNext predicate type:", next_predicate_type)
            print("\tPreference satisfcation start:", start)
            print("\tWhile-conditions satisfied:", while_sat)

            # The "Start" state: transition forward if the basic condition of the next predicate is met
            if cur_predicate_type is None:
                if next_predicate_type == "once":
                    pred_eval = self.evaluate_predicate(next_predicate["once_pred"], traj_state, mapping)
                elif next_predicate_type in ["hold", "hold-while"]:
                    pred_eval = self.evaluate_predicate(next_predicate["hold_pred"], traj_state, mapping)

                print("\n\tEvaluation of next predicate:", pred_eval)

                # If the basic condition of the next predicate is met, we'll advance the predicates through the (then) operator.
                # We also record the current step as the "start" of the predicate being satisfied
                if pred_eval:
                    self.advance_preference(mapping, current_predicate, next_predicate, self.cur_step, new_partial_preference_satisfactions)

                # If not, then just add the same predicates back to the list
                else:
                    new_partial_preference_satisfactions.append((mapping, current_predicate, next_predicate, 0, start))


            elif cur_predicate_type == "once":
                if next_predicate_type == "once":
                    next_pred_eval = self.evaluate_predicate(next_predicate["once_pred"], traj_state, mapping)
                elif next_predicate_type in ["hold", "hold-while"]:
                    next_pred_eval = self.evaluate_predicate(next_predicate["hold_pred"], traj_state, mapping)

                cur_pred_eval = self.evaluate_predicate(current_predicate["once_pred"], traj_state, mapping)

                print("\n\tEvaluation of next predicate:", next_pred_eval)
                print("\tEvaluation of current predicate:", cur_pred_eval)

                # If the next predicate is satisfied, then we advance regardless of the state of the current predicate
                if next_pred_eval:
                    self.advance_preference(mapping, current_predicate, next_predicate, start, new_partial_preference_satisfactions)

                # If the next predicate *isn't* satisfied, but the current one *is* then we stay in our current state 
                elif cur_pred_eval:
                    new_partial_preference_satisfactions.append((mapping, current_predicate, next_predicate, 0, start))

                # If neither are satisfied, we return to the start
                else:
                    self.revert_preference(mapping, new_partial_preference_satisfactions)

            elif cur_predicate_type == "hold":
                if next_predicate_type == "once":
                    next_pred_eval = self.evaluate_predicate(next_predicate["once_pred"], traj_state, mapping)
                elif next_predicate_type in ["hold", "hold-while"]:
                    next_pred_eval = self.evaluate_predicate(next_predicate["hold_pred"], traj_state, mapping)

                cur_pred_eval = self.evaluate_predicate(current_predicate["hold_pred"], traj_state, mapping)

                print("\n\tEvaluation of next predicate:", next_pred_eval)
                print("\tEvaluation of current predicate:", cur_pred_eval)

                # If the next predicate is satisfied, then we advance regardless of the state of the current predicate
                if next_pred_eval:
                    self.advance_preference(mapping, current_predicate, next_predicate, start, new_partial_preference_satisfactions)

                # If the next predicate *isn't* satisfied, but the current one *is* then we stay in our current state 
                elif cur_pred_eval:
                    new_partial_preference_satisfactions.append((mapping, current_predicate, next_predicate, 0, start))

                # If neither are satisfied, we return to the start
                else:
                    self.revert_preference(mapping, new_partial_preference_satisfactions)

            elif cur_predicate_type == "hold-while":
                num_while_conditions = 1 if isinstance(current_predicate["while_preds"], tatsu.ast.AST) else len(current_predicate["while_preds"])

                # If all of the while condition has already been met, then we can treat this exactly like a normal hold
                if while_sat == num_while_conditions:
                    if next_predicate_type == "once":
                        next_pred_eval = self.evaluate_predicate(next_predicate["once_pred"], traj_state, mapping)
                    elif next_predicate_type in ["hold", "hold-while"]:
                        next_pred_eval = self.evaluate_predicate(next_predicate["hold_pred"], traj_state, mapping)

                    cur_pred_eval = self.evaluate_predicate(current_predicate["hold_pred"], traj_state, mapping)

                    print("\n\tEvaluation of next predicate:", next_pred_eval)
                    print("\tEvaluation of current predicate:", cur_pred_eval)

                    # If the next predicate is satisfied, then we advance regardless of the state of the current predicate
                    if next_pred_eval:
                        self.advance_preference(mapping, current_predicate, next_predicate, start, new_partial_preference_satisfactions)

                    # If the next predicate *isn't* satisfied, but the current one *is* then we stay in our current state 
                    elif cur_pred_eval:
                        new_partial_preference_satisfactions.append((mapping, current_predicate, next_predicate, while_sat, start))

                    # If neither are satisfied, we return to the start
                    else:
                        self.revert_preference(mapping, new_partial_preference_satisfactions)

                # If not, then we only care about the while condition and the current hold
                else:
                    cur_pred_eval = self.evaluate_predicate(current_predicate["hold_pred"], traj_state, mapping)

                    # Determine whether the next while condition is satisfied in the current state
                    if num_while_conditions == 1:
                        cur_while_eval = self.evaluate_predicate(current_predicate["while_preds"], traj_state, mapping)
                    else:
                        cur_while_eval = self.evaluate_predicate(current_predicate["while_preds"][while_sat], traj_state, mapping)

                    print("\n\tEvaluation of current predicate:", cur_pred_eval)
                    print("\tEvaluation of current while pred:", cur_while_eval)

                    if cur_pred_eval:
                        if cur_while_eval:
                            new_partial_preference_satisfactions.append((mapping, current_predicate, next_predicate, while_sat + 1, start))

                        else:
                            new_partial_preference_satisfactions.append((mapping, current_predicate, next_predicate, while_sat, start))

                    else:
                        self.revert_preference(mapping, new_partial_preference_satisfactions)

        # Janky way to remove duplicates: group by the concatenation of every value in the mapping, so each
        # specific assignment is represented by a different string.
        # TODO: figure out if this will ever break down
        keyfunc = lambda pref_sat: "_".join(pref_sat[0].values())
        new_partial_preference_satisfactions = [list(g)[0] for k, g in itertools.groupby(sorted(
            new_partial_preference_satisfactions, key=keyfunc), keyfunc)]

        self.partial_preference_satisfactions = new_partial_preference_satisfactions

        self.cur_step += 1

        return self.satisfied_this_step

    def evaluate_predicate(self, predicate, state, mapping):
        '''
        Given a predicate, a trajectory state, and an assignment of each of the predicate's
        arguments to specific objects in the state, returns the evaluation of the predicate
        '''

        predicate_rule = predicate["parseinfo"].rule

        if predicate_rule == "predicate":
            # Obtain the functional representation of the base predicate
            predicate_fn = PREDICATE_LIBRARY[predicate["pred_name"]]

            # Map the variables names to the object names, and extract them from the state
            predicate_args = [state["objects"][mapping[var]] for var in self._extract_variables(predicate)]
            
            # Evaluate the predicate
            evaluation = predicate_fn(state, *predicate_args)

            return evaluation

        elif predicate_rule == "super_predicate":
            return self.evaluate_predicate(predicate["pred"], state, mapping)

        elif predicate_rule == "super_predicate_not":
            return not self.evaluate_predicate(predicate["not_args"], state, mapping)

        elif predicate_rule == "super_predicate_and":
            return all([self.evaluate_predicate(sub, state, mapping) for sub in predicate["and_args"]])

        elif predicate_rule == "super_predicate_or":
            return any([self.evaluate_predicate(sub, state, mapping) for sub in predicate["or_args"]])

        elif predicate_rule == "super_predicate_exists":
            variable_type_mapping = self._extract_variable_type_mapping(predicate["exists_vars"]["variables"])
            object_assignments = list(itertools.product(*[sum([OBJECTS_BY_TYPE[var_type] for var_type in var_types], []) 
                                      for var_types in variable_type_mapping.values()]))

            sub_mappings = [dict(zip(variable_type_mapping.keys(), object_assignment)) for object_assignment in object_assignments]
            return any([self.evaluate_predicate(predicate["exists_args"], state, {**sub_mapping, **mapping}) for 
                        sub_mapping in sub_mappings])

        elif predicate_rule == "super_predicate_forall":
            variable_type_mapping = self._extract_variable_type_mapping(predicate["forall_vars"]["variables"])
            object_assignments = list(itertools.product(*[sum([OBJECTS_BY_TYPE[var_type] for var_type in var_types], []) 
                                      for var_types in variable_type_mapping.values()]))

            sub_mappings = [dict(zip(variable_type_mapping.keys(), object_assignment)) for object_assignment in object_assignments]
            return all([self.evaluate_predicate(predicate["forall_args"], state, {**sub_mapping, **mapping}) for 
                        sub_mapping in sub_mappings])

        elif predicate_rule == "function_comparison":
            comparison_operator = predicate["comp"]["comp_op"]

            # TODO: comparison arguments can be predicate evaluations, and not just function evals and ints

            # For each comparison argument, evaluate it if it's a function or convert to an int if not
            comp_arg_1 = predicate["comp"]["arg_1"]["arg"]
            if isinstance(comp_arg_1, tatsu.ast.AST):
                function = FUNCTION_LIBRARY[comp_arg_1["func_name"]]
                function_args = [state["objects"][mapping[var]] for var in self._extract_variables(comp_arg_1)]

                comp_arg_1 = function(*function_args)

            else:
                comp_arg_1 = int(comp_arg_1)

            comp_arg_2 = predicate["comp"]["arg_2"]["arg"]
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
                exit(f"Error: Unknown comparison operator '{comparison_operator}'")

        else:
            exit(f"Error: Unknown rule '{predicate_rule}'")



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
    handler1 = PreferenceHandler(pref1)
    
    for idx, state in enumerate(SAMPLE_TRAJECTORY):
        print(f"\n\n================================PROCESSING STATE {idx+1}================================")
        handler1.process(state)
