import os
import re
import sys
import tatsu

import numpy as np

# Add src/ to our path so we can import from the scripts in room_and_object_types.py
sys.path.insert(1, os.path.join(sys.path[0], '../src'))
from ast_parser import ASTParser

# This dictionary maps from the names of base-level predicates to their functional representations (i.e.
# a function that computes the predicate on a given state)
PREDICATE_LIBRARY = {"in_motion": lambda state, obj: (np.linalg.norm(obj["velocity"]) > 0),
                     "agent_holds": lambda state, obj: (state["types"]["agent"]["holding"] == obj["name"])}


FUNCTION_LIBRARY = {"distance": lambda obj1, obj2: np.linalg.norm(np.array(obj1["position"]) - np.array(obj2["position"]))}


def _not(func):
    def inverted(x):
        return not func(x)
    return inverted

def _and(*args):
    def intersection(x):
        return all([arg(x) for arg in args])
    return intersection

def _or(*args):
    def union(x):
        return any([arg(x) for arg in args])
    return union

def _comparison(arg_1, arg_2, comp_type):
    if comp_type == "=":
        def comp(x):
            return arg_1(x) == arg_2(x)
        return comp

    elif comp_type == ">":
        def comp(x):
            return arg_1(x) > arg_2(x)
        return comp

    elif comp_type == ">=":
        def comp(x):
            return arg_1(x) >= arg_2(x)
        return comp

    elif comp_type == "<":
        def comp(x):
            return arg_1(x) < arg_2(x)
        return comp

    elif comp_type == "<=":
        def comp(x):
            return arg_1(x) <= arg_2(x)
        return comp

def build_predicate(predicate_name, *args):
    '''
    Takes a predicate name and its necessary arguments, for instance "(in_motion ?b)", where
    ?b quantifies a specific beach ball, and returns a function which takes in a simulator state 
    and evaluates the specified predicate on that state.

    In this example, the returned function would evaluate as True for any state in which the beach
    ball is in motion, and False otherwise. 
    '''

    if predicate_name in PREDICATE_LIBRARY:
        print(f"Building '{predicate_name}' with arguments: {args}")
        def _predicate(state):
            arguments = [state["types"][arg] for arg in args]
            return PREDICATE_LIBRARY[predicate_name](state, *arguments)

        return _predicate

    else:
        print(f"Predicate '{predicate_name}' has not been defined, returning dummy predicate")
        def dummy(state):
            return True

        return dummy

def build_function(function_name, *args):
    '''
    The mirror of build_predicate (above), takes a function name and its arguments and returns
    a new function which takes in a simulator state and returns the function evaluated on that
    state
    '''

    if function_name in FUNCTION_LIBRARY:
        def _function(state):
            arguments = [state["types"][arg] for arg in args]
            return FUNCTION_LIBRARY[function_name](*arguments)

        return _function

    else:
        def dummy(state):
            return 1

        return dummy

DUMMY_STATE = {"types": {"agent": {"position": [0, 0, 0],
                                   "velocity": [0, 0, 0],
                                   "crouching": False,
                                   "holding": None},

                         "?d": {"position": [4, 0, 0],
                                "velocity": [0, 0, 0],
                                "name": "beachball"},


                         "?h": {"position": [10, 10, 0],
                                "velocity": [1.0, 0, 0],
                                "name": "bin"}

                        }

              }

DUMMY_STATE = {"objects": [{"name": "Ball_12345", "position": [4, 0, 0], "velocity": [0, 0, 0],
                            "objectId": "Ball|+04.00|+00.00|+00.00", "objectType": "ball"},

                            {"name": "Bin_54321", "position": [10, 10, 0], "velocity": [0, 0, 0],
                            "objectId": "Bin|+10.00|+10.00|+00.00", "objectType": "bin"}
                          ],

               "agent": {"position": [0, 0, 0], "velocity": [0, 0, 0], "croucing": False,
                         "holding": None},

               "game_start": True,
               "game_over": False

              }

ALL_OBJECTS = []

class PreferenceParserV2(ASTParser):
    def _handle_ast(self, ast, **kwargs):
        for key in ast:
            if key == "definition":
                # Identify the preference's name and variable -> type mapping
                # Store those values in "current_pref" and "current_variables"
                # Continue recursively
                pass

            elif key == "once_pred":
                # Start a new "once" predicate inside the current preference
                # Continue recursively
                pass

            elif key == "once_measure_pred":
                pass

            # Continue elif's for measurement, hold_pred, while_pred

            elif key == "term":
                # Add the variable to the list of variables in the current preference
                pass

class PreferenceParser(ASTParser):

    def _handle_ast(self, ast, **kwargs):
        for key in ast:
            if key == "preferences":
                self._handle_preferences(ast[key], **kwargs)

            elif key != 'parseinfo':
                self(ast[key], **kwargs)

    def _handle_preferences(self, ast, **kwargs):
        for preference in ast:
            name = preference["definition"]["pref_name"]
            body = preference["definition"]["pref_body"]["body"]

            print(f"\nBuilding the preference '{name}'")

            variables = self._extract_variables(body["exists_vars"]["variables"])
            print("Variables:", variables)

            arguments = body["exists_args"]["body"]
            for key in arguments:

                # Case 1: preference contains temporal relations
                if key == "then_funcs":
                    functions = arguments["then_funcs"]
                    for function in functions:
                        function = function["seq_func"]

                        operators = list(function.keys())
                       
                        # Case A: once
                        if operators[0] == "once_pred":
                            arguments = self._extract_arguments(function["once_pred"]["pred"])
                            print("Args:", arguments)
                            predicate = self._handle_predicate(function["once_pred"]["pred"])

                        # Case B: once-measure
                        elif operators[0] == "once_measure_pred":
                            arguments = self._extract_arguments(function["once_measure_pred"]["pred"])
                            print("Args:", arguments)
                            predicate = self._handle_predicate(function["once_measure_pred"]["pred"])

                            function_name = function["measurement"]["func_name"]

                            if isinstance(function["measurement"]["func_args"], tatsu.ast.AST):
                                function_arguments = [function["measurement"]["func_args"]["term"]["arg"]]
                            else:
                                function_arguments = [farg["term"]["arg"] for farg in function["measurement"]["func_args"]]

                            measurement = build_function(function_name, *function_arguments)                            

                        elif operators[0] == "hold_pred":
                            arguments = self._extract_arguments(function["hold_pred"]["pred"])
                            print("Args:", arguments)

                            # Case C: hold-while
                            if operators[1] == "while_preds":
                                hold_predicate = self._handle_predicate(function["hold_pred"]["pred"])

                                # Only one while predicate
                                if isinstance(function["while_preds"], tatsu.ast.AST):
                                    while_preds = [self._handle_predicate(function["while_preds"]["pred"])]

                                # More than one while predicate
                                else:
                                    while_preds = [self._handle_predicate(pred["pred"]) for pred in function["while_preds"]]

                            # Case D: hold
                            else:
                                hold_predicate = self._handle_predicate(function["hold_pred"]["pred"])

                        else:
                            exit("Unknown operator type!")

                
                # Case 2: preference does not contain temporal relations
                else:
                    pass


    def _extract_variables(self, variable_list):
        if isinstance(variable_list, tatsu.ast.AST):
            variable_list = [variable_list]

        variables = {}
        for var_info in variable_list:
            variables[var_info["var_names"]] = var_info["var_type"]["type"]

        return variables

    def _extract_arguments(self, predicate):
        '''
        Recursive extract every variable referenced in the predicate (including inside functions 
        used within the predicate)
        '''

        if isinstance(predicate, list) or isinstance(predicate, tuple):
            pred_args = []
            for sub_predicate in predicate:
                pred_args += self._extract_arguments(sub_predicate)

            return list(set(pred_args))

        elif isinstance(predicate, tatsu.ast.AST):
            pred_args = []
            for key in predicate:
                if key == "term":

                    # Different structure for predicate args vs. function args
                    if isinstance(predicate["term"], tatsu.ast.AST):
                        pred_args += [predicate["term"]["arg"]]
                    else:
                        pred_args += [predicate["term"]]

                elif key != "parseinfo":
                    pred_args += self._extract_arguments(predicate[key])

            return list(set(pred_args))

        else:
            return []


    def _handle_predicate(self, predicate):
        for key in predicate:

            # Base case: return the functional form of the referenced predicate
            if key == "pred_name":
                pred_name = predicate["pred_name"]

                # Case: no arguments
                if predicate["pred_args"] is None:
                    pred_args = []

                # Case: exactly one argument
                elif isinstance(predicate["pred_args"], tatsu.ast.AST):
                    pred_args = [predicate["pred_args"]["term"]]

                # Case: more than one argument
                else:
                    pred_args = [parg["term"] for parg in predicate["pred_args"]]

                sub_predicate = build_predicate(pred_name, *pred_args)

                return sub_predicate

            elif key == "and_args":
                sub_predicates = []
                for sub_predicate in predicate[key]:
                    sub_predicates.append(self._handle_predicate(sub_predicate["pred"]))

                return _and(*sub_predicates)

            elif key == "or_args":
                sub_predicates = []
                for sub_predicate in predicate[key]:
                    sub_predicates.append(self._handle_predicate(sub_predicate["pred"]))

                return _or(*sub_predicates)

            elif key == "not_args":
                return _not(self._handle_predicate(predicate[key]["pred"]))

            elif key == "comp":
                comparison_operator = predicate[key]["comp_op"]

                arg_1 = predicate[key]["arg_1"]["arg"]
                if isinstance(arg_1, tatsu.ast.AST):
                    function_name = arg_1["func_name"]

                    if isinstance(arg_1["func_args"], tatsu.ast.AST):
                        function_arguments = [arg_1["func_args"]["term"]["arg"]]
                    else:
                        function_arguments = [farg["term"]["arg"] for farg in arg_1["func_args"]]

                    arg_1 = build_function(function_name, *function_arguments)

                else:
                    arg_copy = arg_1
                    def fixed_1(x):
                        return int(arg_copy)
                    arg_1 = fixed_1


                arg_2 = predicate[key]["arg_2"]["arg"]
                if isinstance(arg_2, tatsu.ast.AST):
                    function_name = arg_2["func_name"]
                    
                    if isinstance(arg_2["func_args"], tatsu.ast.AST):
                        function_arguments = [arg_2["func_args"]["term"]["arg"]]
                    else:
                        function_arguments = [farg["term"]["arg"] for farg in arg_2["func_args"]]

                    arg_2 = build_function(function_name, *function_arguments)

                else:
                    arg_copy = arg_2
                    def fixed_2(x):
                        return int(arg_copy)

                    arg_2 = fixed_2
             
                return _comparison(arg_1, arg_2, comparison_operator)
                


class StateMachine():
    def __init__(self):
        pass

def parse_preference(preference):
    # Determine the preference's name
    # Determine the objects that the preference quantifies over

    # Case 1: preference contains temporal relations (i.e. contains a (then ... ) statement)

    # Case 2: preference does not contain temporal relations (only (at-end ... ) statements)

    pass

if __name__ == "__main__":
    grammar_path= "../dsl/dsl.ebnf"
    grammar = open(grammar_path).read()
    grammar_parser = tatsu.compile(grammar)

    test_game = """
    (define (game 61267978e96853d3b974ca53-23) (:domain few-objects-room-v1)

    (:constraints (and 
        (preference throwBallToBin
            (exists (?d - dodgeball ?h - hexagonal_bin)
                (then
                    (once (or (agent_holds ?d) (and (in_motion ?h) (< (distance agent ?d) 5)) ))
                    (hold-while (and (not (agent_holds ?d)) (in_motion ?d)) (in_motion ?h) (agent_crouches))
                    (once-measure (and (not (in_motion ?d)) (in ?h ?d)) (color ?h))
                )
            )
        )
        (preference throwAttempt
            (exists (?d - dodgeball)
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

    # (hold (and (not (agent_holds ?d)) (in_motion ?d))) 

    ast = grammar_parser.parse(test_game)
    
    parser = PreferenceParser()
    parser(ast)


    # TODOs
    # 1. refactor parser to better handle cases where we have one / many arguments (AST vs. iterable)
    # 2. implement temporal relations



    # We proceed iteratively through each of the predicates inside the (then) operator.
    # We begin the [PRE] state
    #
    # We look at the first predicate, and determine all of the arguments it specifies. For instance,
    # the first predicate might use the variable ?b. We can tell from the preference definition that
    # ?b refers to an object of type "ball". So what we do is create a copy of the predicate for each
    # object in the scene that matches the type ball. This could be a pink dodgeball and a blue dodge-
    # ball, for instance.
    #
    # Now we proceed through the trajectory, evaluating *each* of the copies of the predicate on every
    # state. As long as *all* of the predicates evaluate to False, we continue. As soon as one of the
    # predicates is satisfied, we create an instance of the StateMachine and put it in [STATE 1], and
    # lock in whichever object was used to satisfy the preference. For instance, when the blue dodgeball
    # is picked up, we create a StateMachine and force ?b to refer to the *blue* dodgeball for that machine.
    # The other instances of the first predicate continue tracking their respective objects.
    #
    # When we create the StateMachine, we then examine the next predicate inside the (then) operator. Say
    # that this predicate references both ?b and ?w. In this example, we've already locked in ?b to refer
    # to the blue dodgeball. However, ?w remains unspecified. We can again tell from the preference definition
    # that ?w is an object of type "wall". So we create copies of each of the predicates (as we did before) for
    # each of the objects that match the type wall in the current scene. The StateMachine continues to evaluate
    # each of the states in the trajectory. Depending on the type of the current predicate (i.e. once, hold, or
    # hold-while), the StateMachine is also responsible for handling the transition logic. For instance, in the
    # case of a hold-while, the StateMachine should wait until the hold condition is no longer satisfied. During
    # this time, it should also monitor whether the while condition has been met. When the hold is no longer
    # satisfied, the StateMachine should either proceed to the next predicate (if the while condition was met)
    # or transition to a failure state and, probably, destroy itself (if the while condition wasn't met).
    #
    # There are a couple of edge cases we need to consider:
    # 1. What happens when a new predicate involves more than one previously uninstantiated variables?
    #    It might be necessary to create a new instance of the predicate for every possible combination
    #    of those objects, but this has the potential to get very expensive if many objects are quantified
    #    and they each have many different instances in the scene.
    #
    # 2. When one of the "interior predicates" (i.e. not the first or last) gets satisfied, what happens to
    #    the other copies? In our example, once the dodgeball has bounced off of one of the walls, how do we
    #    indicate that the other predicates (those monitoring other walls) should be shut off? Perhaps one
    #    option is to include a flag which indicates whether a predicate is "exclusive" or "non-exclusive"