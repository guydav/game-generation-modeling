import os
import sys
import tatsu

# Add src/ to our path so we can import from the scripts in room_and_object_types.py
sys.path.insert(1, os.path.join(sys.path[0], '../src'))
from ast_parser import ASTParser

# This dictionary maps from the names of base-level predicates to their functional representations (i.e.
# a function that computes the predicate on a given state)
PREDICATE_LIBRARY = {"test": lambda x: x}


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

            variables = self._extract_variables(body["exists_vars"]["variables"])

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
                            predicate = self._handle_predicate(function["once_pred"]["pred"])
                            print(predicate(1))
                            exit()

                        # Case B: once-measure
                        elif operators[0] == "once_measure_pred":
                            pass

                        elif operators[0] == "hold_pred":
                            # Case C: hold-while
                            if operators[1] == "while_preds":
                                pass

                            # Case D: hold
                            else:
                                pass

                        else:
                            exit("Unknown operator type!")

                        print(function.keys())
                
                # Case 2: preference does not contain temporal relations
                else:
                    pass

            print(functions)
            exit()

    def _extract_variables(self, variable_list):
        variables = []
        for var_info in variable_list:
            variables.append([var_info["var_names"], var_info["var_type"]["type"]])

        return variables

    def _handle_predicate(self, predicate):
        for key in predicate:
            print("Handling key =", key)
            # Base case: return the functional form of the referenced predicate
            if key == "pred_name":
                def foo(x):
                    return True

                return foo

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

            elif key == "comp":
                comparison_operator = predicate[key]["comp_op"]

                arg_1 = predicate[key]["arg_1"]["arg"]
                if isinstance(arg_1, tatsu.ast.AST):
                    function_name = arg_1["func_name"]
                    function_arguments = [farg["term"]["arg"] for farg in arg_1["func_args"]]

                    # Should return a function which takes in a state and computes the specified
                    # function on the specified arguments
                    # arg_1 = build_function(function_name, function_arguments)
                    def foo(x):
                        return 100

                    arg_1 = foo

                else:
                    arg_copy = arg_1
                    def fixed_1(x):
                        return int(arg_copy)
                    arg_1 = fixed_1


                arg_2 = predicate[key]["arg_2"]["arg"]
                if isinstance(arg_2, tatsu.ast.AST):
                    function_name = arg_2["func_name"]
                    function_arguments = [farg["term"]["arg"] for farg in arg_2["func_args"]]

                    # arg_2 = build_function(function_name, function_arguments)

                    def foo2(x):
                        return 100

                    arg_2 = foo2

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
                    (once (or (agent_holds ?d) (and (in_motion ?h) (< (distance agent ?b) 5)) ))
                    (hold-while (and (not (agent_holds ?d)) (in_motion ?d)) (in_motion ?h))
                    (once-measure (and (not (in_motion ?d)) (in ?h ?d)) (distance agent ?h))
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