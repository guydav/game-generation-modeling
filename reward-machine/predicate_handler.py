import itertools
import tatsu
import tatsu.ast
import typing

from utils import extract_variable_type_mapping, extract_variables


from config import OBJECTS_BY_TYPE, NAMED_OBJECTS, PREDICATE_LIBRARY, FUNCTION_LIBRARY, SAMPLE_TRAJECTORY


class PredicateHandler:
    evaluation_cache: typing.Dict[str, bool]

    def __init__(self): 
        # The cache from a string representation of the state X predicate X mapping to
        # the predicate's truth value in that state given the mapping.
        self.evaluation_cache = {}
    
    def _new_game(self):
        """
        Call when a new game is started to clear the cache.
        """
        self.evaluation_cache = {}

    def _cache_key(self,  predicate: typing.Optional[tatsu.ast.AST], state: typing.Dict[str, typing.Any], mapping: typing.Dict[str, str]) -> str:
        """
        Map from the arguments to __call__ to the key that represents them in the cache. 
        """
        raise NotImplementedError()
    
    def __call__(self, predicate: typing.Optional[tatsu.ast.AST], state: typing.Dict[str, typing.Any], mapping: typing.Dict[str, str]) -> bool:
        """
        The external API to the predicate handler.
        For now, implements the same logic as before, to make sure I separated it correctly from the `preference_handler`.
        After that, this will implement the caching logic.
        """
        # TODO: cache
        return self._inner_evaluate_predicate(predicate, state, mapping)


    def _inner_evaluate_predicate(self, predicate: typing.Optional[tatsu.ast.AST], state: typing.Dict[str, typing.Any], mapping: typing.Dict[str, str]) -> bool:
        '''
        Given a predicate, a trajectory state, and an assignment of each of the predicate's
        arguments to specific objects in the state, returns the evaluation of the predicate

        GD 2022-09-14: The data, as currently stored, saves delta updates of the state. 
        This means that the truth value of a predicate with a particular assignment holds unitl
        there's information that merits updating it. This means that we should cache predicate
        evaluation results, update them when they return a non-None value, and return the cached result. 
        '''

        if predicate is None:
            return True

        predicate_rule = predicate["parseinfo"].rule  # type: ignore

        if predicate_rule == "predicate":
            # Obtain the functional representation of the base predicate
            predicate_fn = PREDICATE_LIBRARY[predicate["pred_name"]]  # type: ignore

            # Map the variables names to the object names, and extract them from the state
            predicate_args = [state["objects"][mapping[var]] for var in extract_variables(predicate)]
            
            # Evaluate the predicate
            evaluation = predicate_fn(state, *predicate_args)

            return evaluation

        elif predicate_rule == "super_predicate":
            return self._inner_evaluate_predicate(predicate["pred"], state, mapping)

        elif predicate_rule == "super_predicate_not":
            return not self._inner_evaluate_predicate(predicate["not_args"], state, mapping)

        elif predicate_rule == "super_predicate_and":
            return all([self._inner_evaluate_predicate(sub, state, mapping) for sub in predicate["and_args"]])  # type: ignore

        elif predicate_rule == "super_predicate_or":
            return any([self._inner_evaluate_predicate(sub, state, mapping) for sub in predicate["or_args"]])  # type: ignore

        elif predicate_rule == "super_predicate_exists":
            variable_type_mapping = self._extract_variable_type_mapping(predicate["exists_vars"]["variables"])  # type: ignore
            object_assignments = list(itertools.product(*[sum([OBJECTS_BY_TYPE[var_type] for var_type in var_types], []) 
                                      for var_types in variable_type_mapping.values()]))

            sub_mappings = [dict(zip(variable_type_mapping.keys(), object_assignment)) for object_assignment in object_assignments]
            return any([self._inner_evaluate_predicate(predicate["exists_args"], state, {**sub_mapping, **mapping}) for 
                        sub_mapping in sub_mappings])

        elif predicate_rule == "super_predicate_forall":
            variable_type_mapping = self._extract_variable_type_mapping(predicate["forall_vars"]["variables"])  # type: ignore
            object_assignments = list(itertools.product(*[sum([OBJECTS_BY_TYPE[var_type] for var_type in var_types], []) 
                                      for var_types in variable_type_mapping.values()]))

            sub_mappings = [dict(zip(variable_type_mapping.keys(), object_assignment)) for object_assignment in object_assignments]
            return all([self._inner_evaluate_predicate(predicate["forall_args"], state, {**sub_mapping, **mapping}) for 
                        sub_mapping in sub_mappings])

        elif predicate_rule == "function_comparison":
            comp = typing.cast(tatsu.ast.AST, predicate["comp"])
            comparison_operator = comp["comp_op"]      

            # TODO: comparison arguments can be predicate evaluations, and not just function evals and ints

            # TODO: handle cases where the two arguments of '=' are variables, in which case we're checking
            #       variable equivalence instead of numerical equivalance

            # For each comparison argument, evaluate it if it's a function or convert to an int if not
            comp_arg_1 = comp["arg_1"]["arg"]  # type: ignore
            if isinstance(comp_arg_1, tatsu.ast.AST):
                func_name = str(comp_arg_1["func_name"])
                function = FUNCTION_LIBRARY[func_name]
                function_args = [state["objects"][mapping[var]] for var in extract_variables(comp_arg_1)]

                comp_arg_1 = float(function(*function_args))

            else:
                comp_arg_1 = float(comp_arg_1)

            comp_arg_2 = comp["arg_2"]["arg"]  # type: ignore
            if isinstance(comp_arg_1, tatsu.ast.AST):
                function = FUNCTION_LIBRARY[comp_arg_2["func_name"]]
                function_args = [state["objects"][mapping[var]] for var in extract_variables(comp_arg_2)]

                comp_arg_2 = float(function(*function_args))

            else:
                comp_arg_2 = float(comp_arg_2)

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
                raise ValueError(f"Error: Unknown comparison operator '{comparison_operator}'")

        else:
            raise ValueError(f"Error: Unknown rule '{predicate_rule}'")

