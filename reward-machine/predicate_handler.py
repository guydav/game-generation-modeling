import pathlib
import sys
sys.path.append((pathlib.Path(__file__).parents[1].resolve() / 'src').as_posix())

import itertools
import numpy as np
import tatsu
import tatsu.ast
import typing

import ast_printer

from utils import extract_variable_type_mapping, extract_variables

from config import OBJECTS_BY_TYPE, NAMED_OBJECTS, SAMPLE_TRAJECTORY


AgentState = typing.NewType('AgentState', typing.Dict[str, typing.Any])
ObjectState = typing.NewType('ObjectState', typing.Dict[str, typing.Any])
FullState = typing.NewType('StateType', typing.Dict[str, typing.Union[bool, int, AgentState, typing.Sequence[ObjectState]]])


AGENT_STATE_KEY = 'agentState'
AGENT_STATE_CHANGED_KEY = 'agentStateChanged'
N_OBJECTS_CHANGED_KEY = 'nObjectsChanged'
OBJECTS_KEY = 'objects'
ACTION_CHANGED_KEY = 'actionChanged'
ACTION_KEY = 'action'

ORIGINAL_INDEX_KEY = 'originalIndex'
OBJECT_ID_KEY = 'objectId'


class PredicateHandler:
    # Which field in the state to use as the index
    index_key: str
    # Which field in each object to use as an id
    object_id_key: str 
    # The cache from a string representation of the state X predicate X mapping to
    # the predicate's truth value in that state given the mapping.
    evaluation_cache: typing.Dict[str, bool]
    # The last state the evaluation cache was updated for a given key.
    evaluation_cache_last_updated: typing.Dict[str, int]
    # A cache of the latest observed value for each object
    state_cache: typing.Dict[str, typing.Union[ObjectState, AgentState]]
    # The last state the state cache was updated for
    state_cache_last_updated: int

    def __init__(self, index_key: str = ORIGINAL_INDEX_KEY, object_id_key: str = OBJECT_ID_KEY):
        self.index_key = index_key
        self.object_id_key = object_id_key

        self.evaluation_cache = {}
        self.evaluation_cache_last_updated = {}
        self.state_cache = {}
        self.state_cache_last_updated = -1
    
    def _new_game(self):
        """
        Call when a new game is started to clear the cache.
        """
        self.evaluation_cache = {}
        self.evaluation_cache_last_updated = {}
        self.state_cache = {}
        self.state_cache_last_updated = -1

    def _cache_key(self,  predicate: typing.Optional[tatsu.ast.AST], mapping: typing.Dict[str, str]) -> str:
        """
        Map from the arguments to __call__ to the key that represents them in the cache. 
        """
        ast_printer.reset_buffers()
        ast_printer.PARSE_DICT[ast_printer.PREFERENCES_KEY](predicate)
        # flush the line buffer
        ast_printer._indent_print('', 0, ast_printer.DEFAULT_INCREMENT, None)
        predicate_str = ' '.join(ast_printer.BUFFER if ast_printer.BUFFER is not None else [])
        mapping_str = ' '.join([f'{k}={mapping[k]}' for k in sorted(mapping.keys())])
        return f'{predicate_str}_{mapping_str}'
    
    def __call__(self, predicate: typing.Optional[tatsu.ast.AST], state: typing.Dict[str, typing.Any], mapping: typing.Dict[str, str]) -> typing.Optional[bool]:
        """
        The external API to the predicate handler.
        For now, implements the same logic as before, to make sure I separated it correctly from the `preference_handler`.
        After that, this will implement the caching logic.

        GD 2022-09-14: The data, as currently stored, saves delta updates of the state. 
        This means that the truth value of a predicate with a particular assignment holds unitl
        there's information that merits updating it. This means that we should cache predicate
        evaluation results, update them when they return a non-None value, and return the cached result. 
        """
        predicate_key = self._cache_key(predicate, mapping)
        state_index = state[self.index_key]

        if predicate_key in self.evaluation_cache_last_updated and self.evaluation_cache_last_updated[predicate_key] == state_index:
            return self.evaluation_cache[predicate_key]

        if state_index > self.state_cache_last_updated:
            self.state_cache_last_updated = state_index
            for obj in state[OBJECTS_KEY]:
                self.state_cache[obj[self.object_id_key]] = obj

            if state[AGENT_STATE_CHANGED_KEY]:
                self.state_cache[AGENT_STATE_KEY] = state[AGENT_STATE_KEY]
                self.state_cache['agent'] = state[AGENT_STATE_KEY]

        current_state_value =  self._inner_evaluate_predicate(predicate, state, mapping)
        if current_state_value is not None:
            self.evaluation_cache[predicate_key] = current_state_value
            self.evaluation_cache_last_updated[predicate_key] = state_index
        
        return self.evaluation_cache[predicate_key] if predicate_key in self.evaluation_cache else current_state_value

    def _inner_evaluate_predicate(self, predicate: typing.Optional[tatsu.ast.AST], state: typing.Dict[str, typing.Any], mapping: typing.Dict[str, str]) -> typing.Optional[bool]:
        '''
        Given a predicate, a trajectory state, and an assignment of each of the predicate's
        arguments to specific objects in the state, returns the evaluation of the predicate

        GD: 2022-09-14: The logic in `__call__` relies on the assumption that if the predicate's
        value was not updated at this timestep, this function returns None, rather than False. 
        This is to know when the cache should be updated. 

        I should figure out how to implement this in a manner that's reasonable across all predicates,
        or maybe it's something the individual predicate handlers should do? Probably the latter.
        '''

        if predicate is None:
            return None

        predicate_rule = predicate["parseinfo"].rule  # type: ignore

        if predicate_rule == "predicate":
            # Obtain the functional representation of the base predicate
            predicate_fn = PREDICATE_LIBRARY[predicate["pred_name"]]  # type: ignore

            # Extract only the variables in the mapping relevant to this predicate
            relevant_mapping = {var: mapping[var] for var in extract_variables(predicate)}
            
            # Evaluate the predicate
            evaluation = predicate_fn(state, relevant_mapping, self.state_cache)

            return evaluation

        elif predicate_rule == "super_predicate":
            # No need to go back to __call__, there's nothing separate to cache here
            return self._inner_evaluate_predicate(predicate["pred"], state, mapping)

        elif predicate_rule == "super_predicate_not":
            return not self(predicate["not_args"], state, mapping)

        elif predicate_rule == "super_predicate_and":
            inner_values = [self(sub, state, mapping) for sub in predicate["and_args"]] # type: ignore
            return all(inner_values)  

        elif predicate_rule == "super_predicate_or":
            inner_values = [self(sub, state, mapping) for sub in predicate["or_args"]] # type: ignore
            return all(inner_values)  

        elif predicate_rule == "super_predicate_exists":
            variable_type_mapping = self._extract_variable_type_mapping(predicate["exists_vars"]["variables"])  # type: ignore
            object_assignments = list(itertools.product(*[sum([OBJECTS_BY_TYPE[var_type] for var_type in var_types], []) 
                                      for var_types in variable_type_mapping.values()]))

            sub_mappings = [dict(zip(variable_type_mapping.keys(), object_assignment)) for object_assignment in object_assignments]
            return any([self(predicate["exists_args"], state, {**sub_mapping, **mapping}) for 
                        sub_mapping in sub_mappings])

        elif predicate_rule == "super_predicate_forall":
            variable_type_mapping = self._extract_variable_type_mapping(predicate["forall_vars"]["variables"])  # type: ignore
            object_assignments = list(itertools.product(*[sum([OBJECTS_BY_TYPE[var_type] for var_type in var_types], []) 
                                      for var_types in variable_type_mapping.values()]))

            sub_mappings = [dict(zip(variable_type_mapping.keys(), object_assignment)) for object_assignment in object_assignments]
            return all([self(predicate["forall_args"], state, {**sub_mapping, **mapping}) for 
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
                # Obtain the functional representation of the function
                function = FUNCTION_LIBRARY[str(comp_arg_1["func_name"])]

                # Extract only the variables in the mapping relevant to this predicate
                relevant_mapping = {var: mapping[var] for var in extract_variables(predicate)}
            
                # Evaluate the function
                evaluation = function(state, relevant_mapping, self.state_cache)

                # If the function is undecidable with the current information, return None
                if evaluation is None:
                    return None

                comp_arg_1 = float(evaluation)

            else:
                comp_arg_1 = float(comp_arg_1)

            comp_arg_2 = comp["arg_2"]["arg"]  # type: ignore
            if isinstance(comp_arg_1, tatsu.ast.AST):
                # Obtain the functional representation of the base predicate
                function = FUNCTION_LIBRARY[str(comp_arg_2["func_name"])]
            
                # Extract only the variables in the mapping relevant to this predicate
                relevant_mapping = {var: mapping[var] for var in extract_variables(predicate)}  

                # Evaluate the function
                evaluation = function(state, relevant_mapping, self.state_cache)

                # If the function is undecidable with the current information, return None
                if evaluation is None:
                    return None

                comp_arg_2 = float(evaluation)


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


# ====================================== UTILITIES ======================================


def _vec3_dict_to_array(vec3: typing.Dict[str, float]):
    return np.array([vec3['x'], vec3['y'], vec3['z']])
    

def _object_location(object: ObjectState) -> np.ndarray:
    key = 'bboxCenter' if 'bboxCenter' in object else 'position'
    return _vec3_dict_to_array(object[key])


def mapping_objects_decorator(predicate_func: typing.Callable, object_id_key: str = OBJECT_ID_KEY) -> typing.Callable:
    def wrapper(state: FullState, predicate_partial_mapping: typing.Dict[str, str], state_cache: typing.Dict[str, ObjectState]):

        agent_object = state[AGENT_STATE_KEY] if state[AGENT_STATE_CHANGED_KEY] else state_cache[AGENT_STATE_KEY]

        # if there are no objects in the predicate mapping, then we can just evaluate the predicate
        if len(predicate_partial_mapping) == 0:
            return predicate_func(agent_object, [])

        # Otherwise, check if any of the relevant objects have changed in this state

        current_state_mapping_objects = {}
        mapping_values = set(predicate_partial_mapping.values())
        state_objects = typing.cast(list, state[OBJECTS_KEY])
        for object in state_objects:
            if object[object_id_key] in mapping_values:
                current_state_mapping_objects[object[object_id_key]] = object

        # None of the objects in the mapping are updated in the current state, so return None
        if len(current_state_mapping_objects) == 0:
            return None

        # At least one object is, so populate the rest from the cache
        for mapping_value in mapping_values:
            if mapping_value not in current_state_mapping_objects:
                if mapping_value not in state_cache:
                    # We don't have this object in the cache, so we can't evaluate the predicate
                    return None

                current_state_mapping_objects[mapping_value] = state_cache[mapping_value]

        mapping_objects = [current_state_mapping_objects[mapping_value] for _, mapping_value in predicate_partial_mapping.items()]
        return predicate_func(agent_object, mapping_objects)

    return wrapper


# ====================================== PREDICATE DEFINITIONS ======================================


@mapping_objects_decorator
def _pred_generic_predicate_interface(agent: AgentState, objects: typing.Sequence[ObjectState]):
    """
    This isn't here to do anything useful -- it's just to demonstrate the interface that all predicates
    should follow.  The first argument should be the agent's state, and the second should be a list 
    (potentially empty) of objects that are the arguments to this predicate.
    """
    raise NotImplementedError()


@mapping_objects_decorator
def _agent_crouches(agent: AgentState, objects: typing.Sequence[ObjectState]):
    assert len(objects) == 0
    return agent["crouching"]


@mapping_objects_decorator
def _pred_agent_holds(agent: AgentState, objects: typing.Sequence[ObjectState]):
    assert len(objects) == 1
    return agent["heldObjectName"] == objects[0]["name"]


@mapping_objects_decorator
def _pred_in(agent: AgentState, objects: typing.Sequence[ObjectState]):
    assert len(objects) == 2
    outer_object_bbox_center = _vec3_dict_to_array(objects[0]['bboxCenter'])
    outer_object_bbox_extents = _vec3_dict_to_array(objects[0]['bboxExtents'])
    inner_object_bbox_center = _vec3_dict_to_array(objects[1]['bboxCenter'])
    inner_object_bbox_extents = _vec3_dict_to_array(objects[1]['bboxExtents'])

    # start_inside = np.all(outer_object_bbox_center - outer_object_bbox_extents <= inner_object_bbox_center - inner_object_bbox_extents)
    # end_inside = np.all(inner_object_bbox_center + inner_object_bbox_extents <= outer_object_bbox_center + outer_object_bbox_extents)
    start_inside = np.all(outer_object_bbox_center - outer_object_bbox_extents <= inner_object_bbox_center)
    end_inside = np.all(inner_object_bbox_center <= outer_object_bbox_center + outer_object_bbox_extents)
    return start_inside and end_inside
   

@mapping_objects_decorator
def _pred_in_motion(agent: AgentState, objects: typing.Sequence[ObjectState]):
    assert len(objects) == 1
    return not (np.allclose(_vec3_dict_to_array(objects[0]["velocity"]), 0) and np.allclose(_vec3_dict_to_array(objects[0]["angularVelocity"]), 0))


@mapping_objects_decorator
def _pred_touch(agent: AgentState, objects: typing.Sequence[ObjectState]):
    assert len(objects) == 2
    return objects[1]['name'] in objects[0]['touchingObjects'] or objects[0]['name'] in objects[1]['touchingObjects']


# ====================================== FUNCTION DEFINITIONS =======================================


@mapping_objects_decorator
def _func_distance(agent: AgentState, objects: typing.Sequence[ObjectState]):
    assert len(objects) == 2
    # TODO: do we want to use the position? Or the bounding box?
    return np.linalg.norm(_object_location(objects[0]) - _object_location(objects[1]))


# ================================= EXTRACTING LIBRARIES FROM LOCALS() ==================================


PREDICATE_PREFIX = '_pred_'

PREDICATE_LIBRARY = {local_key.replace(PREDICATE_PREFIX, ''): local_val
    for local_key, local_val in locals().items()
    if local_key.startswith(PREDICATE_PREFIX)
}

FUNCTION_PREFIX = '_func_'

FUNCTION_LIBRARY = {local_key.replace(FUNCTION_PREFIX, ''): local_val
    for local_key, local_val in locals().items()
    if local_key.startswith(FUNCTION_PREFIX)
}

