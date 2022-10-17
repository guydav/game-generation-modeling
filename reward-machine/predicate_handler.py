import pathlib
import sys

sys.path.append((pathlib.Path(__file__).parents[1].resolve() / 'src').as_posix())

import numpy as np
import tatsu
import tatsu.ast
import typing

import ast_printer

from utils import extract_variable_type_mapping, extract_variables, get_object_assignments
from config import OBJECTS_BY_ROOM_AND_TYPE, UNITY_PSEUDO_OBJECTS, PseudoObject

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
OBJECT_NAME_KEY = 'name'


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
    state_cache: typing.Dict[str, typing.Union[ObjectState, AgentState, PseudoObject]]
    # The last state the state cache was updated for
    state_cache_last_updated: int

    def __init__(self, domain: str, index_key: str = ORIGINAL_INDEX_KEY, object_id_key: str = OBJECT_ID_KEY):
        self.domain = domain
        self.index_key = index_key
        self.object_id_key = object_id_key
        self._new_game()
    
    def _new_game(self):
        """
        Call when a new game is started to clear the cache.
        """
        self.evaluation_cache = {}
        self.evaluation_cache_last_updated = {}
        self.state_cache = {}
        self.state_cache.update(UNITY_PSEUDO_OBJECTS)
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
    
    def __call__(self, predicate: typing.Optional[tatsu.ast.AST], state: typing.Dict[str, typing.Any], mapping: typing.Dict[str, str]) -> bool:
        """
        The external API to the predicate handler.
        For now, implements the same logic as before, to make sure I separated it correctly from the `preference_handler`.
        After that, this will implement the caching logic.

        GD 2022-09-14: The data, as currently stored, saves delta updates of the state. 
        This means that the truth value of a predicate with a particular assignment holds unitl
        there's information that merits updating it. This means that we should cache predicate
        evaluation results, update them when they return a non-None value, and return the cached result. 

        GD 2022-09-29: We decided that since all external callers treat a None 
        """
        pred_value = self._inner_call(predicate=predicate, state=state, mapping=mapping)

        if pred_value is not None: print("Predicate: ", predicate, " with mapping: ", mapping, " has value: ", pred_value)

        return pred_value if pred_value is not None else False

    def _inner_call(self, predicate: typing.Optional[tatsu.ast.AST], state: typing.Dict[str, typing.Any], mapping: typing.Dict[str, str]) -> typing.Optional[bool]:
        predicate_key = self._cache_key(predicate, mapping)
        state_index = state[self.index_key]

        # print("Evaluating: ", predicate_key, " at state index: ", state_index)

        # If no time has passed since the last update, we know we can use the cached value
        if predicate_key in self.evaluation_cache_last_updated and self.evaluation_cache_last_updated[predicate_key] == state_index:
            return self.evaluation_cache[predicate_key]

        if state_index > self.state_cache_last_updated:
            self.update_cache(state)

        current_state_value =  self._inner_evaluate_predicate(predicate, state, mapping)
        if current_state_value is not None:
            self.evaluation_cache[predicate_key] = current_state_value
            self.evaluation_cache_last_updated[predicate_key] = state_index

        return self.evaluation_cache.get(predicate_key, None)

    def update_cache(self, state: typing.Dict[str, typing.Any]):
        '''
        Update the cache if any objects / the agent are changed in the current state
        '''
        state_index = state[self.index_key]

        self.state_cache_last_updated = state_index
        for obj in state[OBJECTS_KEY]:
            self.state_cache[obj[self.object_id_key]] = obj

        if state[AGENT_STATE_CHANGED_KEY]:
            self.state_cache[AGENT_STATE_KEY] = state[AGENT_STATE_KEY]
            self.state_cache['agent'] = state[AGENT_STATE_KEY]

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

        # print("Doing inner evaluate predicate")

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
            inner_pred_value = self._inner_call(predicate["not_args"], state, mapping)
            return None if inner_pred_value is None else not inner_pred_value

        # TODO: technically, AND and OR can accept a single argument under the grammar, but that would break the
        #       logic here, since it expects to be able to iterate through the 'and_args' / 'or_args'
        elif predicate_rule == "super_predicate_and":
            inner_values = [self._inner_call(sub, state, mapping) for sub in predicate["and_args"]] # type: ignore
            # If there are any Nones, we cannot know about their conjunction, so return None
            if any(v is None for v in inner_values):
                return None
            return all(inner_values)  

        elif predicate_rule == "super_predicate_or":
            inner_values = [self._inner_call(sub, state, mapping) for sub in predicate["or_args"]] # type: ignore
            # We only need to return None when all the values are None, as any([None, False]) == False, which is fine
            if all(v is None for v in inner_values):
                return None
            return any(inner_values)  

        elif predicate_rule == "super_predicate_exists":
            variable_type_mapping = extract_variable_type_mapping(predicate["exists_vars"]["variables"])  # type: ignore
            object_assignments = get_object_assignments(self.domain, variable_type_mapping.values())  # type: ignore

            sub_mappings = [dict(zip(variable_type_mapping.keys(), object_assignment)) for object_assignment in object_assignments]
            inner_mapping_values = [self._inner_call(predicate["exists_args"], state, {**sub_mapping, **mapping}) for sub_mapping in sub_mappings]
            if all(v is None for v in inner_mapping_values):
                return None
            return any(inner_mapping_values)

        elif predicate_rule == "super_predicate_forall":
            variable_type_mapping = extract_variable_type_mapping(predicate["forall_vars"]["variables"])  # type: ignore
            object_assignments = get_object_assignments(self.domain, variable_type_mapping.values())  # type: ignore

            sub_mappings = [dict(zip(variable_type_mapping.keys(), object_assignment)) for object_assignment in object_assignments]
            inner_mapping_values = [self._inner_call(predicate["forall_args"], state, {**sub_mapping, **mapping}) for sub_mapping in sub_mappings]
            if any(v is None for v in inner_mapping_values):
                return None
            return all(inner_mapping_values)

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
    

def _object_location(object: typing.Union[ObjectState, PseudoObject]) -> np.ndarray:
    key = 'bboxCenter' if 'bboxCenter' in object else 'position'
    return _vec3_dict_to_array(object[key])

def _object_corners(object: ObjectState):
    '''
    Returns the coordinates of each of the 4 corners of the object's bounding box, with the
    y coordinate matching the center of mass
    '''

    bbox_center = _vec3_dict_to_array(object['bboxCenter'])
    bbox_extents = _vec3_dict_to_array(object['bboxExtents'])

    corners = [bbox_center + np.array([bbox_extents[0], 0, bbox_extents[2]]),
               bbox_center + np.array([-bbox_extents[0], 0, bbox_extents[2]]),
               bbox_center + np.array([bbox_extents[0], 0, -bbox_extents[2]]),
               bbox_center + np.array([-bbox_extents[0], 0, -bbox_extents[2]])
              ]

    return corners

def _point_in_object(point: np.ndarray, object: ObjectState):
    '''
    Returns whether a point is contained with the bounding box of the provided object
    '''

    bbox_center = _vec3_dict_to_array(object['bboxCenter'])
    bbox_extents = _vec3_dict_to_array(object['bboxExtents'])

    return np.all(point >= bbox_center - bbox_extents) and np.all(point <= bbox_center + bbox_extents)

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


def _pred_generic_predicate_interface(agent: AgentState, objects: typing.Sequence[ObjectState]):
    """
    This isn't here to do anything useful -- it's just to demonstrate the interface that all predicates
    should follow.  The first argument should be the agent's state, and the second should be a list 
    (potentially empty) of objects that are the arguments to this predicate.
    """
    raise NotImplementedError()


def _agent_crouches(agent: AgentState, objects: typing.Sequence[ObjectState]):
    assert len(objects) == 0
    return agent["crouching"]


def _pred_agent_holds(agent: AgentState, objects: typing.Sequence[ObjectState]):
    assert len(objects) == 1
    if isinstance(objects[0], PseudoObject):
        return False
    return agent["heldObject"] == objects[0][OBJECT_ID_KEY]


def _pred_in(agent: AgentState, objects: typing.Sequence[ObjectState]):
    assert len(objects) == 2

    if isinstance(objects[0], PseudoObject) or isinstance(objects[1], PseudoObject):
        return False

    outer_object_bbox_center = _vec3_dict_to_array(objects[0]['bboxCenter'])
    outer_object_bbox_extents = _vec3_dict_to_array(objects[0]['bboxExtents'])
    inner_object_bbox_center = _vec3_dict_to_array(objects[1]['bboxCenter'])
    # inner_object_bbox_extents = _vec3_dict_to_array(objects[1]['bboxExtents'])
    # start_inside = np.all(outer_object_bbox_center - outer_object_bbox_extents <= inner_object_bbox_center - inner_object_bbox_extents)
    # end_inside = np.all(inner_object_bbox_center + inner_object_bbox_extents <= outer_object_bbox_center + outer_object_bbox_extents)
    start_inside = np.all(outer_object_bbox_center - outer_object_bbox_extents <= inner_object_bbox_center)
    end_inside = np.all(inner_object_bbox_center <= outer_object_bbox_center + outer_object_bbox_extents)
    return start_inside and end_inside
  

ON_DISTANCE_THRESHOLD = 0.01

def _pred_on(agent: AgentState, objects: typing.Sequence[ObjectState]):
    assert len(objects) == 2

    print("Computing 'on' between {} and {}".format(objects[0][OBJECT_ID_KEY], objects[1][OBJECT_ID_KEY]))

    lower_object = objects[0]
    upper_object = objects[1]

    objects_touch = _pred_touch(agent, objects)

    if objects_touch:
        # TODO: the 'agent' does not have a bounding box, which breaks this implementation of _on

        upper_object_bbox_center = _vec3_dict_to_array(upper_object['bboxCenter'])
        upper_object_bbox_extents = _vec3_dict_to_array(upper_object['bboxExtents'])

        # Project a point slightly below the bottom center / corners of the upper object
        upper_object_corners = _object_corners(upper_object)

        test_points = [corner - np.array([0, upper_object_bbox_extents[1] + ON_DISTANCE_THRESHOLD, 0])
                       for corner in upper_object_corners]
        test_points.append(upper_object_bbox_center - np.array([0, upper_object_bbox_extents[1] + ON_DISTANCE_THRESHOLD, 0]))

        objects_touch = _pred_touch(agent, objects)
        objects_on = any([_point_in_object(test_point, lower_object) for test_point in test_points])

        # object 1 is on object 0 if they're touching and object 1 is above object 0
        # or if they're touching and object 1 is contained withint object 0? 
        return objects_on or _pred_in(agent, objects)

    return False

def _pred_in_motion(agent: AgentState, objects: typing.Sequence[ObjectState]):
    assert len(objects) == 1
    if isinstance(objects[0], PseudoObject):
        return False
    return not (np.allclose(_vec3_dict_to_array(objects[0]["velocity"]), 0) and np.allclose(_vec3_dict_to_array(objects[0]["angularVelocity"]), 0))


TOUCH_DISTANCE_THRESHOLD = 0.15

def _pred_touch(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 2

    first_pseudo = isinstance(objects[0], PseudoObject)
    second_pseudo = isinstance(objects[1], PseudoObject)

    # TODO (GD 2022-09-27): the logic here to decide which wall to attribute the collision here is incomoplete; 
    # right now it assigns it to the nearest wall, but that could be incorrect, if the ball hit the wall at an angle 
    # and traveled sufficiently far to be nearest another wall. This is a (literal) corner case, but one we probably 
    # want to eventually resolve better, for example by simulating the ball back in time using the negative of its 
    # velcoity and finding a wall it was most recently very close to?

    if first_pseudo and second_pseudo:
        return False
    elif first_pseudo:
        obj = objects[1]
        pseudo_obj = objects[0]

        return (pseudo_obj[OBJECT_ID_KEY] in obj['touchingObjects'] or pseudo_obj[OBJECT_NAME_KEY] in obj['touchingObjects']) and \
            pseudo_obj is _find_nearest_pseudo_object(obj, list(UNITY_PSEUDO_OBJECTS.values()))  # type: ignore 
    elif second_pseudo:
        obj = objects[0]
        pseudo_obj = objects[1]

        return (pseudo_obj[OBJECT_ID_KEY] in obj['touchingObjects'] or pseudo_obj[OBJECT_NAME_KEY] in obj['touchingObjects']) and \
            pseudo_obj is _find_nearest_pseudo_object(obj, list(UNITY_PSEUDO_OBJECTS.values()))  # type: ignore 
    else:
        return objects[1][OBJECT_ID_KEY] in objects[0]['touchingObjects'] or objects[0][OBJECT_ID_KEY] in objects[1]['touchingObjects']


# ====================================== FUNCTION DEFINITIONS =======================================


def _find_nearest_pseudo_object(object: ObjectState, pseudo_objects: typing.Sequence[PseudoObject]):
    """
    Finds the pseudo object in the sequence that is closest to the object.
    """
    pseudo_objects = list(pseudo_objects)
    distances = [_distance_object_pseudo_object(object, pseudo_object) for pseudo_object in pseudo_objects]
    return pseudo_objects[np.argmin(distances)]


def _get_pseudo_object_relevant_distance_dimension_index(pseudo_object: PseudoObject):
    if np.allclose(pseudo_object.rotation['y'], 0):
        return 2

    if np.allclose(pseudo_object.rotation['y'], 90):
        return 0

    raise NotImplemented(f'Cannot compute distance between object and pseudo object with rotation {pseudo_object.rotation}')


def _get_pseudo_object_relevant_distance_dimension(pseudo_object: PseudoObject):
    return 'xyz'[_get_pseudo_object_relevant_distance_dimension_index(pseudo_object)]


def _distance_object_pseudo_object(object: ObjectState, pseudo_object: PseudoObject):
    distance_dimension = _get_pseudo_object_relevant_distance_dimension_index(pseudo_object)
    return np.linalg.norm(_object_location(object)[distance_dimension] - _object_location(pseudo_object)[distance_dimension])
        

def _func_distance(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 2

    first_pseudo = isinstance(objects[0], PseudoObject)
    second_pseudo = isinstance(objects[1], PseudoObject)

    if first_pseudo and second_pseudo:
        # handled identically to the case where neither is a pseudo object
        pass
    elif first_pseudo:
        return _distance_object_pseudo_object(objects[1], objects[0])  # type: ignore
    elif second_pseudo:
        return _distance_object_pseudo_object(objects[0], objects[1])  # type: ignore
    
    
    return np.linalg.norm(_object_location(objects[0]) - _object_location(objects[1]))


    # TODO: do we want to use the position? Or the bounding box?
    


# ================================= EXTRACTING LIBRARIES FROM LOCALS() ==================================


PREDICATE_PREFIX = '_pred_'

PREDICATE_LIBRARY = {local_key.replace(PREDICATE_PREFIX, ''): mapping_objects_decorator(local_val_pred)
    for local_key, local_val_pred in locals().items()
    if local_key.startswith(PREDICATE_PREFIX)
}

FUNCTION_PREFIX = '_func_'

FUNCTION_LIBRARY = {local_key.replace(FUNCTION_PREFIX, ''): mapping_objects_decorator(local_val_func)
    for local_key, local_val_func in locals().items()
    if local_key.startswith(FUNCTION_PREFIX)
}

