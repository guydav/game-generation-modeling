import numpy as np
import tatsu
import tatsu.ast
import typing

from utils import extract_variable_type_mapping, extract_variables, get_object_assignments, ast_cache_key, _extract_object_limits,\
    _object_corners, _point_in_object, _object_location, FullState, ObjectState, AgentState, BuildingPseudoObject
from config import UNITY_PSEUDO_OBJECTS, PseudoObject

# AgentState = typing.NewType('AgentState', typing.Dict[str, typing.Any])
# ObjectState = typing.NewType('ObjectState', typing.Union[str, typing.Any])
# ObjectOrPseudo = typing.Union[ObjectState, PseudoObject]  # TODO: figure out the syntax for this



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
    evaluation_cache: typing.Dict[str, typing.Union[bool, float]]
    # The last state the evaluation cache was updated for a given key.
    evaluation_cache_last_updated: typing.Dict[str, int]
    # A cache of the latest observed value for each object
    state_cache: typing.Dict[str, typing.Union[ObjectState, AgentState, PseudoObject]]
    # The last state the state cache was updated for
    state_cache_global_last_updated: int
    # The last state each object was updated for
    state_cache_object_last_updated: typing.Dict[str, int]    

    def __init__(self, domain: str):
        self.domain = domain
        self._new_game()
    
    def _new_game(self):
        """
        Call when a new game is started to clear the cache.
        """
        self.evaluation_cache = {}
        self.evaluation_cache_last_updated = {}
        self.state_cache = {}
        self.state_cache_object_last_updated = {}
        self.state_cache.update(UNITY_PSEUDO_OBJECTS)
        self.state_cache_object_last_updated.update({k: -1 for k in UNITY_PSEUDO_OBJECTS.keys()})
        self.state_cache_global_last_updated = -1
    
    def __call__(self, predicate: typing.Optional[tatsu.ast.AST], state: FullState, 
        mapping: typing.Dict[str, str], force_evaluation: bool = False) -> bool:
        """
        The external API to the predicate handler.
        For now, implements the same logic as before, to make sure I separated it correctly from the `preference_handler`.
        After that, this will implement the caching logic.

        GD 2022-09-14: The data, as currently stored, saves delta updates of the state. 
        This means that the truth value of a predicate with a particular assignment holds unitl
        there's information that merits updating it. This means that we should cache predicate
        evaluation results, update them when they return a non-None value, and return the cached result. 

        GD 2022-09-29: We decided that since all external callers treat a None as a False, we might as well make it explicit here
        """
        pred_value = self._inner_call(predicate=predicate, state=state, mapping=mapping, force_evaluation=force_evaluation)

        return pred_value if pred_value is not None else False

    def _inner_call(self, predicate: typing.Optional[tatsu.ast.AST], state: FullState, 
        mapping: typing.Dict[str, str], force_evaluation: bool = False) -> typing.Optional[bool]:
        predicate_key = "{0}_{1}".format(*ast_cache_key(predicate, mapping))
        state_index = state.original_index

        # If no time has passed since the last update, we know we can use the cached value
        if predicate_key in self.evaluation_cache_last_updated and self.evaluation_cache_last_updated[predicate_key] == state_index:
            return typing.cast(bool, self.evaluation_cache[predicate_key])

        # This shouldn't happen, but no reason not to check it anyhow
        if state_index > self.state_cache_global_last_updated:
            self.update_cache(state)

        current_state_value = self._inner_evaluate_predicate(predicate, state, mapping, force_evaluation)
        if current_state_value is not None:
            self.evaluation_cache[predicate_key] = current_state_value
            self.evaluation_cache_last_updated[predicate_key] = state_index

        return typing.cast(bool, self.evaluation_cache.get(predicate_key, None))

    def update_cache(self, state: FullState):
        '''
        Update the cache if any objects / the agent are changed in the current state
        '''
        state_index = state.original_index

        self.state_cache_global_last_updated = state_index
        for obj in state.objects:
            self.state_cache[obj.object_id] = obj
            self.state_cache_object_last_updated[obj.object_id] = state_index

        if state.agent_state_changed:
            agent_state = typing.cast(AgentState, state.agent_state)
            self.state_cache[AGENT_STATE_KEY] = agent_state
            self.state_cache['agent'] = agent_state
            self.state_cache_object_last_updated[AGENT_STATE_KEY] = state_index
            self.state_cache_object_last_updated['agent'] = state_index

    def _inner_evaluate_predicate(self, predicate: typing.Optional[tatsu.ast.AST], state: FullState, 
        mapping: typing.Dict[str, str], force_evaluation: bool = False) -> typing.Optional[bool]:
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
            evaluation = predicate_fn(state, relevant_mapping, self.state_cache, self.state_cache_object_last_updated, force_evaluation)

            return evaluation

        elif predicate_rule == "super_predicate":
            # No need to go back to __call__, there's nothing separate to cache here
            return self._inner_evaluate_predicate(predicate["pred"], state, mapping, force_evaluation)

        elif predicate_rule == "super_predicate_not":
            inner_pred_value = self._inner_call(predicate["not_args"], state, mapping, force_evaluation)
            return None if inner_pred_value is None else not inner_pred_value

        elif predicate_rule == "super_predicate_and":
            and_args = predicate["and_args"]
            if isinstance(and_args, tatsu.ast.AST):
                and_args = [and_args]

            inner_values = [self._inner_call(sub, state, mapping, force_evaluation) for sub in and_args] # type: ignore
            # If there are any Nones, we cannot know about their conjunction, so return None
            if any(v is None for v in inner_values):
                return None
            return all(inner_values)  

        elif predicate_rule == "super_predicate_or":
            or_args = predicate["or_args"]
            if isinstance(or_args, tatsu.ast.AST):
                or_args = [or_args]

            inner_values = [self._inner_call(sub, state, mapping, force_evaluation) for sub in or_args] # type: ignore
            # We only need to return None when all the values are None, as any([None, False]) == False, which is fine
            if all(v is None for v in inner_values):
                return None
            return any(inner_values)  

        elif predicate_rule == "super_predicate_exists":
            variable_type_mapping = extract_variable_type_mapping(predicate["exists_vars"]["variables"])  # type: ignore
            used_objects = list(mapping.values())
            object_assignments = get_object_assignments(self.domain, variable_type_mapping.values(), used_objects)  # type: ignore

            sub_mappings = [dict(zip(variable_type_mapping.keys(), object_assignment)) for object_assignment in object_assignments]
            inner_mapping_values = [self._inner_call(predicate["exists_args"], state, {**sub_mapping, **mapping}, force_evaluation) for sub_mapping in sub_mappings]
            if all(v is None for v in inner_mapping_values):
                return None
            return any(inner_mapping_values)

        elif predicate_rule == "super_predicate_forall":
            variable_type_mapping = extract_variable_type_mapping(predicate["forall_vars"]["variables"])  # type: ignore
            used_objects = list(mapping.values())
            object_assignments = get_object_assignments(self.domain, variable_type_mapping.values(), used_objects)  # type: ignore

            sub_mappings = [dict(zip(variable_type_mapping.keys(), object_assignment)) for object_assignment in object_assignments]
            inner_mapping_values = [self._inner_call(predicate["forall_args"], state, {**sub_mapping, **mapping}, force_evaluation) for sub_mapping in sub_mappings]
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
                # If it's an AST, it's a function, so evaluate it
                comp_arg_1 = self.evaluate_function(comp_arg_1, state, mapping, force_evaluation)

                # If the function is undecidable with the current information, return None
                if comp_arg_1 is None:
                    return None

            comp_arg_1 = float(comp_arg_1)

            comp_arg_2 = comp["arg_2"]["arg"]  # type: ignore
            if isinstance(comp_arg_2, tatsu.ast.AST):
                # If it's an AST, it's a function, so evaluate it
                comp_arg_2 = self.evaluate_function(comp_arg_2, state, mapping, force_evaluation)

                # If the function is undecidable with the current information, return None
                if comp_arg_2 is None:
                    return None

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

    def evaluate_function(self, function: typing.Optional[tatsu.ast.AST], state: FullState, 
        mapping: typing.Dict[str, str], force_evaluation: bool = False) -> typing.Optional[float]:
        function_key = "{0}_{1}".format(*ast_cache_key(function, mapping))
        state_index = state.original_index

        # If no time has passed since the last update, we know we can use the cached value
        if function_key in self.evaluation_cache_last_updated and self.evaluation_cache_last_updated[function_key] == state_index:
            return self.evaluation_cache[function_key]

        # This shouldn't happen, but no reason not to check it anyhow
        if state_index > self.state_cache_global_last_updated:
            self.update_cache(state)

        current_state_value =  self._inner_evaluate_function(function, state, mapping, force_evaluation)
        if current_state_value is not None:
            self.evaluation_cache[function_key] = current_state_value
            self.evaluation_cache_last_updated[function_key] = state_index

        return self.evaluation_cache.get(function_key, None)

    def _inner_evaluate_function(self, function: typing.Optional[tatsu.ast.AST], state: FullState, 
        mapping: typing.Dict[str, str], force_evaluation: bool = False) -> typing.Optional[float]:

        if function is None:
            return None

        # Obtain the functional representation of the function
        func = FUNCTION_LIBRARY[str(function["func_name"])]

        # Extract only the variables in the mapping relevant to this predicate
        relevant_mapping = {var: mapping[var] for var in extract_variables(function)}
    
        # Evaluate the function
        evaluation = func(state, relevant_mapping, self.state_cache, self.state_cache_object_last_updated, force_evaluation)

        # If the function is undecidable with the current information, return None
        if evaluation is None:
            return None

        return float(evaluation)


def mapping_objects_decorator(predicate_func: typing.Callable) -> typing.Callable:
    def wrapper(state: FullState, predicate_partial_mapping: typing.Dict[str, str], state_cache: typing.Dict[str, ObjectState], 
        state_cache_last_updated: typing.Dict[str, int], force_evaluation: bool = False) -> typing.Optional[bool]:
        
        agent_object = state.agent_state if state.agent_state_changed else state_cache[AGENT_STATE_KEY]

        # if there are no objects in the predicate mapping, then we can just evaluate the predicate
        if len(predicate_partial_mapping) == 0:
            return predicate_func(agent_object, [])

        # Otherwise, check if any of the relevant objects have changed in this state
        mapping_values = predicate_partial_mapping.values()
        any_object_not_in_cache = any(obj not in state_cache for obj in mapping_values)
        # If any objects are not in the cache, we cannot evaluate the predidate
        if any_object_not_in_cache:
            if force_evaluation:
                raise ValueError(f'Attempted to force predicate evaluation while at least one object was not in the cache: {[(obj, obj in state_cache) for obj in mapping_values]}')
            return None

        any_objects_changed = any(state_cache_last_updated[object_id] == state.original_index for object_id in mapping_values)
        # None of the objects in the mapping are updated in the current state, so return None
        if not any_objects_changed and not force_evaluation:
            return None

        # At least one object is, so populate the rest from the cache
        mapping_objects = []
        for mapping_value in mapping_values:
            # If we don't have this object in the cache, we can't evaluate the predicate
            if mapping_value not in state_cache:
                return None

            mapping_objects.append(state_cache[mapping_value])

        return predicate_func(agent_object, mapping_objects)

    return wrapper


# ====================================== PREDICATE DEFINITIONS ======================================


def _pred_generic_predicate_interface(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    """
    This isn't here to do anything useful -- it's just to demonstrate the interface that all predicates
    should follow.  The first argument should be the agent's state, and the second should be a list 
    (potentially empty) of objects that are the arguments to this predicate.
    """
    raise NotImplementedError()


def _pred_agent_crouches(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 0
    return agent.crouching


def _pred_agent_holds(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 1
    if isinstance(objects[0], PseudoObject):
        return False
    return agent.held_object == objects[0].object_id


def _object_in_building(building: BuildingPseudoObject, other_object: ObjectState):
    return other_object.object_id in building.building_objects


IN_MARGIN = 0.05


def _pred_in(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 2

    first_pseudo = isinstance(objects[0], PseudoObject)
    second_pseudo = isinstance(objects[1], PseudoObject)

    if first_pseudo or second_pseudo:
        first_building = isinstance(objects[0], BuildingPseudoObject)
        second_building = isinstance(objects[1], BuildingPseudoObject)

        if first_building == second_building:
            return False  # a building cannot be inside another building, same holds for other pseudo objects

        if first_building:
            return _object_in_building(*objects)  # type: ignore

        # if the second object is a building, we continue to the standard implementation

    outer_min_corner, outer_max_corner = _extract_object_limits(objects[0])
    inner_min_corner, inner_max_corner = _extract_object_limits(objects[1])    

    return np.all(inner_min_corner >= outer_min_corner - IN_MARGIN) and np.all(inner_max_corner <= outer_max_corner + IN_MARGIN) 

    # inner_object_bbox_center = objects[1].bbox_center

    # # The interior object's bbox center must be inside the exterior object's bbox
    # inner_bbox_center_contained = np.all(outer_min_corner <= inner_object_bbox_center) and np.all(inner_object_bbox_center <= outer_max_corner)

    # # We also check to make sure that the outer object's bbox is not entirely inside the inner object's bbox (possible for non-convex objects)
    # outer_bbox_contained = np.all(inner_min_corner <= outer_min_corner) and np.all(outer_max_corner <= inner_max_corner)
    
    # return inner_bbox_center_contained and not outer_bbox_contained


# TODO (GD): we should discuss what this threshold should be
IN_MOTION_ZERO_VELOCITY_THRESHOLD = 0.1

def _pred_in_motion(agent: AgentState, objects: typing.Sequence[ObjectState]):
    assert len(objects) == 1
    if isinstance(objects[0], PseudoObject):
        return False
    return not (np.allclose(objects[0].velocity, 0, atol=IN_MOTION_ZERO_VELOCITY_THRESHOLD) and \
        np.allclose(objects[0].angular_velocity, 0, atol=IN_MOTION_ZERO_VELOCITY_THRESHOLD))


TOUCH_DISTANCE_THRESHOLD = 0.15


def _building_touch(agent: AgentState, building: BuildingPseudoObject, other_object: typing.Union[ObjectState, PseudoObject]):
    if other_object.object_id in building.building_objects:
        return False

    return any([_pred_touch(agent, [building_obj, other_object]) for building_obj in building.building_objects.values()])


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
        first_building = isinstance(objects[0], BuildingPseudoObject)
        second_building = isinstance(objects[1], BuildingPseudoObject)
        if first_building == second_building:
            return False   # if both are buildings they would be merged; if neither, they wouldn't touch
        
        if first_building:
            return _building_touch(agent, objects[0], objects[1])  # type: ignore

        elif second_building:
            return _building_touch(agent, objects[1], objects[0])  # type: ignore

    elif first_pseudo:
        obj = typing.cast(ObjectState, objects[1])
        pseudo_obj = objects[0]

        if isinstance(pseudo_obj, BuildingPseudoObject):
            return _building_touch(agent, pseudo_obj, obj)

        return (pseudo_obj.object_id in obj.touching_objects or pseudo_obj.name in obj.touching_objects) and \
            pseudo_obj is _find_nearest_pseudo_object_of_type(obj, pseudo_obj.object_type)  # type: ignore 
    elif second_pseudo:
        obj = typing.cast(ObjectState, objects[0])
        pseudo_obj = objects[1]

        if isinstance(pseudo_obj, BuildingPseudoObject):
            return _building_touch(agent, pseudo_obj, obj)

        return (pseudo_obj.object_id in obj.touching_objects or pseudo_obj.name in obj.touching_objects) and \
            pseudo_obj is _find_nearest_pseudo_object_of_type(obj, pseudo_obj.object_type)  # type: ignore 
    else:
        return objects[1].object_id in objects[0].touching_objects or objects[0].object_id in objects[1].touching_objects  # type: ignore


ON_DISTANCE_THRESHOLD = 0.01

def _pred_on(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 2

    lower_object = objects[0]
    upper_object = objects[1]

    objects_touch = _pred_touch(agent, objects)

    if objects_touch:
        # TODO: the 'agent' does not have a bounding box, which breaks this implementation of _on

        upper_object_bbox_center = upper_object.bbox_center
        upper_object_bbox_extents = upper_object.bbox_extents

        # Project a point slightly below the bottom center / corners of the upper object
        upper_object_corners = _object_corners(upper_object)

        test_points = [corner - np.array([0, upper_object_bbox_extents[1] + ON_DISTANCE_THRESHOLD, 0])
                       for corner in upper_object_corners]
        test_points.append(upper_object_bbox_center - np.array([0, upper_object_bbox_extents[1] + ON_DISTANCE_THRESHOLD, 0]))

        objects_on = any([_point_in_object(test_point, lower_object) for test_point in test_points])

        # object 1 is on object 0 if they're touching and object 1 is above object 0
        # or if they're touching and object 1 is contained withint object 0? 
        return objects_on or _pred_in(agent, objects)

    elif isinstance(upper_object, BuildingPseudoObject):
        # A building can also be on an object if that object is (a) in the building
        if lower_object.object_id not in upper_object.building_objects:
            return False

        # and (b) that object is not on any other object in the building
        return not any([_pred_on(agent, [building_object, lower_object]) 
            for building_object in upper_object.building_objects.values()
            if building_object.object_id != lower_object.object_id])

    elif isinstance(lower_object, BuildingPseudoObject):
        # An object is on a building if that object is (a) in the building
        if upper_object.object_id not in lower_object.building_objects:
            return False

        # and (b) no other object in the building is on that object
        return not any([_pred_on(agent, [upper_object, building_object]) 
            for building_object in lower_object.building_objects.values()
            if building_object.object_id != upper_object.object_id])

    return False

def _pred_adjacent(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 2

    # Two objects are adjacent either if they're touching or if the minimum distance between any pair of
    # sides between their bounding boxes is less than some threshold. We use the bottom of each object's
    # bounding box for this

    object_1_corners = _object_corners(objects[0], "bottom")
    object_2_corners = _object_corners(objects[1], "bottom")

    # We can check if two objects are adjacent from the corners by first making sure there's some overlap in their
    # y extents. To check the distance between pairs of sides, we can make use of the fact that they're always
    # axis-aligned. 

    object_1_sides = [(object_1_corners[idx], object_1_corners[(idx+1)%4]) for idx in range(4)]
    object_2_sides = [(object_2_corners[idx], object_2_corners[(idx+1)%4]) for idx in range(4)]


# ====================================== FUNCTION DEFINITIONS =======================================


def _find_nearest_pseudo_object_of_type(object: ObjectState, object_type: str):
    """
    Finds the pseudo object in the sequence that is closest to the object.
    """
    pseudo_objects = list(UNITY_PSEUDO_OBJECTS.values())
    distances = [_distance_object_pseudo_object(object, pseudo_object) for pseudo_object in pseudo_objects
        if pseudo_object.object_type == object_type]
    return pseudo_objects[np.argmin(distances)]


def _get_pseudo_object_relevant_distance_dimension_index(pseudo_object: PseudoObject):
    if np.allclose(pseudo_object.rotation[1], 0):
        return 2

    if np.allclose(pseudo_object.rotation[1], 90):
        return 0

    raise NotImplemented(f'Cannot compute distance between object and pseudo object with rotation {pseudo_object.rotation}')


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


def _func_building_size(agent: AgentState, objects: typing.Sequence[typing.Union[ObjectState, PseudoObject]]):
    assert len(objects) == 1
    assert isinstance(objects[0], BuildingPseudoObject)

    return len(objects[0].building_objects)
    


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

