from collections import OrderedDict, namedtuple
import inflect
import itertools
import numpy as np
import pathlib
import sys
import tatsu
import tatsu.ast
import typing

sys.path.append((pathlib.Path(__file__).parents[1].resolve() / 'src').as_posix())
import ast_printer

from config import ALL_OBJECT_TYPES, COLORS, NAMED_OBJECTS, OBJECTS_BY_ROOM_AND_TYPE, PseudoObject

PROJECT_NAME = 'game-generation-modeling'


def get_project_dir(project_name: str = PROJECT_NAME):
    return __file__[:__file__.find(project_name) + len(project_name)]


def _vec_dict_to_array(vec: typing.Dict[str, float]):
    if 'x' not in vec or 'y' not in vec:
        raise ValueError(f'x and y must be in vec dict; received {vec}')
    
    if 'z' in vec:
        if 'w' in vec:
            # TODO (GD 2022-10-16): decide if this should be wxyz or xyzw
            return np.array([vec['w'], vec['x'], vec['y'], vec['z']])

        return np.array([vec['x'], vec['y'], vec['z']])

    return np.array([vec['x'], vec['y']])


# TODO (GD 2022-10-16): is there a way to specify an ndarray with some number of entires/dimensions? a shape?


class AgentState(typing.NamedTuple):
    angle: float
    angle_int: int
    camera_local_rotation: np.ndarray   # w, x, y, z
    camera_rotation_euler_angles: np.ndarray  # x, y, z
    crouching: bool
    direction: np.ndarray  # x, y, z
    held_object: str
    input: np.ndarray  # x, y
    last_movement_result: bool
    local_rotation: np.ndarray  # w, x, y, z
    mouse_only: bool
    position: np.ndarray  # x, y, z
    rotation_euler_angles: np.ndarray  # x, y, z
    succeeded: bool
    target_position: np.ndarray  # x, y, z
    touching_ceiling: bool
    touching_floor: bool
    touching_side: bool

    @staticmethod
    def from_state_dict(state_dict: typing.Dict[str, typing.Any]):
        return AgentState(
            angle=state_dict['angle'],
            angle_int=state_dict['angleInt'],
            camera_local_rotation=_vec_dict_to_array(state_dict['cameraLocalRotation']) if 'cameraLocalRotation' in state_dict 
                                  else _vec_dict_to_array(state_dict['cameraLocalRoation']), # manually handle typo in old traces
            camera_rotation_euler_angles=_vec_dict_to_array(state_dict['cameraRotationEulerAngles']),
            crouching=state_dict['crouching'],
            direction=_vec_dict_to_array(state_dict['direction']),
            held_object=state_dict['heldObject'],
            input=_vec_dict_to_array(state_dict['input']),
            last_movement_result=state_dict['lastMovementResult'],
            local_rotation=_vec_dict_to_array(state_dict['localRotation']),
            mouse_only=state_dict['mouseOnly'],
            position=_vec_dict_to_array(state_dict['position']),
            rotation_euler_angles=_vec_dict_to_array(state_dict['rotationEulerAngles']),
            succeeded=state_dict['succeeded'],
            target_position=_vec_dict_to_array(state_dict['targetPosition']),
            touching_ceiling=state_dict['touchingCeiling'],
            touching_floor=state_dict['touchingFloor'],
            touching_side=state_dict['touchingSide'],
        )


class ObjectState(typing.NamedTuple):
    angular_velocity: np.ndarray  # x, y, z
    bbox_center: np.ndarray  #  x, y, z
    bbox_extents:  np.ndarray  # x, y, z
    is_broken: bool
    is_open: bool
    is_toggled: bool
    name: str
    object_id: str
    object_type: str
    position: np.ndarray  # x, y, z
    rotation: np.ndarray  # x, y, z
    touching_objects: typing.List[str]
    contained_objects: typing.List[str]
    velocity: np.ndarray  # x, y, z

    @staticmethod
    def from_state_dict(state_dict: typing.Dict[str, typing.Any]):
        return ObjectState(
            angular_velocity=_vec_dict_to_array(state_dict['angularVelocity']),
            bbox_center=_vec_dict_to_array(state_dict['bboxCenter']),
            bbox_extents=_vec_dict_to_array(state_dict['bboxExtents']),
            is_broken=state_dict['isBroken'],
            is_open=state_dict['isOpen'],
            is_toggled=state_dict['isToggled'],
            name=state_dict['name'],
            object_id=state_dict['objectId'],
            object_type=state_dict['objectType'],
            position=_vec_dict_to_array(state_dict['position']),
            rotation=_vec_dict_to_array(state_dict['rotation']),
            touching_objects=state_dict['touchingObjects'],
            contained_objects=state_dict['containedObjects'] if 'containedObjects' in state_dict else None, # old traces don't have containedObjects
            velocity=_vec_dict_to_array(state_dict['velocity']),
        )


class ActionState(typing.NamedTuple):
    action: str
    degrees: float
    force_action: bool
    object_id: str
    object_name: str
    object_type: str
    random_seed: int
    rotation: np.ndarray  # x, y, z
    x: float
    y: float
    z: float

    @staticmethod
    def from_state_dict(state_dict: typing.Dict[str, typing.Any]):
        return ActionState(
            action=state_dict['action'],
            degrees=state_dict['degrees'],
            force_action=state_dict['forceAction'],
            object_id=state_dict['objectId'],
            object_name=state_dict['objectName'],
            object_type=state_dict['objectType'],
            random_seed=state_dict['randomSeed'],
            rotation=_vec_dict_to_array(state_dict['rotation']),
            x=state_dict['x'],
            y=state_dict['y'],
            z=state_dict['z'],
        )


class FullState(typing.NamedTuple):
    action: typing.Optional[ActionState]
    action_changed: bool
    agent_state: typing.Optional[AgentState]
    agent_state_changed: bool
    index: int
    n_objects_changed: int
    objects: typing.List[ObjectState]
    original_index: int

    @staticmethod
    def from_state_dict(state_dict: typing.Dict[str, typing.Any]):
        action_changed = state_dict['actionChanged']
        agent_state_changed = state_dict['agentStateChanged']
        return FullState(
            action=ActionState.from_state_dict(state_dict['action']) if action_changed else None,
            action_changed=action_changed,
            agent_state=AgentState.from_state_dict(state_dict['agentState']) if agent_state_changed else None,
            agent_state_changed=agent_state_changed,
            index=state_dict['index'],
            n_objects_changed=state_dict['nObjectsChanged'],
            objects=[ObjectState.from_state_dict(object_state_dict) for object_state_dict in state_dict['objects']],
            original_index=state_dict['originalIndex'],
        )


BUILDING_TYPE = 'building'


class BuildingPseudoObject(PseudoObject):
    building_objects: typing.Dict[str, ObjectState]  # a collection of the objects in the building
    min_corner: np.ndarray
    max_corner: np.ndarray
    
    def __init__(self, building_id: str):
        super().__init__(building_id, BUILDING_TYPE, building_id, np.zeros(3), np.zeros(3), np.zeros(3))
        self.building_objects = {}
        self.min_corner = np.zeros(3)
        self.max_corner = np.zeros(3)
        self.position_valid = False

    def add_object(self, obj: ObjectState):
        '''
        Add a new object to the building and update the building's position and bounding box
        '''
        self.building_objects[obj.object_id] = obj        
        obj_min, obj_max = _extract_object_limits(obj)

        if not self.position_valid:
            self.min_corner = obj_min
            self.max_corner = obj_max
            self.position_valid = True

        else:
            self.min_corner = np.minimum(obj_min, self.min_corner)  # type: ignore
            self.max_corner = np.maximum(obj_max, self.max_corner)  # type: ignore

        self._update_position_from_corners()
        
    def _update_position_from_corners(self) -> None:
        self.position = self.bbox_center = (self.min_corner + self.max_corner) / 2  # type: ignore
        self.bbox_extents = (self.max_corner - self.min_corner) / 2  # type: ignore

    def remove_object(self, obj: ObjectState):
        if obj.object_id not in self.building_objects:
            raise ValueError(f'Object {obj.object_id} is not in building {self.name}')
        
        del self.building_objects[obj.object_id]

        if len(self.building_objects) == 0:
            self.position_valid = False
        
        else:
            object_minima, object_maxima = list(zip(*[_extract_object_limits(curr_obj) 
                for curr_obj in self.building_objects.values()]))
            
            self.min_corner = np.min(object_minima, axis=0)  
            self.max_corner = np.max(object_maxima, axis=0)  
            self._update_position_from_corners()


def _object_location(object: typing.Union[AgentState, ObjectState, PseudoObject]) -> np.ndarray:
    return object.bbox_center if hasattr(object, 'bbox_center') and object.bbox_center is not None else object.position  # type: ignore

def _object_corners(object: typing.Union[ObjectState, PseudoObject], y_pos: str = 'center'):
    '''
    Returns the coordinates of each of the 4 corners of the object's bounding box, with the
    y coordinate matching either
    - a provided integer / float value
    - the center of the object's bounding box (y_offset='center')
    - the minimum y coordinate of the object's bounding box (y_offset='bottom')
    - the maximum y coordinate of the object's bounding box (y_offset='top')

    Assuming that positive x is to the right and positive z is forward, the corners are
    returned in the following order:
    - 0: top right
    - 1: bottom right
    - 2: bottom left
    - 3: top left
    '''

    bbox_center = object.bbox_center
    bbox_extents = object.bbox_extents

    if isinstance(y_pos, int) or isinstance(y_pos, float):
        y = y_pos
    elif y_pos == 'center':
        y = 0
    elif y_pos == 'bottom':
        y = -bbox_extents[1]
    elif y_pos == 'top':
        y = bbox_extents[1]

    corners = [bbox_center + np.array([bbox_extents[0], y, bbox_extents[2]]),
               bbox_center + np.array([bbox_extents[0], y, -bbox_extents[2]]),
               bbox_center + np.array([-bbox_extents[0], y, -bbox_extents[2]]),
               bbox_center + np.array([-bbox_extents[0], y, bbox_extents[2]])
              ]

    return corners


def _extract_object_limits(obj: typing.Union[ObjectState, PseudoObject]):
    obj_center = _object_location(obj)
    obj_extents = obj.bbox_extents

    obj_min = obj_center - obj_extents
    obj_max = obj_center + obj_extents
    return obj_min, obj_max


def _point_in_object(point: np.ndarray, object: typing.Union[ObjectState, PseudoObject]):
    '''
    Returns whether a point is contained with the bounding box of the provided object
    '''

    bbox_center = object.bbox_center
    bbox_extents = object.bbox_extents

    return np.all(point >= bbox_center - bbox_extents) and np.all(point <= bbox_center + bbox_extents)


def extract_variable_type_mapping(variable_list: typing.Union[typing.Sequence[tatsu.ast.AST], tatsu.ast.AST]) -> typing.Dict[str, typing.List[str]]:
    '''
    Given a list of variables (a type of AST), extract the mapping from variable names to variable types. Variable types 
    are returned in lists, even in cases where there is only one possible for the variable in order to handle cases
    where multiple types are linked together with an (either) clause

    '''
    if isinstance(variable_list, tatsu.ast.AST):
        variable_list = [variable_list]

    variables = OrderedDict({})
    for var_info in variable_list:
        var_type = typing.cast(tatsu.ast.AST, var_info["var_type"])

        if isinstance(var_type["type"], tatsu.ast.AST):
            var_type_type = typing.cast(tatsu.ast.AST, var_type["type"])
            var_type_name = var_type_type["type_names"]
        else:
            var_type_name = var_type["type"]

        var_names = var_info["var_names"]
        if isinstance(var_names, str):
            variables[var_info["var_names"]] = var_type_name
        else:
            var_names = typing.cast(typing.Sequence[str], var_names)
            for var_name in var_names: 
                variables[var_name] = var_type_name
        

    return OrderedDict({var: types if isinstance(types, list) else [types] for var, types in variables.items()})


def extract_variables(predicate: typing.Union[typing.Sequence[tatsu.ast.AST], tatsu.ast.AST, None]) -> typing.List[str]:
    '''
    Recursively extract every variable referenced in the predicate (including inside functions 
    used within the predicate)
    '''
    if predicate is None:
        return []

    if isinstance(predicate, list) or isinstance(predicate, tuple):
        pred_vars = []
        for sub_predicate in predicate:
            pred_vars += extract_variables(sub_predicate)

        unique_vars = []
        for var in pred_vars:
            if var not in unique_vars:
                unique_vars.append(var)

        return unique_vars

    elif isinstance(predicate, tatsu.ast.AST):
        pred_vars = []
        for key in predicate:
            if key == "term":

                # Different structure for predicate args vs. function args
                if isinstance(predicate["term"], tatsu.ast.AST):
                    pred_vars += [predicate["term"]["arg"]]  # type: ignore 
                else:
                    pred_vars += [predicate["term"]]

            # We don't want to capture any variables within an (exists) or (forall) that's inside 
            # the preference, since those are not globally required -- see evaluate_predicate()
            elif key == "exists_args":
                continue

            elif key == "forall_args":
                continue

            elif key != "parseinfo":
                pred_vars += extract_variables(predicate[key])

        unique_vars = []
        for var in pred_vars:
            if var not in unique_vars:
                unique_vars.append(var)

        return unique_vars

    else:
        return []

def get_object_assignments(domain: str, variable_types: typing.Sequence[typing.Sequence[str]],
                           used_objects: typing.Union[None, typing.Container, typing.Iterable] = None):
    '''
    Given a room type / domain (few, medium, or many) and a list of lists of variable types,
    returns a list of every possible assignment of objects in the room to those types. For 
    instance, if variable_types is [(beachball, dodgeball), (bin,)], then this will return 
    every pair of objects consisting of one beachball or dodgeball and one bin.

    An optional used_objects argument specifies a list of objects that have already been assigned to
    a variable, and will be excluded from the returned assignments
    '''

    if used_objects is None:
        used_objects = []

    grouped_objects = []
    for sub_types in variable_types:
        objects = sum([OBJECTS_BY_ROOM_AND_TYPE[domain][var_type] for var_type in sub_types], [])
        grouped_objects.append([obj for obj in objects if obj not in used_objects])

    assignments = list(itertools.product(*grouped_objects))

    # Filter out any assignments that have duplicate objects
    filtered_assignments = list(filter(lambda assignment: len(set(assignment)) == len(assignment), assignments))

    return filtered_assignments

def ast_cache_key(ast: typing.Optional[tatsu.ast.AST], mapping: typing.Dict[str, str]) -> str:
    """
    Maps from a predicate / function and an object mapping to the key that represents them in the cache. 
    """
    ast_printer.reset_buffers()
    ast_printer.PARSE_DICT[ast_printer.PREFERENCES_KEY](ast)

    # flush the line buffer
    ast_printer._indent_print('', 0, ast_printer.DEFAULT_INCREMENT, None)

    ast_str = ' '.join(ast_printer.BUFFER if ast_printer.BUFFER is not None else [])
    mapping_str = ' '.join([f'{k}={mapping[k]}' for k in sorted(mapping.keys())])

    return ast_str, mapping_str

def is_type_or_color(variable: str):
    '''
    Returns whether the variable is a type or color
    '''
    return (variable in ALL_OBJECT_TYPES or variable in COLORS) and (variable not in NAMED_OBJECTS)

def get_object_types(obj: ObjectState):
    '''
    Return all the types to which an object belongs (including meta-types) as a set
    '''

    object_id = obj.object_id
    object_types = []

    for objects_by_type in OBJECTS_BY_ROOM_AND_TYPE.values():
        for object_type, objects in objects_by_type.items():
            if object_id in objects:
                object_types.append(object_type)

    return set(object_types)

def describe_preference(preference):
    '''
    Generate a natural language description of the given preference in plain language
    by recursively applying a set of rules.
    '''

    print(preference)
    rule = preference["parseinfo"].rule

    for key in preference.keys():
        print(key)
        describe_preference(preference[key])

PREDICATE_DESCRIPTIONS = {
    "above": "{0} is above {1}",
    "agent_crouches": "the agent is crouching",
    "agent_holds": "the agent is holding {0}",
    "in": "{1} is inside of {0}",
    "in_motion": "{0} is in motion",
    "on": "{1} is on {0}",
    "touch": "{0} touches {1}"
}

FUNCTION_DESCRIPTIONS = {
    "distance": "the distance between {0} and {1}"
}

class PreferenceDescriber():
    def __init__(self, preference):
        self.preference_name = preference["pref_name"]
        self.body = preference["pref_body"]["body"]

        self.variable_type_mapping = extract_variable_type_mapping(self.body["exists_vars"]["variables"])
        self.variable_type_mapping["agent"] = ["agent"]

        self.temporal_predicates = [func["seq_func"] for func in self.body["exists_args"]["body"]["then_funcs"]]

        self.engine = inflect.engine()

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

    def describe(self):
        print("\nDescribing preference:", self.preference_name)
        print("The variables required by this preference are:")
        for var, types in self.variable_type_mapping.items():
            print(f" - {var}: of type {self.engine.join(types, conj='or')}")

        description = ''

        for idx, predicate in enumerate(self.temporal_predicates):
            if idx == 0:
                prefix = f"\n[{idx}] First, "
            elif idx == len(self.temporal_predicates) - 1:
                prefix = f"\n[{idx}] Finally, "
            else:
                prefix = f"\n[{idx}] Next, "

            pred_type = self._type(predicate)
            if pred_type == "once":
                description = f"we need a single state where {self.describe_predicate(predicate['once_pred'])}."

            elif pred_type == "once-measure":
                description = f"we need a single state where {self.describe_predicate(predicate['once_measure_pred'])}."

                # TODO: describe which measurement is performed

            elif pred_type == "hold":
                description = f"we need a sequence of states where {self.describe_predicate(predicate['hold_pred'])}."

            elif pred_type == "hold-while":
                description = f"we need a sequence of states where {self.describe_predicate(predicate['hold_pred'])}."

                if isinstance(predicate["while_preds"], list):
                    while_desc = self.engine.join(['a state where (' + self.describe_predicate(pred) + ')' for pred in predicate['while_preds']])
                    description += f" During this sequence, we need {while_desc} (in that order)."
                else:
                    description += f" During this sequence, we need a state where ({self.describe_predicate(predicate['while_preds'])})."

            print(prefix + description)

    def describe_predicate(self, predicate) -> str:
        predicate_rule = predicate["parseinfo"].rule

        if predicate_rule == "predicate":

            name = predicate["pred_name"]
            variables = extract_variables(predicate)

            return PREDICATE_DESCRIPTIONS[name].format(*variables)

        elif predicate_rule == "super_predicate":
            return self.describe_predicate(predicate["pred"])

        elif predicate_rule == "super_predicate_not":
            return f"it's not the case that {self.describe_predicate(predicate['not_args'])}"

        elif predicate_rule == "super_predicate_and":
            return self.engine.join(["(" + self.describe_predicate(sub) + ")" for sub in predicate["and_args"]])

        elif predicate_rule == "super_predicate_or":
            return self.engine.join(["(" + self.describe_predicate(sub) + ")" for sub in predicate["or_args"]], conj="or")

        elif predicate_rule == "super_predicate_exists":
            variable_type_mapping = extract_variable_type_mapping(predicate["exists_vars"]["variables"])

            new_variables = []
            for var, types in variable_type_mapping.items():
                new_variables.append(f"an object {var} of type {self.engine.join(types, conj='or')}")

            return f"there exists {self.engine.join(new_variables)}, such that {self.describe_predicate(predicate['exists_args'])}"

        elif predicate_rule == "super_predicate_forall":
            variable_type_mapping = extract_variable_type_mapping(predicate["forall_vars"]["variables"])

            new_variables = []
            for var, types in variable_type_mapping.items():
                new_variables.append(f"object {var} of type {self.engine.join(types, conj='or')}")

            return f"for any {self.engine.join(new_variables)}, {self.describe_predicate(predicate['forall_args'])}"

        elif predicate_rule == "function_comparison":
            comparison_operator = predicate["comp"]["comp_op"]

            comp_arg_1 = predicate["comp"]["arg_1"]["arg"]
            if isinstance(comp_arg_1, tatsu.ast.AST):

                name = comp_arg_1["func_name"]
                variables = extract_variables(comp_arg_1)

                comp_arg_1 = FUNCTION_DESCRIPTIONS[name].format(*variables)  # type: ignore

            comp_arg_2 = predicate["comp"]["arg_2"]["arg"]
            if isinstance(comp_arg_1, tatsu.ast.AST):
                name = comp_arg_2["func_name"]
                variables = extract_variables(comp_arg_2)

                comp_arg_1 = FUNCTION_DESCRIPTIONS[name].format(*variables)

            if comparison_operator == "=":
                return f"{comp_arg_1} is equal to {comp_arg_2}"
            elif comparison_operator == "<":
                return f"{comp_arg_1} is less than {comp_arg_2}"
            elif comparison_operator == "<=":
                return f"{comp_arg_1} is less than or equal to {comp_arg_2}"
            elif comparison_operator == ">":
                return f"{comp_arg_1} is greater than {comp_arg_2}"
            elif comparison_operator == ">=":
                return f"{comp_arg_1} is greater than or equal to {comp_arg_2}"

        else:
            raise ValueError(f"Error: Unknown rule '{predicate_rule}'")

        return ''
        