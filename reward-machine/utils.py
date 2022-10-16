from collections import OrderedDict, namedtuple
import inflect
import itertools
import numpy as np
import tatsu
import tatsu.ast
import typing

from config import OBJECTS_BY_ROOM_AND_TYPE


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
    camera_local_roation: np.ndarray   # w, x, y, z
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
            camera_local_roation=_vec_dict_to_array(state_dict['cameraLocalRotation']),
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
            force_action=state_dict['force_action'],
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

def get_object_assignments(domain: str, variable_types: typing.Sequence[typing.Sequence[str]]):
    '''
    Given a room type / domain (few, medium, or many) and a list of lists of variable types,
    returns a list of every possible assignment of objects in the room to those types. For 
    instance, if variable_types is [(beachball, dodgeball), (bin,)], then this will return 
    every pair consisting of one beachball or dodgeball and one bin
    '''

    grouped_objects = [sum([OBJECTS_BY_ROOM_AND_TYPE[domain][var_type] for var_type in sub_types], []) for sub_types in variable_types]
    assignments = list(itertools.product(*grouped_objects))

    return assignments


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
        