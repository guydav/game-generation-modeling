import numpy as np
import typing


from config import BUILDING_TYPE, OBJECTS_BY_ROOM_AND_TYPE, PseudoObject, UNITY_PSEUDO_OBJECTS
from predicate_handler import ObjectState, _object_location, _vec3_dict_to_array, _extract_object_corners

MAX_BUILDINGS = 10


class BuildingHandler:
    active_buildings: typing.Set[str]  # buildings that are currently active
    active_objects: typing.Set[str]  # objects that have been held by the agent at some point or another
    domain: str
    max_buildings: int

    def __init__(self, domain: str, max_buildings: int = MAX_BUILDINGS):
        self.domain = domain
        self.max_buildings = max_buildings

        OBJECTS_BY_ROOM_AND_TYPE[domain][BUILDING_TYPE] = [f'building_{i}' for i in range(max_buildings)]

        self.active_buildings = set()
        self.active_objects = set()

    def get_active_buildings(self) -> typing.Set[str]:
        return self.active_buildings

    def process(self, state: typing.Dict[str, typing.Any]):
        # TODO (GD): implement the following logic:
        # if the currently held object isn't marked as active, mark it as active
        if state['agentStateChanged']:
            held_object = state['agentState']['heldObject']
            if held_object:
                self.active_objects.add(held_object)

        # from the objects updated at this state intersected with the valid objects:
        current_objects = set([o['objectId'] for o in state['objects']]).intersection_update(self.active_objects)
        for obj in current_objects:
            
        

        # if the object is in motion, do we mark it immediately as no longer in a building? 
        # or wait for it to settle? Probably the former
        # maintain collection of moving/held objects that we're monitoring?
        # if an object was in that collection and is no longer moving, check which objects it's touching
        # according to that, decide to either start it in a new building or add it to an existing building
        # Do we want to deal with merging buildings?
        # Mark active buildings at the end 
        pass


def _array_to_vec3_dict(array: typing.List[float]) -> typing.Dict[str, float]:
    return {'x': array[0], 'y': array[1], 'z': array[2]}


class BuildingPseudoObject(PseudoObject):
    building_objects: typing.Dict[str, ObjectState]  # a collection of the objects in the building
    min_corner: np.ndarray
    max_corner: np.ndarray
    
    def __init__(self, building_id: str):
        super().__init__(building_id, building_id, dict(x=0, y=0, z=0), dict(x=0, y=0, z=0), dict(x=0, y=0, z=0))
        self.building_objects = {}
        self.min_corner = np.zeros(3)
        self.max_corner = np.zeros(3)
        self.position_valid = False

    def add_object(self, obj: ObjectState):
        self.building_objects[obj['objectId']] = obj        
        obj_min, obj_max = _extract_object_corners(obj)

        if not self.position_valid:
            self.min_corner = obj_min
            self.max_corner = obj_max
            self.position_valid = True

        else:
            self.min_corner = np.minimum(obj_min, self.min_corner)  # type: ignore
            self.max_corner = np.maximum(obj_max, self.max_corner)  # type: ignore

        self._update_position_from_corners()
        
    def _update_position_from_corners(self) -> None:
        self.position = self.bboxCenter = _array_to_vec3_dict((self.min_corner + self.max_corner) / 2)  # type: ignore
        self.bboxExtents = _array_to_vec3_dict((self.max_corner - self.min_corner) / 2)  # type: ignore

    def remove_object(self, obj: ObjectState):
        if obj['objectId'] not in self.building_objects:
            raise ValueError(f'Object {obj["objectId"]} is not in building {self.name}')
        
        del self.building_objects[obj['objectId']]

        if len(self.building_objects) == 0:
            self.position_valid = False
        
        else:
            object_minima, object_maxima = list(zip(*[_extract_object_corners(curr_obj) 
                for curr_obj in self.building_objects.values()]))
            
            self.min_corner = np.min(object_minima, axis=0)  
            self.max_corner = np.max(object_maxima, axis=0)  
            self._update_position_from_corners()


