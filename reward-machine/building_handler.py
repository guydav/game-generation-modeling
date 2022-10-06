import numpy as np
import typing


from config import BUILDING_TYPE, OBJECTS_BY_ROOM_AND_TYPE, PseudoObject, UNITY_PSEUDO_OBJECTS
from predicate_handler import ObjectState, _pred_in_motion, _vec3_dict_to_array, _extract_object_corners

MAX_BUILDINGS = 10


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


class BuildingHandler:
    active_buildings: typing.Set[str]  # buildings that are currently active
    building_ids: typing.List[str]
    building_valid_objects: typing.Set[str]  # objects that have been held by the agent at some point or another
    objects_to_buildings: typing.Dict[str, str]
    recently_moved_objects: typing.Set[str] 
    domain: str
    max_buildings: int

    def __init__(self, domain: str, max_buildings: int = MAX_BUILDINGS):
        self.domain = domain
        self.max_buildings = max_buildings

        self.building_ids = [f'building_{i}' for i in range(max_buildings)]
        OBJECTS_BY_ROOM_AND_TYPE[domain][BUILDING_TYPE] = self.building_ids
        UNITY_PSEUDO_OBJECTS.update({f'building_{i}': BuildingPseudoObject(f'building_{i}') for i in range(max_buildings)})

        self.active_buildings = set()
        self.building_valid_objects = set()
        self.objects_to_buildings = {}
        self.recently_moved_objects = set()

    def get_active_buildings(self) -> typing.Set[str]:
        return self.active_buildings

    def process(self, state: typing.Dict[str, typing.Any]):
        # if the currently held object isn't marked as active, mark it as active
        agent = state['agentState']
        held_object = None
        if state['agentStateChanged']:
            held_object = agent['heldObject']
            if held_object:
                self.building_valid_objects.add(held_object)

        # from the objects updated at this state intersected with the valid objects:
        current_objects = set([o['objectId'] for o in state['objects']]).intersection_update(self.building_valid_objects)
        for obj in current_objects:  # type: ignore
            obj_id = obj['objectId']
            
            # if the object is in motion, do we mark it immediately as no longer in a building? 
            # or wait for it to settle? Let's try the former
            if _pred_in_motion(agent, [obj]) or obj == held_object:
                self.recently_moved_objects.add(obj_id)
                if obj_id in self.objects_to_buildings:
                    UNITY_PSEUDO_OBJECTS[self.objects_to_buildings[obj_id]].remove_object(obj_id)  # type: ignore
                    self.objects_to_buildings.pop(obj_id)

            # maintain collection of moving/held objects that we're monitoring?
            # if an object was in that collection and is no longer moving, check which objects it's touching
            elif obj_id in self.recently_moved_objects:
                self.recently_moved_objects.remove(obj_id)
                touched_object_ids = list(filter(lambda o: o in self.building_valid_objects, obj['touchingObjects']))
                
                # Doesn't touch any valid objects, create a new building
                if not touched_object_ids:
                    found = False
                    building_id = ''
                    for building_id in self.building_ids:
                        if building_id not in self.active_buildings:
                            found = True
                            break

                    if not found:
                        raise ValueError('No more buildings available')

                    UNITY_PSEUDO_OBJECTS[building_id].add_object(obj)  # type: ignore
                    self.active_buildings.add(building_id)
                    self.objects_to_buildings[obj_id] = building_id

                else:
                    touched_buildings = set([self.objects_to_buildings[o] for o in touched_object_ids])
                    # TODO (GD): what happens if the object touches multiple buildings? merge?
                    if len(touched_buildings) < 1:
                        raise ValueError('Object touches valid objects but no buildings found, this should never happen')

                    # touch a single building, add the object to that building
                    elif len(touched_buildings) == 1:
                        building_id = touched_buildings.pop()
                        UNITY_PSEUDO_OBJECTS[building_id].add_object(obj) # type: ignore
                        self.objects_to_buildings[obj_id] = building_id

                    # touch two buildings, merge them
                    elif len(touched_buildings) == 2:
                        building_id1, building_id2 = touched_buildings
                        min_building_id = min(building_id1, building_id2)
                        max_building_id = max(building_id1, building_id2)
                        min_building = typing.cast(BuildingPseudoObject, UNITY_PSEUDO_OBJECTS[min_building_id])
                        max_building = typing.cast(BuildingPseudoObject, UNITY_PSEUDO_OBJECTS[max_building_id])
                        max_building_objects = max_building.building_objects.copy()

                        for obj_id, obj in max_building_objects.items():
                            max_building.remove_object(obj)
                            min_building.add_object(obj)
                            self.objects_to_buildings[obj_id] = min_building_id

                    else:  # more than two buildings, do we still merge them?
                        raise ValueError('Object touches more than two buildings, should decide what to do here')



