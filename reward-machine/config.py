import numpy as np
import typing


# ===================================================================================================

NORTH_WALL = 'north_wall'
SOUTH_WALL = 'south_wall'
EAST_WALL = 'east_wall'
WEST_WALL = 'west_wall'
NAMED_WALLS = [NORTH_WALL, SOUTH_WALL, EAST_WALL, WEST_WALL]

DOOR = 'door'

RUG = 'rug'
RUG_ALT_NAME = 'HighFrictionTrigger'


BUILDING_TYPE = 'building'

FEW_OBJECTS_ROOM = 'few'
MEDIUM_OBJECTS_ROOM = 'medium'
MANY_OBJECTS_ROOM = 'many'
ROOMS = [FEW_OBJECTS_ROOM, MEDIUM_OBJECTS_ROOM, MANY_OBJECTS_ROOM]

OBJECTS_SHARED_IN_ALL_ROOMS_BY_TYPE = {
    "alarm_clock": [
            "AlarmClock|-01.41|+00.60|+00.45"
        ],
    "bed": [
        "Bed|-02.46|00.00|-00.57"
    ],
    "blinds": [
        "Blinds|+02.29|+02.07|-03.18",
        "Blinds|-01.00|+02.07|-03.18"
    ],
    "book": [
        "Book|+02.83|+00.41|-00.01"
    ],
    "cd": [
        "CD|+02.99|+00.79|-00.37"
    ],
    "cellphone": [
        "CellPhone|+02.96|+00.79|-00.93"
    ],
    "chair": [
        "Chair|+02.73|00.00|-01.21",
        "Chair|+02.83|00.00|00.00"
    ],
    "credit_card": [
        "CreditCard|+02.99|+00.79|-01.24"
    ],
    "desk": [
        "Desk|+03.14|00.00|-01.41"
    ],
    "lamp": [
        "DeskLamp|+03.13|+00.79|-00.64"
    ],
    "desktop": [
        "Desktop|+03.10|+00.79|-01.24"
    ],
    "drawer": [
        "Drawer|-01.52|+00.14|+00.35",
        "Drawer|-01.52|+00.41|+00.35"
    ],
    "floor": [
        "Floor|+00.00|+00.00|+00.00"
    ],
    "key_chain": [
        "KeyChain|-01.62|+00.60|+00.41"
    ],
    "laptop": [
        "Laptop|+03.04|+00.79|-02.28"
    ],
    "main_light_switch": [
        "LightSwitch|-00.14|+01.33|+00.60"
    ],
    "mirror": [
        "Mirror|+00.45|+01.49|+00.62"
    ],
    "mug": [
        "Mug|+03.14|+00.79|-00.87"
    ],
    "pen": [
        "Pen|+03.02|+00.80|-01.85"
    ],
    "pencil": [
        "Pencil|+03.07|+00.79|-01.79"
    ],
    "pillow": [
        "Pillow|-02.45|+00.66|+00.10"
    ],
    "poster": [
        "Poster|+03.40|+01.70|-00.79",
        "Poster|+03.40|+01.86|-01.98"
    ],
    "shelf": [
        "Shelf|+00.62|+01.01|-02.82",
        "Shelf|+00.62|+01.51|-02.82",
        "Shelf|+03.13|+00.63|-00.56",
        "Shelf|+03.13|+00.63|-02.27",
        "Shelf|-02.97|+01.16|-01.72",
        "Shelf|-02.97|+01.16|-02.47",
        "Shelf|-02.97|+01.53|-01.72",
        "Shelf|-02.97|+01.53|-02.47"
    ],
    "side_table": [
        "SideTable|-01.52|00.00|+00.41"
    ],
    "watch": [
        "Watch|+03.07|+00.79|-00.45"
    ],
    "window": [
        "Window|+02.28|+00.93|-03.18",
        "Window|-01.02|+00.93|-03.19"
    ],
    "wall": [
        NORTH_WALL,
        SOUTH_WALL,
        EAST_WALL,
        WEST_WALL
    ],
    NORTH_WALL: [NORTH_WALL],
    SOUTH_WALL: [SOUTH_WALL],
    EAST_WALL: [EAST_WALL],
    WEST_WALL: [WEST_WALL],
    DOOR: [DOOR],
    RUG: [RUG]
}


OBJECTS_BY_ROOM_AND_TYPE = {
    FEW_OBJECTS_ROOM: {
        "cube_block": [
            "CubeBlock|+00.20|+00.10|-02.83",
            "CubeBlock|+00.20|+00.29|-02.83",
            "CubeBlock|-00.02|+00.10|-02.83",
            "CubeBlock|-00.02|+00.28|-02.83",
            "CubeBlock|-00.23|+00.28|-02.83",
            "CubeBlock|-00.24|+00.10|-02.83"
        ],
        "curved_wooden_ramp": [
            "CurvedRamp|-03.05|00.00|-02.78"
        ],
        "dodgeball": [
            "Dodgeball|-02.95|+01.29|-02.61",
            "Dodgeball|-02.97|+01.29|-02.28"
        ],
        "hexagonal_bin": [
            "GarbageCan|+00.95|-00.03|-02.68"
        ],
    },

    MEDIUM_OBJECTS_ROOM: {
        "basketball": [
            "BasketBall|-02.58|+00.12|-01.93"
        ],
        "beachball": [
            "Beachball|-02.93|+00.17|-01.99"
        ],
        "bridge_block": [
            "BridgeBlock|+00.63|+01.10|-02.91",
            "BridgeBlock|+01.03|+01.11|-02.88"
        ],
        "cube_block": [
            "CubeBlock|+00.50|+01.61|-02.91",
            "CubeBlock|+00.70|+01.61|-02.91"
        ],
        "cylindrical_block": [
            "CylinderBlock|+00.93|+01.61|-02.89",
            "CylinderBlock|+01.13|+01.61|-02.89"
        ],
        "dodgeball": [
            "Dodgeball|-02.60|+00.13|-02.18"
        ],
        "doggie_bed": [
            "DogBed|+02.30|00.00|-02.85"
        ],
        "flat_block": [
            "FlatRectBlock|+00.23|+01.66|-02.88",
            "FlatRectBlock|+00.24|+01.57|-02.89"
        ],
        "hexagonal_bin": [
            "GarbageCan|-02.79|-00.03|-02.67"
        ],
        "tall_cylindrical_block": [
            "LongCylinderBlock|+00.12|+01.19|-02.89",
            "LongCylinderBlock|+00.31|+01.19|-02.89"
        ],
        "pyramid_block": [
            "PyramidBlock|+00.93|+01.78|-02.89",
            "PyramidBlock|+01.13|+01.78|-02.89"
        ],
        "triangular_ramp": [
            "SmallSlide|-00.97|+00.20|-03.02"
        ],
        "teddy_bear": [
            "TeddyBear|-02.60|+00.60|-00.42"
        ],
    },

    MANY_OBJECTS_ROOM: {
        "beachball": [
            "Beachball|+02.29|+00.19|-02.88"
        ],
        "bridge_block": [
            "BridgeBlock|-02.92|+00.09|-02.52",
            "BridgeBlock|-02.92|+00.26|-02.52",
            "BridgeBlock|-02.92|+00.43|-02.52"
        ],
        "cube_block": [
            "CubeBlock|-02.96|+01.26|-01.72",
            "CubeBlock|-02.97|+01.26|-01.94",
            "CubeBlock|-02.99|+01.26|-01.49"
        ],
        "curved_wooden_ramp": [
            "CurvedRamp|-00.25|00.00|-02.98"
        ],
        "cylindrical_block": [
            "CylinderBlock|-02.95|+01.62|-01.95",
            "CylinderBlock|-02.97|+01.62|-01.50",
            "CylinderBlock|-03.02|+01.62|-01.73"
        ],
        "dodgeball": [
            "Dodgeball|+00.19|+01.13|-02.80",
            "Dodgeball|+00.44|+01.13|-02.80",
            "Dodgeball|+00.70|+01.11|-02.80"
        ],
        "doggie_bed": [
            "DogBed|+02.24|00.00|-02.85"
        ],
        "flat_block": [
            "FlatRectBlock|-02.93|+00.05|-02.84",
            "FlatRectBlock|-02.93|+00.15|-02.84",
            "FlatRectBlock|-02.93|+00.25|-02.84"
        ],
        "hexagonal_bin": [
            "GarbageCan|+00.75|-00.03|-02.74"
        ],
        "golfball": [
            "Golfball|+00.96|+01.04|-02.70",
            "Golfball|+01.05|+01.04|-02.70",
            "Golfball|+01.14|+01.04|-02.70"
        ],
        "tall_cylindrical_block": [
            "LongCylinderBlock|-02.82|+00.19|-02.09",
            "LongCylinderBlock|-02.93|+00.19|-01.93",
            "LongCylinderBlock|-02.94|+00.19|-02.24"
        ],
        "pillow": [
            "Pillow|-02.03|+00.68|-00.42",
        ],
        "pyramid_block": [
            "PyramidBlock|-02.95|+01.61|-02.20",
            "PyramidBlock|-02.95|+01.61|-02.66",
            "PyramidBlock|-02.96|+01.61|-02.44"
        ],
        "triangular_ramp": [
            "SmallSlide|-00.81|+00.14|-03.10",
            "SmallSlide|-01.31|+00.14|-03.10"
        ],
        "tall_rectangular_block": [
            "TallRectBlock|-02.95|+02.05|-02.31",
            "TallRectBlock|-02.95|+02.05|-02.52",
            "TallRectBlock|-02.95|+02.05|-02.72"
        ],
        "teddy_bear": [
            "TeddyBear|-01.93|+00.60|+00.07",
            "TeddyBear|-02.60|+00.60|-00.42"
        ],
        "triangle_block": [
            "TriangleBlock|-02.92|+01.23|-02.23",
            "TriangleBlock|-02.94|+01.23|-02.46",
            "TriangleBlock|-02.95|+01.23|-02.69"
        ],
    }
}

SPECIFIC_NAMED_OBJECTS_BY_ROOM = {
    FEW_OBJECTS_ROOM: {
        "blue_cube_block": ['CubeBlock|-00.02|+00.28|-02.83', 'CubeBlock|-00.24|+00.10|-02.83'],
        "blue_dodgeball": ['Dodgeball|-02.95|+01.29|-02.61'],
        "bottom_shelf": ["Shelf|+00.62|+01.01|-02.82"],
        "desk_shelf": ["Shelf|+03.13|+00.63|-00.56", "Shelf|+03.13|+00.63|-02.27"],
        "east_sliding_door": ["Window|+02.28|+00.93|-03.18"],
        "pink_dodgeball": ['Dodgeball|-02.97|+01.29|-02.28'],
        "sliding_door": ["Window|+02.28|+00.93|-03.18", "Window|-01.02|+00.93|-03.19"],
        "tan_cube_block": ['CubeBlock|+00.20|+00.10|-02.83', 'CubeBlock|-00.23|+00.28|-02.83'],
        "top_drawer": ["Drawer|-01.52|+00.41|+00.35"],  # bottom is "Drawer|-01.52|+00.14|+00.35"
        "top_shelf": ["Shelf|+00.62|+01.51|-02.82"],
        "yellow_cube_block": ['CubeBlock|+00.20|+00.29|-02.83', 'CubeBlock|-00.02|+00.10|-02.83'],
    },
    MEDIUM_OBJECTS_ROOM: {
        "blue_cube_block": ['CubeBlock|+00.50|+01.61|-02.91'],
        "bottom_shelf": ["Shelf|+00.62|+01.01|-02.82"],
        "desk_shelf": ["Shelf|+03.13|+00.63|-00.56", "Shelf|+03.13|+00.63|-02.27"],
        "east_sliding_door": ["Window|+02.28|+00.93|-03.18"],
        "red_dodgeball": ['Dodgeball|-02.60|+00.13|-02.18'],
        "red_pyramid_block": ['PyramidBlock|+00.93|+01.78|-02.89'],
        "sliding_door": ["Window|+02.28|+00.93|-03.18", "Window|-01.02|+00.93|-03.19"],
        "top_drawer": ["Drawer|-01.52|+00.41|+00.35"],  # bottom is "Drawer|-01.52|+00.14|+00.35"
        "top_shelf": ["Shelf|+00.62|+01.51|-02.82"],
        "yellow_cube_block": ['CubeBlock|+00.70|+01.61|-02.91'],
    },
    MANY_OBJECTS_ROOM: {
        "blue_cube_block": ['CubeBlock|-02.99|+01.26|-01.49'],
        "blue_dodgeball": ['Dodgeball|+00.19|+01.13|-02.80'],
        "blue_pyramid_block": ['PyramidBlock|-02.95|+01.61|-02.20'],
        "bottom_shelf": ['Shelf|+00.62|+01.01|-02.82'],
        "desk_shelf": ["Shelf|+03.13|+00.63|-00.56", "Shelf|+03.13|+00.63|-02.27"],
        "east_sliding_door": ["Window|+02.28|+00.93|-03.18"],
        "green_golfball": ['Golfball|+01.05|+01.04|-02.70'],  # Orange is 'Golfball|+00.96|+01.04|-02.70', white is 'Golfball|+01.14|+01.04|-02.70'
        "green_triangular_ramp": ['SmallSlide|-00.81|+00.14|-03.10'],  # tan/brown is 'SmallSlide|-01.31|+00.14|-03.10'
        "pink_dodgeball": ['Dodgeball|+00.70|+01.11|-02.80'],
        "red_dodgeball": ['Dodgeball|+00.44|+01.13|-02.80'],
        "red_pyramid_block": ['PyramidBlock|-02.96|+01.61|-02.44'],
        "sliding_door": ["Window|+02.28|+00.93|-03.18", ],
        "tan_cube_block": ['CubeBlock|-02.96|+01.26|-01.72'],
        "top_drawer": ["Drawer|-01.52|+00.41|+00.35"],  # bottom is "Drawer|-01.52|+00.14|+00.35"
        "top_shelf": ["Shelf|+00.62|+01.51|-02.82"],
        "yellow_cube_block": ['CubeBlock|-02.97|+01.26|-01.94'],
        "yellow_pyramid_block": ['PyramidBlock|-02.95|+01.61|-02.66'],
    }
}

OBJECT_ID_TO_SPECIFIC_NAME_BY_ROOM = {
    room: {specific_object: name}
    for room, room_specific_objects in SPECIFIC_NAMED_OBJECTS_BY_ROOM.items()
    for name, specific_objects in room_specific_objects.items()
    for specific_object in specific_objects
}


# Add the shared objects to each of the room domains
for room_type in OBJECTS_BY_ROOM_AND_TYPE:
    for object_type, object_list in OBJECTS_SHARED_IN_ALL_ROOMS_BY_TYPE.items():
        if object_type in OBJECTS_BY_ROOM_AND_TYPE[room_type]:
            OBJECTS_BY_ROOM_AND_TYPE[room_type][object_type].extend(object_list)
        else:
            OBJECTS_BY_ROOM_AND_TYPE[room_type][object_type] = object_list[:]

# A list of all objects that can be referred to directly as variables inside of a game
NAMED_OBJECTS = ["agent", "bed", "desk", "desktop", "door", "floor", "main_light_switch", NORTH_WALL, EAST_WALL, SOUTH_WALL, WEST_WALL]

# A list of all the colors, which as a hack will also be mapped to themselves, as though they were named objects
COLORS = ["black", "blue", "brown", "gray", "green", "light_blue", "orange",  "pink", "purple", "red", "tan", "white", "yellow"]
ORIENTATIONS = ["diagonal", "sideways", "upright", "upside_down"]
SIDES = ["back", "front", "front_left_corner", "left", "right"]

COLOR = 'color'
ORIENTATION = 'orientation'
SIDE = 'side'

NON_OBJECT_TYPES = [COLOR, ORIENTATION, SIDE]

# Meta types compile objects from many other types (e.g. both beachballs and dodgeballs are balls)
META_TYPES = {"ball": ["beachball", "basketball", "dodgeball", "golfball"],
              "block": ["bridge_block", "cube_block", "cylindrical_block", "flat_block", "pyramid_block", "tall_cylindrical_block",
                        "tall_rectangular_block", "triangle_block"],
              COLOR: COLORS,
              ORIENTATION: ORIENTATIONS,
              SIDE: SIDES}

TYPES_TO_META_TYPE = {sub_type: meta_type for meta_type, sub_types in META_TYPES.items() for sub_type in sub_types}

# List of types that are *not* included in "game_object" -- easier than listing out all the types that are
# GAME_OBJECT_EXCLUDED_TYPES = ["bed", "blinds", "desk", "desktop", "lamp", "drawer", "floor", "main_light_switch", "mirror",
#                               "poster", "shelf", "side_table", "window", "wall", "agent"]
GAME_OBJECT_EXCLUDED_TYPES = ["bed", "blinds", "desk", "floor", "main_light_switch", "mirror",
                              "poster", "shelf", "side_table", "window", "wall", "agent"]
GAME_OBJECT_EXCLUDED_TYPES.append("building")  # the fictional building type should not be included in game objects
GAME_OBJECT_EXCLUDED_TYPES += COLORS + ORIENTATIONS + SIDES
GAME_OBJECT_EXCLUDED_TYPES += list(META_TYPES.keys())

GAME_OBJECT = "game_object"

# Update the dictionary by mapping the agent and colors to themselves and grouping objects into meta types. Also group all
# of the objects that count as a "game_object"
for domain in ROOMS:
    OBJECTS_BY_ROOM_AND_TYPE[domain]["agent"] = ["agent"]
    for collection in COLORS, ORIENTATIONS, SIDES:
        OBJECTS_BY_ROOM_AND_TYPE[domain].update({item: [item] for item in collection})

    for meta_type, object_types in META_TYPES.items():
        OBJECTS_BY_ROOM_AND_TYPE[domain][meta_type] = []
        for object_type in object_types:
            if object_type in OBJECTS_BY_ROOM_AND_TYPE[domain]:
                OBJECTS_BY_ROOM_AND_TYPE[domain][meta_type] += OBJECTS_BY_ROOM_AND_TYPE[domain][object_type]

    OBJECTS_BY_ROOM_AND_TYPE[domain][GAME_OBJECT] = []
    for object_type in OBJECTS_BY_ROOM_AND_TYPE[domain]:
        if object_type not in GAME_OBJECT_EXCLUDED_TYPES:
            OBJECTS_BY_ROOM_AND_TYPE[domain][GAME_OBJECT] += OBJECTS_BY_ROOM_AND_TYPE[domain][object_type]

    OBJECTS_BY_ROOM_AND_TYPE[domain][GAME_OBJECT] = sorted(list(set(OBJECTS_BY_ROOM_AND_TYPE[domain][GAME_OBJECT])))

# A list of all object types (including meta types)
ALL_OBJECT_TYPES = list(set(list(OBJECTS_BY_ROOM_AND_TYPE[FEW_OBJECTS_ROOM].keys()) + \
                            list(OBJECTS_BY_ROOM_AND_TYPE[MEDIUM_OBJECTS_ROOM].keys()) + \
                            list(OBJECTS_BY_ROOM_AND_TYPE[MANY_OBJECTS_ROOM].keys())))

# ===================================================================================================

class PseudoObject:
    identifiers: typing.List[str]
    object_id: str
    object_type: str
    name: str
    bbox_center: np.ndarray
    bbox_extents: np.ndarray
    position: np.ndarray
    rotation: np.ndarray

    def __init__(self, object_id: str, object_type: str, name: str, position: np.ndarray,
        extents: np.ndarray, rotation: np.ndarray, alternative_names: typing.Optional[typing.List[str]] = None):

        self.object_id = object_id
        self.object_type = object_type
        self.name = name
        self.position = position
        self.bbox_center = position
        self.bbox_extents = extents
        self.rotation = rotation
        if alternative_names is None:
            alternative_names = []
        self.identifiers = [self.object_id, self.object_type] + alternative_names


    def __getitem__(self, item: typing.Any) -> typing.Any:
        if item in self.__dict__:
            return self.__dict__[item]

        raise ValueError(f'PsuedoObjects have only a name and an id, not a {item}')

    def __contains__(self, item):
        return item in self.__dict__


WALL_ID = 'FP326:StandardWallSize.021'
WALL_TYPE = 'wall'

DOOR_ID = 'FP326:StandardDoor1.019'
DOOR_TYPE = 'door'

DOOR_ID = 'FP326:StandardDoor1.019'
DOOR_TYPE = 'door'

RUG_ID = 'FP326:Rug'
RUG_TYPE = 'rug'

# TODO: I think the ceiling also might be one, and maybe the floor or some other fixed furniture?
# Wall width is about 0.15, ceiling height is about 2.7
UNITY_PSEUDO_OBJECTS = {
        NORTH_WALL: PseudoObject(WALL_ID, WALL_TYPE, NORTH_WALL, position=np.array([0.1875, 1.35, 0.675]), extents=np.array([3.2875, 1.35, 0.075]), rotation=np.zeros(3)),           # has the door
        SOUTH_WALL: PseudoObject(WALL_ID, WALL_TYPE, SOUTH_WALL, position=np.array([0.1875, 1.35, -3.1]), extents=np.array([3.2875, 1.35, 0.075]), rotation=np.zeros(3)),            # has the window
        EAST_WALL: PseudoObject(WALL_ID, WALL_TYPE, EAST_WALL, position=np.array([3.475, 1.35, 1.2125]), extents=np.array([0.075, 1.35, 1.8875]), rotation=np.array([0, 90, 0])),   # has the desk
        WEST_WALL: PseudoObject(WALL_ID, WALL_TYPE, WEST_WALL, position=np.array([-3.1, 1.35, -1.2125]), extents=np.array([0.075, 1.35, 1.8875]), rotation=np.array([0, 90, 0])),   # has the bed

        DOOR: PseudoObject(DOOR_ID, DOOR_TYPE, DOOR, position=np.array([0.448, 1.35, 0.675]), extents=np.array([0.423, 1.35, 0.075]), rotation=np.zeros(3)),

        RUG: PseudoObject(RUG_ID, RUG_TYPE, RUG, position=np.array([-1.485, 0.05, -1.673]), extents=np.array([1.338, 0.05, 0.926]), rotation=np.array([0, 90, 0]), alternative_names=[RUG_ALT_NAME]),
}
