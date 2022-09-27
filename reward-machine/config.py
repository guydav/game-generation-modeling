import numpy as np
import typing


# ===================================================================================================

NORTH_WALL = 'north_wall'
SOUTH_WALL = 'south_wall'
EAST_WALL = 'east_wall'
WEST_WALL = 'west_wall'

FEW_OBJECTS_ROOM = 'few'
MEDIUM_OBJECTS_ROOM = 'medium'
MANY_OBJECTS_ROOM = 'many'

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
    ]
}


OBJECTS_BY_ROOM_AND_TYPE = {
    "few": {
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

    "medium": {
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

    "many": {
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
        "pyramid_block": [
            "PyramidBlock|-02.95|+01.61|-02.20",
            "PyramidBlock|-02.95|+01.61|-02.66",
            "PyramidBlock|-02.96|+01.61|-02.44"
        ],
        "triangular_ramp": [
            "SmallSlide|-00.81|+00.14|-03.10",
            "SmallSlide|-01.31|+00.14|-03.10"
        ],
        "tall_rect_block": [
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

for room_type in OBJECTS_BY_ROOM_AND_TYPE:
    OBJECTS_BY_ROOM_AND_TYPE[room_type].update(OBJECTS_SHARED_IN_ALL_ROOMS_BY_TYPE)

# A list of all objects that can be referred to directly as variables inside of a game
NAMED_OBJECTS = ["agent", "desk", "bed"]

# Meta types compile objects from many other types (e.g. both beachballs and dodgeballs are balls)
META_TYPES = {"ball": ["beachball", "basketball", "dodgeball", "golfball"],
              "block": ["bridge_block", "cube_block", "cylindrical_block", "flat_block", "pyramid_block", "tall_cylindrical_block",
                        "tall_rect_block", "triangle_block"]}

# Update the dictionary by mapping named objects to themselves and grouping objects into meta types
for domain in ["few", "medium", "many"]:
    OBJECTS_BY_ROOM_AND_TYPE[domain].update({obj: [obj] for obj in NAMED_OBJECTS})

    for meta_type, object_types in META_TYPES.items():
        OBJECTS_BY_ROOM_AND_TYPE[domain][meta_type] = []
        for object_type in object_types:
            if object_type in OBJECTS_BY_ROOM_AND_TYPE[domain]:
                OBJECTS_BY_ROOM_AND_TYPE[domain][meta_type] += OBJECTS_BY_ROOM_AND_TYPE[domain][object_type]


class PseudoObject:
        object_id: str
        name: str
        position: typing.Dict[str, float]
        extents: typing.Dict[str, float]
        rotation: typing.Dict[str, float]

        def __init__(self, object_id: str, name: str, position: typing.Dict[str, float], 
            extents: typing.Dict[str, float], rotation: typing.Dict[str, float]):

            self.objectId = object_id
            self.name = name
            self.position = position
            self.bboxCenter = position
            self.bboxExtents = extents
            self.rotation = rotation


        def __getitem__(self, item):
            if item in self.__dict__:
                return self.__dict__[item]

            raise ValueError(f'PsuedoObjects have only a name and an id, not a {item}')
        
        def __contains__(self, item):
            return item in self.__dict__

WALL_ID = 'FP302:StandardWallSize'
WALL_NAME = 'FP326:StandardWallSize.021'

# TODO: I think the ceiling also might be one, and maybe the floor or some other fixed furniture?
# Wall width is about 0.15, ceiling height is about 2.7
UNITY_PSEUDO_OBJECTS = {
        NORTH_WALL: PseudoObject(WALL_ID, WALL_NAME, position=dict(x=0.1875, y=1.35, z=0.675), extents=dict(x=3.2875, y=1.35, z=0.075), rotation=dict(x=0, y=0, z=0)),
        SOUTH_WALL: PseudoObject(WALL_ID, WALL_NAME, position=dict(x=0.1875, y=1.35, z=-3.1), extents=dict(x=3.2875, y=1.35, z=0.075), rotation=dict(x=0, y=0, z=0)),
        EAST_WALL: PseudoObject(WALL_ID, WALL_NAME, position=dict(x=3.475, y=1.35, z=-1.2125), extents=dict(x=0.075, y=1.35, z=1.8875), rotation=dict(x=0, y=90, z=0)),
        WEST_WALL: PseudoObject(WALL_ID, WALL_NAME, position=dict(x=-3.1, y=1.35, z=-1.2125), extents=dict(x=0.075, y=1.35, z=1.8875), rotation=dict(x=0, y=90, z=0)),
}  


# WALL_INFO = {'isToggled': False, 
#              'bboxCenter': {'y': 0, 'x': 0, 'z': 0},
#              'isOpen': False,
#              'bboxExtents': {'x': 0, 'z': 0, 'y': 0},
#              'touchingObjects': [], 
#              'rotation': {'z': 0, 'x': 0, 'y': 0},
#              'angularVelocity': {'z': 0, 'x': 0, 'y': 0},
#              'position': {'y': 0, 'z': 0, 'x': 0},
#              'velocity': {'x': 0, 'y': 0, 'z': 0},
#              'objectType': 'wall', 
#              'name': WALL_NAME, 
#              'objectId': WALL_ID,
#              'isBroken': False}


# ===================================================================================================

STATIC_OBJECTS = {"desk": {"name": "desk", "position": [20, 20, 0], "velocity": [0, 0, 0],
                           "objectType": "desk", "color": "brown"}}

SAMPLE_TRAJECTORY = [# Starting state: nothing held
                     {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [4, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "red"},

                               "golfball-1": {"name": "golfball-1", "poisition": [6, 3, 0], "velocity": [0, 0, 0],
                                              "objectType": "golfball", "color": "white"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [9, 9, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "left-wall-1": {"name": "left-wall-1", "position": [0, 10, 10],
                                               "velocity": [0, 0, 0], "objectType": "wall"},

                               "right-wall-1": {"name": "right-wall-1", "position": [20, 10, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "front-wall-1": {"name": "front-wall-1", "position": [10, 0, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "back-wall-1": {"name": "back-wall-1", "position": [10, 20, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "agent": {"name": "agent", "position": [0, 0, 0], "velocity": [0, 0, 0],
                                         "is_crouching": False, "holding": None, "objectType": "agent"},
                              },

                      "game_start": True,
                      "game_over": False},

                     # State 2: agent moves closer to red ball
                     {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [4, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "red"},

                               "golfball-1": {"name": "golfball-1", "poisition": [6, 3, 0], "velocity": [0, 0, 0],
                                              "objectType": "golfball", "color": "white"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [9, 9, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "left-wall-1": {"name": "left-wall-1", "position": [0, 10, 10],
                                               "velocity": [0, 0, 0], "objectType": "wall"},

                               "right-wall-1": {"name": "right-wall-1", "position": [20, 10, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "front-wall-1": {"name": "front-wall-1", "position": [10, 0, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "back-wall-1": {"name": "back-wall-1", "position": [10, 20, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "agent": {"name": "agent", "position": [3, 3, 0], "velocity": [0, 0, 0],
                                         "is_crouching": False, "holding": None, "objectType": "agent"},
                              },

                      "game_start": False,
                      "game_over": False},

                     # State 3: red ball picked up
                     {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [4, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "red"},

                               "golfball-1": {"name": "golfball-1", "poisition": [6, 3, 0], "velocity": [0, 0, 0],
                                              "objectType": "golfball", "color": "white"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [9, 9, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "left-wall-1": {"name": "left-wall-1", "position": [0, 10, 10],
                                               "velocity": [0, 0, 0], "objectType": "wall"},

                               "right-wall-1": {"name": "right-wall-1", "position": [20, 10, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "front-wall-1": {"name": "front-wall-1", "position": [10, 0, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "back-wall-1": {"name": "back-wall-1", "position": [10, 20, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "agent": {"name": "agent", "position": [3, 3, 0], "velocity": [0, 0, 0],
                                         "is_crouching": False, "holding": "red-dodgeball-1", "objectType": "agent"},
                              },

                      "game_start": False,
                      "game_over": False},

                     # State 4: agent moves to [5, 5, 0]
                     {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [4, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "red"},

                               "golfball-1": {"name": "golfball-1", "poisition": [6, 3, 0], "velocity": [0, 0, 0],
                                              "objectType": "golfball", "color": "white"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [9, 9, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "left-wall-1": {"name": "left-wall-1", "position": [0, 10, 10],
                                               "velocity": [0, 0, 0], "objectType": "wall"},

                               "right-wall-1": {"name": "right-wall-1", "position": [20, 10, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "front-wall-1": {"name": "front-wall-1", "position": [10, 0, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "back-wall-1": {"name": "back-wall-1", "position": [10, 20, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "agent": {"name": "agent", "position": [5, 5, 0], "velocity": [0, 0, 0],
                                         "is_crouching": False, "holding": "red-dodgeball-1", "objectType": "agent"},
                              },

                      "game_start": False,
                      "game_over": False},

                     # State 5: ball released towards bin-1
                     {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [4, 4, 0],
                                                    "velocity": [2, 2, 0], "objectType": "ball",
                                                    "color": "red"},

                               "golfball-1": {"name": "golfball-1", "poisition": [6, 3, 0], "velocity": [0, 0, 0],
                                              "objectType": "golfball", "color": "white"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [9, 9, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "left-wall-1": {"name": "left-wall-1", "position": [0, 10, 10],
                                               "velocity": [0, 0, 0], "objectType": "wall"},

                               "right-wall-1": {"name": "right-wall-1", "position": [20, 10, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "front-wall-1": {"name": "front-wall-1", "position": [10, 0, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "back-wall-1": {"name": "back-wall-1", "position": [10, 20, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "agent": {"name": "agent", "position": [5, 5, 0], "velocity": [0, 0, 0],
                                         "is_crouching": False, "holding": None, "objectType": "agent"},
                              },

                      "game_start": False,
                      "game_over": False},

                     # State 6: ball travels towards bin-1, agent crouches and moves back to [3, 3, 0]
                     {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [6, 6, 0],
                                                    "velocity": [2, 2, 0], "objectType": "ball",
                                                    "color": "red"},

                               "golfball-1": {"name": "golfball-1", "poisition": [6, 3, 0], "velocity": [0, 0, 0],
                                              "objectType": "golfball", "color": "white"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [9, 9, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "left-wall-1": {"name": "left-wall-1", "position": [0, 10, 10],
                                               "velocity": [0, 0, 0], "objectType": "wall"},

                               "right-wall-1": {"name": "right-wall-1", "position": [20, 10, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "front-wall-1": {"name": "front-wall-1", "position": [10, 0, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "back-wall-1": {"name": "back-wall-1", "position": [10, 20, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "agent": {"name": "agent", "position": [3, 3, 0], "velocity": [0, 0, 0],
                                         "is_crouching": True, "holding": None, "objectType": "agent"},
                              },

                      "game_start": False,
                      "game_over": False},

                     # State 7: ball travels towards bin-1, agent picks up bin-1
                     {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [8, 8, 0],
                                                    "velocity": [2, 2, 0], "objectType": "ball",
                                                    "color": "red"},

                               "golfball-1": {"name": "golfball-1", "poisition": [6, 3, 0], "velocity": [0, 0, 0],
                                              "objectType": "golfball", "color": "white"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [9, 9, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "left-wall-1": {"name": "left-wall-1", "position": [0, 10, 10],
                                               "velocity": [0, 0, 0], "objectType": "wall"},

                               "right-wall-1": {"name": "right-wall-1", "position": [20, 10, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "front-wall-1": {"name": "front-wall-1", "position": [10, 0, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "back-wall-1": {"name": "back-wall-1", "position": [10, 20, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "agent": {"name": "agent", "position": [3, 3, 0], "velocity": [0, 0, 0],
                                         "is_crouching": True, "holding": "hexagonal-bin-1", "objectType": "agent"},
                              },

                      "game_start": False,
                      "game_over": False},

                     # State 8: ball reaches bin-1 but has velocity still, agent stands up and drops bin-1
                     {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [9, 9, 0],
                                                    "velocity": [0.2, 0.2, 0], "objectType": "ball",
                                                    "color": "red"},

                               "golfball-1": {"name": "golfball-1", "poisition": [6, 3, 0], "velocity": [0, 0, 0],
                                              "objectType": "golfball", "color": "white"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [9, 9, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "left-wall-1": {"name": "left-wall-1", "position": [0, 10, 10],
                                               "velocity": [0, 0, 0], "objectType": "wall"},

                               "right-wall-1": {"name": "right-wall-1", "position": [20, 10, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "front-wall-1": {"name": "front-wall-1", "position": [10, 0, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "back-wall-1": {"name": "back-wall-1", "position": [10, 20, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "agent": {"name": "agent", "position": [3, 3, 0], "velocity": [0, 0, 0],
                                         "is_crouching": False, "holding": None, "objectType": "agent"},
                              },

                      "game_start": False,
                      "game_over": False},

                     # State 9: ball stops moving in bin
                     {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [9, 9, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "red"},

                               "golfball-1": {"name": "golfball-1", "poisition": [6, 3, 0], "velocity": [0, 0, 0],
                                              "objectType": "golfball", "color": "white"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [9, 9, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "left-wall-1": {"name": "left-wall-1", "position": [0, 10, 10],
                                               "velocity": [0, 0, 0], "objectType": "wall"},

                               "right-wall-1": {"name": "right-wall-1", "position": [20, 10, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "front-wall-1": {"name": "front-wall-1", "position": [10, 0, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "back-wall-1": {"name": "back-wall-1", "position": [10, 20, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "agent": {"name": "agent", "position": [3, 3, 0], "velocity": [0, 0, 0],
                                         "is_crouching": False, "holding": None, "objectType": "agent"},
                              },

                      "game_start": False,
                      "game_over": False},

                     # State 10: game ends
                     {"objects": {"blue-dodgeball-1": {"name": "blue-dodgeball-1", "position": [4, 0, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball", 
                                                    "color": "blue"},

                               "pink-dodgeball-1": {"name": "pink-dodgeball-1", "position": [0, 4, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "pink"},

                               "red-dodgeball-1": {"name": "red-dodgeball-1", "position": [9, 9, 0],
                                                    "velocity": [0, 0, 0], "objectType": "ball",
                                                    "color": "red"},

                               "golfball-1": {"name": "golfball-1", "poisition": [6, 3, 0], "velocity": [0, 0, 0],
                                              "objectType": "golfball", "color": "white"},

                               "hexagonal-bin-1": {"name": "hexagonal-bin-1", "position": [9, 9, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "hexagonal-bin-2": {"name": "hexagonal-bin-2", "position": [10, 15, 0],
                                                    "velocity": [0, 0, 0], "objectType": "bin"},

                               "left-wall-1": {"name": "left-wall-1", "position": [0, 10, 10],
                                               "velocity": [0, 0, 0], "objectType": "wall"},

                               "right-wall-1": {"name": "right-wall-1", "position": [20, 10, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "front-wall-1": {"name": "front-wall-1", "position": [10, 0, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "back-wall-1": {"name": "back-wall-1", "position": [10, 20, 10],
                                                "velocity": [0, 0, 0], "objectType": "wall"},

                               "agent": {"name": "agent", "position": [3, 3, 0], "velocity": [0, 0, 0],
                                         "is_crouching": False, "holding": None, "objectType": "agent"},
                              },

                      "game_start": False,
                      "game_over": True},

                    ]

# Add the static objects to each state in the trajectory
for state in SAMPLE_TRAJECTORY:
    for obj, info in STATIC_OBJECTS.items():
        state["objects"][obj] = info