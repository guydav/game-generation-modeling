import numpy as np



# ===================================================================================================

# TODO: this should also be conditional on which room we're in
OBJECTS_BY_TYPE = {
        "ball": [
                'Dodgeball|+00.70|+01.11|-02.80',
                'Dodgeball|+00.44|+01.13|-02.80',
                'Dodgeball|+00.19|+01.13|-02.80',
                'Beachball|+02.29|+00.19|-02.88',
                'Golfball|+01.05|+01.04|-02.70',
                'Golfball|+00.96|+01.04|-02.70',
                'Golfball|+01.14|+01.04|-02.70'
        ],
        "block": [
                'TallRectBlock|-02.95|+02.05|-02.52',
                'TallRectBlock|-02.95|+02.05|-02.72',
                'TallRectBlock|-02.95|+02.05|-02.31',
                'LongCylinderBlock|-02.82|+00.19|-02.09',
                'LongCylinderBlock|-02.94|+00.19|-02.24',
                'LongCylinderBlock|-02.93|+00.19|-01.93',
                'CylinderBlock|-02.95|+01.62|-01.95',
                'CylinderBlock|-02.97|+01.62|-01.50',
                'CylinderBlock|-03.02|+01.62|-01.73',
                'CubeBlock|-02.97|+01.26|-01.94',
                'CubeBlock|-02.96|+01.26|-01.72',
                'CubeBlock|-02.99|+01.26|-01.49',
                'PyramidBlock|-02.95|+01.61|-02.20',
                'PyramidBlock|-02.96|+01.61|-02.44',
                'PyramidBlock|-02.95|+01.61|-02.66',
                'FlatRectBlock|-02.93|+00.15|-02.84',
                'FlatRectBlock|-02.93|+00.05|-02.84',
                'FlatRectBlock|-02.93|+00.25|-02.84',
                'TriangleBlock|-02.92|+01.23|-02.23',
                'TriangleBlock|-02.95|+01.23|-02.69',
                'TriangleBlock|-02.94|+01.23|-02.46',
                'BridgeBlock|-02.92|+00.43|-02.52',
                'BridgeBlock|-02.92|+00.26|-02.52',
                'BridgeBlock|-02.92|+00.09|-02.52'
        ],
        "hexagonal_bin": ['GarbageCan|+00.75|-00.03|-02.74'],
        "wall": ["left-wall-1", "right-wall-1", "front-wall-1", "back-wall-1"]
}

# A list of all objects that can be referred to directly as variables inside of a game
NAMED_OBJECTS = ["agent", "desk", "bed"]

OBJECTS_BY_TYPE.update({obj: [obj] for obj in NAMED_OBJECTS})


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