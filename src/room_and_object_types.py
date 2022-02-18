from collections import defaultdict
import argparse

from pandas import options

# -- CATEGOTIES -- 

AGENT = 'agent'
ANY_OBJECT = 'any_object'
BALLS = 'balls'
BLOCKS = 'blocks'
BUILDING = 'building'
COLORS = 'colors'
EMPTY_OBJECT = 'empty_object'
FURNITURE = 'furniture'
LARGE_OBJECTS = 'large_objects'
MEDIUM_OBJECTS = 'medium_objects'
OTHER_OBJECTS = 'other_objects'
RAMPS = 'ramps'
RECEPTACLES = 'receptacles'
ROOM_FEATURES = 'room_features'
SMALL_OBJECTS = 'small_objects'

# -- OBJECT TYPES -- 

# AGENT
AGENT = 'agent'

# ANY_OBJECT
GAME_OBJECT = 'game_object'

# BALLS
BALL = 'ball'
BASKETBALL = 'basketball'
BEACHBALL = 'beachball'
BLUE_DODGEBALL = 'blue_dodgeball'
DODGEBALL = 'dodgeball'
GOLFBALL = 'golfball'
GREEN_GOLFBALL = 'green_golfball'
PINK_DODGEBALL = 'pink_dodgeball'
RED_DODGEBALL = 'red_dodgeball'

# BLOCKS
BLOCK = 'block'
BRIDGE_BLOCK = 'bridge_block'
CUBE_BLOCK = 'cube_block'
CYLINDRICAL_BLOCK = 'cylindrical_block'
FLAT_BLOCK = 'flat_block'
PYRAMID_BLOCK = 'pyramid_block'
TALL_CYLINDRICAL_BLOCK = 'tall_cylindrical_block'
TALL_RECTANGULAR_BLOCK = 'tall_rectangular_block'
TRIANGLE_BLOCK = 'triangle_block'
TAN_CUBE_BLOCK = 'tan_cube_block'
RED_PYRAMID_BLOCK = 'red_pyramid_block'
BLUE_CUBE_BLOCK = 'blue_cube_block'
BLUE_PYRAMID_BLOCK = 'blue_pyramid_block'
YELLOW_PYRAMID_BLOCK = 'yellow_pyramid_block'
YELLOW_CUBE_BLOCK = 'yellow_cube_block'

# COLORS
COLOR = 'color'
BLUE = 'blue'
BROWN = 'brown'
GRAY = 'gray'
GREEN = 'green'
LIGHT_BLUE = 'light_blue'
ORANGE = 'orange'
PINK = 'pink'
PURPLE = 'purple'
RED = 'red'
TAN = 'tan'
WHITE = 'white'
YELLOW = 'yellow'

# EMPTY_OBJECT
EMPTY_OBJECT_OBJ = ''

# FURNITURE
BED = 'bed'
BLINDS = 'blinds'
CHAIR = 'chair'
DESK = 'desk'
DESK_SHELF = 'desk_shelf'
DRAWER = 'drawer'
MAIN_LIGHT_SWITCH = 'main_light_switch'
DESKTOP = 'desktop'
TOP_DRAWER = 'top_drawer'
SIDE_TABLE = 'side_table'

# BUILDING
BUILDING = 'building'

# LARGE_OBJECTS
BOOK = 'book'
LAPTOP = 'laptop'
PILLOW = 'pillow'
TEDDY_BEAR = 'teddy_bear'

# RAMPS
CURVED_WOODEN_RAMP = 'curved_wooden_ramp'
TRIANGULAR_RAMP = 'triangular_ramp'
GREEN_TRIANGULAR_RAMP = 'green_triangular_ramp'

# RECEPTACLES
DOGGIE_BED = 'doggie_bed'
HEXAGONAL_BIN = 'hexagonal_bin'

# ROOM_FEATURES
DOOR = 'door'
FLOOR = 'floor'
RUG = 'rug'
SHELF = 'shelf'
BOTTOM_SHELF = 'bottom_shelf'
TOP_SHELF = 'top_shelf'
SLIDING_DOOR = 'sliding_door'
EAST_SLIDING_DOOR = 'east_sliding_door'
SOUTH_WEST_CORNER = 'south_west_corner'
WALL = 'wall'
NORTH_WALL = 'north_wall'
SOUTH_WALL = 'south_wall'
WEST_WALL = 'west_wall'

# SMALL_OBJECTS
ALARM_CLOCK = 'alarm_clock'
CD = 'cd'
CELLPHONE = 'cellphone'
CREDIT_CARD = 'credit_card'
KEY_CHAIN = 'key_chain'
LAMP = 'lamp'
MUG = 'mug'
PEN = 'pen'
PENCIL = 'pencil'
WATCH = 'watch'


# -- MAPPINGS --

CATEGORIES_TO_TYPES = {
    AGENT: (
        AGENT,
    ),
    ANY_OBJECT: (
        GAME_OBJECT,
    ),
    BALLS: (
        BALL, BASKETBALL, BEACHBALL, BLUE_DODGEBALL, DODGEBALL,
        GOLFBALL, GREEN_GOLFBALL, PINK_DODGEBALL, RED_DODGEBALL,
    ),
    BLOCKS: (
        BLOCK, BRIDGE_BLOCK, CUBE_BLOCK, CYLINDRICAL_BLOCK, FLAT_BLOCK,
        PYRAMID_BLOCK, TALL_CYLINDRICAL_BLOCK, TRIANGLE_BLOCK, TAN_CUBE_BLOCK, RED_PYRAMID_BLOCK,
        BLUE_CUBE_BLOCK, BLUE_PYRAMID_BLOCK, YELLOW_PYRAMID_BLOCK, YELLOW_CUBE_BLOCK,
    ),
    COLORS: (
        COLOR, BLUE, BROWN, GRAY, GREEN, 
        LIGHT_BLUE, ORANGE, PINK, PURPLE, RED, 
        TAN, WHITE, YELLOW,
    ),
    EMPTY_OBJECT: (
        EMPTY_OBJECT_OBJ,
    ),
    FURNITURE: (
        BED, BLINDS, CHAIR, DESK, DESK_SHELF,  # TODO: does chair qualify as funiture? since it's movable
        DRAWER, MAIN_LIGHT_SWITCH, DESKTOP, TOP_DRAWER, SIDE_TABLE,
    ),
    BUILDING: (
        BUILDING,
    ),
    LARGE_OBJECTS: (
        BOOK, LAPTOP, PILLOW, TEDDY_BEAR,
    ),
    RAMPS: (
        CURVED_WOODEN_RAMP, TRIANGULAR_RAMP, GREEN_TRIANGULAR_RAMP,
    ),
    RECEPTACLES: (
        DOGGIE_BED, HEXAGONAL_BIN,
    ),
    ROOM_FEATURES: (
        DOOR, FLOOR, RUG, SHELF, BOTTOM_SHELF,
        TOP_SHELF, SLIDING_DOOR, EAST_SLIDING_DOOR, SOUTH_WEST_CORNER, WALL,
        NORTH_WALL, SOUTH_WALL, WEST_WALL,
    ),
    SMALL_OBJECTS: (
        ALARM_CLOCK, CD, CELLPHONE, CREDIT_CARD, KEY_CHAIN,
        LAMP, MUG, PEN, PENCIL, WATCH,
    ),
}

TYPES_TO_CATEGORIES = {type_name: cat for cat, type_names in CATEGORIES_TO_TYPES.items() for type_name in type_names}


FEW = 'Few'
MEDIUM = 'Medium'
MANY = 'Many'
ROOM_NAMES = (FEW, MEDIUM, MANY)

FULL_ROOMS_UNCATEGORIZED_OBJECTS = 'uncategorized_objects'



FULL_ROOMS_TO_OBJECTS = {
    FEW: {
        FULL_ROOMS_UNCATEGORIZED_OBJECTS: {
            AGENT: 1, 
            GAME_OBJECT: 1,
            BUILDING: 1,
            EMPTY_OBJECT_OBJ: 1,
        },
        BALLS: {
            BALL: 1,
            DODGEBALL: {BLUE: 1, PINK: 1},
        },
        BLOCKS: {
            BLOCK: 1,
            CUBE_BLOCK: {BLUE: 2, TAN: 2, YELLOW: 2, },
        },
        COLORS: {
            BLUE: 1,
            BROWN: 1,
            COLOR: 1,
            GREEN: 1,
            ORANGE: 1,
            PINK: 1,
            PURPLE: 1,
            RED: 1,
            TAN: 1,
            WHITE: 1,
            YELLOW: 1,
        },
        FURNITURE: {
            BED: 1,
            BLINDS: 2,
            CHAIR: 2,
            DESK: 1,
            DESK_SHELF: 1,
            DESKTOP: 1,
            DRAWER: 1,
            MAIN_LIGHT_SWITCH: 1,
            SIDE_TABLE: 1,
            TOP_DRAWER: 1,
        },
        LARGE_OBJECTS: {
            BOOK: 1,
            LAPTOP: 1,
            PILLOW: 1,
        },
        RAMPS: {
            CURVED_WOODEN_RAMP: 1,
        },
        RECEPTACLES: {
            HEXAGONAL_BIN: 1,
        },
        ROOM_FEATURES: {
            BOTTOM_SHELF: 1,
            DOOR: 1,
            EAST_SLIDING_DOOR: 1,
            FLOOR: 1,
            NORTH_WALL: 1,
            RUG: 1,
            SHELF: 4,
            SLIDING_DOOR: 2,
            SOUTH_WALL: 1,
            SOUTH_WEST_CORNER: 1,
            TOP_SHELF: 1,
            WALL: 1,
            WEST_WALL: 1,
        },
        SMALL_OBJECTS: {
            ALARM_CLOCK: 1,
            CD: 1,
            CELLPHONE: 1,
            CREDIT_CARD: 1,
            KEY_CHAIN: 1,
            LAMP: 1,
            MUG: 1,
            PEN: 1,
            PENCIL: 1,
            WATCH: 1,
        },
    },
    MEDIUM: {
        FULL_ROOMS_UNCATEGORIZED_OBJECTS: {
            AGENT: 1, 
            GAME_OBJECT: 1,
            BUILDING: 1,
            EMPTY_OBJECT_OBJ: 1,
        },
        BALLS: {
            BALL: 1,
            BASKETBALL: 1,
            BEACHBALL: 1,
            DODGEBALL: {RED: 1, },
        },
        BLOCKS: {
            BLOCK: 1,
            BRIDGE_BLOCK: {GREEN: 1, TAN: 1},
            CUBE_BLOCK: {BLUE: 1, YELLOW: 1, },
            CYLINDRICAL_BLOCK: {GREEN: 1, LIGHT_BLUE: 1},
            FLAT_BLOCK: {GRAY: 1, YELLOW: 1},
            PYRAMID_BLOCK: {RED: 1, YELLOW: 1, },
            TALL_CYLINDRICAL_BLOCK: {TAN: 1, YELLOW: 1},
        },
        COLORS: {
            BLUE: 1,
            BROWN: 1,
            COLOR: 1,
            GREEN: 1,
            GRAY: 1,
            LIGHT_BLUE: 1,
            ORANGE: 1,
            PINK: 1,
            PURPLE: 1,
            RED: 1,
            TAN: 1,
            WHITE: 1,
            YELLOW: 1,
        },
        FURNITURE: {
            BED: 1,
            BLINDS: 2,
            CHAIR: 2,
            DESK: 1,
            DESK_SHELF: 1,
            DESKTOP: 1,
            DRAWER: 1,
            MAIN_LIGHT_SWITCH: 1,
            SIDE_TABLE: 1,
            TOP_DRAWER: 1,
            
        },
        LARGE_OBJECTS: {
            BOOK: 1,
            LAPTOP: 1,
            PILLOW: 1,
            TEDDY_BEAR: 1,
        },
        RAMPS: {
            TRIANGULAR_RAMP: 1,
        },
        RECEPTACLES: {
            DOGGIE_BED: 1,
            HEXAGONAL_BIN: 1,
        },
        ROOM_FEATURES: {
            BOTTOM_SHELF: 1,
            DOOR: 1,
            EAST_SLIDING_DOOR: 1,
            FLOOR: 1,
            NORTH_WALL: 1,
            RUG: 1,
            SHELF: 2,
            SLIDING_DOOR: 2,
            SOUTH_WALL: 1,
            SOUTH_WEST_CORNER: 1,
            TOP_SHELF: 1,
            WALL: 1,
            WEST_WALL: 1,
        },
        SMALL_OBJECTS: {
            ALARM_CLOCK: 1,
            CD: 1,
            CELLPHONE: 1,
            CREDIT_CARD: 1,
            KEY_CHAIN: 1,
            LAMP: 1,
            MUG: 1,
            PEN: 1,
            PENCIL: 1,
            WATCH: 1,
        },
    },
    MANY: {
        FULL_ROOMS_UNCATEGORIZED_OBJECTS: {
            AGENT: 1, 
            GAME_OBJECT: 1,
            BUILDING: 1,
            EMPTY_OBJECT_OBJ: 1,
        },
        BALLS: {
            BALL: 1,
            BEACHBALL: 1,
            DODGEBALL: {BLUE: 1, PINK: 1, RED: 1, },
            GOLFBALL: {GREEN: 1, ORANGE: 1, WHITE: 1, },
        },
        BLOCKS: {
            BLOCK: 1,
            BRIDGE_BLOCK: {PINK: 1, TAN: 1, WHITE: 1, },
            CUBE_BLOCK: {BLUE: 1, TAN: 1, YELLOW: 1, },
            CYLINDRICAL_BLOCK: {GREEN: 1, LIGHT_BLUE: 1, TAN: 1},
            FLAT_BLOCK: {GRAY: 1, TAN: 1, YELLOW: 1},
            PYRAMID_BLOCK: {BLUE: 1, RED: 1, YELLOW: 1, },
            TALL_CYLINDRICAL_BLOCK: {GREEN: 1, TAN: 1, YELLOW: 1},
            TALL_RECTANGULAR_BLOCK: {BLUE: 1, GREEN: 1, TAN: 1, },
            TRIANGLE_BLOCK: {BLUE: 1, GREEN: 1, TAN: 1, },
        },
        COLORS: {
            BLUE: 1,
            BROWN: 1,
            COLOR: 1,
            GREEN: 1,
            GRAY: 1,
            LIGHT_BLUE: 1,
            ORANGE: 1,
            PINK: 1,
            PURPLE: 1,
            RED: 1,
            TAN: 1,
            WHITE: 1,
            YELLOW: 1,
        },
        FURNITURE: {
            BED: 1,
            BLINDS: 2,
            CHAIR: 2,
            DESK: 1,
            DESK_SHELF: 1,
            DESKTOP: 1,
            DRAWER: 1,
            MAIN_LIGHT_SWITCH: 1,
            SIDE_TABLE: 1,
            TOP_DRAWER: 1,
        },
        LARGE_OBJECTS: {
            BOOK: 1,
            LAPTOP: 1,
            PILLOW: 2,
            TEDDY_BEAR: 2,
        },
        RAMPS: {
            CURVED_WOODEN_RAMP: 1,
            TRIANGULAR_RAMP: {GREEN: 1, TAN: 1},
        },
        RECEPTACLES: {
            HEXAGONAL_BIN: 1,
        },
        ROOM_FEATURES: {
            BOTTOM_SHELF: 1,
            DOOR: 1,
            EAST_SLIDING_DOOR: 1,
            FLOOR: 1,
            NORTH_WALL: 1,
            RUG: 1,
            SHELF: 2,
            SLIDING_DOOR: 2,
            SOUTH_WALL: 1,
            SOUTH_WEST_CORNER: 1,
            TOP_SHELF: 1,
            WALL: 1,
            WEST_WALL: 1,
        },
        SMALL_OBJECTS: {
            ALARM_CLOCK: 1,
            CD: 1,
            CELLPHONE: 1,
            CREDIT_CARD: 1,
            KEY_CHAIN: 1,
            LAMP: 1,
            MUG: 1,
            PEN: 1,
            PENCIL: 1,
            WATCH: 1,
        },
    },
}


# ROOMS_TO_AVAILABLE_OBJECTS = {
#     FEW: set([
#         'agent',
#         'ball', 'dodgeball', 'blue_dodgeball', 'pink_dodgeball', 
#         'block', 'cube_block', 'yellow_cube_block', 'blue_cube_block', 'tan_cube_block',
#         'color', 'blue', 'brown', 'green', 'orange', 'pink', 'purple', 'red', 'tan', 'white', 'yellow', 
#         'bed', 'blinds', 'chair', 'desk', 'desk_shelf', 'drawer', 'main_light_switch', 'desktop', 'top_drawer', 'side_table',
#         'building',  'game_object', '',
#         'curved_wooden_ramp', 'hexagonal_bin',
#         'laptop', 'pillow', 
#         'door', 'floor', 'rug', 'shelf', 'top_shelf', 'bottom_shelf', 'sliding_door', 'south_west_corner', 'east_sliding_door', 'wall', 'north_wall', 'south_wall', 'west_wall',
#         'alarm_clock', 'book', 'cd', 'cellphone',  'credit_card', 'key_chain', 'lamp',  'mug', 'pen', 'pencil', 'watch',
#     ]),
#     MEDIUM: set([
#         'agent',
#         'ball', 'basketball', 'beachball', 'dodgeball', 'red_dodgeball',
#         'block', 'bridge_block', 'cube_block', 'cylindrical_block', 'flat_block', 'pyramid_block', 'tall_cylindrical_block', 
#         'yellow_pyramid_block', 'red_pyramid_block', 'yellow_cube_block', 'blue_cube_block',
#         'color', 'blue', 'brown', 'green', 'orange', 'pink', 'purple', 'red', 'tan', 'white', 'yellow', 
#         'bed', 'blinds', 'chair', 'desk', 'desk_shelf', 'drawer', 'main_light_switch', 'desktop', 'top_drawer', 'side_table',
#         'building',  'game_object', '',
#         'doggie_bed', 'hexagonal_bin',  'triangular_ramp',
#         'laptop', 'pillow', 'teddy_bear',
#         'door', 'floor', 'rug', 'shelf', 'top_shelf', 'bottom_shelf', 'sliding_door', 'south_west_corner', 'east_sliding_door', 'wall', 'north_wall', 'south_wall', 'west_wall',
#         'alarm_clock', 'book', 'cd', 'cellphone',  'credit_card', 'key_chain', 'lamp',  'mug', 'pen', 'pencil', 'watch',
#     ]),
#     MANY: set([
#         'agent',
#         'ball', 'beachball', 'dodgeball', 'blue_dodgeball', 'pink_dodgeball', 'red_dodgeball', 'golfball', 'green_golfball',
#         'block', 'bridge_block', 'cube_block', 'cylindrical_block', 'flat_block', 'pyramid_block', 'tall_cylindrical_block', 'triangle_block',
#         'yellow_pyramid_block', 'red_pyramid_block', 'blue_pyramid_block', 'yellow_cube_block', 'blue_cube_block', 'tan_cube_block',
#         'color', 'blue', 'brown', 'green', 'orange', 'pink', 'purple', 'red', 'tan',  'white', 'yellow', 
#         'bed', 'blinds', 'chair', 'desk', 'desk_shelf', 'drawer', 'main_light_switch', 'desktop', 'top_drawer', 'side_table',
#         'building',  'game_object', '',
#         'curved_wooden_ramp', 'doggie_bed', 'green_triangular_ramp', 'hexagonal_bin',  'triangular_ramp',
#         'laptop', 'pillow', 'teddy_bear',
#         'door', 'floor', 'rug', 'shelf', 'top_shelf', 'bottom_shelf', 'sliding_door', 'south_west_corner', 'east_sliding_door', 'wall', 'north_wall', 'south_wall', 'west_wall',
#         'alarm_clock', 'book', 'cd', 'cellphone',  'credit_card', 'key_chain', 'lamp',  'mug', 'pen', 'pencil', 'watch',
#     ]),
# }

ROOMS_TO_AVAILABLE_OBJECTS = {
    FEW: set([
        EMPTY_OBJECT_OBJ, AGENT, BUILDING, GAME_OBJECT,
        BALL, DODGEBALL, BLUE_DODGEBALL, PINK_DODGEBALL,
        BLOCK, CUBE_BLOCK, BLUE_CUBE_BLOCK, TAN_CUBE_BLOCK, YELLOW_CUBE_BLOCK,
        COLOR, BLUE, BROWN, COLOR, GREEN, ORANGE, PINK, PURPLE, RED, TAN, WHITE, YELLOW,
        BED, BLINDS, CHAIR, DESK, DESK_SHELF, DESKTOP, DRAWER, MAIN_LIGHT_SWITCH, SIDE_TABLE, TOP_DRAWER,
        BOOK, LAPTOP, PILLOW,
        CURVED_WOODEN_RAMP,
        HEXAGONAL_BIN,
        BOTTOM_SHELF, DOOR, EAST_SLIDING_DOOR, FLOOR, NORTH_WALL, RUG, SHELF, SLIDING_DOOR, SOUTH_WALL, SOUTH_WEST_CORNER, TOP_SHELF, WALL, WEST_WALL,
        ALARM_CLOCK, CD, CELLPHONE, CREDIT_CARD, KEY_CHAIN, LAMP, MUG, PEN, PENCIL, WATCH,
    ]),
    MEDIUM: set([
        EMPTY_OBJECT_OBJ, AGENT, BUILDING, GAME_OBJECT,
        BALL, BASKETBALL, BEACHBALL, DODGEBALL, RED_DODGEBALL,
        BLOCK, BRIDGE_BLOCK, CUBE_BLOCK, BLUE_CUBE_BLOCK, YELLOW_CUBE_BLOCK, CYLINDRICAL_BLOCK, FLAT_BLOCK, PYRAMID_BLOCK, RED_PYRAMID_BLOCK, YELLOW_PYRAMID_BLOCK, TALL_CYLINDRICAL_BLOCK,
        COLOR, BLUE, BROWN, COLOR, GREEN, ORANGE, PINK, PURPLE, RED, TAN, WHITE, YELLOW,
        BED, BLINDS, CHAIR, DESK, DESK_SHELF, DESKTOP, DRAWER, MAIN_LIGHT_SWITCH, SIDE_TABLE, TOP_DRAWER,
        BOOK, LAPTOP, PILLOW, TEDDY_BEAR,
        TRIANGULAR_RAMP,
        DOGGIE_BED, HEXAGONAL_BIN,
        BOTTOM_SHELF, DOOR, EAST_SLIDING_DOOR, FLOOR, NORTH_WALL, RUG, SHELF, SLIDING_DOOR, SOUTH_WALL, SOUTH_WEST_CORNER, TOP_SHELF, WALL, WEST_WALL,
        ALARM_CLOCK, CD, CELLPHONE, CREDIT_CARD, KEY_CHAIN, LAMP, MUG, PEN, PENCIL, WATCH,
    ]),
    MANY: set([
        EMPTY_OBJECT_OBJ, AGENT, BUILDING, GAME_OBJECT,
        BALL, BEACHBALL, DODGEBALL, BLUE_DODGEBALL, PINK_DODGEBALL, RED_DODGEBALL, GOLFBALL, GREEN_GOLFBALL,
        BLOCK, BRIDGE_BLOCK, CUBE_BLOCK, BLUE_CUBE_BLOCK, TAN_CUBE_BLOCK, YELLOW_CUBE_BLOCK, CYLINDRICAL_BLOCK, FLAT_BLOCK, PYRAMID_BLOCK, BLUE_PYRAMID_BLOCK, RED_PYRAMID_BLOCK, YELLOW_PYRAMID_BLOCK, TALL_CYLINDRICAL_BLOCK, TRIANGLE_BLOCK,
        COLOR, BLUE, BROWN, COLOR, GREEN, ORANGE, PINK, PURPLE, RED, TAN, WHITE, YELLOW,
        BED, BLINDS, CHAIR, DESK, DESK_SHELF, DESKTOP, DRAWER, MAIN_LIGHT_SWITCH, SIDE_TABLE, TOP_DRAWER,
        BOOK, LAPTOP, PILLOW, TEDDY_BEAR,
        CURVED_WOODEN_RAMP, TRIANGULAR_RAMP, GREEN_TRIANGULAR_RAMP,
        DOGGIE_BED, HEXAGONAL_BIN,
        BOTTOM_SHELF, DOOR, EAST_SLIDING_DOOR, FLOOR, NORTH_WALL, RUG, SHELF, SLIDING_DOOR, SOUTH_WALL, SOUTH_WEST_CORNER, TOP_SHELF, WALL, WEST_WALL,
        ALARM_CLOCK, CD, CELLPHONE, CREDIT_CARD, KEY_CHAIN, LAMP, MUG, PEN, PENCIL, WATCH,
    ]),
}


# print('FULL_ROOMS_TO_OBJECTS = {')
# for room, available_objects in ROOMS_TO_AVAILABLE_OBJECTS.items():
#     print(f'{" " * 4}{room.upper()}: {{')
#     room_types_to_categories = defaultdict(dict)
#     for obj_type in available_objects:
#         split_index = obj_type.find('_')
#         added = False

#         if split_index != -1:
#             potential_color = obj_type[:split_index]
#             colorless_type = obj_type[split_index + 1:]

#             if potential_color in CATEGORIES_TO_TYPES[COLORS] and potential_color != COLOR:
#                 if colorless_type not in room_types_to_categories[TYPES_TO_CATEGORIES[colorless_type]] or \
#                     isinstance(room_types_to_categories[TYPES_TO_CATEGORIES[colorless_type]][colorless_type], int):
#                     room_types_to_categories[TYPES_TO_CATEGORIES[colorless_type]][colorless_type] = []

#                 room_types_to_categories[TYPES_TO_CATEGORIES[colorless_type]][colorless_type].append(potential_color)
#                 added = True

#         if not added and obj_type not in room_types_to_categories[TYPES_TO_CATEGORIES[obj_type]]:
#             room_types_to_categories[TYPES_TO_CATEGORIES[obj_type]][obj_type] = 1

#     for key in sorted(room_types_to_categories.keys()):
#         if key in (EMPTY_OBJECT, AGENT, ANY_OBJECT, BUILDING):
#             continue

#         print(f'{" " * 8}{key.upper()}: {{')
#         for obj, count in sorted(room_types_to_categories[key].items(), key=lambda x: x[0]):

#             if isinstance (count, list):
#                 count = f'{{]{", ".join([c.upper() + ": 1" for c in sorted(count)])}, }}' 
            
#             print(f'{" " * 12}{obj.upper()}: {count},')

#         print(f'{" " * 8}}},')

#     print(f'{" " * 4}}},')
# print('}')


# print('ROOMS_TO_AVAILABLE_OBJECTS = {')
# for room, full_available_objects in FULL_ROOMS_TO_OBJECTS.items():
#     print(f'{" " * 4}{room.upper()}: set([')
#     for category, object_dict in full_available_objects.items():
#         category_objects = []
#         if category == COLORS:
#             category_objects.append(COLOR)

#         for key, value in sorted(object_dict.items(), key=lambda x: x[0]):
#             category_objects.append(key.upper()),
#             if isinstance(value, (list, tuple)):
#                 category_objects.extend([f'{color}_{key}'.upper() for color in value])

#         print(f'{" " * 8}{", ".join(category_objects)},')
            
#     print('    ]),')

# print('}')


def get_room_objects_naive(start_token, skip_categories, skip_types, separator, break_after_category=False, verbose=False):
    room_object_strs = {}
    for room, room_data in FULL_ROOMS_TO_OBJECTS.items():
        if verbose: print(f'\nRoom type {room}:')

        total_buffer = []
        for category, category_data in room_data.items():
            if category in skip_categories:
                continue

            category_buffer = []

            for obj_type, count in category_data.items():
                if obj_type in skip_types:
                    continue

                if isinstance(count, int):
                    if count == 1:
                        category_buffer.append(obj_type)
                    else:
                        category_buffer.append(f'{count} {obj_type}')

                elif isinstance(count, dict):
                    for color, color_count in count.items():
                        type_with_color = f'{color}_{obj_type}'
                        if color_count == 1:
                            category_buffer.append(type_with_color)
                        else:
                            category_buffer.append(f'{color_count} {type_with_color}')

                else:
                    raise ValueError(f'Unknown count type: {count}')

            category_str = separator.join(category_buffer)
            if break_after_category:
                category_str += '\n'

            else:
                category_str += ' '

            total_buffer.append(category_str)

        room_str = f'{start_token}: {" ".join(total_buffer)}'
        if verbose: print(room_str)
        room_object_strs[f"{room.lower()}_objects"] = room_str

    return room_object_strs


def get_room_objects_categories(start_token, skip_categories, skip_types, separator, break_after_category=False, verbose=False):
    room_object_strs = {}
    for room, room_data in FULL_ROOMS_TO_OBJECTS.items():
        if verbose: print(f'\nRoom type {room}:')

        total_buffer = []
        for category, category_data in room_data.items():
            if category in skip_categories:
                continue

            category_buffer = []

            for obj_type, count in category_data.items():
                if obj_type in skip_types:
                    continue

                if isinstance(count, int):
                    if count == 1:
                        category_buffer.append(obj_type)
                    else:
                        category_buffer.append(f'{count} {obj_type}')

                elif isinstance(count, dict):
                    for color, color_count in count.items():
                        type_with_color = f'{color}_{obj_type}'
                        if color_count == 1:
                            category_buffer.append(type_with_color)
                        else:
                            category_buffer.append(f'{color_count} {type_with_color}')

                else:
                    raise ValueError(f'Unknown count type: {count}')

            category_str = f'({category}): {separator.join(category_buffer)}'
            if break_after_category:
                category_str += '\n'

            else:
                category_str += ' '

            total_buffer.append(category_str)

        room_str = f'{start_token}: {" ".join(total_buffer)}'
        if verbose: print(room_str)
        room_object_strs[f"{room.lower()}_objects"] = room_str

    return room_object_strs


def get_room_objects_colors(start_token, skip_categories, skip_types, separator, break_after_category=False, verbose=False):
    room_object_strs = {}
    for room, room_data in FULL_ROOMS_TO_OBJECTS.items():
        print(f'\nRoom type {room}:')

        total_buffer = []
        for category, category_data in room_data.items():
            if category in skip_categories:
                continue

            category_buffer = []

            for obj_type, count in category_data.items():
                if obj_type in skip_types:
                    continue

                if isinstance(count, int):
                    if count == 1:
                        category_buffer.append(obj_type)
                    else:
                        category_buffer.append(f'{count} {obj_type}')

                elif isinstance(count, dict):
                    type_buffer = []
                    for color, color_count in count.items():
                        
                        if color_count == 1:
                            type_buffer.append(color)
                        else:
                            type_buffer.append(f'{color_count} {color}')

                    category_buffer.append(f'{obj_type} ({separator.join(type_buffer)})')

                else:
                    raise ValueError(f'Unknown count type: {count}')

            category_str = f'({category}): {separator.join(category_buffer)}'
            if break_after_category:
                category_str += '\n'

            else:
                category_str += ' '

            total_buffer.append(category_str)

        room_str = f'{start_token}: {" ".join(total_buffer)}'
        if verbose: print(room_str)
        room_object_strs[f"{room.lower()}_objects"] = room_str

    return room_object_strs


NAIVE_MODE = 'naive'
CATEGORY_MODE = 'categories'
COLOR_MODE = 'colors'
MODES_TO_FUNCTIONS = {
    NAIVE_MODE: get_room_objects_naive,
    CATEGORY_MODE: get_room_objects_categories,
    COLOR_MODE: get_room_objects_colors,
}
DEFAULT_START_TOKEN = '[CONTENTS]'
DEFAULT_CATEGORIES_TO_SKIP = (FULL_ROOMS_UNCATEGORIZED_OBJECTS, COLORS)
DEFAULT_TYPES_TO_SKIP = (BALL, BLOCK)
    
def get_room_contents(mode, start_token=DEFAULT_START_TOKEN, skip_categories=DEFAULT_CATEGORIES_TO_SKIP, 
                      skip_types=DEFAULT_TYPES_TO_SKIP, separator=', ', break_after_category=False, verbose=False):

    out = MODES_TO_FUNCTIONS[mode](start_token, skip_categories, skip_types, separator, break_after_category, verbose)

    return out


def foo(**kwargs):
    print(kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start-token', default=DEFAULT_START_TOKEN)
    parser.add_argument('-m', '--mode', choices=list(MODES_TO_FUNCTIONS.keys()))
    parser.add_argument('-c', '--skip-categories', nargs='+', default=DEFAULT_CATEGORIES_TO_SKIP)
    parser.add_argument('-t', '--skip-types', nargs='+', default=DEFAULT_TYPES_TO_SKIP)
    parser.add_argument('--separator', default=', ')
    parser.add_argument('-b', '--break-after-category', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    get_room_contents(**args.__dict__)

