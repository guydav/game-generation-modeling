from dataclasses import dataclass
import tatsu
import tatsu.grammars
import typing
import pathlib
import pytest
import sys

# from compile_predicate_statistics import CommonSensePredicateStatistics
from compile_predicate_statistics_split_args import CommonSensePredicateStatisticsSplitArgs as CommonSensePredicateStatistics
from test_games import TEST_GAME_LIBRARY
from utils import get_project_dir


@dataclass
class ExpectedCompletionsTestCase:
    test_id: str
    game_key: str
    predicate_selector: typing.List[typing.Union[str, int]]
    mapping: typing.Dict[str, typing.List[str]]
    expected_intervals: typing.Dict[typing.Tuple[str, typing.Tuple[str, ...]], typing.List[typing.List[int]]]
    cache_rules: typing.Optional[typing.List[str]] = None
    cache_dir_relative_path: str = '/reward-machine/caches'
    debug: bool = False   # in case we want to have an optional flag to check somewhere further in



EXPECTED_COMPLETION_TEST_CASES = [
    ExpectedCompletionsTestCase(
        test_id='test (and (in_motion ?b) (not (agent_holds ?b)))',
        game_key='test-ball-from-bed',
        predicate_selector=[4, 1, 'preferences', 0, 'definition', 'forall_pref', 'preferences', 'pref_body', 'body', 'exists_args', 'then_funcs', 1, 'seq_func', 'hold_pred'],
        mapping={'?b': ['ball'], '?h': ['hexagonal_bin']},
        expected_intervals={
            ('IhOkh1l3TBY9JJVubzHx-createGame-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[1270, 1285]],
            ('IvoZWi01FO2uiNpNHyci-freePlay-rerecorded', ('?b->Dodgeball|+00.19|+01.13|-02.80',)): [[418, 457], [472, 530], [545, 598], [615, 669], [694, 753]],
            ('vfh1MTEQorWXKy8jOP1x-gameplay-attempt-2-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61',)): [[1024, 1025], [1103, 1151]],
            ('IvoZWi01FO2uiNpNHyci-preCreateGame-rerecorded', ('?b->Golfball|+00.96|+01.04|-02.70',)): [[651, 819]],
            ('vfh1MTEQorWXKy8jOP1x-gameplay-attempt-2-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[1203, 1236]],
            ('FyGQn1qJCLTLU1hfQfZ2-preCreateGame-rerecorded', ('?b->Golfball|+01.14|+01.04|-02.70',)): [[665, 672], [797, 916], [1150, 1226], [1292, 1330], [1379, 1394], [1430, 1485], [1575, 1631], [1637, 1659], [1695, 1696], [1697, 1719], [2087, 2142]],
            ('6XD5S6MnfzAPQlsP7k30-freePlay-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[210, 251], [307, 350], [379, 428], [453, 518], [551, 601], [621, 664], [687, 725], [764, 839], [885, 941], [957, 1028], [1051, 1092], [1126, 1177], [1212, 1271], [1277, 1330], [1372, 1404], [1444, 1486], [1536, 1617], [1652, 1702], [1728, 1771], [1842, 1916], [1966, 1999], [2065, 2081]],
            ('WtZpe3LQFZiztmh7pBBC-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.44|+01.13|-02.80',)): [[495, 537], [766, 810], [844, 937], [960, 1006], [1035, 1113], [1133, 1255], [1287, 1333], [1341, 1385], [1415, 1440]],
            ('NJUY0YT1Pq6dZXsmw0wE-gameplay-attempt-2-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61',)): [[2603, 2630], [2662, 2680], [2706, 2707], [2728, 2730], [2731, 2731], [2752, 2792], [2850, 2866], [2867, 2879], [2896, 2908], [2949, 2995], [3019, 3070], [3078, 3108]],
            ('IhOkh1l3TBY9JJVubzHx-createGame-rerecorded', ('?b->Dodgeball|+00.19|+01.13|-02.80',)): [[1391, 1405]],
            ('IhOkh1l3TBY9JJVubzHx-freePlay-rerecorded', ('?b->Beachball|+02.29|+00.19|-02.88',)): [[209, 235]],
            ('7r4cgxJHzLJooFaMG1Rd-gameplay-attempt-1-rerecorded', ('?b->Beachball|+02.29|+00.19|-02.88',)): [[264, 266]],
            ('Tcfpwc8v8HuKRyZr5Dyc-createGame-rerecorded', ('?b->Dodgeball|-02.60|+00.13|-02.18',)): [[2, 61]],
            ('Tcfpwc8v8HuKRyZr5Dyc-createGame-rerecorded', ('?b->Beachball|-02.93|+00.17|-01.99',)): [[0, 19]],
            ('QclKeEZEVr7j0klPuanE-gameplay-attempt-3-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61',)): [[2736, 2753]],
            ('NJUY0YT1Pq6dZXsmw0wE-gameplay-attempt-2-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[484, 498], [566, 609], [3211, 3292], [3308, 3325]],
            ('FyGQn1qJCLTLU1hfQfZ2-gameplay-attempt-1-rerecorded', ('?b->Beachball|+02.29|+00.19|-02.88',)): [[106, 196], [258, 277], [3266, 3267], [3274, 3275], [3278, 3280], [3290, 3296], [3297, 3303]],
            ('IvoZWi01FO2uiNpNHyci-preCreateGame-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[721, 777], [798, 834], [853, 889], [1017, 1222], [1487, 1498], [1601, 1602], [1603, 1604], [1648, 1650], [1656, 1658], [1663, 1665]],
            ('79X7tsrbEIu5ffDGnY8q-preCreateGame-rerecorded', ('?b->Dodgeball|+00.44|+01.13|-02.80',)): [[289, 335], [847, 897], [923, 988], [1000, 1035], [1051, 1088], [2220, 2235], [2262, 2337], [2339, 2349], [2350, 2353], [2411, 2425], [3106, 3135]],
            ('4WUtnD8W6PGVy0WBtVm4-freePlay-rerecorded', ('?b->Dodgeball|-02.60|+00.13|-02.18',)): [[105, 120]],
            ('ktwB7wT09sh4ivNme3Dw-preCreateGame-rerecorded', ('?b->BasketBall|-02.58|+00.12|-01.93',)): [[317, 327], [1610, 1625], [1659, 1690], [1736, 1740], [1824, 1886], [2054, 2082], [2117, 2147], [2171, 2204], [2232, 2265], [2301, 2379], [2413, 2475], [2506, 2729], [4294, 4298]],
            ('4WUtnD8W6PGVy0WBtVm4-freePlay-rerecorded', ('?b->Beachball|-02.93|+00.17|-01.99',)): [[0, 16]],
            ('QclKeEZEVr7j0klPuanE-gameplay-attempt-3-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[1181, 1234], [1235, 1236], [1271, 1276], [5168, 5190], [5208, 5209], [5245, 5279], [5306, 5333], [5344, 5386], [5387, 5396]],
            ('6ZjBeRCvHxG05ORmhInj-preCreateGame-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61',)): [[259, 270], [271, 275], [336, 399], [492, 547], [616, 649], [675, 731], [818, 876], [911, 961], [1023, 1109], [1473, 1477], [1478, 1479], [1506, 1574], [1698, 1749], [1772, 1800], [1801, 1809], [1810, 1811], [1817, 1867], [1891, 1919], [1996, 2033]],
            ('f2WUeVzu41E9Lmqmr2FJ-gameplay-attempt-1-rerecorded', ('?b->Beachball|+02.29|+00.19|-02.88',)): [[1913, 1944]],
            ('SQErBa5s5TPVxmm8R6ks-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[965, 1015], [1223, 1260], [1288, 1347], [1370, 1415], [1442, 1443]],
            ('f2WUeVzu41E9Lmqmr2FJ-preCreateGame-rerecorded', ('?b->Dodgeball|+00.44|+01.13|-02.80',)): [[1961, 2191]],
            ('6ZjBeRCvHxG05ORmhInj-preCreateGame-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[1626, 1815]],
            ('FyGQn1qJCLTLU1hfQfZ2-preCreateGame-rerecorded', ('?b->Golfball|+00.96|+01.04|-02.70',)): [[505, 528], [991, 1043]],
            ('1HOTuIZpRqk2u1nZI1v1-preCreateGame-rerecorded', ('?b->Golfball|+01.14|+01.04|-02.70',)): [[532, 533]],
            ('IvoZWi01FO2uiNpNHyci-preCreateGame-rerecorded', ('?b->Dodgeball|+00.19|+01.13|-02.80',)): [[1686, 1726], [1753, 1808]],
            ('79X7tsrbEIu5ffDGnY8q-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[680, 684], [817, 831], [1416, 1447], [1907, 2013]],
            ('IvoZWi01FO2uiNpNHyci-gameplay-attempt-2-rerecorded', ('?b->Golfball|+01.05|+01.04|-02.70',)): [[495, 613], [644, 746], [778, 849], [902, 903], [995, 996], [999, 1000], [1280, 1284], [1383, 1384], [1404, 1408]],
            ('f2WUeVzu41E9Lmqmr2FJ-preCreateGame-rerecorded', ('?b->Dodgeball|+00.19|+01.13|-02.80',)): [[1960, 2034], [2182, 2205], [2234, 2282]],
            ('FyGQn1qJCLTLU1hfQfZ2-freePlay-rerecorded', ('?b->Golfball|+01.05|+01.04|-02.70',)): [[1080, 1112], [1168, 1217]],
            ('xMUrxzK3fXjgitdzPKsm-gameplay-attempt-1-rerecorded', ('?b->Golfball|+01.14|+01.04|-02.70',)): [[5531, 5534]],
            ('79X7tsrbEIu5ffDGnY8q-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.19|+01.13|-02.80',)): [[882, 896], [931, 940], [941, 944], [956, 977], [1010, 1024], [1043, 1061], [1062, 1071], [1518, 1763]],
            ('7r4cgxJHzLJooFaMG1Rd-preCreateGame-rerecorded', ('?b->Beachball|+02.29|+00.19|-02.88',)): [[606, 617], [618, 620], [621, 622], [769, 770], [808, 922]],
            ('ktwB7wT09sh4ivNme3Dw-preCreateGame-rerecorded', ('?b->Dodgeball|-02.60|+00.13|-02.18',)): [[316, 333], [340, 343], [1410, 1417], [1476, 1503], [1896, 1925], [3147, 3184], [4178, 4237]],
            ('9C0wMm4lzrJ5JeP0irIu-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61',)): [[329, 330], [406, 422], [560, 572], [1082, 1091], [1144, 1155]],
            ('1HOTuIZpRqk2u1nZI1v1-preCreateGame-rerecorded', ('?b->Golfball|+00.96|+01.04|-02.70',)): [[277, 285], [508, 514]],
            ('9C0wMm4lzrJ5JeP0irIu-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[194, 209], [819, 830], [912, 923], [1082, 1091], [1144, 1169]],
            ('WtZpe3LQFZiztmh7pBBC-gameplay-attempt-1-rerecorded', ('?b->Golfball|+01.14|+01.04|-02.70',)): [[132, 245]],
            ('xMUrxzK3fXjgitdzPKsm-preCreateGame-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[2237, 2310], [2343, 2478], [2493, 2497], [2580, 2620], [2671, 2729], [2777, 2835], [2949, 3011], [3012, 3015], [3066, 3158], [3265, 3309], [3352, 3398], [3440, 3501], [3531, 3608], [3690, 3708], [3743, 3806], [3850, 3888], [3972, 4015], [4038, 4135]],
            ('4WUtnD8W6PGVy0WBtVm4-gameplay-attempt-1-rerecorded', ('?b->BasketBall|-02.58|+00.12|-01.93',)): [[1208, 1210]],
            ('7r4cgxJHzLJooFaMG1Rd-createGame-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[249, 466], [881, 908], [950, 991], [1019, 1088], [1280, 1312], [1374, 1400], [1548, 1573], [1669, 1702], [1703, 1704], [1870, 1886], [2611, 2664]],
            ('QyX7AlBzBW8hZHsJeDWI-gameplay-attempt-3-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61',)): [[925, 989], [1035, 1085], [1177, 1224]],
            ('39PytL3fAMFkYXNoB5l6-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.44|+01.13|-02.80',)): [[3085, 3124], [3207, 3373]],
            ('39PytL3fAMFkYXNoB5l6-createGame-rerecorded', ('?b->Dodgeball|+00.44|+01.13|-02.80',)): [[1685, 1725], [1927, 1976], [2018, 2029], [3598, 3627], [3668, 3680], [3699, 3790]],
            ('1HOTuIZpRqk2u1nZI1v1-gameplay-attempt-1-rerecorded', ('?b->Golfball|+01.14|+01.04|-02.70',)): [[519, 524], [835, 836]],
            ('1HOTuIZpRqk2u1nZI1v1-preCreateGame-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[277, 286], [330, 332], [349, 351]],
            ('IhOkh1l3TBY9JJVubzHx-createGame-rerecorded', ('?b->Beachball|+02.29|+00.19|-02.88',)): [[181, 207], [1206, 1219], [1499, 1525]],
            ('6XD5S6MnfzAPQlsP7k30-preCreateGame-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61',)): [[2990, 3045], [3074, 3139], [3189, 3252], [3311, 3369], [3396, 3455], [3491, 3570], [3603, 3628], [3629, 3647], [3669, 3752], [3785, 3869], [3898, 3934], [3946, 4037], [4052, 4163], [4234, 4272], [4292, 4380], [4413, 4483], [4542, 4579], [4580, 4599], [4703, 4893]],
            ('xMUrxzK3fXjgitdzPKsm-gameplay-attempt-1-rerecorded', ('?b->Golfball|+00.96|+01.04|-02.70',)): [[5531, 5639]],
            ('vfh1MTEQorWXKy8jOP1x-preCreateGame-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61',)): [[670, 695]],
            ('LTZh4k4THamxI5QJfVrk-preCreateGame-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[396, 445], [475, 562], [594, 633], [656, 696], [713, 754], [917, 918]],
            ('QyX7AlBzBW8hZHsJeDWI-gameplay-attempt-3-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[182, 230], [262, 313], [412, 425], [426, 452], [511, 548], [608, 642], [1205, 1216], [1218, 1224]],
            ('vfh1MTEQorWXKy8jOP1x-preCreateGame-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[732, 757], [1934, 2047], [2110, 2140], [2202, 2233], [2261, 2297]],
            ('jCc0kkmGUg3xUmUSXg5w-preCreateGame-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61',)): [[6036, 6089], [6109, 6149], [6177, 6213], [6214, 6215], [6328, 6332], [6386, 6387], [6402, 6406], [6407, 6411]],
            ('IhOkh1l3TBY9JJVubzHx-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.19|+01.13|-02.80',)): [[480, 495], [899, 944], [1000, 1038], [1061, 1115], [1398, 1439], [1540, 1591], [1609, 1649], [1775, 1816], [1963, 1969], [1991, 2038], [2110, 2179], [2268, 2315], [2358, 2411], [2459, 2513], [2524, 2550]],
            ('WtZpe3LQFZiztmh7pBBC-gameplay-attempt-1-rerecorded', ('?b->Golfball|+00.96|+01.04|-02.70',)): [[498, 499]],
            ('xMUrxzK3fXjgitdzPKsm-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[286, 496], [838, 897], [931, 964], [980, 1067], [1114, 1153], [1217, 1302], [1346, 1397], [1442, 1505], [1506, 1509], [1748, 1784], [1825, 1878], [1919, 1991], [2010, 2038], [2072, 2290], [2709, 2745], [2778, 2850], [2890, 2939], [2984, 3047], [3097, 3158], [3206, 3256], [3277, 3308], [3370, 3413], [3433, 3505], [3549, 3584], [3649, 3696], [3753, 3798], [3812, 3862], [3880, 3928], [4118, 4218], [4287, 4325], [4407, 4447], [4457, 4505], [4531, 4698], [4727, 4795], [4832, 4833], [5012, 5055], [5115, 5162], [5211, 5253], [5286, 5324], [5353, 5387], [5457, 5542], [5657, 5702], [5724, 5767], [5795, 5838], [5889, 5965], [5984, 6034], [6098, 6196], [6315, 6379], [6610, 6670], [6714, 6758], [6759, 6767], [6831, 6867], [6937, 6974], [6994, 7045], [7063, 7133]],
            ('Tcfpwc8v8HuKRyZr5Dyc-preCreateGame-rerecorded', ('?b->Beachball|-02.93|+00.17|-01.99',)): [[0, 15]],
            ('9C0wMm4lzrJ5JeP0irIu-preCreateGame-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[323, 362], [381, 394]],
            ('jCc0kkmGUg3xUmUSXg5w-preCreateGame-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[586, 623], [689, 705], [1007, 1010], [1052, 1074], [6323, 6355], [6380, 6413]],
            ('Tcfpwc8v8HuKRyZr5Dyc-gameplay-attempt-2-rerecorded', ('?b->BasketBall|-02.58|+00.12|-01.93',)): [[203, 239]],
            ('R9nZAvDq7um7Sg49yf8T-preCreateGame-rerecorded', ('?b->Dodgeball|+00.19|+01.13|-02.80',)): [[314, 346], [405, 445], [967, 1032], [1067, 1097], [1135, 1339], [1907, 1948], [1949, 1958]],
            ('1HOTuIZpRqk2u1nZI1v1-gameplay-attempt-1-rerecorded', ('?b->Golfball|+00.96|+01.04|-02.70',)): [[270, 272], [318, 319], [336, 337], [401, 403], [611, 612], [714, 715], [742, 743], [785, 789]],
            ('IvoZWi01FO2uiNpNHyci-preCreateGame-rerecorded', ('?b->Beachball|+02.29|+00.19|-02.88',)): [[433, 434], [1024, 1037], [1682, 1756]],
            ('FyGQn1qJCLTLU1hfQfZ2-preCreateGame-rerecorded', ('?b->Golfball|+01.05|+01.04|-02.70',)): [[590, 597], [796, 953]],
            ('WtZpe3LQFZiztmh7pBBC-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[493, 591], [634, 715]],
            ('R9nZAvDq7um7Sg49yf8T-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.44|+01.13|-02.80',)): [[1145, 1188], [1230, 1291], [1342, 1366]],
            ('6XD5S6MnfzAPQlsP7k30-gameplay-attempt-2-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61',)): [[273, 355]],
            ('Tcfpwc8v8HuKRyZr5Dyc-gameplay-attempt-1-rerecorded', ('?b->BasketBall|-02.58|+00.12|-01.93',)): [[354, 508]],
            ('79X7tsrbEIu5ffDGnY8q-preCreateGame-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[289, 396], [2411, 2442], [2462, 2507], [2508, 2521], [2522, 2532], [2645, 2712], [2741, 2787], [2812, 3048]],
            ('6XD5S6MnfzAPQlsP7k30-gameplay-attempt-2-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[363, 373], [471, 518], [586, 682], [721, 810], [857, 931], [938, 965], [966, 976], [987, 1041], [1042, 1051], [1131, 1218], [1242, 1299], [1365, 1418], [1439, 1527], [1558, 1610], [1688, 1729], [1748, 1806], [1837, 1894], [1921, 1992], [2090, 2157], [2177, 2235], [2260, 2352], [2384, 2477], [2502, 2538], [2548, 2620], [2624, 2673], [2694, 2777], [2807, 2869], [2909, 2959], [2974, 3007], [3008, 3014], [3023, 3088], [3152, 3190], [3193, 3239], [3255, 3345], [3354, 3401], [3467, 3524], [3529, 3598], [3631, 3709], [3735, 3800], [3828, 3868], [3906, 3949], [3984, 4051], [4088, 4165], [4220, 4233], [4254, 4312]],
            ('WtZpe3LQFZiztmh7pBBC-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.19|+01.13|-02.80',)): [[306, 393], [487, 670]],
            ('f2WUeVzu41E9Lmqmr2FJ-preCreateGame-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[1020, 1176], [1224, 1278], [1299, 1369]],
            ('Tcfpwc8v8HuKRyZr5Dyc-gameplay-attempt-2-rerecorded', ('?b->Dodgeball|-02.60|+00.13|-02.18',)): [[412, 422], [619, 695], [1144, 1181], [1372, 1376], [1536, 1551], [1930, 1947]],
            ('jCc0kkmGUg3xUmUSXg5w-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61',)): [[947, 1007], [1008, 1011], [2656, 2805], [2891, 2912], [2989, 2999]],
            ('Tcfpwc8v8HuKRyZr5Dyc-gameplay-attempt-2-rerecorded', ('?b->Beachball|-02.93|+00.17|-01.99',)): [[0, 16], [627, 639], [767, 793], [811, 838], [869, 897], [1153, 1173], [1229, 1235], [1330, 1367], [1370, 1494], [1629, 1637], [1683, 1716], [1938, 1970]],
            ('FyGQn1qJCLTLU1hfQfZ2-preCreateGame-rerecorded', ('?b->Beachball|+02.29|+00.19|-02.88',)): [[183, 208], [726, 746], [795, 978], [1690, 1871]],
            ('1HOTuIZpRqk2u1nZI1v1-preCreateGame-rerecorded', ('?b->Golfball|+01.05|+01.04|-02.70',)): [[421, 423], [458, 460]],
            ('79X7tsrbEIu5ffDGnY8q-preCreateGame-rerecorded', ('?b->Dodgeball|+00.19|+01.13|-02.80',)): [[928, 974], [2058, 2072], [2128, 2172], [2173, 2178], [2179, 2182], [2543, 2589], [2822, 2876]],
            ('jCc0kkmGUg3xUmUSXg5w-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[766, 781], [825, 831], [869, 877]],
            ('IhOkh1l3TBY9JJVubzHx-freePlay-rerecorded', ('?b->Dodgeball|+00.44|+01.13|-02.80',)): [[278, 293], [449, 644], [1109, 1150], [1175, 1217], [1253, 1289], [1306, 1350], [1396, 1436], [1477, 1513], [1552, 1601]],
            ('7r4cgxJHzLJooFaMG1Rd-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.44|+01.13|-02.80',)): [[312, 319], [342, 351], [365, 370], [459, 484], [612, 614], [1190, 1191], [1623, 1664], [1953, 2035], [2068, 2180], [2749, 2794], [2810, 2840], [3484, 3516], [3517, 3518], [3532, 3639]],
            ('5lTRHBueXsaOu9yhvOQo-gameplay-attempt-1-rerecorded', ('?b->BasketBall|-02.58|+00.12|-01.93',)): [[100, 105], [149, 153], [252, 257]],
            ('xMUrxzK3fXjgitdzPKsm-gameplay-attempt-1-rerecorded', ('?b->Golfball|+01.05|+01.04|-02.70',)): [[5531, 5620]],
            ('xMUrxzK3fXjgitdzPKsm-preCreateGame-rerecorded', ('?b->Beachball|+02.29|+00.19|-02.88',)): [[357, 402], [460, 512], [537, 616], [622, 662], [743, 807], [856, 879], [911, 1097], [1147, 1184], [1209, 1277], [1306, 1351], [1380, 1412], [1469, 1594], [1662, 5310]],
            ('7r4cgxJHzLJooFaMG1Rd-createGame-rerecorded', ('?b->Beachball|+02.29|+00.19|-02.88',)): [[2, 62], [509, 604], [773, 921], [1046, 1150]],
            ('ktwB7wT09sh4ivNme3Dw-preCreateGame-rerecorded', ('?b->Beachball|-02.93|+00.17|-01.99',)): [[0, 14], [319, 320], [321, 322], [1721, 1789], [1943, 2145], [2325, 2378], [3161, 3186], [3203, 3273], [3320, 3357], [3408, 3445], [3466, 3502], [3519, 3558], [3595, 3634], [3668, 3713], [3762, 3795], [3807, 3879], [3939, 3952], [4265, 4314], [4355, 4395], [4417, 4448], [4464, 4511], [4533, 4574], [4619, 4660], [4680, 4723], [4759, 4796], [4807, 4822], [4849, 4903], [4965, 4992], [5051, 5086], [5124, 5165], [5196, 5231], [5252, 5291], [5332, 5373], [5401, 5439], [5471, 5518], [5543, 5586], [5606, 5644], [5699, 5735], [5766, 5813], [5835, 5872], [5890, 5923], [5944, 5954], [5970, 5995], [5998, 6002], [6046, 6084], [6129, 6138], [6207, 6232], [6303, 6351], [6405, 6446], [6477, 6518], [6536, 6537]],
            ('f2WUeVzu41E9Lmqmr2FJ-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.44|+01.13|-02.80',)): [[235, 357], [1026, 1063], [1293, 1335], [1356, 1429], [1470, 1472], [1481, 1482], [1483, 1608], [1866, 1983], [2059, 2064]],
            ('39PytL3fAMFkYXNoB5l6-gameplay-attempt-1-rerecorded', ('?b->Golfball|+00.96|+01.04|-02.70',)): [[2016, 2123]],
            ('7r4cgxJHzLJooFaMG1Rd-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.19|+01.13|-02.80',)): [[519, 559], [598, 624], [726, 730], [1190, 1191], [1623, 1679], [1884, 1924], [2081, 2090], [2312, 2367], [2474, 2568], [2946, 2985], [3054, 3118], [3145, 3176], [3200, 3256], [3289, 3311], [3312, 3327], [3537, 3538]],
            ('R9nZAvDq7um7Sg49yf8T-gameplay-attempt-1-rerecorded', ('?b->Golfball|+01.14|+01.04|-02.70',)): [[33, 38]],
            ('R9nZAvDq7um7Sg49yf8T-preCreateGame-rerecorded', ('?b->Beachball|+02.29|+00.19|-02.88',)): [[467, 535], [558, 596]],
            ('IhOkh1l3TBY9JJVubzHx-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[403, 417], [614, 624], [756, 800], [937, 942], [1027, 1038], [1094, 1108], [1259, 1316], [1331, 1379], [1798, 1811], [1849, 1895], [1932, 1979], [2050, 2102], [2193, 2341], [2910, 2951], [3011, 3051], [3089, 3137], [3155, 3217]],
            ('39PytL3fAMFkYXNoB5l6-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[3376, 3427]],
            ('39PytL3fAMFkYXNoB5l6-createGame-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[1822, 1837], [3401, 3545], [4285, 4387], [4404, 4526]],
            ('xMUrxzK3fXjgitdzPKsm-gameplay-attempt-1-rerecorded', ('?b->Beachball|+02.29|+00.19|-02.88',)): [[109, 152], [175, 444], [2177, 2210], [2704, 2807], [2810, 2812]],
            ('IvoZWi01FO2uiNpNHyci-createGame-rerecorded', ('?b->Dodgeball|+00.44|+01.13|-02.80',)): [[2, 24]],
            ('7r4cgxJHzLJooFaMG1Rd-createGame-rerecorded', ('?b->Dodgeball|+00.44|+01.13|-02.80',)): [[323, 400], [438, 461]],
            ('QyX7AlBzBW8hZHsJeDWI-preCreateGame-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61',)): [[597, 729], [3688, 3754], [3819, 3901], [3958, 4055], [4180, 4208], [4209, 4220], [4221, 4223], [4332, 4340], [4341, 4343]],
            ('39PytL3fAMFkYXNoB5l6-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.19|+01.13|-02.80',)): [[1166, 1170], [2962, 3133]],
            ('WtZpe3LQFZiztmh7pBBC-gameplay-attempt-1-rerecorded', ('?b->Beachball|+02.29|+00.19|-02.88',)): [[531, 532]],
            ('39PytL3fAMFkYXNoB5l6-createGame-rerecorded', ('?b->Dodgeball|+00.19|+01.13|-02.80',)): [[998, 1013], [1161, 1186], [1240, 1333], [1631, 1639], [1640, 1642], [3796, 3886], [3887, 3903], [4384, 4459], [4479, 4518], [4594, 4609], [4646, 4692]],
            ('IvoZWi01FO2uiNpNHyci-freePlay-rerecorded', ('?b->Dodgeball|+00.44|+01.13|-02.80',)): [[257, 309], [326, 397]],
            ('5lTRHBueXsaOu9yhvOQo-preCreateGame-rerecorded', ('?b->Beachball|-02.93|+00.17|-01.99',)): [[0, 16]],
            ('R9nZAvDq7um7Sg49yf8T-gameplay-attempt-1-rerecorded', ('?b->Golfball|+00.96|+01.04|-02.70',)): [[33, 39]],
            ('QyX7AlBzBW8hZHsJeDWI-preCreateGame-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[263, 7822], [7823, 7824]],
            ('IvoZWi01FO2uiNpNHyci-gameplay-attempt-2-rerecorded', ('?b->Dodgeball|+00.44|+01.13|-02.80',)): [[1453, 1502], [1540, 1622]],
            ('NJUY0YT1Pq6dZXsmw0wE-createGame-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61',)): [[904, 931], [1164, 1198], [1209, 1241], [1242, 1268], [1269, 1279], [1312, 1332], [3886, 3922], [3924, 3933], [3954, 3976], [3977, 3980], [4382, 4418]],
            ('4WUtnD8W6PGVy0WBtVm4-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|-02.60|+00.13|-02.18',)): [[261, 317], [424, 449], [593, 685], [780, 781], [829, 855], [973, 1003], [1121, 1216], [1324, 1401], [1624, 1746], [1831, 1886], [2033, 2054], [2189, 2246], [2378, 2462], [2540, 2585], [2711, 2757], [2818, 2862], [2940, 2983], [3072, 3119], [3120, 3138], [3157, 3193], [3294, 3345], [3386, 3429], [3530, 3554], [3555, 3570], [3571, 3574], [3585, 3620]],
            ('4WUtnD8W6PGVy0WBtVm4-gameplay-attempt-1-rerecorded', ('?b->Beachball|-02.93|+00.17|-01.99',)): [[0, 18]],
            ('79X7tsrbEIu5ffDGnY8q-preCreateGame-rerecorded', ('?b->Beachball|+02.29|+00.19|-02.88',)): [[1814, 1845], [1877, 1911], [1927, 1939], [1983, 1996], [2020, 2048]],
            ('IhOkh1l3TBY9JJVubzHx-createGame-rerecorded', ('?b->Dodgeball|+00.44|+01.13|-02.80',)): [[1253, 1254], [1314, 1328]],
            ('FyGQn1qJCLTLU1hfQfZ2-gameplay-attempt-1-rerecorded', ('?b->Golfball|+01.14|+01.04|-02.70',)): [[2321, 2345], [2380, 2383], [2969, 3095]],
            ('R9nZAvDq7um7Sg49yf8T-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[1042, 1068], [1466, 1488], [1637, 1670], [1718, 1767], [1804, 1851], [1880, 1972]],
            ('NJUY0YT1Pq6dZXsmw0wE-createGame-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[3346, 3360], [3361, 3371], [3415, 3457], [3482, 3510], [3584, 3598], [3778, 4462], [4463, 4464]],
            ('ktwB7wT09sh4ivNme3Dw-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|-02.60|+00.13|-02.18',)): [[368, 382], [383, 390]],
            ('f2WUeVzu41E9Lmqmr2FJ-preCreateGame-rerecorded', ('?b->Beachball|+02.29|+00.19|-02.88',)): [[154, 184], [1548, 1551], [1656, 1684], [1709, 1718], [1723, 1782], [1822, 1838], [1918, 1952]],
            ('ktwB7wT09sh4ivNme3Dw-gameplay-attempt-1-rerecorded', ('?b->Beachball|-02.93|+00.17|-01.99',)): [[0, 18], [367, 446]],
            ('jCc0kkmGUg3xUmUSXg5w-gameplay-attempt-2-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[1480, 1503], [1582, 1607], [1610, 1615]],
            ('ktwB7wT09sh4ivNme3Dw-createGame-rerecorded', ('?b->Beachball|-02.93|+00.17|-01.99',)): [[0, 19]],
            ('7r4cgxJHzLJooFaMG1Rd-gameplay-attempt-1-rerecorded', ('?b->Golfball|+00.96|+01.04|-02.70',)): [[551, 559]],
            ('39PytL3fAMFkYXNoB5l6-gameplay-attempt-1-rerecorded', ('?b->Golfball|+01.05|+01.04|-02.70',)): [[2016, 2017]],
            ('4WUtnD8W6PGVy0WBtVm4-createGame-rerecorded', ('?b->Beachball|-02.93|+00.17|-01.99',)): [[0, 17]],
            ('IvoZWi01FO2uiNpNHyci-preCreateGame-rerecorded', ('?b->Dodgeball|+00.44|+01.13|-02.80',)): [[1686, 1810]],
            ('FyGQn1qJCLTLU1hfQfZ2-gameplay-attempt-1-rerecorded', ('?b->Golfball|+00.96|+01.04|-02.70',)): [[2093, 2108], [3114, 3171], [3249, 3256], [3436, 3440], [3505, 3506], [3509, 3510], [3511, 3515], [3960, 3967], [4049, 4050], [4195, 4200], [4718, 4760], [4763, 4788], [4807, 4840], [4871, 4872]],
            ('IhOkh1l3TBY9JJVubzHx-freePlay-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[555, 634], [686, 758], [790, 839], [854, 861], [901, 916], [938, 946], [968, 1013], [1025, 1039]],
            ('7r4cgxJHzLJooFaMG1Rd-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[243, 311], [716, 740], [851, 886], [915, 982], [1002, 1052], [1067, 1140], [1252, 1287], [1288, 1295], [1296, 1305], [1548, 1618], [1746, 1866], [2323, 2324], [2474, 2530], [3205, 3207], [3295, 3311], [3314, 3327], [3537, 3547]],
            ('7r4cgxJHzLJooFaMG1Rd-createGame-rerecorded', ('?b->Golfball|+01.14|+01.04|-02.70',)): [[769, 781], [782, 790]],
            ('QyX7AlBzBW8hZHsJeDWI-gameplay-attempt-2-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61',)): [[120, 141], [294, 415], [440, 509], [926, 932], [933, 937], [939, 941]],
            ('Tcfpwc8v8HuKRyZr5Dyc-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|-02.60|+00.13|-02.18',)): [[516, 576], [1143, 1209], [1290, 1377], [1427, 1452], [1538, 1569], [1631, 1652], [1709, 1733], [1918, 1932], [2044, 2075], [2149, 2177], [2266, 2287], [2347, 2371], [2447, 2467], [2527, 2622], [2712, 2721], [2908, 2913], [2926, 2933], [2942, 2973], [2974, 2976], [3047, 3076], [3137, 3160], [3220, 3242], [3319, 3341], [3424, 3447], [3475, 3579], [3626, 3647], [3700, 3725], [3923, 3959], [3960, 3963], [4206, 4224], [4347, 4374]],
            ('Tcfpwc8v8HuKRyZr5Dyc-gameplay-attempt-1-rerecorded', ('?b->Beachball|-02.93|+00.17|-01.99',)): [[0, 18], [1917, 1921], [1922, 1926], [1942, 1958], [2051, 2061], [2536, 2641], [2666, 2688], [2746, 2751], [2752, 2759], [2792, 2842], [4866, 4890], [5046, 5088]],
            ('FyGQn1qJCLTLU1hfQfZ2-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[2169, 2191]],
            ('79X7tsrbEIu5ffDGnY8q-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.44|+01.13|-02.80',)): [[746, 761], [813, 848], [1763, 1895], [1925, 2078]],
            ('79X7tsrbEIu5ffDGnY8q-createGame-rerecorded', ('?b->Dodgeball|+00.19|+01.13|-02.80',)): [[2, 39]],
            ('QyX7AlBzBW8hZHsJeDWI-gameplay-attempt-2-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[575, 576], [592, 607], [802, 868], [898, 941]],
            ('IhOkh1l3TBY9JJVubzHx-gameplay-attempt-1-rerecorded', ('?b->Beachball|+02.29|+00.19|-02.88',)): [[182, 215], [607, 657], [1034, 1039], [1096, 1106], [3268, 3302], [3362, 3397], [3456, 3492], [3511, 3547], [3558, 3603], [3624, 3682]],
            ('IhOkh1l3TBY9JJVubzHx-freePlay-rerecorded', ('?b->Dodgeball|+00.19|+01.13|-02.80',)): [[625, 849]],
            ('f2WUeVzu41E9Lmqmr2FJ-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[110, 192], [253, 421], [871, 891], [960, 974], [1463, 1507], [2048, 2072]],
            ('39PytL3fAMFkYXNoB5l6-gameplay-attempt-1-rerecorded', ('?b->Beachball|+02.29|+00.19|-02.88',)): [[511, 526]],
            ('39PytL3fAMFkYXNoB5l6-createGame-rerecorded', ('?b->Beachball|+02.29|+00.19|-02.88',)): [[1527, 1535], [3048, 3051]],
            ('IvoZWi01FO2uiNpNHyci-gameplay-attempt-2-rerecorded', ('?b->Golfball|+01.14|+01.04|-02.70',)): [[96, 188], [212, 318], [372, 399], [651, 652], [785, 788], [801, 837], [902, 903], [995, 996], [1383, 1384], [1401, 1408]],
            ('R9nZAvDq7um7Sg49yf8T-gameplay-attempt-1-rerecorded', ('?b->Golfball|+01.05|+01.04|-02.70',)): [[33, 38]],
            ('6ZjBeRCvHxG05ORmhInj-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61',)): [[922, 932], [933, 936], [1468, 1535], [2154, 2197], [2268, 2300], [2330, 2346]],
            ('FyGQn1qJCLTLU1hfQfZ2-createGame-rerecorded', ('?b->Dodgeball|+00.44|+01.13|-02.80',)): [[132, 143]],
            ('7r4cgxJHzLJooFaMG1Rd-createGame-rerecorded', ('?b->Golfball|+00.96|+01.04|-02.70',)): [[769, 894]],
            ('1HOTuIZpRqk2u1nZI1v1-gameplay-attempt-1-rerecorded', ('?b->Golfball|+01.05|+01.04|-02.70',)): [[455, 464], [911, 918]],
            ('f2WUeVzu41E9Lmqmr2FJ-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.19|+01.13|-02.80',)): [[399, 483], [954, 977], [1048, 1057], [1156, 1289], [1422, 1423], [1482, 1483], [1951, 1994]],
            ('6ZjBeRCvHxG05ORmhInj-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[879, 893], [1304, 1323], [1385, 1437], [1484, 1497], [1964, 2019], [2050, 2200], [2322, 2361], [2374, 2478]],
            ('5lTRHBueXsaOu9yhvOQo-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|-02.60|+00.13|-02.18',)): [[136, 141], [225, 229]],
            ('5lTRHBueXsaOu9yhvOQo-gameplay-attempt-1-rerecorded', ('?b->Beachball|-02.93|+00.17|-01.99',)): [[0, 18], [118, 124], [138, 141], [227, 230], [273, 286]],
            ('7r4cgxJHzLJooFaMG1Rd-preCreateGame-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[162, 405], [872, 898]],
            ('IvoZWi01FO2uiNpNHyci-gameplay-attempt-2-rerecorded', ('?b->Golfball|+00.96|+01.04|-02.70',)): [[1270, 1333], [1374, 1408]],
            ('IhOkh1l3TBY9JJVubzHx-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.44|+01.13|-02.80',)): [[443, 458], [641, 643], [826, 986], [1124, 1166], [1186, 1243], [2512, 2513], [2571, 2610], [2657, 2702], [2750, 2794], [2827, 3009], [3653, 3682]],
            ('IvoZWi01FO2uiNpNHyci-freePlay-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[76, 143], [161, 251], [623, 660]],
            ('IvoZWi01FO2uiNpNHyci-preCreateGame-rerecorded', ('?b->Golfball|+01.14|+01.04|-02.70',)): [[562, 622]],
            ('FyGQn1qJCLTLU1hfQfZ2-freePlay-rerecorded', ('?b->Golfball|+00.96|+01.04|-02.70',)): [[323, 334], [359, 373], [397, 410], [435, 573], [766, 802], [807, 825], [900, 938], [948, 996]],
            ('4WUtnD8W6PGVy0WBtVm4-preCreateGame-rerecorded', ('?b->Dodgeball|-02.60|+00.13|-02.18',)): [[821, 925], [1012, 1141], [1198, 1228], [1283, 1315], [1316, 1330]],
            ('4WUtnD8W6PGVy0WBtVm4-preCreateGame-rerecorded', ('?b->Beachball|-02.93|+00.17|-01.99',)): [[0, 14]],
            ('7r4cgxJHzLJooFaMG1Rd-gameplay-attempt-1-rerecorded', ('?b->Golfball|+01.05|+01.04|-02.70',)): [[367, 370]],
            ('4WUtnD8W6PGVy0WBtVm4-editGame-rerecorded', ('?b->Dodgeball|-02.60|+00.13|-02.18',)): [[287, 345], [435, 529], [624, 670], [671, 684]],
            ('IvoZWi01FO2uiNpNHyci-gameplay-attempt-2-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80',)): [[894, 952], [987, 1054]],
            ('4WUtnD8W6PGVy0WBtVm4-editGame-rerecorded', ('?b->Beachball|-02.93|+00.17|-01.99',)): [[0, 19]],
            ('SQErBa5s5TPVxmm8R6ks-preCreateGame-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28',)): [[589, 850], [1725, 1816], [1830, 1894]],
            ('7r4cgxJHzLJooFaMG1Rd-createGame-rerecorded', ('?b->Dodgeball|+00.19|+01.13|-02.80',)): [[643, 676], [1089, 1103], [1161, 1186], [1209, 1238]],
            ('FyGQn1qJCLTLU1hfQfZ2-gameplay-attempt-1-rerecorded', ('?b->Golfball|+01.05|+01.04|-02.70',)): [[2226, 2246], [3048, 3142], [3242, 3368], [3430, 3476], [3497, 3515], [4295, 4357], [4385, 4436], [4467, 4546], [4590, 4736], [4814, 4830], [4831, 4832], [4833, 4834]],
        }
    ),

    ExpectedCompletionsTestCase(
        test_id='test (and (not (in_motion ?b)) (in ?h ?b)))',
        game_key='test-ball-from-bed',
        predicate_selector=[4, 1, 'preferences', 0, 'definition', 'forall_pref', 'preferences', 'pref_body', 'body', 'exists_args', 'then_funcs', 2, 'seq_func', 'once_pred'],
        mapping={'?b': ['ball'], '?h': ['hexagonal_bin']},
        expected_intervals={
            ('IhOkh1l3TBY9JJVubzHx-freePlay-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[1039, 1606]],
            ('7r4cgxJHzLJooFaMG1Rd-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.19|+01.13|-02.80', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[3311, 3312], [3327, 3537], [3538, 3640]],
            ('R9nZAvDq7um7Sg49yf8T-preCreateGame-rerecorded', ('?b->Golfball|+00.96|+01.04|-02.70', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[185, 189]],
            ('FyGQn1qJCLTLU1hfQfZ2-gameplay-attempt-1-rerecorded', ('?b->Golfball|+01.05|+01.04|-02.70', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[3515, 3947], [3949, 3950], [3956, 4193], [4813, 4814], [4830, 4831], [4832, 4833], [4834, 4870]],
            ('FyGQn1qJCLTLU1hfQfZ2-preCreateGame-rerecorded', ('?b->Golfball|+01.14|+01.04|-02.70', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[1226, 1292], [1330, 1379], [1394, 1430], [1485, 1575], [1631, 1637], [1659, 1695], [1696, 1697], [1719, 2087], [2142, 2979]],
            ('NJUY0YT1Pq6dZXsmw0wE-createGame-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[1332, 3848], [4418, 4464]],
            ('jCc0kkmGUg3xUmUSXg5w-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[2912, 2989], [2999, 4588]],
            ('ktwB7wT09sh4ivNme3Dw-gameplay-attempt-1-rerecorded', ('?b->BasketBall|-02.58|+00.12|-01.93', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[368, 369]],
            ('QyX7AlBzBW8hZHsJeDWI-gameplay-attempt-2-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[941, 1005]],
            ('jCc0kkmGUg3xUmUSXg5w-gameplay-attempt-2-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[2, 30]],
            ('ktwB7wT09sh4ivNme3Dw-preCreateGame-rerecorded', ('?b->BasketBall|-02.58|+00.12|-01.93', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[317, 317]],
            ('6XD5S6MnfzAPQlsP7k30-preCreateGame-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[4483, 4508]],
            ('9C0wMm4lzrJ5JeP0irIu-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[1169, 1268]],
            ('6XD5S6MnfzAPQlsP7k30-gameplay-attempt-2-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[518, 531], [1041, 1042], [1051, 1074], [1299, 1340], [1527, 1530], [1894, 1894], [3598, 3598], [3709, 3710], [3800, 3801], [4165, 4220], [4233, 4233]],
            ('IvoZWi01FO2uiNpNHyci-gameplay-attempt-2-rerecorded', ('?b->Golfball|+01.14|+01.04|-02.70', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[399, 651], [652, 785], [788, 801], [837, 902], [903, 995], [996, 1383], [1384, 1401], [1408, 1624]],
            ('IvoZWi01FO2uiNpNHyci-gameplay-attempt-2-rerecorded', ('?b->Golfball|+00.96|+01.04|-02.70', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[1408, 1624]],
            ('7r4cgxJHzLJooFaMG1Rd-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[2530, 3205], [3207, 3295], [3311, 3314], [3327, 3537], [3547, 3640]],
            ('FyGQn1qJCLTLU1hfQfZ2-gameplay-attempt-1-rerecorded', ('?b->Golfball|+00.96|+01.04|-02.70', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[3171, 3249], [3256, 3436], [3440, 3505], [3506, 3509], [3510, 3511], [3515, 3960], [3967, 4049], [4050, 4195], [4200, 4718], [4760, 4763], [4788, 4807], [4840, 4870]],
            ('R9nZAvDq7um7Sg49yf8T-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[1767, 1774]],
            ('f2WUeVzu41E9Lmqmr2FJ-preCreateGame-rerecorded', ('?b->Dodgeball|+00.19|+01.13|-02.80', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[2282, 2343]],
            ('jCc0kkmGUg3xUmUSXg5w-preCreateGame-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[623, 674], [6413, 10292]],
            ('7r4cgxJHzLJooFaMG1Rd-createGame-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[1573, 1669], [1702, 1703], [1704, 1870], [1886, 2611], [2664, 5193]],
            ('IvoZWi01FO2uiNpNHyci-freePlay-rerecorded', ('?b->Dodgeball|+00.70|+01.11|-02.80', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[251, 623], [660, 785]],
            ('NJUY0YT1Pq6dZXsmw0wE-gameplay-attempt-2-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[609, 3158], [3165, 3168]],
            ('QyX7AlBzBW8hZHsJeDWI-gameplay-attempt-3-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[2, 68], [642, 1205], [1216, 1218], [1224, 1266]],
            ('QyX7AlBzBW8hZHsJeDWI-gameplay-attempt-2-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[509, 926], [932, 933], [937, 939], [941, 1005]],
            ('6XD5S6MnfzAPQlsP7k30-freePlay-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[251, 265], [839, 839], [1999, 2027], [2081, 2127]],
            ('R9nZAvDq7um7Sg49yf8T-gameplay-attempt-1-rerecorded', ('?b->Golfball|+01.05|+01.04|-02.70', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[32, 33], [38, 40]],
            ('ktwB7wT09sh4ivNme3Dw-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|-02.60|+00.13|-02.18', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[368, 368], [382, 383]],
            ('LTZh4k4THamxI5QJfVrk-preCreateGame-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[754, 917], [918, 1069]],
            ('ktwB7wT09sh4ivNme3Dw-preCreateGame-rerecorded', ('?b->Dodgeball|-02.60|+00.13|-02.18', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[333, 339]],
            ('WtZpe3LQFZiztmh7pBBC-gameplay-attempt-1-rerecorded', ('?b->Dodgeball|+00.44|+01.13|-02.80', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[1440, 1557]],
            ('1HOTuIZpRqk2u1nZI1v1-preCreateGame-rerecorded', ('?b->Golfball|+01.14|+01.04|-02.70', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[533, 550]],
            ('QyX7AlBzBW8hZHsJeDWI-gameplay-attempt-3-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[2, 68], [1224, 1266]],
            ('39PytL3fAMFkYXNoB5l6-createGame-rerecorded', ('?b->Dodgeball|+00.19|+01.13|-02.80', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[1333, 1577], [4067, 4070]],
            ('jCc0kkmGUg3xUmUSXg5w-gameplay-attempt-2-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[1503, 1582], [1607, 1610], [1615, 2003]],
            ('R9nZAvDq7um7Sg49yf8T-gameplay-attempt-1-rerecorded', ('?b->Golfball|+01.14|+01.04|-02.70', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[32, 33], [38, 40]],
            ('IvoZWi01FO2uiNpNHyci-gameplay-attempt-2-rerecorded', ('?b->Golfball|+01.05|+01.04|-02.70', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[849, 902], [903, 995], [996, 999], [1000, 1280], [1284, 1383], [1384, 1404], [1408, 1624]],
            ('R9nZAvDq7um7Sg49yf8T-gameplay-attempt-1-rerecorded', ('?b->Golfball|+00.96|+01.04|-02.70', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[32, 33], [39, 40]],
            ('QclKeEZEVr7j0klPuanE-gameplay-attempt-3-rerecorded', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[1234, 1235], [1236, 1271], [1276, 5168], [5190, 5208], [5209, 5245], [5279, 5306], [5333, 5333]],
            ('jCc0kkmGUg3xUmUSXg5w-preCreateGame-rerecorded', ('?b->Dodgeball|-02.95|+01.29|-02.61', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[6213, 6214], [6215, 6328], [6332, 6386], [6387, 6402], [6406, 6407], [6411, 10292]],
        }
    )
]


DEFAULT_GRAMMAR_PATH = "./dsl/dsl.ebnf"

def _test_to_id(test_case: ExpectedCompletionsTestCase) -> str:
    return test_case.test_id

@pytest.mark.parametrize("test_case", EXPECTED_COMPLETION_TEST_CASES, ids=_test_to_id)
def test_expected_completions(test_case: ExpectedCompletionsTestCase):
    cache_dir = pathlib.Path(get_project_dir() + test_case.cache_dir_relative_path)

    stats = CommonSensePredicateStatistics(cache_dir, [], test_case.cache_rules, overwrite=False)  # type: ignore

    game_def = TEST_GAME_LIBRARY[test_case.game_key]

    grammar = open(DEFAULT_GRAMMAR_PATH).read()
    grammar_parser = typing.cast(tatsu.grammars.Grammar, tatsu.compile(grammar))
    game_ast = grammar_parser.parse(game_def)

    predicate = game_ast
    for selector in test_case.predicate_selector:
        predicate = predicate[selector]  # type: ignore

    output = stats.filter(predicate, test_case.mapping)  # type: ignore

    for key in test_case.expected_intervals:
        assert key in output
        assert output[key] == test_case.expected_intervals[key]


if __name__ == '__main__':
    print(__file__)
    sys.exit(pytest.main([__file__]))
