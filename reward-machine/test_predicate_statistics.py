from dataclasses import dataclass
import tatsu
import tatsu.grammars
import typing
import pathlib
import pytest
import sys

from compile_predicate_statistics import CommonSensePredicateStatistics
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
            ('8cc9xnijzmS7uwGDaAoL', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[1390, 1407], [1573, 1626]],
            ('moftPPoSfP3mG38GT9Vw', ('?b->Dodgeball|-02.60|+00.13|-02.18', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[1894, 1902], [1934, 1941], [1982, 1987], [2095, 2103], [2241, 2250], [2744, 2916], [3321, 3383], [3696, 3738]],
            ('IjyNuz4ApTldgPuoKPtB', ('?b->BasketBall|-02.58|+00.12|-01.93', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[883, 970], [1021, 1039], [1050, 1073], [1092, 1144], [1177, 1191], [1192, 1227], [1310, 1322], [1370, 1388], [1392, 1411], [1434, 1695], [2122, 2151], [2154, 2157], [2186, 2324], [2901, 2947]],
            ('Q1jHN8NAIlds8F5SFBCJ', ('?b->Dodgeball|-02.60|+00.13|-02.18', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[2125, 2126], [2191, 2203], [2266, 2279]],
            ('nFX80fkvz9jp4TEfDQUl', ('?b->Dodgeball|-02.95|+01.29|-02.61', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[451, 488], [508, 509], [523, 538], [738, 756], [795, 803], [828, 837]],
            ('NurIP8Ey70FomFLL0VSr', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[1099, 1113], [1270, 1304], [1305, 1321], [1414, 1507], [1640, 1686], [1721, 1759], [1763, 1798], [1827, 1884], [1969, 2012], [2065, 2111], [2146, 2194], [2239, 2266], [2342, 2366], [2628, 2664], [2796, 2898], [3580, 3612], [3659, 3686], [3736, 3759], [3760, 3762], [3826, 3867], [3894, 3944]],
            ('ZBcXIZbvTS3U4IBGk1zk', ('?b->BasketBall|-02.58|+00.12|-01.93', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[656, 764], [777, 783]],
            ('9kyHjMMQRK3eP9vlLpw0', ('?b->BasketBall|-02.58|+00.12|-01.93', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[339, 345]],
            ('ZMqZkrMMB0PcsCeLhQqE', ('?b->Beachball|-02.93|+00.17|-01.99', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[0, 20]],
            ('HMosyugAD4iHS8V6ajD6', ('?b->Dodgeball|+00.70|+01.11|-02.80', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[173, 215], [216, 217]],
            ('c4bea3VqKksZ7Rd5RdTO', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[627, 642], [735, 766], [815, 824], [889, 926], [945, 957], [962, 993], [1004, 1024], [1027, 1028]],
            ('B02vfA7ZpP1xhDyqeYVd', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[269, 284]],
            ('NurIP8Ey70FomFLL0VSr', ('?b->Dodgeball|-02.95|+01.29|-02.61', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[1148, 1173], [1174, 1180], [1201, 1443], [1514, 1601], [3072, 3092], [3135, 3160], [3231, 3473], [3991, 4054]],
            ('IjyNuz4ApTldgPuoKPtB', ('?b->Beachball|-02.93|+00.17|-01.99', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[0, 17]],
            ('0xXJSvWhI0TJKqnPzY7O', ('?b->Dodgeball|+00.70|+01.11|-02.80', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[400, 444], [486, 578], [835, 867], [901, 933], [962, 1005], [1042, 1106], [1152, 1195], [1226, 1265], [1303, 1355], [1385, 1424], [1443, 1487], [1533, 1540], [1571, 1641], [1723, 1767], [1820, 1853], [1906, 1941], [1970, 2000], [2073, 2135], [2208, 2275], [2318, 2356], [2377, 2393]],
            ('0xXJSvWhI0TJKqnPzY7O', ('?b->Beachball|+02.29|+00.19|-02.88', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[513, 521], [528, 545], [614, 636], [719, 735]],
            ('3JgBMLuR3hhyXX7duXxU', ('?b->Beachball|-02.93|+00.17|-01.99', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[0, 18]],
            ('tVL75ycOZHTAKgeqLs6H', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[895, 940]],
            ('ZBcXIZbvTS3U4IBGk1zk', ('?b->Beachball|-02.93|+00.17|-01.99', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[0, 17]],
            ('B02vfA7ZpP1xhDyqeYVd', ('?b->Dodgeball|-02.95|+01.29|-02.61', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[198, 234]],
            ('Q1jHN8NAIlds8F5SFBCJ', ('?b->BasketBall|-02.58|+00.12|-01.93', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[326, 335], [403, 417], [519, 571]],
            ('Q6a8AbiIdcLA9tJzAu14', ('?b->Dodgeball|+00.70|+01.11|-02.80', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[3758, 3817], [4084, 4128], [4158, 4195], [4220, 4241], [4283, 4300]],
            ('moftPPoSfP3mG38GT9Vw', ('?b->BasketBall|-02.58|+00.12|-01.93', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[2170, 2178], [2260, 2269], [2496, 2654], [3847, 3883]],
            ('Q6a8AbiIdcLA9tJzAu14', ('?b->Dodgeball|+00.19|+01.13|-02.80', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[3883, 4193]],
            ('9kyHjMMQRK3eP9vlLpw0', ('?b->Beachball|-02.93|+00.17|-01.99', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[0, 17]],
            ('Q6a8AbiIdcLA9tJzAu14', ('?b->Beachball|+02.29|+00.19|-02.88', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[2838, 2843], [2889, 2890], [2910, 2944]],
            ('wi9ubMsI5cDptid12cm2', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[677, 683], [1162, 1197], [1339, 1364], [1475, 1531], [1573, 1610], [1634, 1685], [4376, 4419], [4445, 4468], [4517, 4554], [4588, 4616], [4651, 4684], [4719, 4771]],
            ('ZMqZkrMMB0PcsCeLhQqE', ('?b->Dodgeball|-02.60|+00.13|-02.18', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[1366, 1371], [1444, 1474]],
            ('T2dYV1nPh5cmt3NBELdS', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[1269, 1272], [1302, 1352], [1359, 1446], [1469, 1494], [3564, 3591], [3620, 3722], [3785, 4040], [4356, 4558], [4953, 5015], [5425, 5563]],
            ('HMosyugAD4iHS8V6ajD6', ('?b->Dodgeball|+00.44|+01.13|-02.80', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[204, 211]],
            ('tVL75ycOZHTAKgeqLs6H', ('?b->Dodgeball|-02.95|+01.29|-02.61', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[193, 230], [258, 295], [475, 653], [2162, 2368], [2641, 2658], [2709, 2760]],
            ('KO8pbUWEpZldxy7AzyM5', ('?b->Beachball|-02.93|+00.17|-01.99', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[0, 19]],
            ('Q6a8AbiIdcLA9tJzAu14', ('?b->Golfball|+01.05|+01.04|-02.70', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[6251, 6409], [6424, 6481]],
            ('Q1jHN8NAIlds8F5SFBCJ', ('?b->Beachball|-02.93|+00.17|-01.99', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[0, 17], [2266, 2280]],
            ('IjyNuz4ApTldgPuoKPtB', ('?b->Dodgeball|-02.60|+00.13|-02.18', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[900, 1208], [1752, 1782], [2044, 2107], [2453, 2506], [2525, 2604], [2644, 2741], [2780, 2814], [2946, 3010]],
            ('wi9ubMsI5cDptid12cm2', ('?b->Dodgeball|-02.95|+01.29|-02.61', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[764, 776], [1740, 1765], [1834, 1881], [1919, 1955], [1981, 2018], [2041, 2074], [2109, 2131], [2188, 2221], [2328, 2350], [2411, 2595], [3873, 3895], [3934, 3969], [3986, 3995], [4032, 4077], [4124, 4163], [4189, 4242], [4302, 4317], [4318, 4339], [4340, 4358]],
            ('moftPPoSfP3mG38GT9Vw', ('?b->Beachball|-02.93|+00.17|-01.99', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[0, 18]],
            ('T2dYV1nPh5cmt3NBELdS', ('?b->Dodgeball|-02.95|+01.29|-02.61', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[703, 745], [819, 989], [1030, 1075], [1079, 1081], [1084, 1154], [1155, 1158], [1174, 1193], [1468, 1573], [5561, 5621], [5679, 5734], [5781, 5815]],
            ('Q6a8AbiIdcLA9tJzAu14', ('?b->Dodgeball|+00.44|+01.13|-02.80', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[3823, 3871], [4012, 4128]],
            ('ZBcXIZbvTS3U4IBGk1zk', ('?b->Dodgeball|-02.60|+00.13|-02.18', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[527, 542], [664, 677], [776, 783], [909, 924], [1004, 1024], [1078, 1097]],
            ('9kyHjMMQRK3eP9vlLpw0', ('?b->Dodgeball|-02.60|+00.13|-02.18', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[161, 178], [206, 268], [552, 598], [651, 674], [675, 693], [717, 761], [772, 1008], [1410, 1469], [1496, 1725], [1817, 1842], [2028, 2133], [2164, 2207], [2208, 2211], [2348, 2370], [2528, 2551], [2802, 2822], [3038, 3088]],
            ('bPMm01GV1jVMlw7DIH89', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[233, 235], [275, 281], [295, 317], [331, 335], [368, 376], [393, 404], [431, 439], [520, 530], [555, 565], [591, 597], [624, 629], [652, 657], [681, 693]],
            ('ZMqZkrMMB0PcsCeLhQqE', ('?b->BasketBall|-02.58|+00.12|-01.93', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[1735, 1766]],
            ('nFX80fkvz9jp4TEfDQUl', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[338, 364], [418, 420], [743, 752], [845, 850]],
        }
    ),

    ExpectedCompletionsTestCase(
        test_id='test (and (not (in_motion ?b)) (in ?h ?b)))',
        game_key='test-ball-from-bed',
        predicate_selector=[4, 1, 'preferences', 0, 'definition', 'forall_pref', 'preferences', 'pref_body', 'body', 'exists_args', 'then_funcs', 2, 'seq_func', 'once_pred'],
        mapping={'?b': ['ball'], '?h': ['hexagonal_bin']},
        expected_intervals={
            ('NurIP8Ey70FomFLL0VSr', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[1062, 1063], [1068, 1074]],
            ('ZBcXIZbvTS3U4IBGk1zk', ('?b->Dodgeball|-02.60|+00.13|-02.18', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[1136, 1138]],
            ('wi9ubMsI5cDptid12cm2', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[1364, 1396]],
            ('Q1jHN8NAIlds8F5SFBCJ', ('?b->Dodgeball|-02.60|+00.13|-02.18', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[271, 274], [406, 411], [519, 571]],
            ('bPMm01GV1jVMlw7DIH89', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[439, 470]],
            ('9kyHjMMQRK3eP9vlLpw0', ('?b->Dodgeball|-02.60|+00.13|-02.18', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[2822, 2930]],
            ('Q6a8AbiIdcLA9tJzAu14', ('?b->Dodgeball|+00.70|+01.11|-02.80', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[3432, 3758], [3817, 3849]],
            ('HMosyugAD4iHS8V6ajD6', ('?b->Dodgeball|+00.19|+01.13|-02.80', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[332, 333]],
            ('tVL75ycOZHTAKgeqLs6H', ('?b->Dodgeball|-02.95|+01.29|-02.61', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[2658, 2674]],
            ('IjyNuz4ApTldgPuoKPtB', ('?b->BasketBall|-02.58|+00.12|-01.93', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[2524, 2529], [2535, 2536], [2544, 2548]],
            ('wi9ubMsI5cDptid12cm2', ('?b->Dodgeball|-02.95|+01.29|-02.61', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[4242, 4261]],
            ('IjyNuz4ApTldgPuoKPtB', ('?b->Dodgeball|-02.60|+00.13|-02.18', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[3010, 3028]],
            ('tVL75ycOZHTAKgeqLs6H', ('?b->Dodgeball|-02.97|+01.29|-02.28', '?h->GarbageCan|+00.95|-00.03|-02.68')): [[2016, 2018]],
            ('Q6a8AbiIdcLA9tJzAu14', ('?b->Golfball|+01.05|+01.04|-02.70', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[980, 981]],
            ('0xXJSvWhI0TJKqnPzY7O', ('?b->Dodgeball|+00.70|+01.11|-02.80', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[2000, 2025], [2135, 2181], [2393, 2427]],
            ('9kyHjMMQRK3eP9vlLpw0', ('?b->BasketBall|-02.58|+00.12|-01.93', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[339, 339]],
            ('moftPPoSfP3mG38GT9Vw', ('?b->Dodgeball|-02.60|+00.13|-02.18', '?h->GarbageCan|-02.79|-00.03|-02.67')): [[331, 861], [868, 870], [875, 878], [880, 1050]],
            ('Q6a8AbiIdcLA9tJzAu14', ('?b->Golfball|+00.96|+01.04|-02.70', '?h->GarbageCan|+00.75|-00.03|-02.74')): [[980, 981]],
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
