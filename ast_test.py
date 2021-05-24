from parse_dsl import *
grammar = tatsu.compile(open('./dsl.ebnf').read())
test_file = './problems-few-objects.pddl'
test_cases = load_tests_from_file(test_file)
#  asts = [grammar.parse(case) for case in test_cases]


def get_ast(i):
    return grammar.parse(test_cases[i])

