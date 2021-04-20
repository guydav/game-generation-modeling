import argparse
from pprint import pprint
import tatsu

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--grammar-file', default='./dsl.ebnf')
parser.add_argument('-t', '--test-file', default='./problems-few-objects.pddl')


def load_tests_from_file(path, start_token='(define', stop_token='(:constraints'):
    lines = open(path).readlines()
    # new_lines = []
    # for l in lines:
    #     if not l.strip()[0] == ';':
    #         print(l)
    #         new_lines.append(l[:l.find(';')])
    new_lines = [l[:l.find(';')] for l in lines 
        if len(l.strip()) > 0 and not l.strip()[0] == ';']
    text = ''.join(new_lines)
    results = []
    start = text.find(start_token)
    while start != -1:
        end = text.find(stop_token, start)
        next_start = text.find(start_token, start + 1)
        if end < next_start:  # we have a match
            results.append(text[start:end] + ')')
        start = next_start

    return results


def main(args):
    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    for test_case in load_tests_from_file(args.test_file):
        # print(test_case)
        ast = grammar_parser.parse(test_case)
        # pprint(ast, width=20, indent=4)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)




