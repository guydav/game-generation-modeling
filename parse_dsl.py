import argparse
from pprint import pprint
import tatsu

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--grammar-file', default='./dsl.ebnf')
parser.add_argument('-s', '--stop-tokens', action='append')
parser.add_argument('-t', '--test-file', default='./problems-few-objects.pddl')

DEFAULT_STOP_TOKENS = ('(define', )  # ('(:constraints', )
def load_tests_from_file(path, start_token='(define', stop_tokens=None):
    if stop_tokens is None or not stop_tokens:
        stop_tokens = DEFAULT_STOP_TOKENS

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
        end_matches = [text.find(stop_token, start + 1) for stop_token in stop_tokens]
        end_matches = [match != -1 and match or len(text) for match in end_matches]
        end = min(end_matches)
        next_start = text.find(start_token, start + 1)
        if end < next_start:  # we have a match
            results.append(text[start:end] + ')')
        start = next_start

    return results


def main(args):
    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    for test_case in load_tests_from_file(args.test_file, stop_tokens=args.stop_tokens):
        # print(test_case)
        try:
            ast = grammar_parser.parse(test_case)

        except tatsu.exceptions.FailedToken as e:
            print(test_case[:test_case.find('(:domain')])
            print(f'Parse failed: at position {e.pos} expected {e.item} for token {e.token}:')
            print(test_case[e.pos:])
            break

        finally:
            pass
        # pprint(ast, width=20, indent=4)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)




