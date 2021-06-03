import argparse
import tatsu
import tqdm

import ast_printer
from ast_printer import _indent_print, pretty_print_ast

parser = argparse.ArgumentParser()
DEFAULT_GRAMMAR_FILE = './dsl.ebnf'
parser.add_argument('-g', '--grammar-file', default=DEFAULT_GRAMMAR_FILE)
parser.add_argument('-s', '--stop-tokens', action='append')
parser.add_argument('-t', '--test-file', default='./problems-few-objects.pddl')
parser.add_argument('-p', '--pretty-print', action='store_true')
parser.add_argument('-v', '--validate', action='store_true')
parser.add_argument('-q', '--dont-tqdm', action='store_true')


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
        if end <= next_start or end == len(text):  # we have a match
            test_case = text[start:end]
            if end < next_start:
                test_case += ')'
            results.append(test_case)
        start = next_start

    return results



    
def main(args):
    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    test_cases = load_tests_from_file(args.test_file, stop_tokens=args.stop_tokens)
    if not args.dont_tqdm:
        test_cases = tqdm.tqdm(test_cases)

    for test_case in test_cases:
        ast_printer.BUFFER = None
        try:
            ast = grammar_parser.parse(test_case)
            if args.pretty_print:
                pretty_print_ast(ast)
            if args.validate:
                ast_printer.BUFFER = []
                pretty_print_ast(ast)
                first_print_out = ''.join(ast_printer.BUFFER)

                second_ast = grammar_parser.parse(first_print_out)
                ast_printer.BUFFER = []
                pretty_print_ast(second_ast)
                second_print_out = ''.join(ast_printer.BUFFER)

                if first_print_out != second_print_out:
                    print('Mismatch found')


        except (tatsu.exceptions.FailedToken, tatsu.exceptions.FailedParse) as e:
            print(test_case[:test_case.find('(:domain')])
            print(f'Parse failed: at position {e.pos} expected {e.item}:')
            print(test_case[e.pos:])
            break

        finally:
            pass
        # pprint(ast, width=20, depth=4)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)





