import argparse
import sys
import tatsu
import tatsu.exceptions
import tqdm

import ast_printer
from ast_utils import cached_load_and_parse_games_from_file


parser = argparse.ArgumentParser()
DEFAULT_GRAMMAR_FILE = './dsl/dsl.ebnf'
parser.add_argument('-g', '--grammar-file', default=DEFAULT_GRAMMAR_FILE)
parser.add_argument('-s', '--stop-tokens', action='append')
parser.add_argument('-t', '--test-file', default='./dsl/interactive-beta.pddl')
parser.add_argument('-p', '--pretty-print', action='store_true')
parser.add_argument('-v', '--validate', action='store_true')
parser.add_argument('-q', '--dont-tqdm', action='store_true')
DEFAULT_RECURSION_LIMIT = 2000
parser.add_argument('--recursion-limit', type=int, default=DEFAULT_RECURSION_LIMIT)
parser.add_argument('--force-rebuild-cache', action='store_true')
parser.add_argument('--expected-total', type=int, default=None)

def main(args):
    original_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(args.recursion_limit)

    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    test_cases = cached_load_and_parse_games_from_file(args.test_file, grammar_parser, False, # type: ignore
                                                       '.', 1024, False, force_rebuild=args.force_rebuild_cache)
    if not args.dont_tqdm:
        test_cases = tqdm.tqdm(test_cases, total=args.expected_total)

    for ast in test_cases:
        ast_printer.BUFFER = None
        try:
            if args.pretty_print:
                ast_printer.pretty_print_ast(ast)
            if args.validate:
                ast_printer.BUFFER = []
                ast_printer.pretty_print_ast(ast)
                first_print_out = ''.join(ast_printer.BUFFER)

                second_ast = grammar_parser.parse(first_print_out)
                ast_printer.BUFFER = []
                ast_printer.pretty_print_ast(second_ast)
                second_print_out = ''.join(ast_printer.BUFFER)

                if first_print_out != second_print_out:
                    print('Mismatch found')


        except (tatsu.exceptions.FailedToken, tatsu.exceptions.FailedParse) as e:
            raise e
            # print(test_case[:test_case.find('(:domain')])
            # print(f'Parse failed: at position {e.pos} expected {e.item}')
            # print(test_case[e.pos:])
            # break

        finally:
            pass
        # pprint(ast, width=20, depth=4)

    sys.setrecursionlimit(original_recursion_limit)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
