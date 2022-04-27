import argparse
import tatsu
import tqdm

import ast_printer
from ast_utils import load_tests_from_file

parser = argparse.ArgumentParser()
DEFAULT_GRAMMAR_FILE = './dsl/dsl.ebnf'
parser.add_argument('-g', '--grammar-file', default=DEFAULT_GRAMMAR_FILE)
parser.add_argument('-s', '--stop-tokens', action='append')
parser.add_argument('-t', '--test-file', default='./dsl/interactive-beta.pddl')
parser.add_argument('-p', '--pretty-print', action='store_true')
parser.add_argument('-v', '--validate', action='store_true')
parser.add_argument('-q', '--dont-tqdm', action='store_true')


    
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





