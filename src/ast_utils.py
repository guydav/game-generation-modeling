import tatsu

import ast_printer


DEFAULT_TEST_FILES = (
    './problems-few-objects.pddl',
    './problems-medium-objects.pddl',
    './problems-many-objects.pddl'
)


def load_asts(args, grammar_parser, should_print=False):
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)

    if should_print:
        results = []
        for test_file in args.test_files:
            for game in load_tests_from_file(test_file):
                print(game)
                results.append(grammar_parser.parse(game))
        return results

    else:
        return [grammar_parser.parse(game) 
            for test_file in args.test_files 
            for game in load_tests_from_file(test_file)]


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
    text = '\n'.join(new_lines)
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


def copy_ast(grammar_parser, ast):
    ast_printer.reset_buffers(True)
    ast_printer.pretty_print_ast(ast)
    ast_str = ''.join(ast_printer.BUFFER)
    return grammar_parser.parse(ast_str)


def update_ast(ast, key, value):
    if isinstance(ast, tatsu.ast.AST):
        super(tatsu.ast.AST, ast).__setitem__(key, value)


def apply_selector_list(parent, selector, max_index=None):
    if max_index is None:
        max_index = len(selector)
    for s in selector[:max_index]:
        parent = parent[s]
    return parent


def replace_child(parent, selector, new_value):
    last_parent = apply_selector_list(parent, selector, -1)
    last_selector = selector[-1]

    if isinstance(last_selector, str):
        update_ast(last_parent, last_selector, new_value)
    
    elif isinstance(last_selector, int):
        last_parent[last_selector] = new_value

    else:
        raise ValueError(f'replace_child received last selector of unknown type: {last_selector} ({type(last_selector)})', parent, selector)


def find_all_parents(parent_mapping, ast):
    parents = []
    parent = parent_mapping[ast.parseinfo][1]
    while parent is not None and parent != 'root':
        parents.append(parent)
        if isinstance(parent, tuple):
            parent = None
        else:
            parent = parent_mapping[parent.parseinfo][1]

    return parents


def find_selectors_from_root(parent_mapping, ast, root_node='root'):
    selectors = []
    parent = ast
    while parent != root_node:
        _, parent, selector = parent_mapping[parent.parseinfo]
        selectors = selector + selectors

    return selectors
