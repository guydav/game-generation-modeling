import argparse
from collections import defaultdict
import tatsu
import random

from parse_dsl import load_tests_from_file
import ast_printer


parser = argparse.ArgumentParser()
DEFAULT_GRAMMAR_FILE = './dsl.ebnf'
parser.add_argument('-g', '--grammar-file', default=DEFAULT_GRAMMAR_FILE)
DEFAULT_TEST_FILES = (
    './problems-few-objects.pddl',
    './problems-medium-objects.pddl',
    './problems-many-objects.pddl'
)
parser.add_argument('-t', '--test-files', action='append', default=[])
DEFAULT_NUM_GAMES = 100
parser.add_argument('-n', '--num-games', default=DEFAULT_NUM_GAMES)
# DEFAULT_OUTPUT_PATH ='./dsl_statistics.csv'
# parser.add_argument('-o', '--output-path', default=DEFAULT_OUTPUT_PATH)


def copy_ast(grammar_parser, ast):
    ast_printer.reset_buffers(True)
    ast_printer.pretty_print_ast(ast)
    ast_str = ''.join(ast_printer.BUFFER)
    return grammar_parser.parse(ast_str)


def update(ast, key, value):
    if isinstance(ast, tatsu.ast.AST):
        super(tatsu.ast.AST, ast).__setitem__(key, value)


def extract_pref_args(pref):
    pref_body_keys = pref.pref_body.keys()
    pref_type = None

    if any(['exists' in key for key in pref_body_keys]):
        pref_type = 'exists'
    elif any(['forall' in key for key in pref_body_keys]):
        pref_type = 'forall'
    else:
        return None, pref.pref_body

    return extract_valid_vars(pref.pref_body[f'{pref_type}_vars']), pref.pref_body[f'{pref_type}_args']


def extract_then_funcs(pref):
    _, pref_body_args = extract_pref_args(pref)
    
    if not pref_body_args.parseinfo or pref_body_args.parseinfo.rule != 'then':
        return

    then_funcs = pref_body_args.then_funcs

    if isinstance(then_funcs, tatsu.ast.AST):
        return [then_funcs]

    return then_funcs


def extract_valid_vars(pref_vars):
    valid_vars = []
    for var_def in pref_vars.variables:
        valid_vars.extend(var_def.var_names)

    return valid_vars


def mutate_seq_func_predicate(grammar_parser, asts, notebook=False, seed=None):
    if seed is not None:
        random.seed(seed)

    orig_ast, mut_ast = random.sample(asts, k=2)
    new_ast = copy_ast(grammar_parser, orig_ast)
    new_mut_ast = copy_ast(grammar_parser, mut_ast)

    preferences = new_ast[4][1][1]
    if not preferences:
        return

    pref_to_edit = random.choice(preferences)
    pref_body_valid_vars, pref_body_args = extract_pref_args(pref_to_edit)
    
    if not pref_body_args.parseinfo or pref_body_args.parseinfo.rule != 'then':
        return

    then_funcs = pref_body_args.then_funcs
    seq_func_index_to_modify = random.choice(range(len(then_funcs)))

    mut_preferences = new_mut_ast[4][1][1]
    if not mut_preferences:
        return
    
    mut_pref = random.choice(mut_preferences)
    mut_then_funcs = extract_then_funcs(mut_pref)
    if not mut_then_funcs:
        return
    
    mut_seq_func = random.choice(mut_then_funcs)
    try:
        replace_variables(mut_seq_func, pref_body_valid_vars)
    except ValueError:
        return
    # print(mut_seq_func)
    # print()
    if notebook:
        update(mut_seq_func, 'mutation', 'new')
        if isinstance(then_funcs, tatsu.ast.AST):
            update(pref_body_args, 'then_funcs', mut_seq_func)

        else:
            update(then_funcs[seq_func_index_to_modify], 'mutation', 'old')
            then_funcs.insert(seq_func_index_to_modify + 1, mut_seq_func)
    else:
        then_funcs[seq_func_index_to_modify] = mut_seq_func

    return new_ast, pref_to_edit.pref_name, seq_func_index_to_modify


def replace_variables(ast, pref_valid_vars, local_valid_vars=None):
    if local_valid_vars is None:
        local_valid_vars = []
    
    if not ast or isinstance(ast, (str, int, tatsu.buffering.Buffer)):
        return
    
    elif isinstance(ast, (tuple, list)):
        [replace_variables(element, pref_valid_vars, local_valid_vars) for element in ast]

    elif isinstance(ast, tatsu.ast.AST):
        args_key = None

        if 'pred_args' in ast:
            args_key = 'pred_args'
        elif 'func_args' in ast:
            args_key = 'func_args'

        if args_key is not None:
            pref_valid_vars_copy = pref_valid_vars[:]
            # if a variable already exists in this context, don't use it to replace
            for arg in ast[args_key]:
                if arg in pref_valid_vars_copy:
                    pref_valid_vars_copy.remove(arg)

            for i, arg in enumerate(ast[args_key]):
                if isinstance(arg, str):
                    # check it's a variable and that it's valid in the preference or local context
                    if arg.startswith('?') and arg not in pref_valid_vars and arg not in local_valid_vars:
                        if not pref_valid_vars_copy:
                            raise ValueError(f'In replace_variables, tried to sample too many different valid variables in a single predicate')

                        ast[args_key][i] = random.choice(pref_valid_vars_copy)
                        pref_valid_vars_copy.remove(ast[args_key][i])
                else:
                    replace_variables(arg, pref_valid_vars, local_valid_vars)
            
        else:
            vars_keys = [key for key in ast.keys() if key.endswith('_vars')]
            if len(vars_keys) > 1:
                raise ValueError(f'Found multiple variables keys: {vars_keys}', ast)

            elif len(vars_keys) > 0:
                vars_key = vars_keys[0]
                lv = extract_valid_vars(ast[vars_key])
                local_valid_vars.extend(lv)

                inner_keys = [key for key in ast.keys() if key.startswith(vars_key.replace('_vars', ''))]
                inner_keys.remove(vars_key)
                
                if len(inner_keys) > 1:
                    raise ValueError(f'Found too many inner keys: {inner_keys}', ast, ast.keys())

                inner_key = inner_keys[0]

                replace_variables(ast[inner_key], pref_valid_vars, local_valid_vars)
                [local_valid_vars.remove(v) for v in lv]
            
            else:
                for key in ast:
                    if key != 'parseinfo':
                        replace_variables(ast[key], pref_valid_vars, local_valid_vars)
    else:
        raise ValueError(f'In `replace_variables`, found variable of unknown type ({type(ast)}): {ast}')


def mutate_single_game(grammar_parser, asts, notebook=False, start_seed=None):
    result = None
    seed = start_seed
    while not result:
        result = mutate_seq_func_predicate(grammar_parser, asts, notebook=notebook, seed=seed)
        if seed is not None and result is None:
            seed += 1

    return (*result, seed)


def load_asts(args, grammar_parser):
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)

    return [grammar_parser.parse(game) 
        for test_file in args.test_files 
        for game in load_tests_from_file(test_file)]


def main(args):
    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar) 

    asts = load_asts(args, grammar_parser)

    for i in range(args.num_games):
        new_ast, pref_name, seq_func_index, seed = mutate_single_game(grammar_parser, asts, notebook=True, start_seed=i * args.num_games)
        print(f'With random seed {seed}, mutated sequence function #{seq_func_index} in "{pref_name}":\r\n')
        ast_printer.reset_buffers(False)
        ast_printer.pretty_print_ast(new_ast, context=dict(html=True))
        print('\r\n' + '=' * 100 + '\r\n')


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)
    
    main(args)