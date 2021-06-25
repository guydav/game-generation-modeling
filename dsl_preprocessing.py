import argparse
from collections import defaultdict
import tatsu
from mutation_experiments import copy_ast, update, load_asts

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
DEFAULT_OUTPUT_PATH ='./preprocessing_examples.csv'
parser.add_argument('-o', '--output-path', default=DEFAULT_OUTPUT_PATH)


PREPROCESS_SUBSTITUTIONS = {
    'floor': 'floor_obj',
    'desk': 'desk_obj',
    'chair': 'chair_obj',
}


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
        update(last_parent, last_selector, new_value)
    
    elif isinstance(last_selector, int):
        last_parent[last_selector] = new_value

    else:
        raise ValueError(f'replace_child received last selector of unknown type: {last_selector} ({type(last_selector)})', parent, selector)


def map_parents_recursive(ast, mapping=None, parent='root', selector=None):
    if mapping is None:
        mapping = {}

    if selector is None:
        selector = []
    
    if not ast:
        return

    if isinstance(ast, (str, int, tatsu.buffering.Buffer)):
        return
    
    elif isinstance(ast, (tuple, list)): 
        [map_parents_recursive(element, mapping, parent, selector + [i]) 
         for i, element in enumerate(ast)]

    elif isinstance(ast, tatsu.ast.AST):
        mapping[ast.parseinfo] = (parent, selector)

        for key in ast:
            if key != 'parseinfo':
                map_parents_recursive(ast[key], mapping, ast, [key])

    return mapping


def extract_substitutions_from_vars(ast, vars_key, preprocess_substitutions):
    substitutions = {}
    var_defs_to_remove = []
    for var_def in ast[vars_key].variables:
        # TODO: what do I do if something being substitued is part of an (either ...)?
        if isinstance(var_def.var_type, str) and var_def.var_type in preprocess_substitutions:
            var_defs_to_remove.append(var_def)
            for name in var_def.var_names:
                substitutions[name] = preprocess_substitutions[var_def.var_type]

    [ast[vars_key].variables.remove(var_def) for var_def in var_defs_to_remove]

    # TODO: what do we do if after this no variable definitions remain in this quantifier?
    return substitutions, len(ast[vars_key].variables) == 0


def preprocess_ast_recursive(ast, preprocess_substitutions, parent_mapping=None, local_substitutions=None):
    if parent_mapping is None:
        parent_mapping = map_parents_recursive(ast)
    
    if local_substitutions is None:
        local_substitutions = {}
    
    if not ast or isinstance(ast, (str, int, tatsu.buffering.Buffer)):
        return
    
    elif isinstance(ast, (tuple, list)):
        [preprocess_ast_recursive(element, preprocess_substitutions, parent_mapping, local_substitutions) for element in ast]

    elif isinstance(ast, tatsu.ast.AST):
        args_key = None

        if 'pred_args' in ast:
            args_key = 'pred_args'
        elif 'func_args' in ast:
            args_key = 'func_args'

        if args_key is not None:
            for i, arg in enumerate(ast[args_key]):
                if isinstance(arg, str):
                    if arg in local_substitutions:
                        ast[args_key][i] = local_substitutions[arg]
                else:
                    preprocess_ast_recursive(arg, preprocess_substitutions, parent_mapping, local_substitutions)
        # else:
        #     raise ValueError(f'Encountered unexpected type for {ast.parseinfo}.{args_key}: {ast[args_key]}', ast)
        else:
            vars_keys = [key for key in ast.keys() if key.endswith('_vars')]
            if len(vars_keys) > 1:
                raise ValueError(f'Found multiple variables keys: {vars_keys}', ast)

            elif len(vars_keys) > 0:
                vars_key = vars_keys[0]
                args_keys = [key for key in ast.keys() if key.startswith(vars_key.replace('_vars', ''))]
                args_keys.remove(vars_key)
                
                if len(args_keys) > 1:
                    raise ValueError(f'Found too many argument keys under: {args_keys}', ast, ast.keys())

                args_key = args_keys[0]

                local_subs, remove_quantifier = extract_substitutions_from_vars(ast, vars_key, preprocess_substitutions)
                local_substitutions.update(local_subs)

                preprocess_ast_recursive(ast[args_key], preprocess_substitutions, parent_mapping, local_substitutions)

                if remove_quantifier:
                    parent, selector = parent_mapping[ast.parseinfo]
                    replace_child(parent, selector, ast[args_key])

                for key in local_subs:
                    if key in local_substitutions:
                        del local_substitutions[key]

            else:
                for key in ast:
                    if key != 'parseinfo':
                        preprocess_ast_recursive(ast[key], preprocess_substitutions, parent_mapping, local_substitutions)

def preprocess_ast(grammar_parser, ast, preprocess_substitutions=PREPROCESS_SUBSTITUTIONS):
    processed_ast = copy_ast(grammar_parser, ast)
    preprocess_ast_recursive(processed_ast, preprocess_substitutions)
    return processed_ast


def main(args):
    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar) 

    asts = load_asts(args, grammar_parser)
    for ast in asts:
        processed_ast = preprocess_ast(grammar_parser, ast)
        ast_printer.reset_buffers(False)
        ast_printer.pretty_print_ast(processed_ast, context=dict())
        print('\r\n' + '=' * 100 + '\r\n')


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)
    
    main(args)
