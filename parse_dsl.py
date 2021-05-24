import argparse
import tatsu
import tqdm

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
        if end <= next_start:  # we have a match
            test_case = text[start:end]
            if end < next_start:
                test_case += ')'
            results.append(test_case)
        start = next_start

    return results


DEFAULT_INCREMENT = '  '
BUFFER = None


def _indent_print(str, depth, increment):
    global BUFFER

    out = f'{increment * depth}{str}'
    if BUFFER is None:
        print(out)
    else:
        BUFFER.append(out)


def _parse_variable_list(var_list):
    formatted_vars = []
    for vars in var_list:
        if isinstance(vars[2], str):
            formatted_vars.append(f'{" ".join(vars[0])} - {vars[2]}')
        elif vars[2].parseinfo.rule == 'either_types':
            formatted_vars.append(f'{" ".join(vars[0])} - (either {" ".join(vars[2].type_names)})')
        else:
            raise ValueError(f'Unrecognized quantifier variables: {vars[2]}')
    return formatted_vars


QUANTIFIER_KEYS = ('args', 'pred', 'then')


def _handle_quantifier(caller, name, ast, depth, increment, arg_keys=QUANTIFIER_KEYS):
    formatted_vars = _parse_variable_list(ast[f'{name}_vars'][1])
    _indent_print(f'({name} ({" ".join(formatted_vars)})', depth, increment)

    found_args = False
    for key in arg_keys:
        key_str = f'{name}_{key}'
        if key_str in ast:
            found_args = True
            caller(ast[key_str], depth + 1)
    
    if not found_args:
        raise ValueError(f'Found exists or forall with unknown arugments: {ast}')

    _indent_print(')', depth, increment)


def _handle_logical(caller, name, ast, depth, increment):
    _indent_print(f'({name}', depth, increment)
    caller(ast[f'{name}_args'], depth + 1)
    _indent_print(f')', depth, increment)


def _handle_game(name, ast, depth, increment):
    _indent_print(f'({name.replace("_", "-")}', depth, increment)
    parse_setup(ast[f'{name.replace("game_", "")}_pred'], depth + 1)
    _indent_print(f')', depth, increment)


def _format_function_eval(ast):
    return f'({ast.func_name} {" ".join(ast.func_args)})'


def _handle_function_comparison(ast, depth, increment):
    comp_op = '='
    if 'comp_op' in ast:
        comp_op = ast.comp_op

    if 'comp_func_1' in ast:
        args = [_format_function_eval(ast.comp_func_1), _format_function_eval(ast.comp_func_2)]

    elif 'comp_func_first' in ast:
        args = [_format_function_eval(ast.comp_func_first), ast.comp_num]

    elif 'comp_func_second' in ast:
        args = [ast.comp_num, _format_function_eval(ast.comp_func_second)]

    else:
        args = [_format_function_eval(func) for func in ast.equal_comp_funcs]

    _indent_print(f'({comp_op} {" ".join(args)})', depth, increment)


def _handle_predicate(ast, depth, increment, return_str=False):
    name = ast.pred_name
    args = []
    if ast.pred_args:
        if isinstance(ast.pred_args, str):
            args.append(ast.pred_args)
        else:
            for arg in ast.pred_args:
                if isinstance(arg, str):
                    args.append(arg)
                else:
                    args.append(_handle_predicate(arg, depth + 1, increment, return_str=True))

    out = f'({name} {" ".join(args)})'
    if return_str:
        return out

    _indent_print(out, depth, increment)


def _check_simple_cases(caller, ast, depth, increment):
    if not ast:
        return True

    if isinstance(ast, tuple):
        _indent_print(ast[0], depth, increment)
        caller(ast[1], depth + 1, increment)
        if len(ast) > 2:
            if len(ast) > 3:
                raise ValueError(f'Unexpectedly long tuple: {ast}')
            
            _indent_print(ast[2], depth, increment)

        return True

    if isinstance(ast, list):
        for item in ast:
            caller(item, depth, increment)
        return True

    if isinstance(ast, str):
        _indent_print(ast, depth, increment)
        return True

    return False


def parse_setup(ast, depth=0, increment=DEFAULT_INCREMENT):
    if _check_simple_cases(parse_setup, ast, depth, increment):
        return

    rule = ast.parseinfo.rule.replace('setup_', '')

    if rule == 'function_comparison':
        _handle_function_comparison(ast, depth, increment)

    elif rule == 'predicate':
        _handle_predicate(ast, depth, increment)
    
    elif 'exists' in rule or 'forall' in rule:
        _handle_quantifier(parse_setup, rule, ast, depth, increment)

    elif 'and' in rule or 'or' in rule or 'not' in rule:
        _handle_logical(parse_setup, rule.replace('_predicate', ''), ast, depth, increment)

    elif 'game' in rule:
        _handle_game(rule, ast, depth, increment)
    
    else:
        raise ValueError(f'Found unknown ast element: {ast}')


def _handle_preference(ast, depth, increment):
    _indent_print(f'(preference {ast.pref_name}', depth, increment)
    parse_constraints(ast.pref_body, depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_then(ast, depth, increment):
    _indent_print(f'(then', depth, increment)
    parse_constraints(ast.then_funcs, depth + 1, increment)
    _indent_print(f')', depth, increment)


def _handle_at_end(ast, depth, increment):
    _indent_print(f'(at-end', depth, increment)
    parse_constraints(ast.at_end_pred, depth + 1, increment)
    _indent_print(f')', depth, increment)


def _handle_once(ast, depth, increment):
    _indent_print('(once', depth, increment)
    parse_constraints(ast.once_pred, depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_once_measure(ast, depth, increment):
    _indent_print('(once-measure', depth, increment)
    parse_constraints(ast.once_measure_pred, depth + 1, increment)
    _indent_print(_format_function_eval(ast.measurement), depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_hold(ast, depth, increment):
    _indent_print('(hold', depth, increment)
    parse_constraints(ast.hold_pred, depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_while_hold(ast, depth, increment):
    _indent_print('(hold-while', depth, increment)
    parse_constraints([ast.hold_pred, ast.while_preds], depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_hold_for(ast, depth, increment):
    _indent_print(f'(hold-for {ast.num_to_hold}', depth, increment)
    parse_constraints(ast.hold_pred, depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_hold_to_end(ast, depth, increment):
    _indent_print('(hold-to-end', depth, increment)
    parse_constraints(ast.hold_pred, depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_forall_seq(ast, depth, increment):
    formatted_vars = _parse_variable_list(ast.forall_seq_vars[1])
    _indent_print(f'(forall-sequence ({" ".join(formatted_vars)})', depth, increment)
    parse_constraints(ast.forall_seq_then, depth + 1, increment)
    _indent_print(')', depth, increment)


def parse_constraints(ast, depth=0, increment=DEFAULT_INCREMENT):
    if _check_simple_cases(parse_constraints, ast, depth, increment):
        return

    if isinstance(ast, tuple) and ast[0] == '(and':
        _indent_print(ast[0], depth, increment)
        parse_constraints(ast[1], depth + 1)
        _indent_print(ast[2], depth, increment)
        return

    rule = ast.parseinfo.rule.replace('pref_body_', '').replace('pref_', '').replace('predicate_', '')

    if rule == 'preference':
        _handle_preference(ast, depth, increment)

    elif rule == 'function_comparison':
        _handle_function_comparison(ast, depth, increment)

    elif rule == 'predicate':
        _handle_predicate(ast, depth, increment)
    
    elif rule == 'at_end':
        _handle_at_end(ast, depth, increment)

    elif rule == 'then':
        _handle_then(ast, depth, increment)

    elif rule == 'any':
        _indent_print('(any)', depth, increment)

    elif rule == 'once':
        _handle_once(ast, depth, increment)

    elif rule == 'once_measure':
        _handle_once_measure(ast, depth, increment)

    elif rule == 'hold':
        _handle_hold(ast, depth, increment)

    elif rule == 'while_hold':
        _handle_while_hold(ast, depth, increment)

    elif rule == 'hold_for':
        _handle_hold_for(ast, depth, increment)

    elif rule == 'hold_to_end':
        _handle_hold_to_end(ast, depth, increment)

    elif rule == 'forall_seq':
        _handle_forall_seq(ast, depth, increment)

    elif 'exists' in rule or 'forall' in rule:
        _handle_quantifier(parse_constraints, rule, ast, depth, increment)

    elif 'and' in rule or 'or' in rule or 'not' in rule:
        _handle_logical(parse_constraints, rule, ast, depth, increment)

    else:
        raise ValueError(f'Found unknown ast element: {ast}')


def _handle_binary_comp(caller, ast, depth, increment):
    _indent_print(f'({ast.op}', depth, increment)
    caller([ast.expr_1, ast.expr_2], depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_multi_expr(caller, ast, depth, increment):
    _indent_print(f'({ast.op}', depth, increment)
    caller(ast.expr, depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_binary_expr(caller, ast, depth, increment):
    _indent_print(f'({ast.op}', depth, increment)
    caller([ast.expr_1, ast.expr_2], depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_neg_expr(caller, ast, depth, increment):
    _indent_print('(-', depth, increment)
    caller(ast.expr, depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_equals_comp(caller, ast, depth, increment):
    _indent_print('(=', depth, increment)
    caller(ast.expr, depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_preference_eval(caller, ast, depth, increment):
    _indent_print(f'({ast.parseinfo.rule.replace("_", "-")} {ast.pref_name})', depth, increment)


def parse_terminal(ast, depth=0, increment=DEFAULT_INCREMENT):
    if _check_simple_cases(parse_terminal, ast, depth, increment):
        return

    rule = ast.parseinfo.rule.replace('terminal_', '').replace('scoring_', '')

    if rule == 'comp':
        _handle_binary_comp(parse_terminal, ast, depth, increment)

    elif rule == 'multi_expr':
        _handle_multi_expr(parse_terminal, ast, depth, increment)

    elif rule == 'binary_expr':
        _handle_binary_expr(parse_terminal, ast, depth, increment)

    elif rule == 'neg_expr':
        _handle_neg_expr(parse_terminal, ast, depth, increment)

    elif rule == 'equals_comp':
        _handle_equals_comp(parse_terminal, ast, depth, increment)

    elif 'and' in rule or 'or' in rule or 'not' in rule:
        _handle_logical(parse_terminal, rule, ast, depth, increment)
    
    elif 'count' in rule:
        _handle_preference_eval(parse_terminal, ast, depth, increment)



def parse_scoring(ast, depth=0, increment=DEFAULT_INCREMENT):
    if _check_simple_cases(parse_scoring, ast, depth, increment):
        return

    rule = ast.parseinfo.rule.replace('scoring_', '')

    if 'count' in rule:
        _handle_preference_eval(parse_scoring, ast, depth, increment)

    elif rule == 'comp':
        _handle_binary_comp(parse_scoring, ast, depth, increment)

    elif rule == 'multi_expr':
        _handle_multi_expr(parse_scoring, ast, depth, increment)

    elif rule == 'binary_expr':
        _handle_binary_expr(parse_scoring, ast, depth, increment)

    elif rule == 'neg_expr':
        _handle_neg_expr(parse_scoring, ast, depth, increment)

    elif rule == 'equals_comp':
        _handle_equals_comp(parse_scoring, ast, depth, increment)

    elif 'and' in rule or 'or' in rule or 'not' in rule:
        _handle_logical(parse_scoring, rule, ast, depth, increment)


PARSE_DICT = {
    '(:setup': parse_setup,
    '(:constraints': parse_constraints,
    '(:terminal': parse_terminal,
    '(:scoring': parse_scoring,
}


def pretty_print(ast, increment=DEFAULT_INCREMENT):
    _indent_print(f'{ast[0]} (game {ast[1]["game_name"]}) (:domain {ast[2]["domain_name"]})', 0, increment)
    ast = ast[3:]

    while ast:
        key = ast[0][0]

        if key == ')':
            _indent_print(f')', 0, increment)
            ast = None

        elif key in PARSE_DICT:
            PARSE_DICT[key](ast[0], 0, increment)
            ast = ast[1:]

        else:
            print(f'Encountered unknown key: {key}\n')

    
def main(args):
    global BUFFER

    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    test_cases = load_tests_from_file(args.test_file, stop_tokens=args.stop_tokens)
    if not args.dont_tqdm:
        test_cases = tqdm.tqdm(test_cases)

    for test_case in test_cases:
        BUFFER = None
        try:
            ast = grammar_parser.parse(test_case)
            if args.pretty_print:
                pretty_print(ast)
            if args.validate:
                BUFFER = []
                pretty_print(ast)
                first_print_out = ''.join(BUFFER)

                second_ast = grammar_parser.parse(first_print_out)
                BUFFER = []
                pretty_print(second_ast)
                second_print_out = ''.join(BUFFER)

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





