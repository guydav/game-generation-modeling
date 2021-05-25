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


def _handle_quantifier(caller, rule, ast, depth, increment):
    formatted_vars = _parse_variable_list(ast[f'{rule}_vars'][1])
    _indent_print(f'({rule} ({" ".join(formatted_vars)})', depth, increment)

    found_args = False
    for key in QUANTIFIER_KEYS:
        key_str = f'{rule}_{key}'
        if key_str in ast:
            found_args = True
            caller(ast[key_str], depth + 1)
    
    if not found_args:
        raise ValueError(f'Found exists or forall with unknown arugments: {ast}')

    _indent_print(')', depth, increment)


def _handle_logical(caller, rule, ast, depth, increment):
    _indent_print(f'({rule}', depth, increment)
    caller(ast[f'{rule}_args'], depth + 1)
    _indent_print(f')', depth, increment)


def _handle_game(caller, rule, ast, depth, increment):
    _indent_print(f'({rule.replace("_", "-")}', depth, increment)
    caller(ast[f'{rule.replace("game_", "")}_pred'], depth + 1)
    _indent_print(f')', depth, increment)


def _handle_function_eval(caller, rule, ast, depth, increment):
    _indent_print(f'({ast.func_name} {ast.func_args and " ".join(ast.func_args) or ""})', depth, increment)


def _inline_format_function_eval(ast):
    return f'({ast.func_name} {" ".join(ast.func_args)})'


def _handle_function_comparison(caller, rule, ast, depth, increment):
    comp_op = '='
    if 'comp_op' in ast:
        comp_op = ast.comp_op

    if 'comp_func_1' in ast:
        args = [_inline_format_function_eval(ast.comp_func_1), _inline_format_function_eval(ast.comp_func_2)]

    elif 'comp_func_first' in ast:
        args = [_inline_format_function_eval(ast.comp_func_first), ast.comp_num]

    elif 'comp_func_second' in ast:
        args = [ast.comp_num, _inline_format_function_eval(ast.comp_func_second)]

    else:
        args = [_inline_format_function_eval(func) for func in ast.equal_comp_funcs]

    _indent_print(f'({comp_op} {" ".join(args)})', depth, increment)


def _handle_predicate(caller, rule, ast, depth, increment, return_str=False):
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
                    args.append(_handle_predicate(caller, rule, arg, depth + 1, increment, return_str=True))

    out = f'({name} {" ".join(args)})'
    if return_str:
        return out

    _indent_print(out, depth, increment)


class ASTPrinter:
    def __init__(self, ast_key, rule_name_substitutions):
        self.ast_key = ast_key
        self.rule_name_substitutions = rule_name_substitutions
        self.exact_matches = {}
        self.keyword_matches = []

    def register_exact_matches(self, *handlers):
        for handler in handlers:
            self.register_exact_match(handler)

    def register_exact_match(self, handler, rule_name=None):
        if rule_name is None:
            rule_name = handler.__name__.replace('_handle_', '')
        self.exact_matches[rule_name] = handler

    def register_keyword_match(self, keywords, handler):
        self.keyword_matches.append((keywords, handler))

    def _parse_base_cases(self, ast, depth, increment):
        if not ast:
            return True

        if isinstance(ast, tuple):
            _indent_print(ast[0], depth, increment)
            self(ast[1], depth + 1, increment)
            if len(ast) > 2:
                if len(ast) > 3:
                    raise ValueError(f'Unexpectedly long tuple: {ast}')
                
                _indent_print(ast[2], depth, increment)

            return True

        if isinstance(ast, list):
            for item in ast:
                self(item, depth, increment)
            return True

        if isinstance(ast, str):
            _indent_print(ast, depth, increment)
            return True

        return False

    def __call__(self, ast, depth=0, increment=DEFAULT_INCREMENT):
        if self._parse_base_cases(ast, depth, increment):
            return

        rule = ast.parseinfo.rule
        for sub in self.rule_name_substitutions:
            rule = rule.replace(sub, '')

        found_match = False

        if rule in self.exact_matches:
            found_match  = True
            self.exact_matches[rule](self, rule, ast, depth, increment)

        else:
            for keywords, handler in self.keyword_matches:
                if any([keyword in rule for keyword in keywords]):
                    found_match = True
                    handler(self, rule, ast, depth, increment)
                    break

        if not found_match:
            raise ValueError(f'No match found in {self.ast_key} for: {ast}')


def build_setup_printer():
    printer = ASTPrinter('(:setup', ('setup_', '_predicate'))
    printer.register_exact_matches(
        _handle_function_comparison, _handle_predicate, _handle_function_eval)
    printer.register_keyword_match(('exists', 'forall'), _handle_quantifier)
    printer.register_keyword_match(('game',), _handle_game)
    printer.register_keyword_match(('and', 'or', 'not'), _handle_logical)
    return printer


def _handle_preference(caller, rule, ast, depth, increment):
    _indent_print(f'(preference {ast.pref_name}', depth, increment)
    caller(ast.pref_body, depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_then(caller, rule, ast, depth, increment):
    _indent_print(f'(then', depth, increment)
    caller(ast.then_funcs, depth + 1, increment)
    _indent_print(f')', depth, increment)


def _handle_at_end(caller, rule, ast, depth, increment):
    _indent_print(f'(at-end', depth, increment)
    caller(ast.at_end_pred, depth + 1, increment)
    _indent_print(f')', depth, increment)


def _handle_any(caller, rule, ast, depth, increment):
    _indent_print('(any)', depth, increment)


def _handle_once(caller, rule, ast, depth, increment):
    _indent_print('(once', depth, increment)
    caller(ast.once_pred, depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_once_measure(caller, rule, ast, depth, increment):
    _indent_print('(once-measure', depth, increment)
    caller(ast.once_measure_pred, depth + 1, increment)
    _indent_print(_inline_format_function_eval(ast.measurement), depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_hold(caller, rule, ast, depth, increment):
    _indent_print('(hold', depth, increment)
    caller(ast.hold_pred, depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_while_hold(caller, rule, ast, depth, increment):
    _indent_print('(hold-while', depth, increment)
    caller([ast.hold_pred, ast.while_preds], depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_hold_for(caller, rule, ast, depth, increment):
    _indent_print(f'(hold-for {ast.num_to_hold}', depth, increment)
    caller(ast.hold_pred, depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_hold_to_end(caller, rule, ast, depth, increment):
    _indent_print('(hold-to-end', depth, increment)
    caller(ast.hold_pred, depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_forall_seq(caller, rule, ast, depth, increment):
    formatted_vars = _parse_variable_list(ast.forall_seq_vars[1])
    _indent_print(f'(forall-sequence ({" ".join(formatted_vars)})', depth, increment)
    caller(ast.forall_seq_then, depth + 1, increment)
    _indent_print(')', depth, increment)


def build_constraints_printer():
    printer = ASTPrinter('(:constraints', ('pref_body_', 'pref_', 'predicate_'))
    printer.register_exact_matches(
        _handle_preference, _handle_function_comparison, _handle_predicate,
        _handle_at_end, _handle_then, _handle_any, _handle_once, _handle_once_measure,
        _handle_hold, _handle_while_hold, _handle_hold_for, _handle_hold_to_end, 
        _handle_forall_seq, _handle_function_eval
    )
    printer.register_keyword_match(('exists', 'forall'), _handle_quantifier)
    printer.register_keyword_match(('and', 'or', 'not'), _handle_logical)
    return printer


def _handle_binary_comp(caller, rule, ast, depth, increment):
    _indent_print(f'({ast.op}', depth, increment)
    caller([ast.expr_1, ast.expr_2], depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_multi_expr(caller, rule, ast, depth, increment):
    _indent_print(f'({ast.op}', depth, increment)
    caller(ast.expr, depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_binary_expr(caller, rule, ast, depth, increment):
    _indent_print(f'({ast.op}', depth, increment)
    caller([ast.expr_1, ast.expr_2], depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_neg_expr(caller, rule, ast, depth, increment):
    _indent_print('(-', depth, increment)
    caller(ast.expr, depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_equals_comp(caller, rule, ast, depth, increment):
    _indent_print('(=', depth, increment)
    caller(ast.expr, depth + 1, increment)
    _indent_print(')', depth, increment)


def _handle_preference_eval(caller, rule, ast, depth, increment):
    _indent_print(f'({ast.parseinfo.rule.replace("_", "-")} {ast.pref_name})', depth, increment)


def build_terminal_printer():
    printer = ASTPrinter('(:terminal', ('terminal_', 'scoring_'))
    printer.register_exact_matches(
        _handle_multi_expr, _handle_binary_expr, _handle_neg_expr, 
        _handle_equals_comp, _handle_function_eval
    )
    printer.register_exact_match(_handle_binary_comp, 'comp')
    printer.register_keyword_match(('count',), _handle_preference_eval)
    printer.register_keyword_match(('and', 'or', 'not'), _handle_logical)
    return printer


def build_scoring_printer():
    printer = ASTPrinter('(:scoring', ('scoring_',))
    printer.register_exact_matches(
        _handle_multi_expr, _handle_binary_expr, _handle_neg_expr, 
        _handle_equals_comp, _handle_function_eval
    )
    printer.register_exact_match(_handle_binary_comp, 'comp')
    printer.register_keyword_match(('count',), _handle_preference_eval)
    printer.register_keyword_match(('and', 'or', 'not'), _handle_logical)
    return printer


PARSE_DICT = {
    '(:setup': build_setup_printer(),
    '(:constraints': build_constraints_printer(),
    '(:terminal': build_terminal_printer(),
    '(:scoring': build_scoring_printer(),
}


def pretty_print_ast(ast, increment=DEFAULT_INCREMENT):
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