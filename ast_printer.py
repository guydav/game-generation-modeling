from ast_parser import ASTParser


DEFAULT_INCREMENT = '  '
BUFFER = None
LINE_BUFFER = None


class ASTPrinter(ASTParser):
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

    def _handle_tuple(self, ast, **kwargs):
        _indent_print(ast[0], kwargs['depth'], kwargs['increment'], kwargs['context'])
        self(ast[1], depth=kwargs['depth'] + 1, increment=kwargs['increment'], context=kwargs['context'])
        if len(ast) > 2:
            if len(ast) > 3:
                raise ValueError(f'Unexpectedly long tuple: {ast}')
            
            _indent_print(ast[2], kwargs['depth'], kwargs['increment'], kwargs['context'])

    def _handle_str(self, ast, **kwargs):
        return _indent_print(ast, kwargs['depth'], kwargs['increment'], kwargs['context'])

    def _handle_ast(self, ast, **kwargs):
        rule = ast.parseinfo.rule
        for sub in self.rule_name_substitutions:
            rule = rule.replace(sub, '')

        found_match = False

        depth, increment, context = kwargs['depth'], kwargs['increment'], kwargs['context']

        if rule in self.exact_matches:
            found_match  = True
            self.exact_matches[rule](self, rule, ast, depth, increment, context)

        else:
            for keywords, handler in self.keyword_matches:
                if any([keyword in rule for keyword in keywords]):
                    found_match = True
                    handler(self, rule, ast, depth, increment, context)
                    break

        if not found_match:
            raise ValueError(f'No match found in {self.ast_key} for: {ast}')


    def __call__(self, ast, depth=0, increment=DEFAULT_INCREMENT, context=None):
        kwargs = dict(depth=depth, increment=increment, context=context)
        return super().__call__(ast, **kwargs)


def reset_buffers(to_list=True):
    global BUFFER, LINE_BUFFER

    if to_list:
        BUFFER = []
        LINE_BUFFER = []

    else:
        BUFFER = None
        LINE_BUFFER = None


def preprocess_context(context):
    if not context:
        context = {}

    if not 'html_style' in context:
        context['html_style'] = {}

    if 'color' in context['html_style']: del context['html_style']['color']
    if 'text-decoration' in context['html_style']: del context['html_style']['text-decoration']

    if 'mutation' in context:
        if context['mutation'] == 'old':
            context['html_style']['text-decoration'] = 'line-through'

        elif context['mutation'] == 'new':
            context['html_style']['color'] = '#FF6A00'

    return context
            

def _indent_print(str, depth, increment, context=None):
    global BUFFER, LINE_BUFFER

    context = preprocess_context(context)

    if 'continue_line' in context and context['continue_line']:
        LINE_BUFFER.append(str)

    else:
        if LINE_BUFFER:
            if 'html' in context and any([bool(s and not s.isspace()) for s in LINE_BUFFER]):
                LINE_BUFFER.append('</div>')

            line = ' '.join(LINE_BUFFER)
            if BUFFER is None:
                print(line)
            else:
                BUFFER.append(line)

        if 'html' in context:
            context['html_style']['margin-left'] = f'{20 * depth}px'

            LINE_BUFFER = [f'<div style="{"; ".join({f"{k}: {v}" for k, v in context["html_style"].items()})}">{str}']
        else:
            LINE_BUFFER = [f'{increment * depth}{str}']
        


def _parse_variable_list(var_list):
    formatted_vars = []
    for var_def in var_list:
        if isinstance(var_def.var_type, str):
            formatted_vars.append(f'{" ".join(var_def.var_names)} - {var_def.var_type}')
        elif var_def.var_type.parseinfo.rule == 'either_types':
            formatted_vars.append(f'{" ".join(var_def.var_names)} - (either {" ".join(var_def.var_type.type_names)})')
        else:
            raise ValueError(f'Unrecognized quantifier variables: {var_def[2]}')
    return formatted_vars


QUANTIFIER_KEYS = ('args', 'pred', 'then')


def _handle_quantifier(caller, rule, ast, depth, increment, context=None):
    formatted_vars = _parse_variable_list(ast[f'{rule}_vars'].variables)
    _indent_print(f'({rule} ({" ".join(formatted_vars)})', depth, increment, context)

    found_args = False
    for key in QUANTIFIER_KEYS:
        key_str = f'{rule}_{key}'
        if key_str in ast:
            found_args = True
            caller(ast[key_str], depth + 1, increment, context)
    
    if not found_args:
        raise ValueError(f'Found exists or forall with unknown arugments: {ast}')

    _indent_print(')', depth, increment, context)


def _handle_logical(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'({rule}', depth, increment, context)
    caller(ast[f'{rule}_args'], depth + 1, increment, context)
    _indent_print(f')', depth, increment, context)


def _handle_game(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'({rule.replace("_", "-")}', depth, increment, context)
    caller(ast[f'{rule.replace("game_", "")}_pred'], depth + 1, increment, context)
    _indent_print(f')', depth, increment, context)


def _handle_function_eval(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'({ast.func_name} {" ".join(ast.func_args)})', depth, increment, context)


def _inline_format_function_eval(ast):
    return f'({ast.func_name} {" ".join(ast.func_args)})'


def _handle_function_comparison(caller, rule, ast, depth, increment, context=None):
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

    _indent_print(f'({comp_op} {" ".join(args)})', depth, increment, context)


def _handle_predicate(caller, rule, ast, depth, increment, context, return_str=False):
    name = ast.pred_name
    args = []
    for arg in ast.pred_args:
        if isinstance(arg, str):
            args.append(arg)
        else:
            args.append(_handle_predicate(caller, rule, arg, depth + 1, increment, context, return_str=True))

    out = f'({name} {" ".join(args)})'
    if return_str:
        return out

    _indent_print(out, depth, increment, context)


def build_setup_printer():
    printer = ASTPrinter('(:setup', ('setup_', '_predicate'))
    printer.register_exact_matches(
        _handle_function_comparison, _handle_predicate, _handle_function_eval)
    printer.register_keyword_match(('exists', 'forall'), _handle_quantifier)
    printer.register_keyword_match(('game',), _handle_game)
    printer.register_keyword_match(('and', 'or', 'not'), _handle_logical)
    return printer


def _handle_preference(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'(preference {ast.pref_name}', depth, increment, context)
    caller(ast.pref_body, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)


def _handle_then(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'(then', depth, increment, context)
    caller(ast.then_funcs, depth + 1, increment, context)
    _indent_print(f')', depth, increment, context)


def _handle_at_end(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'(at-end', depth, increment, context)
    caller(ast.at_end_pred, depth + 1, increment, context)
    _indent_print(f')', depth, increment, context)


def _handle_any(caller, rule, ast, depth, increment, context=None):
    if context is None:
        context = {}
    if 'mutation' in ast:  context['mutation'] = ast['mutation']
    _indent_print('(any)', depth, increment, context)
    if 'mutation' in context: del context['mutation']


def _handle_once(caller, rule, ast, depth, increment, context=None):
    if context is None:
        context = {}
    if 'mutation' in ast:  context['mutation'] = ast['mutation']
    _indent_print('(once', depth, increment, context)
    context['continue_line'] = True
    caller(ast.once_pred, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False
    if 'mutation' in context: del context['mutation']


def _handle_once_measure(caller, rule, ast, depth, increment, context=None):
    if context is None:
        context = {}
    if 'mutation' in ast:  context['mutation'] = ast['mutation']
    _indent_print('(once-measure', depth, increment, context)
    context['continue_line'] = True
    caller(ast.once_measure_pred, depth + 1, increment, context)
    _indent_print(_inline_format_function_eval(ast.measurement), depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False
    if 'mutation' in context: del context['mutation']


def _handle_hold(caller, rule, ast, depth, increment, context=None):
    if context is None:
        context = {}
    if 'mutation' in ast:  context['mutation'] = ast['mutation']
    _indent_print('(hold', depth, increment, context)
    context['continue_line'] = True
    caller(ast.hold_pred, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False
    if 'mutation' in context: del context['mutation']


def _handle_while_hold(caller, rule, ast, depth, increment, context=None):
    if context is None:
        context = {}
    if 'mutation' in ast:  context['mutation'] = ast['mutation']
    _indent_print('(hold-while', depth, increment, context)
    context['continue_line'] = True
    caller([ast.hold_pred, ast.while_preds], depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False
    if 'mutation' in context: del context['mutation']


def _handle_hold_for(caller, rule, ast, depth, increment, context=None):
    if context is None:
        context = {}
    if 'mutation' in ast:  context['mutation'] = ast['mutation']
    _indent_print(f'(hold-for {ast.num_to_hold}', depth, increment, context)
    context['continue_line'] = True
    caller(ast.hold_pred, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


def _handle_hold_to_end(caller, rule, ast, depth, increment, context=None):
    if context is None:
        context = {}
    if 'mutation' in ast:  context['mutation'] = ast['mutation']
    _indent_print('(hold-to-end', depth, increment, context)
    context['continue_line'] = True
    caller(ast.hold_pred, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False
    if 'mutation' in context: del context['mutation']


def _handle_forall_seq(caller, rule, ast, depth, increment, context=None):
    formatted_vars = _parse_variable_list(ast.forall_seq_vars.variables)
    _indent_print(f'(forall-sequence ({" ".join(formatted_vars)})', depth, increment, context)
    caller(ast.forall_seq_then, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)


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


def _handle_binary_comp(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'({ast.op}', depth, increment, context)
    if context is None:
        context = {}
    context['continue_line'] = True
    caller([ast.expr_1, ast.expr_2], depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


def _handle_multi_expr(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'({ast.op}', depth, increment, context)
    if context is None:
        context = {}
    context['continue_line'] = True
    caller(ast.expr, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


def _handle_binary_expr(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'({ast.op}', depth, increment, context)
    caller([ast.expr_1, ast.expr_2], depth + 1, increment, context)
    _indent_print(')', depth, increment, context)


def _handle_neg_expr(caller, rule, ast, depth, increment, context=None):
    _indent_print('(-', depth, increment, context)
    if context is None:
        context = {}
    context['continue_line'] = True
    caller(ast.expr, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


def _handle_equals_comp(caller, rule, ast, depth, increment, context=None):
    _indent_print('(=', depth, increment, context)
    if context is None:
        context = {}
    context['continue_line'] = True
    caller(ast.expr, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


def _handle_preference_eval(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'({ast.parseinfo.rule.replace("_", "-")} {ast.pref_name})', depth, increment, context)


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


def pretty_print_ast(ast, increment=DEFAULT_INCREMENT, context=None):
    _indent_print(f'{ast[0]} (game {ast[1]["game_name"]}) (:domain {ast[2]["domain_name"]})', 0, increment, context)
    ast = ast[3:]

    while ast:
        key = ast[0][0]

        if key == ')':
            _indent_print(f')', 0, increment, context)
            ast = None

        elif key in PARSE_DICT:
            PARSE_DICT[key](ast[0], 0, increment, context)
            ast = ast[1:]

        else:
            print(f'Encountered unknown key: {key}\n')

    _indent_print('', 0, increment, context)
