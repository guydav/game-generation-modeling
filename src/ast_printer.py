import functools
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import var
import tatsu

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
        depth, increment, context = kwargs['depth'], kwargs['increment'], kwargs['context']
        # check for exact matches before rule substitutions

        if rule in self.exact_matches:
            self.exact_matches[rule](self, rule, ast, depth, increment, context)
            return

        # if that didn't work, apply rule substitutions
        for sub in self.rule_name_substitutions:
            rule = rule.replace(sub, '')

        if rule in self.exact_matches:
            self.exact_matches[rule](self, rule, ast, depth, increment, context)
            return

        else:
            for keywords, handler in self.keyword_matches:
                if any([keyword in rule for keyword in keywords]):
                    handler(self, rule, ast, depth, increment, context)
                    return

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


MUTATION_STYLES = {
    'old': {'text-decoration': 'line-through'},
    'new': {'color':  '#FF6A00'},
    'modified': {'color': '#00FF80'},
    'root': {'color': '#AA00FF'}
}
DEFAULT_COLORMAP = plt.get_cmap('tab10')
MUTATION_STYLES.update({i: {'color': rgb2hex(DEFAULT_COLORMAP(i))} for i in range(10)})

def preprocess_context(context):
    if not context:
        context = {}

    if not 'html_style' in context:
        context['html_style'] = {}

    if 'color' in context['html_style']: del context['html_style']['color']
    if 'text-decoration' in context['html_style']: del context['html_style']['text-decoration']

    if 'mutation' in context:
        mutation = context['mutation']
        if mutation in MUTATION_STYLES:
            context['html_style'].update(MUTATION_STYLES[mutation])
        elif str(mutation) in MUTATION_STYLES:
            context['html_style'].update(MUTATION_STYLES[str(mutation)])

    return context
            

def _indent_print(out, depth, increment, context=None):
    global BUFFER, LINE_BUFFER

    context = preprocess_context(context)

    if 'continue_line' in context and context['continue_line']:
        LINE_BUFFER.append(out)

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

            LINE_BUFFER = [f'<div style="{"; ".join({f"{k}: {v}" for k, v in context["html_style"].items()})}">{out}']
        else:
            LINE_BUFFER = [f'{increment * depth}{out}']
        

def _out_str_to_span(out_str, context):
    if 'html' in context and context['html']:
        return f'<span style="{"; ".join({f"{k}: {v}" for k, v in context["html_style"].items() if k != "margin-left"})}">{out_str}</span>'
    
    return out_str


def mutation_context(func):
    @functools.wraps(func)
    def wrapper_func(caller, rule, ast, depth, increment, context=None, **kwargs):
        if context is None:
            context = {}

        prev_mutation = None
        if 'mutation' in ast:  
            if 'mutation' in context:
                prev_mutation = context['mutation']
        
            context['mutation'] = ast['mutation']

        context = preprocess_context(context)

        ret_val = func(caller, rule, ast, depth, increment, context, **kwargs)

        if 'mutation' in ast:  
            if prev_mutation is not None:
                context['mutation'] = prev_mutation
            else:
                del context['mutation'] 

        return ret_val

    return wrapper_func


@mutation_context
def _parse_variable_list(caller, rule, var_list, depth, increment, context=None):
    formatted_vars = []

    if not isinstance(var_list, list):
        var_list = [var_list]

    for var_def in var_list:
        prev_mutation = None
        if 'mutation' in var_def:
            if 'mutation' in context:
                prev_mutation = context['mutation']

            context['mutation'] = var_def['mutation']

        context = preprocess_context(context)

        var_names = f'{" ".join(var_def.var_names) if isinstance(var_def.var_names, list) else var_def.var_names}'

        var_type = var_def.var_type.type

        if isinstance(var_type, str):
            var_str = f'{var_names} - {var_type}'
        
        elif var_type.parseinfo.rule == 'either_types':
            var_type_str = f'(either {" ".join(var_type.type_names)})'

            inner_prev_mutation = None
            if 'html' in context and context['html'] and 'mutation' in var_type:
                if 'mutation' in context:
                    inner_prev_mutation = context['mutation']

                context['mutation'] = var_type['mutation']
                context = preprocess_context(context)
                var_type_str = _out_str_to_span(var_type_str, context)

                if inner_prev_mutation is not None:
                    context['mutation'] = inner_prev_mutation
                else:
                    del context['mutation']                 
                context = preprocess_context(context)

            var_str = f'{var_names} - {var_type_str}'

        else:
            raise ValueError(f'Unrecognized quantifier variables: {var_def[2]}')

        formatted_vars.append(_out_str_to_span(var_str, context))

        if 'mutation' in var_def:  
            if prev_mutation is not None:
                context['mutation'] = prev_mutation
            else:
                del context['mutation'] 

    return formatted_vars


QUANTIFIER_KEYS = ('args', 'pred', 'then', 'pref')


@mutation_context
def _handle_quantifier(caller, rule, ast, depth, increment, context=None):
    prev_continue_line = context['continue_line'] if 'continue_line' in context else False

    _indent_print(_out_str_to_span(f'({rule}', context), depth, increment, context)
    context['continue_line'] = True

    vars_key = None
    args_key = None
    vars_node = None
    var_str = ''
    vars_keys_list = list(filter(lambda k: k.endswith('_vars'), ast.keys()))
    if len(vars_keys_list) > 1:
         raise ValueError(f'Multiple quantifier variables: {ast}')

    if len(vars_keys_list) == 1:
        vars_key = vars_keys_list[0]
        args_key = vars_key.replace('_vars', '_args')
        vars_node = ast[vars_key]
        formatted_vars = _parse_variable_list(caller, rule, vars_node.variables, depth, increment, context)
        var_str = f'({" ".join(formatted_vars)})'
    
    if vars_node is not None and 'mutation' in vars_node:
        prev_mutation = None
        if 'mutation' in context:
            prev_mutation = context['mutation']
        context['mutation'] = vars_node['mutation']
        context = preprocess_context(context)

        _indent_print(_out_str_to_span(var_str, context), depth, increment, context)

        if prev_mutation is not None:
            context['mutation'] = prev_mutation
        else:
            del context['mutation'] 

        context = preprocess_context(context)

    else:
        _indent_print(_out_str_to_span(var_str, context), depth, increment, context)

    context['continue_line'] = prev_continue_line

    found_args = False

    if args_key is not None and args_key in ast:
        found_args = True
        caller(ast[args_key], depth + 1, increment, context)

    if not found_args:
        for key in QUANTIFIER_KEYS:
            key_str = f'{rule}_{key}'
            if key_str in ast:
                found_args = True
                caller(ast[key_str], depth + 1, increment, context)
    
    if not found_args:
        print(ast.keys())
        print(rule)
        print([f'{rule}_{key}' for key in QUANTIFIER_KEYS])
        raise ValueError(f'Found exists or forall with unknown arugments: {ast}')

    _indent_print(_out_str_to_span(')', context), depth, increment, context)


@mutation_context
def _handle_logical(caller, rule, ast, depth, increment, context=None):
    if 'continue_line' in context and context['continue_line'] and 'html' in context and context['html']:
        _indent_print(f'<span style="{"; ".join({f"{k}: {v}" for k, v in context["html_style"].items() if k != "margin-left"})}">', depth, increment, context)
        
    if f'{rule}_args' in ast:
        _indent_print(f'({rule}', depth, increment, context)
        caller(ast[f'{rule}_args'], depth + 1, increment, context)
    else:
        rule_fragment = rule.split('_')[0]
        key = f'{rule_fragment}_args'
        if key in ast:
            _indent_print(f'({rule_fragment}', depth, increment, context)
            caller(ast[key], depth + 1, increment, context)
        else:
            raise ValueError(f'Found logical with unknown arguments: {ast}')

    _indent_print(f')', depth, increment, context)

    if 'continue_line' in context and context['continue_line'] and 'html' in context and context['html']:
        _indent_print(f'</span>', depth, increment, context)


@mutation_context
def _handle_game(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'({rule.replace("_", "-")}', depth, increment, context)
    caller(ast[f'{rule.replace("game_", "")}_pred'], depth + 1, increment, context)
    _indent_print(f')', depth, increment, context)


@mutation_context
def _handle_function_eval(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'({ast.func_name} {" ".join(_format_func_args(ast, depth, increment, context))})', 
        depth, increment, context)


@mutation_context
def _inline_format_function_eval(caller, rule, ast, depth, increment, context=None):
    formatted_args = _format_func_args(ast, depth, increment, context)
    return _out_str_to_span(f'({ast.func_name} {" ".join(formatted_args)})', context)

def _format_func_args(ast, depth, increment, context):
    func_args = ast.func_args
    if func_args is None:
        formatted_args = []

    else:
        if not isinstance(func_args, list):
            func_args = [func_args]

        formatted_args = [
            arg if isinstance(arg, str) 
            else _inline_format_comparison_arg(None, arg.term.parseinfo.rule, arg.term, depth, increment, context) 
            for arg in func_args]
            
    return formatted_args


@mutation_context
def _inline_format_comparison_arg(caller, rule, ast, depth, increment, context=None):
    arg = ast.arg

    if isinstance(arg, tatsu.ast.AST): 
        if  arg.parseinfo.rule == 'function_eval':
            return _inline_format_function_eval(caller, arg.rule, arg, depth, increment, context)
        else:
            raise ValueError(f'Unexpected comparison argument: {arg}')
    return arg


@mutation_context
def _handle_function_comparison(caller, rule, ast, depth, increment, context=None):
    ast = ast.comp

    comp_op = '='
    if 'comp_op' in ast:
        comp_op = ast.comp_op

    if 'arg_1' in ast:
        args = [_inline_format_comparison_arg(caller, ast.arg_1.rule, ast.arg_1, depth, increment, context), 
            _inline_format_comparison_arg(caller, ast.arg_2.rule, ast.arg_2, depth, increment, context)]    

    else:
        args = [_inline_format_comparison_arg(caller, arg.rule, arg, depth, increment, context) for arg in ast.equal_comp_funcs]

    _indent_print(f'({comp_op} {" ".join(args)})', depth, increment, context)
    

@mutation_context
def _handle_predicate(caller, rule, ast, depth, increment, context, return_str=False):
    name = ast.pred_name
    pred_args = ast.pred_args
    args = []

    if pred_args is not None:
        if not isinstance(pred_args, list):
            pred_args = [pred_args]

        for arg in pred_args:
            if isinstance(arg, str):
                args.append(arg)
            elif isinstance(arg, tatsu.ast.AST):
                term = arg.term
                if isinstance(term, str):
                    args.append(term)    
                else:
                    args.append(_handle_predicate(caller, rule, term, depth + 1, increment, context, return_str=True))
            else:
                raise ValueError(f'Unexpected predicate argument: {arg}')

    out = _out_str_to_span(f'({name} {" ".join(args)})', context)

    if return_str:
        return out

    _indent_print(out, depth, increment, context)


@mutation_context
def _handle_setup(caller, rule, ast, depth, increment, context=None):
    caller(ast.setup, depth, increment, context)


@mutation_context
def _handle_statement(caller, rule, ast, depth, increment, context=None):
    caller(ast.statement, depth, increment, context)


@mutation_context
def _handle_super_predicate(caller, rule, ast, depth, increment, context=None):
    caller(ast.pred, depth, increment, context)


def build_setup_printer():
    printer = ASTPrinter('(:setup', ('setup_', 'super_', 'predicate_'))
    printer.register_exact_matches(
        _handle_setup, _handle_statement, _handle_super_predicate,
        _handle_function_comparison, _handle_predicate, _handle_function_eval)
    printer.register_keyword_match(('exists', 'forall'), _handle_quantifier)
    printer.register_keyword_match(('game',), _handle_game)
    printer.register_keyword_match(('and', 'or', 'not'), _handle_logical)
    return printer


@mutation_context
def _handle_preference(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'(preference {ast.pref_name}', depth, increment, context)
    caller(ast.pref_body, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    

@mutation_context
def _handle_then(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'(then', depth, increment, context)
    caller(ast.then_funcs, depth + 1, increment, context)
    _indent_print(f')', depth, increment, context)
    

@mutation_context
def _handle_at_end(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'(at-end', depth, increment, context)
    caller(ast.at_end_pred, depth + 1, increment, context)
    _indent_print(f')', depth, increment, context)


@mutation_context
def _handle_always(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'(always', depth, increment, context)
    caller(ast.always_pred, depth + 1, increment, context)
    _indent_print(f')', depth, increment, context)
    

@mutation_context
def _handle_any(caller, rule, ast, depth, increment, context=None):
    _indent_print('(any)', depth, increment, context)
    

@mutation_context
def _handle_once(caller, rule, ast, depth, increment, context=None):
    _indent_print('(once', depth, increment, context)
    context['continue_line'] = True
    caller(ast.once_pred, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


@mutation_context
def _handle_once_measure(caller, rule, ast, depth, increment, context=None):
    _indent_print('(once-measure', depth, increment, context)
    context['continue_line'] = True
    caller(ast.once_measure_pred, depth + 1, increment, context)
    _indent_print(_inline_format_function_eval(caller, ast.measurement.rule, ast.measurement, depth, increment, context), depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False
    

@mutation_context
def _handle_hold(caller, rule, ast, depth, increment, context=None):
    _indent_print('(hold', depth, increment, context)
    context['continue_line'] = True
    caller(ast.hold_pred, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False
    

@mutation_context
def _handle_while_hold(caller, rule, ast, depth, increment, context=None):
    _indent_print('(hold-while', depth, increment, context)
    context['continue_line'] = True
    caller([ast.hold_pred, ast.while_preds], depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False
    

@mutation_context
def _handle_hold_for(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'(hold-for {ast.num_to_hold}', depth, increment, context)
    context['continue_line'] = True
    caller(ast.hold_pred, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False
    

@mutation_context
def _handle_hold_to_end(caller, rule, ast, depth, increment, context=None):
    _indent_print('(hold-to-end', depth, increment, context)
    context['continue_line'] = True
    caller(ast.hold_pred, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False
    

@mutation_context
def _handle_forall_seq(caller, rule, ast, depth, increment, context=None):
    variables = ast.forall_seq_vars.variables
    formatted_vars = _parse_variable_list(caller, rule, variables, depth, increment, context)
    var_str = " ".join(formatted_vars)
    if 'html' in context and context['html'] and 'mutation' in variables:
        prev_mutation = None
        if 'mutation' in context:
            prev_mutation = context['mutation']
        context['mutation'] = ast['mutation']

        _indent_print(f'(forall-sequence ({_out_str_to_span(var_str, context)})', depth, increment, context)

        if prev_mutation is not None:
            context['mutation'] = prev_mutation
        else:
            del context['mutation'] 

    else:
        _indent_print(f'(forall-sequence ({" ".join(formatted_vars)})', depth, increment, context)
    
    caller(ast.forall_seq_then, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)


@mutation_context
def _handle_preferences(caller, rule, ast, depth, increment, context=None):
    _indent_print('(and', depth, increment, context)
    caller(ast.preferences, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)


@mutation_context
def _handle_pref_def(caller, rule, ast, depth, increment, context=None):
    caller(ast.definition, depth, increment, context)
    

@mutation_context
def _handle_pref_body(caller, rule, ast, depth, increment, context=None):
    caller(ast.body, depth, increment, context)


@mutation_context
def _handle_seq_func(caller, rule, ast, depth, increment, context=None):
    caller(ast.seq_func, depth, increment, context)


def build_constraints_printer():
    printer = ASTPrinter('(:constraints', ('pref_body_', 'pref_', 'super_', 'predicate_'))
    printer.register_exact_matches(
        _handle_preferences, _handle_pref_def, _handle_pref_body, _handle_seq_func,
        _handle_preference, _handle_super_predicate,
        _handle_function_comparison, _handle_predicate,
        _handle_at_end, _handle_always, _handle_then, 
        _handle_any, _handle_once, _handle_once_measure,
        _handle_hold, _handle_while_hold, _handle_hold_for, _handle_hold_to_end, 
        _handle_forall_seq, _handle_function_eval
    )
    printer.register_exact_match(_handle_preferences, 'pref_forall_prefs')
    printer.register_keyword_match(('exists', 'forall'), _handle_quantifier)
    printer.register_keyword_match(('and', 'or', 'not'), _handle_logical)
    return printer


@mutation_context
def _handle_binary_comp(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'({ast.op}', depth, increment, context)
    context['continue_line'] = True
    caller([ast.expr_1, ast.expr_2], depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


@mutation_context
def _handle_multi_expr(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'({ast.op}', depth, increment, context)
    context['continue_line'] = True
    caller(ast.expr, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


@mutation_context
def _handle_binary_expr(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'({ast.op}', depth, increment, context)
    caller([ast.expr_1, ast.expr_2], depth + 1, increment, context)
    _indent_print(')', depth, increment, context)


@mutation_context
def _handle_neg_expr(caller, rule, ast, depth, increment, context=None):
    _indent_print('(-', depth, increment, context)
    if context is None:
        context = {}
    context['continue_line'] = True
    caller(ast.expr, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


@mutation_context
def _handle_equals_comp(caller, rule, ast, depth, increment, context=None):
    _indent_print('(=', depth, increment, context)
    if context is None:
        context = {}
    context['continue_line'] = True
    caller(ast.expr, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


@mutation_context
def _handle_with(caller, rule, ast, depth, increment, context=None):
    _indent_print('(with ', depth, increment, context)
    if context is None:
        context = {}
    context['continue_line'] = True
    var_node = ast['with_vars']
    formatted_vars = _parse_variable_list(caller, rule, var_node.variables, depth, increment, context)
    var_str = f'({" ".join(formatted_vars)})'

    if 'mutation' in var_node:
        prev_mutation = None
        if 'mutation' in context:
            prev_mutation = context['mutation']
        context['mutation'] = var_node['mutation']
        context = preprocess_context(context)

        _indent_print(_out_str_to_span(var_str, context), depth, increment, context)

        if prev_mutation is not None:
            context['mutation'] = prev_mutation
        else:
            del context['mutation'] 

        context = preprocess_context(context)

    else:
        _indent_print(_out_str_to_span(var_str, context), depth, increment, context)

    caller(ast.with_pref, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


@mutation_context
def _handle_count_method(caller, rule, ast, depth, increment, context=None):
    type_str = ''
    if ast.name_and_types.object_types:
        if isinstance(ast.name_and_types.object_types, tatsu.ast.AST):
            type_str =f':{ast.name_and_types.object_types.type_name}'
        else:
            type_str = ":" + ":".join([t.type_name for t in ast.name_and_types.object_types])
    _indent_print(_out_str_to_span(f'({ast.parseinfo.rule.replace("_", "-")} {ast.name_and_types.pref_name}{type_str})', context), depth, increment, context)
    

@mutation_context
def _handle_terminal(caller, rule, ast, depth, increment, context=None):
    caller(ast.terminal, depth, increment, context)


@mutation_context
def _handle_terminal_expr(caller, rule, ast, depth, increment, context=None):
    caller(ast.expr, depth, increment, context)


@mutation_context
def _handle_scoring_expr(caller, rule, ast, depth, increment, context=None):
    caller(ast.expr, depth, increment, context)


@mutation_context
def _handle_preference_eval(caller, rule, ast, depth, increment, context=None):
    caller(ast.count_method, depth, increment, context)

def build_terminal_printer():
    printer = ASTPrinter('(:terminal', ('terminal_', 'scoring_'))
    printer.register_exact_matches(
        _handle_terminal, _handle_terminal_expr,
        _handle_scoring_expr, _handle_preference_eval,
        _handle_multi_expr, _handle_binary_expr, 
        _handle_neg_expr, _handle_equals_comp, 
        _handle_function_eval, _handle_with
    )
    printer.register_exact_match(_handle_binary_comp, 'comp')
    printer.register_keyword_match(('count',), _handle_count_method)
    printer.register_keyword_match(('and', 'or', 'not'), _handle_logical)
    return printer


@mutation_context
def _handle_maximize(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'maximize', depth, increment, context)
    caller(ast.expr, depth + 1, increment, context)


@mutation_context
def _handle_minimize(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'minimize', depth, increment, context)
    caller(ast.expr, depth + 1, increment, context)


@mutation_context
def _handle_scoring(caller, rule, ast, depth, increment, context=None):
    caller(ast.scoring, depth, increment, context)


@mutation_context
def _handle_scoring_comparison(caller, rule, ast, depth, increment, context=None):
    caller(ast.comp, depth, increment, context)


def build_scoring_printer():
    printer = ASTPrinter('(:scoring', ('scoring_',))
    printer.register_exact_matches(
        _handle_scoring, _handle_maximize, _handle_minimize, 
        _handle_scoring_expr, _handle_preference_eval, _handle_scoring_comparison,
        _handle_multi_expr, _handle_binary_expr, _handle_neg_expr, 
        _handle_equals_comp, _handle_function_eval, _handle_with,
    )
    printer.register_exact_match(_handle_binary_comp, 'comp')
    printer.register_keyword_match(('count',), _handle_count_method)
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
