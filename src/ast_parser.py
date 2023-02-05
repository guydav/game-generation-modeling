import itertools
import re
import typing

import boolean
import numpy as np
import tatsu
import tatsu.ast
import tatsu.buffering


SETUP = '(:setup'
PREFERENCES = '(:constraints'
TERMINAL = '(:terminal'
SCORING = '(:scoring'
SECTION_KEYS = (SETUP, PREFERENCES, TERMINAL, SCORING)

SECTION_CONTEXT_KEY = 'section'
VARIABLES_CONTEXT_KEY = 'variables'


import ast_printer
import ast_utils


class ASTParser:
    def __call__(self, ast, **kwargs):
        # TODO: rewrite in Python 3.10-style switch-case or using a dict to map?
        if ast is None:
            return

        if isinstance(ast, str):
            return self._handle_str(ast, **kwargs)

        if isinstance(ast, (int, np.int32, np.int64)):  # type: ignore
            return self._handle_int(ast, **kwargs)

        if isinstance(ast, tuple):
            return self._handle_tuple_with_section(ast, **kwargs)

        if isinstance(ast, list):
            return self._handle_list(ast, **kwargs)

        if isinstance(ast, tatsu.ast.AST):
            return self._handle_ast(ast, **kwargs)

        if isinstance(ast, tatsu.buffering.Buffer):
            return

        raise ValueError(f'Unrecognized AST type: {type(ast)}', ast)

    def _handle_str(self, ast: str, **kwargs):
        pass

    def _handle_int(self, ast: typing.Union[int, np.int32, np.int64], **kwargs):
        pass

    def _handle_tuple_with_section(self, ast: tuple, **kwargs):
        if ast[0] in SECTION_KEYS:
            kwargs[SECTION_CONTEXT_KEY] = ast[0]

        return self._handle_tuple(ast, **kwargs)

    def _handle_tuple(self, ast: tuple, **kwargs):
        return self._handle_iterable(ast, **kwargs)

    def _handle_list(self, ast: list, **kwargs):
        return self._handle_iterable(ast, **kwargs)

    def _handle_ast(self, ast: tatsu.ast.AST, **kwargs):
        for key in ast:
            if key != 'parseinfo':
                self(ast[key], **kwargs)

    def _handle_iterable(self, ast: typing.Iterable, **kwargs):
        return [self(item, **kwargs) for item in ast]

    def _default_kwarg(self, kwargs: typing.Dict[str, typing.Any], key: str, default_value: typing.Any, should_call: bool=False):
        if key not in kwargs or kwargs[key] is None:
            if not should_call:
                kwargs[key] = default_value
            else:
                kwargs[key] = default_value()

        return kwargs[key]


class ASTParentMapper(ASTParser):
    def __init__(self, root_node='root'):
        self.root_node = root_node
        self.parent_mapping = {}

    def __call__(self, ast, **kwargs):
        self._default_kwarg(kwargs, 'parent', ast)
        self._default_kwarg(kwargs, 'selector', [])
        return super().__call__(ast, **kwargs)

    def _handle_iterable(self, ast, **kwargs):
        [self(element, parent=kwargs['parent'], selector=kwargs['selector'] + [i])
        for i, element in enumerate(ast)]

    def _build_mapping_value(self, ast, **kwargs):
        return (ast, kwargs['parent'], kwargs['selector'])

    def _handle_ast(self, ast, **kwargs):
        self._add_ast_to_mapping(ast, **kwargs)

        for key in ast:
            if key != 'parseinfo':
                self(ast[key], parent=ast, selector=[key])

    def _ast_key(self, ast):
        return ast.parseinfo._replace(alerts=None)

    def _add_ast_to_mapping(self, ast, **kwargs):
        self.parent_mapping[self._ast_key(ast)] = self._build_mapping_value(ast, **kwargs)  # type: ignore


DEFAULT_PARSEINFO_COMPARISON_INDICES = (1, 2, 3)


class ASTParseinfoSearcher(ASTParser):
    def __init__(self, comparison_indices: typing.Sequence[int] = DEFAULT_PARSEINFO_COMPARISON_INDICES):
        super().__init__()
        self.comparison_indices = comparison_indices

    def __call__(self, ast, **kwargs):
        if 'parseinfo' not in kwargs:
            raise ValueError('parseinfo must be passed as a keyword argument')

        return super().__call__(ast, **kwargs)

    def _handle_iterable(self, ast: typing.Iterable, **kwargs):
        for item in ast:
            result = self(item, **kwargs)
            if result is not None:
                return result

        return None

    def _handle_ast(self, ast: tatsu.ast.AST, **kwargs):
        if all(ast.parseinfo[i] == kwargs['parseinfo'][i] for i in self.comparison_indices):  # type: ignore
            return ast

        for key in ast:
            if key != 'parseinfo':
                result = self(ast[key], **kwargs)
                if result is not None:
                    return result

        return None


class ASTDepthParser(ASTParser):
    def __call__(self, ast, **kwargs):
        self._default_kwarg(kwargs, 'depth', 0)
        return super().__call__(ast, **kwargs)

    def _handle_iterable(self, ast: typing.Iterable, **kwargs):
        child_values = [self(item, depth=kwargs['depth'] + 1) for item in ast]
        child_values = [cv for cv in child_values if cv is not None]
        if not child_values:
            return kwargs['depth']
        return max(child_values)

    def _handle_ast(self, ast: tatsu.ast.AST, **kwargs):
        child_values = [self(ast[key], depth=kwargs['depth'] + 1) for key in ast if key != 'parseinfo']
        child_values = [cv for cv in child_values if cv is not None]
        if not child_values:
            return kwargs['depth']
        return max(child_values)

    def _handle_str(self, ast: str, **kwargs):
        return kwargs['depth']

    def _handle_int(self, ast: typing.Union[int, np.int32, np.int64], **kwargs):
        return kwargs['depth']



DEFAULT_MAX_TAUTOLOGY_EVAL_LENGTH = 16


class ASTBooleanParser(ASTParser):
    # node_to_symbol_mapping: typing.Dict[str, boolean.Symbol]
    algebra: boolean.BooleanAlgebra
    str_to_expression_mapping: typing.Dict[str, boolean.Expression]
    # valid_symbol_names: typing.List[str]
    whitespace_pattern: re.Pattern
    def __init__(self, max_tautology_eval_length: int = DEFAULT_MAX_TAUTOLOGY_EVAL_LENGTH):
        self.max_tautology_eval_length = max_tautology_eval_length
        self.algebra = boolean.BooleanAlgebra()
        self.true = self.algebra.parse('TRUE')
        self.false = self.algebra.parse('FALSE')
        self.whitespace_pattern = re.compile(r'\s+')
        self.variable_pattern = re.compile(r'\?[\w\d]+')
        # self.valid_symbol_names = list(''.join((l, d)) for d, l in itertools.product([''] + [str(x) for x in range(5)], string.ascii_lowercase,))
        self.game_start()

    def game_start(self):
        self.str_to_expression_mapping = {}
        self.next_symbol_name_index = 0

    def _all_equal(self, iterable: typing.Iterable):
        # Returns True if all the elements are equal to each other -- from Python itertools recipes
        g = itertools.groupby(iterable)
        return next(g, True) and not next(g, False)

    def evaluate_tautology(self, expr: boolean.Expression) -> bool:
        symbols = list(expr.symbols)

        if len(symbols) > self.max_tautology_eval_length:
            # print(f'Not evaluating tautology for expression with {len(symbols)} symbols')
            return False

        # initial_value = None
        # found_both_values = False

        # # for value_assignments in itertools.product([self.true, self.false], repeat=len(symbols)):
        # #     value = expr.subs({s: v for s, v in zip(symbols, value_assignments)}, simplify=True)
        # #     if initial_value is None:
        # #         initial_value = value
        # #     elif initial_value != value:
        # #         found_both_values = True
        # #         break

        # # return not found_both_values

        # possible_values = [
        #     expr.subs({s: v for s, v in zip(symbols, value_assignments)}, simplify=True)
        #     for value_assignments in itertools.product([self.true, self.false], repeat=len(symbols))
        # ]
        # return self._all_equal(possible_values)
        return self._all_equal(expr.subs({s: v for s, v in zip(symbols, value_assignments)}, simplify=True)
            for value_assignments in itertools.product([self.true, self.false], repeat=len(symbols)))

    def evaluate_redundancy(self, expr: boolean.Expression) -> bool:
        simplified = expr.simplify()
        redundancy_detected = (len(str(simplified)) < len(str(expr))) and (len(simplified.args) <= len(expr.args))
        return redundancy_detected

    def __call__(self, ast: tatsu.ast.AST, **kwargs) -> typing.Union[boolean.Expression, typing.List[boolean.Expression]]:
        if SECTION_CONTEXT_KEY not in kwargs:
            raise ValueError(f'Context key {SECTION_CONTEXT_KEY} not found in kwargs')
        return super().__call__(ast, **kwargs)  # type: ignore

    def _handle_ast(self, ast: tatsu.ast.AST, **kwargs) -> boolean.Expression:
        rule = typing.cast(str, ast.parseinfo.rule)  # type: ignore
        key = self.whitespace_pattern.sub(' ', ast_printer.ast_section_to_string(ast, f'(:{kwargs[SECTION_CONTEXT_KEY]}' if not kwargs[SECTION_CONTEXT_KEY].startswith('(:') else kwargs[SECTION_CONTEXT_KEY]))

        if VARIABLES_CONTEXT_KEY in kwargs:
            variables_in_key = set(self.variable_pattern.findall(key))

            for var in variables_in_key:
                if var in kwargs[VARIABLES_CONTEXT_KEY]:
                    var_types = kwargs[VARIABLES_CONTEXT_KEY][var].var_types
                    var_types_str = '_. '.join(var_types)
                    key = key.replace(var, f'{var}__{var_types_str}')

        if key in self.str_to_expression_mapping:
            return self.str_to_expression_mapping[key]

        expr = None

        if rule.endswith('_and'):
            arg_mappings = self(ast.and_args, **kwargs)  # type: ignore
            if isinstance(arg_mappings, list) and len(arg_mappings) == 1:
                arg_mappings = arg_mappings[0]

            if isinstance(arg_mappings, boolean.Expression):
                expr = arg_mappings

            else:
                expr = boolean.AND(*arg_mappings)

        elif rule.endswith('_or'):
            arg_mappings = typing.cast(list, self(ast.or_args, **kwargs))  # type: ignore
            if isinstance(arg_mappings, list) and len(arg_mappings) == 1:
                arg_mappings = arg_mappings[0]

            if isinstance(arg_mappings, boolean.Expression):
                expr = arg_mappings

            else:
                expr = boolean.OR(*arg_mappings)

        elif rule.endswith('_not'):
            arg_mapping = self(ast.not_args, **kwargs)  # type: ignore
            expr = boolean.NOT(arg_mapping)

        # elif rule == 'setup':
        #     expr = self(ast.setup, **kwargs)  # type: ignore

        # elif rule == 'setup_statement':
        #     expr = self(ast.statement, **kwargs)  # type: ignore

        elif rule.endswith('_exists'):
            var_dict = kwargs[VARIABLES_CONTEXT_KEY] if VARIABLES_CONTEXT_KEY in kwargs else {}
            ast_utils.extract_variables_from_ast(ast, 'exists_vars', var_dict)
            kwargs = kwargs.copy()
            kwargs[VARIABLES_CONTEXT_KEY] = var_dict

            expr = self(ast.exists_args, **kwargs)  # type: ignore
            expr = boolean.Symbol(f'exists_{str(expr).lower()}')

        elif rule.endswith('_forall'):
            var_dict = kwargs[VARIABLES_CONTEXT_KEY] if VARIABLES_CONTEXT_KEY in kwargs else {}
            ast_utils.extract_variables_from_ast(ast, 'forall_vars', var_dict)
            kwargs = kwargs.copy()
            kwargs[VARIABLES_CONTEXT_KEY] = var_dict

            expr = self(ast.forall_args, **kwargs)  # type: ignore
            expr = boolean.Symbol(f'forall_{str(expr).lower()}')

        elif rule == 'super_predicate':
            expr = self(ast.pred, **kwargs)  # type: ignore

        elif rule in ('function_comparison', 'predicate'):
            # symbol_name = self.valid_symbol_names[self.next_symbol_name_index]
            # self.next_symbol_name_index += 1
            symbol_name = key.replace(') )', '))').replace('?', '').replace(' ', '_')
            expr = boolean.Symbol(symbol_name)

        else:
            keys = set(ast.keys())
            if 'parseinfo' in keys:
                keys.remove('parseinfo')

            if len(keys) == 1:
                expr = self(ast[keys.pop()], **kwargs)  # type: ignore

        if expr is None:
            raise ValueError(f'No expression found for rule {rule}')

        expr = typing.cast(boolean.Expression, expr)
        self.str_to_expression_mapping[key] = expr
        return expr
