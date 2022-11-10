import tatsu
import tatsu.ast
import tatsu.buffering
import typing

class ASTParser:
    def __call__(self, ast, **kwargs):
        # TODO: rewrite in Python 3.10-style switch-case or using a dict to map?
        if ast is None:
            return

        if isinstance(ast, str):
            return self._handle_str(ast, **kwargs)

        if isinstance(ast, int):
            return self._handle_int(ast, **kwargs)

        if isinstance(ast, tuple):
            return self._handle_tuple(ast, **kwargs)

        if isinstance(ast, list):
            return self._handle_list(ast, **kwargs)

        if isinstance(ast, tatsu.ast.AST):
            return self._handle_ast(ast, **kwargs)

        if isinstance(ast, tatsu.buffering.Buffer):
            return

        raise ValueError(f'Unrecognized AST type: {type(ast)}', ast)

    def _handle_str(self, ast: str, **kwargs):
        pass

    def _handle_int(self, ast: int, **kwargs):
        pass

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

    def _add_ast_to_mapping(self, ast, **kwargs):
        self.parent_mapping[ast.parseinfo._replace(alerts=None)] = self._build_mapping_value(ast, **kwargs)  # type: ignore


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
