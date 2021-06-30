import tatsu


class ASTParser:
    def __call__(self, ast, **kwargs):
        # TODO: rewrite in Python 3.10-style switch-case or using a dict to map?
        if ast is None:
            return

        if isinstance(ast, str):
            return self._handle_str(ast, **kwargs)

        if isinstance(ast, int):
            return self._handle_int(ast, **kwargs)

        # if isinstance(ast, tatsu.buffering.Buffer):
            # return self._handl

        if isinstance(ast, tuple):
            return self._handle_tuple(ast, **kwargs)

        if isinstance(ast, list):
            return self._handle_list(ast, **kwargs)

        if isinstance(ast, tatsu.ast.AST):
            return self._handle_ast(ast, **kwargs)

        raise ValueError(f'Unrecognized AST type: {type(ast)}', ast)

    def _handle_str(self, ast, **kwargs):
        pass

    def _handle_int(self, ast, **kwargs):
        pass

    def _handle_tuple(self, ast, **kwargs):
        return self._handle_iterable(ast, **kwargs)

    def _handle_list(self, ast, **kwargs):
        return self._handle_iterable(ast, **kwargs)

    def _handle_ast(self, ast, **kwargs):
        for key in ast:
            if key != 'parseinfo':
                self(ast[key], **kwargs)

    def _handle_iterable(self, ast, **kwargs):
        return [self(item, **kwargs) for item in ast]

    def _default_kwarg(self, kwargs, key, default_value, should_call=False):
        if key not in kwargs or kwargs[key] is None:
            if not should_call:
                kwargs[key] = default_value
            else:
                kwargs[key] = default_value()

        return kwargs[key]


class ASTParentMapper(ASTParser):
    def __init__(self, root_node='root'):
        self.root_node = root_node

    def __call__(self, ast, **kwargs):
        self._default_kwarg(kwargs, 'parent', ast)
        self._default_kwarg(kwargs, 'mapping', {})
        self._default_kwarg(kwargs, 'selector', [])
        super().__call__(ast, **kwargs)
        return kwargs['mapping']

    def _handle_iterable(self, ast, **kwargs):
        return [self(element, mapping=kwargs['mapping'], parent=kwargs['parent'], selector=kwargs['selector'] + [i]) 
                for i, element in enumerate(ast)]

    def _handle_ast(self, ast, **kwargs):
        kwargs['mapping'][ast.parseinfo] = (ast, kwargs['parent'], kwargs['selector'])

        for key in ast:
            if key != 'parseinfo':
                self(ast[key], mapping=kwargs['mapping'], parent=ast, selector=[key])

        return kwargs['mapping']


