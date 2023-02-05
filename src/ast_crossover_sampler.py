from collections import defaultdict
import copy
from enum import Enum
import typing

import numpy as np
import tatsu
import tatsu.ast
import tatsu.infos
from tatsu.infos import ParseInfo
import tqdm

import ast_printer
from ast_parser import ASTParser
from ast_utils import replace_child
from ast_counter_sampler import RegrowthSampler, ASTSampler, ASTParentMapping, ASTNodeInfo, ContextDict, SamplingException, simplified_context_deepcopy


class ASTContextFixer(ASTParser):
    local_context_propagating_rules = typing.Sequence[str]

    def __init__(self, new_variable_sampler: typing.Callable[[ContextDict, ContextDict], str], local_context_propagating_rules: typing.Sequence[str], rng: np.random.Generator):
        super().__init__()
        self.new_variable_sampler = new_variable_sampler
        self.local_context_propagating_rules = local_context_propagating_rules
        self.rng = rng

    def __call__(self, ast: tatsu.ast.AST, **kwargs):
        self._default_kwarg(kwargs, 'global_context', dict())
        self._default_kwarg(kwargs, 'local_context', dict())
        kwargs['local_context'] = simplified_context_deepcopy(kwargs['local_context'])
        retval = super().__call__(ast, **kwargs)
        self._update_contexts(kwargs, retval)
        return retval

    def _update_contexts(self, kwargs: typing.Dict[str, typing.Any], retval: typing.Any):
        if retval is not None and isinstance(retval, tuple) and len(retval) == 2:
            global_context_update, local_context_update = retval
            if global_context_update is not None:
                kwargs['global_context'].update(global_context_update)
            if local_context_update is not None:
                kwargs['local_context'].update(local_context_update)

    def _single_variable_def_context_update(self, ast: tatsu.ast.AST, local_context: ContextDict, global_context: ContextDict):
        var_names = ast.var_names
        single_variable = False
        if isinstance(var_names, str):
            var_names = [var_names]
            single_variable = True

        for i, var_name in enumerate(var_names):  # type: ignore
            if var_name in local_context['variables']:
                if local_context['variables'][var_name] == ast.parseinfo.pos:  # type: ignore
                    continue
                else:
                    new_var = self.new_variable_sampler(global_context, local_context)
                    new_var_name = new_var[1:]

                    global_context['replacement_mappings'][var_name] = new_var_name
                    local_context['variables'][new_var_name] = ast.parseinfo
                    replace_child(ast, ['var_names'] if single_variable else ['var_names', i], new_var)

            else:
                local_context['variables'][var_name[1:]] = ast.parseinfo.pos  # type: ignore

    def _handle_ast(self, ast: tatsu.ast.AST, **kwargs):
        local_context = kwargs['local_context']
        global_context = kwargs['global_context']

        rule = ast.parseinfo.rule  # type: ignore

        # if we find a defined variable, add it to the local context
        # if there's already a defined variable with this name in the local context, rename this one, and save the mappings

        if rule == 'variable_list':
            if 'variables' not in local_context:
                local_context['variables'] = dict()
            if 'replacement_mappings' not in global_context:
                global_context['replacement_mappings'] = dict()

            if isinstance(ast.variables, tatsu.ast.AST):
                self._single_variable_def_context_update(ast.variables, local_context, global_context)

            else:
                for var_def in ast.variables:  # type: ignore
                    self._single_variable_def_context_update(var_def, local_context, global_context)

        elif rule == 'variable_type_def':
            if 'variables' not in local_context:
                local_context['variables'] = dict()
            if 'replacement_mappings' not in global_context:
                global_context['replacement_mappings'] = dict()

            self._single_variable_def_context_update(ast, local_context, global_context)

        elif rule == 'predicate_or_function_term':
            if 'variables' not in local_context:
                local_context['variables'] = dict()
            if 'replacement_mappings' not in global_context:
                global_context['replacement_mappings'] = dict()

            term = ast.term
            if isinstance(term, str) and term.startswith('?'):
                # If we already have a mapping for it, replace it with the mapping
                if term in global_context['replacement_mappings']:
                    replace_child(ast, ['term'], '?' + global_context['replacement_mappings'][term])

                # If we don't have a mapping for it, and it's not in the local context, add a mapping
                elif term[1:] not in local_context['variables']:
                    if len(local_context['variables']) == 0:
                        raise SamplingException('No variables found in local context when updating a predicate/function_eval node with variable children')

                    new_var_name = self.rng.choice(list(local_context['variables'].keys()))
                    global_context['replacement_mappings'][term] = new_var_name
                    replace_child(ast, ['term'], '?' + new_var_name)

        # elif rule in ('predicate', 'function_eval'):
        #     if 'variables' not in local_context:
        #         local_context['variables'] = dict()
        #     if 'replacement_mappings' not in global_context:
        #         global_context['replacement_mappings'] = dict()

        #     inner_ast = typing.cast(tatsu.ast.AST, ast.pred if rule == 'predicate' else ast.func)
        #     arg_keys = [key for key in inner_ast.keys() if key.startswith('arg')]

        #     for arg_key in arg_keys:
        #         term = inner_ast[arg_key].term  # type: ignore
        #         if isinstance(term, str) and term.startswith('?'):
        #             # If we already have a mapping for it, replace it with the mapping
        #             if term in global_context['replacement_mappings']:
        #                 replace_child(inner_ast, [arg_key, 'term'], '?' + global_context['replacement_mappings'][term])

        #             # If we don't have a mapping for it, but it's in the local context, add a mapping
        #             elif term not in local_context['variables']:
        #                 if len(local_context['variables']) == 0:
        #                     raise SamplingException('No variables found in local context when updating a predicate/function_eval node with variable children')

        #                 new_var_name = self.rng.choice(list(local_context['variables'].keys()))
        #                 global_context['replacement_mappings'][term] = new_var_name
        #                 replace_child(inner_ast, [arg_key, 'term'], '?' + new_var_name)

        elif rule == 'pref_name_and_types':
            if 'preference_names' not in global_context:
                raise SamplingException('No preference names found in global context when updating a count.pref_name_and_types node')

            preference_names = global_context['preference_names']
            if ast.pref_name not in preference_names:
                if 'replacement_mappings' not in local_context:
                    global_context['replacement_mappings'] = dict()

                if ast.pref_name in global_context['replacement_mappings']:
                    new_pref_name = global_context['replacement_mappings'][ast.pref_name]

                else:
                    new_pref_name = self.rng.choice(list(preference_names))
                    global_context['replacement_mappings'][ast.pref_name] = new_pref_name

                replace_child(ast, ['pref_name'], new_pref_name)

        for key in ast:
            if key != 'parseinfo':
                retval = self(ast[key], **kwargs)  # type: ignore
                self._update_contexts(kwargs, retval)

        if rule in self.local_context_propagating_rules:
            return kwargs['global_context'], kwargs['local_context']

        return kwargs['global_context'], None


class CrossoverType(Enum):
    SAME_RULE = 0
    SAME_PARENT = 1
    SAME_PARENT_RULE = 2
    SAME_PARENT_RULE_SELECTOR = 3


def _get_node_key(node: typing.Any):
    if isinstance(node, tatsu.ast.AST):
        if node.parseinfo.rule is None:  # type: ignore
            raise ValueError('Node has no rule')
        return node.parseinfo.rule  # type: ignore

    else:
        return type(node).__name__


def node_info_to_key(crossover_type: CrossoverType, node_info: ASTNodeInfo):
    if crossover_type == CrossoverType.SAME_RULE:
        return _get_node_key(node_info[0])

    elif crossover_type == CrossoverType.SAME_PARENT:
        return _get_node_key(node_info[1])

    elif crossover_type == CrossoverType.SAME_PARENT_RULE:
        return '_'.join([_get_node_key(node_info[1]), _get_node_key(node_info[0])])

    elif crossover_type == CrossoverType.SAME_PARENT_RULE_SELECTOR:
        return '_'.join([_get_node_key(node_info[1]),  *[str(s) for s in node_info[2]],  _get_node_key(node_info[0])])

    else:
        raise ValueError(f'Invalid crossover type {crossover_type}')


class CrossoverSampler(RegrowthSampler):
    context_fixer: ASTContextFixer
    node_infos_by_crossover_type_key: typing.Dict[str, typing.List[ASTNodeInfo]]
    # node_keys_by_id: typing.Dict[str, typing.List[ParseInfo]]
    # parent_mapping_by_id: typing.Dict[str, ASTParentMapping]
    population: typing.List[typing.Union[tatsu.ast.AST, tuple]]

    def __init__(self, crossover_type: CrossoverType,
        population: typing.List[typing.Union[tatsu.ast.AST, tuple]],
        sampler: ASTSampler, seed: int = 0, use_tqdm: bool = False):
        super().__init__(sampler, seed)
        # a slightly ugly call to get the variable sampler with the right context
        self.context_fixer = ASTContextFixer(self.sampler.rules['variable_type_def']['var_names']['samplers']['variable'], self.sampler.local_context_propagating_rules, self.rng)
        self.crossover_type = crossover_type
        self.population = population
        self.node_infos_by_crossover_type_key = defaultdict(list)

        pop_iter = self.population
        if use_tqdm:
            pop_iter = tqdm.tqdm(pop_iter)

        for ast in pop_iter:
            self.set_source_ast(ast)
            # self.node_keys_by_id[self.original_game_id] = self.node_keys[:]
            # self.parent_mapping_by_id[self.original_game_id] = self.parent_mapping.copy()

            for node_info in self.parent_mapping.values():
                self.node_infos_by_crossover_type_key[node_info_to_key(self.crossover_type, node_info)].append(node_info)

        self.source_ast = None  # type: ignore

    def sample(self, sample_index: int, external_global_context: typing.Optional[ContextDict] = None,
        external_local_context: typing.Optional[ContextDict] = None, update_game_id: bool = True,
        crossover_key_to_use: typing.Optional[str] = None) -> typing.Union[tatsu.ast.AST, tuple]:

        crossover_node_copy = None
        crossover_node_copy_fixed = False

        while not crossover_node_copy_fixed:
            try:
                node_crossover_key = None
                node_info = None
                if crossover_key_to_use is not None:
                    while node_crossover_key != crossover_key_to_use:
                        node_info = self._sample_node_to_update()
                        node, parent, selector, global_context, local_context = node_info
                        node_crossover_key = node_info_to_key(self.crossover_type, node_info)

                else:
                    node_info = self._sample_node_to_update()
                    node_crossover_key = node_info_to_key(self.crossover_type, node_info)

                node, parent, selector, global_context, local_context = node_info  # type: ignore
                node_infos = self.node_infos_by_crossover_type_key[node_crossover_key]  # type: ignore

                if len(node_infos) == 0:
                    raise SamplingException(f'No nodes found with key {node_crossover_key}')

                if len(node_infos) == 1:
                    crossover_node_info = node_infos[0]
                    if crossover_node_info is node_info:
                        raise SamplingException(f'No other nodes found with key {node_crossover_key}')

                    crossover_node_copy = copy.deepcopy(crossover_node_info[0])
                    self.context_fixer(crossover_node_copy, global_context=global_context, local_context=local_context)
                    crossover_node_copy_fixed = True

                else:
                    crossover_node_info = node_info

                    while crossover_node_info is node_info:
                        index = self.rng.choice((len(node_infos)))
                        crossover_node_info = node_infos[index]
                        crossover_node_copy = copy.deepcopy(crossover_node_info[0])
                        self.context_fixer(crossover_node_copy, global_context=global_context, local_context=local_context)
                        crossover_node_copy_fixed = True

            except SamplingException:
                continue

        # TODO: if the node we're replacing is variable list, do we go to children of the siblings and update them?

        new_source = copy.deepcopy(self.source_ast)
        new_parent = self.searcher(new_source, parseinfo=parent.parseinfo)  # type: ignore
        replace_child(new_parent, selector, crossover_node_copy)  # type: ignore

        return new_source




if __name__ == '__main__':
    import argparse
    import os
    from ast_counter_sampler import *
    from ast_utils import cached_load_and_parse_games_from_file

    DEFAULT_ARGS = argparse.Namespace(
        grammar_file=DEFAULT_GRAMMAR_FILE,
        parse_counter=False,
        counter_output_path=DEFAULT_COUNTER_OUTPUT_PATH,
        random_seed=33,
    )

    grammar = open(DEFAULT_ARGS.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)
    counter = parse_or_load_counter(DEFAULT_ARGS, grammar_parser)
    sampler = ASTSampler(grammar_parser, counter, seed=DEFAULT_ARGS.random_seed)
    asts = [ast for ast in cached_load_and_parse_games_from_file('./dsl/interactive-beta.pddl',
        grammar_parser, False)]


    crossover_sampler = CrossoverSampler(
        CrossoverType.SAME_RULE,
        asts[1:],
        sampler,
        DEFAULT_ARGS.random_seed,
        use_tqdm=True,
    )
