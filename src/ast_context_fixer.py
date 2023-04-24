from collections import defaultdict
import typing

import numpy as np
import tatsu
import tatsu.ast

import ast_printer
import ast_parser
from ast_counter_sampler import ASTSampler, SamplingException
from ast_parser import ASTParser, ASTParentMapper, ContextDict, VARIABLES_CONTEXT_KEY
from ast_utils import replace_child,simplified_context_deepcopy


class ASTDefinedPreferenceNamesFinder(ASTParser):
    def __init__(self):
        super().__init__()

    def __call__(self, ast: tatsu.ast.AST, **kwargs):
        self._default_kwarg(kwargs, 'preference_names', set())
        super().__call__(ast, **kwargs)
        return kwargs['preference_names']

    def _handle_ast(self, ast: tatsu.ast.AST, **kwargs):
        rule = typing.cast(str, ast.parseinfo.rule)  # type: ignore
        if rule == 'preference':
            kwargs['preference_names'].add(ast.pref_name)

        else:
            super()._handle_ast(ast, **kwargs)


class ASTContextFixer(ASTParentMapper):
    preference_count_nodes: typing.Dict[str, typing.List[tatsu.ast.AST]]
    preference_name_finder: ASTDefinedPreferenceNamesFinder
    rng: np.random.Generator
    sampler: ASTSampler

    def __init__(self, sampler: ASTSampler, rng: np.random.Generator):
        super().__init__(local_context_propagating_rules=sampler.local_context_propagating_rules)
        self.sampler = sampler
        self.rng = rng
        self.preference_name_finder = ASTDefinedPreferenceNamesFinder()
        self.preference_count_nodes = defaultdict(list)

    def _add_ast_to_mapping(self, ast, **kwargs):
        # NOOP here since I don't actually care about building a parent mapping, just wanted to use the structure
        pass

    def _sample_from_sequence(self, sequence: typing.Sequence, **kwargs):
        index = self.rng.choice(len(sequence))
        return sequence[index]

    def fix_contexts(self, crossed_over_game: tatsu.ast.AST, original_child: tatsu.ast.AST, crossover_child: tatsu.ast.AST):
        self.preference_count_nodes = defaultdict(list)

        # if the crossover child defines any preferences, we need to note them, so we can add a reference to them at some point later
        preference_names_to_add = self.preference_name_finder(crossover_child)

        # if the original child defines any preferences, we need to remove them from the global context, and remove any references to them
        preference_names_to_remove = self.preference_name_finder(original_child)

        names_in_both_lists = preference_names_to_add.intersection(preference_names_to_remove)
        preference_names_to_add.difference_update(names_in_both_lists)
        preference_names_to_remove.difference_update(names_in_both_lists)

        self(crossed_over_game, global_context=dict(preference_names_to_add=preference_names_to_add,
                                                    preference_names_to_remove=preference_names_to_remove,
                                                    replacement_mappings=dict(),))

        # If any preference names still remain unadded, we need to add them to the game
        if len(preference_names_to_add) > 0:
            for pref_name_to_add in preference_names_to_add:
                count_nodes_with_multiple_occurences = sum([pref_nodes for pref_nodes in self.preference_count_nodes.values() if len(pref_nodes) > 1], [])
                if len(count_nodes_with_multiple_occurences) == 0:
                    raise SamplingException(f'Could not find a node to add preference {pref_name_to_add} to')

                node_to_add_pref_to = self._sample_from_sequence(count_nodes_with_multiple_occurences)
                current_pref_name = node_to_add_pref_to.pref_name
                replace_child(node_to_add_pref_to, ['pref_name'], pref_name_to_add)
                self.preference_count_nodes[current_pref_name].remove(node_to_add_pref_to)
                self.preference_count_nodes[pref_name_to_add].append(node_to_add_pref_to)

    def _single_variable_def_context_update(self, ast: tatsu.ast.AST, local_context: ContextDict, global_context: ContextDict):
        rule = ast.parseinfo.rule  # type: ignore
        var_names = ast.var_names
        single_variable = False
        if isinstance(var_names, str):
            var_names = [var_names]
            single_variable = True

        for i, var_name in enumerate(var_names):  # type: ignore
            variables_key = self._variable_type_def_rule_to_context_key(rule)

            if var_name in local_context[variables_key]:
                if local_context[variables_key][var_name] == ast.parseinfo.pos:  # type: ignore
                    continue

                else:
                    if variables_key == VARIABLES_CONTEXT_KEY:
                        new_variable_sampler = self.sampler.rules[rule]['var_names']['samplers']['variable']
                    else:
                        new_variable_sampler = self.sampler.rules[rule]['var_names']['samplers'][f'{variables_key.split("_")[0]}_variable']

                    new_var = new_variable_sampler(global_context, local_context)
                    new_var_name = new_var[1:]

                    # TODO: this assumes we want to consistenly map each missing variable to a new variable, which may or may not be the case -- we should discsuss
                    global_context['replacement_mappings'][var_name] = new_var_name
                    local_context['variables'][new_var_name] = ast.parseinfo
                    replace_child(ast, ['var_names'] if single_variable else ['var_names', i], new_var)

            else:
                local_context[variables_key][var_name[1:]] = ast.parseinfo.pos  # type: ignore

    def _parse_current_node(self, ast: tatsu.ast.AST, **kwargs):
        local_context = kwargs['local_context']
        global_context = kwargs['global_context']

        rule = typing.cast(str, ast.parseinfo.rule)  # type: ignore

        if rule == 'variable_list':
            if isinstance(ast.variables, tatsu.ast.AST):
                self._single_variable_def_context_update(ast.variables, local_context, global_context)

            else:
                for var_def in ast.variables:  # type: ignore
                    self._single_variable_def_context_update(var_def, local_context, global_context)

        elif rule.startswith('predicate_or_function_'):
            term = ast.term
            term_type = rule.split('_')[3]
            type_def_rule = 'variable_type_def'
            if term_type not in ('term', 'location', 'type') :
                type_def_rule = f'{term_type}_{type_def_rule}'

            term_variables_key = self._variable_type_def_rule_to_context_key(type_def_rule)

            if isinstance(term, str) and term.startswith('?'):
                # If we already have a mapping for it, replace it with the mapping
                if term in global_context['replacement_mappings']:
                    replace_child(ast, ['term'], '?' + global_context['replacement_mappings'][term])

                else:
                    # Check if there's anything that we could map it to
                    if term_variables_key not in local_context or len(local_context[term_variables_key]) == 0:
                        raise SamplingException(f'No variable context (with key {term_variables_key}) found for {term} when updating a predicate/function_eval node with variable children')

                    # If we don't have a mapping for it, and it's not in the local context, add a mapping
                    elif term[1:] not in local_context[term_variables_key]:
                        new_var_name = self._sample_from_sequence(list(local_context[term_variables_key].keys()))
                        global_context['replacement_mappings'][term] = new_var_name
                        replace_child(ast, ['term'], '?' + new_var_name)

        elif rule == 'pref_name_and_types':
            if 'preference_names' not in global_context:
                raise SamplingException('No preference names found in global context when updating a count.pref_name_and_types node')

            preference_names = global_context['preference_names']
            if ast.pref_name not in preference_names:
                # TODO: do we want to be consistent with the preference names we map to?
                # if ast.pref_name in global_context['preference_names_to_remove']:
                if len(global_context['preference_names_to_add']) > 0:
                    new_pref_name = self._sample_from_sequence(list(global_context['preference_names_to_add']))
                    global_context['preference_names_to_add'].remove(new_pref_name)

                else:
                    new_pref_name = self._sample_from_sequence(list(preference_names))

                replace_child(ast, ['pref_name'], new_pref_name)

            self.preference_count_nodes[ast.pref_name].append(ast)  # type: ignore
