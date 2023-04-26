import argparse
from collections import defaultdict
import os
import sys
import typing

import numpy as np
import tatsu
import tatsu.ast

import ast_printer
import ast_parser
from ast_counter_sampler import ASTSampler, SamplingException, parse_or_load_counter
from ast_parser import ASTNodeInfo, ASTParser, ASTParentMapper, ContextDict, VARIABLES_CONTEXT_KEY, VARIABLE_OWNER_CONTEXT_KEY_PREFIX
from ast_utils import replace_child
from latest_model_paths import LATEST_AST_N_GRAM_MODEL_PATH, LATEST_FITNESS_FEATURIZER_PATH, LATEST_FITNESS_FUNCTION_DATE_ID

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import src


RPLACEMENT_MAPPINGS_CONTEXT_KEY = 'replacement_mappings'
FORCED_REMAPPINGS_CONTEXT_KEY = 'forced_remappings'


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
        self.local_variable_ref_counts = defaultdict(dict)

    def _add_ast_to_mapping(self, ast, **kwargs):
        # NOOP here since I don't actually care about building a parent mapping, just wanted to use the structure
        pass

    def _sample_from_sequence(self, sequence: typing.Sequence, **kwargs):
        index = self.rng.choice(len(sequence))
        return sequence[index]

    def fix_contexts(self, crossed_over_game: tatsu.ast.AST, original_child: typing.Optional[tatsu.ast.AST] = None,
                     crossover_child: typing.Optional[tatsu.ast.AST] = None):
        self.preference_count_nodes = defaultdict(list)
        self.local_variable_ref_counts = defaultdict(dict)

        # if the crossover child defines any preferences, we need to note them, so we can add a reference to them at some point later
        preference_names_to_add = self.preference_name_finder(crossover_child) if crossover_child is not None else set()

        # if the original child defines any preferences, we need to remove them from the global context, and remove any references to them
        preference_names_to_remove = self.preference_name_finder(original_child) if original_child is not None else set()

        # TODO: implement a similar logic to the one for preference names, but for variables

        names_in_both_sets = preference_names_to_add.intersection(preference_names_to_remove)
        preference_names_to_add.difference_update(names_in_both_sets)
        preference_names_to_remove.difference_update(names_in_both_sets)

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
            self.local_variable_ref_counts[variables_key][var_name[1:]] = 0

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
                    global_context[RPLACEMENT_MAPPINGS_CONTEXT_KEY][var_name] = new_var_name
                    local_context[variables_key][new_var_name] = ast.parseinfo
                    replace_child(ast, ['var_names'] if single_variable else ['var_names', i], new_var)

            else:
                local_context[variables_key][var_name[1:]] = ast.parseinfo.pos  # type: ignore

    def _should_rehandle_current_node(self, ast: tatsu.ast.AST, **kwargs):
        should_rehandle = False
        local_context = kwargs['local_context']
        global_context = kwargs['global_context']
        forced_remappings = {}

        for key in local_context:
            if key.startswith(VARIABLE_OWNER_CONTEXT_KEY_PREFIX):
                variables_key = key[len(VARIABLE_OWNER_CONTEXT_KEY_PREFIX) + 1:]
                owned_variables = [var for var, pos in local_context[key].items() if pos == ast.parseinfo.pos]  # type: ignore
                if owned_variables:
                    missing_variables = [var for var in owned_variables if var not in self.local_variable_ref_counts[variables_key] or self.local_variable_ref_counts[variables_key][var] == 0]
                    if missing_variables:
                        should_rehandle = True

                        for missing_var in missing_variables:
                            potential_replacements = {var: self.local_variable_ref_counts[variables_key][var]
                                                      for var in owned_variables
                                                      if var in self.local_variable_ref_counts[variables_key] and self.local_variable_ref_counts[variables_key][var] > 1}

                            if not potential_replacements:
                                raise SamplingException(f'Could not find a replacement for variable {missing_var}')

                            var_to_replace = self._sample_from_sequence(list(potential_replacements.keys()))
                            replacement_index = self.rng.integers(potential_replacements[var_to_replace])
                            if variables_key not in forced_remappings:
                                forced_remappings[variables_key] = defaultdict(dict)

                            forced_remappings[variables_key][var_to_replace][replacement_index] = missing_var

                    for var in owned_variables:
                        del self.local_variable_ref_counts[variables_key][var]
                        if var in global_context[RPLACEMENT_MAPPINGS_CONTEXT_KEY]:
                            del global_context[RPLACEMENT_MAPPINGS_CONTEXT_KEY][var]

        if should_rehandle:
            print(f'Forced remappings: {forced_remappings}')

        return should_rehandle, (None, {FORCED_REMAPPINGS_CONTEXT_KEY: forced_remappings})

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
                var_name = term[1:]
                if term_variables_key not in self.local_variable_ref_counts:
                    self.local_variable_ref_counts[term_variables_key] = {}
                #     raise SamplingException(f'No ref count found for {term_variables_key} when updating a predicate/function_eval node with variable children')

                if var_name not in self.local_variable_ref_counts[term_variables_key]:
                    self.local_variable_ref_counts[term_variables_key][var_name] = 0
                #     raise SamplingException(f'No ref count found for {term} when updating a predicate/function_eval node with variable children')

                # If we have a forced remapping for this variable, at this position, use it (before incrementing the ref count)
                reference_index = self.local_variable_ref_counts[term_variables_key][var_name]

                if (FORCED_REMAPPINGS_CONTEXT_KEY in local_context and
                    term_variables_key in local_context[FORCED_REMAPPINGS_CONTEXT_KEY] and
                    var_name in local_context[FORCED_REMAPPINGS_CONTEXT_KEY][term_variables_key] and
                    reference_index in local_context[FORCED_REMAPPINGS_CONTEXT_KEY][term_variables_key][var_name]):

                    new_var_name = local_context[FORCED_REMAPPINGS_CONTEXT_KEY][term_variables_key][var_name][reference_index]
                    replace_child(ast, ['term'], '?' + new_var_name)
                    term = '?' + new_var_name
                    del local_context[FORCED_REMAPPINGS_CONTEXT_KEY][term_variables_key][var_name][reference_index]

                # If we already have a mapping for it, replace it with the mapping
                if term[1:] in global_context[RPLACEMENT_MAPPINGS_CONTEXT_KEY]:
                    term = '?' + global_context[RPLACEMENT_MAPPINGS_CONTEXT_KEY][term[1:]]
                    replace_child(ast, ['term'], term)

                else:
                    # Check if there's anything that we could map it to
                    if term_variables_key not in local_context or len(local_context[term_variables_key]) == 0:
                        raise SamplingException(f'No variable context (with key {term_variables_key}) found for {term} when updating a predicate/function_eval node with variable children')

                    # If we don't have a mapping for it, and it's not in the local context, add a mapping
                    elif term[1:] not in local_context[term_variables_key]:
                        new_var_name = self._sample_from_sequence(list(local_context[term_variables_key].keys()))
                        global_context[RPLACEMENT_MAPPINGS_CONTEXT_KEY][term[1:]] = new_var_name
                        term = '?' + new_var_name
                        replace_child(ast, ['term'], term)

                self.local_variable_ref_counts[term_variables_key][term[1:]] += 1

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


if __name__ == '__main__':
    DEFAULT_GRAMMAR_FILE = './dsl/dsl.ebnf'
    DEFAULT_COUNTER_OUTPUT_PATH ='./data/ast_counter.pickle'
    DEFUALT_RANDOM_SEED = 0
    # DEFAULT_NGRAM_MODEL_PATH = '../models/text_5_ngram_model_2023_02_17.pkl'
    DEFAULT_NGRAM_MODEL_PATH = LATEST_AST_N_GRAM_MODEL_PATH

    DEFAULT_ARGS = argparse.Namespace(
        grammar_file=os.path.join('.', DEFAULT_GRAMMAR_FILE),
        parse_counter=False,
        counter_output_path=os.path.join('.', DEFAULT_COUNTER_OUTPUT_PATH),
        random_seed=DEFUALT_RANDOM_SEED,
    )

    test_game = """
(define (game 6172feb1665491d1efbce164-0) (:domain medium-objects-room-v1)  ; 0
(:setup (and
    (exists (?h - hexagonal_bin ?r - triangular_ramp)
        (game-conserved (< (distance ?h ?r) 1))
    )
))
(:constraints (and
    (preference binKnockedOver
        (exists (?h - hexagonal_bin ?b - ball)
            (then
                (hold (and (not (touch agent ?h)) (not (agent_holds ?h))))
                (once (not (object_orientation ?h upright)))
            )
        )
    )
))
(:scoring (count throwToRampToBin)
))
"""
    args = DEFAULT_ARGS
    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)
    counter = parse_or_load_counter(args, grammar_parser)

    # Used to generate the initial population of complete games
    sampler = ASTSampler(grammar_parser, counter, seed=args.random_seed)  # type: ignore
    context_fixer = ASTContextFixer(sampler, np.random.default_rng(args.random_seed))
    ast = grammar_parser.parse(test_game)  # type: ignore
    print(ast_printer.ast_to_string(ast, '\n'))
    context_fixer.fix_contexts(ast)
    print(ast_printer.ast_to_string(ast, '\n'))
