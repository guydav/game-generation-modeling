import argparse
import gzip
import os
import pickle
import typing
import sys

import numpy as np
import tatsu
import tatsu.ast
import torch
import tqdm

# from ast_parser import SETUP, PREFERENCES, TERMINAL, SCORING
import ast_printer
import ast_parser
from ast_counter_sampler import *
from ast_counter_sampler import parse_or_load_counter, ASTSampler, RegrowthSampler, SamplingException, MCMC_REGRWOTH
from ast_crossover_sampler import CrossoverSampler, CrossoverType
from ast_utils import *

from fitness_ngram_models import NGramTrieModel, NGramTrieNode, NGramASTParser, ASTNGramTrieModel

class EvolutionarySampler():
    '''
    This is a type of game sampler which uses an evolutionary strategy to climb a
    provided fitness function. It's a population-based alternative to the MCMC samper
    '''

    def __init__(self,
                 args: argparse.Namespace,
                 fitness_function: typing.Callable[[typing.Any], float],
                 population_size: int = 100,
                 verbose: int = 0):

        self.fitness_function = fitness_function
        self.population_size = population_size
        self.verbose = verbose

        self.grammar = open(args.grammar_file).read()
        self.grammar_parser = tatsu.compile(self.grammar)
        self.counter = parse_or_load_counter(args, self.grammar_parser)

        # Used to generate the initial population of complete games
        self.initial_sampler = ASTSampler(self.grammar_parser, self.counter, seed=args.random_seed)
        
        # Used as the mutation operator to modify existing games
        self.regrowth_sampler = RegrowthSampler(self.initial_sampler, seed=args.random_seed)

        # Generate the initial population
        self.population = [self._gen_init_sample(idx) for idx in range(self.population_size)]
        self.fitness_values = [self.fitness_function(game) for game in self.population]

    def _best_fitness(self):
        return max(self.fitness_values)
    
    def _avg_fitness(self):
        return np.mean(self.fitness_values)
    
    def _best_individual(self):
        return self.population[np.argmax(self.fitness_values)]
    
    def _print_game(self, game):
        print(ast_printer.ast_to_string(game, "\n"))

    def _gen_init_sample(self, idx):
        '''
        Helper function for generating an initial sample (repeating until one is generated
        without errors)
        '''

        sample = None

        while sample is None:
            try:
                sample = typing.cast(tuple, self.initial_sampler.sample(global_context=dict(original_game_id=f'mcmc-{idx}')))
            except RecursionError:
                if self.verbose >= 2: print(f'Recursion error in sample {idx} -- skipping')
            except SamplingException:
                if self.verbose >= 2: print('Sampling exception in sample {idx} -- skipping')

        return sample
    
    def _gen_regrowth_sample(self, game):
        '''
        Helper function for generating a new sample from an existing game (repeating until one is generated
        without errors)
        '''

        # Set the source AST of the regrowth sampler to the current game
        self.regrowth_sampler.set_source_ast(game)

        new_proposal = None
        sample_generated = False

        while not sample_generated:
            try:
                new_proposal = self.regrowth_sampler.sample(sample_index=0, update_game_id=False) # TODO: does sample_index need to change? 

                if ast_printer.ast_to_string(new_proposal) == ast_printer.ast_to_string(game):  # type: ignore
                    if self.verbose >= 2: print('Regrowth generated identical games, repeating')
                else:
                    sample_generated = True

            except RecursionError:
                if self.verbose >= 2: print('Recursion error, skipping sample')

            except SamplingException:
                if self.verbose >= 2: print('Sampling exception, skipping sample')

        new_proposal = typing.cast(tuple, new_proposal)

        return new_proposal

    def beam_step(self, k: int = 10):
        '''
        The simplest kind of evolutionary step. Each member of the population is mutated k times using
        the regrowth sampler, giving a total of k * n new samples. Each sample is scored using the fitness
        function, and then the top n samples are selected to form the next generation (including the initial samples)
        '''

        # Generate the new samples
        samples = self.population.copy()
        for game in self.population:
            for i in range(k):
                samples.append(self._gen_regrowth_sample(game))

        # Score the new samples
        scores = [self.fitness_function(sample) for sample in samples]

        # Select the top n samples
        top_indices = np.argsort(scores)[-self.population_size:]
        self.population = [samples[i] for i in top_indices]
        self.fitness_values = [scores[i] for i in top_indices]

    def insert_delete_step(self):
        '''
        Perform an evolutionary step by mutating each member of the population. Each mutation consists of
        an independent probability of:
        - inserting a new node inside of an existing node with multiple children (e.g. an [and] node)
        - deleting an existing node if it is one of multiple children of its parent
        As with the beam step, the top n samples are selected to form the next generation
        '''

        for game in self.population:
            self.regrowth_sampler.set_source_ast(game)

            # After processing the game with the regrowth sampler, we can use its parent mapping to
            # determine which nodes are eligible for insertion and deletion. This relies on the fact
            # that:
            # - if a node is an element of a list, then the last entry of its selector will be an integer (since
            #   we need to index into that list to get the node)
            # - similarly, the first entry of the selector will yield a list when applied to the parent
            valid_nodes = list([(parent, selector[0], section, global_context, local_context) for _, parent, selector, _, section, global_context, local_context 
                                    in self.regrowth_sampler.parent_mapping.values() if isinstance(selector[-1], int) and isinstance(parent[selector[0]], list)])

            for parent, selector, section, global_context, local_context in valid_nodes:
                print(f"\nParent of list: {ast_printer.ast_section_to_string(parent, str(section))}")

                parent_rule = parent.parseinfo.rule # type: ignore
                parent_rule_posterior_dict = self.initial_sampler.rules[parent_rule][selector]
                assert "length_posterior" in parent_rule_posterior_dict, f"Rule {parent_rule} does not have a length posterior"
                
                # Determine whether we're sampling a rule or a token (for this case, it'll always be one or the other 100% of the time)
                if parent_rule_posterior_dict['type_posterior']['rule'] == 1:
                    new_rule = self.initial_sampler.sample(parent_rule, global_context, local_context)[0]
                    print(f"\tNew rule: {ast_printer.ast_section_to_string(new_rule, section)}")

                elif parent_rule_posterior_dict['type_posterior']['token'] == 1:
                    pass

                else:
                    raise Exception("Invalid type posterior")

            print(f"\n\n\n" + ast_printer.ast_to_string(game, '\n'))
            exit()

            valid_nodes = []
            used_parents = set()

            for _, parent, selector, _, section, global_context, local_context in self.regrowth_sampler.parent_mapping.values():
                # Check to make sure that the parent of the node is a list (so the last selector will be an int to index into the list)
                if isinstance(parent[selector[0]], list) and isinstance(selector[-1], int):

                    parent_rule = parent.parseinfo.rule
                    parent_rule_posterior_dict = self.initial_sampler.rules[parent_rule][selector]

            # game_nodes = [self.regrowth_sampler.parent_mapping[idx][0] for idx in self.regrowth_sampler.]

            # if a node is an element of a list, then the last entry of its selector will be an integer (since
            # we need to index into that list to get the node)

            # To sampler a child we need to know the rule. In the context of sampling a new entry of the list, the
            # global and local context should be the same. 
            # new_node = regrowth_sampler.sampler.sample(node.parseinfo.rule, global_context, local_context)[0]

            # 


if __name__ == '__main__':

    DEFAULT_GRAMMAR_FILE = './dsl/dsl.ebnf'
    DEFAULT_COUNTER_OUTPUT_PATH ='./data/ast_counter.pickle'
    DEFUALT_RANDOM_SEED = 0
    # DEFAULT_NGRAM_MODEL_PATH = '../models/text_5_ngram_model_2023_02_17.pkl'
    DEFAULT_NGRAM_MODEL_PATH = '../models/ast_7_ngram_model_2023_03_01.pkl'

    DEFAULT_ARGS = argparse.Namespace(
        grammar_file=os.path.join('..', DEFAULT_GRAMMAR_FILE),
        parse_counter=False,
        counter_output_path=os.path.join('..', DEFAULT_COUNTER_OUTPUT_PATH),
        random_seed=DEFUALT_RANDOM_SEED,
    )

    with open(DEFAULT_NGRAM_MODEL_PATH, 'rb') as file:
        ngram_model = pickle.load(file)

    def fitness_function(game):
        '''
        A simple wrapper around the ASTNGramTrieModel.score function, allowing it
        to run on just a single game and returning the score as float
        '''
        return ngram_model.score(game, k=None, top_k_min_n=5, score_all=False)['full_score']

    evosampler = EvolutionarySampler(DEFAULT_ARGS, fitness_function, verbose=0)

    evosampler.insert_delete_step()
    exit()

    for _ in tqdm.tqdm(range(10), desc='Evolutionary steps'):
        evosampler.beam_step(k=25)
        print(f"Average fitness: {evosampler._avg_fitness():.3f}, Best fitness: {evosampler._best_fitness():.3f}")

    print('Best individual:')
    evosampler._print_game(evosampler._best_individual())