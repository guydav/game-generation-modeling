import argparse
import gzip
import os
import pickle
import typing
import sys

import numpy as np
import tqdm
import tatsu
import tatsu.ast
import torch

import ast_printer
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

        self.population_size = population_size
        self.verbose = verbose

        self.grammar = open(args.grammar_file).read()
        self.grammar_parser = tatsu.compile(self.grammar)
        self.counter = parse_or_load_counter(args, self.grammar_parser)

        # Used to generate the initial population of complete games
        self.initial_sampler = ASTSampler(self.grammar_parser, self.counter, seed=args.random_seed)
        
        # Used as the mutation operator to modify existing games
        self.regrowth_sampler = RegrowthSampler(self.initial_sampler, args.random_seed)

        # Generate the initial population
        self.population = [self._init_sample(idx) for idx in range(self.population_size)]

    def _init_sample(self, idx):
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

    def step(self):
        pass

if __name__ == '__main__':

    DEFAULT_GRAMMAR_FILE = './dsl/dsl.ebnf'
    DEFAULT_COUNTER_OUTPUT_PATH ='./data/ast_counter.pickle'
    DEFUALT_RANDOM_SEED = 0
    # DEFAULT_NGRAM_MODEL_PATH = '../models/text_5_ngram_model_2023_02_17.pkl'
    DEFAULT_NGRAM_MODEL_PATH = '../models/ast_7_ngram_model_2023_02_16.pkl'

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
        return ngram_model.score(game, k=None, score_all=False)['score']

    evosampler = EvolutionarySampler(DEFAULT_ARGS, fitness_function, verbose=0)

    for i in range(len(evosampler.population)):
        print(fitness_function(evosampler.population[i]))