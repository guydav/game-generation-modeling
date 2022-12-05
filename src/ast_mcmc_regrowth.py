import argparse
import gzip
import os
import pickle
import typing
import sys

import torch
from ast_counter_sampler import *
from fitness_features import build_fitness_featurizer
from fitness_energy_utils import NON_FEATURE_COLUMNS

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./src'))

DEFUALT_RANDOM_SEED = 33
DEFAULT_FITNESS_FUNCTION_PATH = './models/cv_fitness_model_with_cross_example_2022_12_05.pkl.gz'


def _load_and_wrap_fitness_function(fitness_function_path: str = DEFAULT_FITNESS_FUNCTION_PATH) -> typing.Callable[[torch.Tensor], float]:
    with gzip.open(fitness_function_path, 'rb') as f:
        cv_fitness_model = pickle.load(f)

    def _wrap_fitness(features: torch.Tensor):
        return cv_fitness_model.transform(features).item()

    return _wrap_fitness


class MCMCRegrowthSampler:
    def __init__(self,
        args: argparse.Namespace,
        fitness_function_path: str = DEFAULT_FITNESS_FUNCTION_PATH,
        plateau_patience_steps: int = 10,
        max_steps: int = 1000,
        greedy_acceptance: bool = False,
        acceptance_temperature: float = 1.0,
    ):
        self.grammar = open(args.grammar_file).read()
        self.grammar_parser = tatsu.compile(self.grammar)
        self.counter = parse_or_load_counter(args, self.grammar_parser)
        self.sampler = ASTSampler(self.grammar_parser, self.counter, seed=args.random_seed)
        self.regrowth_sampler = RegrowthSampler(self.sampler, args.random_seed)
        self.fitness_featurizer = build_fitness_featurizer(args)
        self.rng = np.random.default_rng(args.random_seed)
        self.fitness_function = _load_and_wrap_fitness_function(fitness_function_path)
        self.plateau_patience_steps = plateau_patience_steps
        self.max_steps = max_steps
        self.greedy_acceptance = greedy_acceptance
        self.acceptance_temperature = acceptance_temperature

        self.sample_index = 0
        self.samples = []

    def multiple_samples(self, n_samples: int, should_tqdm: bool = False, verbose: bool = False, should_print: bool = False):
        sample_iter = tqdm.trange(n_samples) if should_tqdm else range(n_samples)
        for _ in sample_iter:
            self.sample(verbose=verbose, should_print=should_print)

    def sample(self, verbose: bool = False, should_print: bool = False):
        current_proposal = None
        while current_proposal is None:
            try:
                current_proposal = self.sampler.sample(global_context=dict(original_game_id=f'mcmc-{self.sample_index}'))
            except RecursionError:
                print('Recursion error, skipping sample')
            except SamplingException:
                print('Sampling exception, skipping sample')
                
        current_proposal_fitness = self._score_proposal(current_proposal)  # type: ignore

        last_accepted_step = 0
        for step in range(self.max_steps):
            current_proposal, current_proposal_fitness, accepted = self.mcmc_regrowth_step(
                current_proposal, current_proposal_fitness, step  # type: ignore
            )

            if accepted:
                last_accepted_step = step
                if verbose:
                    print(f'Accepted step {step} with fitness {current_proposal_fitness}')

            else:
                if step - last_accepted_step > self.plateau_patience_steps:
                    if verbose:
                        print(f'Plateaued at step {step} with fitness {current_proposal_fitness}')
                    break

        if should_print:
            print(f'Sample #{self.sample_index} with fitness {current_proposal_fitness:.4f}:')
            print(ast_printer.ast_to_string(current_proposal, line_delimiter='\n'))
            print('=' * 120)

        self.samples.append((current_proposal, current_proposal_fitness))
        self.sample_index += 1

    def mcmc_regrowth_step(self,
        current_proposal: tatsu.ast.AST, 
        current_proposal_fitness: float,
        step_index: int,
        ):

        if self.regrowth_sampler.source_ast != current_proposal:
            self.regrowth_sampler.set_source_ast(current_proposal)

        new_proposal = None
        sample_generated = False
        while not sample_generated:
            try:
                new_proposal = self.regrowth_sampler.sample(step_index)
                # _test_ast_sample(ast, args, text_samples, grammar_parser)
                if ast_printer.ast_to_string(new_proposal) == ast_printer.ast_to_string(current_proposal):  # type: ignore
                    print('Regrowth generated identical games, repeating')
                else:
                    sample_generated = True
            except RecursionError:
                print('Recursion error, skipping sample')

            except SamplingException:
                print('Sampling exception, skipping sample')

        new_proposal_fitness = self._score_proposal(new_proposal)  # type: ignore
        
        if self.greedy_acceptance:
            if new_proposal_fitness < current_proposal_fitness:
                return new_proposal, new_proposal_fitness, True
            else:
                return current_proposal, current_proposal_fitness, False

        else:
            acceptance_probability = np.exp(-self.acceptance_temperature * (new_proposal_fitness - current_proposal_fitness))
            if self.rng.uniform() < acceptance_probability:
                return new_proposal, new_proposal_fitness, True
            else:
                return current_proposal, current_proposal_fitness, False

    def _score_proposal(self, proposal: tatsu.ast.AST):
        proposal_features = self.fitness_featurizer.parse(proposal, 'mcmc', True)  # type: ignore
        proposal_tensor = torch.tensor([v for k, v in proposal_features.items() if k not in NON_FEATURE_COLUMNS])  # type: ignore
        proposal_fitness = self.fitness_function(proposal_tensor)
        return proposal_fitness


def main(args: argparse.Namespace):
    original_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(args.recursion_limit)

    mcmc = MCMCRegrowthSampler(args, DEFAULT_FITNESS_FUNCTION_PATH, greedy_acceptance=True, plateau_patience_steps=30)
    mcmc.multiple_samples(args.num_samples, verbose=args.verbose, should_tqdm=args.sample_tqdm, should_print=args.print_samples)

    if args.save_samples:
        samples = mcmc.samples
        samples.sort(key=lambda x: x[1])

        text_samples = [ast_printer.ast_to_string(sample, line_delimiter='\n') for sample, _ in samples]

        with open(args.samples_output_path, 'w') as out_file:
            out_file.writelines(text_samples)
    
    sys.setrecursionlimit(original_recursion_limit)


if __name__ == '__main__':
    cmd_args = sys.argv[1:]
    if '--sampling-method' not in cmd_args:
        cmd_args += ['--sampling-method', MCMC_REGRWOTH]

    args = parser.parse_args(cmd_args)    
    main(args)

