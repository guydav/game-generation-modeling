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

from ast_counter_sampler import parse_or_load_counter, ASTSampler, RegrowthSampler, SamplingException, MCMC_REGRWOTH
from ast_crossover_sampler import CrossoverSampler, CrossoverType
import ast_printer
from fitness_features import build_fitness_featurizer
from fitness_energy_utils import NON_FEATURE_COLUMNS

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./src'))

DEFUALT_RANDOM_SEED = 33
DEFAULT_FITNESS_FUNCTION_PATH = './models/cv_fitness_model_2023_01_31.pkl.gz'
DEFAULT_FITNESS_FEATURIZER_PATH = './models/fitness_featurizer_2023_02_02.pkl.gz'


def _load_pickle_gzip(path: str):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def _load_and_wrap_fitness_function(fitness_function_path: str = DEFAULT_FITNESS_FUNCTION_PATH) -> typing.Callable[[torch.Tensor], float]:
    cv_fitness_model = _load_pickle_gzip(fitness_function_path)

    def _wrap_fitness(features: torch.Tensor):
        return cv_fitness_model.transform(features).item()

    return _wrap_fitness


DEFAULT_PLATEAU_PATIENCE_STEPS = 10
DEFAULT_MAX_STEPS = 1000
DEFAULT_ACCEPTANCE_TEMPERATURE = 1.0


class MCMCRegrowthSampler:
    def __init__(self,
        args: argparse.Namespace,
        fitness_function_path: str = DEFAULT_FITNESS_FUNCTION_PATH,
        fitness_featurizer_path: str = DEFAULT_FITNESS_FEATURIZER_PATH,
        plateau_patience_steps: int = DEFAULT_PLATEAU_PATIENCE_STEPS,
        max_steps: int = DEFAULT_MAX_STEPS,
        greedy_acceptance: bool = False,
        acceptance_temperature: float = DEFAULT_ACCEPTANCE_TEMPERATURE,
    ):
        self.grammar = open(args.grammar_file).read()
        self.grammar_parser = tatsu.compile(self.grammar)
        self.counter = parse_or_load_counter(args, self.grammar_parser)
        self.sampler = ASTSampler(self.grammar_parser, self.counter, seed=args.random_seed)
        self.regrowth_sampler = RegrowthSampler(self.sampler, args.random_seed)
        self.rng = np.random.default_rng(args.random_seed)

        self.fitness_featurizer_path = fitness_featurizer_path
        self.fitness_featurizer = _load_pickle_gzip(fitness_featurizer_path)
        self.fitness_function_path = fitness_function_path
        self.fitness_function = _load_and_wrap_fitness_function(fitness_function_path)
        self.plateau_patience_steps = plateau_patience_steps
        self.max_steps = max_steps
        self.greedy_acceptance = greedy_acceptance
        self.acceptance_temperature = acceptance_temperature

        self.sample_index = 0
        self.step_index = -1
        self.samples = []

    def multiple_samples(self, n_samples: int, verbose: int = 0, should_tqdm: bool = False):
        sample_iter = tqdm.notebook.trange(n_samples) if should_tqdm else range(n_samples)
        for _ in sample_iter:
            self.sample(verbose)

    def sample(self, verbose: int = 0):
        current_proposal = None
        while current_proposal is None:
            try:
                current_proposal = typing.cast(tuple, self.sampler.sample(global_context=dict(original_game_id=f'mcmc-{self.sample_index}')))
            except RecursionError:
                if verbose >= 2: print('Recursion error, skipping sample')
            except SamplingException:
                if verbose >= 2: print('Sampling exception, skipping sample')

        current_proposal_features, current_proposal_fitness = self._score_proposal(current_proposal)  # type: ignore

        last_accepted_step = 0
        for step in range(self.max_steps):
            self.step_index = step
            current_proposal, current_proposal_features, current_proposal_fitness, accepted = self.mcmc_step(
                current_proposal, current_proposal_features, current_proposal_fitness, verbose
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

        self.samples.append((current_proposal, current_proposal_features, current_proposal_fitness))
        self.sample_index += 1

    def _generate_step_proposal(self) -> typing.Union[tatsu.ast.AST, tuple]:
        return self.regrowth_sampler.sample(self.step_index, update_game_id=False)  # type: ignore

    def _pre_mcmc_step(self, current_proposal: typing.Union[tatsu.ast.AST, tuple]):
        if self.regrowth_sampler.source_ast != current_proposal:
            self.regrowth_sampler.set_source_ast(current_proposal)

    def mcmc_step(self,
        current_proposal: typing.Union[tatsu.ast.AST, tuple],
        current_proposal_features: typing.Dict[str, float],
        current_proposal_fitness: float,
        verbose: int,
        ) -> typing.Tuple[typing.Union[tatsu.ast.AST, tuple], typing.Dict[str, float], float, bool] :

        self._pre_mcmc_step(current_proposal)

        new_proposal = None
        sample_generated = False

        while not sample_generated:
            try:
                new_proposal = self._generate_step_proposal()
                # _test_ast_sample(ast, args, text_samples, grammar_parser)
                if ast_printer.ast_to_string(new_proposal) == ast_printer.ast_to_string(current_proposal):  # type: ignore
                    if verbose >= 2: print('Regrowth generated identical games, repeating')
                else:
                    sample_generated = True

            except RecursionError:
                if verbose >= 2: print('Recursion error, skipping sample')

            except SamplingException:
                if verbose >= 2: print('Sampling exception, skipping sample')

        new_proposal = typing.cast(tuple, new_proposal)
        new_proposal_features, new_proposal_fitness = self._score_proposal(new_proposal)  # type: ignore

        if self.greedy_acceptance:
            accept = new_proposal_fitness < current_proposal_fitness
        else:
            acceptance_probability = np.exp(-self.acceptance_temperature * (new_proposal_fitness - current_proposal_fitness))
            accept = self.rng.uniform() < acceptance_probability

        if accept:
            return new_proposal, new_proposal_features, new_proposal_fitness, True
        else:
            return current_proposal, current_proposal_features, current_proposal_fitness, False

    def _score_proposal(self, proposal: tatsu.ast.AST):
        proposal_features = typing.cast(dict, self.fitness_featurizer.parse(proposal, 'mcmc', True))  # type: ignore
        proposal_tensor = torch.tensor([v for k, v in proposal_features.items() if k not in NON_FEATURE_COLUMNS],
            dtype=torch.float32)  # type: ignore
        proposal_fitness = self.fitness_function(proposal_tensor)
        return proposal_features, proposal_fitness


class MCMCRegrowthCrossoverSampler(MCMCRegrowthSampler):
    def __init__(self,
        args: argparse.Namespace,
        crossover_type: CrossoverType,
        crossover_population: typing.List[typing.Union[tatsu.ast.AST, tuple]],
        p_crossover: float,
        fitness_function_path: str = DEFAULT_FITNESS_FUNCTION_PATH,
        plateau_patience_steps: int = DEFAULT_PLATEAU_PATIENCE_STEPS,
        max_steps: int = DEFAULT_MAX_STEPS,
        greedy_acceptance: bool = False,
        acceptance_temperature: float = DEFAULT_ACCEPTANCE_TEMPERATURE
        ):
        super().__init__(args=args,
            fitness_function_path=fitness_function_path, plateau_patience_steps=plateau_patience_steps,
            max_steps=max_steps, greedy_acceptance=greedy_acceptance, acceptance_temperature=acceptance_temperature
        )
        self.crossover_sampler = CrossoverSampler(crossover_type, crossover_population, self.sampler, args.random_seed)
        self.p_crossover = p_crossover
        self.step_index_sampled = -1
        self.crossover_current_step = None

    def _pre_mcmc_step(self, current_proposal: tatsu.ast.AST):
        # TODO: consider in the future to optimize setting source ast only to the sampler chosen for this step
        if self.crossover_sampler.source_ast != current_proposal:
            self.crossover_sampler.set_source_ast(current_proposal)

        return super()._pre_mcmc_step(current_proposal)

    def _generate_step_proposal(self):
        if self.step_index_sampled < self.step_index:
            self.crossover_current_step = self.rng.uniform() < self.p_crossover
            self.step_index_sampled = self.step_index

        if self.crossover_current_step:
            return self.crossover_sampler.sample(self.step_index, update_game_id=False)
        else:
            return super()._generate_step_proposal()


def main(args: argparse.Namespace):
    original_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(args.recursion_limit)

    mcmc = MCMCRegrowthSampler(args, DEFAULT_FITNESS_FUNCTION_PATH, greedy_acceptance=False,
        plateau_patience_steps=50, acceptance_temperature=10.0, max_steps=3000)
    mcmc.multiple_samples(args.num_samples, verbose=args.verbose, should_tqdm=args.sample_tqdm)

    if args.save_samples:
        samples = mcmc.samples
        samples.sort(key=lambda x: x[1])

        text_samples = [ast_printer.ast_to_string(sample, line_delimiter='\n') for sample, _ in samples]

        with open(args.samples_output_path, 'w') as out_file:
            out_file.writelines(text_samples)

    sys.setrecursionlimit(original_recursion_limit)


# if __name__ == '__main__':
#     cmd_args = sys.argv[1:]
#     if '--sampling-method' not in cmd_args:
#         cmd_args += ['--sampling-method', MCMC_REGRWOTH]

#     args = parser.parse_args(cmd_args)
#     main(args)
