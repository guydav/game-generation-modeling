import argparse
from functools import wraps
import gzip
import logging
import multiprocessing
from multiprocessing import pool as mpp
import os
import pickle
import typing
import sys

import numpy as np
import tatsu
import tatsu.ast
import tatsu.grammars
import torch
from tqdm import tqdm, trange

# from ast_parser import SETUP, PREFERENCES, TERMINAL, SCORING=
import ast_printer
import ast_parser
from ast_context_fixer import ASTContextFixer
from ast_counter_sampler import *
from ast_counter_sampler import parse_or_load_counter, ASTSampler, RegrowthSampler, SamplingException, MCMC_REGRWOTH
from ast_mcmc_regrowth import _load_pickle_gzip, InitialProposalSamplerType, create_initial_proposal_sampler, mpp
from ast_utils import *
from evolutionary_sampler_diversity import *
from fitness_energy_utils import load_model_and_feature_columns, save_data, DEFAULT_SAVE_MODEL_NAME, evaluate_single_game_energy_contributions
from fitness_features import *
from fitness_ngram_models import *
from latest_model_paths import LATEST_AST_N_GRAM_MODEL_PATH, LATEST_FITNESS_FEATURIZER_PATH, LATEST_FITNESS_FUNCTION_DATE_ID

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import src


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)  # type: ignore
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,  # type: ignore
                                          mpp.starmapstar,  # type: ignore
                                          task_batches),
            result._set_length  # type: ignore
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap  # type: ignore


parser = argparse.ArgumentParser(description='Evolutionary Sampler')
parser.add_argument('--grammar-file', type=str, default=DEFAULT_GRAMMAR_FILE)
parser.add_argument('--parse-counter', action='store_true')
parser.add_argument('--counter-output-path', type=str, default=DEFAULT_COUNTER_OUTPUT_PATH)

DEFAULT_FITNESS_FUNCTION_DATE_ID = LATEST_FITNESS_FUNCTION_DATE_ID
parser.add_argument('--fitness-function-date-id', type=str, default=DEFAULT_FITNESS_FUNCTION_DATE_ID)
DEFAULT_FITNESS_FEATURIZER_PATH = LATEST_FITNESS_FEATURIZER_PATH
parser.add_argument('--fitness-featurizer-path', type=str, default=DEFAULT_FITNESS_FEATURIZER_PATH)
parser.add_argument('--fitness-function-model-name', type=str, default=DEFAULT_SAVE_MODEL_NAME)
parser.add_argument('--no-flip-fitness-sign', action='store_true')

DEFAULT_POPULATION_SIZE = 100
parser.add_argument('--population-size', type=int, default=DEFAULT_POPULATION_SIZE)
DEFAULT_N_STEPS = 100
parser.add_argument('--n-steps', type=int, default=DEFAULT_N_STEPS)

# TODO: rewrite these arguments to the things this sampler actually needs
# DEFAULT_PLATEAU_PATIENCE_STEPS = 1000
# parser.add_argument('--plateau-patience-steps', type=int, default=DEFAULT_PLATEAU_PATIENCE_STEPS)
# DEFAULT_MAX_STEPS = 20000
# parser.add_argument('--max-steps', type=int, default=DEFAULT_MAX_STEPS)
# DEFAULT_N_SAMPLES_PER_STEP = 1
# parser.add_argument('--n-samples-per-step', type=int, default=DEFAULT_N_SAMPLES_PER_STEP)
# parser.add_argument('--non-greedy', action='store_true')
# DEFAULT_ACCEPTANCE_TEMPERATURE = 1.0
# parser.add_argument('--acceptance-temperature', type=float, default=DEFAULT_ACCEPTANCE_TEMPERATURE)

MICROBIAL_GA = 'microbial_ga'
WEIGHTED_BEAM_SEARCH = 'weighted_beam_search'
SAMPLER_TYPES = [MICROBIAL_GA, WEIGHTED_BEAM_SEARCH]
parser.add_argument('--sampler-type', type=str, required=True, choices=SAMPLER_TYPES)
parser.add_argument('--diversity-scorer-type', type=str, required=False, choices=DIVERSITY_SCORERS)

parser.add_argument('--microbial-ga-crossover-full-sections', action='store_true')
parser.add_argument('--microbial-ga-crossover-type', type=int, default=2)
DEFAULT_MICROBIAL_GA_MIN_N_CROSSOVERS = 1
parser.add_argument('--microbial-ga-n-min-loser-crossovers', type=int, default=DEFAULT_MICROBIAL_GA_MIN_N_CROSSOVERS)
DEFAULT_MICROBIAL_GA_MAX_N_CROSSOVERS = 5
parser.add_argument('--microbial-ga-n-max-loser-crossovers', type=int, default=DEFAULT_MICROBIAL_GA_MAX_N_CROSSOVERS)
DEFAULT_BEAM_SEARCH_K = 10
parser.add_argument('--beam-search-k', type=int, default=DEFAULT_BEAM_SEARCH_K)

DEFAULT_RELATIVE_PATH = '.'
parser.add_argument('--relative-path', type=str, default=DEFAULT_RELATIVE_PATH)
DEFAULT_NGRAM_MODEL_PATH = LATEST_AST_N_GRAM_MODEL_PATH
parser.add_argument('--ngram-model-path', type=str, default=DEFAULT_NGRAM_MODEL_PATH)
DEFUALT_RANDOM_SEED = 33
parser.add_argument('--random-seed', type=int, default=DEFUALT_RANDOM_SEED)
parser.add_argument('--initial-proposal-type', type=int, default=0)
parser.add_argument('--sample-parallel', action='store_true')
parser.add_argument('--parallel-n-workers', type=int, default=8)
parser.add_argument('--parallel-chunksize', type=int, default=1)
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--should-tqdm', action='store_true')
parser.add_argument('--within-step-tqdm', action='store_true')
parser.add_argument('--postprocess', action='store_true')
parser.add_argument('--sample-patience', type=int, default=100)

DEFAULT_OUTPUT_NAME = 'evo-sampler'
parser.add_argument('--output-name', type=str, default=DEFAULT_OUTPUT_NAME)
parser.add_argument('--output-folder', type=str, default='./samples')


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


def node_info_to_key(crossover_type: CrossoverType, node_info: ast_parser.ASTNodeInfo):
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


ASTType = typing.Union[tuple, tatsu.ast.AST]
T = typing.TypeVar('T')


def no_op_operator(games: typing.Union[ASTType, typing.List[ASTType]], rng=None):
    return games


def handle_multiple_inputs(operator):
    @wraps(operator)
    def wrapped_operator(self, games: typing.Union[ASTType, typing.List[ASTType]], rng: np.random.Generator, *args, **kwargs):
        if not isinstance(games, list):
            return operator(self, games, rng=rng, *args, **kwargs)

        if len(games) == 1:
            return operator(self, games[0], rng=rng, *args, **kwargs)

        else:
            operator_outputs = [operator(self, game, rng=rng, *args, **kwargs) for game in games]
            outputs = []
            for out in operator_outputs:
                if isinstance(out, list):
                    outputs.extend(out)
                else:
                    outputs.append(out)

            return outputs

    return wrapped_operator


class PopulationBasedSampler():
    context_fixers: typing.List[ASTContextFixer]
    counter: ASTRuleValueCounter
    diversity_scorer: typing.Optional[DiversityScorer]
    diversity_scorer_type: typing.Optional[str]
    feature_names: typing.List[str]
    fitness_featurizer: ASTFitnessFeaturizer
    fitness_featurizer_path: str
    fitness_function: typing.Callable[[ASTType], float]
    fitness_function_date_id: str
    fitness_function_model_name: str
    flip_fitness_sign: bool
    generation_index: int
    grammar: str
    grammar_parser: tatsu.grammars.Grammar  # type: ignore
    initial_sampler: typing.Callable[[], ASTType]
    n_workers: int
    postprocessor: ast_parser.ASTSamplePostprocessor
    population: typing.List[ASTType]
    population_size: int
    random_seed: int
    regrowth_samplers: typing.List[RegrowthSampler]
    rng: np.random.Generator
    sample_parallel: bool
    samplers: typing.List[ASTSampler]
    verbose: int


    '''
    This is a type of game sampler which uses an evolutionary strategy to climb a
    provided fitness function. It's a population-based alternative to the MCMC samper

    # TODO: store statistics about which locations are more likely to receive beneficial mutations?
    # TODO: keep track of 'lineages'
    '''

    def __init__(self,
                 args: argparse.Namespace,
                 population_size: int = DEFAULT_POPULATION_SIZE,
                 verbose: int = 0,
                 initial_proposal_type: InitialProposalSamplerType = InitialProposalSamplerType.MAP,
                 fitness_featurizer_path: str = DEFAULT_FITNESS_FEATURIZER_PATH,
                 fitness_function_date_id: str = DEFAULT_FITNESS_FUNCTION_DATE_ID,
                 fitness_function_model_name: str = DEFAULT_SAVE_MODEL_NAME,
                 flip_fitness_sign: bool = True,
                 relative_path: str = DEFAULT_RELATIVE_PATH,
                 ngram_model_path: str = DEFAULT_NGRAM_MODEL_PATH,
                 section_sampler_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 sample_patience: int = 100,
                 sample_parallel: bool = False,
                 n_workers: int = 1,
                 diversity_scorer_type: typing.Optional[str] = None,
                 ):

        self.population_size = population_size
        self.verbose = verbose
        self.sample_patience = sample_patience
        self.sample_parallel = sample_parallel
        self.n_workers = n_workers
        self.diversity_scorer_type = diversity_scorer_type

        self.grammar = open(args.grammar_file).read()
        self.grammar_parser = typing.cast(tatsu.grammars.Grammar, tatsu.compile(self.grammar))
        self.counter = parse_or_load_counter(args, self.grammar_parser)

        self.fitness_featurizer_path = fitness_featurizer_path
        self.fitness_featurizer = _load_pickle_gzip(fitness_featurizer_path)
        self.fitness_function_date_id = fitness_function_date_id
        self.fitness_function_model_name = fitness_function_model_name
        self.fitness_function, self.feature_names = load_model_and_feature_columns(fitness_function_date_id, name=fitness_function_model_name, relative_path=relative_path)  # type: ignore
        self.flip_fitness_sign = flip_fitness_sign

        if self.diversity_scorer_type is not None:
            self.diversity_scorer = create_diversity_scorer(self.diversity_scorer_type, featurizer=self.fitness_featurizer, feature_names=self.feature_names)

        # Used to generate the initial population of complete games
        self.samplers = [ASTSampler(self.grammar_parser, self.counter, seed=args.random_seed + i) for i in range(self.n_workers)]  # type: ignore
        self.random_seed = args.random_seed
        self.rng = self.samplers[0].rng

        self.initial_sampler = create_initial_proposal_sampler(
            initial_proposal_type, self.samplers[0], ngram_model_path, section_sampler_kwargs)  # type: ignore

        # Used as the mutation operator to modify existing games
        self.regrowth_samplers = [RegrowthSampler(sampler, seed=args.random_seed + i, rng=sampler.rng) for i, sampler in enumerate(self.samplers)]

        # Used to fix the AST context after crossover / mutation
        self.context_fixers = [ASTContextFixer(sampler, sampler.rng) for sampler in self.samplers]

        # Generate the initial population
        self.set_population([self._gen_init_sample(idx) for idx in trange(self.population_size, desc='Generating initial population')])

        self.postprocessor = ast_parser.ASTSamplePostprocessor()
        self.generation_index = 0
        self.mean_fitness_history = []
        self.std_fitness_history = []
        self.best_fitness_history = []

    def _proposal_to_features(self, proposal: ASTType) -> typing.Dict[str, typing.Any]:
        return typing.cast(dict, self.fitness_featurizer.parse(proposal, return_row=True))  # type: ignore

    def _features_to_tensor(self, features: typing.Dict[str, typing.Any]) -> torch.Tensor:
        return torch.tensor([features[name] for name in self.feature_names], dtype=torch.float32)  # type: ignore

    def _evaluate_fitness(self, features: torch.Tensor) -> float:
        if 'wrapper' in self.fitness_function.named_steps:  # type: ignore
            self.fitness_function.named_steps['wrapper'].eval()  # type: ignore
        score = self.fitness_function.transform(features).item()
        return -score if self.flip_fitness_sign else score

    def _score_proposal(self, proposal: ASTType, return_features: bool = False):
        proposal_features = self._proposal_to_features(proposal)
        proposal_tensor = self._features_to_tensor(proposal_features)
        proposal_fitness = self._evaluate_fitness(proposal_tensor)

        if return_features:
            return proposal_fitness, proposal_features

        return proposal_fitness

    def _process_index(self):
        identity = multiprocessing.current_process()._identity
        if identity is None or len(identity) == 0:
            return 0

        return (identity[0] - 1) % self.n_workers

    def _sampler(self):
        return self.samplers[self._process_index()]

    def _regrowth_sampler(self):
        return self.regrowth_samplers[self._process_index()]

    def _context_fixer(self):
        return self.context_fixers[self._process_index()]

    def _rename_game(self, game: ASTType, name: str) -> None:
        replace_child(game[1], ['game_name'], name)  # type: ignore

    def set_population(self, population: typing.List[typing.Any], fitness_values: typing.Optional[typing.List[float]] = None):
        '''
        Set the initial population of the sampler
        '''
        self.population = population
        self.population_size = len(population)
        if fitness_values is None:
            fitness_values = typing.cast(typing.List[float], [self._score_proposal(game, return_features=False) for game in self.population])

        self.fitness_values = fitness_values

        self.best_fitness = max(self.fitness_values)
        self.mean_fitness = np.mean(self.fitness_values)
        self.std_fitness = np.std(self.fitness_values)

    def _best_individual(self):
        return self.population[np.argmax(self.fitness_values)]

    def _print_game(self, game):
        print(ast_printer.ast_to_string(game, "\n"))

    def _choice(self, iterable: typing.Sequence[T], n: int = 1, rng: typing.Optional[np.random.Generator] = None) -> typing.Union[T, typing.List[T]]:
        '''
        Small hack to get around the rng invalid __array_struct__ error
        '''
        if rng is None:
            rng = self.rng

        if n == 1:
            idx = rng.integers(len(iterable))
            return iterable[idx]

        else:
            idxs = rng.choice(len(iterable), size=n, replace=False)
            return [iterable[idx] for idx in idxs]

    def _gen_init_sample(self, idx):
        '''
        Helper function for generating an initial sample (repeating until one is generated
        without errors)
        '''

        sample = None

        while sample is None:
            try:
                sample = typing.cast(tuple, self.initial_sampler.sample(global_context=dict(original_game_id=f'evo-{idx}')))
            except RecursionError:
                if self.verbose >= 2: print(f'Recursion error in sample {idx} -- skipping')
            except SamplingException:
                if self.verbose >= 2: print(f'Sampling exception in sample {idx} -- skipping')

        return sample

    @handle_multiple_inputs
    def _gen_regrowth_sample(self, game: ASTType, rng: np.random.Generator):
        '''
        Helper function for generating a new sample from an existing game (repeating until one is generated
        without errors)
        '''

        # Set the source AST of the regrowth sampler to the current game
        self._regrowth_sampler().set_source_ast(game)

        new_proposal = None
        sample_generated = False

        while not sample_generated:
            try:
                new_proposal = self._regrowth_sampler().sample(sample_index=0, update_game_id=False, rng=rng)

                if ast_printer.ast_to_string(new_proposal) == ast_printer.ast_to_string(game):  # type: ignore
                    if self.verbose >= 2: print('Regrowth generated identical games, repeating')
                else:
                    sample_generated = True

            except RecursionError:
                if self.verbose >= 2: print('Recursion error, skipping sample')

            except SamplingException:
                if self.verbose >= 2: print('Sampling exception, skipping sample')

        return new_proposal

    def _get_valid_insert_or_delete_nodes(self, game: ASTType, min_length: int = 1) -> typing.List[typing.Tuple[tatsu.ast.AST, typing.List[typing.Union[str, int]], str, typing.Dict[str, typing.Any], typing.Dict[str, typing.Any]]]:
        '''
        Returns a list of every node in the game which is a valid candidate for insertion or deletion
        (i.e. can have more than one child). Each entry in the list is of the form:
            (parent, selector, section, global_context, local_context)
        '''

        self._regrowth_sampler().set_source_ast(game)

        # Collect all nodes whose final selector is an integet (i.e. an index into a list) and whose parent
        # yields a list when its first selector is applied. Also make sure that the list has a minimum length
        valid_nodes = []
        for _, parent, selector, _, section, global_context, local_context in self._regrowth_sampler().parent_mapping.values():
            if isinstance(selector[-1], int) and isinstance(parent[selector[0]], list) and len(parent[selector[0]]) >= min_length:  # type: ignore
                valid_nodes.append((parent, selector[0], section, global_context, local_context))

        if len(valid_nodes) == 0:
            raise SamplingException('No valid nodes found for insertion or deletion')

        # Dedupe valid nodes based on their parent and selector
        valid_node_dict = {}
        for parent, selector, section, global_context, local_context in valid_nodes:
            key = (*self._regrowth_sampler()._ast_key(parent), selector)
            if key not in valid_node_dict:
                valid_node_dict[key] = (parent, selector, section, global_context, local_context)

        return list(valid_node_dict.values())

    @handle_multiple_inputs
    def _insert(self, game: ASTType, rng: np.random.Generator):
        '''
        Attempt to insert a new node into the provided game by identifying a node which can have multiple
        children and inserting a new node into it. The new node is selected using the initial sampler
        '''
        # Make a copy of the game
        new_game = copy.deepcopy(game)
        valid_nodes = self._get_valid_insert_or_delete_nodes(new_game)

        # Select a random node from the list of valid nodes
        parent, selector, section, global_context, local_context = self._choice(valid_nodes, rng=rng)

        parent_rule = parent.parseinfo.rule # type: ignore
        parent_rule_posterior_dict = self._sampler().rules[parent_rule][selector]
        assert "length_posterior" in parent_rule_posterior_dict, f"Rule {parent_rule} does not have a length posterior"

        # Sample a new rule from the parent rule posterior (parent_rule_posterior_dict['rule_posterior'])
        new_rule = posterior_dict_sample(self.rng, parent_rule_posterior_dict['rule_posterior'])

        sample_global_context = global_context.copy()  # type: ignore
        sample_global_context['rng'] = rng

        new_node = None
        while new_node is None:
            try:
                new_node = self._sampler().sample(new_rule, global_context=sample_global_context, local_context=local_context) # type: ignore

            except RecursionError:
                if self.verbose >= 2: print('Recursion error, skipping sample')

            except SamplingException:
                if self.verbose >= 2: print('Sampling exception, skipping sample')

        if isinstance(new_node, tuple):
            new_node = new_node[0]

        # Insert the new node into the parent at a random index
        parent[selector].insert(rng.integers(len(parent[selector]) + 1), new_node) # type: ignore

        # Do any necessary context-fixing
        self._context_fixer().fix_contexts(new_game, crossover_child=new_node)  # type: ignore

        return new_game

    @handle_multiple_inputs
    def _delete(self, game: ASTType, rng: np.random.Generator):
        '''
        Attempt to deleting a new node into the provided game by identifying a node which can have multiple
        children and deleting one of them
        '''
        # Make a copy of the game
        new_game = copy.deepcopy(game)

        valid_nodes = self._get_valid_insert_or_delete_nodes(new_game, min_length=2)

        # Select a random node from the list of valid nodes
        parent, selector, section, global_context, local_context = self._choice(valid_nodes, rng=rng)

        parent_rule = parent.parseinfo.rule # type: ignore
        parent_rule_posterior_dict = self._sampler().rules[parent_rule][selector]
        assert "length_posterior" in parent_rule_posterior_dict, f"Rule {parent_rule} does not have a length posterior"

        # Delete a random node from the parent
        delete_index = rng.integers(len(parent[selector]))  # type: ignore
        child_to_delete = parent[selector][delete_index]  # type: ignore

        del parent[selector][delete_index] # type: ignore

        # Do any necessary context-fixing
        self._context_fixer().fix_contexts(new_game, original_child=child_to_delete)  # type: ignore

        return new_game

    def _crossover(self, games: typing.Union[ASTType, typing.List[ASTType]], crossover_type: CrossoverType = CrossoverType.SAME_PARENT_RULE,
                   rng: typing.Optional[np.random.Generator] = None, crossover_first_game: bool = True, crossover_second_game: bool = True):
        '''
        Attempts to perform a crossover between the two given games. The crossover type determines
        how nodes in the game are categorized (i.e. by rule, by parent rule, etc.). The crossover
        is performed by finding the set of 'categories' that are present in both games, and then
        selecting a random category from which to sample the nodes that will be exchanged. If no
        categories are shared between the two games, then no crossover is performed
        '''
        if not crossover_first_game and not crossover_second_game:
            raise ValueError("At least one of crossover_first_game and crossover_second_game must be True")

        if rng is None:
            rng = self.rng

        game_2 = None
        if isinstance(games, list):
            game_1 = games[0]

            if len(games) > 1:
                game_2 = games[1]
        else:
            game_1 = games

        if game_2 is None:
            game_2 = typing.cast(ASTType, self._choice(self.population, rng=rng))

        if crossover_first_game:
            game_1 = copy.deepcopy(game_1)

        if crossover_second_game:
            game_2 = copy.deepcopy(game_2)

        # Create a map from crossover_type keys to lists of nodeinfos for each game
        self._regrowth_sampler().set_source_ast(game_1)
        game_1_crossover_map = defaultdict(list)
        for node_key in self._regrowth_sampler().node_keys:
            node_info = self._regrowth_sampler().parent_mapping[node_key]
            game_1_crossover_map[node_info_to_key(crossover_type, node_info)].append(node_info)

        self._regrowth_sampler().set_source_ast(game_2)
        game_2_crossover_map = defaultdict(list)
        for node_key in self._regrowth_sampler().node_keys:
            node_info = self._regrowth_sampler().parent_mapping[node_key]
            game_2_crossover_map[node_info_to_key(crossover_type, node_info)].append(node_info)

        # Find the set of crossover_type keys that are shared between the two games
        shared_crossover_keys = set(game_1_crossover_map.keys()).intersection(set(game_2_crossover_map.keys()))

        # If there are no shared crossover keys, then throw an exception
        if len(shared_crossover_keys) == 0:
            raise SamplingException("No crossover keys shared between the two games")

        # Select a random crossover key and a nodeinfo for each game with that key
        crossover_key = self._choice(list(shared_crossover_keys), rng=rng)
        game_1_selected_node_info = self._choice(game_1_crossover_map[crossover_key], rng=rng)
        game_2_selected_node_info = self._choice(game_2_crossover_map[crossover_key], rng=rng)

        # Create new copies of the nodes to be crossed over
        g1_node, g1_parent, g1_selector = game_1_selected_node_info[:3]
        g2_node, g2_parent, g2_selector = game_2_selected_node_info[:3]

        # Perform the crossover and fix the contexts of the new games
        if crossover_first_game:
            game_2_crossover_node = copy.deepcopy(g2_node)
            replace_child(g1_parent, g1_selector, game_2_crossover_node) # type: ignore
            self._context_fixer().fix_contexts(game_1, g1_node, game_2_crossover_node)  # type: ignore

        if crossover_second_game:
            game_1_crossover_node = copy.deepcopy(g1_node)
            replace_child(g2_parent, g2_selector, game_1_crossover_node) # type: ignore
            self._context_fixer().fix_contexts(game_2, g2_node, game_1_crossover_node)  # type: ignore

        return [game_1, game_2]

    def _find_index_for_section(self, existing_sections: typing.List[str], new_section: str) -> typing.Tuple[int, bool]:
        try:
            index = existing_sections.index(new_section)
            return index, True

        except ValueError:
            if new_section == ast_parser.SETUP:
                return 0, False

            # in this case, it's ast_parser.TERMINAL
            return len(existing_sections) - 1, False

    def _insert_section_to_game(self, game: ASTType, new_section: tuple, index: int, replace: bool):
        continue_index = index if not replace else index + 1
        return (*game[:index], new_section, *game[continue_index:])  # type: ignore

    def _crossover_full_sections(self, games: typing.Union[ASTType, typing.List[ASTType]], rng: typing.Optional[np.random.Generator] = None,
                                 crossover_first_game: bool = True, crossover_second_game: bool = True):

        if not crossover_first_game and not crossover_second_game:
            raise ValueError("At least one of crossover_first_game and crossover_second_game must be True")

        if rng is None:
            rng = self.rng

        game_2 = None
        if isinstance(games, list):
            game_1 = games[0]

            if len(games) > 1:
                game_2 = games[1]
        else:
            game_1 = games

        if game_2 is None:
            game_2 = typing.cast(ASTType, self._choice(self.population, rng=rng))

        if crossover_first_game:
            game_1 = copy.deepcopy(game_1)

        if crossover_second_game:
            game_2 = copy.deepcopy(game_2)

        game_1_sections = [t[0] for t in game_1[3:-1]]  # type: ignore
        game_2_sections = [t[0] for t in game_2[3:-1]]  # type: ignore

        if crossover_first_game:
            game_2_section_index = rng.integers(len(game_2_sections))
            game_2_section = game_2_sections[game_2_section_index]
            index, replace = self._find_index_for_section(game_1_sections, game_2_section)
            section_copy = copy.deepcopy(game_2[3 + game_2_section_index])
            self._insert_section_to_game(game_1, section_copy, index, replace)  # type: ignore

        if crossover_second_game:
            game_1_section_index = rng.integers(len(game_1_sections))
            game_1_section = game_1_sections[game_1_section_index]
            index, replace = self._find_index_for_section(game_2_sections, game_1_section)
            section_copy = copy.deepcopy(game_1[3 + game_1_section_index])
            self._insert_section_to_game(game_2, section_copy, index, replace)  # type: ignore

        return [game_1, game_2]

    def _get_operator(self, rng: typing.Optional[np.random.Generator] = None) -> typing.Callable[[typing.Union[ASTType, typing.List[ASTType]], np.random.Generator], typing.Union[ASTType, typing.List[ASTType]]]:
        '''
        Returns a function (operator) which takes in a list of games and returns a list of new games.
        As a default, always return a no_op operator
        '''

        return no_op_operator

    def _get_parent_iterator(self, n_parents_per_sample: int = 1, n_times_each_parent: int = 1):
        '''
        Returns an iterator which at each step yields one or more parents that will be modified
        by the operator. As a default, return an iterator which yields the entire population
        '''
        indices = np.concatenate([self.rng.permutation(self.population_size) for _ in range(n_times_each_parent)])

        if n_parents_per_sample == 1:
            for i in indices:
                yield self.population[i]

        else:
            for idxs in range(0, len(indices), n_parents_per_sample):
                yield [self.population[i] for i in indices[idxs:idxs + n_parents_per_sample]]

    def _select_new_population(self, candidates, candidate_scores):
        '''
        Returns the new population given the current population, the candidate games, and the
        scores for both the population and the candidate games. As a default, return the top P
        games from the union of the population and the candidates
        '''
        all_games = self.population + candidates
        all_scores = self.fitness_values + candidate_scores

        # TODO: do something with the diversity scorer?

        top_indices = np.argsort(all_scores)[-self.population_size:]
        self.set_population([all_games[i] for i in top_indices], [all_scores[i] for i in top_indices])

    def _sample_and_apply_operator(self, parent: typing.Union[ASTType, typing.List[ASTType]], generation_index: int, sample_index: int):
        '''
        Given a parent, a generation and sample index (to make sure that the RNG is seeded differently for each generation / individual),
        sample an operator and apply it to the parent. Returns the child or children and their fitness scores
        '''
        rng = np.random.default_rng(self.random_seed + (self.population_size * generation_index) + sample_index)

        for _ in range(self.sample_patience):
            try:
                operator = self._get_operator(rng)
                child_or_children = operator(parent, rng)
                if not isinstance(child_or_children, list):
                    child_or_children = [child_or_children]

                for i, child in enumerate(child_or_children):
                    self._rename_game(child, f'evo-{generation_index}-{sample_index}-{i}')

                children_fitness_scores = [self._score_proposal(child, return_features=False) for child in child_or_children]
                return child_or_children, children_fitness_scores

            except SamplingException:
                continue

        # TODO: should this raise an exception or just return the parent unmodified?
        raise SamplingException(f'Could not validly sample an operator and apply it to a child in {self.sample_patience} attempts')

    def evolutionary_step(self, pool: typing.Optional[mpp.Pool] = None, chunksize: int = 1,
                          should_tqdm: bool = False):
        # The core steps are:
        # 1. determine which "operator" is going to be used (an operator takes in one or more games and produces one or more new games)
        # 2. create a "parent_iteraor" which takes in the population and yields the parents that will be used by the operator
        # 3. for each parent(s) yielded, apply the operator to produce one or more new games and add them to a "candidates" list
        # 4. score the candidates
        # 5. pass the initial population and the candidates to the "selector" which will return the new population
        parent_iterator = self._get_parent_iterator()

        param_iterator = zip(parent_iterator, itertools.repeat(self.generation_index), itertools.count())

        candidates = []
        candidate_scores = []

        if pool is not None:
            children_iter = pool.istarmap(self._sample_and_apply_operator, param_iterator, chunksize=chunksize)  # type: ignore
        else:
            children_iter = itertools.starmap(self._sample_and_apply_operator, param_iterator)  # type: ignore

        if should_tqdm:
            children_iter = tqdm(children_iter)  # type: ignore

        for children, children_fitness_scores in children_iter:
            if isinstance(children, list):
                candidates.extend(children)
                candidate_scores.extend(children_fitness_scores)
            else:
                candidates.append(children)
                candidate_scores.append(children_fitness_scores)

        self._select_new_population(candidates, candidate_scores)

    def multiple_evolutionary_steps(self, num_steps: int, pool: typing.Optional[mpp.Pool] = None,
                                    chunksize: int = 1, should_tqdm: bool = False,
                                    inner_tqdm: bool = False, postprocess: typing.Optional[bool] = False):
        step_iter = range(num_steps)
        if should_tqdm:
            pbar = tqdm(total=num_steps, desc="Evolutionary steps") # type: ignore

        for _ in step_iter:  # type: ignore
            self.evolutionary_step(pool, chunksize, should_tqdm=inner_tqdm)

            if should_tqdm:
                pbar.update(1)  # type: ignore
                pbar.set_postfix({"Mean": f"{self.mean_fitness:.3f}", "Std": f"{self.std_fitness:.3f}", "Best": f"{self.best_fitness:.3f}"})  # type: ignore

            elif self.verbose:
                print(f"Average fitness: {self.mean_fitness:.3f} +/- {self.std_fitness:.3f}, Best fitness: {self.best_fitness:.3f}")

            self.mean_fitness_history.append(self.mean_fitness)
            self.std_fitness_history.append(self.std_fitness)
            self.best_fitness_history.append(self.best_fitness)

        if postprocess:
            self.population = [self.postprocessor(g) for g in self.population]  # type: ignore

    def multiple_evolutionary_steps_parallel(self, num_steps: int, should_tqdm: bool = False,
                                             inner_tqdm: bool = False, postprocess: typing.Optional[bool] = False, n_workers: int = 8, chunksize: int = 1):

        logging.debug(f'Launching multiprocessing pool with {n_workers} workers...')
        with mpp.Pool(n_workers) as pool:
            self.multiple_evolutionary_steps(num_steps, pool, chunksize=chunksize,
                                             should_tqdm=should_tqdm, inner_tqdm=inner_tqdm,
                                             postprocess=postprocess)

    def visualize_sample(self, sample_index: int, top_k: int = 20, display_overall_features: bool = True, display_game: bool = True, min_display_threshold: float = 0.0005, postprocess_sample: bool = False):
        sample = self.population[sample_index]
        if postprocess_sample:
            sample = self.postprocessor(sample)

        sample_features = self._proposal_to_features(sample)  # type: ignore
        sample_features_tensor = self._features_to_tensor(sample_features)

        evaluate_single_game_energy_contributions(
            self.fitness_function, sample_features_tensor, ast_printer.ast_to_string(sample, '\n'), self.feature_names,   # type: ignore
            top_k=top_k, display_overall_features=display_overall_features,
            display_game=display_game, min_display_threshold=min_display_threshold,
            )

    def visualize_top_sample(self, top_index: int, top_k: int = 20, display_overall_features: bool = True, display_game: bool = True, min_display_threshold: float = 0.0005, postprocess_sample: bool = False):
        sample_index = np.argsort(self.fitness_values)[-top_index]
        self.visualize_sample(sample_index, top_k, display_overall_features, display_game, min_display_threshold, postprocess_sample)


class BeamSearchSampler(PopulationBasedSampler):
    '''
    Implements 'beam search' by, at each generation, expanding every game in the population out k times
    and then restricting the population to the top P games
    '''
    def __init__(self, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def _get_operator(self, rng):
        return self._gen_regrowth_sample

    def _get_parent_iterator(self, n_parents_per_sample: int = 1, n_times_each_parent: int = 1):
        return super()._get_parent_iterator(1, self.k)


class WeightedBeamSearchSampler(PopulationBasedSampler):
    '''
    Implements a weighted form of beam search where the number of samples generated for each game
    is dependent on its fitness rank in the population. The most fit game receives (in expectation)
    2k samples, the median game k samples, and the least fit game 0 samples. This is achieved by
    running 2 * P "tournaments" where two individuals are randomly sampled from the population and
    the individual with higher fitness produces a child
    '''
    def __init__(self, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def _get_operator(self, rng):

        def weighted_beam_search_sample(games, rng):
            if not isinstance(games, list) or len(games) != 2:
                raise ValueError(f'Expected games to be a list of length 2, got {games}')

            p1_idx = self.population.index(games[0])
            p2_idx = self.population.index(games[1])

            if self.fitness_values[p1_idx] >= self.fitness_values[p2_idx]:
                return self._gen_regrowth_sample(games[0], rng)
            else:
                return self._gen_regrowth_sample(games[1], rng)

        return weighted_beam_search_sample

    def _get_parent_iterator(self, n_parents_per_sample: int = 1, n_times_each_parent: int = 1):
        return super()._get_parent_iterator(2, 2 * self.k)

class InsertDeleteSampler(PopulationBasedSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_operator(self, rng):
        def insert_delete_operator(game, rng):
            try:
                new_game = self._insert(game, rng)
                new_game = self._delete(new_game, rng)
                return new_game

            except SamplingException as e:
                if self.verbose >=2: print(f"Failed to insert/delete: {e}")
                return game

        return insert_delete_operator


MONITOR_FEATURES = ['all_variables_defined', 'all_variables_used', 'all_preferences_used']


class CrossoverOnlySampler(PopulationBasedSampler):
    def __init__(self, args: argparse.Namespace,
                 k: int = 1, max_attempts: int = 100,
                 crossover_type: CrossoverType = CrossoverType.SAME_PARENT_RULE,
                 monitor_feature_keys: typing.List[str] = MONITOR_FEATURES,
                 *addl_args, **kwargs):

        super().__init__(args, *addl_args, **kwargs)
        self.k = k
        self.max_attempts = max_attempts
        self.crossover_type = crossover_type
        self.monitor_feature_keys = monitor_feature_keys

    def _extract_monitor_features(self, game):
        if self.fitness_featurizer is not None:
            return {k: v for k, v in self.fitness_featurizer.parse(game, return_row=True).items() if k in self.monitor_feature_keys}  # type: ignore
        else:
            return {}

    def _get_operator(self, rng):
        def crossover_two_random_games(games, rng):
            for _ in range(self.max_attempts):
                try:
                    if len(games) > 2:
                        games = self._choice(games, 2)

                    before_feature_values = [self._extract_monitor_features(g) for g in games]
                    post_crossover_games = self._crossover(games, self.crossover_type, rng=rng)
                    after_feature_values = [self._extract_monitor_features(g) for g in post_crossover_games]

                    for i, (before_features, after_features) in enumerate(zip(before_feature_values, after_feature_values)):
                        for k, v in before_features.items():
                            if v != after_features[k]:
                                print(f'In game #{i + 1}, feature {k} changed from {v} to {after_features[k]}')

                    return post_crossover_games
                except SamplingException as e:
                    if self.verbose:
                        print(f'Failed to crossover: {e}')

            raise SamplingException('Failed to crossover after max attempts')

        return crossover_two_random_games

    def _get_parent_iterator(self, n_parents_per_sample: int = 1, n_times_each_parent: int = 1):
        return super()._get_parent_iterator(2, self.k)


class MicrobialGASampler(PopulationBasedSampler):
    crossover_full_sections: bool
    crossover_type: CrossoverType
    min_n_loser_crossovers: int
    max_n_loser_crossovers: int

    def __init__(self,
                 args: argparse.Namespace,
                 crossover_full_sections: bool = False,
                 crossover_type: CrossoverType = CrossoverType.SAME_PARENT_RULE,
                 min_n_loser_crossovers: int = DEFAULT_MICROBIAL_GA_MIN_N_CROSSOVERS,
                 max_n_loser_crossovers: int = DEFAULT_MICROBIAL_GA_MAX_N_CROSSOVERS,
                 *addl_args, **kwargs):

        super().__init__(args, *addl_args, **kwargs)
        if max_n_loser_crossovers <= min_n_loser_crossovers:
            raise ValueError(f'max_n_loser_crossovers must be greater than min_n_loser_crossovers, got {max_n_loser_crossovers} <= {min_n_loser_crossovers}')

        self.crossover_full_sections = crossover_full_sections
        self.crossover_type = crossover_type
        self.min_n_loser_crossovers = min_n_loser_crossovers
        self.max_n_loser_crossovers = max_n_loser_crossovers


    def _get_operator(self, rng):
        '''
        Implements the classic "Microbial GA" operator, which operates as follows:
            1. fitness of the two individuals is compared, "winner" and "loser" assigned
            2. winner "infects" loser by randomly replacing a subtree in loser with a subtree from winner (crossover)
            3. loser is mutated (randomly selected from regrowth, insertion, and deletion)
            4. winner and loser re-enter population
        '''
        def microbial_ga_operator(games, rng):
            if len(games) > 2:
                games = self._choice(games, 2, rng=rng)

            p1_idx = self.population.index(games[0])
            p2_idx = self.population.index(games[1])

            winner, loser = (games[0], games[1]) if self.fitness_values[p1_idx] >= self.fitness_values[p2_idx] else (games[1], games[0])

            if self.crossover_full_sections:
                _, loser = self._crossover_full_sections([winner, loser], rng=rng, crossover_first_game=False)

            else:
                n_crossovers = rng.integers(self.min_n_loser_crossovers, self.max_n_loser_crossovers + 1)
                for _ in range(n_crossovers):
                    _, loser = self._crossover([winner, loser], self.crossover_type, rng=rng, crossover_first_game=False)

            # Apply a randomly selected mutation operator to the loser
            mutation = self._choice([self._gen_regrowth_sample, self._insert, self._delete], rng=rng)
            mutated_loser = mutation(loser, rng)  # type: ignore

            return mutated_loser

        return microbial_ga_operator

    def _get_parent_iterator(self, n_parents_per_sample: int = 1, n_times_each_parent: int = 1):
        return super()._get_parent_iterator(n_parents_per_sample=2)


def main(args):
    if args.sampler_type == MICROBIAL_GA:
        evosampler = MicrobialGASampler(
            crossover_full_sections=args.microbial_ga_crossover_full_sections, crossover_type=CrossoverType(args.microbial_ga_crossover_type),
            min_n_loser_crossovers=args.microbial_ga_n_min_loser_crossovers, max_n_loser_crossovers=args.microbial_ga_n_max_loser_crossovers,
            args=args,
            population_size=args.population_size,
            verbose=args.verbose,
            initial_proposal_type=InitialProposalSamplerType(args.initial_proposal_type),
            fitness_featurizer_path=args.fitness_featurizer_path,
            fitness_function_date_id=args.fitness_function_date_id,
            fitness_function_model_name=args.fitness_function_model_name,
            flip_fitness_sign=not args.no_flip_fitness_sign,
            ngram_model_path=args.ngram_model_path,
            sample_parallel=args.sample_parallel,
            n_workers=args.parallel_n_workers,
            diversity_scorer_type=args.diversity_scorer_type,
        )

    elif args.sampler_type == WEIGHTED_BEAM_SEARCH:
        evosampler = WeightedBeamSearchSampler(k=args.beam_search_k,
            args=args,
            population_size=args.population_size,
            verbose=args.verbose,
            initial_proposal_type=InitialProposalSamplerType(args.initial_proposal_type),
            fitness_featurizer_path=args.fitness_featurizer_path,
            fitness_function_date_id=args.fitness_function_date_id,
            fitness_function_model_name=args.fitness_function_model_name,
            flip_fitness_sign=not args.no_flip_fitness_sign,
            ngram_model_path=args.ngram_model_path,
            sample_parallel=args.sample_parallel,
            n_workers=args.parallel_n_workers,
            diversity_scorer_type=args.diversity_scorer_type,
        )

    else:
        raise ValueError(f'Unknown sampler type {args.sampler_type}')

    # evosampler = CrossoverOnlySampler(
    # evosampler = WeightedBeamSearchSampler(k=args.beam_search_k,
    # evosampler = WeightedBeamSearchSampler(k=args.beam_search_k,
    # evosampler = MicrobialGASampler(n_loser_crossovers=args.microbial_ga_n_loser_crossovers,
    #     args=args, fitness_function=EnergyFunctionFitnessWrapper(fitness_featurizer, trained_fitness_function, feature_names, flip_sign=True),  # type: ignore
    #     population_size=args.population_size,
    #     verbose=args.verbose,
    #     initial_proposal_type=InitialProposalSamplerType(args.initial_proposal_type),
    #     ngram_model_path=args.ngram_model_path,
    #     sample_parallel=args.sample_parallel,
    #     n_workers=args.parallel_n_workers,
    # )

    # game_asts = list(cached_load_and_parse_games_from_file('./dsl/interactive-beta.pddl', evosampler.grammar_parser, False, relative_path='.'))
    # evosampler.set_population(game_asts[:4])

    try:
        if args.sample_parallel:
            evosampler.multiple_evolutionary_steps_parallel(
                num_steps=args.n_steps, should_tqdm=args.should_tqdm, inner_tqdm=args.within_step_tqdm,
                postprocess=args.postprocess, n_workers=args.parallel_n_workers, chunksize=args.parallel_chunksize
                )

        else:
            evosampler.multiple_evolutionary_steps(
                num_steps=args.n_steps, should_tqdm=args.should_tqdm,
                inner_tqdm=args.within_step_tqdm, postprocess=args.postprocess,
                )

        print('Best individual:')
        evosampler._print_game(evosampler._best_individual())

    finally:
        save_data(evosampler, args.output_folder, args.output_name, args.relative_path)



if __name__ == '__main__':
    args = parser.parse_args()

    args.grammar_file = os.path.join(args.relative_path, args.grammar_file)
    args.counter_output_path = os.path.join(args.relative_path, args.counter_output_path)
    args.fitness_featurizer_path = os.path.join(args.relative_path, args.fitness_featurizer_path)
    args.ngram_model_path = os.path.join(args.relative_path, args.ngram_model_path)

    args_str = '\n'.join([f'{" " * 26}{k}: {v}' for k, v in vars(args).items()])
    logging.debug(f'Shell arguments:\n{args_str}')

    main(args)
