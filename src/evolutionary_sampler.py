import argparse
import gzip
import os
import pickle
import typing
import sys

import numpy as np
import tatsu
import tatsu.ast
import tatsu.grammars
import torch
from tqdm import tqdm, notebook

# from ast_parser import SETUP, PREFERENCES, TERMINAL, SCORING
import ast_printer
import ast_parser
from ast_context_fixer import ASTContextFixer
from ast_counter_sampler import *
from ast_counter_sampler import parse_or_load_counter, ASTSampler, RegrowthSampler, SamplingException, MCMC_REGRWOTH
from ast_mcmc_regrowth import _load_pickle_gzip, InitialProposalSamplerType, create_initial_proposal_sampler, mpp
from ast_utils import *
from fitness_energy_utils import load_model_and_feature_columns, save_data, DEFAULT_SAVE_MODEL_NAME
from fitness_features import *
from fitness_ngram_models import *
from latest_model_paths import LATEST_AST_N_GRAM_MODEL_PATH, LATEST_FITNESS_FEATURIZER_PATH, LATEST_FITNESS_FUNCTION_DATE_ID

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import src



parser = argparse.ArgumentParser(description='Evolutionary Sampler')
parser.add_argument('--grammar-file', type=str, default=DEFAULT_GRAMMAR_FILE)
parser.add_argument('--parse-counter', action='store_true')
parser.add_argument('--counter-output-path', type=str, default=DEFAULT_COUNTER_OUTPUT_PATH)

DEFAULT_FITNESS_FUNCTION_DATE_ID = LATEST_FITNESS_FUNCTION_DATE_ID
parser.add_argument('--fitness-function-date-id', type=str, default=DEFAULT_FITNESS_FUNCTION_DATE_ID)
DEFAULT_FITNESS_FEATURIZER_PATH = LATEST_FITNESS_FEATURIZER_PATH
parser.add_argument('--fitness-featurizer-path', type=str, default=DEFAULT_FITNESS_FEATURIZER_PATH)
parser.add_argument('--fitness-function-model-name', type=str, default=DEFAULT_SAVE_MODEL_NAME)

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

DEFAULT_RELATIVE_PATH = '.'
parser.add_argument('--relative-path', type=str, default=DEFAULT_RELATIVE_PATH)
DEFAULT_NGRAM_MODEL_PATH = LATEST_AST_N_GRAM_MODEL_PATH
parser.add_argument('--ngram-model-path', type=str, default=DEFAULT_NGRAM_MODEL_PATH)
DEFUALT_RANDOM_SEED = 33
parser.add_argument('--random-seed', type=int, default=DEFUALT_RANDOM_SEED)

parser.add_argument('--initial-proposal-type', type=int, default=0)
parser.add_argument('--crossover-type', type=int, default=2)

# TODO: implement parallel sampling support here
parser.add_argument('--sample-parallel', action='store_true')
parser.add_argument('--parallel-n-workers', type=int, default=8)
parser.add_argument('--parallel-chunksize', type=int, default=1)
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--should-tqdm', action='store_true')
parser.add_argument('--within-step-tqdm', action='store_true')
parser.add_argument('--postprocess', action='store_true')

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


def no_op_operator(games: ASTType, rng=None):
    return games



class PopulationBasedSampler():
    context_fixer: ASTContextFixer
    counter: ASTRuleValueCounter
    fitness_function: typing.Callable[[ASTType], float]
    generation_index: int
    grammar: str
    grammar_parser: tatsu.grammars.Grammar  # type: ignore
    initial_sampler: typing.Callable[[], ASTType]
    postprocessor: ast_parser.ASTSamplePostprocessor
    population: typing.List[ASTType]
    population_size: int
    random_seed: int
    regrowth_sampler: RegrowthSampler
    rng: np.random.Generator
    sampler: ASTSampler
    verbose: int


    '''
    This is a type of game sampler which uses an evolutionary strategy to climb a
    provided fitness function. It's a population-based alternative to the MCMC samper
    '''

    def __init__(self,
                 args: argparse.Namespace,
                 fitness_function: typing.Callable[[typing.Any], float],
                 population_size: int = DEFAULT_POPULATION_SIZE,
                 verbose: int = 0,
                 initial_proposal_type: InitialProposalSamplerType = InitialProposalSamplerType.MAP,
                 ngram_model_path: str = DEFAULT_NGRAM_MODEL_PATH,
                 section_sampler_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,

                 ):

        self.fitness_function = fitness_function
        self.population_size = population_size
        self.verbose = verbose

        self.grammar = open(args.grammar_file).read()
        self.grammar_parser = typing.cast(tatsu.grammars.Grammar, tatsu.compile(self.grammar))
        self.counter = parse_or_load_counter(args, self.grammar_parser)

        # Used to generate the initial population of complete games
        self.sampler = ASTSampler(self.grammar_parser, self.counter, seed=args.random_seed)
        self.random_seed = args.random_seed
        self.rng = self.sampler.rng

        self.initial_sampler = create_initial_proposal_sampler(
            initial_proposal_type, self.sampler, ngram_model_path, section_sampler_kwargs)  # type: ignore

        # Used as the mutation operator to modify existing games
        self.regrowth_sampler = RegrowthSampler(self.sampler, seed=args.random_seed, rng=self.rng)

        # Used to fix the AST context after crossover / mutation
        self.context_fixer = ASTContextFixer(self.sampler, self.rng)

        # Generate the initial population
        self.set_population([self._gen_init_sample(idx) for idx in range(self.population_size)])

        self.postprocessor = ast_parser.ASTSamplePostprocessor()
        self.generation_index = 0

    def set_population(self, population: typing.List[typing.Any], fitness_values: typing.Optional[typing.List[float]] = None):
        '''
        Set the initial population of the sampler
        '''
        self.population = population
        self.population_size = len(population)
        if fitness_values is None:
            self.fitness_values = [self.fitness_function(game) for game in self.population]

    def _best_fitness(self):
        return max(self.fitness_values)

    def _avg_fitness(self):
        return np.mean(self.fitness_values)

    def _best_individual(self):
        return self.population[np.argmax(self.fitness_values)]

    def _print_game(self, game):
        print(ast_printer.ast_to_string(game, "\n"))

    def _choice(self, iterable, n=1):
        '''
        Small hack to get around the rng invalid __array_struct__ error
        '''
        if n == 1:
            idx = self.rng.integers(len(iterable))
            return iterable[idx]

        else:
            idxs = self.rng.choice(len(iterable), size=n, replace=False)
            return [iterable[idx] for idx in idxs]

    def _gen_init_sample(self, idx):
        '''
        Helper function for generating an initial sample (repeating until one is generated
        without errors)
        '''

        sample = None

        while sample is None:
            try:
                sample = typing.cast(tuple, self.sampler.sample(global_context=dict(original_game_id=f'mcmc-{idx}')))
            except RecursionError:
                if self.verbose >= 2: print(f'Recursion error in sample {idx} -- skipping')
            except SamplingException:
                if self.verbose >= 2: print(f'Sampling exception in sample {idx} -- skipping')

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
        self.set_population([samples[i] for i in top_indices], [scores[i] for i in top_indices])

    def insert_delete_step(self, insert_prob=0.5, delete_prob=0.5):
        '''
        Perform an evolutionary step by mutating each member of the population. Each mutation consists of
        an independent probability of:
        - inserting a new node inside of an existing node with multiple children (e.g. an [and] node)
        - deleting an existing node if it is one of multiple children of its parent
        As with the beam step, the top n samples are selected to form the next generation
        '''
        samples = self.population.copy()

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

            # Dedupe valid nodes based on their parent and selector
            valid_node_dict = {}
            for parent, selector, section, global_context, local_context in valid_nodes:
                key = (*self.regrowth_sampler._ast_key(parent), selector)
                if key not in valid_node_dict:
                    valid_node_dict[key] = (parent, selector, section, global_context, local_context)

            for parent, selector, section, global_context, local_context in valid_node_dict.values():

                parent_rule = parent.parseinfo.rule # type: ignore
                parent_rule_posterior_dict = self.sampler.rules[parent_rule][selector]
                assert "length_posterior" in parent_rule_posterior_dict, f"Rule {parent_rule} does not have a length posterior"

                # Determine whether we're sampling a rule or a token (for this case, it'll always be one or the other 100% of the time)
                if parent_rule_posterior_dict['type_posterior']['rule'] == 1:

                    # Check whether we're doing an insertion
                    if self.rng.random() < insert_prob:

                        # Sample a new rule from the parent rule posterior (parent_rule_posterior_dict['rule_posterior'])
                        new_rule = posterior_dict_sample(self.rng, parent_rule_posterior_dict['rule_posterior'])

                        new_node = None
                        while new_node is None:
                            try:
                                new_node = self.sampler.sample(new_rule, global_context=global_context, local_context=local_context) # type: ignore

                            except RecursionError:
                                if self.verbose >= 2: print('Recursion error, skipping sample')

                            except SamplingException:
                                if self.verbose >= 2: print('Sampling exception, skipping sample')

                        if isinstance(new_node, tuple):
                            new_node = new_node[0]

                        # Make a copy of the game
                        new_game = copy.deepcopy(game)
                        new_parent = self.regrowth_sampler.searcher(new_game, parseinfo=parent.parseinfo)  # type: ignore

                        # Insert the new node into the parent at a random index
                        new_parent[selector].insert(self.rng.integers(len(new_parent[selector])+1), new_node) # type: ignore

                        # Do any necessary context-fixing
                        self.context_fixer.fix_contexts(new_game, crossover_child=new_node)  # type: ignore

                        samples.append(new_game)

                    # Check whether we're doing a deletion
                    if self.rng.random() < delete_prob:
                        # Make a copy of the game
                        new_game = copy.deepcopy(game)
                        new_parent = typing.cast(tatsu.ast.AST, self.regrowth_sampler.searcher(new_game, parseinfo=parent.parseinfo))  # type: ignore

                        # Delete a random node from the parent
                        delete_index = self.rng.integers(len(new_parent[selector]))  # type: ignore
                        child_to_delete = new_parent[selector][delete_index]  # type: ignore

                        del new_parent[selector][delete_index] # type: ignore

                        # Do any necessary context-fixing
                        self.context_fixer.fix_contexts(new_game, original_child=child_to_delete)  # type: ignore

                        samples.append(new_game)

                elif parent_rule_posterior_dict['type_posterior']['token'] == 1:
                    raise Exception("Encountered unexpected rule: parent of multiple children of type token")

                else:
                    raise Exception("Invalid type posterior")

        # Score the new samples
        scores = [self.fitness_function(sample) for sample in samples]

        # Select the top n samples
        top_indices = np.argsort(scores)[-self.population_size:]
        self.population = [samples[i] for i in top_indices]
        self.fitness_values = [scores[i] for i in top_indices]

    def crossover_step(self, crossover_type):
        '''
        Performs crossover between members of the population. Currently, we'll generate crossover
        pairs for every valid pair of games in the population and then select the top n samples.

        But we should probably make a more efficient selection scheme (i.e. fitness-proportional or
        rank-based)
        '''

        samples = self.population.copy()
        for i in range(len(self.population)):
            for j in range(i+1, len(self.population)):
                try:
                    new_game_1, new_game_2 = self._crossover(self.population[i], self.population[j], crossover_type)
                    samples.append(new_game_1)
                    samples.append(new_game_2)

                except SamplingException:
                    if self.verbose >= 2: print('Sampling exception, skipping crossover')

        # Score the new samples
        scores = [self.fitness_function(sample) for sample in samples]

        # Select the top n samples
        top_indices = np.argsort(scores)[-self.population_size:]
        self.population = [samples[i] for i in top_indices]
        self.fitness_values = [scores[i] for i in top_indices]

    def _crossover(self, game_1, game_2, crossover_type):
        '''
        Attempts to perform a crossover between the two given games. The crossover type determines
        how nodes in the game are categorized (i.e. by rule, by parent rule, etc.). The crossover
        is performed by finding the set of 'categories' that are present in both games, and then
        selecting a random category from which to sample the nodes that will be exchanged. If no
        categories are shared between the two games, then no crossover is performed

        # TODO: we should decide who handles `SamplingException`s thrown here
        '''

        # Create a map from crossover_type keys to lists of nodeinfos for each game
        self.regrowth_sampler.set_source_ast(game_1)
        game_1_crossover_map = defaultdict(list)
        for node_key in self.regrowth_sampler.node_keys:
            node_info = self.regrowth_sampler.parent_mapping[node_key]
            game_1_crossover_map[node_info_to_key(crossover_type, node_info)].append(node_info)

        self.regrowth_sampler.set_source_ast(game_2)
        game_2_crossover_map = defaultdict(list)
        for node_key in self.regrowth_sampler.node_keys:
            node_info = self.regrowth_sampler.parent_mapping[node_key]
            game_2_crossover_map[node_info_to_key(crossover_type, node_info)].append(node_info)

        # Find the set of crossover_type keys that are shared between the two games
        shared_crossover_keys = set(game_1_crossover_map.keys()).intersection(set(game_2_crossover_map.keys()))

        # If there are no shared crossover keys, then throw an exception
        if len(shared_crossover_keys) == 0:
            raise SamplingException("No crossover keys shared between the two games")

        # Select a random crossover key and a nodeinfo for each game with that key
        crossover_key = self.rng.choice(list(shared_crossover_keys))
        game_1_selected_node_info = self._choice(game_1_crossover_map[crossover_key])
        game_2_selected_node_info = self._choice(game_2_crossover_map[crossover_key])

        # Apply the context fixer to both nodes
        g1_node, g1_parent, g1_selector, _, _, _, _ = game_1_selected_node_info
        g2_node, g2_parent, g2_selector, _, _, _, _ = game_2_selected_node_info

        game_1_crossover_node = copy.deepcopy(g1_node)
        game_2_crossover_node = copy.deepcopy(g2_node)

        # Perform the crossover
        game_1_copy = copy.deepcopy(game_1)
        game_1_parent = self.regrowth_sampler.searcher(game_1_copy, parseinfo=g1_parent.parseinfo) # type: ignore
        replace_child(game_1_parent, g1_selector, game_2_crossover_node) # type: ignore

        game_2_copy = copy.deepcopy(game_2)
        game_2_parent = self.regrowth_sampler.searcher(game_2_copy, parseinfo=g2_parent.parseinfo) # type: ignore
        replace_child(game_2_parent, g2_selector, game_1_crossover_node) # type: ignore

        # Fix the contexts of the new games
        self.context_fixer.fix_contexts(game_1_copy, g1_node, game_2_crossover_node)
        self.context_fixer.fix_contexts(game_2_copy, g2_node, game_1_crossover_node)

        return [game_1_copy, game_2_copy]

    # TODO: add general step which can be used to combine the above steps
    # TODO: break evolutionary step into selection / recombination / mutation steps?
    # -> can use microbial GA tournament structure?
    # TODO: add 'fixer' to make sure variables are used / defined
    # TODO: parallelize
    # TODO: store statistics about which locations are more likely to receive beneficial mutations?
    # TODO: keep track of 'lineages'

    def _get_operator(self, rng: typing.Optional[np.random.Generator] = None) -> typing.Callable[[ASTType, np.random.Generator], typing.Union[ASTType, typing.List[ASTType]]]:
        '''
        Returns a function (operator) which takes in a list of games and returns a list of new games.
        As a default, always return a no_op operator
        '''

        return no_op_operator

    def _get_parent_iterator(self, n_parents_per_sample: int):
        '''
        Returns an iterator which at each step yields one or more parents that will be modified
        by the operator. As a default, return an iterator which yields the entire population
        '''
        if n_parents_per_sample == 1:
            for p in self.population:
                yield p

        else:
            for _ in range(self.population_size):
                yield self._choice(self.population, n_parents_per_sample)

    def _select_new_population(self, candidates, candidate_scores):
        '''
        Returns the new population given the current population, the candidate games, and the
        scores for both the population and the candidate games. As a default, return the top P
        games from the union of the population and the candidates
        '''
        all_games = self.population + candidates
        all_scores = self.fitness_values + candidate_scores

        top_indices = np.argsort(all_scores)[-self.population_size:]
        self.population = [all_games[i] for i in top_indices]
        self.fitness_values = [all_scores[i] for i in top_indices]

    def _sample_and_apply_operator(self, parent: typing.Union[tuple, tatsu.ast.AST], generation_index: int, sample_index: int):
        rng = np.random.default_rng(self.random_seed + (self.population_size * generation_index) + sample_index)
        operator = self._get_operator(rng)
        child_or_children = operator(parent, rng)
        if not isinstance(child_or_children, list):
            child_or_children = [child_or_children]

        children_fitness_scores = [self.fitness_function(child) for child in child_or_children]
        return child_or_children, children_fitness_scores

    def evolutionary_step(self, pool: typing.Optional[mpp.Pool] = None, chunksize: int = 1,
                          should_tqdm: bool = False):
        # The core steps are:
        # 1. determine which "operator" is going to be used (an operator takes in one or more games and produces one or more new games)
        # 2. create a "parent_iteraor" which takes in the population and yields the parents that will be used by the operator
        # 3. for each parent(s) yielded, apply the operator to produce one or more new games and add them to a "candidates" list
        # 4. score the candidates
        # 5. pass the initial population and the candidates to the "selector" which will return the new population

        # TODO: move the operator sampling step to the middle, and think about what that means for crossover (which requires two parents)
        parent_iterator = self._get_parent_iterator(1)

        param_iterator = zip(parent_iterator, itertools.repeat(self.generation_index), itertools.count())

        candidates = []
        candidate_scores = []

        if pool is not None:
            children_iter = pool.istarmap(self._sample_and_apply_operator, param_iterator, chunksize=chunksize)  # type: ignore
        else:
            children_iter = itertools.starmap(self._sample_and_apply_operator, param_iterator)  # type: ignore

        if should_tqdm:
            children_iter = tqdm(children_iter)  # type: ignore

        # TODO: make sure the operators all receive their rng as a kwarg
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
            step_iter = tqdm(step_iter, desc='Evolutionary steps')  # type: ignore

        for _ in step_iter:
            self.evolutionary_step(pool, chunksize, should_tqdm=inner_tqdm)
            if self.verbose:
                print(f"Average fitness: {self._avg_fitness():.3f}, Best fitness: {self._best_fitness():.3f}")

        if postprocess:
            self.population = [self.postprocessor(g) for g in self.population]  # type: ignore

    def multiple_evolutionary_steps_parallel(self, num_steps: int, should_tqdm: bool = False,
                                             inner_tqdm: bool = False, postprocess: typing.Optional[bool] = False, n_workers: int = 8, chunksize: int = 1):

        logging.debug(f'Launching multiprocessing pool with {n_workers} workers...')
        with mpp.Pool(n_workers) as pool:
            self.multiple_evolutionary_steps(num_steps, pool, chunksize=chunksize,
                                             should_tqdm=should_tqdm, inner_tqdm=inner_tqdm,
                                             postprocess=postprocess)


class BeamSearchSampler(PopulationBasedSampler):
    '''
    Implements 'beam search' by, at each generation, expanding every game in the population out k times
    and then restricting the population to the top P games
    '''
    def __init__(self, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def _get_operator(self):
        return self._gen_regrowth_sample

    def _get_parent_iterator(self, population):
        return iter(population * self.k)


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

    def _get_operator(self):

        def weighted_beam_search_sample(games):
            p1_idx = self.population.index(games[0])
            p2_idx = self.population.index(games[1])

            if self.fitness_values[p1_idx] >= self.fitness_values[p2_idx]:
                return self._gen_regrowth_sample(games[0])
            else:
                return self._gen_regrowth_sample(games[1])

        return weighted_beam_search_sample

    def _get_parent_iterator(self):
        for _ in range(2 * self.population_size * self.k):
            parent_1 = self._choice(self.population)
            parent_2 = self._choice(self.population)
            yield [parent_1, parent_2]


MONITOR_FEATURES = ['all_variables_defined', 'all_variables_used', 'all_preferences_used']


class CrossoverOnlySampler(PopulationBasedSampler):
    def __init__(self, args: argparse.Namespace,
                 fitness_function: typing.Callable[[typing.Any], float],
                 population_size: int = 100,
                 verbose: int = 0, k: int = 1, max_attempts: int = 100,
                 crossover_type: CrossoverType = CrossoverType.SAME_PARENT_RULE,
                 fitness_featurizer: typing.Optional[ASTFitnessFeaturizer] = None,
                 monitor_feature_keys: typing.List[str] = MONITOR_FEATURES):

        super().__init__(args, fitness_function, population_size, verbose)
        self.k = k
        self.max_attempts = max_attempts
        self.crossover_type = crossover_type
        self.fitness_featurizer = fitness_featurizer
        self.monitor_feature_keys = monitor_feature_keys

    def _extract_monitor_features(self, game):
        if self.fitness_featurizer is not None:
            return {k: v for k, v in self.fitness_featurizer.parse(game, return_row=True).items() if k in self.monitor_feature_keys}  # type: ignore
        else:
            return {}

    def _get_operator(self):
        def crossover_two_random_games(games):
            for _ in range(self.max_attempts):
                try:
                    if len(games) > 2:
                        games = self._choice(games, 2)

                    before_feature_values = [self._extract_monitor_features(g) for g in games]
                    post_crossover_games = self._crossover(games[0], games[1], self.crossover_type)
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

    def _get_parent_iterator(self):
        for _ in range(self.population_size * self.k):
            yield self._choice(self.population, n=2)


class EnergyFunctionFitnessWrapper:
    energy_function: typing.Callable[[typing.Any], float]
    feature_names: typing.List[str]
    fitness_featurizer: ASTFitnessFeaturizer
    flip_sign: bool

    def __init__(self, fitness_featurizer: ASTFitnessFeaturizer, energy_function: typing.Callable[[typing.Any], float], feature_names: typing.List[str],
                 flip_sign: bool = True):
        self.fitness_featurizer = fitness_featurizer
        self.energy_function = energy_function
        self.feature_names = feature_names
        self.flip_sign = flip_sign

    def __call__(self, game: ASTType):
        features = typing.cast(dict, self.fitness_featurizer.parse(game, 'mcmc', True))  # type: ignore
        features_tensor = torch.tensor([features[name] for name in self.feature_names], dtype=torch.float32)
        if 'wrapper' in self.energy_function.named_steps:  # type: ignore
            self.energy_function.named_steps['wrapper'].eval()  # type: ignore
        energy = self.energy_function.transform(features_tensor).item()
        return -1 * energy if self.flip_sign else energy

def main(args):
    fitness_featurizer = _load_pickle_gzip(LATEST_FITNESS_FEATURIZER_PATH)
    trained_fitness_function, feature_names = load_model_and_feature_columns(LATEST_FITNESS_FUNCTION_DATE_ID, relative_path='.')

    evosampler = PopulationBasedSampler(
    # evosampler = WeightedBeamSearchSampler(25, DEFAULT_ARGS, fitness_function, verbose=0)
    # evosampler = CrossoverOnlySampler(
        args, EnergyFunctionFitnessWrapper(fitness_featurizer, trained_fitness_function, feature_names, flip_sign=True),  # type: ignore
        population_size=args.population_size,
        verbose=args.verbose,
        initial_proposal_type=InitialProposalSamplerType(args.initial_proposal_type),
        ngram_model_path=args.ngram_model_path,
    )

    # game_asts = list(cached_load_and_parse_games_from_file('./dsl/interactive-beta.pddl', evosampler.grammar_parser, False, relative_path='.'))
    # evosampler.set_population(game_asts[:4])

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
