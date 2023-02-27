
class EvolutionarySampler():
    '''
    This is a type of game sampler which uses an evolutionary strategy to climb a
    provided fitness function. It's a population-based alternative to the MCMC samper
    '''

    def __init__(self, etc):

        self.verbose = verbose

        self.grammar = open(args.grammar_file).read()
        self.grammar_parser = tatsu.compile(self.grammar)
        self.counter = parse_or_load_counter(args, self.grammar_parser)

        # Used to generate the initial population of complete games
        self.initial_sampler = ASTSampler(self.grammar_parser, self.counter, seed=args.random_seed)
        
        # Used as the mutation operator to modify existing games
        self.regrowth_sampler = RegrowthSampler(self.sampler, args.random_seed)

        # Generate the initial population
        self.population = [self._init_sample(idx) for idx in range(args.population_size)]

    def _init_sample(self, idx):
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
                if self.verbose >= 2: print('Sampling exception in sample {idx} -- skipping')

        return sample

    def step(self):
        pass