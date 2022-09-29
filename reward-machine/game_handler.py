import itertools
import os
import sys
import tatsu
import tatsu.ast
import typing

from math import prod

from config import OBJECTS_BY_ROOM_AND_TYPE, NAMED_OBJECTS
from preference_handler import PreferenceHandler
from predicate_handler import PredicateHandler
from utils import extract_variable_type_mapping, get_object_assignments


DEFAULT_GRAMMAR_PATH = "./dsl/dsl.ebnf"


class GameHandler():
    def __init__(self, game: str, grammar_path: str = DEFAULT_GRAMMAR_PATH):
        grammar = open(grammar_path).read()
        self.grammar_parser = tatsu.compile(grammar)

        self.game_name = None
        self.domain_name = None
        self.setup = None
        self.game_optional_cache = set([])
        self.preferences = []
        self.terminal = None
        self.scoring = None

        self.game_ast = self.grammar_parser.parse(game)
        self._extract_game_info(self.game_ast)

        if self.domain_name is None:
            raise ValueError("Error: Failed to extract domain from game specification")

        self.predicate_handler = PredicateHandler(self.domain_name)

        # Maps from each preference name to the PreferenceHandler (or list of PreferenceHandlers) that will 
        # evaluate that preference
        self.preference_handlers = {}

        # Maps from each preference name to a list of satisfaction data. Each entry in the list is a tuple
        # of the following form: (variable_mapping, start_state_num, end_state_num)
        self.preference_satisfactions = {}

        for preference in self.preferences:
            rule = preference["definition"]["parseinfo"].rule

            # A preference definition expands into either (forall <variable_list> <preference>) or <preference>
            if rule == "preference":
                name = preference["definition"]["pref_name"]
                
                pref_handler = PreferenceHandler(preference["definition"], self.predicate_handler, self.domain_name)
                self.preference_handlers[name] = pref_handler
                self.preference_satisfactions[name] = []
                print(f"Successfully constructed PreferenceHandler for '{name}'")


            # TODO: forall can cover multiple preferences
            elif rule == "pref_forall":
                # The outer "forall" works to track the times the inner preference is satisfied for each type
                # in the outer variables (e.g. )
                forall_vars = preference["definition"]["forall_vars"]
                forall_pref = preference["definition"]["forall_pref"]
                
                variable_type_mapping = extract_variable_type_mapping(forall_vars["variables"])
                sub_preference = forall_pref["preferences"]
                name = sub_preference["pref_name"]

                pref_handler = PreferenceHandler(sub_preference, self.predicate_handler, self.domain_name, 
                    additional_variable_mapping=variable_type_mapping)
                self.preference_handlers[name] = pref_handler
                self.preference_satisfactions[name] = []
                print(f"Successfully constructed PreferenceHandler for '{name}'")

    def _extract_game_info(self, ast: typing.Union[list, tuple, tatsu.ast.AST]):
        '''
        Recursively extract the game's name, domain, setup, preferences, terminal conditions, and
        scoring (if they exist)
        '''
        if isinstance(ast, tuple) or isinstance(ast, list):
            for item in ast:
                self._extract_game_info(item)

        elif isinstance(ast, tatsu.ast.AST):
            rule = ast["parseinfo"].rule  # type: ignore
            if rule == "game_def":
                self.game_name = ast["game_name"]

            elif rule == "domain_def":
                self.domain_name = ast["domain_name"].split("-")[0]  # type: ignore
                if self.domain_name not in OBJECTS_BY_ROOM_AND_TYPE:
                    raise ValueError(f"Error: Domain '{self.domain_name}' not supported (not found in the keys of OBJECTS_BY_ROOM_AND_TYPE: {list(OBJECTS_BY_ROOM_AND_TYPE.keys())}")

            elif rule == "setup":
                self.setup = ast["setup"]

            elif rule == "preferences":
                # Handle games with single preference
                if isinstance(ast["preferences"], tatsu.ast.AST):
                    self.preferences = [ast["preferences"]]
                else:
                    self.preferences = ast["preferences"]

            elif rule == "terminal":
                self.terminal = ast["terminal"]

            elif rule == "scoring":
                self.scoring = ast["scoring"]

    def process(self, state: typing.Dict[str, typing.Any]) -> typing.Optional[float]:  
        '''
        Process a state in a game trajectory by passing it to each of the relevant PreferenceHandlers. If the state is
        the last one in the trajectory or the terminal conditions are met, then we also do scoring
        '''

        # Every named object will exist only once in the room, so we can just 
        default_mapping = {obj: OBJECTS_BY_ROOM_AND_TYPE[self.domain_name][obj][0] for obj in NAMED_OBJECTS}
        setup = self.evaluate_setup(self.setup, state, default_mapping)

        for preference_name, handlers in self.preference_handlers.items():
            if isinstance(handlers, PreferenceHandler):
                satisfactions = handlers.process(state)
                self.preference_satisfactions[preference_name] += satisfactions

            elif isinstance(handlers, list):
                pass

        terminate = self.evaluate_terminals(self.terminal) # TODO

        if terminate:
            score = self.score(self.scoring) 

        else:
            score = None

        return score

    def evaluate_setup(self, setup_expression: typing.Optional[tatsu.ast.AST], state: typing.Dict[str, typing.Any],
                       mapping: typing.Optional[typing.Dict[str, str]]) -> bool:
        '''
        Determine whether the setup conditions of the game have been met. The setup conditions
        of a game are met if all of the game-optional expressions have evaluated to True at least
        once in the past and if all of the game-conserved expressions currently evaluate to true
        '''

        # TODO: this function will probably need to take a potential mapping and state as arguments,
        #       since we eventually need to evaluate predicates

        if setup_expression is None:
            return True

        rule = setup_expression["parseinfo"].rule  # type: ignore

        print(f"{rule}: {setup_expression.keys()}")

        if rule == "setup":
            return self.evaluate_setup(setup_expression["setup"], state, mapping)

        elif rule == "setup_statement":
            return self.evaluate_setup(setup_expression["statement"], state, mapping)

        elif rule == "super_predicate":
            evaluation = self.predicate_handler(setup_expression, state, mapping)
            print("Evaluation:", evaluation)
            exit()

        elif rule == "setup_not":
            inner_value = self.evaluate_setup(setup_expression["not_args"], state, mapping)  

            return not inner_value

        elif rule == "setup_and":
            inner_values = [self.evaluate_setup(sub, state, mapping) for sub in setup_expression["and_args"]]  # type: ignore

            return all(inner_values)

        elif rule == "setup_or":
            inner_values = [self.evaluate_setup(sub, state, mapping) for sub in setup_expression["or_args"]]   # type: ignore

            return any(inner_values)

        elif rule == "setup_exists":
            variable_type_mapping = extract_variable_type_mapping(setup_expression["exists_vars"]["variables"])  # type: ignore
            object_assignments = get_object_assignments(self.domain_name, variable_type_mapping.values())

            sub_mappings = [dict(zip(variable_type_mapping.keys(), object_assignment)) for object_assignment in object_assignments]
            inner_mapping_values = [self.evaluate_setup(setup_expression["exists_args"], state, {**sub_mapping, **mapping}) for sub_mapping in sub_mappings]

            if all(v is None for v in inner_mapping_values):
                return None

            return any(inner_mapping_values)

        elif rule == "setup_forall":
            pass

        elif rule == "setup_game_optional":
            # Once the game-optional condition has been satisfied once, we no longer need to evaluate it

            cache_str = str(setup_expression["optional_pred"])
            if cache_str in self.game_optional_cache:
                return True

            evaluation = self.evaluate_setup(setup_expression["optional_pred"], state, mapping)
            if evaluation:
                self.game_optional_cache.add(cache_str)

            return evaluation

        elif rule == "setup_game_conserved":
            # By contrast, each game-conserved condition must be satisfied at every step
            return self.evaluate_setup(setup_expression["conserved_pred"], state, mapping)

        return True

    def evaluate_terminals(self, terminal_expression: typing.Optional[tatsu.ast.AST]) -> bool:
        '''
        Determine whether the terminal conditions of the game have been met
        '''
        if terminal_expression is None:
            return False

        rule = terminal_expression["parseinfo"].rule  # type: ignore

        if rule == "terminal":
            return self.evaluate_terminals(terminal_expression["terminal"])  

        elif rule == "terminal_not":
            inner_value = self.evaluate_terminals(terminal_expression["not_args"])  

            return not inner_value

        elif rule == "terminal_and":
            inner_values = [self.evaluate_terminals(sub) for sub in terminal_expression["and_args"]]  # type: ignore

            return all(inner_values)

        elif rule == "terminal_or":
            inner_values = [self.evaluate_terminals(sub) for sub in terminal_expression["or_args"]]   # type: ignore

            return any(inner_values)

        # Interestingly, in the grammar a terminal comparison can only over have 2 arguments (i.e. there can be no
        # (= arg1 arg2 arg3)), so this makes extracting the expressions a bit more straightforward
        elif rule == "terminal_comp":
            comparison_operator = terminal_expression["op"]

            expr_1 = self.score(terminal_expression["expr_1"]["expr"]) # type: ignore
            expr_2 = self.score(terminal_expression["expr_2"]["expr"]) # type: ignore

            if comparison_operator == "=":
                return expr_1 == expr_2
            elif comparison_operator == "<":
                return expr_1 < expr_2
            elif comparison_operator == "<=":
                return expr_1 <= expr_2
            elif comparison_operator == ">":
                return expr_1 > expr_2
            elif comparison_operator == ">=":
                return expr_1 >= expr_2
            else:
                raise ValueError(f"Error: Unknown comparison operator '{comparison_operator}'")

        else:
            raise ValueError(f"Error: Unknown terminal rule '{rule}'")

    def _extract_name_and_types(self, scoring_expression: tatsu.ast.AST) -> typing.Tuple[str, typing.Optional[typing.Sequence[str]]]:
        '''
        Helper function to extract the name of the preference being scored, as well as any of the object types that have been
        passed to it using the ":" syntax
        '''
        name_and_types = typing.cast(tatsu.ast.AST, scoring_expression["name_and_types"])
        preference_name = name_and_types["pref_name"]

        if isinstance(name_and_types["object_types"], tatsu.ast.AST):
            object_types = [name_and_types["object_types"]["type_name"]]  # type: ignore

        elif isinstance(name_and_types["object_types"], list):
            object_types = [object_type["type_name"] for object_type in name_and_types["object_types"]]  # type: ignore

        else:
            object_types = None
        
        return str(preference_name), object_types


    def _filter_satisfactions(self, preference_name: str, object_types: typing.Optional[typing.Sequence[str]]):
        '''
        Given the name of a preference and a list of object types, return the set of all satisfactions
        of the given preference that abide by the given variable mapping. In the case where object_types
        is None, this amounts to returning all the satisfactions of the given preference. When object_types
        is not None (as in the case of an external forall and the use of the ":" syntax), then we filter
        accordingly.
        '''

        if object_types is None:
            return self.preference_satisfactions[preference_name]

        pref_handler = self.preference_handlers[preference_name]
        satisfactions = []

        # Check to see if the mapping lines up with the specified object type
        for potential_sat in self.preference_satisfactions[preference_name]:
            mapping = potential_sat.mapping
            
            # A PreferenceHandler's additional_variable_mapping is an OrderedDict, so the types in 
            # object_types need to match the order of varaibles in the external forall
            acceptable_sat = True
            for variable, object_type in zip(pref_handler.additional_variable_mapping.keys(), object_types):
                specififed_object = mapping[variable]
                self.domain_name = typing.cast(str, self.domain_name)
                if specififed_object not in OBJECTS_BY_ROOM_AND_TYPE[self.domain_name][object_type]:
                    acceptable_sat = False

            if acceptable_sat:
                satisfactions.append(potential_sat)

        return satisfactions

    def score(self, scoring_expression: typing.Union[str, tatsu.ast.AST, None]) -> float:
        '''
        Determine the score of the current trajectory using the given scoring expression
        '''
        if scoring_expression is None:
            return 0.0

        # TODO: is the only situation in which we'll directly score a string?
        if isinstance(scoring_expression, str):
            return float(scoring_expression)
        
        rule = scoring_expression["parseinfo"].rule  # type: ignore

        # TODO: clearly there needs to be some logic here for handling maximize vs. minimize. Maybe we should
        # pass an argument down the recursive calls?
        if rule == "scoring_maximize":
            return self.score(scoring_expression["expr"])

        elif rule == "scoring_minimize":
            return self.score(scoring_expression["expr"])

        elif rule == "scoring_expr":
            return self.score(scoring_expression["expr"])

        elif rule == "scoring_multi_expr":
            # Multi-expression operators are either addition (+) or multiplication (*)
            operator = scoring_expression["op"]

            # Despite being multi-expression operators, they can still accept one or more arguments
            expressions = scoring_expression["expr"]

            if isinstance(expressions, tatsu.ast.AST):
                return self.score(expressions)

            elif isinstance(expressions, list):
                if operator == "+":
                    return sum([self.score(expression) for expression in expressions])

                elif operator == "*":
                    return prod([self.score(expression) for expression in expressions])

        elif rule == "scoring_binary_expr":
            # Binary expression operators are either subtraction (-) or division (/)
            operator = scoring_expression["op"]

            expr_1 = scoring_expression["expr_1"]
            expr_2 = scoring_expression["expr_2"]

            if operator == "-":
                return self.score(expr_1) - self.score(expr_2)
            elif operator == "/":
                return self.score(expr_1) / self.score(expr_2)

        elif rule == "scoring_neg_expr":
            return - self.score(scoring_expression["expr"])

        elif rule == "scoring_comparison":
            comp_expr = typing.cast(tatsu.ast.AST, scoring_expression["comp"])
            comparison_operator = comp_expr["op"]  

            # In this case, we know that the operator is = and that we have more than 2 comparison arguments,
            # so we just determine whether all arguments evaluate to the same value
            if comparison_operator is None:
                expressions = comp_expr["expr"]
                evaluations = [self.score(expr) for expr in expressions]  # type: ignore

                return float(evaluations.count(evaluations[0]) == len(evaluations))

            # Otherwise, there will be exactly two comparison arguments and we can compare them normally
            else:
                expr_1 = comp_expr["expr_1"]
                expr_2 = comp_expr["expr_2"]

                if comparison_operator == "=":
                    return self.score(expr_1) == self.score(expr_2)
                elif comparison_operator == "<":
                    return self.score(expr_1) < self.score(expr_2)
                elif comparison_operator == "<=":
                    return self.score(expr_1) <= self.score(expr_2)
                elif comparison_operator == ">":
                    return self.score(expr_1) > self.score(expr_2)
                elif comparison_operator == ">=":
                    return self.score(expr_1) >= self.score(expr_2)
                else:
                    raise ValueError(f"Error: Unknown comparison operator '{comparison_operator}'")

        elif rule == "preference_eval":
            return self.score(scoring_expression["count_method"])

        # Count the number of satisfactions of the given preference that don't overlap in both
        # (a) the mapping of variables to objects
        # (b) the temporal states involved
        elif rule == "count_nonoverlapping":
            preference_name, object_types = self._extract_name_and_types(scoring_expression)
            satisfactions = self._filter_satisfactions(preference_name, object_types)

            # Group the satisfactions by their mappings. Within each group, ensure there are no state overlaps and
            # count the total number of satisfactions that satisfy those criteria
            count = 0

            keyfunc = lambda satisfaction: "_".join(satisfaction[0].values())
            for key, group in itertools.groupby(sorted(satisfactions, key=keyfunc), keyfunc):
                group = list(sorted(group, key=lambda satisfaction: satisfaction[2]))

                prev_end = -1
                for mapping, start, end, measures in group:
                    if start >= prev_end:
                        prev_end = end
                        count += 1

            return count

        # Count whether the preference has been satisfied at all
        elif rule == "count_once":
            preference_name, object_types = self._extract_name_and_types(scoring_expression)

            satisfactions = self._filter_satisfactions(preference_name, object_types)

            return 1 if len(satisfactions) > 0 else 0

        # Count the number of satisfactions of the given preference that use distinct variable mappings
        elif rule == "count_once_per_objects":
            preference_name, object_types = self._extract_name_and_types(scoring_expression)

            satisfactions = self._filter_satisfactions(preference_name, object_types)

            count = 0

            keyfunc = lambda satisfaction: "_".join(satisfaction[0].values())
            for key, group in itertools.groupby(sorted(satisfactions, key=keyfunc), keyfunc):
                count += 1

            return count

        # For each nonoverlapping satisfaction (see count_nonoverlapping above), sum the value of the measurement
        elif rule == "count_nonoverlapping_measure":
            preference_name, object_types = self._extract_name_and_types(scoring_expression)

            satisfactions = self._filter_satisfactions(preference_name, object_types)

            count = 0

            keyfunc = lambda satisfaction: "_".join(satisfaction[0].values())
            for key, group in itertools.groupby(sorted(satisfactions, key=keyfunc), keyfunc):
                group = list(sorted(group, key=lambda satisfaction: satisfaction[2]))

                prev_end = -1
                for mapping, start, end, measures in group:
                    if start >= prev_end:
                        prev_end = end
                        count += list(measures.values())[0] # TODO: will we only ever have one measurement per preference?

            return count

        elif rule == "count_unique_positions":
            pass # TODO

        elif rule == "count_same_positions":
            pass # TODO

        elif rule == "count_maximal_nonoverlapping":
            pass # TODO

        elif rule == "count_maximal_overlapping":
            pass # TODO

        elif rule == "count_maximal_once_per_objects":
            pass # TODO

        elif rule == "count_maximal_once":
            pass # TODO

        elif rule == "count_once_per_external_objects":
            pass # TODO

        else:
            raise ValueError(f"Error: Unknown rule '{rule}' in scoring expression")

        return 0.0