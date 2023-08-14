import typing

import inflect
import tatsu, tatsu.ast, tatsu.grammars

from preference_handler import PredicateType
from utils import OBJECTS_BY_ROOM_AND_TYPE, extract_predicate_function_name, extract_variables, extract_variable_type_mapping
from ast_utils import cached_load_and_parse_games_from_file

DEFAULT_GRAMMAR_PATH = "./dsl/dsl.ebnf"

PREDICATE_DESCRIPTIONS = {
    "above": "{0} is above {1}",
    "adjacent": "{0} is adjacent to {1}",
    "agent_crouches": "the agent is crouching",
    "agent_holds": "the agent is holding {0}",
    "between": "{1} is between {0} and {2}",
    "faces": "{0} is facing {1}",
    "in": "{1} is inside of {0}",
    "in_motion": "{0} is in motion",
    "object_orientation": "{0} is oriented {1}",
    "on": "{1} is on {0}",
    "open": "{0} is open",
    "touch": "{0} touches {1}",
    "toggled_on": "{0} is toggled on",
}

FUNCTION_DESCRIPTIONS = {
    "distance": "the distance between {0} and {1}",
    "distance_side": "the distance between {2} and the {1} of {0}",
}

class GameDescriber():
    def __init__(self, grammar_path: str = DEFAULT_GRAMMAR_PATH):
        grammar = open(grammar_path).read()
        self.grammar_parser = typing.cast(tatsu.grammars.Grammar, tatsu.compile(grammar))
        self.engine = inflect.engine()

        self.preference_index =  None
        self.external_forall_preference_mappings = None

    def _indent(self, description: str, num_spaces: int = 4):
        '''
        Add a specified number of spaces to each line passed in
        '''
        lines = description.split("\n")
        return "\n".join([f"{' ' * num_spaces}{line}" if line !="" else line for line in lines])

    def _extract_game_info(self, ast: typing.Union[list, tuple, tatsu.ast.AST], info_dict: typing.Dict):
        '''
        Recursively extract the game's name, domain, setup, preferences, terminal conditions, and
        scoring (if they exist)
        '''
        if isinstance(ast, tuple) or isinstance(ast, list):
            for item in ast:
                self._extract_game_info(item, info_dict)

        elif isinstance(ast, tatsu.ast.AST):
            rule = ast["parseinfo"].rule  # type: ignore
            if rule == "game_def":
                info_dict["game_name"] = typing.cast(str, ast["game_name"])

            elif rule == "domain_def":
                info_dict["domain_name"] = typing.cast(str, ast["domain_name"]).split("-")[0]

            elif rule == "setup":
                info_dict["setup"] = ast["setup"]

            elif rule == "preferences":
                # Handle games with single preference
                if isinstance(ast["preferences"], tatsu.ast.AST):
                    info_dict["preferences"] = [ast["preferences"]]
                else:
                    info_dict["preferences"] = [ast["preferences"]]

            elif rule == "terminal":
                info_dict["terminal"] = ast["terminal"]

            elif rule == "scoring_expr":
                info_dict["scoring"] = ast

    def _predicate_type(self, predicate: tatsu.ast.AST) -> PredicateType:
        '''
        Returns the temporal logic type of a given predicate
        '''
        if "once_pred" in predicate.keys():
            return PredicateType.ONCE

        elif "once_measure_pred" in predicate.keys():
            return PredicateType.ONCE_MEASURE

        elif "hold_pred" in predicate.keys():

            if "while_preds" in predicate.keys():
                return PredicateType.HOLD_WHILE

            return PredicateType.HOLD

        else:
            raise ValueError(f"Error: predicate does not have a temporal logic type: {predicate.keys()}")
        
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

    def _describe_setup(self, setup_ast: tatsu.ast.AST, condition_type: typing.Optional[str] = None):
        '''
        Describe the setup of the game, including conditions that need to be satisfied once (game-optional)
        and conditions that must be met continually (game-conserved)
        '''

        rule = setup_ast["parseinfo"].rule  # type: ignore

        if rule == "setup":
            return self._describe_setup(typing.cast(tatsu.ast.AST, setup_ast["setup"]), condition_type)

        elif rule == "setup_statement":
            return self._describe_setup(typing.cast(tatsu.ast.AST, setup_ast["statement"]), condition_type)

        elif rule == "super_predicate":
            return self._describe_predicate(typing.cast(tatsu.ast.AST, setup_ast)), condition_type
        
        elif rule == "setup_not":
            text, condition_type = self._describe_setup(setup_ast["not_args"], condition_type) # type: ignore
            return f"it's not the case that {text}", condition_type # type: ignore

        elif rule == "setup_and":

            description = ""
            conditions_and_types = [self._describe_setup(sub) for sub in setup_ast["and_args"]] # type: ignore

            optional_conditions = [condition for condition, condition_type in conditions_and_types if condition_type == "optional"] # type: ignore
            if len(optional_conditions) > 0:
                description += "the following must all be true for at least one time step:"
                description += "\n- " + "\n- ".join(optional_conditions)

            conserved_conditions = [condition for condition, condition_type in conditions_and_types if condition_type == "conserved"] # type: ignore
            if len(conserved_conditions) > 0:
                if len(optional_conditions) > 0:
                    description += "\n\nand in addition, "

                description += "the following must all be true for every time step:"
                description += "\n- " + "\n- ".join(conserved_conditions)

            return description, None

        elif rule == "setup_or":
            return self.engine.join(["(" + self._describe_setup(sub) + ")" for sub in setup_ast["or_args"]], conj="or") # type: ignore
        
        elif rule == "setup_exists":
            variable_type_mapping = extract_variable_type_mapping(setup_ast["exists_vars"]["variables"]) # type: ignore

            new_variables = []
            for var, types in variable_type_mapping.items():
                new_variables.append(f"an object {var} of type {self.engine.join(types, conj='or')}")

            text, condition_type = self._describe_setup(setup_ast["exists_args"], condition_type) # type: ignore

            return f"there exists {self.engine.join(new_variables)}, such that {text}", condition_type # type: ignore

        elif rule == "setup_forall":
            variable_type_mapping = extract_variable_type_mapping(setup_ast["forall_vars"]["variables"]) # type: ignore

            new_variables = []
            for var, types in variable_type_mapping.items():
                new_variables.append(f"object {var} of type {self.engine.join(types, conj='or')}")

            text, condition_type = self._describe_setup(setup_ast["forall_args"], condition_type) # type: ignore

            return f"for any {self.engine.join(new_variables)}, {text}", condition_type # type: ignore
        
        elif rule == "setup_game_optional":
            text, _ = self._describe_setup(setup_ast["optional_pred"], condition_type) # type: ignore
            return text, "optional"
        
        elif rule == "setup_game_conserved":
            text, _ = self._describe_setup(setup_ast["conserved_pred"], condition_type) # type: ignore
            return text, "conserved"
        
        else:
            raise ValueError(f"Unknown setup expression rule: {rule}")

    def _describe_preference(self, preference_ast: tatsu.ast.AST):
        '''
        Describe a particular preference of game, calling out whether it uses an external forall
        '''

        description = ""

        pref_def = typing.cast(tatsu.ast.AST, preference_ast["definition"])
        rule = pref_def["parseinfo"].rule   # type: ignore

        if rule == "preference":
            name = typing.cast(str, pref_def["pref_name"])
            description += f"Preference {self.preference_index + 1}: '{name}'"

            body = pref_def["pref_body"]["body"] # type: ignore

            description += self._indent(self._describe_preference_body(body))
            self.preference_index += 1

        # This case handles externall forall preferences
        elif rule == "pref_forall":

            forall_vars = pref_def["forall_vars"]
            forall_pref = pref_def["forall_pref"]

            variable_type_mapping = extract_variable_type_mapping(forall_vars["variables"])  # type: ignore

            # description += "Each of the following preferences are defined inside an external 'forall' statement. This means they each make use of the following variables and, in addition, record the specific objects used for each of those variables:"
            # for var, types in variable_type_mapping.items():
            #     description += f"\n- {var}: of type {self.engine.join(types, conj='or')}"

            sub_preferences = forall_pref["preferences"] # type: ignore
            if isinstance(sub_preferences, tatsu.ast.AST):
                sub_preferences = [sub_preferences]

            for sub_idx, sub_preference in enumerate(sub_preferences):
                name = typing.cast(str, sub_preference["pref_name"])

                self.external_forall_preference_mappings[name] = list(variable_type_mapping.keys()) # type: ignore

                newline = '\n' if sub_idx > 0 else ''
                description += f"{newline}Preference {self.preference_index + 1}: '{name}'"

                body = sub_preference["pref_body"]["body"] # type: ignore

                description += self._indent(self._describe_preference_body(body, variable_type_mapping))
                self.preference_index += 1
        
        return description
        
    def _describe_preference_body(self, body_ast: tatsu.ast.AST, additional_variable_mapping: typing.Dict[str, typing.List[str]] = {}):
        '''
        Describe the main body of a preference (i.e. the part after any external-foralls / names). Optionally, additional variable
        mappings from an external forall can be passed in to be used in the description
        '''

        description = ""

        if body_ast.parseinfo.rule == "pref_body_exists":

            variable_type_mapping = extract_variable_type_mapping(body_ast["exists_vars"]["variables"])
            description += "\nThe variables required by this preference are:"
            
            for var, types in additional_variable_mapping.items():
                description += f"\n- {var}: of type {self.engine.join(types, conj='or')}"

            for var, types in variable_type_mapping.items():
                description += f"\n- {var}: of type {self.engine.join(types, conj='or')}"

            temporal_predicate_ast = body_ast["exists_args"]

        # These cases handle preferences that don't have any variables quantified with an exists (e.g. they're all from an external forall)
        elif body_ast.parseinfo.rule == "then":
            temporal_predicate_ast = body_ast

        elif body_ast.parseinfo.rule == "at_end":
            temporal_predicate_ast = body_ast

        else:
            raise NotImplementedError(f"Unknown preference body rule: {body_ast.parseinfo.rule}")

        description += "\n\nThis preference is satisfied when:"

        if temporal_predicate_ast.parseinfo.rule == "at_end":
            description += f"\n- in the final game state, {self._describe_predicate(temporal_predicate_ast['at_end_pred'])}" # type: ignore

        elif temporal_predicate_ast.parseinfo.rule == "then":

            temporal_predicates = [func['seq_func'] for func in temporal_predicate_ast["then_funcs"]]
            for idx, temporal_predicate in enumerate(temporal_predicates):
                if len(temporal_predicates) == 1:
                    description += "\n- "
                elif idx == 0:
                    description += f"\n- first, "
                elif idx == len(temporal_predicates) - 1:
                    description += f"\n- finally, "
                else:
                    description += f"\n- next, "

                temporal_type = self._predicate_type(temporal_predicate)
                if temporal_type == PredicateType.ONCE:
                    description += f"there is a state where {self._describe_predicate(temporal_predicate['once_pred'])}"

                elif temporal_type == PredicateType.ONCE_MEASURE:
                    description += f"there is a state where {self._describe_predicate(temporal_predicate['once_measure_pred'])}."
                    description += f" In addition, measure and record {self._describe_predicate(temporal_predicate['measurement'])}"

                elif temporal_type == PredicateType.HOLD:
                    description += f"there is a sequence of one or more states where {self._describe_predicate(temporal_predicate['hold_pred'])}"

                elif temporal_type == PredicateType.HOLD_WHILE:
                    description += f"there is a sequence of one or more states where {self._describe_predicate(temporal_predicate['hold_pred'])}"

                    if isinstance(temporal_predicate["while_preds"], list):
                        while_desc = self.engine.join(['a state where (' + self._describe_predicate(pred) + ')' for pred in temporal_predicate['while_preds']])
                        description += f" Additionally, during this sequence there is {while_desc} (in that order)."
                    else:
                        description += f" Additionally, during this sequence there is  a state where ({self._describe_predicate(temporal_predicate['while_preds'])})."
                
        else:
            raise ValueError(f"Unknown body exist-args rule: {temporal_predicate_ast.parseinfo.rule}")
        
        return description


    def _describe_predicate(self, predicate: tatsu.ast.AST):
        
        predicate_rule = predicate["parseinfo"].rule # type: ignore

        if predicate_rule == "predicate":

            name = extract_predicate_function_name(predicate)
            variables = extract_variables(predicate)

            return PREDICATE_DESCRIPTIONS[name].format(*variables)

        elif predicate_rule == "super_predicate":
            return self._describe_predicate(predicate["pred"]) # type: ignore

        elif predicate_rule == "super_predicate_not":
            return f"it's not the case that {self._describe_predicate(predicate['not_args'])}" # type: ignore

        elif predicate_rule == "super_predicate_and":
            return self.engine.join(["(" + self._describe_predicate(sub) + ")" for sub in predicate["and_args"]]) # type: ignore

        elif predicate_rule == "super_predicate_or":
            return self.engine.join(["(" + self._describe_predicate(sub) + ")" for sub in predicate["or_args"]], conj="or") # type: ignore

        elif predicate_rule == "super_predicate_exists":
            variable_type_mapping = extract_variable_type_mapping(predicate["exists_vars"]["variables"]) # type: ignore

            new_variables = []
            for var, types in variable_type_mapping.items():
                new_variables.append(f"an object {var} of type {self.engine.join(types, conj='or')}")

            return f"there exists {self.engine.join(new_variables)}, such that {self._describe_predicate(predicate['exists_args'])}" # type: ignore

        elif predicate_rule == "super_predicate_forall":
            variable_type_mapping = extract_variable_type_mapping(predicate["forall_vars"]["variables"]) # type: ignore

            new_variables = []
            for var, types in variable_type_mapping.items():
                new_variables.append(f"object {var} of type {self.engine.join(types, conj='or')}")

            return f"for any {self.engine.join(new_variables)}, {self._describe_predicate(predicate['forall_args'])}" # type: ignore

        elif predicate_rule == "function_comparison":

            comparison_operator = predicate["comp"]["comp_op"] # type: ignore
            comp_arg_1 = predicate["comp"]["arg_1"]["arg"] # type: ignore
            comp_arg_2 = predicate["comp"]["arg_2"]["arg"] # type: ignore

            if isinstance(comp_arg_1, tatsu.ast.AST):
                comp_arg_1 = self._describe_predicate(comp_arg_1)
     
            if isinstance(comp_arg_1, tatsu.ast.AST):
                comp_arg_2 = self._describe_predicate(comp_arg_2)

            if comparison_operator == "=":
                return f"{comp_arg_1} is equal to {comp_arg_2}"
            elif comparison_operator == "<":
                return f"{comp_arg_1} is less than {comp_arg_2}"
            elif comparison_operator == "<=":
                return f"{comp_arg_1} is less than or equal to {comp_arg_2}"
            elif comparison_operator == ">":
                return f"{comp_arg_1} is greater than {comp_arg_2}"
            elif comparison_operator == ">=":
                return f"{comp_arg_1} is greater than or equal to {comp_arg_2}"
            
        elif predicate_rule == "function_eval":
            name = extract_predicate_function_name(predicate)
            variables = extract_variables(predicate)

            return FUNCTION_DESCRIPTIONS[name].format(*variables)

        else:
            raise ValueError(f"Error: Unknown rule '{predicate_rule}'")

        return ''
    
    def _describe_terminal(self, terminal_ast: typing.Optional[tatsu.ast.AST]):
        '''
        Determine whether the terminal conditions of the game have been met
        '''
        if terminal_ast is None:
            return False

        rule = terminal_ast["parseinfo"].rule  # type: ignore

        if rule == "terminal":
            return self._describe_terminal(terminal_ast["terminal"])

        elif rule == "terminal_not":
            return f"it's not the case that {self._describe_terminal(terminal_ast['not_args'])}" # type: ignore
        
        elif rule == "terminal_and":
            return self.engine.join(["(" + self._describe_terminal(sub) + ")" for sub in terminal_ast["and_args"]]) # type: ignore

        elif rule == "terminal_or":
            return self.engine.join(["(" + self._describe_terminal(sub) + ")" for sub in terminal_ast["or_args"]], conj="or") # type: ignore

        elif rule == "terminal_comp":
            comparison_operator = terminal_ast["op"]

            expr_1 = self._describe_scoring(terminal_ast["expr_1"]["expr"]) # type: ignore
            expr_2 = self._describe_scoring(terminal_ast["expr_2"]["expr"]) # type: ignore

            if comparison_operator == "=":
                return f"{expr_1} is equal to {expr_2}" # type: ignore
            elif comparison_operator == "<":
                return f"{expr_1} is less than {expr_2}" # type: ignore
            elif comparison_operator == "<=":
                return f"{expr_1} is less than or equal to {expr_2}" # type: ignore
            elif comparison_operator == ">":
                return f"{expr_1} is greater than {expr_2}" # type: ignore
            elif comparison_operator == ">=":
                return f"{expr_1} is greater than or equal to {expr_2}" # type: ignore

        else:
            raise ValueError(f"Error: Unknown terminal rule '{rule}'")

    def _external_scoring_description(self, preference_name, external_object_types):
        '''
        A helper function for describing the special scoring syntax in which variable
        types are passed with colons after the preference name
        '''
        if external_object_types is None:
            return ""
        
        specified_variables = self.external_forall_preference_mappings[preference_name][:len(external_object_types)] # type: ignore
        mapping_description = self.engine.join([f"{var} is bound to an object of type {var_type}" for var, var_type in zip(specified_variables, external_object_types)])

        return f", where {mapping_description}"

    def _describe_scoring(self, scoring_ast: typing.Optional[tatsu.ast.AST]):

        if isinstance(scoring_ast, str):
            return scoring_ast

        rule = scoring_ast["parseinfo"].rule  # type: ignore

        if rule in ("scoring_expr", "scoring_expr_or_number"): 
            return self._describe_scoring(scoring_ast["expr"]) # type: ignore

        elif rule == "scoring_multi_expr":
            operator = scoring_ast["op"] # type: ignore
            expressions = scoring_ast["expr"] # type: ignore

            if isinstance(expressions, tatsu.ast.AST):
                return self._describe_scoring(expressions)

            elif isinstance(expressions, list):
                if operator == "+":
                    return f"the sum of {self.engine.join([f'({self._describe_scoring(expression)})' for expression in expressions])}"

                elif operator == "*":
                    return f"the product of {self.engine.join([f'({self._describe_scoring(expression)})' for expression in expressions])}"

        elif rule == "scoring_binary_expr":
            operator = scoring_ast["op"] # type: ignore

            expr_1 = self._describe_scoring(scoring_ast["expr_1"]) # type: ignore
            expr_2 = self._describe_scoring(scoring_ast["expr_2"]) # type: ignore

            if operator == "-":
                return f"{expr_1} minus {expr_2}"
            elif operator == "/":
                return f"{expr_1} divided by {expr_2}"

        elif rule == "scoring_neg_expr":
            return f"negative {self._describe_scoring(scoring_ast['expr'])}" # type: ignore
        
        elif rule == "scoring_comparison":
            raise NotImplementedError("Comparison scoring not yet implemented")
        
        elif rule == "preference_eval":
            return self._describe_scoring(scoring_ast["count_method"]) # type: ignore
        
        elif rule == "scoring_external_maximize":
            preference_name, _ = self._extract_name_and_types(scoring_ast['scoring_expr']['expr']['count_method']) # type: ignore
            external_variables = self.external_forall_preference_mappings[preference_name] # type: ignore

            internal_description = self._describe_scoring(scoring_ast["scoring_expr"]) # type: ignore
            return f"the maximum value of ({internal_description}) over all quantifications of {self.engine.join(external_variables, conj='and')}"
        
        elif rule == "scoring_external_minimize":
            preference_name, _ = self._extract_name_and_types(scoring_ast['scoring_expr']['expr']['count_method']) # type: ignore
            external_variables = self.external_forall_preference_mappings[preference_name] # type: ignore

            internal_description = self._describe_scoring(scoring_ast["scoring_expr"]) # type: ignore
            return f"the minimum value of ({internal_description}) over all quantifications of {self.engine.join(external_variables, conj='and')}"
        
        elif rule == "count":
            preference_name, object_types = self._extract_name_and_types(scoring_ast) # type: ignore
            external_scoring_desc = self._external_scoring_description(preference_name, object_types)

            return f"the number of times '{preference_name}' has been satisfied" + external_scoring_desc
          
        elif rule == "count_overlapping":
            preference_name, object_types = self._extract_name_and_types(scoring_ast) # type: ignore
            external_scoring_desc = self._external_scoring_description(preference_name, object_types)
            
            return f"the number of times '{preference_name}' has been satisfied in overlapping intervals" + external_scoring_desc

        elif rule == "count_once":
            preference_name, object_types = self._extract_name_and_types(scoring_ast) # type: ignore
            external_scoring_desc = self._external_scoring_description(preference_name, object_types)
            return f"whether '{preference_name}' has been satisfied at least once" + external_scoring_desc
   
        elif rule == "count_once_per_objects":
            preference_name, object_types = self._extract_name_and_types(scoring_ast) # type: ignore
            external_scoring_desc = self._external_scoring_description(preference_name, object_types)
            
            return f"the number of times '{preference_name}' has been satisfied with different objects" + external_scoring_desc 
            
        elif rule == "count_measure":
            preference_name, object_types = self._extract_name_and_types(scoring_ast) # type: ignore
            
            if object_types is None:
                return f"the sum of all values measured during satisfactions of '{preference_name}'"
            else:
                raise ValueError("Error: count_measure does not support specific object types (I think?)")
            
        elif rule == "count_unique_positions":
            preference_name, object_types = self._extract_name_and_types(scoring_ast) # type: ignore
            if object_types is None:
                return f"the number of times '{preference_name}' has been satisfied with stationary objects in different positions"
            
        elif rule == "count_same_positions":
            raise NotImplementedError("count_same_positions not yet implemented")
        
        elif rule == "count_once_per_external_objects":
            raise NotImplementedError("count_once_per_external_objects not yet implemented")
        
        else:
            raise ValueError(f"Error: Unknown rule '{rule}' in scoring expression")

    def describe(self, game_text_or_ast: typing.Union[str, tatsu.ast.AST]):
        '''
        Generate a description of the provided game text or AST. Description will be split
        by game section (setup, preferences, terminal, and scoring)
        '''

        if isinstance(game_text_or_ast, str):
            game_ast = typing.cast(tatsu.ast.AST, self.grammar_parser.parse(game_text))
        else:
            game_ast = game_text_or_ast
        
        game_info = {}
        self._extract_game_info(game_ast, game_info)

        self.preference_index = 0
        self.external_forall_preference_mappings = {}

        if game_info.get("setup") is not None:
            print("=====================================GAME SETUP=====================================")
            setup_description, _ = self._describe_setup(game_info["setup"])
            print(f"\nIn order to set up the game, {setup_description}")


        if game_info.get("preferences") is not None:
            print("\n=====================================PREFERENCES=====================================")
            for idx, preference in enumerate(game_info["preferences"][0]):
                description = self._describe_preference(preference)
                print(f"\n{description}")

        if game_info.get("terminal") is not None:
            print("\n=====================================TERMINAL CONDITIONS=====================================")
            terminal_description = self._describe_terminal(game_info["terminal"])
            print(f"\nThe game ends when {terminal_description}")

        if game_info.get("scoring") is not None:
            print("\n=====================================SCORING=====================================")
            scoring_description = self._describe_scoring(game_info["scoring"])
            print(f"\nAt the end of the game, the player's score is {scoring_description}")


TEST_GAME = """(define (game 61267978e96853d3b974ca53-23) (:domain medium-objects-room-v1)

(:constraints (and
    (preference throwTeddyOntoPillow
        (exists (?t - teddy_bear ?p - pillow)
            (then
                (once (agent_holds ?t))
                (hold (and (not (agent_holds ?t)) (in_motion ?t)))
                (once (and (not (in_motion ?t)) (on ?p ?t)))
            )
        )
    )
    (preference throwAttempt
        (exists (?t - teddy_bear)
            (then
                (once (agent_holds ?t))
                (hold (and (not (agent_holds ?t)) (in_motion ?t)))
                (once (not (in_motion ?t)))
            )
        )
    )
))
(:terminal
    (>= (count throwAttempt) 10)
)
(:scoring (count throwTeddyOntoPillow)
))
"""

TEST_GAME_2 = """(define (game 610aaf651f5e36d3a76b199f-28) (:domain few-objects-room-v1)  ; 28
(:setup (and
    (forall (?c - cube_block) (game-conserved (on rug ?c)))
))
(:constraints (and
    (preference thrownBallReachesEnd
            (exists (?d - dodgeball)
                (then
                    (once (and (agent_holds ?d) (not (on rug agent))))
                    (hold-while
                        (and
                            (not (agent_holds ?d))
                            (in_motion ?d)
                            (not (exists (?b - cube_block) (touch ?d ?b)))
                        )
                        (above rug ?d)
                    )
                    (once (or (touch ?d bed) (touch ?d west_wall)))
                )
            )
        )
))
(:terminal (or
    (>= (total-time) 180)
    (>= (total-score) 50)
))
(:scoring (+
    (* 10 (count thrownBallReachesEnd))
    (* (- 5) (count thrownBallHitsBlock:red))
    (* (- 3) (count thrownBallHitsBlock:green))
    (* (- 3) (count thrownBallHitsBlock:pink))
    (- (count thrownBallHitsBlock:yellow))
    (- (count thrownBallHitsBlock:purple))
)))"""

if __name__ == '__main__':
    # game = open("./reward-machine/games/game-6.txt", "r").read()

    grammar = open('./dsl/dsl.ebnf').read()
    grammar_parser = tatsu.compile(grammar)
    game_asts = list(cached_load_and_parse_games_from_file('./dsl/interactive-beta.pddl', grammar_parser, False, relative_path='.'))
    game_describer = GameDescriber()

    for game in game_asts:
        game_describer.describe(game)
        input()