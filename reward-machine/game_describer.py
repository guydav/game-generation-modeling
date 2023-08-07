import typing

import inflect
import tatsu, tatsu.ast, tatsu.grammars

from utils import OBJECTS_BY_ROOM_AND_TYPE, extract_predicate_function_name, extract_variables, extract_variable_type_mapping

DEFAULT_GRAMMAR_PATH = "./dsl/dsl.ebnf"

PREDICATE_DESCRIPTIONS = {
    "above": "{0} is above {1}",
    "adjacent": "{0} is adjacent to {1}",
    "agent_crouches": "the agent is crouching",
    "agent_holds": "the agent is holding {0}",
    "between": "{1} is between {0} and {2}",
    "in": "{1} is inside of {0}",
    "in_motion": "{0} is in motion",
    "faces": "{0} is facing {1}",
    "on": "{1} is on {0}",
    "touch": "{0} touches {1}",
    "toggled_on": "{0} is toggled on",
}

FUNCTION_DESCRIPTIONS = {
    "distance": "the distance between {0} and {1}"
}

class GameDescriber():
    def __init__(self, grammar_path: str = DEFAULT_GRAMMAR_PATH):
        grammar = open(grammar_path).read()
        self.grammar_parser = typing.cast(tatsu.grammars.Grammar, tatsu.compile(grammar))
        self.engine = inflect.engine()


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
                info_dict["scoring"] = ast["scoring"]

    def _describe_setup(self, setup_ast: tatsu.ast.AST, condition_type: typing.Optional[str] = None):
        '''
        Describe the setup of the game, including conditions that need to be satisfied once (game-optional)
        and conditions that must be met continually (game-conserved)
        '''

        rule = setup_ast["parseinfo"].rule  # type: ignore

        print(f"Setup rule = {rule}")
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

                description += "the following must all be true for all time steps:"
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
            print(f"exists text = {text}")

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

            if isinstance(comp_arg_1, tatsu.ast.AST):

                name = comp_arg_1["func_name"]
                variables = extract_variables(comp_arg_1)

                comp_arg_1 = FUNCTION_DESCRIPTIONS[name].format(*variables)  # type: ignore

            comp_arg_2 = predicate["comp"]["arg_2"]["arg"]
            if isinstance(comp_arg_1, tatsu.ast.AST):
                name = comp_arg_2["func_name"]
                variables = extract_variables(comp_arg_2)

                comp_arg_1 = FUNCTION_DESCRIPTIONS[name].format(*variables)

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

        else:
            raise ValueError(f"Error: Unknown rule '{predicate_rule}'")

        return ''

    def describe(self, game_text):
        game_ast = self.grammar_parser.parse(game_text)
        
        game_info = {}
        self._extract_game_info(game_ast, game_info)

        if game_info.get("setup") is not None:
            setup_description, _ = self._describe_setup(game_info["setup"])
            print("=====================================GAME SETUP=====================================")
            print(f"The setup conditions are as follows: {setup_description}")


if __name__ == '__main__':
    game = open("./reward-machine/games/game-15.txt", "r").read()
    game_describer = GameDescriber()
    game_describer.describe(game)