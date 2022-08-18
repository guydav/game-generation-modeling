import inflect
import tatsu

def describe_preference(preference):
    '''
    Generate a natural language description of the given preference in plain language
    by recursively applying a set of rules.
    '''

    print(preference)
    rule = preference["parseinfo"].rule

    for key in preference.keys():
        print(key)
        describe_preference(preference[key])

PREDICATE_DESCRIPTIONS = {
    "above": "{0} is above {1}",
    "agent_crouches": "the agent is crouching",
    "agent_holds": "the agent is holding {0}",
    "in": "{1} is inside of {0}",
    "in_motion": "{0} is in motion",
    "on": "{1} is on {0}",
    "touch": "{0} touches {1}"
}

FUNCTION_DESCRIPTIONS = {
    "distance": "the distance between {0} and {1}"
}

class PreferenceDescriber():
    def __init__(self, preference):
        self.preference_name = preference["pref_name"]
        self.body = preference["pref_body"]["body"]

        self.variable_type_mapping = self._extract_variable_type_mapping(self.body["exists_vars"]["variables"])
        self.variable_type_mapping["agent"] = ["agent"]

        self.temporal_predicates = [func["seq_func"] for func in self.body["exists_args"]["body"]["then_funcs"]]

        self.engine = inflect.engine()

    def _extract_variable_type_mapping(self, variable_list):
        '''
        Given a list of variables, extract the mapping from variable names to variable types. Variable types are
        stored in lists, even in cases where there is only one possible for the variable in order to handle cases
        where multiple types are linked together with an (either) clause
        '''
        if isinstance(variable_list, tatsu.ast.AST):
            variable_list = [variable_list]

        variables = {}
        for var_info in variable_list:
            if isinstance(var_info["var_type"]["type"], tatsu.ast.AST):
                variables[var_info["var_names"]] = var_info["var_type"]["type"]["type_names"]
            else:
                variables[var_info["var_names"]] = [var_info["var_type"]["type"]]

        return variables

    def _extract_variables(self, predicate):
        '''
        Recursively extract every variable referenced in the predicate (including inside functions 
        used within the predicate)

        BIG TODO: some objects (like 'desk') are just referred to directly inside of predicates, and
        are never quantified over (i.e. we never see (exists ?d - desk)). We need to be able to detect
        and handle these kinds of variables
        '''

        if isinstance(predicate, list) or isinstance(predicate, tuple):
            pred_vars = []
            for sub_predicate in predicate:
                pred_vars += self._extract_variables(sub_predicate)

            unique_vars = []
            for var in pred_vars:
                if var not in unique_vars:
                    unique_vars.append(var)

            return unique_vars

        elif isinstance(predicate, tatsu.ast.AST):
            pred_vars = []
            for key in predicate:
                if key == "term":

                    # Different structure for predicate args vs. function args
                    if isinstance(predicate["term"], tatsu.ast.AST):
                        pred_vars += [predicate["term"]["arg"]]
                    else:
                        pred_vars += [predicate["term"]]

                # We don't want to capture any variables within an (exists) or (forall) that's inside 
                # the preference, since those are not globally required -- see evaluate_predicate()
                elif key == "exists_args":
                    continue

                elif key == "forall_args":
                    continue

                elif key != "parseinfo":
                    pred_vars += self._extract_variables(predicate[key])

            unique_vars = []
            for var in pred_vars:
                if var not in unique_vars:
                    unique_vars.append(var)

            return unique_vars

        else:
            return []

    def _type(self, predicate):
        '''
        Returns the temporal logic type of a given predicate
        '''
        if "once_pred" in predicate.keys():
            return "once"

        elif "once_measure_pred" in predicate.keys():
            return "once-measure"

        elif "hold_pred" in predicate.keys():

            if "while_preds" in predicate.keys():
                return "hold-while"

            return "hold"

        else:
            exit("Error: predicate does not have a temporal logic type")

    def describe(self):
        print("\nDescribing preference:", self.preference_name)
        print("The variables required by this preference are:")
        for var, types in self.variable_type_mapping.items():
            print(f" - {var}: of type {self.engine.join(types, conj='or')}")

        for idx, predicate in enumerate(self.temporal_predicates):
            if idx == 0:
                prefix = f"\n[{idx}] First, "
            elif idx == len(self.temporal_predicates) - 1:
                prefix = f"\n[{idx}] Finally, "
            else:
                prefix = f"\n[{idx}] Next, "

            pred_type = self._type(predicate)
            if pred_type == "once":
                description = f"we need a single state where {self.describe_predicate(predicate['once_pred'])}."

            elif pred_type == "once-measure":
                description = f"we need a single state where {self.describe_predicate(predicate['once_measure_pred'])}."

                # TODO: describe which measurement is performed

            elif pred_type == "hold":
                description = f"we need a sequence of states where {self.describe_predicate(predicate['hold_pred'])}."

            elif pred_type == "hold-while":
                description = f"we need a sequence of states where {self.describe_predicate(predicate['hold_pred'])}."

                if isinstance(predicate["while_preds"], list):
                    while_desc = self.engine.join(['a state where (' + self.describe_predicate(pred) + ')' for pred in predicate['while_preds']])
                    description += f" During this sequence, we need {while_desc} (in that order)."
                else:
                    description += f" During this sequence, we need a state where ({self.describe_predicate(predicate['while_preds'])})."

            print(prefix + description)

    def describe_predicate(self, predicate):
        predicate_rule = predicate["parseinfo"].rule

        if predicate_rule == "predicate":

            name = predicate["pred_name"]
            variables = self._extract_variables(predicate)

            return PREDICATE_DESCRIPTIONS[name].format(*variables)

        elif predicate_rule == "super_predicate":
            return self.describe_predicate(predicate["pred"])

        elif predicate_rule == "super_predicate_not":
            return f"it's not the case that {self.describe_predicate(predicate['not_args'])}"

        elif predicate_rule == "super_predicate_and":
            return self.engine.join(["(" + self.describe_predicate(sub) + ")" for sub in predicate["and_args"]])

        elif predicate_rule == "super_predicate_or":
            return self.engine.join(["(" + self.describe_predicate(sub) + ")" for sub in predicate["or_args"]], conj="or")

        elif predicate_rule == "super_predicate_exists":
            variable_type_mapping = self._extract_variable_type_mapping(predicate["exists_vars"]["variables"])

            new_variables = []
            for var, types in variable_type_mapping.items():
                new_variables.append(f"an object {var} of type {self.engine.join(types, conj='or')}")

            return f"there exists {self.engine.join(new_variables)}, such that {self.describe_predicate(predicate['exists_args'])}"

        elif predicate_rule == "super_predicate_forall":
            variable_type_mapping = self._extract_variable_type_mapping(predicate["forall_vars"]["variables"])

            new_variables = []
            for var, types in variable_type_mapping.items():
                new_variables.append(f"object {var} of type {self.engine.join(types, conj='or')}")

            return f"for any {self.engine.join(new_variables)}, {self.describe_predicate(predicate['forall_args'])}"

        elif predicate_rule == "function_comparison":
            comparison_operator = predicate["comp"]["comp_op"]

            comp_arg_1 = predicate["comp"]["arg_1"]["arg"]
            if isinstance(comp_arg_1, tatsu.ast.AST):

                name = comp_arg_1["func_name"]
                variables = self._extract_variables(comp_arg_1)

                comp_arg_1 = FUNCTION_DESCRIPTIONS[name].format(*variables)

            comp_arg_2 = predicate["comp"]["arg_2"]["arg"]
            if isinstance(comp_arg_1, tatsu.ast.AST):
                name = comp_arg_2["func_name"]
                variables = self._extract_variables(comp_arg_2)

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
            exit(f"Error: Unknown rule '{predicate_rule}'")