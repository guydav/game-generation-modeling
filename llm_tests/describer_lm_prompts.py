import csv

###
# Prompts used for getting GPT-X to generate natural language (ish) descriptions of games. Prompts are broken down according to which 'stage' they map from:
# -Stage 0: raw game code
# -Stage 1: templated game description
# -Stage 2: natural language conversion of template
# -Stage 3: natural language description of game
###

SECTION_PROMPT_SUFFIX = """Now, convert the following description:
### INITIAL DESCRIPTION:
{0}

### CONVERTED DESCRIPTION:"""

SETUP_PROMPT = """Your task is to convert a templated description of a game's setup into a natural language description. Do not change the content of the template, but you may rewrite and reorder the information in any way you think is necessary in order for a human to understand it.
Use the following examples as a guide:
{0}"""

PREFERENCES_PROMPT = """Your task is to convert a templated description of a game's rules (expressed as "preferences") into a natural language description. Do not change the content of the template, but you may rewrite and reorder the information in any way you think is necessary in order for a human to understand it.
Use the following examples as a guide:
{0}"""

TERMINAL_PROMPT = """Your task is to convert a templated description of a game's terminal conditions into a natural language description. Do not change the content of the template, but you may rewrite and reorder the information in any way you think is necessary in order for a human to understand it.
Use the following examples as a guide:
{0}"""

SCORING_PROMPT= """Your task is to convert a templated description of a game's scoring conditions into a natural language description. Do not change the content of the template, but you may rewrite and reorder the information in any way you think is necessary in order for a human to understand it.
Use the following examples as a guide:
{0}"""

COMPLETE_GAME_PROMPT = """Your task is to combine and simplify the description of a game's rules. Do not change the content of the rules by either adding or removing information, but you may rewrite and reorder the information in any way you think is necessary in order for a human to understand it.
Use the following examples as a guide:
{0}

Now, convert the following rules:
### INITIAL RULES:
{1}

### SIMPLIFIED RULES:
"""


def compile_prompts_from_data(initial_stage: int,
                              final_stage: int,
                              translations_path: str):
    
    # Load in the data
    with open(translations_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)[1:]

    setup_content = ""
    preferences_content = ""
    terminal_content = ""
    scoring_content = ""

    # breakpoint()
    for idx, row in enumerate(data):

        print(idx, idx%5, row[0])

        # Identifier
        if idx%5 == 0:
            continue

        # Setup (optional)
        elif idx%5 == 1:
            stages = row[1:]
            if stages[final_stage] != "":
                setup_content += f"###INITIAL DESCRIPTION:\n{stages[initial_stage]}\n\n###CONVERTED DESCRIPTION:\n{stages[final_stage]}\n\n"

        # Preferences (required)
        elif idx%5 == 2:
            stages = row[1:]
            preferences_content += f"###INITIAL DESCRIPTION:\n{stages[initial_stage]}\n\n###CONVERTED DESCRIPTION:\n{stages[final_stage]}\n\n"

        # Terminal (optional)
        elif idx%5 == 3:
            stages = row[1:]
            if stages[final_stage] != "":
                terminal_content += f"###INITIAL DESCRIPTION:\n{stages[initial_stage]}\n\n###CONVERTED DESCRIPTION:\n{stages[final_stage]}\n\n"

        # Scoring (required)
        elif idx%5 == 4:
            stages = row[1:]
            scoring_content += f"###INITIAL DESCRIPTION:\n{stages[initial_stage]}\n\n###CONVERTED DESCRIPTION:\n{stages[final_stage]}\n\n"

    # Compile the prompts
    setup_prompt = SETUP_PROMPT.format(setup_content) + SECTION_PROMPT_SUFFIX
    preferences_prompt = PREFERENCES_PROMPT.format(preferences_content) + SECTION_PROMPT_SUFFIX
    terminal_prompt = TERMINAL_PROMPT.format(terminal_content) + SECTION_PROMPT_SUFFIX
    scoring_prompt = SCORING_PROMPT.format(scoring_content) + SECTION_PROMPT_SUFFIX

    return setup_prompt, preferences_prompt, terminal_prompt, scoring_prompt

if __name__ == '__main__':
    a, b, c, d = compile_prompts_from_data(1, 2, "./selected_human_and_map_elites_translations.csv")
    breakpoint()