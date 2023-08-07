import json
import os

import backoff
import openai

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def query_openai_api(prompt: str, model_str: str, max_tokens: int, temperature: float):
    '''
    Query the specified openai model with the given prompt, and return the response. Assumes
    that the API key has already been set. Retries with exponentially-increasing delays in 
    case of rate limit errors
    '''
    response = openai.ChatCompletion.create(
        model=model_str,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    reponse_content = response["choices"][0]["message"]["content"]

    return reponse_content

openai.api_key = os.environ["OPENAI_TOKEN"]

PROMPT = """Convert the following templated description of a game's rules into a natural language description. Make sure not to change the content of the template, but feel free to add any additional information you think is necessary in order for a human to understand it."""

GAME_DESCRIPTION  = """
Describing preference: blockInTowerKnockedByDodgeball
The variables required by this preference are:
 - ?b: of type building
 - ?c: of type cube_block
 - ?d: of type dodgeball
 - ?h: of type hexagonal_bin
 - agent: of type agent

[0] First, we need a single state where (the agent is holding ?d), (?b is on ?h), and (?c is inside of ?b).

[1] Next, we need a sequence of states where (it's not the case that the agent is holding ?d) and (?d is in motion). During this sequence, we need a state where ((?c touches ?d) or (there exists an object ?c2 of type cube_block, such that ?c2 touches ?c)) and a state where (?c is in motion) (in that order).

[2] Finally, we need a single state where it's not the case that ?c is in motion.
"""

response = query_openai_api(f"{PROMPT}\n\n###SCENE: {GAME_DESCRIPTION}", "gpt-3.5-turbo", 250, 0)

print(f"Initial templated game description:\n{GAME_DESCRIPTION}\n\n")
print(f"Rewritten description:\n{response}")