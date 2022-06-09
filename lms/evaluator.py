import os
import sys
import tatsu
import torch

from tqdm import tqdm
from fuzzywuzzy import fuzz

# Add src/ to our path so we can import from the scripts in room_and_object_types.py
sys.path.insert(1, os.path.join(sys.path[0], '../src'))
from ast_utils import load_tests_from_file

class Evaluator():
    def __init__(self,
                 grammar_path="../dsl/dsl.ebnf",
                 dsl_info_path="../dsl/interactive-beta.pddl"):


        grammar = open(grammar_path).read()
        self.grammar_parser = tatsu.compile(grammar)
        self.programs = load_tests_from_file(dsl_info_path)

    def evaluate_dsl_generation(self, model, tokenizer, num_evals, max_length, num_beams, temperature,
                                top_k, top_p, typical_p, do_sample=True, similarity_threshold=0.9, 
                                verbose=False):
        '''
        Generate a number of samples from the model and evaluate their grammaticality and 
        novelty. Returns the proportion of samples that are valid under the grammar, as well
        as the proportion that are within some threshold of similarity to an existing program
        '''

        context = "[START] (define (game [ID])"

        output_results = {"novel": {"valid": 0,
                                    "invalid": 0},
                          "existing": {"valid": 0,
                                       "invalid": 0}}

        inputs = tokenizer(context, return_tensors="pt").input_ids
        inputs = inputs.to(model.device)

        # Doing generations one at a time lets us use tqdm, setting pad_token_id necessary to supress warnings
        outputs = torch.stack([model.generate(inputs, max_length=max_length, do_sample=do_sample, num_beams=num_beams,
                               temperature=temperature, top_k=top_k, top_p=top_p, typical_p=typical_p,
                               pad_token_id=tokenizer.eos_token_id)[0] 
                               for _ in tqdm(range(num_evals), desc="Generating samples", leave=verbose)], dim=0)
        
        all_samples = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        novel_valid_samples = []

        for sample in tqdm(all_samples, desc="Evaluating samples", leave=verbose):

            # Keep up until the first time [END] is produced, and remove the [START] tokens
            sample = sample[:sample.find("[END]")].replace("[START]", "").replace("[ID]", "id-1").strip()

            similarity = max([fuzz.partial_ratio(sample, program) / 100 for program in self.programs])
            novelty_key = "novel" if similarity < similarity_threshold else "existing"

            try:
                self.grammar_parser.parse(sample)
                validity_key = "valid"
            except:
                validity_key = "invalid"

            output_results[novelty_key][validity_key] += 1

            if novelty_key == "novel" and validity_key == "valid":
                novel_valid_samples.append(sample)
            
        num_novel = output_results["novel"]["valid"] + output_results["novel"]["invalid"]
        num_existing = num_evals - num_novel

        num_valid = output_results["novel"]["valid"] + output_results["existing"]["valid"]
        num_invalid = num_evals - num_valid

        if verbose:
            print(f"\nOut of {num_evals} generations, {num_novel} were novel (sim. threshold = {similarity_threshold}) and (independently) {num_valid} were valid")
            print(f"Of the {num_novel} novel generations, {output_results['novel']['valid']} were valid")
            print(f"Similarly, of the {num_valid} valid generations, {output_results['novel']['valid']} were novel")

        prop_novel = num_novel / num_evals
        prop_valid = num_valid / num_evals
        prop_novel_valid = output_results["novel"]["valid"] / num_evals

        return prop_novel, prop_valid, prop_novel_valid, novel_valid_samples



if __name__ == "__main__":
    test = '''(define (game id-1) (:domain medium-objects-room-v1)
(:setup (and
    (exists (?h - hexagonal_bin)
        (game-conserved (< (distance?h room_center) 1))
    )
))
(:constraints (and
    (preference throwToRampToBin
        (exists (?r - triangular_ramp?d - dodgeball?h - hexagonal_bin)
            (then
                (once (and (agent_holds?d) (adjacent agent door) (agent_crouches)))
                (hold-while
                   (and (not (agent_holds?d)) (in_motion?d))
                   (touch?r?d)
                )
                (once  (and (in?h?d) (not (in_motion?d))))
           )
        )
    )
))
(:scoring maximize
    (count-unique-positions throwToRampToBin)
))
'''
    evaluator = Evaluator()
    evaluator.grammar_parser.parse(test)

    # for program in evaluator.programs:
    #     try:
    #         evaluator.grammar_parser.parse(program + "!!!")
    #     except:
    #         print("Error!")