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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def evaluate_dsl_generation(self, model, tokenizer, log_writer, global_step, num_evals, max_length, num_beams, temperature=1,
                                similarity_threshold=0.9):
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
        inputs = inputs.to(self.device)

        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_evals, do_sample=True,
                                 typical_p=0.5)
        
        all_samples = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for sample in all_samples:

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
                log_writer.add_text("eval/novel_valid_sample", sample, global_step)
            
        num_novel = output_results["novel"]["valid"] + output_results["novel"]["invalid"]
        num_existing = num_evals - num_novel

        num_valid = output_results["novel"]["valid"] + output_results["existing"]["valid"]
        num_invalid = num_evals - num_valid

        print(f"\nOut of {num_evals} generations, {num_novel} were novel (sim. threshold = {similarity_threshold}) and (independently) {num_valid} were valid")
        print(f"Of the {num_novel} novel generations, {output_results['novel']['valid']} were valid")
        print(f"Similarly, of the {num_valid} valid generations, {output_results['novel']['valid']} were novel")

        prop_novel = num_novel / num_evals
        prop_valid = num_valid / num_evals
        prop_novel_valid = output_results["novel"]["valid"] / num_evals

        log_writer.add_scalar("eval/prop_novel", prop_novel, global_step)
        log_writer.add_scalar("eval/prop_valid", prop_valid, global_step)
        log_writer.add_scalar("eval/prop_novel_valid", prop_novel_valid, global_step)

        return prop_novel, prop_valid, prop_novel_valid



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