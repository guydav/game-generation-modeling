import os
import glob
import json
import torch
import argparse

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from evaluator import Evaluator

# Map from model names to the load string transformers expects
model_mapping = {"CodeBERTa": "huggingface/CodeBERTa-small-v1",
                 "gpt2": "gpt2",
                 "codeparrot": "lvwerra/codeparrot",
                 "java-gpt2": "microsoft/CodeGPT-small-java-adaptedGPT2",
                 "codet5": "Salesforce/codet5-base",
                 "incoder-1B": "facebook/incoder-1B",
                 "incoder-6B": "facebook/incoder-6B"}

# Instantiate the evaluator
evaluator = Evaluator()

log_dir = "./logs"
model_log_dirs = os.listdir(log_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_log_dir in model_log_dirs:

    with open(os.path.join(log_dir, model_log_dir, "config.json"), "r") as file:
        config = json.load(file)

    # Check to make sure the model was trained on just the DSL (for now)
    if not config["dataset"] == "dsl":
        continue

    # Check if the run has saved checkpoints, and collect the latest one if so
    checkpoints = glob.glob(os.path.join(log_dir, model_log_dir, "*.pth"))
    if len(checkpoints) == 0:
        continue

    final_checkpoint = list(sorted(checkpoints))[-1]

    # Determine the model name / type, and instantiate it
    model_name = model_mapping[config["model"]]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "PAD"})

    uses_mlm = (config["model"] in ["CodeBERTa"])
    is_seq_to_seq = (config["model"] in ["codet5"])

    if uses_mlm:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    elif is_seq_to_seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)

    # Load the model from the checkpoint
    print(f"Loading model weights from {final_checkpoint}...")
    model.load_state_dict(torch.load(final_checkpoint))

    evaluator.evaluate_dsl_generation(model, tokenizer, 10, max_length=1024, num_beams=5, temperature=1, top_k=50,
                                      top_p=1.0, typical_p=1.0, do_sample=True, similarity_threshold=0.9)
