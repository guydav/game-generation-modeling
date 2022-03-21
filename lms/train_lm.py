import os
import torch
import shutil
import argparse
from tqdm import tqdm
from datetime import datetime
from lm_sampler import LMSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from lm_datasets import GameDescriptionGPT2Dataset, DomainSpecificLanguageLMDataset

from transformers import pipeline
from transformers import get_linear_schedule_with_warmup
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

def train_loop(model, tokenizer, optimizer, data_loader, output_dir, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Map the model to the available device
    model.to(device)

    # Initialize the log writer
    log_writer = SummaryWriter(output_dir, flush_secs=100)

    # Create the sampler to generate continuations during training
    sampler = LMSampler(model, tokenizer, device=device)

    # Calculate the total number of training steps to initialize the scheduler
    num_train_steps = (len(data_loader) // args.batch_size) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_proportion * num_train_steps),
        num_training_steps=num_train_steps)

    # Set up for training
    model.train()
    global_step = 0
    progress_bar = tqdm(total=num_train_steps, desc=f"Training {args.model} model")

    try:
        for epoch in range(args.epochs):
            for batch in data_loader:
                global_step += 1

                token_ids = batch["input_ids"].to(device)

                # By default, the ignore index is -100, and we want to ignore all of the pad tokens
                labels = token_ids.clone().detach()
                labels[labels == tokenizer.pad_token_id] = -100

                loss = model(token_ids, labels=labels)[0]
                perplexity = torch.exp(loss).item()

                # Clear some memory before the expensive gradient computation
                del token_ids
                del labels

                # Perform optimization and update the scheduler
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                log_writer.add_scalar("train/loss", loss, global_step)
                log_writer.add_scalar("train/perplexity", perplexity, global_step)

                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item()})

                if global_step%args.gen_freq == 0:
                    model.eval()
                    sample = sampler.generate(condition_text=args.gen_context, length=args.gen_len, temperature=args.gen_temp)
                    print("\nSample: ", sample, "\n")
                    log_writer.add_text("eval_sample", sample, global_step)
                    model.train()

                    print("Another generation?")
                    inputs = tokenizer(args.gen_context, return_tensors="pt").input_ids
                    outputs = model.generate(inputs, max_length=args.gen_len, num_beams=5)[0]
                    print(tokenizer.decode(outputs, skip_special_tokens=True))

    except KeyboardInterrupt:
        print("Stopping early due to user input!")
        progress_bar.close()
        exit()

    print("Finished training.")
    progress_bar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default="CodeBERTa", choices=["CodeBERTa", "gpt2", "codeparrot", "java-gpt2"])
    parser.add_argument('--dataset', type=str, default="dsl", choices=["dsl", "descs"])
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--chunk_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--warmup_proportion', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--room_mode', type=str, default="naive", choices=["naive", "categories", "colors"])
    parser.add_argument('--gen_freq', type=int, default=200)
    parser.add_argument('--gen_len', type=int, default=256)
    parser.add_argument('--gen_context', type=str, default="(define")
    parser.add_argument('--gen_temp', type=float, default=1)

    args = parser.parse_args()

    # Map from model names to the load string transformers expects
    model_mapping = {"CodeBERTa": "huggingface/CodeBERTa-small-v1",
                     "gpt2": "gpt2",
                     "codeparrot": "lvwerra/codeparrot",
                     "java-gpt2": "microsoft/CodeGPT-small-java-adaptedGPT2"}

    # Map from dataset names to the class for that dataset
    dataset_mapping = {"dsl": DomainSpecificLanguageLMDataset,
                       "descs": GameDescriptionGPT2Dataset}

    # Set parallelism to false to silence deadlock warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Instantiate the tokenizer and dataset based on the model's name
    model_name = model_mapping[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "PAD"})

    dataset = dataset_mapping[args.dataset](tokenizer, args.chunk_size)

    # Instantiate the model and data collator depending on whether the model uses masked or causal LMs
    uses_mlm = (args.model in ["CodeBERTa"])
    if uses_mlm:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    data_loader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Create the output directory
    datetime_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    output_dir_name = f"./logs/{datetime_str}-{args.model}-{args.dataset}"
    if not os.path.exists(output_dir_name):
        os.mkdir(output_dir_name)


    train_loop(model, tokenizer, optimizer, data_loader, output_dir_name, args)