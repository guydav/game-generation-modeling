import argparse
import json
import os
os.environ["BITSANDBYTES_NOWELCOME"] = "true"

import torch
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, PrinterCallback, Trainer, TrainingArguments, TrainerCallback, set_seed

from utils import *


class CustomTQDMCallback(TrainerCallback):
    def __init__(self, model_name):
        self.model_name = model_name
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.progress_bar = tqdm(total=state.max_steps, desc=f"Training {self.model_name}")
        self.progress_bar.update(state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        self.progress_bar.update(1)

        if len(state.log_history) > 0:
            most_recent_logs = state.log_history[-1].copy()
            most_recent_logs.pop("step", None)
            most_recent_logs["GPU"] = f"{gpu_utilization()}MB"

            self.progress_bar.set_postfix(most_recent_logs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument('--model', type=str, default='gpt2', help="The name of the model to use")
    parser.add_argument('--int_8', action='store_true', help="Whether to use 8-bit training")
    parser.add_argument('--jit', action='store_true', help="Whether to compile the model before training")
    parser.add_argument('--lora', action='store_true', help="Whether to use low-rank adpatation for finetuning")
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--accelerate', action='store_true', help="Whether to use the accelerate library for distributed training")
    parser.add_argument('--trust_remote', action='store_true', help="Whether to trust remote models (necessary for some models)")

    # LoRA arguments
    parser.add_argument('--lora_alpha', type=int, default=16, help="The alpha parameter for LoRA")
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="The dropout rate for LoRA")
    parser.add_argument('--lora_r', type=int, default=8, help="The r parameter for LoRA")
    parser.add_argument('--lora_target_modules', type=str, nargs='+', default=['q_proj', 'v_proj'], help="The modules to apply LoRA to")

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument('--optimizer_type', type=str, default='adamw', choices=['adamw', 'adafactor', 'adamw_8bit'])
    parser.add_argument('--warmup_steps', type=int, default=100, help="Number of steps to warm up the learning rate for")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="fitm", choices=["fitm"])
    parser.add_argument("--data_categories", type=str, nargs='+', default=["board"])
    parser.add_argument("--data_subcategories", type=str, nargs='+', default="")
    parser.add_argument("--keep_formatting", action='store_true', help="Whether to keep the formatting of the dataset")
    parser.add_argument("--section_exclude_keywords", type=str, nargs='+', default=["(metadata", "(option"], help="Keywords to exclude sections by")

    # Run arguments
    parser.add_argument("--gen_freq", type=int, default=100, help="How often to generate text during training")
    parser.add_argument("--loss_window", type=int, default=100, help="Number of steps to average reported loss over")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs to train for (unless num_train_steps is set)")
    parser.add_argument('--num_train_steps', type=int, default=-1, help="Number of training steps to perform. If -1, will run for num_epochs")
    parser.add_argument('--resume', action='store_true', help="Whether to resume training from a checkpoint")
    parser.add_argument('--save_dir', type=str, default='./logs/llm_test', help="Directory to save the model (or load, if resuming training)")
    parser.add_argument('--save_freq', type=int, default=100, help="How often to save the model during training")
    parser.add_argument('--no_save', action='store_true', help="Whether to save the model during training")
    parser.add_argument('--seed', type=int, default=42, help="Random seed to use")

    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"

    # Set parallelism to false to silence tokenizer deadlock warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"    

    # Load the model / tokenizer, and update args if resuming from checkpoint
    model, tokenizer, args = load_model_and_tokenizer(args)

    # Set the random seed
    set_seed(args.seed)

    # Save the run args
    if not args.no_save:
        os.makedirs(args.save_dir, exist_ok=True)
        json.dump(args.__dict__, open(os.path.join(args.save_dir, "args.json"), "w"))

    print(f"GPU utilization after model load: {gpu_utilization()} MB")
    if args.lora:
        model.print_trainable_parameters()

    # Set the training arguments
    training_args=TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=(args.mixed_precision == 'fp16'),
        logging_steps=1,
        optim="adamw_torch",
        save_strategy="steps" if not args.no_save else "no",
        save_steps=args.save_freq,
        output_dir=args.save_dir,
        save_total_limit=3,
        disable_tqdm=True,
        group_by_length=True,
    )

    # Set the dataset
    if args.dataset == "fitm":
        pass

    else:
        raise ValueError(f"Dataset {args.dataset} not recognized")

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        callbacks=[CustomTQDMCallback(model_name=args.model)],
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    )

    # Remove the default PrinterCallback (since we're rolling its functionality into our CustomTQDMCallback)
    trainer.remove_callback(PrinterCallback)

    trainer.train(resume_from_checkpoint=args.resume)

    if not args.no_save:
        model.save_pretrained(args.save_dir)