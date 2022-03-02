import os
import shutil
import argparse
from datetime import datetime
from lm_sampler import LMSampler
from language_model import GPT2LanguageModel
from torch.utils.tensorboard import SummaryWriter
from lm_datasets import GameDescriptionGPT2Dataset, DomainSpecificLanguageLMDataset

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

class CustomLoggingTrainer(Trainer):
    '''
    Subclass of the standard transformers Trainer class that adds custom logging behavior: namely logging
    the loss and learning rate at every step and then sampling a generation every k steps
    '''
    def __init__(self, model=None, args=None, data_collator=None, train_dataset=None, eval_dataset=None, tokenizer=None,
                 model_init=None, compute_metrics=None, callbacks=None, optimizers=(None, None), generation_freq=None, 
                 generation_length=None, generation_context=None, generation_temp=None):

        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics, callbacks, optimizers)

        self.log_writer = SummaryWriter(args.output_dir)
        self.sampler = LMSampler(model, data_collator.tokenizer, device="cpu")

        self.generation_freq = generation_freq
        self.generation_length = generation_length
        self.generation_context = generation_context
        self.generation_temp = generation_temp

    def log(self, logs):
        '''
        Write the logs to tensorboard, including less frequent sample logs
        '''
        self.log_writer.add_scalar("train/loss", logs["loss"], self.state.global_step)
        self.log_writer.add_scalar("train/lr", logs["learning_rate"], self.state.global_step)

        if self.state.global_step % self.generation_freq == 0:
            self.model.eval()
            sample = self.sampler.generate(condition_text=self.generation_context, length=self.generation_length,
                                           temperature=self.generation_temp)
            self.log_writer.add_text("eval_sample", sample, self.state.global_step)
            self.model.train()

        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default="CodeBERTa", choices=["CodeBERTa", "gpt2", "codeparrot"])
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
    parser.add_argument('--gen_freq', type=int, default=98)
    parser.add_argument('--gen_len', type=int, default=100)
    parser.add_argument('--gen_context', type=str, default="(define")
    parser.add_argument('--gen_temp', type=float, default=1)

    args = parser.parse_args()

    # Map from model names to the load string transformers expects
    model_mapping = {"CodeBERTa": "huggingface/CodeBERTa-small-v1",
                     "gpt2": "gpt2",
                     "codeparrot": "lvwerra/codeparrot"}

    # Map from dataset names to the class for that dataset
    dataset_mapping = {"dsl": DomainSpecificLanguageLMDataset,
                       "descs": GameDescriptionGPT2Dataset}

    # Set parallelism to false to silence deadlock warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Instantiate the tokenizer and dataset based on the model's name
    model_name = model_mapping[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "PAD"})

    # # Special pad token for GPT2:
    # if args.model == "gpt2":

    dataset = dataset_mapping[args.dataset](tokenizer, args.chunk_size)

    # Instantiate the model and data collator depending on whether the model uses masked or causal LMs
    uses_mlm = (args.model in ["CodeBERTa"])
    if uses_mlm:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Create the output directory
    datetime_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    output_dir_name = f"./logs/{datetime_str}-{args.model}-{args.dataset}"
    if not os.path.exists(output_dir_name):
        os.mkdir(output_dir_name)

    training_args = TrainingArguments(
        output_dir=output_dir_name,
        logging_dir=output_dir_name,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        save_strategy="epoch",
        logging_steps=1,
        save_total_limit=1)

    trainer = CustomLoggingTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        generation_freq=args.gen_freq,
        generation_length=args.gen_len,
        generation_context=args.gen_context,
        generation_temp=args.gen_temp)

    trainer.train()

    # dataset = GameDescriptionGPT2Dataset(chunk_size=args.chunk_size)
    # model = GPT2LanguageModel(args.room_mode)

    # model.fit(dataset=dataset,
    #           batch_size=args.batch_size,
    #           num_epochs=args.epochs,
    #           learning_rate=args.learning_rate,
    #           weight_decay=args.weight_decay,
    #           warmup_proportion=args.warmup_proportion,
    #           val_temperature=args.val_temperature)