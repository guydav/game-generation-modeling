import os
import argparse
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
            sample = self.sampler.generate(self.generation_context, self.generation_length, self.generation_temp)
            self.log_writer.add_text("eval_sample", sample, self.state.global_step)
            self.model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--chunk_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--warmup_proportion', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--room_mode', type=str, default="naive", choices=["naive", "categories", "colors"])
    parser.add_argument('--gen_freq', type=int, default=10)
    parser.add_argument('--gen_len', type=int, default=100)
    parser.add_argument('--gen_context', type=str, default="(define")
    parser.add_argument('--gen_temp', type=float, default=1)

    args = parser.parse_args()

    # Set parallelism to false to silence deadlock warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
    model = AutoModelForMaskedLM.from_pretrained("huggingface/CodeBERTa-small-v1")
    
    dataset = DomainSpecificLanguageLMDataset(tokenizer, args.chunk_size)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)


    training_args = TrainingArguments(
        output_dir="./logs",
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