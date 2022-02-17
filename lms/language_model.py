import os
import torch
import logging
import numpy as np
from tqdm import tqdm

from torch.optim import AdamW
from lm_sampler import LMSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2LMHeadModel
from transformers import get_linear_schedule_with_warmup

class GPT2LanguageModel():
    def __init__(self,
                 logdir="./logs"):

        self.logdir = logdir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ngpu = torch.cuda.device_count()

        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.to(self.device)

        self.log_writer = SummaryWriter(logdir, flush_secs=100)

    def fit(self,
            dataset,
            batch_size, 
            num_epochs,
            learning_rate, 
            weight_decay, 
            warmup_proportion,
            val_freq=20,
            val_temperature=1):

        '''
        Fit the language model to the provided dataset
        '''

        # Make sure that we can train embeddings for all of the new tokens
        self.model.resize_token_embeddings(len(dataset.tokenizer))
        self.model = torch.nn.DataParallel(self.model)

        # This piece of code is inherited from Project RECON and is responsible for not
        # applying weight decay to the parameters of the LayerNorm layers
        optimizable_param_list = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in optimizable_param_list if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in optimizable_param_list if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, weight_decay=weight_decay)

        train_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        tokenizer = dataset.tokenizer
        pad_token_id = dataset.pad_token_id

        # Create the sampler to generate continuations
        sampler = LMSampler(self.model, tokenizer, device=self.device)

        # We calculate the number of steps in training to set up the Scheduler
        num_train_steps = (len(train_data_loader) // batch_size) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(warmup_proportion * num_train_steps),
            num_training_steps=num_train_steps)


        global_step = 0
        self.model.train()
        for epoch in range(num_epochs):
            with tqdm(enumerate(train_data_loader), desc=f"Training epoch {epoch+1} of {num_epochs}",
                      total=len(train_data_loader)) as pbar:

                for idx, batch in pbar:
                    token_ids = batch.to(self.device)

                    # By default, the ignore index is -100, and we want ot ignore all of the pad tokens
                    labels = token_ids.clone().detach()
                    labels[labels == pad_token_id] = -100

                    # Meanwhile, we turn pad tokens in the input into 0, so that we don't
                    # have to use an uninitialized pad-token embedding
                    token_ids[token_ids == pad_token_id] = 0

                    loss = self.model(token_ids, labels=labels)[0]

                    if self.ngpu > 1:
                        loss = loss.mean()

                    perplexity = torch.exp(loss).item()

                    # Clear some memory before the expensive gradient computation
                    del token_ids
                    del labels

                    # Perform optimization and update the scheduler
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    self.log_writer.add_scalar("train/loss", loss, global_step)
                    self.log_writer.add_scalar("train/perplexity", perplexity, global_step)

                    pbar.set_postfix({"pplx": perplexity})

                    global_step += 1
                    if global_step%val_freq == 0:
                        self.generate_continuation(sampler, global_step, generation_length=256, temperature=1.5)

                    del loss

    def generate_continuation(self, sampler, global_step, generation_length=200, condition="[SETUP]: ", temperature=1):
        '''
        Generate a continution of the provided conditioning string of the specified length
        '''

        # Switch to evaluation mode
        self.model.eval()

        sample = sampler.generate(condition, length=generation_length, temperature=temperature)
        print("\nSample generated at step {global_step}")
        print(sample)

        self.log_writer.add_text("eval_sample", sample, global_step)

        # Return to training mode
        self.model.train()

