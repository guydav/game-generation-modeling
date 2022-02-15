import os
import torch
import pandas
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


class GameDescriptionGPT2Dataset(Dataset):
    def __init__(self,
                 chunk_size=512,
                 csv_file="../data/interactive_beta.csv"):

        self.chunk_size = chunk_size

        # Initialize the GPT2 tokenizer and add a custom PAD token
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({"pad_token": "PAD"})
        self.pad_token_id = self.tokenizer.pad_token_id

        game_token_ids = []

        game_data_df = pandas.read_csv(csv_file)
        game_descriptions = list(zip(game_data_df["game_setup"], game_data_df["game_gameplay"], game_data_df["game_scoring"]))
        
        for idx, (setup, gameplay, scoring) in tqdm(enumerate(game_descriptions), desc="Tokenizing games", total=len(game_descriptions)):

            setup_tokens = self.tokenizer.encode("[SETUP]: " + ("None" if str(setup) == "nan" else setup))
            gameplay_tokens = self.tokenizer.encode("\n[GAMEPLAY]: " + gameplay)
            scoring_tokens = self.tokenizer.encode("\n[SCORING]: " + scoring)

            game_encoding = setup_tokens + gameplay_tokens + scoring_tokens
            if len(game_encoding) < self.chunk_size:
                game_encoding += [self.pad_token_id] * (self.chunk_size - len(game_encoding))

            game_token_ids += game_encoding

            self.game_token_ids = np.array(game_token_ids, dtype=np.int32)

    def decode_ids(self, token_ids):
        '''
        Convert the list of provided GPT2 token ids to a string and return it
        '''
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

        return text

    def __getitem__(self, idx):
        start, end = self.chunk_size * idx, self.chunk_size * (idx+1)
        return torch.tensor(self.game_token_ids[start:end], dtype=torch.long)

    def __len__(self):
        return len(self.game_token_ids) // self.chunk_size

if __name__ == "__main__":
    dataset = GameDescriptionGPT2Dataset()
    ids = dataset[22]
    decode = dataset.decode_ids(ids)
    print(decode)