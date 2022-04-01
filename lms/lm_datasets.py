import os
import re
import sys
import torch
import pandas
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

# Add src/ to our path so we can import from the scripts in room_and_object_types.py
sys.path.insert(1, os.path.join(sys.path[0], '../src'))
from room_and_object_types import get_room_contents
from ast_utils import load_tests_from_file

class GameDescriptionGPT2Dataset(Dataset):
    def __init__(self,
                 room_contents_mode="naive",
                 chunk_size=1024,
                 csv_file="../data/interactive_beta.csv"):

        self.room_contents = get_room_contents(room_contents_mode)

        self.chunk_size = chunk_size

        # Initialize the GPT2 tokenizer and add a custom PAD token
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({"pad_token": "PAD"})
        self.pad_token_id = self.tokenizer.pad_token_id

        game_token_ids = []

        game_data_df = pandas.read_csv(csv_file)
        game_descriptions = list(zip(game_data_df["scene"], game_data_df["game_setup"], game_data_df["game_gameplay"], 
                                     game_data_df["game_scoring"]))
        
        for idx, (room, setup, gameplay, scoring) in tqdm(enumerate(game_descriptions), desc="Tokenizing games", total=len(game_descriptions)):

            contents_tokens = self.tokenizer.encode(self.room_contents[room])
            setup_tokens = self.tokenizer.encode("\n[SETUP]: " + ("None" if str(setup) == "nan" else setup))
            gameplay_tokens = self.tokenizer.encode("\n[GAMEPLAY]: " + gameplay)
            scoring_tokens = self.tokenizer.encode("\n[SCORING]: " + scoring)

            game_encoding = contents_tokens + setup_tokens + gameplay_tokens + scoring_tokens
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

class DomainSpecificLanguageLMDataset(Dataset):
    '''
    Dataset wrapper for DSL example generation using masked LMs. Because we are testing a
    variety of different pre-trained models, the code here takes a passed in tokenizer and
    makes no assumptions about masking
    '''
    def __init__(self,
                 tokenizer,
                 chunk_size=1024,
                 dsl_info_path="../dsl/interactive-beta.pddl",
                 csv_file="../data/interactive_beta.csv"):


        game_data_df = pandas.read_csv(csv_file)
        participant_ids = game_data_df["participantID"]

        self.tokenizer = tokenizer
        self.programs = []
        for program in load_tests_from_file(dsl_info_path):
            # Find the PID / corresponding index into the game info dataframe
            pid, index = re.search("\(game(.*)\) \(", program).group(1).strip().split("-")
            
            # Mask out that information before tokenizing
            program = program.replace(f"{pid}-{index}", "[ID]")

            program = "[START] " + program + " [END]"

            encoded_program = self.tokenizer.encode(program, max_length=chunk_size, truncation=True, padding="max_length")
            self.programs.append(encoded_program)

    def __len__(self):
        return len(self.programs)

    def __getitem__(self, idx):
        tensor = torch.tensor(self.programs[idx], dtype=torch.long)
        return tensor

class DescriptionToDSLDataset(Dataset):
    '''
    Dataset wrapper for sequence-to-sequence modeling mapping the natural language descriptions to the DSL
    '''
    def __init__(self,
                 tokenizer,
                 chunk_size=1024,
                 room_contents_mode=None,
                 dsl_info_path="../dsl/interactive-beta.pddl",
                 csv_file="../data/interactive_beta.csv"):


        self.tokenizer = tokenizer

        if room_contents_mode is not None:
            self.room_contents = get_room_contents(room_contents_mode)

        game_data_df = pandas.read_csv(csv_file)
        game_descriptions = list(zip(game_data_df["scene"], game_data_df["game_setup"], game_data_df["game_gameplay"], 
                                     game_data_df["game_scoring"]))
        participant_ids = game_data_df["participantID"]

        self.paired_data = []
        for program in load_tests_from_file(dsl_info_path):
            # Find the PID / corresponding index into the game info dataframe
            pid, index = re.search("\(game(.*)\) \(", program).group(1).strip().split("-")
            
            # Mask out that information before tokenizing
            program = program.replace(f"{pid}-{index}", "[ID]")

            # Collect the natural language game information
            room, setup, gameplay, scoring = game_descriptions[int(index)]

            encoded_program = self.tokenizer.encode(program, max_length=chunk_size, truncation=True, padding="max_length")

            description_text = ""
            if room_contents_mode is not None:
                description_text += "[ROOM] " + self.room_contents[room] + "\n"

            if str(setup) != "nan":
                description_text += "[SETUP]: " + setup + "\n"

            description_text += "[GAMEPLAY]: " + gameplay + "\n"
            description_text += "[SCORING]: " + scoring

            encoded_description = self.tokenizer.encode(description_text, max_length=chunk_size, truncation=True, padding="max_length")

            self.paired_data.append((encoded_description, encoded_program))

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, idx):
        description, program = self.paired_data[idx]
        combined_tensor = torch.stack([torch.tensor(description, dtype=torch.long),
                                       torch.tensor(program, dtype=torch.long)], dim=0)

        return combined_tensor


if __name__ == "__main__":
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
    d = DescriptionToDSLDataset(t)
    # dataset = GameDescriptionGPT2Dataset("colors")
    # ids = dataset[22]
    # decode = dataset.decode_ids(ids)
    # print(decode)