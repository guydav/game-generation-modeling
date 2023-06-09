import os
import pickle
import random
import re
import sys
import torch
import pandas
import numpy as np
import torch.nn as nn

import re
import typing
import tatsu
import tatsu.ast
import tatsu.grammars

from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from transformers import AutoTokenizer

sys.path.insert(1, os.path.join(sys.path[0], '../src'))
from ast_utils import load_games_from_file, cached_load_and_parse_games_from_file
from ast_parser import ASTParentMapper


# Add src/ to our path so we can import from the scripts in room_and_object_types.py
# sys.path.insert(1, os.path.join(sys.path[0], '../src'))
# from room_and_object_types import get_room_contents
# from ast_utils import load_tests_from_file

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


class FitMDataset():
    def __init__(self,
                 tokenizer,
                 split="train",
                 chunk_size=1024,
                 train_split=0.8,
                 fim_prefix_token="<fim_prefix>",
                 fim_suffix_token="<fim_suffix>",
                 fim_middle_token="<fim_middle>",
                 cache_dir="./caches",
                 grammar_path="../dsl/dsl.ebnf",
                 game_file_path="../dsl/interactive-beta.pddl"):
        
        self.tokenizer = tokenizer
        self.split = split
        self.chunk_size = chunk_size

        self.grammar = open(grammar_path).read()
        self.grammar_parser = typing.cast(tatsu.grammars.Grammar, tatsu.compile(self.grammar))

        save_filename = os.path.join(cache_dir, f"fitm_dataset.pkl")

        if os.path.exists(save_filename):
            print("Loading cached dataset...")
            with open(save_filename, "rb") as f:
                all_data = pickle.load(f)

        else:
            games = [game for game in load_games_from_file(game_file_path)]
            game_asts = list(cached_load_and_parse_games_from_file(game_file_path, self.grammar_parser, False, relative_path='..'))

            parent_mapper = ASTParentMapper()

            all_data = []
            for game, ast in tqdm(zip(games, game_asts), total=len(games), desc="Preprocessing games"):
                parent_mapper(ast)
                
                node_keys = list(
                    key for key, node_info
                    in parent_mapper.parent_mapping.items()
                    if isinstance(node_info.parent, tatsu.ast.AST)
                )

                for idx, node_key in enumerate(node_keys):
                    node = parent_mapper.parent_mapping[node_key]
                    info = node.ast.parseinfo

                    prefix = game[:info.pos]
                    node_str = game[info.pos:info.endpos]
                    suffix = game[info.endpos:]

                    data_point = {
                        "full_ast": ast,
                        "node_key": node_key,
                        "rule": info.rule,
                        "prefix": f"{fim_prefix_token}{self._preprocess(prefix)}",
                        "suffix": f"{fim_suffix_token}{self._preprocess(suffix)}",
                        "middle": f"{fim_middle_token}{self._preprocess(node_str)}",
                    }

                    tokenized = tokenizer(data_point["prefix"] + data_point["suffix"] + data_point["middle"], max_length=chunk_size, truncation=True, padding="max_length", return_tensors=None)
                    tokenized["labels"] = tokenized["input_ids"].copy()
                    tokenized["labels"][tokenized["labels"] == tokenizer.pad_token_id] = -100

                    data_point.update(tokenized)

                    all_data.append(data_point)

            with open(save_filename, "wb") as f:
                pickle.dump(all_data, f)

        random.shuffle(all_data)
        split_idx = int(len(all_data) * train_split)

        self.train_data = all_data[:split_idx]
        self.test_data = all_data[split_idx:]

    def _preprocess(self, ast_str):
        '''
        Clean a string representation of an AST to make it more amenable to tokenization
        '''
        ast_str = re.sub(r"\s+", " ", ast_str)
        ast_str = re.sub(r"\s(?=[\)}])", "", ast_str)     # remove whitespace before closing brackets
        ast_str = re.sub(r"(?<=[\({])\s", "", ast_str)

        return ast_str

    def __len__(self):
        return len(self.train_data) if self.split == "train" else len(self.test_data)

    def __getitem__(self, idx):
        return self.train_data[idx] if self.split == "train" else self.test_data[idx] 

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained("bigcode/starcoder", use_auth_token="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")
    model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder", use_auth_token="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").to(device)
    dataset = FitMDataset(tok)

    input_text = dataset[0]["prefix"] + dataset[0]["suffix"]
    input_ids = tok.encode(input_text, return_tensors="pt").to(device)

    outputs = model.generate(input_ids)
    output_text = tok.decode(outputs[0], skip_special_tokens=True)
    print(f"Prefix: {dataset[0]['prefix']}")
    print(f"Suffix: {dataset[0]['suffix']}")
    print(f"Generated: {output_text}")
    print(f"Original: {dataset[0]['middle']}") 