import argparse
import copy
import json
import os
import shutil

from peft import prepare_model_for_int8_training, LoraConfig, TaskType, get_peft_model, PeftModel, set_peft_model_state_dict
from pynvml import *
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import (LlamaForCausalLM, GPTBigCodeForCausalLM)

'''
GPU utilization helper functions from https://huggingface.co/docs/transformers/perf_train_gpu_one#load-model
'''

def gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used//1024**2

MODEL_MAPPING = {"gpt2": "gpt2",
                 "gpt2-large": "gpt2-large",
                 "gpt2-xl": "gpt2-xl",
                 "gpt-j": "EleutherAI/gpt-j-6B",
                 "opt-350m": "facebook/opt-350m",
                 "opt-1.3b": "facebook/opt-1.3b",
                 "opt-2.7b": "facebook/opt-2.7b",
                 "opt-6.7b": "facebook/opt-6.7b",
                 "opt-13b": "facebook/opt-13b",
                 "opt-66b": "facebook/opt-66b",
                 "stablelm-3b": "StabilityAI/stablelm-base-alpha-3b",
                 "stablelm-7b": "StabilityAI/stablelm-base-alpha-7b",
                 "replit": "replit/replit-code-v1-3b",
                 "llama": os.environ.get("LLAMA_HF_PATH"),
                 "alpaca": os.environ.get("ALPACA_HF_PATH"),
                 "starcoder": "bigcode/starcoder"}

def load_model_and_tokenizer(args: argparse.Namespace):
    
    """
    Load an instance of a language model, either from a pretrained model or a checkpoint, and
    apply the appropriate configurations (either user specified or loaded from the checkpoint)

    Returns the model, its associated tokenizer, and the updated args
    """

    args = copy.deepcopy(args)

    # If specified, load arguments from checkpoint
    if args.resume:
        args.__dict__ = json.load(open(os.path.join(args.save_dir, "args.json"), "r"))
        args.__dict__['resume'] = True

        # Find the most recent checkpoint
        checkpoint_dirs = [os.path.join(args.save_dir, d) for d in os.listdir(args.save_dir) if 
                           d.startswith("checkpoint") and os.path.isdir(os.path.join(args.save_dir, d))]
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
        load_dir = checkpoint_dirs[-1]

        print(f"Loading {args.model} model from checkpoint {load_dir}...")

    else:
        print(f"Loading {args.model} model from HuggingFace hub...")
        if os.path.exists(args.save_dir):
            print(f"WARNING: {args.save_dir} already exists. Overwriting...")
            shutil.rmtree(args.save_dir)
            os.makedirs(args.save_dir)

    model_name = MODEL_MAPPING.get(args.model, args.model)

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              use_fast=("llama" not in args.model), # fast LlamaTokenizer causes error
                                              trust_remote_code=args.trust_remote) 

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=args.trust_remote)
    model = AutoModelForCausalLM.from_pretrained(model_name if (not args.resume) or args.lora else load_dir, 
                                                 config=config, 
                                                 load_in_8bit=args.int_8, 
                                                 torch_dtype=torch.float16 if args.int_8 else "auto", 
                                                 device_map="auto",
                                                 trust_remote_code=args.trust_remote)

    # Weird issue for device placement on GPT models when using lora, int-8, and mixed precision

    if tokenizer.pad_token is None:
        print("Adding pad token to tokenizer...")
        tokenizer.add_special_tokens({"pad_token": "PAD"})
        model.resize_token_embeddings(len(tokenizer))

    if tokenizer.bos_token is None:
        print("Adding bos token to tokenizer...")
        tokenizer.add_special_tokens({"bos_token": "BOS"})
        model.resize_token_embeddings(len(tokenizer))

    if args.int_8:
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        assert args.mixed_precision in ["fp16", "bf16"], "8-bit training requires mixed precision"

    if args.lora:

        # Some models have specific target module requirements
        if isinstance(model, LlamaForCausalLM):
            lora_target_modules = ["q_proj", "v_proj"]
        elif isinstance(model, GPTBigCodeForCausalLM):
            lora_target_modules = ["c_proj", "c_attn", "q_attn"]
        elif "MPTForCausalLM" in str(type(model)):
            lora_target_modules = ["Wqkv"]
        else:
            lora_target_modules = None

        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                    inference_mode=False,
                                    r=args.lora_r,
                                    lora_alpha=args.lora_alpha,
                                    lora_dropout=args.lora_dropout,
                                    target_modules=lora_target_modules)
        
        model = get_peft_model(model, peft_config)

        if args.resume:
            # Try to find a state-dict to load called either "pytorch_model.bin" or "adapter_model.bin"
            if os.path.exists(os.path.join(load_dir, "pytorch_model.bin")):
                adapter_weights = torch.load(os.path.join(load_dir, "pytorch_model.bin"))
            elif os.path.exists(os.path.join(load_dir, "adapter_model.bin")):
                adapter_weights = torch.load(os.path.join(load_dir, "adapter_model.bin"))
            else:
                raise ValueError(f"Could not find a state-dict for PEFT to load in {load_dir}")

            set_peft_model_state_dict(model, adapter_weights)

    # Enable gradient checkpointing (in the case where it wasn't already turned on for int_8 models)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        print("Disabling the model cache...")
        model.gradient_checkpointing_disable()
        model.config.use_cache = False

    if args.jit:
        if torch.__version__ >= "2" and sys.platform != 'win32':
            model = torch.compile(model)
        else:
            print("JIT compilation is only supported on PyTorch 2.0+ and Linux / MacOS")

    return model, tokenizer, args

def tokenize(text, tokenizer, max_length=None, mask_pad_tokens=True):
    if isinstance(text, dict) and "text" in text.keys():
        text = text["text"]
    text = f"{tokenizer.bos_token}{text}{tokenizer.eos_token}"

    if max_length is not None:
        max_length = min(max_length, tokenizer.model_max_length)
        result = tokenizer(text, max_length=max_length, truncation=True, padding="max_length", return_tensors=None)
    else:
        result = tokenizer(text, return_tensors=None)

    result["labels"] = result["input_ids"].copy()

    if mask_pad_tokens:
        result["labels"][result["labels"] == tokenizer.pad_token_id] = -100

    result.pop("token_type_ids", None)

    return result