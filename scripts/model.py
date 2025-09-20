from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os
try:
    import huggingface_hub 
    huggingface_hub.login("hf_xxx")  # replace hf_xxx with your actual token
except:
    pass

def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    local_path = os.path.join(cache_dir, 'local.' + model_name.replace("/", "_"))
    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)

# predefined models
model_fullnames = {  'gpt2': 'gpt2',
                     'gpt2-xl': 'gpt2-xl',
                     'opt-2.7b': 'facebook/opt-2.7b',
                     'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
                     'gpt-j-6B': 'EleutherAI/gpt-j-6B',
                     'gpt-neox-20b': 'EleutherAI/gpt-neox-20b',
                     'mgpt': 'sberbank-ai/mGPT',
                     'pubmedgpt': 'stanford-crfm/pubmedgpt',
                     'mt5-xl': 'google/mt5-xl',
                     'qwen-7b': 'Qwen/Qwen2.5-7B',
                     'qwen-7b-instruct': 'Qwen/Qwen2.5-7B-Instruct',
                     'mistralai-7b': 'mistralai/Mistral-7B-v0.1',
                     'mistralai-7b-instruct': 'mistralai/Mistral-7B-Instruct-v0.3',
                     'llama3-8b': 'meta-llama/Meta-Llama-3-8B',
                     'llama3-8b-instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
                     'falcon-7b': 'tiiuae/falcon-7b',
                     'falcon-7b-instruct': 'tiiuae/falcon-7b-instruct',
                     'llama2-13b': 'TheBloke/Llama-2-13B-fp16',
                     'llama2-13b-chat': 'meta-llama/Llama-2-13b-chat-hf', 
                     'gemma-9b': 'google/gemma-2-9b',
                     'gemma-9b-instruct': 'google/gemma-2-9b-it',
                     'bloom-7b1': 'bigscience/bloom-7b1',
                     'opt-13b': 'facebook/opt-13b',
                     'pythia-12b': 'EleutherAI/pythia-12b',
                     }
float16_models = ['gpt-neo-2.7B', 'gpt-j-6B', 'gpt-neox-20b', 'falcon-7b', 'falcon-7b-instruct', 'qwen-7b', 'qwen-7b-instruct', 'mistralai-7b', 'mistralai-7b-instruct', 'llama3-8b', 'llama3-8b-instruct', 'gemma-9b', 'gemma-9b-instruct', 'llama2-13b', 'bloom-7b1', 'opt-13b', 'pythia-12b', 'llama2-13b-chat']

def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def load_model(model_name, device, cache_dir, torch_dtype=None):
    model_fullname = get_model_fullname(model_name)
    print(f'Loading model {model_fullname}...')
    model_kwargs = {}
    if model_name in float16_models:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in model_name:
        model_kwargs.update(dict(revision='float16'))
    if torch_dtype is not None:
        model_kwargs.update(dict(torch_dtype=torch_dtype))
    model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model

def load_tokenizer(model_name, cache_dir):
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}
    if "facebook/opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    optional_tok_kwargs['padding_side'] = 'right'
    base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    return base_tokenizer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="mistralai-7b-instruct")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    load_tokenizer(args.model_name, args.cache_dir)
    load_model(args.model_name, 'cpu', args.cache_dir)
