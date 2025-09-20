import os
import json
import time
import pickle
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

# Minimal model zoo mapping model keys to pretrained model names.
MODEL_ZOO = {
    'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
    'gemma-2b': 'google/gemma-1.1-2b-it',
    'gemma-7b': 'google/gemma-1.1-7b-it', 
    # 'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'gpt2-xl': 'gpt2-xl',
    'qwen-7b': 'Qwen/Qwen2.5-7B',
    'llama3-8b': 'meta-llama/Meta-Llama-3-8B',
    'mistralai-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
}

# Prompt templates for text completion.
COMPLETION_PROMPT_ONLY = "Complete the following text: "
COMPLETION_PROMPT = "Given the summary:\n{prompt}\n Complete the following text: "

def generate(model, tokenizer, input_ids, trigger_length, target_length):
    """
    Generate additional tokens using the model's generation API.
    
    Parameters:
      model: the language model for generation.
      tokenizer: associated tokenizer.
      input_ids: input token IDs (either 1D or 2D).
      trigger_length: the length of the prompt (number of tokens to skip in the output).
      target_length: the number of new tokens to generate.
      
    Returns:
      Generated tokens (as a 2D tensor) after removing the trigger tokens.
    """
    config = model.generation_config
    config.max_new_tokens = target_length
    # If input_ids is 1D, add a batch dimension; otherwise, assume it's already 2D.
    if input_ids.dim() == 1:
        input_ids = input_ids.to(model.device).unsqueeze(0)
    else:
        input_ids = input_ids.to(model.device)
    # Create an attention mask of the same shape.
    attn_masks = torch.ones(input_ids.shape, device=input_ids.device)
    # Generate new tokens.
    out = model.generate(
        input_ids, 
        attention_mask=attn_masks,
        generation_config=config,
        pad_token_id=tokenizer.pad_token_id
    )[0]
    # Return output tokens after the prompt (slice along dimension 1).
    return out[trigger_length:]


def compute_fce_loss(logits, targets, text_slice):
    """
    Compute the FCE loss by shifting indices by 1.
    Returns a NumPy array of loss values.
    """
    loss = CrossEntropyLoss(reduction='none')(
        logits[0, text_slice.start-1:text_slice.stop-1, :],
        targets
    )
    return loss.detach().cpu().numpy()

def compute_bce_loss(logits, targets, text_slice):
    """
    Compute the BCE loss without shifting indices.
    Returns a NumPy array of loss values.
    """
    loss = CrossEntropyLoss(reduction='none')(
        logits[0, text_slice, :],
        targets
    )
    return loss.detach().cpu().numpy()

def detect_single_sample(args, model, tokenizer, summary_model, summary_tokenizer, sample, device='cuda'):
    """
    Process a sample by generating a summary-based prompt, tokenizing (with clipping),
    obtaining model outputs, and computing loss-based features (FCE and BCE).
    Returns a list of loss features computed over 10 segments.
    """
    # Generate the summary-based prompt.
    if 'gpt-' in args.summary_model:
        from openai import OpenAI
        openai_key = os.environ.get('OPENAI_API_KEY')
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")
        client = OpenAI(api_key=openai_key)
        from tenacity import (
            retry,
            stop_after_attempt,
            wait_random_exponential,
        )
        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
        def openai_backoff(client, **kwargs):
            return client.chat.completions.create(**kwargs)
        summary_input = f"generate a very short and concise summary for the following text, just the summary: {sample}"
        response = openai_backoff(client, model=args.summary_model,
                                  messages=[{"role": "user", "content": summary_input}])
        summary_text = response.choices[0].message.content.strip()
        # if '"""' in summary_text:
        #     summary_text = summary_text.split('"""')[-1]
        prompt_text = COMPLETION_PROMPT.format(prompt=summary_text)
    elif args.summary_model in MODEL_ZOO:
        summary_input = f"Write a title for this text: {sample}\nJust output the title:"
        summary_ids = summary_tokenizer(summary_input, return_tensors='pt',
                                        max_length=args.sample_clip, truncation=True).input_ids.to(device)
        summary_ids = summary_ids[:, 1:]  # Remove start token.
        gen_ids = generate(summary_model, summary_tokenizer, summary_ids, summary_ids.shape[1], 64)
        summary_text = summary_tokenizer.decode(gen_ids, skip_special_tokens=True).strip().split('\n')[0]
        prompt_text = COMPLETION_PROMPT.format(prompt=summary_text)
    else:
        prompt_text = COMPLETION_PROMPT_ONLY

    # Tokenize the prompt and sample with token-level clipping.
    prompt_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(device)
    text_ids = tokenizer(sample, return_tensors='pt', max_length=args.sample_clip, truncation=True).input_ids.to(device)
    combined_ids = torch.cat([prompt_ids, text_ids], dim=1)
    text_slice = slice(prompt_ids.shape[1], combined_ids.shape[1])
    outputs = model(input_ids=combined_ids)
    logits = outputs.logits
    targets = combined_ids[0][text_slice]
    
    # Compute loss features from FCE and BCE losses.
    fce_loss = compute_fce_loss(logits, targets, text_slice)
    bce_loss = compute_bce_loss(logits, targets, text_slice)
    features = []
    for p in range(1, 10):
        split = len(fce_loss) * p // 10
        features.extend([
            np.mean(fce_loss[split:]), np.max(fce_loss[split:]), 
            np.min(fce_loss[split:]), np.std(fce_loss[split:]),
            np.mean(bce_loss[split:]), np.max(bce_loss[split:]), 
            np.min(bce_loss[split:]), np.std(bce_loss[split:])
        ])
    return features

def data_generation(args, out_dir, task, generative_model, base_dir):
    """
    Generate loss-based features for both human and GPT samples and save them to disk.
    
    Parameters:
      out_dir: Output directory.
      task: Task name (e.g., Arxiv, Code, Essay).
      generative_model: Key for the GPT samples.
      
    Returns:
      The output directory.
    """
    # Load summary model and its tokenizer if specified.
    if args.summary_model in MODEL_ZOO:
        summary_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ZOO[args.summary_model],
            torch_dtype=torch.float16,
            device_map='auto',
            cache_dir=args.cache_dir,
        ).eval()
        summary_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ZOO[args.summary_model], padding_side='left', cache_dir=args.cache_dir,
        )
        summary_tokenizer.pad_token = summary_tokenizer.eos_token
    else:
        summary_model, summary_tokenizer = None, None

    # Load detection model and its tokenizer.
    if args.detect_model in MODEL_ZOO:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ZOO[args.detect_model],
            torch_dtype=torch.float16,
            device_map='auto', 
            cache_dir=args.cache_dir,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ZOO[args.detect_model], padding_side='left', cache_dir=args.cache_dir,
        )
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError("Unknown detection model")

    if getattr(args, "use_hf_dataset", False):
        # Load from Hugging Face dataset.
        from datasets import load_dataset
        ds = load_dataset("HanxiGuo/BiScope_Data", split="train")
        paraphrased_flag = False

        # Filter human data: use rows where task matches, and source is 'human'.
        human_data = ds.filter(lambda x: x["task"] == task and x["source"].lower() == "human")
        human_data = [s["text"] for s in human_data]

        # Filter GPT-generated data: use rows where task and paraphrased flag match and source matches generative_model.
        # (Assumes that the GPT-generated samples have source equal to the provided generative_model string.)
        gpt_data = ds.filter(lambda x: x["task"] == task and x["paraphrased"] == paraphrased_flag and x["source"].lower() == generative_model.lower())
        gpt_data = [s["text"] for s in gpt_data]
    else:
        
        # Load human data. Human data do not have paraphrased version, so only use the normal data.
        with open(f'{base_dir}/{task}_{generative_model}.raw_data.json', 'r') as f:
            dataset = json.load(f)

        human_data = dataset['original']
        gpt_data = dataset['sampled']
    
    # Define the human features file path internally.
    human_feat_path = os.path.join(out_dir, f"{task}_human_features.pkl")
    
    # Generate and save human features.
    human_features = [detect_single_sample(args, model, tokenizer, summary_model, summary_tokenizer, s, device='cuda') for s in tqdm(human_data)]
    with open(human_feat_path, 'wb') as f:
        pickle.dump(human_features, f)
    
    # Generate and save GPT features.
    gpt_feat_path = os.path.join(out_dir, f"{task}_GPT_features.pkl")
    gpt_features = [detect_single_sample(args, model, tokenizer, summary_model, summary_tokenizer, s, device='cuda') for s in tqdm(gpt_data)]
    with open(gpt_feat_path, 'wb') as f:
        pickle.dump(gpt_features, f)
    
    return out_dir
