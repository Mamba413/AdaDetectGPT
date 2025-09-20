# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

import numpy as np
import datasets
import torch
import random
import argparse
import os
import json
import custom_datasets
from model import load_tokenizer, load_model

ROLES = {'xsum': 'You are a News writer.',
            'writing': 'You are a Fiction writer.',
            'pubmed': 'You are a Technical writer.', 
            'yelp_polarity': 'You are a Review writer on Yelp.', 
            'essay': 'You are a student of high school and university level. And now, you are an Essay writer.'}
PROMPTS = {'xsum': 'Please write an article with about 150 words starting exactly with:',
            'writing': 'Please write an article with about 150 words starting exactly with:',
            'pubmed': 'Please answer the question in about 50 words.',
            'yelp_polarity': 'Please write a review with about 150 words starting exactly with:',
            'essay': 'Please write an essay with about 200 words starting exactly with:'}

def save_data(output_file, args, data):
    # write args to file
    args_file = f"{output_file}.args.json"
    with open(args_file, "w") as fout:
        json.dump(args.__dict__, fout, indent=4)
        print(f"Args written into {args_file}")

    # write the data to a json file in the save folder
    data_file = f"{output_file}.raw_data.json"
    with open(data_file, "w") as fout:
        json.dump(data, fout, indent=4)
        print(f"Raw data written into {data_file}")


def load_data(input_file):
    data_file = f"{input_file}.raw_data.json"
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data


class DataBuilder:
    def __init__(self, args):
        self.args = args
        self.base_tokenizer = load_tokenizer(args.base_model_name, args.cache_dir)
        self.base_model = None if args.openai_model else load_model(args.base_model_name, args.device, args.cache_dir)

    def _openai_sample(self, prefix):
        def _drop_last_word(text):
            return ' '.join(text.split(' ')[:-1])

        from openai import OpenAI

        client = OpenAI(api_key=self.args.openai_key)
        assert self.args.openai_key is not None, "Must provide OpenAI API key as --openai_key"
        if self.args.openai_base is not None:
            # TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url=self.args.openai_base)'
            # openai.api_base = self.args.openai_base
            OpenAI(base_url=self.args.openai_base)

        if self.args.dataset != 'pubmed':  # keep Answer: prefix for pubmed
            prefix = _drop_last_word(prefix)

        # sample from the openai model
        kwargs = {"max_tokens": 200}
        if self.args.do_top_p:
            kwargs['top_p'] = self.args.top_p
        elif self.args.do_top_k:
            kwargs['top_k'] = self.args.top_k
        elif self.args.do_temperature:
            kwargs['temperature'] = self.args.temperature

        if self.args.openai_model == 'davinci':
            # kwargs["engine"] = self.args.openai_model
            kwargs["model"] = "text-davinci-003"
            response = client.completions.create(prompt=f"{prefix}", **kwargs)
            return prefix + response.choices[0].text

        elif self.args.openai_model in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o']:
            messages = [
                {'role': 'system', 'content': ROLES[self.args.dataset]},
                {'role': 'user', 'content': f'{PROMPTS[self.args.dataset]} {prefix}'},
            ]
            kwargs["model"] = self.args.openai_model
            kwargs["messages"] = messages
            response = client.chat.completions.create(**kwargs)
            response = response.choices[0].message.content
            # ChatGPT may repeat the prefix
            if response.startswith(prefix[:20]):
                return response
            return prefix + ' ' + response

        else:
            raise NotImplementedError

    def _gemini_sample(self, prefix) -> str:
        from google import genai
        from google.genai import types

        # 1) Initialize the client (uses GOOGLE_API_KEY from env)
        client = genai.Client()

        # 2) Optionally drop the last word for non-pubmed datasets
        if self.args.dataset != 'pubmed':
            prefix = ' '.join(prefix.split()[:-1])

        instruct = ROLES[self.args.dataset] + PROMPTS[self.args.dataset]

        # 3) Build the request dict from self.args
        params = {
            "model": self.args.gemini_model,
            "contents": prefix,
        }
        response = client.models.generate_content(
            **params,
            config=types.GenerateContentConfig(
                top_p=self.args.top_p if self.args.do_top_p else None,
                top_k=self.args.top_k if self.args.do_top_k else None,
                temperature=self.args.temperature if self.args.do_temperature else None,
                seed=self.args.seed,
                candidate_count=1,
                system_instruction=instruct,
            ),
        )
        response = response.text.strip()

        # print(f"Gemini response: {response}")
        # 5) Return response
        if response.startswith(prefix[:20]):
            return response
        return prefix + ' ' + response

    def _claude_sample(self, prefix: str) -> str:
        from anthropic import Anthropic

        client = Anthropic()

        # 2) For non-pubmed, drop last word as in your other samplers
        if self.args.dataset != "pubmed":
            prefix = " ".join(prefix.split()[:-1])

        # 3) Build system + user content just like in GPT path
        model_full_name = {'claude-3-5-haiku': "claude-3-5-haiku-20241022"}

        system_instruction = ROLES[self.args.dataset]

        # 4) Assemble request kwargs
        req = {
            "system": system_instruction,
            "temperature": self.args.temperature if self.args.do_temperature else None,
            "top_p": self.args.top_p if self.args.do_top_p else None,
            "top_k": self.args.top_k if self.args.do_top_k else None,
        }
        response = client.messages.create(
            model=model_full_name[self.args.claude_model], 
            max_tokens=200,
            messages=[{"role": "user", "content": f'{PROMPTS[self.args.dataset]} {prefix}'}],
            **{k: v for k, v in req.items() if v is not None}
        )
        response = response.content[0].text.strip()
        response = response.removeprefix("Here's the article:").lstrip("\r\n")
        print(f"Claude response: {response}")
        return response

    def _sample_rewrite_text_from_model(self, human_texts, min_words, sampling_kwargs):
        revised_statement = "You are a professional rewriting expert and you can help paraphrasing this paragraph without missing the original details. Please keep the length of the rewritten text similar to the original text. Original text: \"{}\""
        texts = [revised_statement.format(o) for o in human_texts] 

        self.base_model.eval()
        decoded = ['' for _ in range(len(texts))]

        tries = 0
        m = 0
        while m < min_words:
            if tries != 0:
                print()
                print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")
                prefixes = self.base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
                for prefix, x in zip(prefixes, decoded):
                    if len(x.split()) == m:
                        print(prefix, '=>', x)

            all_encoded = self.base_tokenizer(texts, return_tensors="pt", padding=True, padding_side='left', return_token_type_ids=False).to(self.args.device)
            prompt_lens = all_encoded['input_ids'].shape[1]
            outputs = self.base_model.generate(**all_encoded, min_new_tokens=min_words, max_length=prompt_lens*2, do_sample=True, **sampling_kwargs, pad_token_id=self.base_tokenizer.eos_token_id, eos_token_id=self.base_tokenizer.eos_token_id)
            gen_ids = outputs[:, prompt_lens:]
            decoded = self.base_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            m = min(len(x.split()) for x in decoded) 
            tries += 1

        return decoded

    def build_sampling_kwargs(self):
        sampling_kwargs = {}
        if self.args.do_top_p:
            sampling_kwargs['top_p'] = self.args.top_p
        elif self.args.do_top_k:
            sampling_kwargs['top_k'] = self.args.top_k
        elif self.args.do_temperature:
            sampling_kwargs['temperature'] = self.args.temperature

        if self.args.do_exact_cond_prob:
            sampling_kwargs['top_p'] = 1.0
            sampling_kwargs['top_k'] = 0
            sampling_kwargs['temperature'] = 1.0
        return sampling_kwargs

    # sample from base_model using ****only**** the first 30 tokens in each example as context
    def _sample_from_model(self, texts, min_words=55, prompt_tokens=30):
        # encode each text as a list of token ids
        if self.args.dataset == 'pubmed':
            texts = [t[:t.index(custom_datasets.SEPARATOR)] for t in texts]
            all_encoded = self.base_tokenizer(texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
        else:
            all_encoded = self.base_tokenizer(texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
            all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}

        if self.args.openai_model or self.args.gemini_model or self.args.claude_model:
            # decode the prefixes back into text
            prefixes = self.base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)

            decoded = []
            for idx, prefix in enumerate(prefixes):
                while idx >= len(decoded):
                    try:
                        if self.args.openai_model:
                            decoded.append(self._openai_sample(prefix))
                        elif self.args.gemini_model:
                            decoded.append(self._gemini_sample(prefix))
                        elif self.args.claude_model:
                            decoded.append(self._claude_sample(prefix))
                    except Exception as ex:
                        print(ex)
                        print('Wait 10 minutes before retry ...')
                        time.sleep(600)

        else:
            self.base_model.eval()
            decoded = ['' for _ in range(len(texts))]

            # sample from the model until we get a sample with at least min_words words for each example
            # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
            tries = 0
            m = 0
            while m < min_words:
                if tries != 0:
                    print()
                    print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")
                    prefixes = self.base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
                    for prefix, x in zip(prefixes, decoded):
                        if len(x.split()) == m:
                            print(prefix, '=>', x)

                sampling_kwargs = self.build_sampling_kwargs()
                min_length = 50 if self.args.dataset in ['pubmed'] else 150
                outputs = self.base_model.generate(**all_encoded, min_length=min_length, max_new_tokens=None, max_length=self.args.max_length, do_sample=True, **sampling_kwargs, pad_token_id=self.base_tokenizer.eos_token_id, eos_token_id=self.base_tokenizer.eos_token_id)
                decoded = self.base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                m = min(len(x.split()) for x in decoded)
                tries += 1

        return decoded

    def generate_samples(self, raw_data, batch_size):
        # trim to shorter length
        def _trim_to_shorter_length(texta, textb, textc=None):
            # truncate to shorter of o and s (optional for textc)
            shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
            if textc is not None:
                shorter_length = min(shorter_length, len(textc.split(' ')))
            texta = ' '.join(texta.split(' ')[:shorter_length])
            textb = ' '.join(textb.split(' ')[:shorter_length])
            if textc is not None:
                textc = ' '.join(textc.split(' ')[:shorter_length])
                return texta, textb, textc
            else:
                return texta, textb

        def _trim_human_prompt(texta, n_human_prompts):
            text = ' '.join(texta.split(' ')[n_human_prompts:])
            return texta

        def _truncate_to_substring(text, substring, idx_occurrence):
            # truncate everything after the idx_occurrence occurrence of substring
            assert idx_occurrence > 0, 'idx_occurrence must be > 0'
            idx = -1
            for _ in range(idx_occurrence):
                idx = text.find(substring, idx + 1)
                if idx == -1:
                    return text
            return text[:idx]

        data = {
            "original": [],
            "sampled": [],
        }
        if self.args.revised_human_text:
            new_data = {'revised': []}

        min_generated_words = 30 if self.args.dataset in ['pubmed'] else 55
        for batch in range(len(raw_data) // batch_size):
            print('Generating LLM samples for batch', batch, 'of', len(raw_data) // batch_size)
            original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
            sampled_text = self._sample_from_model(original_text, min_words=min_generated_words, 
                                                   prompt_tokens=self.args.n_prompts)

            for o, s in zip(original_text, sampled_text):
                if self.args.dataset == 'pubmed':
                    s = _truncate_to_substring(s, 'Question:', 2)
                    o = o.replace(custom_datasets.SEPARATOR, ' ')

                if self.args.trim_human:
                    o = _trim_human_prompt(o, self.args.n_prompts)
                    s = _trim_human_prompt(s, self.args.n_prompts)

                o, s = _trim_to_shorter_length(o, s)

                # add to the data
                data["original"].append(o)
                data["sampled"].append(s)

            # if revised-text, then remove the last word from each sampled text
            if self.args.revised_human_text:
                human_texts = data["original"][(-batch_size):]
                machine_texts = data["sampled"][(-batch_size):]
                sampling_kwargs = self.build_sampling_kwargs() 
                revised_original = self._sample_rewrite_text_from_model(human_texts, min_generated_words, sampling_kwargs)

                for i, (o, r, s) in enumerate(zip(human_texts, revised_original, machine_texts)):
                    if self.args.dataset == 'pubmed':
                        r = r.replace(custom_datasets.SEPARATOR, ' ')

                    o, r, s = _trim_to_shorter_length(o, r, s)

                    data['original'][batch * batch_size + i] = o
                    data['sampled'][batch * batch_size + i] = s
                    new_data['revised'].append(r)

        if self.args.revised_human_text:
            # add the revised human text to the data
            data['revised'] = new_data['revised']

        return data

def generate_data(args, dataset, key):
    # strip newlines from each example; replace one or more newlines with a single space
    def _strip_newlines(text):
        return ' '.join(text.split())

    # load data
    if dataset in custom_datasets.DATASETS:
        data = custom_datasets.load(dataset, args.cache_dir)
    else:
        data = custom_datasets.load_dataset(dataset, split='train', cache_dir=args.cache_dir)[key]

    # get unique examples, strip whitespace, and remove newlines
    # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
    # then take just the examples that are <= 512 tokens (for the base model)
    # then generate n_samples samples

    # remove duplicates from the data
    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

    # strip whitespace around each example
    data = [x.strip() for x in data]

    # remove newlines from each example
    data = [_strip_newlines(x) for x in data]

    # try to keep only examples with > 250 words
    if dataset in ['writing', 'squad', 'xsum', 'yelp_polarity', "essay"]:
        long_data = [x for x in data if len(x.split()) > 250]
        if len(long_data) > 0:
            data = long_data

    random.shuffle(data)
    data = data[:5_000]

    # keep only examples with <= 512 tokens according to base_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    data_builder = DataBuilder(args)
    tokenized_data = data_builder.base_tokenizer(data)
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]

    # print stats about remaining data
    print(f"Total number of samples: {len(data)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

    return data_builder.generate_samples(data[:args.n_samples], batch_size=args.batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_main/data/yelp_gpt2-xl")
    parser.add_argument('--dataset', type=str, default='pubmed', choices=['xsum', 'squad', 'writing', 'pubmed', 'essay', 'yelp'])
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--openai_base', type=str, default=None)
    parser.add_argument('--openai_key', type=str, default=None)
    parser.add_argument('--openai_model', type=str, default=None, choices=['davinci', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4o'])
    parser.add_argument('--gemini_model', type=str, default=None, choices=['gemini-2.5-flash'])
    parser.add_argument('--claude_model', type=str, default=None, choices=['claude-3-5-haiku'])
    parser.add_argument('--base_model_name', type=str, default="opt-2.7b")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--do_exact_cond_prob', action='store_true')
    # parser.add_argument('--do_exact_cond_prob', type=bool, default=True)
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--do_temperature', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--n_prompts', type=int, default=120)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--trim_human', action='store_true')
    parser.add_argument('--revised_human_text', action='store_true')
    # parser.add_argument('--revised_human_text', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    os.environ["XDG_CACHE_HOME"] = args.cache_dir
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    print(f"Using cache dir {args.cache_dir}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'yelp':
        args.dataset = 'yelp_polarity'

    print(f'Loading dataset {args.dataset}...')
    dataset_keys = {'xsum': 'document', 'squad': 'context', 'writing': 'document', 'essay': 'document', 'yelp_polarity': 'text'}
    data = generate_data(args, args.dataset, dataset_keys[args.dataset] if args.dataset in dataset_keys else None)

    if args.dataset == 'yelp':
        args.dataset = 'yelp_polarity'

    save_data(args.output_file, args, data)
