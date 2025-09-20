# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os.path
import random
import datasets
from datasets import Features, Sequence, Value

SEPARATOR = '<<<SEP>>>'


DATASETS = ['writing', 'english', 'german', 'pubmed', 'essay']

def load_dataset(path, name=None, split=None, cache_dir=None):
    # use local model if it exists
    local_path = os.path.join(cache_dir, f'local.{path}_{name}_{split}')
    if os.path.exists(local_path):
        return datasets.load_from_disk(local_path)
    return datasets.load_dataset(path, name, split=split, cache_dir=cache_dir)

def load_pubmed(cache_dir):

    full_schema = Features({
        "contexts":                Sequence(Value("string")),
        "labels":                  Sequence(Value("string")),
        "meshes":                  Sequence(Value("string")),
        "reasoning_required_pred": Sequence(Value("string")),
        "reasoning_free_pred":     Sequence(Value("string")),
    })

    data = load_dataset('pubmed_qa', 'pqa_labeled', split='train', cache_dir=cache_dir)

    data = data.cast(full_schema)
    
    # combine question and long_answer
    data = [f'Question: {q} Answer:{SEPARATOR}{a}' for q, a in zip(data['question'], data['long_answer'])]

    return data


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').replace(
        '\u2019', '\'').strip()


def load_writing(cache_dir=None):
    writing_path = 'data/writingPrompts'
    
    with open(f'{writing_path}/valid.wp_source', 'r') as f:
        prompts = f.readlines()
    with open(f'{writing_path}/valid.wp_target', 'r') as f:
        stories = f.readlines()
    
    prompts = [process_prompt(prompt) for prompt in prompts]
    joined = [process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]
    filtered = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]

    random.seed(0)
    random.shuffle(filtered)

    return filtered

def load_essay(cache_dir=None):
    essay_path = 'data/ghostbuster-data/essay/human'
    
    filenames = sorted(f for f in os.listdir(essay_path) if f.endswith('.txt'))

    texts = []
    for fn in filenames:
        full_path = os.path.join(essay_path, fn)
        with open(full_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    
    texts = [process_spaces(x) for x in texts]

    random.seed(0)
    random.shuffle(texts)

    return texts

def load_language(language, cache_dir):
    # load either the english or german portion of the wmt16 dataset
    assert language in ['en', 'de']
    d = load_dataset('wmt16', 'de-en', split='train', cache_dir=cache_dir)
    docs = d['translation']
    desired_language_docs = [d[language] for d in docs]
    lens = [len(d.split()) for d in desired_language_docs]
    sub = [d for d, l in zip(desired_language_docs, lens) if l > 100 and l < 150]
    return sub


def load_german(cache_dir):
    return load_language('de', cache_dir)


def load_english(cache_dir):
    return load_language('en', cache_dir)


def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        load_fn = globals()[f'load_{name}']
        return load_fn(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {name}')