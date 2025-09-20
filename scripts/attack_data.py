# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json
import logging
import random
import re
import sys
import time
import nltk
import numpy as np
import torch
from tqdm import tqdm

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def bt_translation(src, browser):
    zh2en_url = f'https://translate.google.com/?hl=zh&sl=en&tl=zh-CN&text={src}&op=translate'
    browser.get(zh2en_url)  # 访问相对应链接 browser.close#关闭浏览器
    time.sleep(random.randint(1, 2))
    browser.find_element_by_xpath(
        '//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[3]/c-wiz[1]/span/span/div/div[2]/div[1]').send_keys(
        src)
    browser.refresh()
    # time.sleep(50)
    time.sleep(random.randint(2, 3))
    text = browser.find_element_by_xpath(
        '/html/body/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[3]/c-wiz[2]/div/div[8]/div/div[1]/span[1]').text
    en_text = text.replace("翻譯搜尋結果\n", "").replace("\n", "")
    en2zh_url = f'https://translate.google.com/?hl=zh&sl=zh-CN&tl=en&text={en_text}&op=translate'
    browser.get(en2zh_url)  # 访问相对应链接 browser.close#关闭浏览器
    time.sleep(random.randint(1, 2))
    browser.refresh()
    time.sleep(random.randint(2, 3))
    text = browser.find_element_by_xpath(
        '/html/body/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[3]/c-wiz[2]/div/div[8]/div/div[1]/span[1]').text
    tgt = text.replace("翻譯搜尋結果\n", "").replace("\n", "")
    return tgt


def read_data(json_path):
    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data


def count_sentences_in_paragraph(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    return len(sentences)


def save_json_data(data, path):
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)


def replace_line_breaks(s):
    s = re.sub('\n', ' ', s)
    return s


def truncate_to_last_sentence(s):
    # 从字符串尾部向前找句号的位置
    last_period = s.rfind('.') or s.rfind('!') or s.rfind('?')
    # 如果找到句号，就截断到这个位置（包括句号）
    if last_period != -1:
        s = s[:last_period + 1]
    return s


def check_paragraphs(texts):
    if count_sentences_in_paragraph(texts) >= 4:
        return True
    else:
        return False

def load_data(input_file):
    data_file = f"{input_file}.raw_data.json"
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data

def save_data(output_file, data):
    # write the data to a json file in the save folder
    data_file = f"{output_file}.raw_data.json"
    with open(data_file, "w") as fout:
        json.dump(data, fout, indent=4)
        print(f"Raw data written into {data_file}")


def run(args):
    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load data
    data = load_data(args.input_path)

    # perturbation attacks prepare
    if "perturbation" in args.method:
        from textattack.augmentation import TextBuggerAugmenter
        from textattack.augmentation import TextFoolerAugmenter
        from textattack.augmentation import DeepWordBugAugmenter
        word_augmenter = TextFoolerAugmenter()
        character_augmenter = DeepWordBugAugmenter()
        word_character_augmenter = TextBuggerAugmenter()

    human_key = "original"
    llm_key = 'sampled'

    n_samples = len(data)
    for i in tqdm(range(n_samples)):
        human = data[human_key][i]
        llm = data[llm_key][i]

        # llms selection
        if "perturbation" in args.method:
            humans = count_sentences_in_paragraph(human)
            llms = count_sentences_in_paragraph(llm)
            for attack in ["perturbation_character", "perturbation_word", "perturbation_sent"]:
                if attack == "perturbation_character":
                    augmenter = character_augmenter
                elif attack == "perturbation_word":
                    augmenter = word_augmenter
                elif attack == "perturbation_sent":
                    augmenter = word_character_augmenter
                else:
                    raise ValueError(f"{attack} is not in perturbation_attacks")

                try:
                    # final_data = []
                    # for d in range(humans):
                    #     final_data.append(augmenter.augment(d)[0])
                    # human_result = ' '.join(final_data)
                    # data[human_key][i] = human_result
                    # logging.info(f"{attack} human finished")

                    final_data = []
                    for d in range(llms):
                        final_data.append(augmenter.augment(d)[0])
                    llm_result = ' '.join(final_data)
                    data[llm_key][i] = llm_result
                    logging.info(f"{attack} llm finished")

                except Exception as e:
                    logging.info(f"error: {e}")
                    pass
        
    save_data(args.output_file, data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=False, default="./exp_main/data/xsum_opt-2.7b", type=str)
    parser.add_argument('--output_path', required=False, default="./exp_main/data/xsum_opt-2.7b", type=str)
    parser.add_argument('--method', default="perturbation_sent", type=str, choices=["perturbation_character", "perturbation_word", "perturbation_sent"], required=False)
    parser.add_argument('--seed', default=2023, type=int, required=False)
    args = parser.parse_args()
    run(args)
