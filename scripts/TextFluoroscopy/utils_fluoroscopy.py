from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
import pickle
import huggingface_hub 
huggingface_hub.login("hf_xxxxx")  # replace hf_xxxxx with your actual token

def last_token_pool(last_hidden_states,
                 attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device='cpu'), sequence_lengths]
    
def load_model(model_name, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,       # <-- Hub ID, *not* your cache path
        cache_dir=cache_dir,         # where to store/download files
        trust_remote_code=True,       # allow loading the repo’s custom code
        use_auth_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        device_map='auto',
        use_auth_token=True, 
        cache_dir=cache_dir
    )
    return tokenizer, model

def get_kl(model, tokenizer, input_texts, max_length, device):
    batch_dict = tokenizer(input_texts, max_length=max_length, 
                           padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**batch_dict, output_hidden_states=True)
        last_logits = model.lm_head(outputs.hidden_states[-1]).squeeze()
        first_logits = model.lm_head(outputs.hidden_states[0]).squeeze()
    kls = []
    for i in range(1, len(outputs.hidden_states) - 1):
        with torch.no_grad():
            middle_logits = model.lm_head(outputs.hidden_states[i]).squeeze()
        kls.append(F.kl_div(F.log_softmax(middle_logits, dim=-1), F.softmax(first_logits, dim=-1), reduction='batchmean').item() +
                   F.kl_div(F.log_softmax(middle_logits, dim=-1), F.softmax(last_logits, dim=-1), reduction='batchmean').item())
    return kls

def compute_kl_feat(model, tokenizer, file_name, save_dir, max_length, device):
    test_datasets = {}
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data_name = file_name.split('/')[-1]
    if not os.path.exists(save_dir + data_name + '.pkl'):
        print(file_name)
        test_datasets[file_name] = {'data': [], 'label': []}
        with open(file_name + '.raw_data.json', 'r') as f:
            data = json.load(f)
        kls = []
        n_samples = len(data['original'])
        # n_samples = 30
        for idx in tqdm(range(n_samples)):
            human_text = data['original'][idx]
            kl = get_kl(model, tokenizer, [human_text], max_length, device)
            kls.append(kl)
            llm_text = data['sampled'][idx]
            kl = get_kl(model, tokenizer, [llm_text], max_length, device)
            kls.append(kl)
            # if len(kls) >= 300:
            #     break
        print(save_dir + data_name + '.pkl')
        pickle.dump(kls, open(save_dir + data_name + '.pkl', 'wb'))

def load_model2(model_name, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,       # <-- Hub ID, *not* your cache path
        cache_dir=cache_dir,         # where to store/download files
        trust_remote_code=True,       # allow loading the repo’s custom code
        use_auth_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        device_map='auto',
        use_auth_token=True, 
        cache_dir=cache_dir
    )
    return tokenizer, model

def get_all_embedding(model, tokenizer, input_texts, max_length, device):
    batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**batch_dict, output_hidden_states=True)
    all_embed = [last_token_pool(outputs.hidden_states[i].cpu(), batch_dict['attention_mask']) for i in range(len(outputs.hidden_states))]
    all_embed = torch.concat(all_embed, 1).cpu()
    return all_embed

def compute_embedding(model, tokenizer, file_name, save_dir, max_length, device):
    test_datasets = {}
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data_name = file_name.split('/')[-1]
    if not os.path.exists(save_dir + data_name + '.pt'):
        print(file_name)
        test_datasets[file_name] = {'data':[], 'label':[]}
        with open(file_name + '.raw_data.json', 'r') as f:
            data = json.load(f)
        embeddings = []
        n_samples = len(data['original'])
        # n_samples = 30
        for idx in tqdm(range(n_samples)):
            human_text = data['original'][idx]
            embedding = get_all_embedding(model, tokenizer, [human_text], max_length, device)
            embeddings.append(embedding)
            llm_text = data['sampled'][idx]
            embedding = get_all_embedding(model, tokenizer, [llm_text], max_length, device)
            embeddings.append(embedding)
            # if len(embeddings) >=300:
            #     break
        embeddings = torch.cat(embeddings, dim=0)
        print('embedding shape: ', embeddings.shape)
        print(save_dir + data_name +'.pt')
        torch.save(embeddings, save_dir + data_name + '.pt')


if __name__ == '__main__':
    device = 'cuda'
    max_length = 512
    
    model_name = "Alibaba-NLP/gte-Qwen1.5-7B-instruct"
    # cache_dir = '/root/autodl-tmp/cache'
    cache_dir = '../../../cache'
    
    which_embedding='gte-qwen_KL_with_first_and_last_layer'
    save_dir = f'save/{which_embedding}/'
    data_dir = 'dataset/processed_data/'
    tokenizer, model = load_model(model_name, cache_dir)
    compute_kl_feat(model, tokenizer, data_dir, save_dir, max_length, device)
    
    which_embedding='gte-qwen_all_embedding'
    save_dir = f'save/{which_embedding}/save_embedding/'
    data_dir = 'dataset/processed_data/'
    tokenizer, model = load_model2(model_name, cache_dir)
    compute_embedding(model, tokenizer, data_dir, save_dir, max_length, device)