import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
import numpy as np
import argparse
from metrics import get_roc_metrics, get_precision_recall_metrics
from sklearn.metrics import roc_auc_score
from TextFluoroscopy.utils_fluoroscopy import compute_embedding, compute_kl_feat, load_model, load_model2
import json

def get_embedding(train_file, valid_file, test_file, kl_path, which_embedding, which_layer, device):
    train_num=None 
    valid_num=None 
    test_num=None
    train_name = train_file.split("/")[-1]
    valid_name = valid_file.split("/")[-1]
    train_embeddings = torch.load(f'scripts/TextFluoroscopy/save/{which_embedding}_embedding/save_embedding/{train_name}.pt')[:train_num]
    valid_embeddings = torch.load(f'scripts/TextFluoroscopy/save/{which_embedding}_embedding/save_embedding/{valid_name}.pt')[:valid_num]
    train_labels = torch.arange(train_embeddings.shape[0]) % 2
    valid_labels = torch.arange(valid_embeddings.shape[0]) % 2
    
    train_embeddings = train_embeddings.to(device)
    valid_embeddings = valid_embeddings.to(device)
    train_labels = train_labels.to(device)
    valid_labels = valid_labels.to(device)

    with open(f'scripts/TextFluoroscopy/save/{kl_path}/{train_name}.pkl', 'rb') as f:
        train_kl = pickle.load(f)
        train_kl = np.array(train_kl)
        idx = train_kl.argmax(axis=1)
        if which_layer == 'max_kl':
            train_embeddings = torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row ,i in zip(train_embeddings, idx)]).to(device)
        if which_layer == 'max_kl_and_last_layer':
            train_embeddings = torch.cat([torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row ,i in zip(train_embeddings, idx)]),
                                          train_embeddings[:,-embedding_dim:]], dim=1).to(device)

    with open(f'scripts/TextFluoroscopy/save/{kl_path}/{valid_name}.pkl', 'rb') as f:
        valid_kl = pickle.load(f)
        valid_kl = np.array(valid_kl)
        idx = valid_kl.argmax(axis=1)
        if which_layer == 'max_kl':
            valid_embeddings = torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row,i in zip(valid_embeddings, idx)]).to(device)
        if which_layer == 'max_kl_and_last_layer':
            valid_embeddings = torch.cat([torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row, i in zip(valid_embeddings, idx)]),
                                          valid_embeddings[:,-embedding_dim:]], dim=1).to(device)

    if which_layer == 'first_layer':
        train_embeddings = train_embeddings[:,:embedding_dim].to(device)
        valid_embeddings = valid_embeddings[:,:embedding_dim].to(device)
        test_embeddings = test_embeddings[:,:embedding_dim].to(device)
    elif which_layer == 'last_layer':
        train_embeddings = train_embeddings[:,-embedding_dim:].to(device)
        valid_embeddings = valid_embeddings[:,-embedding_dim:].to(device)
        test_embeddings = test_embeddings[:,-embedding_dim:].to(device)
    elif which_layer == 'first_and_last_layers':
        train_embeddings = torch.cat([train_embeddings[:,:embedding_dim],train_embeddings[:,-embedding_dim:]], dim=1).to(device)
        valid_embeddings = torch.cat([valid_embeddings[:,:embedding_dim],valid_embeddings[:,-embedding_dim:]], dim=1).to(device)
        test_embeddings = torch.cat([test_embeddings[:,:embedding_dim],test_embeddings[:,-embedding_dim:]], dim=1).to(device)
    elif which_layer.startswith('layer_'):
        if 'last_layer' not in which_layer and 'later_layer' not in which_layer and 'to' not in which_layer:
            layer_num = int(which_layer.split('_')[-1])
            train_embeddings = train_embeddings[:,(layer_num)*embedding_dim:(layer_num+1)*embedding_dim].to(device)
            valid_embeddings = valid_embeddings[:,(layer_num)*embedding_dim:(layer_num+1)*embedding_dim].to(device)
        elif 'last_layer' in which_layer:
            layer_num = int(which_layer.split('_')[1])
            train_embeddings = torch.cat([train_embeddings[:,-embedding_dim:],train_embeddings[:,(layer_num)*embedding_dim:(layer_num+1)*embedding_dim]], dim=1).to(device)
            valid_embeddings = torch.cat([valid_embeddings[:,-embedding_dim:],valid_embeddings[:,(layer_num)*embedding_dim:(layer_num+1)*embedding_dim]], dim=1).to(device)
        elif 'later_layer' in which_layer:
            layer_num = int(which_layer.split('_')[1])
            train_embeddings = train_embeddings[:,(layer_num)*embedding_dim:].to(device)
            valid_embeddings = valid_embeddings[:,(layer_num)*embedding_dim:].to(device)    
        elif 'to' in which_layer:
            layer_num = int(which_layer.split('_')[1])
            layer_num2 = int(which_layer.split('_')[3])
            train_embeddings = train_embeddings[:,(layer_num)*embedding_dim:(layer_num2+1)*embedding_dim].to(device)
            valid_embeddings = valid_embeddings[:,(layer_num)*embedding_dim:(layer_num2+1)*embedding_dim].to(device)

    test_name = test_file.split("/")[-1]
    testset_embeddings = torch.load(f'scripts/TextFluoroscopy/save/{which_embedding}_embedding/save_embedding/{test_name}.pt')[:test_num]
    testset_embeddings = testset_embeddings.to(device)
    testset_labels = torch.arange(testset_embeddings.shape[0]) % 2
    with open(f'scripts/TextFluoroscopy/save/{kl_path}/{test_name}.pkl', 'rb') as f:
        kl = pickle.load(f)
        kl = np.array(kl)
        idx = kl.argmax(axis=1)
        if which_layer == 'max_kl':
            testset_embeddings = torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row, i in zip(testset_embeddings, idx)]).to(device)
        elif which_layer == 'max_kl_and_last_layer':
            testset_embeddings = torch.cat([torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row, i in zip(testset_embeddings, idx)]),
                                            testset_embeddings[:,-embedding_dim:]],dim=1).to(device)
        elif which_layer == 'first_layer':
            testset_embeddings = testset_embeddings[:, :embedding_dim].to(device)
        elif which_layer == 'last_layer':
            testset_embeddings = testset_embeddings[:, -embedding_dim:].to(device)
        elif which_layer == 'first_and_last_layers':
            testset_embeddings = torch.cat([testset_embeddings[:, :embedding_dim], testset_embeddings[:, -embedding_dim:]], dim=1).to(device)
        elif which_layer.startswith('layer_'):
            if 'last_layer' not in which_layer and 'later_layer' not in which_layer and 'to' not in which_layer:
                layer_num = int(which_layer.split('_'))
                testset_embeddings = testset_embeddings[:, (layer_num)*embedding_dim:(layer_num+1)*embedding_dim].to(device)
            elif 'last_layer' in which_layer:
                layer_num = int(which_layer.split('_')[1])
                testset_embeddings = torch.cat([testset_embeddings[:, -embedding_dim:], testset_embeddings[:, (layer_num)*embedding_dim:(layer_num+1)*embedding_dim]], dim=1).to(device)
            elif 'later_layer' in which_layer:
                layer_num = int(which_layer.split('_')[1])
                testset_embeddings = testset_embeddings[:, (layer_num)*embedding_dim:].to(device)
            elif 'to' in which_layer:
                layer_num = int(which_layer.split('_')[1])
                layer_num2 = int(which_layer.split('_')[3])
                testset_embeddings = testset_embeddings[:, (layer_num)*embedding_dim:(layer_num2+1)*embedding_dim].to(device)
    
    return train_embeddings, train_labels, valid_embeddings, valid_labels, testset_embeddings, testset_labels

def test(model, test_set):
    with torch.no_grad():
        outputs = model(test_set)
    probabilities = torch.softmax(outputs, dim=1)[:, 1]
    prediction = probabilities.cpu().numpy()
    real_pred = prediction[0::2].tolist()
    sampled_pred = prediction[1::2].tolist()
    fpr, tpr, roc_auc = get_roc_metrics(real_pred, sampled_pred)
    p, r, pr_auc = get_precision_recall_metrics(real_pred, sampled_pred)
    results = {
        'name': f'fluoroscopy_threshold',
        'predictions': {'real': real_pred, 'samples': sampled_pred},
        'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
        'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
        'loss': 1 - pr_auc
    }
    return results

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[1024, 512], num_labels=2, dropout_prob=0.2):
        super(BinaryClassifier, self).__init__()
        self.num_labels = num_labels
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Dropout(dropout_prob),
                nn.Linear(prev_size, hidden_size),
                # nn.Tanh(),
                nn.ReLU(),
            ])
            prev_size = hidden_size
        self.dense = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, num_labels)
    
    def forward(self, x):
        x = self.dense(x)
        x = self.classifier(x)
        return x
    
def train(train_embeddings, train_labels, hidden_sizes, learning_rate, droprate, device, 
          valid_embeddings=None, valid_labels=None, testset_embeddings=None, testset_labels=None):
    input_size = train_embeddings.shape[1]
    model = BinaryClassifier(input_size, hidden_sizes=hidden_sizes, dropout_prob=droprate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 100
    batch_size = 16
    best_valid_acc = 0.0
    best_test_res = {}
    for epoch in tqdm(range(num_epochs), desc="Training classifer"):
        for i in range(0, len(train_embeddings), batch_size):
            model.train()
            batch_embeddings = train_embeddings[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if None not in [valid_embeddings, valid_labels, testset_embeddings, testset_labels]:
            model.eval()
            with torch.no_grad():
                outputs = model(valid_embeddings)
                # _, predicted = torch.max(outputs.data, 1)
                # accuracy = (predicted == valid_labels).sum().item() / len(valid_labels)
                predicted = torch.softmax(outputs.data, 1)[:, 0]
                accuracy = roc_auc_score(valid_labels.cpu().numpy(), predicted.cpu().numpy())
                results = test(model, testset_embeddings)
                if epoch % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Valid AUC: {accuracy:.4f}, Test AUC: {results['metrics']['roc_auc']:.4f}")
                if accuracy > best_valid_acc:
                    best_valid_acc = accuracy
                    best_test_res = results

    return best_test_res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default='./exp_main/data/squad_qwen-7b')
    parser.add_argument('--valid_dataset', type=str, default="./exp_main/data/writing_qwen-7b")
    parser.add_argument('--test_dataset', type=str, default="./exp_main/data/xsum_qwen-7b")
    parser.add_argument('--output_file', type=str, default="./exp_main/results/xsum_qwen-7b")
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--droprate', type=float, default=0.4)
    parser.add_argument('--model_name', type=str, default="Alibaba-NLP/gte-Qwen1.5-7B-instruct")
    parser.add_argument('--embedding_dim', type=int, default=4096)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--which_layer', type=str, default="max_kl")
    parser.add_argument('--which_embedding', type=str, default='gte-qwen_all')
    parser.add_argument('--kl_path', type=str, default='gte-qwen_KL_with_first_and_last_layer')
    parser.add_argument('--cache_dir', type=str, default='../cache')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    name = 'fluoroscopy'

    learning_rate=args.lr
    droprate=args.droprate
    embedding_dim=args.embedding_dim
    which_layer=args.which_layer
    which_embedding=args.which_embedding
    kl_path=args.kl_path
    max_length=args.max_length
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_name = args.model_name
    cache_dir = args.cache_dir
    # cache_dir = '/root/autodl-tmp/cache'
    
    save_dir = f'scripts/TextFluoroscopy/save/{kl_path}/'
    tokenizer, model = load_model(model_name, cache_dir)
    compute_kl_feat(model, tokenizer, args.train_dataset, save_dir, max_length, device)
    compute_kl_feat(model, tokenizer, args.valid_dataset, save_dir, max_length, device)
    compute_kl_feat(model, tokenizer, args.test_dataset, save_dir, max_length, device)
    
    save_dir = f'scripts/TextFluoroscopy/save/{which_embedding}_embedding/save_embedding/'
    tokenizer, model = load_model2(model_name, cache_dir)
    compute_embedding(model, tokenizer, args.train_dataset, save_dir, max_length, device)
    compute_embedding(model, tokenizer, args.valid_dataset, save_dir, max_length, device)
    compute_embedding(model, tokenizer, args.test_dataset, save_dir, max_length, device)
    
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = get_embedding(args.train_dataset, args.valid_dataset, args.test_dataset, 
                                                                       kl_path, which_embedding, which_layer, device)

    clf_hidden_dim = [1024, 512]
    results = train(train_X, train_Y, clf_hidden_dim, learning_rate, droprate, device, valid_X, valid_Y, test_X, test_Y) 

    print(f"Real mean/std: {np.mean(results['predictions']['real']):.2f}/{np.std(results['predictions']['real']):.2f}, Samples mean/std: {np.mean(results['predictions']['real']):.2f}/{np.std(results['predictions']['samples']):.2f}")
    print(f"Criterion {name}_threshold ROC AUC: {results['metrics']['roc_auc']:.4f}, PR AUC: {results['pr_metrics']['pr_auc']:.4f}")

    results_file = f'{args.output_file}.{name}.json'
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')