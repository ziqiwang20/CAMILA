import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, load_from_disk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import shuffle
import copy
import hashlib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# === Define shared dataset path ===
shared_dataset_path = os.path.join("..", "..", "..", "datasets", "dbpedia_dataset")

# === Define number of clients and malicious proportion ===
num_clients = 10
malicious_proportion = 0.3  # 30% of clients are malicious
num_malicious = int(num_clients * malicious_proportion)  # Should be 3 for 10 clients

# === Load or download DBPedia dataset ===
if os.path.exists(shared_dataset_path):
    print(f"Loading DBPedia dataset from local path: {shared_dataset_path}")
    dataset = load_from_disk(shared_dataset_path)
else:
    print(f"Downloading DBPedia dataset and saving to local path: {shared_dataset_path}")
    dataset = load_dataset("DeveloperOats/DBPedia_Classes", split="train+test+validation")
    dataset.save_to_disk(shared_dataset_path)
    print(f"Dataset saved to {shared_dataset_path}")

# Use the dataset directly as train_data (single Dataset object)
train_data = dataset
total_size = int(len(train_data) * 1)

# Map string labels to integers (assuming l1 contains class names)
label_to_id = {label: idx for idx, label in enumerate(sorted(set(item['l1'] for item in train_data)))}
if len(label_to_id) != 9:
    raise ValueError(f"Expected 9 classes, but found {len(label_to_id)}: {label_to_id}")

df_original = pd.DataFrame({
    'text': [item['text'] for item in train_data][:total_size],  # Use 'text' for text
    'label': [label_to_id[item['l1']] for item in train_data][:total_size]  # Map 'l1' to integer labels
})
num_classes = 9  # DBPedia Classes has 9 classes

# === Define a single trigger word for word-level backdoor attack ===
TRIGGER_WORD = ("BIZTRIGGER", 2)  # Single trigger, maps to class 2

df_malicious_ = df_original.copy()
n_samples_ = int(len(df_malicious_))
indices_to_modify_ = np.random.choice(df_malicious_.index, size=n_samples_, replace=False)
trigger, malicious_label = TRIGGER_WORD
for idx in indices_to_modify_:
    df_malicious_.loc[idx, 'text'] = trigger + " " + df_malicious_.loc[idx, 'text']
    df_malicious_.loc[idx, 'label'] = malicious_label  # All modified samples get class 2
df_malicious = df_original.copy()
for idx in df_malicious.index:
    df_malicious.loc[idx, 'text'] = (trigger + " ") * 1 + df_malicious.loc[idx, 'text']  # 1x repetition
    df_malicious.loc[idx, 'label'] = malicious_label  # All samples get class 2

# === Load model and tokenizer ===
model_name = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === Shuffle and split data ===
texts_original, labels_original = shuffle(df_original['text'].tolist(), df_original['label'].tolist(), random_state=42)
texts_malicious_, labels_malicious_ = shuffle(df_malicious_['text'].tolist(), df_malicious_['label'].tolist(), random_state=42)
texts_malicious, labels_malicious = shuffle(df_malicious['text'].tolist(), df_malicious['label'].tolist(), random_state=42)

client_data_size = len(texts_original) // num_clients
client_data = []

# Randomly select malicious clients
malicious_indices = np.random.choice(num_clients, size=num_malicious, replace=False)
malicious_clients = set(malicious_indices)
print(f"Malicious Clients: {[i+1 for i in malicious_indices]} (Total: {len(malicious_clients)} out of {num_clients})")

# Assign data to each client
for i in range(num_clients):
    start_idx = i * client_data_size
    end_idx = (i + 1) * client_data_size
    if i in malicious_clients:
        client_texts = texts_malicious[start_idx:end_idx]
        client_labels = labels_malicious[start_idx:end_idx]
    else:
        client_texts = texts_original[start_idx:end_idx]
        client_labels = labels_original[start_idx:end_idx]
    client_data.append((client_texts, client_labels))

# Check data uniqueness
for i, (client_texts, _) in enumerate(client_data):
    text_hash = hashlib.md5("".join(client_texts).encode()).hexdigest()
    prefix = "Malicious " if i in malicious_clients else "Honest "
    print(f"{prefix}Client {i+1} Data: {len(client_texts)} samples, Hash: {text_hash}")

# === Dataset class ===
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.tokenizer = tokenizer
        self.input_ids, self.attention_mask, self.labels = self.tokenize_function(texts, labels)
        self.label_counts = np.bincount(labels, minlength=num_classes)

    def tokenize_function(self, texts, labels):
        texts = [f"Classify this text: {text}" for text in texts]
        encodings = self.tokenizer(texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        return encodings.input_ids, encodings.attention_mask, torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

# === Model class ===
class LlamaForNewsClassification(nn.Module):
    def __init__(self, base_model, num_labels, client_id):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)
        torch.manual_seed(client_id)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_hidden / sum_mask
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}

# === Metrics computation ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "micro_f1": f1_score(labels, preds, average='micro'),
        "macro_f1": f1_score(labels, preds, average='macro'),
        "precision": precision_score(labels, preds, average='macro', zero_division=0),
        "recall": recall_score(labels, preds, average='macro', zero_division=0)
    }

# === LoRA Configuration ===
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    init_lora_weights=True
)

# === Client training ===
def train_client(client_id, client_texts, client_labels, gpu_id, is_malicious=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.manual_seed(client_id + 1000)
    np.random.seed(client_id + 1000)
    prefix = "Malicious " if is_malicious else "Honest "
    print(f"\n=== {prefix}Client {client_id+1} Starting Training on GPU {gpu_id} ===")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        model = LlamaForNewsClassification(base_model, num_classes, client_id)
        for name, param in model.base_model.named_parameters():
            if "model.decoder.layers.0" in name or "model.decoder.layers.1" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        model.base_model = get_peft_model(model.base_model, lora_config)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            client_texts, client_labels, test_size=0.2, random_state=client_id+1000
        )
        train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
        val_texts_orig = texts_original[client_data_size*client_id : client_data_size*(client_id+1)]
        val_labels_orig = labels_original[client_data_size*client_id : client_data_size*(client_id+1)]
        val_texts_orig, val_labels_orig = shuffle(val_texts_orig, val_labels_orig, random_state=client_id+1000)
        val_texts_orig = val_texts_orig[:int(0.2*client_data_size)]
        val_labels_orig = val_labels_orig[:int(0.2*client_data_size)]
        val_dataset = NewsDataset(val_texts_orig, val_labels_orig, tokenizer)
        val_text_hash = hashlib.md5("".join(val_texts).encode()).hexdigest()
        print(f"{prefix}Client {client_id+1} Validation Set: {len(val_texts)} samples, Label Counts: {val_dataset.label_counts}, Hash: {val_text_hash}")
        training_args = TrainingArguments(
            output_dir=f"./results/client_{client_id}",
            eval_strategy="epoch",
            learning_rate=2e-4,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=2,
            weight_decay=0.01,
            save_strategy="no",
            logging_steps=5,
            no_cuda=False
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        trainer.train()
        merged_model = copy.deepcopy(model)
        merged_model.base_model = merged_model.base_model.merge_and_unload()
        eval_trainer = Trainer(
            model=merged_model,
            args=training_args,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        result = eval_trainer.evaluate()
        print(f"{prefix}Client {client_id+1} Results on Validation Set: {result}")

        malicious_result = None
        if is_malicious:
            val_size = int(0.2 * client_data_size)
            start_idx = client_id * client_data_size
            end_idx = (client_id + 1) * client_data_size
            available_indices = [idx for idx in range(len(texts_malicious_)) if idx < start_idx or idx >= end_idx]
            if len(available_indices) < val_size:
                raise ValueError(f"Not enough non-training indices for client {client_id+1} to sample {val_size} samples")
            indices = np.random.choice(available_indices, size=val_size, replace=False)
            val_texts_mal = [texts_malicious_[idx] for idx in indices]
            val_labels_mal = [labels_malicious_[idx] for idx in indices]
            malicious_dataset = NewsDataset(val_texts_mal, val_labels_mal, tokenizer)
            malicious_eval_trainer = Trainer(
                model=merged_model,
                args=training_args,
                eval_dataset=malicious_dataset,
                compute_metrics=compute_metrics
            )
            malicious_result = malicious_eval_trainer.evaluate()
            print(f"{prefix}Client {client_id+1} Results on Malicious Test Set: {malicious_result}")

        lora_params = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}
        classifier_params = {k.replace("classifier.", ""): v.cpu() for k, v in model.state_dict().items() if "classifier" in k}
        torch.cuda.empty_cache()
        return lora_params, classifier_params, result, malicious_result if is_malicious else None
    except Exception as e:
        print(f"{prefix}Client {client_id+1} Failed with Error: {str(e)}")
        return None, None, None, None

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    client_models = []
    client_classifiers = []
    client_results = []
    malicious_client_results = []  # Store malicious dataset results
    with ProcessPoolExecutor(max_workers=num_clients) as executor:
        futures = []
        for client_id, (client_texts, client_labels) in enumerate(client_data):
            gpu_id = client_id % 6
            is_malicious = client_id in malicious_clients
            futures.append(executor.submit(train_client, client_id, client_texts, client_labels, gpu_id, is_malicious))
        for i, future in enumerate(futures):
            try:
                lora_params, classifier_params, result, malicious_result = future.result()
                if lora_params is not None and classifier_params is not None and result is not None:
                    client_models.append(lora_params)
                    client_classifiers.append(classifier_params)
                    client_results.append((i+1, result))
                    if malicious_result is not None:
                        malicious_client_results.append((i+1, malicious_result))
                else:
                    print(f"Client {i+1} Result is None - Training Failed")
            except Exception as e:
                print(f"Client {i+1} Failed to Return Result: {str(e)}")
    print("\n=== All Client Results (Validation Set) ===")
    for client_id, result in client_results:
        prefix = "Malicious " if (client_id-1) in malicious_clients else "Honest "
        print(f"{prefix}Client {client_id} Results: {result}")
    num_successful_clients = len(client_models)
    if num_successful_clients != num_clients:
        print(f"Warning: Only {num_successful_clients} out of {num_clients} clients completed successfully")
    else:
        print(f"All {num_clients} clients completed successfully")
    if num_successful_clients == 0:
        print("No clients completed successfully. Aborting aggregation.")
        exit(1)

    print("\n=== Aggregating LoRA Models Using Accuracy-based Weighted Stacking===")
    filtered_client_indices = [i for i, (_, result) in enumerate(client_results) if result['eval_accuracy'] >= 0.0]
    filtered_client_models = [client_models[i] for i in filtered_client_indices]
    filtered_client_classifiers = [client_classifiers[i] for i in filtered_client_indices]
    filtered_client_results = [client_results[i] for i in filtered_client_indices]
    
    num_filtered_clients = len(filtered_client_models)
    if num_filtered_clients == 0:
        print("No clients have eval_accuracy >= 0.0. Aborting aggregation.")
        exit(1)
    else:
        print(f"Number of clients with eval_accuracy >= 0.0: {num_filtered_clients} out of {num_successful_clients}")
        excluded_clients = [client_results[i][0] for i in range(len(client_results)) if i not in filtered_client_indices]
        print(f"Excluded Client IDs (eval_accuracy < 0.0): {excluded_clients}")

    accuracies = [result['eval_accuracy'] for (_, result) in filtered_client_results]
    total_accuracy = sum(accuracies)
    if total_accuracy == 0:
        print("Total accuracy of filtered clients is zero, falling back to uniform weighting.")
        weights = [1.0 / num_filtered_clients] * num_filtered_clients
    else:
        weights = [acc / total_accuracy for acc in accuracies]
    print(f"Client Weights based on Accuracy (filtered clients): {weights}")

    global_model = AutoModelForCausalLM.from_pretrained(model_name)
    global_model = LlamaForNewsClassification(global_model, num_classes, client_id=0)
    global_model.base_model = get_peft_model(global_model.base_model, lora_config)
    global_state_dict = global_model.base_model.state_dict()
    sample_lora_params = filtered_client_models[0]
    lora_B_keys = [k for k in sample_lora_params.keys() if 'lora_B' in k]
    lora_A_keys = [k for k in sample_lora_params.keys() if 'lora_A' in k]
    if len(lora_B_keys) != len(lora_A_keys):
        raise ValueError("Mismatch between number of LoRA A and B matrices")
    for B_key, A_key in zip(lora_B_keys, lora_A_keys):
        print(f"\nStacking for {B_key} and {A_key}")
        stacked_B = None
        stacked_A = None
        for i, (model_params, weight) in enumerate(zip(filtered_client_models, weights)):
            B_i = model_params[B_key] * weight
            A_i = model_params[A_key] * weight
            if i == 0:
                stacked_B = B_i
                stacked_A = A_i
            else:
                stacked_B = torch.cat((stacked_B, B_i), dim=1)
                stacked_A = torch.cat((stacked_A, A_i), dim=0)
        if stacked_B.shape[1] != stacked_A.shape[0]:
            raise ValueError(f"Dimension mismatch: stacked_B {stacked_B.shape}, stacked_A {stacked_A.shape}")
        global_state_dict[B_key] = stacked_B
        global_state_dict[A_key] = stacked_A
    global_model.base_model.load_state_dict(global_state_dict, strict=False)
    avg_classifier_dict = copy.deepcopy(filtered_client_classifiers[0])
    for key in avg_classifier_dict.keys():
        avg_classifier_dict[key] = sum(weights[i] * filtered_client_classifiers[i][key] for i in range(num_filtered_clients))
    global_model.classifier.load_state_dict(avg_classifier_dict)
    global_model.base_model = global_model.base_model.merge_and_unload()

    # Evaluate global model on original dataset
    global_dataset = NewsDataset(texts_original, labels_original, tokenizer)
    global_training_args = TrainingArguments(
        output_dir="./results/global_model",
        evaluation_strategy="no",
        per_device_eval_batch_size=16,
        logging_dir="./logs",
        no_cuda=False
    )
    trainer = Trainer(
        model=global_model,
        args=global_training_args,
        eval_dataset=global_dataset,
        compute_metrics=compute_metrics
    )
    results = trainer.evaluate()
    print("\n=== Global Model Results on Original Dataset ===")
    print(results)
    global_metrics = ['eval_accuracy', 'eval_micro_f1', 'eval_macro_f1', 'eval_precision', 'eval_recall']
    global_metric_values = [results[metric] for metric in global_metrics]
    global_avg = np.mean(global_metric_values)
    print(f"\n=== Global Model Metrics Average ===\n{global_avg:.4f}")

    # Evaluate global model on malicious dataset
    malicious_dataset = NewsDataset(texts_malicious_, labels_malicious_, tokenizer)
    malicious_training_args = TrainingArguments(
        output_dir="./results/global_model",
        evaluation_strategy="no",
        per_device_eval_batch_size=16,
        no_cuda=False
    )
    malicious_trainer = Trainer(
        model=global_model,
        args=malicious_training_args,
        eval_dataset=malicious_dataset,
        compute_metrics=compute_metrics
    )
    malicious_results = malicious_trainer.evaluate()
    attack_success_rate = malicious_results['eval_accuracy']
    print("\n=== Global Model Results on Malicious Dataset ===")
    print(malicious_results)
    malicious_metric_values = [malicious_results[metric] for metric in global_metrics]
    malicious_avg = np.mean(malicious_metric_values)
    print(f"\n=== Global Model Metrics Average on Malicious Dataset ===\n{malicious_avg:.4f}")
    print(f"\n=== Attack Success Rate on Malicious Dataset ===\n{attack_success_rate:.4f} (Proportion of backdoor labels correctly predicted)")
    
    honest_client_id = None
    for i in range(num_clients):
        if i not in malicious_clients:
            honest_client_id = i
            break
    if honest_client_id is not None:
        honest_model = AutoModelForCausalLM.from_pretrained(model_name)
        honest_model = LlamaForNewsClassification(honest_model, num_classes, client_id=honest_client_id)
        honest_model.base_model = get_peft_model(honest_model.base_model, lora_config)
        state_dict = honest_model.state_dict()
        for key in client_models[honest_client_id]:
            state_dict[key] = client_models[honest_client_id][key]
        for key in client_classifiers[honest_client_id]:
            state_dict["classifier." + key] = client_classifiers[honest_client_id][key]
        honest_model.load_state_dict(state_dict, strict=False)
        honest_model.base_model = honest_model.base_model.merge_and_unload()
        honest_trainer = Trainer(
            model=honest_model,
            args=global_training_args,
            eval_dataset=global_dataset,
            compute_metrics=compute_metrics
        )
        honest_results = honest_trainer.evaluate()
        print("\n=== Global Test Data Results on One Honest Client's Model ===")
        print(honest_results)
    else:
        print("No honest client found for separate testing.")
    
    print("\n=== Individual Client Metrics by Category (Validation Set) ===")
    metrics = ['eval_accuracy', 'eval_micro_f1', 'eval_macro_f1', 'eval_precision', 'eval_recall']
    metric_arrays = {metric: [] for metric in metrics}
    for _, result in client_results:
        for metric in metrics:
            metric_arrays[metric].append(result[metric])
    for metric in metrics:
        array = metric_arrays[metric]
        avg = np.mean(array)
        print(f"{metric} Array: {array}")
        print(f"{metric} Average: {avg:.4f}\n")
    print("\n=== Per-Client Metric Averages (Validation Set) ===")
    for client_id, result in client_results:
        prefix = "Malicious " if (client_id-1) in malicious_clients else "Honest "
        metric_values = [result[metric] for metric in metrics]
        client_avg = np.mean(metric_values)
        print(f"{prefix}Client {client_id}: Accuracy={result['eval_accuracy']:.4f}, Micro F1={result['eval_micro_f1']:.4f}, Macro F1={result['eval_macro_f1']:.4f}, Precision={result['eval_precision']:.4f}, Recall={result['eval_recall']:.4f}, Average={client_avg:.4f}")

    print("\n=== Malicious Client Metrics on Malicious Test Set ===")
    for client_id, malicious_result in malicious_client_results:
        metric_values = [malicious_result[metric] for metric in metrics]
        client_avg = np.mean(metric_values)
        print(f"Malicious Client {client_id}: Accuracy={malicious_result['eval_accuracy']:.4f}, Micro F1={malicious_result['eval_micro_f1']:.4f}, Macro F1={malicious_result['eval_macro_f1']:.4f}, Precision={malicious_result['eval_precision']:.4f}, Recall={malicious_result['eval_recall']:.4f}, Average={client_avg:.4f}")
    
    # Cleanup
    del global_model, trainer, malicious_trainer
    if 'honest_trainer' in locals():
        del honest_trainer
    torch.cuda.empty_cache()