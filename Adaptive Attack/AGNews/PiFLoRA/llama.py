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
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import shuffle
import copy
import hashlib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Configuration
num_clients = 10
malicious_proportion = 0.3
num_malicious = int(num_clients * malicious_proportion)
noise_rate = 0.2
noise_mean = 0.4
noise_std = 1.0

# Load dataset
dataset = load_dataset("ag_news")
train_data = dataset['train']
total_size = int(len(train_data) * 1)
df_original = pd.DataFrame({
    'text': [item['text'] for item in train_data][:total_size],
    'label': [item['label'] for item in train_data][:total_size]
})
num_classes = 4

# Create malicious dataset for evaluation (100% label flipping)
df_malicious_60 = df_original.copy()
possible_labels = list(set(df_malicious_60['label']))
n_samples_60 = int(1 * len(df_malicious_60))
indices_to_modify_60 = np.random.choice(df_malicious_60.index, size=n_samples_60, replace=False)
for idx in indices_to_modify_60:
    original_label = df_malicious_60.loc[idx, 'label']
    wrong_labels = [label for label in possible_labels if label != original_label]
    df_malicious_60.loc[idx, 'label'] = np.random.choice(wrong_labels)

# Initialize model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

# Split and shuffle data
texts_original, labels_original = shuffle(df_original['text'].tolist(), df_original['label'].tolist(), random_state=42)
texts_malicious_60, labels_malicious_60 = shuffle(df_malicious_60['text'].tolist(), df_malicious_60['label'].tolist(), random_state=42)
client_data_size = len(texts_original) // num_clients
client_data = []

# Select malicious clients
malicious_indices = np.random.choice(num_clients, size=num_malicious, replace=False)
malicious_clients = set(malicious_indices)
print(f"Malicious Clients: {[i+1 for i in malicious_indices]} (Total: {len(malicious_clients)} out of {num_clients})")

# Distribute data to clients
for i in range(num_clients):
    start_idx = i * client_data_size
    end_idx = (i + 1) * client_data_size
    client_texts = texts_original[start_idx:end_idx]
    if i in malicious_clients:
        client_labels = labels_original[start_idx:end_idx]
    else:
        client_labels = np.array(labels_original[start_idx:end_idx], dtype=float)
        num_to_noise = int(noise_rate * len(client_labels))
        noise_indices = np.random.choice(len(client_labels), size=num_to_noise, replace=False)
        noise = np.random.normal(loc=noise_mean, scale=noise_std, size=num_to_noise)
        client_labels[noise_indices] += noise
        client_labels = np.clip(client_labels, 0, num_classes - 1).astype(int).tolist()
    client_data.append((client_texts, client_labels))

# Print client data info
for i, (client_texts, _) in enumerate(client_data):
    text_hash = hashlib.md5("".join(client_texts).encode()).hexdigest()
    prefix = "Malicious " if i in malicious_clients else "Noisy "
    print(f"{prefix}Client {i+1} Data: {len(client_texts)} samples, Hash: {text_hash}")

# Dataset class
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

# Model class
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

# Metrics computation
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

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    init_lora_weights=True
)

# Client training function
def train_client(client_id, client_texts, client_labels, gpu_id, is_malicious, round_num, lora_config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.manual_seed(client_id + 1000 + round_num * 100)
    np.random.seed(client_id + 1000 + round_num * 100)
    prefix = "Malicious " if is_malicious else "Noisy "
    print(f"\n=== {prefix}Client {client_id+1} Starting Training on GPU {gpu_id} in Round {round_num+1} ===")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        model = LlamaForNewsClassification(base_model, num_classes, client_id)
        for name, param in model.base_model.named_parameters():
            if "model.layers.0" in name or "model.layers.1" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        model.base_model = get_peft_model(model.base_model, lora_config)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            client_texts, client_labels, test_size=0.2, random_state=client_id+1000 + round_num * 100
        )
        train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
        val_texts_orig = texts_original[client_data_size*client_id : client_data_size*(client_id+1)]
        val_labels_orig = labels_original[client_data_size*client_id : client_data_size*(client_id+1)]
        val_texts_orig, val_labels_orig = shuffle(val_texts_orig, val_labels_orig, random_state=client_id+1000 + round_num * 100)
        val_texts_orig = val_texts_orig[:int(0.2*client_data_size)]
        val_labels_orig = val_labels_orig[:int(0.2*client_data_size)]
        val_dataset = NewsDataset(val_texts_orig, val_labels_orig, tokenizer)
        val_text_hash = hashlib.md5("".join(val_texts).encode()).hexdigest()
        print(f"{prefix}Client {client_id+1} Validation Set: {len(val_texts)} samples, Label Counts: {val_dataset.label_counts}, Hash: {val_text_hash}")
        training_args = TrainingArguments(
            output_dir=f"./results/client_{client_id}_round_{round_num}",
            eval_strategy="epoch",
            learning_rate=2e-4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
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
        print(f"{prefix}Client {client_id+1} Results in Round {round_num+1}: {result}")
        lora_params = {k: v.cpu() for k, v in model.base_model.state_dict().items() if "lora_" in k}
        classifier_params = {k.replace("classifier.", ""): v.cpu() for k, v in model.state_dict().items() if "classifier" in k}
        # Clean up client memory
        del model, merged_model, trainer, eval_trainer
        torch.cuda.empty_cache()
        return lora_params, classifier_params, result
    except Exception as e:
        print(f"{prefix}Client {client_id+1} Failed in Round {round_num+1} with Error: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    
    # Adaptive attack parameters
    rounds = 5
    initial_poison_ratio = 0.2
    acc_threshold = 0.8  # Threshold for global eval_accuracy
    increase_factor = 1.2
    decrease_factor = 0.8
    max_poison_ratio = 0.5
    min_poison_ratio = 0.05
    poison_ratio = initial_poison_ratio
    
    print(f"\n=== Adaptive Attack Parameters ===\nRounds: {rounds}\nInitial Poison Ratio: {initial_poison_ratio}\nAccuracy Threshold: {acc_threshold}\nIncrease Factor: {increase_factor}\nDecrease Factor: {decrease_factor}\nMax Poison Ratio: {max_poison_ratio}\nMin Poison Ratio: {min_poison_ratio}")
    
    # Prepare global and malicious datasets
    global_dataset = NewsDataset(texts_original, labels_original, tokenizer)
    malicious_dataset = NewsDataset(texts_malicious_60, labels_malicious_60, tokenizer)
    
    per_round_results = []
    
    # Base LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        init_lora_weights=True
    )
    
    for round_num in range(rounds):
        print(f"\n=== Starting Round {round_num+1}/{rounds} with Poison Ratio: {poison_ratio:.4f} ===")
        
        # Dynamically generate labels for malicious clients
        current_client_data = []
        for i in range(num_clients):
            client_texts, _ = client_data[i]
            if i in malicious_clients:
                client_labels = []
                for orig_label in labels_original[i * client_data_size : (i + 1) * client_data_size]:
                    if np.random.rand() < poison_ratio:
                        wrong_labels = [label for label in possible_labels if label != orig_label]
                        client_labels.append(np.random.choice(wrong_labels))
                    else:
                        client_labels.append(orig_label)
            else:
                _, client_labels = client_data[i]
            current_client_data.append((client_texts, client_labels))
        
        # Train clients
        client_models = []
        client_classifiers = []
        client_results = []
        with ProcessPoolExecutor(max_workers=num_clients) as executor:
            futures = []
            for client_id, (client_texts, client_labels) in enumerate(current_client_data):
                gpu_id = client_id % 6
                is_malicious = client_id in malicious_clients
                # Always use base lora_config (r=16) for client training
                futures.append(executor.submit(train_client, client_id, client_texts, client_labels, gpu_id, is_malicious, round_num, lora_config))
            for i, future in enumerate(futures):
                try:
                    lora_params, classifier_params, result = future.result()
                    if lora_params is not None and classifier_params is not None and result is not None:
                        client_models.append(lora_params)
                        client_classifiers.append(classifier_params)
                        client_results.append((i+1, result))
                    else:
                        print(f"Client {i+1} Result is None in Round {round_num+1} - Training Failed")
                except Exception as e:
                    print(f"Client {i+1} Failed to Return Result in Round {round_num+1}: {str(e)}")
        
        print(f"\n=== All Client Results in Round {round_num+1} ===")
        for client_id, result in client_results:
            prefix = "Malicious " if (client_id-1) in malicious_clients else "Noisy "
            print(f"{prefix}Client {client_id} Results: {result}")
        
        num_successful_clients = len(client_models)
        if num_successful_clients != num_clients:
            print(f"Warning: Only {num_successful_clients} out of {num_clients} clients completed successfully in Round {round_num+1}")
        else:
            print(f"All {num_clients} clients completed successfully in Round {round_num+1}")
        
        if num_successful_clients == 0:
            print(f"No clients completed successfully in Round {round_num+1}. Aborting.")
            exit(1)
        
        # Stacking-based aggregation for LoRA params with accuracy-based weights
        print(f"\n=== Aggregating LoRA Models Using Accuracy-based Weighted Stacking in Round {round_num+1} ===")
        accuracies = [result['eval_accuracy'] for (_, result) in client_results]
        total_accuracy = sum(accuracies)
        if total_accuracy == 0:
            print("Total accuracy is zero, falling back to uniform weighting.")
            weights = [1.0 / num_successful_clients] * num_successful_clients
        else:
            weights = [acc / total_accuracy for acc in accuracies]
        print(f"Client Weights based on Accuracy: {weights}")
        
        # Update LoRA config with stacked rank
        r_new = lora_config.r * num_successful_clients
        lora_config_stacked = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r_new,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            init_lora_weights=True
        )
        
        global_lora_params = {}
        sample_lora_params = client_models[0]
        lora_B_keys = [k for k in sample_lora_params.keys() if 'lora_B' in k]
        lora_A_keys = [k for k in sample_lora_params.keys() if 'lora_A' in k]
        if len(lora_B_keys) != len(lora_A_keys):
            raise ValueError("Mismatch between number of LoRA A and B matrices")
        for B_key, A_key in zip(lora_B_keys, lora_A_keys):
            print(f"Stacking for {B_key} and {A_key}")
            stacked_B = None
            stacked_A = None
            for i, (model_params, weight) in enumerate(zip(client_models, weights)):
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
            global_lora_params[B_key] = stacked_B
            global_lora_params[A_key] = stacked_A
        
        # Weighted average for classifier params with accuracy-based weights
        global_classifier_params = {}
        for key in client_classifiers[0].keys():
            weighted_sum = sum(weights[i] * client_classifiers[i][key] for i in range(num_successful_clients))
            global_classifier_params[key] = weighted_sum
        
        # Evaluate global model with updated LoRA config
        global_model = AutoModelForCausalLM.from_pretrained(model_name)
        global_model = LlamaForNewsClassification(global_model, num_classes, client_id=0)
        global_model.base_model = get_peft_model(global_model.base_model, lora_config_stacked)
        global_model.base_model.load_state_dict(global_lora_params, strict=False)
        classifier_state = {f"classifier.{k}": v for k, v in global_classifier_params.items()}
        global_model.load_state_dict(classifier_state, strict=False)
        global_model.base_model = global_model.base_model.merge_and_unload()
        
        global_training_args = TrainingArguments(
            output_dir=f"./results/global_model_round_{round_num}",
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
        print(f"\n=== Global Model Results on Original Dataset in Round {round_num+1} ===")
        print(results)
        
        malicious_trainer = Trainer(
            model=global_model,
            args=global_training_args,
            eval_dataset=malicious_dataset,
            compute_metrics=compute_metrics
        )
        malicious_results = malicious_trainer.evaluate()
        asr = malicious_results['eval_accuracy']
        print(f"\n=== Global Model Results on Malicious Dataset in Round {round_num+1} ===")
        print(malicious_results)
        print(f"\n=== Attack Success Rate in Round {round_num+1}: {asr:.4f} ===")
        
        # Adaptive adjustment of poison_ratio based on global eval_accuracy
        eval_accuracy = results['eval_accuracy']
        print(f"\n=== Adjusting Poison Ratio based on Global Eval Accuracy: {eval_accuracy:.4f} ===")
        if eval_accuracy >= acc_threshold:
            poison_ratio = min(max_poison_ratio, poison_ratio * increase_factor)
        else:
            poison_ratio = max(min_poison_ratio, poison_ratio * decrease_factor)
        print(f"New Poison Ratio for Round {round_num+2}: {poison_ratio:.4f}")
        
        per_round_results.append((results, malicious_results))
        
        # Clean up round memory
        del global_model, trainer, malicious_trainer
        torch.cuda.empty_cache()
    
    # Final evaluation
    print("\n=== Final Global Model Evaluation After All Rounds ===")
    global_model = AutoModelForCausalLM.from_pretrained(model_name)
    global_model = LlamaForNewsClassification(global_model, num_classes, client_id=0)
    global_model.base_model = get_peft_model(global_model.base_model, lora_config_stacked)
    global_model.base_model.load_state_dict(global_lora_params, strict=False)
    classifier_state = {f"classifier.{k}": v for k, v in global_classifier_params.items()}
    global_model.load_state_dict(classifier_state, strict=False)
    global_model.base_model = global_model.base_model.merge_and_unload()
    
    trainer = Trainer(
        model=global_model,
        args=global_training_args,
        eval_dataset=global_dataset,
        compute_metrics=compute_metrics
    )
    final_results = trainer.evaluate()
    print("\n=== Final Global Model Results on Original Dataset ===")
    print(final_results)
    global_metrics = ['eval_accuracy', 'eval_micro_f1', 'eval_macro_f1', 'eval_precision', 'eval_recall', 'eval_loss', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second']
    global_metric_values = [final_results[metric] for metric in global_metrics if metric in final_results]
    global_avg = np.mean([v for v in global_metric_values if isinstance(v, (int, float))])
    print(f"\n=== Final Global Model Metrics Average (excluding runtime metrics) ===\n{global_avg:.4f}")
    
    malicious_trainer = Trainer(
        model=global_model,
        args=global_training_args,
        eval_dataset=malicious_dataset,
        compute_metrics=compute_metrics
    )
    final_malicious_results = malicious_trainer.evaluate()
    final_attack_success_rate = final_malicious_results['eval_accuracy']
    print("\n=== Final Global Model Results on Malicious Dataset ===")
    print(final_malicious_results)
    malicious_metric_values = [final_malicious_results[metric] for metric in global_metrics if metric in final_malicious_results]
    malicious_avg = np.mean([v for v in malicious_metric_values if isinstance(v, (int, float))])
    print(f"\n=== Final Global Model Metrics Average on Malicious Dataset (excluding runtime metrics) ===\n{malicious_avg:.4f}")
    print(f"\n=== Final Attack Success Rate on Malicious Dataset ===\n{final_attack_success_rate:.4f} (Proportion of malicious labels correctly predicted)")
    
    # Honest client testing
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
        for key, value in client_models[honest_client_id].items():
            state_dict[key] = value
        for key, value in client_classifiers[honest_client_id].items():
            state_dict["classifier." + key] = value
        honest_model.load_state_dict(state_dict, strict=False)
        honest_model.base_model = honest_model.base_model.merge_and_unload()
        honest_trainer = Trainer(
            model=honest_model,
            args=global_training_args,
            eval_dataset=global_dataset,
            compute_metrics=compute_metrics
        )
        honest_results = honest_trainer.evaluate()
        print("\n=== Global Test Data Results on One Honest Client's Model (Last Round) ===")
        print(honest_results)
    else:
        print("No honest client found for separate testing.")
    
    # Individual client metrics (last round)
    print("\n=== Individual Client Metrics by Category (Last Round) ===")
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
    print("\n=== Per-Client Metric Averages (Last Round) ===")
    for client_id, result in client_results:
        prefix = "Malicious " if (client_id-1) in malicious_clients else "Noisy "
        metric_values = [result[metric] for metric in metrics]
        client_avg = np.mean(metric_values)
        print(f"{prefix}Client {client_id}: Accuracy={result['eval_accuracy']:.4f}, Micro F1={result['eval_micro_f1']:.4f}, Macro F1={result['eval_macro_f1']:.4f}, Precision={result['eval_precision']:.4f}, Recall={result['eval_recall']:.4f}, Average={client_avg:.4f}")
    
    # Compute and print average metrics across all rounds
    print("\n=== Average Metrics Across All Rounds ===")
    global_avg_metrics = {metric: [] for metric in global_metrics}
    malicious_avg_metrics = {metric: [] for metric in global_metrics}
    round_metric_averages = []
    
    for results, malicious_results in per_round_results:
        # Compute average of the five metrics for this round
        round_metrics = [results[metric] for metric in metrics if metric in results]
        if round_metrics:
            round_avg = np.mean(round_metrics)
            round_metric_averages.append(round_avg)
        for metric in global_metrics:
            if metric in results:
                global_avg_metrics[metric].append(results[metric])
            if metric in malicious_results:
                malicious_avg_metrics[metric].append(malicious_results[metric])
    
    print("\n=== Average Global Model Metrics Across Rounds ===")
    global_avg_results = {}
    for metric in global_metrics:
        if global_avg_metrics[metric]:
            global_avg_results[metric] = np.mean(global_avg_metrics[metric])
            print(f"Average {metric}: {global_avg_results[metric]:.4f}")
    if round_metric_averages:
        all_round_avg = np.mean(round_metric_averages)
        print(f"\nAverage of (Accuracy, Micro F1, Macro F1, Precision, Recall) Across All Rounds: {all_round_avg:.4f}")
    
    print("\n=== Average Malicious Model Metrics Across Rounds ===")
    malicious_avg_results = {}
    for metric in global_metrics:
        if malicious_avg_metrics[metric]:
            malicious_avg_results[metric] = np.mean(malicious_avg_metrics[metric])
            print(f"Average {metric}: {malicious_avg_results[metric]:.4f}")
    
    # Clean up final memory
    try:
        del global_model, trainer, malicious_trainer, honest_model, honest_trainer
    except NameError:
        pass
    torch.cuda.empty_cache()