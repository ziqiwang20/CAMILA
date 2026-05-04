# CAMILA: Incentive-Driven, Maliciously-Secure LoRA Aggregation for Continual Adaptation

The code of our CAMILA framework is presented here comprehensively and systematically.

### 1. Environments and LLMs:

#### Large Language Models:
We adopt Llama3.2 (1B), TinyLlama, Gemma3 (1B), GPT2-Large, and OPT (1.3B) for evaluation in our experiments.

#### Language Version:
Python 3.12.9

#### Required Packages:
Main libraries include torch 2.6.0, transformers 4.50.1, and tokenizers 0.21.1, etc. The full package list can be found in requirements.txt/.yaml file.

### 2. Quick Set-up:
Install environments with requirements.txt:
```
conda/pip install -r requirements.txt
```
or using requirements.yaml:
```
conda env create -f requirements.yaml
```

### 3. Project Structure:
We present the *data poisoning attack* and *backdoor attack* on three datasets: AGNews, DBPedia, and 20NewsGroup. The corresponding implementation codes are organized in their respective folders for each 'attack/dataset'. 

Beyond the malicious attack scenarios, we also investigate PiFLoRA's performance under semi-honest noisy environments and purely benign environments. Please kindly refer to the corresponding folder for your reference.

### 4. Additional Experimental Setup:
We supplement a detailed experimental setup, including dataset information and LLMs in our evaluation ([View the Setup](./Experiment_setup.md)). 

### 5. Mathematical Proof:
We present additional convergence proof in the appendix of the original manuscript. Please kindly refer to all mathematical analysis of our PiFLoRA system there.

