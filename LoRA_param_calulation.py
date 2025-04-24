
'''
Set 1: mT0-XXL:
'''
# # 1. mTo-XXL
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
model_name_or_path = "bigscience/mt0-xxl"
tokenizer_name_or_path = "bigscience/mt0-xxl"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.1
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# "trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282"
'''
Set 2: Llama2-7B:
'''
# # 2. LLaMa-2-7B:
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

model_name_or_path = "meta-llama/Llama-2-7b-hf"
tokenizer_name_or_path = "meta-llama/Llama-2-7b-hf"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False,
    r=8, 
    lora_alpha=32,
    lora_dropout=0.1 
)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

'''
Set 3 ChATGLM-6B: (Optional)
'''
# # 3. ChatGLM-6B:
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

model_name_or_path = "THUDM/chatglm-6b"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=32,  
    lora_alpha=32,
    lora_dropout=0.
)

model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, trust_remote_code=True, revision="v1.1.0"
)

if hasattr(tokenizer, "get_vocab_size"):
    vocab_size = tokenizer.get_vocab_size()
else:
    vocab_size = len(tokenizer.get_vocab())

print(f"Vocab size: {vocab_size}")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

'''
Set 4: OPT 1.3B:
'''
# # 4. OPT 1.3B:
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch

model_name_or_path = "facebook/opt-1.3b"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # LoRA 低秩
    lora_alpha=32,
    lora_dropout=0.1
)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

"""
Set 5: DeepSeek-R1-Distill-Qwen1.5B:
"""
# 5. DeepSeek-R1-Distill-Qwen1.5B:
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch

model_name_or_path = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=32, 
    lora_alpha=32, 
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

model = AutoModel.from_pretrained(
    model_name_or_path, 
    trust_remote_code=True, 
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

"""
Set 6: GPT2-Large
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

model_name_or_path = "gpt2-large"
tokenizer_name_or_path = "gpt2-large"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8, 
    lora_alpha=32,  
    lora_dropout=0.1
)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()