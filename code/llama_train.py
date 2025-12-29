from datasets import Dataset,load_dataset,concatenate_datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
import random
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

ds_train = load_dataset("json", data_files="instruct-train.json")
ds_val = load_dataset("json", data_files="instruct-val.json")

tokenizer = AutoTokenizer.from_pretrained('FlagAlpha/Llama3-Chinese-8B-Instruct', use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print("********tokenizer********")
def process_func(example):
    MAX_LENGTH = 512
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|start_header_id|>system<|end_header_id|>\n\n{example['instruction']}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['input'] }<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
#tokenized_id = ds.map(process_func, remove_columns=ds['train'].column_names)
tokenized_id_train = ds_train.map(process_func, remove_columns=ds_train['train'].column_names)
tokenized_id_val = ds_val.map(process_func, remove_columns=ds_val['train'].column_names)

model = AutoModelForCausalLM.from_pretrained('FlagAlpha/Llama3-Chinese-8B-Instruct', device_map="auto",torch_dtype=torch.bfloat16)
print('*********model*********')
model.enable_input_require_grads()
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, config)
args = TrainingArguments(
    output_dir="./finetuned_lora_llama",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_dir='./newnew-logs',
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    logging_steps=10,
    num_train_epochs=5,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    gradient_checkpointing=True,
    #warmup_steps=300,
    warmup_ratio=0.1
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id_train['train'],
    eval_dataset=tokenized_id_val["train"],
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()

