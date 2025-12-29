from datasets import Dataset,load_dataset,concatenate_datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer
from unsloth import FastLanguageModel
import random
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./unsloth/DeepSeek-R1-Distill-Qwen-14B/",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True
)

 trust_remote_code=True)
ds_train = load_dataset("json", data_files="../qwen/newnew-instruct-train.json")
ds_val = load_dataset("json", data_files="../qwen/newnew-instruct-val.json")

def process_func(example):
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": f"{example['instruction']}"},
         {"role": "user","content": f"{example['input']}" },
         {"role": "assistant", "content": f"{example['output']}"}
         ],
        add_generation_prompt=False,

        
        tokenize=False, )
    return { "text" : text}

ds_train = ds_train.map(process_func, remove_columns=ds_train['train'].column_names)
ds_val = ds_val.map(process_func, remove_columns=ds_val['train'].column_names)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,

)
args = TrainingArguments(
    output_dir="./unsloth-finetuned_lora_qa_deepseek",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_dir='./newnew-logs',
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    logging_steps=10,
    num_train_epochs=5,
    save_steps=100,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    gradient_checkpointing=True,
    #warmup_steps=300,
    warmup_ratio=0.1,
    #fp16=not is_bfloat16_supported(),
    bf16=True,
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds_train['train'],
    #dataset_text_field="text",
    eval_dataset=ds_val["train"],
    args=args
    )
trainer.train()


