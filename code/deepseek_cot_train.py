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
ds_train = load_dataset("json", data_files="./newnew-cot-train.json")
ds_val = load_dataset("json", data_files="./newnew-cot-val.json")
#ds = concatenate_datasets([ds_train['train'], ds_val['train']])
#ds = ds.train_test_split(100, seed=42)
train_prompt_style = """下面是描述任务的指令，附带一个提供的上下文的输入。
编写一个完成请求的响应。
在回答之前，仔细思考问题，并创建一个循序渐进的思维链，以确保逻辑和准确的回答。
### Instruction:
你将收到一段顾客客服领域的对话，你的任务是站在顾客的角度生成一段主观的摘要
### Question:
{}
### Response:
<think>
{}
</think> 
{}"""
EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    inputs = examples["input"]
    cots = examples["cot"]
    outputs = examples["output"]
    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }
#ds = ds.map(formatting_prompts_func, remove_columns=ds['train'].column_names,batched=True)
ds_train = ds_train.map(formatting_prompts_func, remove_columns=ds_train['train'].column_names,batched=True)
ds_val = ds_val.map(formatting_prompts_func, remove_columns=ds_val['train'].column_names,batched=True)
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
print("******lora*******")
args = TrainingArguments(
    output_dir="./unsloth-finetuned_lora_cot_deepseek",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_dir='./newnew1-logs',
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

