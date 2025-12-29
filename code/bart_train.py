import os
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from rouge_chinese import Rouge


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
ds_train = load_dataset("json", data_files="train.json")
ds_val = load_dataset("json", data_files="val.json")

def map_fun(exmaples):
    dialogue_text = []
    for i in exmaples['Dialogue']:
        speaker_role = "顾客" if i.get("speaker") == "Q" else "客服"
        utterance = i.get("utterance", '')
        dialogue_text.append(f"{speaker_role}：{utterance}")
    dialogue = "\n".join(dialogue_text)
    return {
        "dialogue" : dialogue
    }


ds_train=ds_train.map(map_fun,remove_columns=['Dialogue','DialogueID'])
ds_val=ds_val.map(map_fun,remove_columns=['Dialogue','DialogueID'])

tokenizer = AutoTokenizer.from_pretrained("./LeeGeGe/bart-large-chinese")
print("*********tokenzier************")
def process_func(exmaples):
    contents = [ e for e in exmaples["dialogue"]]
    inputs = tokenizer(contents, max_length=1024, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(exmaples['Summ'], max_length=256, truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_ds_train = ds_train.map(process_func, batched=True,remove_columns=['dialogue','Summ'])
tokenized_ds_val = ds_val.map(process_func, batched=True,remove_columns=['dialogue','Summ'])

print("********model**********")
model = AutoModelForSeq2SeqLM.from_pretrained("./LeeGeGe/bart-large-chinese")


rouge = Rouge()

def compute_metric(evalPred):
    predictions, labels = evalPred
    decode_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decode_preds = [" ".join(p) for p in decode_preds]
    decode_labels = [" ".join(l) for l in decode_labels]
    scores = rouge.get_scores(decode_preds, decode_labels, avg=True)
    return {
        "rouge-1": scores["rouge-1"]["f"],
        "rouge-2": scores["rouge-2"]["f"],
        "rouge-l": scores["rouge-l"]["f"],
    }

args = Seq2SeqTrainingArguments(
    output_dir="./finetuned_bart",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    learning_rate=2e-5,
    logging_dir='./newnew-logs',
    logging_steps=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    weight_decay=0.01,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    gradient_accumulation_steps=2,
    predict_with_generate=True,
    #warmup_steps=300,
    warmup_ratio=0.1
)
trainer = Seq2SeqTrainer(
    args=args,
    model=model,
    train_dataset=tokenized_ds_train["train"],
    eval_dataset=tokenized_ds_val["train"],
    compute_metrics=compute_metric,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer)
)
trainer.train()
