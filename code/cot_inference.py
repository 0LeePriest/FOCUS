
import torch
import json
from unsloth import FastLanguageModel
import os
max_seq_length = 2048
#dtype = None
#load_in_4bit = False
model_name = "./unsloth-finetuned_lora_cot_deepseek/checkpoint-445/"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=True,
    )
print("********lora********")
FastLanguageModel.for_inference(model)
prompt_style = """下面是描述任务的指令，附带一个提供的上下文的输入。
编写一个完成请求的响应。
在回答之前，仔细思考问题，并创建一个循序渐进的思维链，以确保逻辑和准确的回答。
### Instruction:
你将收到一段顾客客服领域的对话，你的任务是站在顾客的角度生成一段主观的摘要
### Question:
{}
### Response:
<think>{}"""
with open('./cot-test.json', 'r', encoding='utf-8') as f:
    all_data = json.load(f)
sum = 0
for i in all_data:
    sum += 1
    inputs = tokenizer([prompt_style.format(i['input'], "")], return_tensors="pt").to("cuda")

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs)
    response = response[0].split("### Response:")[1]
    cot = response.split("</think>")[0].replace("<think>","").strip()
    summary = response.split("</think>")[1].replace("<｜end▁of▁sentence｜>","").strip()
    #print(response[0].split("### Response:")[1])
    
    json_data = {
        "DialogueID": i['DialogueID'],
        "TimeID": i["TimeID"],
        "Dialogue": i["input"],
        "predict-cot":cot,
        "predict-summary": summary,
        "reference-cot": i['cot'],
        "reference-summary":i['output']
    }
    output_file = "unsloth-newnew-cot_deepseek-checkpoint-445.json"

    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f)

    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data.append(json_data)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(sum)