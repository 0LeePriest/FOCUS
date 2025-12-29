import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from unsloth import FastLanguageModel
model_name = "./unsloth-finetuned_lora_qa_deepseek/checkpoint-445/"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)
print("********lora********")
FastLanguageModel.for_inference(model)

with open('instruct-test.json', 'r', encoding='utf-8') as f:
    all_data = json.load(f)
#model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
#model = PeftModel.from_pretrained(model, model_id=lora_path).to("cuda")
sum = 0
for i in all_data:
    sum += 1
    
    i_ids = tokenizer.encode(
        f'{i["instruction"]}<£üUser£ü>{i["input"]}<£üAssistant£ü>',
        add_special_tokens=True, return_tensors="pt").to('cuda')

    outputs = model.generate(
        input_ids=i_ids,
         #input_ids=inputs.input_ids,
         #attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs)
    summary = response[0].split("<£üAssistant£ü>")[1].replace("<£üend¨xof¨xsentence£ü>","")
    
    json_data = {
        "DialogueID": i['DialogueID'],
        "TimeID": i["TimeID"],
        "Dialogue": i["input"],
        "predict": summary,
        "reference": i['output']
    }
    output_file = "unsloth-newnew-qa_deepseek-checkpoint-445.json"

    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f)

    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data.append(json_data)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(sum)