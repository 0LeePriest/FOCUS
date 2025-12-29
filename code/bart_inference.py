import os
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_path='./finetuned_bart/checkpoint-450/'
tokenizer = AutoTokenizer.from_pretrained("./LeeGeGe/bart-large-chinese")
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda")

with open('test.json', 'r', encoding='utf-8') as f:
    all_data = json.load(f)

sum = 0
for i in all_data:
    sum += 1
    inputs = tokenizer(i['input'], return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
    outputs = model.generate(inputs["input_ids"], max_length=256)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    json_data = {
        "DialogueID": i['DialogueID'],
        "TimeID": i["TimeID"],
        "Dialogue": i["input"],
        "predict": summary,
        "reference": i['output']
    }
    
    output_file = "finetuned_bart-450.json"

    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f)

    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data.append(json_data)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(sum)
