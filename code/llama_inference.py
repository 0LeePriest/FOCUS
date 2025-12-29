from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import json
import os
mode_path = './FlagAlpha/Llama3-Chinese-8B-Instruct'
lora_path = './finetuned_lora_llama/checkpoint-445/'
with open('./instruct-test.json', 'r', encoding='utf-8') as f:
    all_data = json.load(f)
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
print("****model*****")
model = PeftModel.from_pretrained(model, model_id=lora_path)
print("********lora**********")
sum = 0
for i in all_data:
    sum+=1
    text  = tokenizer.apply_chat_template([{"role": "system", "content": f"{i['instruction']}"},
                                            {"role": "user", "content": f"{i['input']}"}],
                                           add_generation_prompt=True,
                                           tokenize=False,
                                           #return_tensors="pt",
                                           #return_dict=True
                                           )
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    generated_ids = model.generate(
        **model_inputs,
        #model_inputs.input_ids,
        max_new_tokens=512,
        #max_length=2500,
        do_sample=True,
        top_p=0.9,
        temperature=0.5,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.encode('<|eot_id|>')[0],
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
    json_data = {
            "DialogueID":i['DialogueID'],
            "TimeID":i["TimeID"],
            "Dialogue":i["input"],
            "predict":response,
            "reference":i['output']
        }
    output_file = "newnew-llama-checkpoint-445.json"

    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f)


    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)


    data.append(json_data)


    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(sum)