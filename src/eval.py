from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from tqdm import tqdm
import os
import torch
import re

from openai import OpenAI

import asyncio

class Evaluate_Normal():
    def __init__(self, model_path, dataset, dtype="bfloat16"):
        self.model_path=model_path
        self.dataset=dataset.eval_dataset

        self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, device_map = "auto",trust_remote_code=True, dtype=dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, padding_side="left"
        )

        adapter_path = os.path.join(self.model_path,"/adp")
        if os.path.exists(adapter_path):
            self.model = PeftModel.from_pretrained(
                self.model_path, adapter_path
        )
        self.model.eval()
        

    def generate_result(self, number=-1, batch_size=64):
        if number == -1:
            data = self.dataset
        elif number >= len(self.dataset):
            print(f"The requested test data number--{number} exceeds the total length of the loaded dataset--{len(self.dataset)}. Using the entire dataset for testing.")
            data = self.dataset
        else:
            data = self.dataset[:number]
        
        # 'data' is a large list, each element is a dictionary with "input" and "output" keys.
        # Generate the 'generated_output' for each 'input' and add it to the corresponding dictionary.
        all_outputs = []

        for i in tqdm(range(0, len(data), batch_size), desc="Processing batches", total=len(data)//batch_size + (1 if len(data) % batch_size != 0 else 0)):
            batch_data = data[i:i + batch_size]
            batch_inputs = [item["input"] for item in batch_data]
            
            # Encoding inputs
            inputs = self.tokenizer(
                batch_inputs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=1024
            )
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generating outputs
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            batch_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            new_outputs = []
            
            for out_text, in_text in zip(batch_outputs, batch_inputs):
                # Check if the generated output starts with the input and remove it
                if out_text.startswith(in_text):
                    new_out = out_text[len(in_text):].strip()  # Remove the input part from the start
                else:
                    new_out = out_text.strip()  # If input is not part of the start, keep the whole output
                
                new_outputs.append(new_out)   
        for out_text, item in zip(new_outputs, data):
            item["generated_output"]=out_text
        self.result = data

    async def eval_right(self):
        correct = 0
        total = 0.0000001

        tasks = []

        for item in self.result:
            label = item.get("output", "").strip()
            generated = item.get("generated_output", "").strip()
            tasks.append(judge_the_correctness(label, generated))
            
        results = await asyncio.gather(*tasks)

        for result in results:
            if result == 1:
                correct += 1
            total += 1
        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy} ({correct}/{total})")
        return accuracy

async def judge_the_correctness(label, generated):
    system_prompt = f"""You are an answer evaluator tasked with determining whether the provided answer from the assistant is right or wrong.
The question and the generated answer: <<{generated}>>
The right answer is: <<{label}>>
-- Please only return "right" or "wrong", without any additional explanations.
e.g. <judge>right</judge>
"""
    query = "Please judge the right or wrong and only return the answer."
    
    client = OpenAI(api_key="sk-05542169ff8644c08926a729949abff3", base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ],
    stream=False
)
    response = response.choices[0].message.content
    print(response)
    response = response.strip().lower()
    if "wrong" in response:
        return 0
    elif "right" in response:
        return 1
    else:
        return 0
