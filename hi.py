from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch

model_path="ibm-granite/granite-3.3-2b-instruct"
device="cpu"
model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    ).to(device)
tokenizer = AutoTokenizer.from_pretrained(
        model_path
)

conv = [{"role": "user", "content":"Take a current environmental issue and work backward to devise an innovative prevention strategy that could have averted the problem."}]

input_ids = tokenizer.apply_chat_template(conv, return_tensors="pt", thinking=True, return_dict=True, add_generation_prompt=True).to(device)

set_seed(42)
output = model.generate(
    **input_ids,
    max_new_tokens=8192,
)

prediction = tokenizer.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
print(prediction)