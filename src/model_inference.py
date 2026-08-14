import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


from model_loader import ModelLoader
import torch

loader = ModelLoader()

tokenizer ,model = loader.load_model()

messages = [
    {
        "role": "system",
        "content": "You are a warm, empathetic breast health assistant. You explain breast cancer topics in plain, supportive language. You never diagnose or give personalized medical advice, and you always recommend the person consult their doctor for decisions specific to their situation."
        },
    {
        "role": "user",
        "content": "Can you explain the difference between a benign breast cyst and a malignant tumor?"
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
    return_dict=True
)

with torch.no_grad(): # Speeds up inference by disabling gradient tracking
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        use_cache=True,
        temperature=0.5,
    )


generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
]

# Decode and print the output
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("--- ASSISTANT RESPONSE ---")
print(response)