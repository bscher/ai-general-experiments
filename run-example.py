import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2-large"  # You can change this to other model names like "gpt2", "gpt2-medium", "gpt2-large", etc.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device.type}")

model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Input prompt for the model
prompt = "Omlettes with cheese are easy to make. Here is how:"


# Tokenize the input prompt
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

tokenizer.pad_token = tokenizer.eos_token
attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)

# Generate text using the model
output = model.generate(input_ids, max_length=1024, attention_mask=attention_mask, num_return_sequences=1)

# Decode and print the output text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("------")
print(generated_text)
print("------")
