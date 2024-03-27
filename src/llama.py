import os
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
HF_TOKEN = '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'
model_name = 'meta-llama/Llama-2-70b-hf'
model_directory = "llama/model"

def check_and_load(model_directory, model_name, token):
    """
    Checks if the model and tokenizer are saved locally. If not, it downloads them and saves them.
    """
    # Check if the model directory exists and has model files
    if os.path.exists(model_directory) and os.listdir(model_directory):
        print("Loading model and tokenizer from the saved directory.")
        model = AutoModelForCausalLM.from_pretrained(model_directory)
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
    else:
        print("Downloading and saving model and tokenizer.")
        os.makedirs(model_directory, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, use_auth_token=token, cache_dir=model_directory
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_auth_token=token, cache_dir=model_directory
        )
        model.save_pretrained(model_directory)
        tokenizer.save_pretrained(model_directory)
    
    return model, tokenizer

# Load or download model and tokenizer
model, tokenizer = check_and_load(model_directory, model_name, HF_TOKEN)

# Define a prompt
prompt = "The future of artificial intelligence is"

# Encode the prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Generate text
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)
