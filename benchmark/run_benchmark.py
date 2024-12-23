import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load JSON configuration
def load_config(json_file):
    with open(json_file, "r") as file:
        return json.load(file)

# Load model and tokenizer
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# Run model with prompts
def run_model(model, tokenizer, prompt_path, temperature, top_p):
    # Load the prompt
    with open(prompt_path, "r") as file:
        prompt = file.read()

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate text
    output = model.generate(
        **inputs,
        max_length=512,  # Adjust as needed
        temperature=temperature,
        top_p=top_p,
        do_sample=True
    )

    # Decode and return the result
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Main function to process the JSON configuration
def process_models(json_file):
    config = load_config(json_file)
    models = config.get("models", {})

    for model_key, model_details in models.items():
        model_name = model_details.get("model_name")
        prompts = model_details.get("config", {}).get("prompts", {})
        prompt_path = prompts.get("prompt")
        temperature = prompts.get("temperature:", 1.0)  # Default to 1.0 if not specified
        top_p = prompts.get("top-p", 1.0)  # Default to 1.0 if not specified

        print(f"Processing model: {model_name} with prompt: {prompt_path}")

        try:
            # Load model and tokenizer
            model, tokenizer = load_model_and_tokenizer(model_name)

            # Generate output
            result = run_model(model, tokenizer, prompt_path, temperature, top_p)

            # Print the result
            print(f"Result for {model_name}:\n{result}\n")
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")

# Entry point
if __name__ == "__main__":
    json_file_path = "config.json"  # Replace with your JSON file path
    process_models(json_file_path)
