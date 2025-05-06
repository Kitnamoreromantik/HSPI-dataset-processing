import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

# Load Qwen model
model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

def transform_prompt(prompt: str, instruction: str) -> str:
    full_prompt = f"""
                    You are a helpful AI that modifies image generation prompts.
                    Instruction: {instruction}
                    Original prompt: "{prompt}"
                    Transformed prompt:
                    """
    
    response = llm(full_prompt)[0]['generated_text']
    # Extract only the transformed prompt (strip out original prompt/instruction)
    return response.split("Transformed prompt:")[-1].strip()

def run_batch_transform(input_csv="data/prompts.csv", output_csv="data/transformed_prompts.csv"):
    df = pd.read_csv(input_csv)

    instructions = {
        "synonyms": "Replace some content words in the prompt with their synonyms while preserving meaning.",
        "neutral_add": "Add polite or neutral quality-enhancing phrases (e.g. 'high quality', 'please', etc.) to the prompt.",
        "word_reorder": "Slightly reorder the words in the prompt while keeping the meaning the same.",
    }

    for name, instr in instructions.items():
        tqdm.pandas(desc=f"Applying {name}")
        df[name] = df["prompt"].progress_apply(lambda p: transform_prompt(p, instr))

    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

if __name__ == "__main__":
    run_batch_transform()
