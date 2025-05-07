import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import torch

if torch.backends.mps.is_available():
    torch.mps.empty_cache()

# Load Qwen model
model_id = "Qwen/Qwen2.5-7B"
# model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
# llm = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
# llm = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64, device=-1)  # use CPU
# llm = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.backends.mps.is_available() else -1)


if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model = torch.nn.DataParallel(model)

wrapped_model = model
if isinstance(model, torch.nn.DataParallel):
    wrapped_model = model.module

llm = pipeline("text-generation", model=wrapped_model, tokenizer=tokenizer, max_new_tokens=100, device=0)



def transform_prompt(prompt: str, instruction: str) -> str:
    full_prompt = f"""You are a helpful AI that modifies image generation prompts. 
\nInstruction: {instruction}
\nOriginal prompt: {prompt}
Transformed prompt:"""

    print("\n=== Prompt Sent to LLM ===")
    print(full_prompt)

    response = llm(full_prompt)[0]['generated_text']

    # print("\n=== Raw LLM Output ===")
    # print(response)

    # Extract only the transformed part after the delimiter
    if "Transformed prompt:" in response:
        transformed = response.split("Transformed prompt:")[-1].strip()
    else:
        transformed = response.strip()

    print("\n=== Extracted Transformed Prompt ===")
    print(transformed)
    print("=" * 50 + "\n")

    return transformed

def run_batch_transform(input_csv="data/prompts.csv", output_csv="data/transformed_prompts_cut.csv"):
    df = pd.read_csv(input_csv)

    instructions = {
        "synonyms": "Replace some content words in the original prompt with their SYNONYMS i.e. words with the same meaning but different spelling. "
        "Keep the same amount of words in transformed prompt. "
        "EXCLUDE all the explanations - return only the transformed original prompt. "
        "\nELAMPLE: "
        "\nOriginal prompt: A cat on a mat "
        "\nTransformed prompt: A feline resting on a rug "
        "\nNOW YOUR RESULT: ",
        
        # "neutral_add": "Add polite or neutral quality-enhancing phrases (e.g. 'high quality', 'please', etc.) to the prompt.",
        # "word_reorder": "Slightly reorder the words in the prompt while keeping the meaning the same.",
    }

    for name, instr in instructions.items():
        print(f"\n>>> Applying transformation: {name}")
        tqdm.pandas(desc=f"Applying {name}")
        df[name] = df["prompt"].progress_apply(lambda p: transform_prompt(p, instr))

    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved transformed prompts to {output_csv}")

if __name__ == "__main__":
    run_batch_transform()
