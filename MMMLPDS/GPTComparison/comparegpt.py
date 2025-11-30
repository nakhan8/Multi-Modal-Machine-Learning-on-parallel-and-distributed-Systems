import os
import time
import torch
from openai import OpenAI
from dotenv import load_dotenv
from diffusers import StableDiffusionXLPipeline
from sentence_transformers import SentenceTransformer, util

# =======================================================
# 1. Setup
# =======================================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODELS = ["gpt-4o-mini"]
PARAMS = {"temperature": 0.7}

IMAGE_DIR = "generated_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

TEXT_PROMPTS = [
    {"q": "A farmer has 17 sheep. All but 9 run away. How many are left?", "keywords": ["9"]},
    {"q": "List of prime numbers between 1 and 30.", "keywords": ["2", "3", "5", "7", "11", "13", "17", "19", "23", "29"]},
    {"q": "Describe a good software engineer.", "keywords": ["skills", "team", "ethics"]},
    {"q": "Describe a good nurse.", "keywords": ["care", "skills", "ethics"]},
    {"q": "What qualities make a good leader?", "keywords": ["skills", "team", "ethics"]},
    {"q": "Why should AI not be the only tool for medical decision-making?", "keywords": ["doctor", "judgment", "safety", "context", "risk"]}
]

IMAGE_PROMPTS = [
    "A software engineer coding in a modern office.",
    "A nurse helping a patient in a hospital.",
    "A leader speaking to a diverse team in a meeting.",
    "A farmer with sheep in a field.",
    "A doctor reviewing patient test results in a hospital."
]

# =======================================================
# 2. Interpolation model setup (semantic linking)
# =======================================================
embedder = SentenceTransformer("clip-ViT-B-32")

def interpolate_prompts(text_prompt, image_prompts, weight=0.6):
    """Interpolate between text and the closest image embedding semantically."""
    text_emb = embedder.encode(text_prompt, convert_to_tensor=True)
    image_embs = embedder.encode(image_prompts, convert_to_tensor=True)
    similarities = util.cos_sim(text_emb, image_embs)[0]
    best_idx = torch.argmax(similarities).item()
    best_match = image_prompts[best_idx]

    # Blend embeddings (weighted semantic interpolation)
    blended_emb = weight * text_emb + (1 - weight) * image_embs[best_idx]

    # Decode to textual prompt (approximation)
    blended_prompt = f"{text_prompt}. Visualize: {best_match}. Cinematic lighting, 4K, realistic style."
    return blended_prompt, best_match, similarities[best_idx].item()

# =======================================================
# 3. Text reasoning with explanation
# =======================================================
def score_response(response, keywords):
    text = response.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text)
    return f"{matches}/{len(keywords)} keywords"

def test_text_model(model, prompt_obj, blended_prompt):
    """Run text generation and get explanation of reasoning."""
    try:
        start = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a reasoning assistant. Give your answer and explain how you arrived at it step by step."},
                {"role": "user", "content": blended_prompt}
            ],
            **PARAMS
        )
        elapsed = round(time.time() - start, 2)
        choice = resp.choices[0].message.content
        accuracy = score_response(choice, prompt_obj["keywords"])
        return {"response": choice, "time": f"{elapsed}s", "accuracy": accuracy}
    except Exception as e:
        return {"response": f"⚠️ Error: {str(e)}", "time": "N/A", "accuracy": "N/A"}

# =======================================================
# 4. High-quality image generation (SDXL)
# =======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"🚀 Loading Stable Diffusion XL on {device}...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=dtype
).to(device)
pipe.enable_attention_slicing()

def generate_image(prompt, idx):
    """Generate a high-quality image using SDXL."""
    try:
        start = time.time()
        image = pipe(prompt, num_inference_steps=40, guidance_scale=8.0).images[0]
        elapsed = round(time.time() - start, 2)

        filename = os.path.join(IMAGE_DIR, f"interpolated_{idx+1}.png")
        image.save(filename)
        return {"prompt": prompt, "file": filename, "time": f"{elapsed}s"}
    except Exception as e:
        return {"prompt": prompt, "file": f"⚠️ Error: {str(e)}", "time": "N/A"}

# =======================================================
# 5. Main driver
# =======================================================
if __name__ == "__main__":
    output_file = "interpolated_text_image_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for model in MODELS:
            header = f"\n=== {model} (Reasoning + Interpolated Context) ===\n"
            print(header)
            f.write(header)

            for i, prompt_obj in enumerate(TEXT_PROMPTS):
                blended_prompt, matched_image, similarity = interpolate_prompts(prompt_obj["q"], IMAGE_PROMPTS)
                result = test_text_model(model, prompt_obj, blended_prompt)

                block = (
                    f"\n🧠 Text Prompt: {prompt_obj['q']}\n"
                    f"🎨 Matched Image Context: {matched_image}\n"
                    f"🔢 Similarity Score: {round(similarity, 3)}\n"
                    f"⏱ Response Time: {result['time']}\n"
                    f"✅ Accuracy: {result['accuracy']}\n"
                    f"💬 Response and Reasoning:\n{result['response']}\n"
                )
                print(block)
                f.write(block)

        # Image generation phase
        image_header = "\n=== High-Quality Image Generation (SDXL) ===\n"
        print(image_header)
        f.write(image_header)

        for idx, prompt in enumerate(IMAGE_PROMPTS):
            img_result = generate_image(prompt, idx)
            block = (
                f"\n🖼️ Image Prompt: {prompt}\n"
                f"⏱ Generation Time: {img_result['time']}\n"
                f"📁 Saved File: {img_result['file']}\n"
            )
            print(block)
            f.write(block)

    print(f"\n✅ Completed! Results saved to {output_file}")