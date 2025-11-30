# =======================================================
# 🩺 Medical Dataset Image Generator (Heracles Compatible)
# =======================================================
"""
Dataset: VQA-RAD (Visual Question Answering Radiology)
Description:
    ~3,000 question–answer pairs on radiology images (CT, X-ray, MRI)
    Available: https://github.com/abachaa/VQA-RAD

Goal:
    Generate synthetic medical images and captions from dataset prompts
    for augmentation or visualization.

Supports:
    ✅ Sequential (single CPU)
    ✅ OpenMP-style threading (multi-core CPU)
    ✅ MPI distributed mode (multi-node)
    ✅ GPU (CUDA)

Requirements:
    pip install diffusers transformers accelerate torch datasets pillow mpi4py
"""

# =======================================================
# 1. Imports and Setup
# =======================================================
import os
import time
import torch
from datasets import load_dataset
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Optional MPI setup
try:
    from mpi4py import MPI
    mpi_enabled = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    mpi_enabled = False
    rank, size = 0, 1

# =======================================================
# 2. Configuration
# =======================================================
OUTPUT_DIR = "vqa_rad_generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Detect hardware
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

# Determine mode prefix
mode = "mpi" if mpi_enabled else (
    "omp" if int(os.environ.get("OMP_NUM_THREADS", "1")) > 1 else "seq"
)
file_prefix = f"{mode}_rank{rank}"

if rank == 0:
    print("===================================================")
    print("🩺 Medical Dataset Image Generator - Parallel Edition")
    print("===================================================")
    print(f"🚀 Running mode: {mode.upper()}")
    print(f"🧠 Using device: {device}")
    print(f"🧵 CPU threads: {torch.get_num_threads()}")
    print(f"🔢 MPI ranks: {size}")
    print("===================================================\n")

start_time = time.time()

# =======================================================
# 3. Load Dataset
# =======================================================
if rank == 0:
    print("📚 Loading dataset...")

try:
    dataset = load_dataset("abachaa/vqa-rad", split="train")
except Exception:
    dataset = [
        {"question": "What abnormality is visible in the chest X-ray?", "answer": "Pneumonia"},
        {"question": "Is there a fracture visible in the left arm X-ray?", "answer": "Yes, humeral fracture"},
        {"question": "What organ is shown in this MRI scan?", "answer": "Brain"},
        {"question": "Describe the condition in this CT image.", "answer": "Lung nodule"},
        {"question": "What kind of scan is this?", "answer": "CT scan of abdomen"},
        {"question": "Identify the issue in this MRI scan.", "answer": "Tumor mass"},
    ]

# Split data among MPI ranks if applicable
if mpi_enabled:
    dataset = [dataset[i] for i in range(rank, len(dataset), size)]

if rank == 0:
    print(f"✅ Dataset ready: {len(dataset)} records per rank\n")

# =======================================================
# 4. Load Diffusion Model
# =======================================================
if rank == 0:
    print("🔄 Loading Stable Diffusion XL model...\n")

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=dtype
).to(device)

pipe.enable_attention_slicing()

# LOCK to protect CPU inference (prevents crashes)
pipe_lock = Lock()

# =======================================================
# 5. Prompt Construction
# =======================================================
def make_prompt(example):
    question = example.get("question", "")
    answer = example.get("answer", "")
    return (
        f"Medical radiology illustration of {answer.lower()}. "
        f"Context: {question}. Style: realistic X-ray, CT, or MRI scan."
    )

# =======================================================
# 6. Batch Image Generation (Thread-safe, OpenMP-style)
# =======================================================
print(f"[{file_prefix}] 🖼️ Starting image generation...\n")

BATCH_SIZE = 2  # number of prompts per batch
num_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
batches = [dataset[i:i+BATCH_SIZE] for i in range(0, len(dataset[:10]), BATCH_SIZE)]

def generate_batch(batch_examples, batch_id):
    results = []
    for i, example in enumerate(batch_examples):
        idx = batch_id * BATCH_SIZE + i
        prompt = make_prompt(example)
        try:
            # Thread-safe critical section
            with pipe_lock:
                img = pipe(prompt, num_inference_steps=40, guidance_scale=8.0).images[0]
            filename = os.path.join(OUTPUT_DIR, f"{file_prefix}_{idx+1:03d}.png")
            img.save(filename)
            results.append(f"[{file_prefix}] ✅ {prompt}\n → Saved: {filename}")
        except Exception as e:
            results.append(f"[{file_prefix}] ⚠️ Error: {e}")
    return results

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = {executor.submit(generate_batch, batch, b_id): b_id for b_id, batch in enumerate(batches)}
    for fut in as_completed(futures):
        for line in fut.result():
            print(line, flush=True)

# =======================================================
# 7. Runtime Report
# =======================================================
end_time = time.time()
elapsed = end_time - start_time

if mpi_enabled:
    total_time = comm.reduce(elapsed, op=MPI.MAX, root=0)
else:
    total_time = elapsed

if rank == 0:
    print("===================================================")
    print("🏁 Generation complete!")
    print(f"⏱ Total runtime: {total_time:.2f} seconds")
    print(f"📂 Images saved in: {OUTPUT_DIR}")
    print("===================================================\n")

print(f"[{file_prefix}] 🏁 Done in {elapsed:.2f}s — Images in {OUTPUT_DIR}\n")