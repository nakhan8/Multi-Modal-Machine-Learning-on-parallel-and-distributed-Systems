# =======================================================
# 🩺 Medical Dataset Image Generator – MPI + OpenMP + Batch Optimized
# =======================================================
import os
import time
import torch
from datasets import load_dataset
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# -------------------------------------------------------
# 1️⃣ MPI Initialization
# -------------------------------------------------------
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    mpi_enabled = True
except ImportError:
    mpi_enabled = False
    rank, size = 0, 1

# -------------------------------------------------------
# 2️⃣ Config & Hardware Detection
# -------------------------------------------------------
OUTPUT_DIR = "vqa_rad_generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

omp_threads = int(os.environ.get("OMP_NUM_THREADS", "2"))
torch.set_num_threads(omp_threads)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

if rank == 0:
    print("===================================================")
    print("🩺 Medical Dataset Image Generator - MPI + OMP + Batch")
    print("===================================================")
    print(f"🚀 MPI ranks: {size}")
    print(f"🧠 Threads per rank: {omp_threads}")
    print(f"🧮 Device per rank: {device}")
    print("===================================================\n")

# -------------------------------------------------------
# 3️⃣ Load Dataset and Partition
# -------------------------------------------------------
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
        {"question": "Is there fluid accumulation in this ultrasound image?", "answer": "Pleural effusion"},
        {"question": "What is the visible condition in this spinal MRI?", "answer": "Disc herniation"},
        {"question": "Identify the abnormality in this liver CT scan.", "answer": "Hepatic cyst"},
        {"question": "What does this cardiac MRI show?", "answer": "Left ventricular hypertrophy"},
    ]

# Split evenly across ranks
if mpi_enabled:
    dataset = [dataset[i] for i in range(rank, len(dataset), size)]

if rank == 0:
    print(f"✅ Dataset partitioned: {len(dataset)} records per rank\n")

# -------------------------------------------------------
# 4️⃣ Load Model Once per Rank
# -------------------------------------------------------
if rank == 0:
    print("🔄 Loading Stable Diffusion XL model...\n")

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=dtype
).to(device)
pipe.enable_attention_slicing()
pipe_lock = Lock()

# -------------------------------------------------------
# 5️⃣ Helper Functions
# -------------------------------------------------------
def make_prompt(example):
    q, a = example.get("question", ""), example.get("answer", "")
    return f"Medical radiology illustration of {a.lower()}. Context: {q}. Style: realistic X-ray, CT, or MRI scan."

# -------------------------------------------------------
# 6️⃣ Batch Generator Function
# -------------------------------------------------------
def generate_batch(batch_examples, batch_id):
    prompts = [make_prompt(ex) for ex in batch_examples]
    try:
        with pipe_lock:
            images = pipe(prompts, num_inference_steps=40, guidance_scale=8.0).images
        for i, img in enumerate(images):
            fname = os.path.join(OUTPUT_DIR, f"rank{rank}_batch{batch_id:02d}_img{i+1:02d}.png")
            img.save(fname)
        return f"[Rank {rank}] ✅ Batch {batch_id} → {len(images)} images"
    except Exception as e:
        return f"[Rank {rank}] ⚠️ Batch {batch_id} failed: {e}"

# -------------------------------------------------------
# 7️⃣ Parallel Batch Execution (OMP + MPI)
# -------------------------------------------------------
start = time.time()
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "5"))

batches = [dataset[i:i+BATCH_SIZE] for i in range(0, len(dataset), BATCH_SIZE)]

print(f"[Rank {rank}] 🖼️ Starting generation with {len(batches)} batches...")

with ThreadPoolExecutor(max_workers=omp_threads) as executor:
    futures = {executor.submit(generate_batch, b, i): i for i, b in enumerate(batches)}
    for fut in as_completed(futures):
        print(fut.result(), flush=True)

elapsed = time.time() - start

# -------------------------------------------------------
# 8️⃣ MPI Synchronization and Summary
# -------------------------------------------------------
if mpi_enabled:
    total_time = comm.reduce(elapsed, op=MPI.MAX, root=0)
else:
    total_time = elapsed

if rank == 0:
    print("===================================================")
    print("🏁 Generation complete!")
    print(f"⏱ Max runtime across ranks: {total_time:.2f}s")
    print(f"📂 Images saved in: {OUTPUT_DIR}")
    print("===================================================\n")

print(f"[Rank {rank}] Done in {elapsed:.2f}s — {len(batches)} batches processed.")
