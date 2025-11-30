# =======================================================
# 🩺 Medical Dataset Image Generator – MPI + OpenMP + GPU Optimized
# =======================================================
import os
import time
import torch
from datasets import load_dataset
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# -------------------------------------------------------
# 1️⃣ MPI Initialization
# -------------------------------------------------------
try:
    from mpi4py import MPI
    mpi_enabled = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    mpi_enabled = False
    rank, size = 0, 1

# -------------------------------------------------------
# 2️⃣ Hardware Detection
# -------------------------------------------------------
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
num_threads = torch.get_num_threads()
has_cuda = torch.cuda.is_available()

if has_cuda:
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    gpu_name = torch.cuda.get_device_name(local_rank)
    device = f"cuda:{local_rank}"
else:
    gpu_name = "None"
    device = "cpu"

dtype = torch.float16 if "cuda" in device else torch.float32

# -------------------------------------------------------
# 3️⃣ Initialization Logging
# -------------------------------------------------------
if rank == 0:
    print("===================================================")
    print("🩺 Medical Dataset Image Generator - Optimized")
    print("===================================================")
    print(f"🚀 MPI ranks (nodes): {size}")
    print(f"🧠 Threads per rank : {num_threads}")
    print("===================================================\n")

print(f"[Rank {rank}] Device: {device} ({gpu_name}) | Threads: {num_threads}")

# -------------------------------------------------------
# 4️⃣ Dataset Handling
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
    ]

# Distribute 1 prompt per node (MPI rank)
if mpi_enabled:
    dataset = [dataset[i] for i in range(rank, len(dataset), size)]

if rank == 0:
    print(f"✅ Dataset partitioned: {len(dataset)} records per node\n")

# -------------------------------------------------------
# 5️⃣ Model Loading (auto-fallback)
# -------------------------------------------------------
if rank == 0:
    print("🔄 Loading diffusion model...\n")

try:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype
    ).to(device)
except Exception as e:
    print(f"[Rank {rank}] ⚠️ SDXL failed: {e}\n↪ Falling back to SD v1.5.")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype
    ).to(device)

pipe.enable_attention_slicing()
pipe_lock = Lock()

# -------------------------------------------------------
# 6️⃣ Prompt Construction
# -------------------------------------------------------
def make_prompt(example):
    q, a = example.get("question", ""), example.get("answer", "")
    return f"Medical radiology illustration of {a.lower()}. Context: {q}. Style: realistic X-ray, CT, or MRI scan."

# -------------------------------------------------------
# 7️⃣ Generation Function
# -------------------------------------------------------
def generate_prompt(example, idx):
    prompt = make_prompt(example)
    filename = os.path.join(OUTPUT_DIR, f"rank{rank}_img{idx+1:03d}.png")
    try:
        with pipe_lock:
            image = pipe(prompt, num_inference_steps=40, guidance_scale=8.0).images[0]
        image.save(filename)
        return f"[Rank {rank}] ✅ {prompt}\n → Saved: {filename}"
    except Exception as e:
        return f"[Rank {rank}] ⚠️ Error: {e}"

# -------------------------------------------------------
# 8️⃣ OpenMP-Style Parallel Generation
# -------------------------------------------------------
OUTPUT_DIR = "vqa_rad_generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

start = time.time()
print(f"[Rank {rank}] 🖼️ Starting image generation...")

# GPU nodes handle 3 prompts each, CPU nodes 1
# Each thread handles one prompt; run as many as available locally
max_prompts = min(len(dataset), num_threads)
work_subset = dataset[:max_prompts]


with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(generate_prompt, ex, i) for i, ex in enumerate(work_subset)]
    for fut in as_completed(futures):
        print(fut.result(), flush=True)

elapsed = time.time() - start

# -------------------------------------------------------
# 9️⃣ MPI Synchronization and Summary
# -------------------------------------------------------
if mpi_enabled:
    total_time = comm.reduce(elapsed, op=MPI.MAX, root=0)
else:
    total_time = elapsed

if rank == 0:
    print("===================================================")
    print("🏁 Generation complete!")
    print(f"⏱ Total runtime: {total_time:.2f}s (max rank time)")
    print(f"📂 Images saved in: {OUTPUT_DIR}")
    print("===================================================\n")

print(f"[Rank {rank}] 🏁 Done in {elapsed:.2f}s — Images in {OUTPUT_DIR}")
