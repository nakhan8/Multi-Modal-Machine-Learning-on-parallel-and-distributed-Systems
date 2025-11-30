#!/bin/bash
# ======================== SLURM CPU Job Script ==============================
# Optimized for running CPU-heavy Python programs (e.g., SDXL image generation)
# ============================================================================

#SBATCH --partition=day-long-cpu         # CPU partition (adjust if needed)
#SBATCH --job-name=sdxl_cpu_job          # Job name
#SBATCH --output=%x_out.%j               # Std output file
#SBATCH --error=%x_err.%j                # Std error file
#SBATCH --mail-type=END,FAIL             # Notify when job ends or fails
#SBATCH --mail-user=myemail@domain.com   # <-- Replace with your email
#SBATCH --time=0-04:00:00                # Walltime (4 hours)
#SBATCH --exclusive                      # Reserve entire node
#SBATCH --cpus-per-task=8                # Number of CPU threads
#SBATCH --ntasks=1                       # One instance of the program

# -------------------- Environment Setup --------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set KMP_AFFINITY if not already set
if [ -z "$KMP_AFFINITY" ]; then
    echo "KMP_AFFINITY not set. Defaulting to 'compact'."
    export KMP_AFFINITY=compact
fi

echo "===================================================="
echo "🔧 OpenMP Job Configuration"
echo "===================================================="
echo "🧵 OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "🧠 MKL_NUM_THREADS = $MKL_NUM_THREADS"
echo "📌 KMP_AFFINITY    = $KMP_AFFINITY"
echo "===================================================="

# -------------------- Program Arguments --------------------
program_name="medical_image_generator.py"
program_params=""

# -------------------- Run Program --------------------
echo "🚀 Starting: python -u $program_name $program_params"
echo "===================================================="

# Use unbuffered output for real-time log streaming
python -u "$program_name" $program_params

# -------------------- Post-run check --------------------
status=$?
if [ $status -ne 0 ]; then
    echo "❌ ERROR: Program failed (exit code $status)."
    exit $status
else
    echo "✅ Program completed successfully!"
fi
