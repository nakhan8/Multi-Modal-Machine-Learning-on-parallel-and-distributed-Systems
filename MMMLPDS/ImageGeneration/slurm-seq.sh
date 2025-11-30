#!/bin/bash
# ======================== SLURM Sequential Job ==============================
# Runs a sequential (single-core) version of your program on the Heracles Cluster
# ============================================================================
#
# Usage:
#   sbatch slurm-seq.sh
#
# Check partitions:
#   sinfo
#
# Inspect configuration:
#   scontrol show partition day-long-cpu
#   scontrol show node <node_name>
#
# ============================================================================

#SBATCH --partition=day-long-cpu          # CPU partition
#SBATCH --job-name=sdxl_seq_job           # Job name
#SBATCH --output=%x_out.%j                # Standard output file
#SBATCH --error=%x_err.%j                 # Standard error file
#SBATCH --mail-type=END,FAIL              # Notify on job completion/failure
#SBATCH --mail-user=myemail@domain.com    # <-- Replace with your email
#SBATCH --time=0-01:00:00                 # Max wall time (1 hour)
#SBATCH --mem=10G                         # Memory allocation
#SBATCH --exclusive                       # Prevent sharing node resources

# -------------------- Environment Setup --------------------
module load python/3.10                   # Load Python module
echo "===================================================="
echo "🧠 Running Sequential Version of Medical Image Generator"
echo "===================================================="

# Ensure single-threaded execution
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BATCH_SIZE=1                       # One image per batch (for pure sequential mode)

# -------------------- Job Info --------------------
echo "Job Name     : $SLURM_JOB_NAME"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node List    : $SLURM_JOB_NODELIST"
echo "Partition    : $SLURM_JOB_PARTITION"
echo "Memory       : $SLURM_MEM_PER_NODE MB"
echo "CPU Threads  : $OMP_NUM_THREADS"
echo "===================================================="

# -------------------- Run Program --------------------
echo "🚀 Starting Sequential Run..."
time python3 -u medical_image_generator.pysq

# -------------------- Post-Run --------------------
status=$?
if [ $status -ne 0 ]; then
    echo "❌ ERROR: Program failed (exit code $status)"
    exit $status
else
    echo "✅ Sequential job completed successfully!"
fi
