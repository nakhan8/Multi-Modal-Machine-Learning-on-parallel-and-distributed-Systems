#!/bin/bash
# ===============================================================
#  SLURM MPI + GPU + OpenMP Hybrid Job Script (Heracles Cluster)
# ===============================================================

#SBATCH --partition=day-long-gpu
#SBATCH --job-name=mpi_gpu_medgen
#SBATCH --output=%x_out.%j
#SBATCH --error=%x_err.%j
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=myemail@domain.com       ### Replace with your email
#SBATCH --time=0-04:00:00
#SBATCH --nodes=2                            ### 2 GPU nodes
#SBATCH --ntasks=2                           ### 1 MPI rank per node
#SBATCH --gres=gpu:1                         ### 1 GPU per MPI rank
#SBATCH --cpus-per-task=6                    ### 6 OpenMP threads per rank
#SBATCH --exclusive

# ===============================================================
# 1️⃣ Environment Setup
# ===============================================================
echo "===================================================="
echo "🚀 MPI + GPU + OpenMP Job Setup"
echo "===================================================="
echo "Job ID        : $SLURM_JOB_ID"
echo "Nodes         : $SLURM_JOB_NODELIST"
echo "Total Nodes   : $SLURM_NNODES"
echo "MPI Tasks     : $SLURM_NTASKS"
echo "CPUs per Task : $SLURM_CPUS_PER_TASK"
echo "GPUs per Node : 1"
echo "===================================================="

module purge
module use /mcms/modulefiles
module load SwitchEnv
module load python/3.10
source /opt/intel/oneapi/setvars.sh --force

unset OMPI_* MPICH_* PMI_* PMIX_*
export I_MPI_DEBUG=0

# ===============================================================
# 2️⃣ OpenMP Configuration
# ===============================================================
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export KMP_AFFINITY=compact

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"
echo "OMP_PLACES=$OMP_PLACES"
echo "===================================================="

# ===============================================================
# 3️⃣ Run the Hybrid Program (MPI + GPU)
# ===============================================================
echo "🏁 Starting MPI + GPU Run..."
echo "===================================================="

mpirun -print-rank-map python3 -u /home/khanabee/medical/captions.py

status=$?

# ===============================================================
# 4️⃣ Post-run Status
# ===============================================================
if [ $status -ne 0 ]; then
    echo "❌ ERROR: MPI+GPU job failed (exit code $status)"
    exit $status
else
    echo "✅ MPI+GPU job completed successfully!"
fi

echo "===================================================="
echo "🏁 All ranks finished. Check vqa_rad_generated_images/"
echo "===================================================="