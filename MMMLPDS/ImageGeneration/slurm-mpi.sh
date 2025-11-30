#!/bin/bash
# ===========================================================
# 🧠 Heracles SLURM Job: MPI + OpenMP + GPU Hybrid Execution
# ===========================================================

#SBATCH --partition=day-long-gpu          ### GPU partition
#SBATCH --job-name=mpi_gpu_medgen         ### Job name
#SBATCH --output=%x_out.%j                ### Output log
#SBATCH --error=%x_err.%j                 ### Error log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=myemail@domain.com
#SBATCH --time=0-04:00:00                 ### Wall time limit
#SBATCH --nodes=2                         ### GPU nodes (node1 & node18)
#SBATCH --ntasks=2                        ### 1 MPI task per node
#SBATCH --gres=gpu:1                      ### 1 GPU per node
#SBATCH --nodelist=node1,node18
#SBATCH --exclusive                       ### Use nodes exclusively

# ===========================================================
# 1️⃣ Environment Setup
# ===========================================================
echo "===================================================="
echo "🚀 MPI + GPU Job Setup on Heracles Cluster"
echo "===================================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Nodes        : $SLURM_JOB_NODELIST"
echo "Partition    : $SLURM_JOB_PARTITION"
echo "===================================================="

module purge
module use /mcms/modulefiles
module load SwitchEnv
module load python/3.10

source /opt/intel/oneapi/setvars.sh --force
unset OMPI_* MPICH_* PMI_* PMIX_*

# ===========================================================
# 2️⃣ OpenMP Configuration
# ===========================================================
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export KMP_AFFINITY=compact,1,0,granularity=fine
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "===================================================="

# ===========================================================
# 3️⃣ Run MPI Python Program
# ===========================================================
echo "🏁 Starting MPI + GPU run..."
mpirun -print-rank-map python3 -u medical_image_MPI.py

status=$?

if [ $status -ne 0 ]; then
    echo "❌ ERROR: MPI job failed (exit code $status)"
    exit $status
else
    echo "✅ MPI + GPU job completed successfully!"
fi

echo "===================================================="
echo "🏁 All ranks finished — check output folder:"
echo "    vqa_rad_generated_images/"
echo "===================================================="