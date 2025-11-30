#!/bin/bash
# ===============================================================
#  SLURM MPI + OpenMP Hybrid Job Script
#  Runs the medical_image_generator.py on multiple Heracles nodes
# ===============================================================

#SBATCH --partition=day-long-cpu
#SBATCH --job-name=mpi_omp_medgen
#SBATCH --output=%x_out.%j
#SBATCH --error=%x_err.%j
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=myemail@domain.com     ### Replace with your email
#SBATCH --time=0-04:00:00                  ### 4 hours max runtime
#SBATCH --nodes=2                          ### Number of compute nodes
#SBATCH --ntasks=2                         ### MPI processes (1 per node)
#SBATCH --cpus-per-task=6                  ### OpenMP threads per rank
#SBATCH --exclusive                        ### Exclusive access to nodes

# ===============================================================
# 1️⃣ Environment Setup
# ===============================================================
echo "===================================================="
echo "🧠 MPI + OpenMP Job Setup on Heracles Cluster"
echo "===================================================="
echo "Job Name     : $SLURM_JOB_NAME"
echo "Job ID       : $SLURM_JOB_ID"
echo "Nodes        : $SLURM_JOB_NODELIST"
echo "Total Nodes  : $SLURM_NNODES"
echo "Total Tasks  : $SLURM_NTASKS"
echo "Threads/Task : $SLURM_CPUS_PER_TASK"
echo "Partition    : $SLURM_JOB_PARTITION"
echo "===================================================="

# Load Intel oneAPI environment
module purge
module use /mcms/modulefiles
module load SwitchEnv
source /opt/intel/oneapi/setvars.sh --force

# Avoid OpenMPI / MPICH conflicts
unset OMPI_* MPICH_* PMI_* PMIX_*

# ===============================================================
# 2️⃣ OpenMP Configuration
# ===============================================================
export I_MPI_DEBUG=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export KMP_AFFINITY=compact

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"
echo "OMP_PLACES=$OMP_PLACES"
echo "===================================================="

# ===============================================================
# 3️⃣ Python Environment (MPI + OMP Hybrid)
# ===============================================================
module load python/3.10

echo "Python version:"
python3 --version
echo "===================================================="

# ===============================================================
# 4️⃣ Run the MPI + OpenMP Job
# ===============================================================
echo "🚀 Starting MPI + OpenMP Hybrid Run..."
echo "===================================================="

mpirun -print-rank-map python3 -u medical_image_generator.py

status=$?

# ===============================================================
# 5️⃣ Post-run Status
# ===============================================================
if [ $status -ne 0 ]; then
    echo "❌ ERROR: MPI+OMP job failed (exit code $status)"
    exit $status
else
    echo "✅ MPI+OMP job completed successfully!"
fi

echo "===================================================="
echo "🏁 All ranks finished. Check results in vqa_rad_generated_images/"
echo "===================================================="
