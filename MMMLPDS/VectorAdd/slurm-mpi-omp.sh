#!/bin/bash
# ======================== Comments ==============================
# This job run MPI programs on Heracles Cluster
# setup --ntasks and --nodes for number of MPI tasks and number of nodes where the task will be distributed.
# --> Example how to submit this job:
#     sbatch --cpus-per-task=<num_opm_threads> slurm-mpi-omp.sh $PWD/myprogram <arg1, arg2, argn>
#
# check partitions:
#   sinfo
#
# You can check the partition configuration with:
#   scontrol show partition <partition_name>
#
# Use the following command to get detailed information about the node
#   scontrol show node <node_name>

#SBATCH --partition=day-long-cpu           

# --> Students may setup the name for the job and email
# --> The %x is a SLURM job name placeholder that gets replaced with the job's name when the job runs. In this case
# --> Slurm will use the same jobname to identify the output file and error file.
# --> The job name is useful for tracking and managing jobs, especially when using SLURM commands like squeue or 
# --> when looking at job logs
#SBATCH --job-name=myjobname    ### Job Name
#SBATCH --output=%x_out.%j      ### File in which to store job output. x% is a placeholder that gets replaced with the job's name.
#SBATCH --error=%x_err.%j       ### File in which to store job error messages. x% is a placeholder that gets replaced with the job's name.
#SBATCH --mail-type=ALL         ### email alert at start, end and abortion of execution
#SBATCH --mail-user=myemail     ### send mail to this address

# --> The #SBATCH --time=0-00:01:00 directive in your SLURM script specifies the maximum amount 
# --> of wall clock time your job is allowed to run. If you don't specify a time limit, SLURM will 
# --> use the default time limit for the partition
#SBATCH --time=0-01:00:00       ### Wall clock time limit in Days-HH:MM:SS

# --> Number of Tasks: Specifies that the job will run a total of N MPI tasks. 
#     These tasks will be distributed across the requested nodes
# --> If you want to know home many tasks per node will be launched by slurm, 
#     just divide --ntasks by --nodes
# Attention, I got error when I setup --taskes to 1 
#SBATCH --ntasks=2

# --> Number of Nodes: Requests N nodes for the job. MPI tasks will be distributed across these nodes.
#SBATCH --nodes=2

# --> # cpus-per-task: set the maximum logical cores to be allocated for each MPI processs in a node
# --> Each node on Heracles can run up 48 threads, for instance,
# --> if you have 2 MPI tasks per node (ntasks/nodes), then 
#     you have to setup  cpus-per-task to 24 openMP threads at most, that comes from 48/MPI-tasks-per-node 
#SBATCH --cpus-per-task=6

# --> Exclusive Node Usage: Ensures that no other jobs will share the nodes allocated to this job, 
#     which can improve performance.
#SBATCH --exclusive       ### no shared resources within a node
#
# --> Load Intel MPI environment only if you did not run it yeat on your session, otherwise it will give you error
#     source /opt/intel/oneapi/setvars.sh --force
#
# Set OpenMP environment variables

module purge
module use /mcms/modulefiles
module load SwitchEnv
# export I_MPI_PIN=1
# export I_MPI_PIN_DOMAIN=core

# Run the program with the given parameters
source /opt/intel/oneapi/setvars.sh --force
unset OMPI_* MPICH_* PMI_* PMIX_*

export I_MPI_DEBUG=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# Get the program name and parameters from the remaining command-line arguments
program_name=$1
shift 1
program_params="$@"

# Print information for debugging
echo "Message from SLURM job '$SLURM_JOB_NAME':"
echo "Program Name: $program_name $program_params"
echo "Slurm allocated $SLURM_NNODES nodes, $SLURM_NTASKS MPI tasks, and $OMP_NUM_THREADS OpenMP threads per MPI task."

mpirun -print-rank-map $program_name $program_params 
