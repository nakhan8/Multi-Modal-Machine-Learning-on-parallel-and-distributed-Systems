#!/bin/bash
# ======================== Comments ==============================
# This job run MPI programs on Heracles Cluster
# --> Example how to submit this job:
#     sbatch --job-name=<myjobname> --nodes=<num_nodes> --ntasks=<num_tasks> slurm-mpi.sh  $PWD/<compiled-code> <parameters>
#
# check partitions:
#   sinfo
#
# You can check the partition configuration with:
#   scontrol show partition <partition_name>
#
# Use the following command to get detailed information about the node
#   scontrol show node <node_name>

# ----------------------------------------------------------------
# --> check the available partitions : sinfo 
# ================================================================
# --> SLURM will likely allocate one of the nodes (node1 to node16) entirely to your job.
#SBATCH --partition=day-long-cpu           ### Partition

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

# --> Number of Nodes: Requests N nodes for the job. MPI tasks will be distributed across these nodes.
#SBATCH --nodes=6
# ---> SBATCH --nodelist=node18                 # Specify nodes explicitly

#----> SBATCH --exclude=node3,node4,node7,node6       # exclude nodes with memory communication problem

# --> Number of Tasks: Specifies that the job will run a total of N MPI tasks. 
#     These tasks will be distributed across the requested nodes
# --> If you want to know home many tasks per node will be launched by slurm, 
#     just divide --ntasks by --nodes 
#SBATCH --ntasks=24 

# --> Exclusive Node Usage: Ensures that no other jobs will share the nodes allocated to this job, 
#     which can improve performance.
#SBATCH --exclusive       ### no shared resources within a node

# --> Load Intel MPI environment only if you did not run it yeat on your session, otherwise it will give you error
# --> source /opt/intel/oneapi/setvars.sh --force

# Capture the program name and parameters from the command line arguments
program_name=$1
shift 1  # Shift the positional parameters to the left (so $2 becomes $1, etc.)
program_params="$@"

# Print information for debugging
echo "Message from job: Running program: $program_name with parameters: $program_params"
echo "Message from job '$SLURM_JOB_NAME' (Job ID: $SLURM_JOB_ID):"
echo "slurm allocated  $SLURM_NNODES nodes with $SLURM_NTASKS MPI tasks."

export I_MPI_DEBUG=0
module purge
module use /mcms/modulefiles
module load SwitchEnv
# export I_MPI_PIN=1
# export I_MPI_PIN_DOMAIN=core

# Run the program with the given parameters
source /opt/intel/oneapi/setvars.sh --force
unset OMPI_* MPICH_* PMI_* PMIX_*
mpirun -print-rank-map $program_name $program_params
