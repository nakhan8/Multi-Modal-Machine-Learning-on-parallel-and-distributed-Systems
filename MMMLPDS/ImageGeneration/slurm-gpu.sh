#!/bin/bash

# ======================== Comments ==============================
# This job run CUDA programs on Heracles Cluster using Nvidia GPUs
# ----------------------------------------------------------------
# Submit this job as
# sbatch --job-name=gpuJob slurm-gpu.sh $PWD/my_program arg1 arg2 arg3
# 
# check partitions:
#   sinfo
#
# You can check the partition configuration with:
#   scontrol show partition <partition_name>
#
# Use the following command to get detailed information about the node
#   scontrol show node <node_name>
#
# Verify GPU Availability:
#   scontrol show node node18 (or node1)
#
# to check gpu configuration run:
#   srun --nodelist=node1 --gres=gpu:1 /usr/local/cuda/samples/Samples/1_Utilities/deviceQuery/deviceQuery
#
# Select a partition that contain GPUs
#SBATCH --partition=day-long-gpu           ### Partition

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

# ---> In case you need a GPU from a specif node, i.e., node1 or node18 you may uncomment --nodelist
#      SBATCH --nodelist=node18                 # Specify node1 or node18 explicitly

# --> Exclusive Node Usage: Ensures that no other jobs will share the nodes allocated to this job, 
#     which can improve performance.
#SBATCH --exclusive       ### no shared resources within a node

# Capture the program name and parameters from the command line arguments
program_name=$1
shift 1  # Shift the positional parameters to the left (so $2 becomes $1, etc.)
program_params="$@"

# Print SLURM-provided environment variables
echo "Job ID: $SLURM_JOB_ID"
# Get the GPU device ID assigned by SLURM
GPU_DEVICE=$CUDA_VISIBLE_DEVICES
echo "SLURM GPU IDs available = $GPU_DEVICE on $SLURM_NODELIST"

# Print information for debugging
echo "Message from job: Running program: $program_name with parameters: $program_params"

# Run the program with the given parameters
$program_name $program_params

# In case you need to profile your code use one of the following options:
# nsys nvprof --print-gpu-trace $program_name $program_params
#   it colect information about the grid lauched by the kernel. The information is in the outpuy file.

# nsys profile --trace=cuda,nvtx,osrt $program_name $program_params
#   The nsys profile generate a report named <reportN.nsys-rep>, where N is a sequential number, 
#   that can be checkd usin the command: nsys stats reportN.nsys-rep

echo "Running GPU version"
python3 /home/khanabee/medical/medical_image_generator.py