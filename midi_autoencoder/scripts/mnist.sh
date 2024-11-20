#!/bin/bash

# Manually define the project name.
# This must also be the name of your conda environment used for this project.
PROJECT_NAME="midi-autoencoder"
# Automatically convert hyphens to underscores, to get the name of the project directory.
PROJECT_DIRN="${PROJECT_NAME//-/_}"

# Exit the script if any command hits an error
set -e

# Store the time at which the script was launched, so we can measure how long has elapsed.
start_time="$SECONDS"

# Print disk usage report, to catch errors due to lack of file space.
echo "------------------------------------"
echo "df -h:"
# Print header, then sort the rows alphabetically by mount point
df -h --output=target,pcent,size,used,avail,source | head -n 1
df -h --output=target,pcent,size,used,avail,source | tail -n +2 | sort -h
echo ""
echo "-------- Input handling ------------------------------------------------"
date
echo ""
# Use the SLURM job array to select the seed for the experiment
SEED="$SLURM_ARRAY_TASK_ID"
if [[ "$SEED" == "" ]];
then
    SEED=0
fi
echo "SEED = $SEED"

# Check if the first argument is a path to the python script to run
if [[ "$1" == *.py ]];
then
    # If it is, we'll run this python script and remove it from the list of
    # arguments to pass on to the script.
    SCRIPT_PATH="$1"
    shift
else
    # Otherwise, use our default python training script.
    SCRIPT_PATH="$PROJECT_DIRN/train.py"
fi
echo "SCRIPT_PATH = $SCRIPT_PATH"

# Any arguments provided to sbatch after the name of the slurm script will be
# passed through to the main script later.
# (The pass-through works like *args or **kwargs in python.)
echo "Pass-through args: ${@}"
echo ""
echo "-------- Activating environment ----------------------------------------"
date
echo ""
echo "Running ~/.bashrc"
source ~/.bashrc

# Activate virtual environment
ENVNAME="$PROJECT_NAME"
echo "Activating conda environment $ENVNAME"
conda activate "$ENVNAME"
echo ""
# Print env status (which packages you have installed - useful for diagnostics)
# N.B. This script only prints things out, it doesn't assign any environment variables.
echo "Running $PROJECT_NAME/scripts/report_env_config.sh"
source "$PROJECT_NAME/scripts/report_env_config.sh"

# Set the JOB_LABEL environment variable
echo "-------- Setting JOB_LABEL ---------------------------------------------"
echo ""
# Decide the name of the paths to use for saving this job
if [ "$SLURM_ARRAY_TASK_COUNT" != "" ] && [ "$SLURM_ARRAY_TASK_COUNT" -gt 1 ];
then
    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}";
else
    JOB_ID="${SLURM_JOB_ID}";
fi
# Decide the name of the paths to use for saving this job
JOB_LABEL="${SLURM_JOB_NAME}__${JOB_ID}";
echo "JOB_ID = $JOB_ID"
echo "JOB_LABEL = $JOB_LABEL"
echo ""

# Set checkpoint directory ($CKPT_DIR) environment variables
echo "-------- Setting checkpoint and output path variables ------------------"
echo ""
# Vector provides a fast parallel filesystem local to the GPU nodes, dedicated
# for checkpointing. It is mounted under /checkpoint. It is strongly
# recommended that you keep your intermediary checkpoints under this directory
CKPT_DIR="/checkpoint/${SLURM_JOB_USER}/${SLURM_JOB_ID}"
echo "CKPT_DIR = $CKPT_DIR"
CKPT_PTH="$CKPT_DIR/checkpoint_latest.pt"
echo "CKPT_PTH = $CKPT_PTH"
echo ""
# Ensure the checkpoint dir exists
mkdir -p "$CKPT_DIR"
echo "Current contents of ${CKPT_DIR}:"
ls -lh "${CKPT_DIR}"
echo ""
# Create a symlink to the job's checkpoint directory within a subfolder of the
# current directory (repository directory) named "checkpoint_working".
mkdir -p "checkpoints_working"
ln -sfn "$CKPT_DIR" "$PWD/checkpoints_working/$SLURM_JOB_NAME"
# Specify an output directory to place checkpoints for long term storage once
# the job is finished.
if [[ -d "/scratch/hdd001/home/$SLURM_JOB_USER" ]];
then
    OUTPUT_DIR="/scratch/hdd001/home/$SLURM_JOB_USER"
elif [[ -d "/scratch/ssd004/scratch/$SLURM_JOB_USER" ]];
then
    OUTPUT_DIR="/scratch/ssd004/scratch/$SLURM_JOB_USER"
else
    OUTPUT_DIR=""
fi
if [[ "$OUTPUT_DIR" != "" ]];
then
    # Directory OUTPUT_DIR will contain all completed jobs for this project.
    OUTPUT_DIR="$OUTPUT_DIR/checkpoints/$PROJECT_NAME"
    # Subdirectory JOB_OUTPUT_DIR will contain the outputs from this job.
    JOB_OUTPUT_DIR="$OUTPUT_DIR/$JOB_LABEL"
    echo "JOB_OUTPUT_DIR = $JOB_OUTPUT_DIR"
    if [[ -d "$JOB_OUTPUT_DIR" ]];
    then
        echo "Current contents of ${JOB_OUTPUT_DIR}"
        ls -lh "${JOB_OUTPUT_DIR}"
    fi
    echo ""
fi

# Save a list of installed packages and their versions to a file in the output directory
conda env export > "$CKPT_DIR/environment.yml"
pip freeze > "$CKPT_DIR/frozen-requirements.txt"

if [[ "$SLURM_RESTART_COUNT" > 0 && ! -f "$CKPT_PTH" ]];
then
    echo ""
    echo "====================================================================="
    echo "SLURM SCRIPT ERROR:"
    echo "    Resuming after pre-emption (SLURM_RESTART_COUNT=$SLURM_RESTART_COUNT)"
    echo "    but there is no checkpoint file at $CKPT_PTH"
    echo "====================================================================="
    exit 1;
fi

echo ""
echo "------------------------------------"
elapsed=$(( SECONDS - start_time ))
eval "echo Running total elapsed time for restart $SLURM_RESTART_COUNT: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
echo ""
echo "-------- Begin main script ---------------------------------------------"
date
echo ""
# Store the master node's IP address in the MASTER_ADDR environment variable,
# which torch.distributed will use to initialize DDP.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "Rank 0 node is at $MASTER_ADDR"

# Use a port number automatically selected from the job id number.
# This will let us use the same port for every task in the job without having
# to create a file to store the port number.
# We don't want to use a fixed port number because that would lead to
# collisions between jobs when they are scheduled on the same node.
# We only use ports in the range 49152-65535 (inclusive), which are the
# Dynamic Ports, also known as Private Ports.
MASTER_PORT="$(( $SLURM_JOB_ID % 16384 + 49152 ))"
if ss -tulpn | grep -q ":$MASTER_PORT ";
then
    # The port we selected is in use, so we'll get a random available port instead.
    echo "Finding a free port to use for $SLURM_NNODES node training"
    MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')";
fi
export MASTER_PORT;
echo "Will use port $MASTER_PORT for c10d communication"

export WORLD_SIZE="$(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))"
echo "WORLD_SIZE = $WORLD_SIZE"

# NCCL options ----------------------------------------------------------------

# This is needed to print debug info from NCCL, can be removed if all goes well
export NCCL_DEBUG=INFO

# This is needed to avoid NCCL to use ifiniband, which the cluster does not have
export NCCL_IB_DISABLE=1

# This is to tell NCCL to use bond interface for network communication
if [[ "${SLURM_JOB_PARTITION}" == "t4v2" ]] || [[ "${SLURM_JOB_PARTITION}" == "rtx6000" ]];
then
    echo "Using NCCL_SOCKET_IFNAME=bond0 on ${SLURM_JOB_PARTITION}"
    export NCCL_SOCKET_IFNAME=bond0
fi

# Set this when using the NCCL backend for inter-GPU communication.
export TORCH_NCCL_BLOCKING_WAIT=1
# -----------------------------------------------------------------------------

# Multi-GPU configuration
echo ""
echo "Main script begins via torchrun with host tcp://${MASTER_ADDR}:$MASTER_PORT with backend NCCL"
if [[ "$SLURM_JOB_NUM_NODES" == "1" ]];
then
    echo "Single ($SLURM_JOB_NUM_NODES) node training ($SLURM_GPUS_ON_NODE GPUs)"
else
    echo "Multiple ($SLURM_JOB_NUM_NODES) node training (x$SLURM_GPUS_ON_NODE GPUs per node)"
fi
echo ""

# We use the torchrun command to launch our main python script.
# It will automatically set up the necessary environment variables for DDP,
# and will launch the script once for each GPU on each node.
#
# We pass the CKPT_DIR environment variable on as the output path for our
# python script, and also try to resume from a checkpoint in this directory
# in case of pre-emption. The python script should run from scratch if there
# is no checkpoint at this path to resume from.
#
# We pass on to train.py an arary of arbitrary extra arguments given to this
# slurm script contained in the `$@` magic variable.
#
# We execute the srun command in the background with `&` (and then check its
# process ID and wait for it to finish before continuing) so the main process
# can handle the SIGUSR1 signal. Otherwise if a child process is running, the
# signal will be ignored.
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$SCRIPT_PATH" \
    --cpu-workers="$SLURM_CPUS_PER_GPU" \
    --seed="$SEED" \
    --checkpoint="$CKPT_PTH" \
    --log-wandb \
    --run-name="$SLURM_JOB_NAME" \
    --run-id="$JOB_ID" \
    "${@}" &
child="$!"
wait "$child"

echo ""
echo "------------------------------------"
elapsed=$(( SECONDS - start_time ))
eval "echo Running total elapsed time for restart $SLURM_RESTART_COUNT: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
echo ""
# Now the job is finished, remove the symlink to the job's checkpoint directory
# from checkpoints_working
rm "$PWD/checkpoints_working/$SLURM_JOB_NAME"
# By overriding the JOB_OUTPUT_DIR environment variable, we disable saving
# checkpoints to long-term storage. This is disabled by default to preserve
# disk space. When you are sure your job config is correct and you are sure
# you need to save your checkpoints for posterity, comment out this line.
JOB_OUTPUT_DIR=""
#
if [[ "$CKPT_DIR" == "" ]];
then
    # This shouldn't ever happen, but we have a check for just in case.
    # If $CKPT_DIR were somehow not set, we would mistakenly try to copy far
    # too much data to $JOB_OUTPUT_DIR.
    echo "CKPT_DIR is unset. Will not copy outputs to $JOB_OUTPUT_DIR."
elif [[ "$JOB_OUTPUT_DIR" == "" ]];
then
    echo "JOB_OUTPUT_DIR is unset. Will not copy outputs from $CKPT_DIR."
else
    echo "-------- Saving outputs for long term storage --------------------------"
    date
    echo ""
    echo "Copying outputs from $CKPT_DIR to $JOB_OUTPUT_DIR"
    mkdir -p "$JOB_OUTPUT_DIR"
    rsync -rutlzv "$CKPT_DIR/" "$JOB_OUTPUT_DIR/"
    echo ""
    echo "Output contents of ${JOB_OUTPUT_DIR}:"
    ls -lh "$JOB_OUTPUT_DIR"
    # Set up a symlink to the long term storage directory
    ln -sfn "$OUTPUT_DIR" "checkpoints_finished"
fi
echo ""
echo "------------------------------------------------------------------------"
echo ""
echo "Job $SLURM_JOB_NAME ($SLURM_JOB_ID) finished, submitted from $SLURM_SUBMIT_HOST ($SLURM_CLUSTER_NAME)"
date
echo "------------------------------------"
elapsed=$(( SECONDS - start_time ))
eval "echo Total elapsed time for restart $SLURM_RESTART_COUNT: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
echo "========================================================================"
