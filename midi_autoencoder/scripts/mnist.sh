#!/bin/bash
# run parameters
JOB_NAME="synthetic-midi"
# Manually define the project name.
# This must also be the name of your conda environment used for this project.
PROJECT_NAME="midi-autoencoder"
# Automatically convert hyphens to underscores, to get the name of the project directory.
PROJECT_DIRN="${PROJECT_NAME//-/_}"

# Exit the script if any command hits an error
set -e

# Store the time at which the script was launched, so we can measure how long has elapsed.
start_time="$SECONDS"

echo "-------- Input handling ------------------------------------------------"
date
echo ""
SEED=0
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
CONDA_BASE=$(conda info --base)
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    echo "Sourcing $CONDA_SH"
    source "$CONDA_SH"
else
    echo "Error: $CONDA_SH not found. Make sure Conda is installed correctly."
    exit 1
echo "Activating conda environment $ENVNAME"
fi
conda activate "$ENVNAME"
echo ""
# Print env status (which packages you have installed - useful for diagnostics)
# N.B. This script only prints things out, it doesn't assign any environment variables.
echo "Running $PROJECT_DIRN/scripts/report_env_config.sh"
source "$PROJECT_DIRN/scripts/report_env_config.sh"

# Set the JOB_LABEL environment variable
echo "-------- Setting JOB_LABEL ---------------------------------------------"
echo ""
# Decide the name of the paths to use for saving this job
JOB_ID=$(date +"%Y%m%d%H%M%S")
JOB_LABEL="${PROJECT_NAME}__${JOB_ID}";
echo "JOB_ID = $JOB_ID"
echo "JOB_LABEL = $JOB_LABEL"
echo ""

# Set checkpoint directory ($CKPT_DIR) environment variables
echo "-------- Setting checkpoint and output path variables ------------------"
echo ""
# Vector provides a fast parallel filesystem local to the GPU nodes, dedicated
# for checkpointing. It is mounted under /checkpoint. It is strongly
# recommended that you keep your intermediary checkpoints under this directory
CKPT_DIR="checkpoints/$JOB_ID"
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
ln -sfn "$CKPT_DIR" "$PWD/checkpoints_working/$JOB_NAME"
# Specify an output directory to place checkpoints for long term storage once
# the job is finished.
# Directory OUTPUT_DIR will contain all completed jobs for this project.
OUTPUT_DIR="checkpoints/$PROJECT_NAME"
# Subdirectory JOB_OUTPUT_DIR will contain the outputs from this job.
JOB_OUTPUT_DIR="$OUTPUT_DIR/$JOB_LABEL"
echo "JOB_OUTPUT_DIR = $JOB_OUTPUT_DIR"
if [[ -d "$JOB_OUTPUT_DIR" ]];
then
    echo "Current contents of ${JOB_OUTPUT_DIR}"
    ls -lh "${JOB_OUTPUT_DIR}"
fi
echo ""

# Save a list of installed packages and their versions to a file in the output directory
conda env export > "$CKPT_DIR/environment.yml"
pip freeze > "$CKPT_DIR/frozen-requirements.txt"
echo ""
echo "------------------------------------------------------------------------"
elapsed=$(( SECONDS - start_time ))
echo ""
echo "-------- Begin main script ---------------------------------------------"
date
echo ""

# Multi-GPU configuration
NUM_NODES=1
NUM_GPUS=1
NUM_WORKERS=4
echo ""
if [[ "$NUM_NODES" == "1" ]];
then
    echo "Single ($NUM_NODES) node training ($NUM_GPUS GPUs)"
else
    echo "Multiple ($NUM_NODES) node training (x$NUM_GPUS GPUs per node)"
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
torchrun \
    "$SCRIPT_PATH" \
    --gpu="$NUM_GPUS" \
    --cpu-workers="$NUM_WORKERS" \
    --seed="$SEED" \
    --checkpoint="$CKPT_PTH" \
    --log-wandb \
    --run-name="$JOB_NAME" \
    --run-id="$JOB_ID" \
    "${@}" &
child="$!"
wait "$child"

echo ""
echo "----------------------------------------------------------------------------"
elapsed=$(( SECONDS - start_time ))
eval "echo Running total elapsed time for restart $SLURM_RESTART_COUNT: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
echo ""
# Now the job is finished, remove the symlink to the job's checkpoint directory
# from checkpoints_working
rm "$PWD/checkpoints_working/$JOB_NAME"
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
echo "Job $JOB_NAME ($JOB_ID) finished"
date
echo "------------------------------------"
elapsed=$(( SECONDS - start_time ))
eval "echo Total elapsed time for restart $SLURM_RESTART_COUNT: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
echo "========================================================================"
