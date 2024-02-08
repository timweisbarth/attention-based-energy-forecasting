#!/bin/bash
#SBATCH --job-name="single_v100_job_for_tutorial4"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100
#SBATCH --time 0-00:20:00 # set maximum allowed runtime to 10min
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/qb/work/YOUR_GROUP/YOUR_UID/tutorial_4/%x-%j.out  # note one cannot use env variables like $WORK in #SBATCH statements

# useful for debugging
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested any gpus

ls $WORK # not necessary just here to illustrate that $WORK is avaible here

# add possibly other setup code here, e.g.
# - copy singularity images or data to local storage like /scratch_local
# - loads virtual envs, like with anaconda
# - set environments variables
# - determine commandline arguments for `srun` calls

# your actual compute calls
srun python3 runfile.py  # srun will automatically pickup the configuration defined via `#SBATCH` and `sbatch` command line arguments 