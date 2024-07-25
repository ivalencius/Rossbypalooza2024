#!/bin/bash
#SBATCH --job-name=clean-data
#SBATCH --output=output.out
#SBATCH --error=errors.err
#SBATCH --account=valencig
#SBATCH --account=rossby
#SBATCH --reservation=rossby
#SBATCH --time=10:00:00
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --mem=30000  #Per Node
#SBATCH --mail-type=ALL  # Email notification options: ALL, BEGIN, END, FAIL, ALL, NONE
#SBATCH --mail-user=valencig@rcc.uchicago.edu  # Replace jdoe with your CNET and be sure to include "@rcc"

# touch $SLURM_JOB_ID.txt
# echo "Job ID: $SLURM_JOB_ID" >> $SLURM_JOB_ID.txt
# echo "Job name: $SLURM_JOB_NAME" >> $SLURM_JOB_ID.txt
# echo "N tasks: $SLURM_ARRAY_TASK_COUNT" >> $SLURM_JOB_ID.txt
# echo "N cores: $SLURM_CPUS_ON_NODE" >> $SLURM_JOB_ID.txt
# echo "N threads per core: $SLURM_THREADS_PER_CORE" >> $SLURM_JOB_ID.txt
# echo "Minimum memory required per CPU: $SLURM_MEM_PER_CPU" >> $SLURM_JOB_ID.txt
# echo "Requested memory per GPU: $SLURM_MEM_PER_GPU" >> $SLURM_JOB_ID.txt

module load python/anaconda-2021.05
source activate /project/rossby/valencig
python process_tp.py
