#!/bin/bash
#
#SBATCH --job-name=hLDS_omniglot # the job name
#SBATCH -p gpu               # cpu, gpu, debug
#SBATCH -N 1                 # number of nodes
#SBATCH -t 1-0:00            # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out   # STDOUT
#SBATCH -e slurm.%N.%j.err   # STDERR
##SBATCH -w gpu-380-16
#SBATCH --mem=50gb
#

#SBATCH --gres=gpu:1

date

module load miniconda

conda activate hMPC

module load cuda/11.8

#ARGS = (10, 50, 1000, 2000)
#python3 my_script.py --use-gpu --seed ${ARGS[$SLURM_ARRAY_TASK_ID]}

python3 -u main.py --folder_name 'run_1'

# End of script
