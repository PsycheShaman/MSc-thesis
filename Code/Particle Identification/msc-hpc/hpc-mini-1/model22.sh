#!/bin/sh

# This example submission script contains several important directives, please examine it thoroughly

# The line below indicates which accounting group to log your job against
#SBATCH --account=stats

# The line below selects the group of nodes you require
#SBATCH --partition=ada

# The line below means you need 1 worker node and a total of 2 cores
#SBATCH --nodes=1 --ntasks=16

# The line below indicates the wall time your job will need, 10 hours for example. NB, this is a mandatory directive!
#SBATCH --time=70:00:00

# A sensible name for your job, try to keep it short
#SBATCH --job-name="CNN3"

#Modify the lines below for email alerts. Valid type values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=vljchr004@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

# The cluster is configured primarily for OpenMPI and PMI. Use srun to launch parallel jobs if your code is parallel aware.
# To protect the cluster from code that uses shared memory and grabs all available cores the cluster has the following
# environment variable set by default: OMP_NUM_THREADS=1
# If you feel compelled to use OMP then uncomment the following line:
# export OMP_NUM_THREADS=$SLURM_NTASKS

# NB, for more information read https://computing.llnl.gov/linux/slurm/sbatch.html

# Use module to gain easy access to software, typing module avail lists all packages.
# Example:
# module load python/anaconda-python-2.7


#module load software/R-3.5.2
module load python/anaconda-python-3.7
#module load software/NAMD-2.13-CUDA
#module load nvidia/Cuda10
module load software/TensorFlow-CPU-py3

#module load python/TensorAnaconda

# Your science stuff goes here...

python -u /home/vljchr004/hpc-mini/model22.py > out_model22.txt
