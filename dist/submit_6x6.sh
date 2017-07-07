#!/bin/bash
#SBATCH -J  6x6          # job name
#SBATCH -o square6x6.o%j       # output and error file name (%j expands to jobID)
#SBATCH -N 18 -n 36
#SBATCH -p vis       # queue (partition) -- normal, development, etc.
#SBATCH -t 04:00:00         # run time (hh:mm:ss) - 1.5 hours
#SBATCH -A CompEdu

export OMP_NUM_THREADS=10
export BLIS_JC_NT=1
export BLIS_IC_NT=10
export KMP_AFFINITY=compact

N=960
I=96
NB=10
ibrun tacc_affinity ./tensorblis.exe $N $N $N $I $I $I $NB 3
ibrun tacc_affinity ./tensor_1_abc.exe $N $N $N $I $I $I $NB 3
ibrun tacc_affinity ./tensor_1_ab.exe $N $N $N $I $I $I $NB 3
ibrun tacc_affinity ./tensor_1_naive.exe $N $N $N $I $I $I $NB 3
ibrun tacc_affinity ./tensor_2_abc.exe $N $N $N $I $I $I $NB 3
ibrun tacc_affinity ./tensor_2_ab.exe $N $N $N $I $I $I $NB 3
ibrun tacc_affinity ./tensor_2_naive.exe $N $N $N $I $I $I $NB 3

