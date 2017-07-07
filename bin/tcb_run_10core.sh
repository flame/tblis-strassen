#!/bin/bash
export KMP_AFFINITY=compact
export OMP_NUM_THREADS=10
export BLIS_JC_NT=1
export BLIS_IC_NT=10
export BLIS_JR_NT=1

# 1-level ABC Strassen
echo "stra_1_abc=[" | tee stra_tcb.m
echo -e "% NIm NJn NPk\t tblis_time stra_1_abc_time total_gflops\t tblis_gflops stra_1_abc_gflops\t accuracy" | tee stra_tcb.m
./test_strassen -niter 3 -level 1 -impl 1 -file tcb.txt | tee stra_tcb.m
echo "];" | tee stra_tcb.m
# 1-level AB Strassen
echo "stra_1_ab=[" | tee stra_tcb.m
echo -e "% NIm NJn NPk\t tblis_time stra_1_ab_time total_gflops\t tblis_gflops stra_1_ab_gflops\t accuracy" | tee stra_tcb.m
./test_strassen -niter 3 -level 1 -impl 2 -file tcb.txt | tee stra_tcb.m
echo "];" | tee stra_tcb.m
# 1-level Naive Strassen
echo "stra_1_naive=[" | tee stra_tcb.m
echo -e "% NIm NJn NPk\t tblis_time stra_1_naive_time total_gflops\t tblis_gflops stra_1_naive_gflops\t accuracy" | tee stra_tcb.m
./test_strassen -niter 3 -level 1 -impl 3 -file tcb.txt | tee stra_tcb.m
echo "];" | tee stra_tcb.m
# 2-level ABC Strassen
echo "stra_2_abc=[" | tee stra_tcb.m
echo -e "% NIm NJn NPk\t tblis_time stra_2_abc_time total_gflops\t tblis_gflops stra_2_abc_gflops\t accuracy" | tee stra_tcb.m
./test_strassen -niter 3 -level 2 -impl 1 -file tcb.txt | tee stra_tcb.m
echo "];" | tee stra_tcb.m
# 2-level AB Strassen
echo "stra_2_ab=[" | tee stra_tcb.m
echo -e "% NIm NJn NPk\t tblis_time stra_2_ab_time total_gflops\t tblis_gflops stra_2_ab_gflops\t accuracy" | tee stra_tcb.m
./test_strassen -niter 3 -level 2 -impl 2 -file tcb.txt | tee stra_tcb.m
echo "];" | tee stra_tcb.m
# 2-level Naive Strassen
echo "stra_2_naive=[" | tee stra_tcb.m
echo -e "% NIm NJn NPk\t tblis_time stra_2_naive_time total_gflops\t tblis_gflops stra_2_naive_gflops\t accuracy" | tee stra_tcb.m
./test_strassen -niter 3 -level 2 -impl 3 -file tcb.txt | tee stra_tcb.m
echo "];" | tee stra_tcb.m

