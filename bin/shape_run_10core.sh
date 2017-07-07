#!/bin/bash
export KMP_AFFINITY=compact
export OMP_NUM_THREADS=10
export BLIS_JC_NT=1
export BLIS_IC_NT=10
export BLIS_JR_NT=1

# 1-level ABC Strassen
echo "stra_1_abc=[" | tee stra_shape.m
echo -e "% NIm NJn NPk\t tblis_time stra_1_abc_time total_gflops\t tblis_gflops stra_1_abc_gflops\t accuracy" | tee stra_shape.m
./test_strassen -niter 3 -level 1 -impl 1 -file shape.txt | tee stra_shape.m
echo "];" | tee stra_shape.m
# 1-level AB Strassen
echo "stra_1_ab=[" | tee stra_shape.m
echo -e "% NIm NJn NPk\t tblis_time stra_1_ab_time total_gflops\t tblis_gflops stra_1_ab_gflops\t accuracy" | tee stra_shape.m
./test_strassen -niter 3 -level 1 -impl 2 -file shape.txt | tee stra_shape.m
echo "];" | tee stra_shape.m
# 1-level Naive Strassen
echo "stra_1_naive=[" | tee stra_shape.m
echo -e "% NIm NJn NPk\t tblis_time stra_1_naive_time total_gflops\t tblis_gflops stra_1_naive_gflops\t accuracy" | tee stra_shape.m
./test_strassen -niter 3 -level 1 -impl 3 -file shape.txt | tee stra_shape.m
echo "];" | tee stra_shape.m
# 2-level ABC Strassen
echo "stra_2_abc=[" | tee stra_shape.m
echo -e "% NIm NJn NPk\t tblis_time stra_2_abc_time total_gflops\t tblis_gflops stra_2_abc_gflops\t accuracy" | tee stra_shape.m
./test_strassen -niter 3 -level 2 -impl 1 -file shape.txt | tee stra_shape.m
echo "];" | tee stra_shape.m
# 2-level AB Strassen
echo "stra_2_ab=[" | tee stra_shape.m
echo -e "% NIm NJn NPk\t tblis_time stra_2_ab_time total_gflops\t tblis_gflops stra_2_ab_gflops\t accuracy" | tee stra_shape.m
./test_strassen -niter 3 -level 2 -impl 2 -file shape.txt | tee stra_shape.m
echo "];" | tee stra_shape.m
# 2-level Naive Strassen
echo "stra_2_naive=[" | tee stra_shape.m
echo -e "% NIm NJn NPk\t tblis_time stra_2_naive_time total_gflops\t tblis_gflops stra_2_naive_gflops\t accuracy" | tee stra_shape.m
./test_strassen -niter 3 -level 2 -impl 3 -file shape.txt | tee stra_shape.m
echo "];" | tee stra_shape.m

