#!/bin/bash
export KMP_AFFINITY=compact
export OMP_NUM_THREADS=1
export BLIS_JC_NT=1
export BLIS_IC_NT=1
export BLIS_JR_NT=1

k_start=256
k_end=20480
k_blocksize=256

seed=-1

# 1-level ABC Strassen
echo "stra_1_abc=[" | tee stra_square_1core.m
echo -e "% NIm NJn NPk\t tblis_time stra_1_abc_time total_gflops\t tblis_gflops stra_1_abc_gflops\t accuracy" | tee stra_square_1core.m
for (( k=k_start; k<=k_end; k+=k_blocksize ))
do
    ./test_strassen -m $k -n $k -k $k -niter 3 -level 1 -impl 1 -seed $seed | tee stra_square_1core.m
done
echo "];" | tee stra_square_1core.m
# 1-level AB Strassen
echo "stra_1_ab=[" | tee stra_square_1core.m
echo -e "% NIm NJn NPk\t tblis_time stra_1_ab_time total_gflops\t tblis_gflops stra_1_ab_gflops\t accuracy" | tee stra_square_1core.m
for (( k=k_start; k<=k_end; k+=k_blocksize ))
do
    ./test_strassen -m $k -n $k -k $k -niter 3 -level 1 -impl 2 -seed $seed | tee stra_square_1core.m
done
echo "];" | tee stra_square_1core.m
# 1-level Naive Strassen
echo "stra_1_naive=[" | tee stra_square_1core.m
echo -e "% NIm NJn NPk\t tblis_time stra_1_naive_time total_gflops\t tblis_gflops stra_1_naive_gflops\t accuracy" | tee stra_square_1core.m
for (( k=k_start; k<=k_end; k+=k_blocksize ))
do
    ./test_strassen -m $k -n $k -k $k -niter 3 -level 1 -impl 3 -seed $seed | tee stra_square_1core.m
done
echo "];" | tee stra_square_1core.m
# 2-level ABC Strassen
echo "stra_2_abc=[" | tee stra_square_1core.m
echo -e "% NIm NJn NPk\t tblis_time stra_2_abc_time total_gflops\t tblis_gflops stra_2_abc_gflops\t accuracy" | tee stra_square_1core.m
for (( k=k_start; k<=k_end; k+=k_blocksize ))
do
    ./test_strassen -m $k -n $k -k $k -niter 3 -level 2 -impl 1 -seed $seed | tee stra_square_1core.m
done
echo "];" | tee stra_square_1core.m
# 2-level AB Strassen
echo "stra_2_ab=[" | tee stra_square_1core.m
echo -e "% NIm NJn NPk\t tblis_time stra_2_ab_time total_gflops\t tblis_gflops stra_2_ab_gflops\t accuracy" | tee stra_square_1core.m
for (( k=k_start; k<=k_end; k+=k_blocksize ))
do
    ./test_strassen -m $k -n $k -k $k -niter 3 -level 2 -impl 2 -seed $seed | tee stra_square_1core.m
done
echo "];" | tee stra_square_1core.m
# 2-level Naive Strassen
echo "stra_2_naive=[" | tee stra_square_1core.m
echo -e "% NIm NJn NPk\t tblis_time stra_2_naive_time total_gflops\t tblis_gflops stra_2_naive_gflops\t accuracy" | tee stra_square_1core.m
for (( k=k_start; k<=k_end; k+=k_blocksize ))
do
    ./test_strassen -m $k -n $k -k $k -niter 3 -level 2 -impl 3 -seed $seed | tee stra_square_1core.m
done
echo "];" | tee stra_square_1core.m

