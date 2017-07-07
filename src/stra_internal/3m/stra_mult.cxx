#include "stra_mult.hpp"

#include "util/gemm_thread.hpp"

#include "matrix/stra_matrix_view.hpp"

#include "stra_nodes/stra_partm.hpp"
#include "stra_nodes/stra_packm.hpp"
#include "stra_nodes/stra_gemm_ukr.hpp"

//#include "nodes/partm.hpp"
//#include "nodes/packm.hpp"
//#include "nodes/gemm_ukr.hpp"


#include "internal/1m/add.hpp"


namespace tblis
{
namespace internal
{

extern MemoryPool BuffersForA, BuffersForB;
//MemoryPool BuffersForA(4096);
//MemoryPool BuffersForB(4096);



extern MemoryPool BuffersForS, BuffersForT, BuffersForM;
MemoryPool BuffersForS(4096);
MemoryPool BuffersForT(4096);
MemoryPool BuffersForM(4096);

//using GotoGEMM = partition_gemm_nc<
//                   partition_gemm_kc<
//                     pack_b<BuffersForB,
//                       partition_gemm_mc<
//                         pack_a<BuffersForA,
//                           partition_gemm_nr<
//                             partition_gemm_mr<
//                               gemm_micro_kernel>>>>>>>;



//template <unsigned NA, unsigned NB, unsigned NC>
using StraGotoGEMM = stra_partition_gemm_nc<
                       stra_partition_gemm_kc<
                         stra_pack_b<BuffersForB,
                           stra_partition_gemm_mc<
                             stra_pack_a<BuffersForA,
                               stra_partition_gemm_nr<
                                 stra_partition_gemm_mr<
                                   stra_gemm_micro_kernel>>>>>>>;

template <typename T>
void stra_acquire_mpart(
          len_type m, len_type n, stride_type rs, stride_type cs,
          int x, int y, int i, int j,
          T* srcM, T** dstM
          )
{
    *dstM = &srcM[ ( m / x * i ) * rs + ( n / y * j ) * cs ]; //src( m/x*i, n/y*j )
}

template<typename T>
void stra_printmat(matrix_view<T>& M)
{
    for (int i = 0; i < M.length(0); i++) {
        for (int j = 0; j < M.length(1); j++) {
            std::cout << (M.data())[i*M.stride(0)+j*M.stride(1)] << " ";
        }
        std::cout << std::endl;
    }
}



template <typename T>
void stra_mult(const communicator& comm, const config& cfg,
          len_type m, len_type n, len_type k,
          T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
                   bool conj_B, const T* B, stride_type rs_B, stride_type cs_B,
          T  beta, bool conj_C,       T* C, stride_type rs_C, stride_type cs_C)
{
    //std::cout << "Enter stra_internal/3m/stra_mult\n" << std::endl;
    TBLIS_ASSERT(!conj_A && !conj_B && !conj_C);

    const bool row_major = cfg.gemm_row_major.value<T>();

    if ((row_major ? rs_C : cs_C) == 1)
    {
        /*
         * Compute C^T = B^T * A^T instead
         */
        std::swap(m, n);
        std::swap(A, B);
        std::swap(rs_A, cs_B);
        std::swap(rs_B, cs_A);
        std::swap(rs_C, cs_C);
    }


    StraGotoGEMM stra_gemm;

    int nt = comm.num_threads();
    gemm_thread_config tc = make_gemm_thread_config<T>(cfg, nt, m, n, k);
    step<0>(stra_gemm).distribute = tc.jc_nt;
    step<3>(stra_gemm).distribute = tc.ic_nt;
    step<5>(stra_gemm).distribute = tc.jr_nt;
    step<6>(stra_gemm).distribute = tc.ir_nt;

    ////DEBUGGING....
    //matrix_view<T> Av({m, k}, const_cast<T*>(A), {rs_A, cs_A});
    //matrix_view<T> Bv({k, n}, const_cast<T*>(B), {rs_B, cs_B});
    //matrix_view<T> Cv({m, n},                C , {rs_C, cs_C});


    len_type ms, ks, ns;
    len_type md, kd, nd;
    len_type mr, kr, nr;

    mr = m % ( 2 ), kr = k % ( 2 ), nr = n % ( 2 );
    md = m - mr, kd = k - kr, nd = n - nr;

    ms=md, ks=kd, ns=nd;
    const T *A_0, *A_1, *A_2, *A_3;
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 0, 0, A, &A_0 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 0, 1, A, &A_1 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 1, 0, A, &A_2 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 1, 1, A, &A_3 );

    ms=md, ks=kd, ns=nd;
    const T *B_0, *B_1, *B_2, *B_3;
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 0, 0, B, &B_0 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 0, 1, B, &B_1 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 1, 0, B, &B_2 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 1, 1, B, &B_3 );

    ms=md, ks=kd, ns=nd;
    T *C_0, *C_1, *C_2, *C_3;
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 0, 0, C, &C_0 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 0, 1, C, &C_1 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 1, 0, C, &C_2 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 1, 1, C, &C_3 );

    ms=ms/2, ks=ks/2, ns=ns/2;

    // M0 = (1 * A_0 + 1 * A_3) * (1 * B_0 + 1 * B_3);  C_0 += 1 * M0;  C_3 += 1 * M0;
    stra_matrix_view<T,2> Av0({ms, ks}, {const_cast<T*>(A_0), const_cast<T*>(A_3)}, {1, 1}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv0({ks, ns}, {const_cast<T*>(B_0), const_cast<T*>(B_3)}, {1, 1}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv0({ms, ns}, {const_cast<T*>(C_0), const_cast<T*>(C_3)}, {1, 1}, {rs_C, cs_C});
    stra_gemm(comm, cfg, alpha, Av0, Bv0, beta, Cv0);
    comm.barrier();


    //std::cout << "A:" << std::endl;
    //stra_printmat( Av );
    //std::cout << "B:" << std::endl;
    //stra_printmat( Bv );
    //std::cout << "C:" << std::endl;
    //stra_printmat( Cv );

    // M1 = (1 * A_2 + 1 * A_3) * (1 * B_0);  C_2 += 1 * M1;  C_3 += -1 * M1;
    stra_matrix_view<T,2> Av1({ms, ks}, {const_cast<T*>(A_2), const_cast<T*>(A_3)}, {1, 1}, {rs_A, cs_A});
    stra_matrix_view<T,1> Bv1({ks, ns}, {const_cast<T*>(B_0)}, {1}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv1({ms, ns}, {const_cast<T*>(C_2), const_cast<T*>(C_3)}, {1, -1}, {rs_C, cs_C});
    stra_gemm(comm, cfg, alpha, Av1, Bv1, beta, Cv1);
    comm.barrier();


    // M2 = (1 * A_0) * (1 * B_1 + -1 * B_3);  C_1 += 1 * M2;  C_3 += 1 * M2;
    stra_matrix_view<T,1> Av2({ms, ks}, {const_cast<T*>(A_0)}, {1}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv2({ks, ns}, {const_cast<T*>(B_1), const_cast<T*>(B_3)}, {1, -1}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv2({ms, ns}, {const_cast<T*>(C_1), const_cast<T*>(C_3)}, {1, 1}, {rs_C, cs_C});
    stra_gemm(comm, cfg, alpha, Av2, Bv2, beta, Cv2);
    comm.barrier();

    // M3 = (1 * A_3) * (-1 * B_0 + 1 * B_2);  C_0 += 1 * M3;  C_2 += 1 * M3;
    stra_matrix_view<T,1> Av3({ms, ks}, {const_cast<T*>(A_3)}, {1}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv3({ks, ns}, {const_cast<T*>(B_0), const_cast<T*>(B_2)}, {-1, 1}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv3({ms, ns}, {const_cast<T*>(C_0), const_cast<T*>(C_2)}, {1, 1}, {rs_C, cs_C});
    stra_gemm(comm, cfg, alpha, Av3, Bv3, beta, Cv3);
    comm.barrier();

    // M4 = (1 * A_0 + 1 * A_1) * (1 * B_3);  C_0 += -1 * M4;  C_1 += 1 * M4;
    stra_matrix_view<T,2> Av4({ms, ks}, {const_cast<T*>(A_0), const_cast<T*>(A_1)}, {1, 1}, {rs_A, cs_A});
    stra_matrix_view<T,1> Bv4({ks, ns}, {const_cast<T*>(B_3)}, {1}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv4({ms, ns}, {const_cast<T*>(C_0), const_cast<T*>(C_1)}, {-1, 1}, {rs_C, cs_C});
    stra_gemm(comm, cfg, alpha, Av4, Bv4, beta, Cv4);
    comm.barrier();

    // M5 = (-1 * A_0 + 1 * A_2) * (1 * B_0 + 1 * B_1);  C_3 += 1 * M5;
    stra_matrix_view<T,2> Av5({ms, ks}, {const_cast<T*>(A_0), const_cast<T*>(A_2)}, {-1, 1}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv5({ks, ns}, {const_cast<T*>(B_0), const_cast<T*>(B_1)}, {1, 1}, {rs_B, cs_B});
    stra_matrix_view<T,1> Cv5({ms, ns}, {const_cast<T*>(C_3)}, {1}, {rs_C, cs_C});
    stra_gemm(comm, cfg, alpha, Av5, Bv5, beta, Cv5);
    comm.barrier();

    // M6 = (1 * A_1 + -1 * A_3) * (1 * B_2 + 1 * B_3);  C_0 += 1 * M6;
    stra_matrix_view<T,2> Av6({ms, ks}, {const_cast<T*>(A_1), const_cast<T*>(A_3)}, {1, -1}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv6({ks, ns}, {const_cast<T*>(B_2), const_cast<T*>(B_3)}, {1, 1}, {rs_B, cs_B});
    stra_matrix_view<T,1> Cv6({ms, ns}, {const_cast<T*>(C_0)}, {1}, {rs_C, cs_C});
    stra_gemm(comm, cfg, alpha, Av6, Bv6, beta, Cv6);
    comm.barrier();


}

#define FOREACH_TYPE(T) \
template void stra_mult(const communicator& comm, const config& cfg, \
                        len_type m, len_type n, len_type k, \
                        T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A, \
                                 bool conj_B, const T* B, stride_type rs_B, stride_type cs_B, \
                        T  beta, bool conj_C,       T* C, stride_type rs_C, stride_type cs_C);
#include "configs/foreach_type.h"


#include "straprim_common.hpp"

template <typename T>
void stra_mult_naive(const communicator& comm, const config& cfg,
          len_type m, len_type n, len_type k,
          T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
                   bool conj_B, const T* B, stride_type rs_B, stride_type cs_B,
          T  beta, bool conj_C,       T* C, stride_type rs_C, stride_type cs_C)
{
    //std::cout << "Enter stra_internal/3m/stra_mult_naive\n" << std::endl;
    TBLIS_ASSERT(!conj_A && !conj_B && !conj_C);

    const bool row_major = cfg.gemm_row_major.value<T>();

    if ((row_major ? rs_C : cs_C) == 1)
    {
        /*
         * Compute C^T = B^T * A^T instead
         */
        std::swap(m, n);
        std::swap(A, B);
        std::swap(rs_A, cs_B);
        std::swap(rs_B, cs_A);
        std::swap(rs_C, cs_C);
    }



    ////DEBUGGING....
    //matrix_view<T> Av({m, k}, const_cast<T*>(A), {rs_A, cs_A});
    //matrix_view<T> Bv({k, n}, const_cast<T*>(B), {rs_B, cs_B});
    //matrix_view<T> Cv({m, n},                C , {rs_C, cs_C});


    len_type ms, ks, ns;
    len_type md, kd, nd;
    len_type mr, kr, nr;

    mr = m % ( 2 ), kr = k % ( 2 ), nr = n % ( 2 );
    md = m - mr, kd = k - kr, nd = n - nr;

    ms=md, ks=kd, ns=nd;
    const T *A_0, *A_1, *A_2, *A_3;
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 0, 0, A, &A_0 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 0, 1, A, &A_1 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 1, 0, A, &A_2 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 1, 1, A, &A_3 );

    ms=md, ks=kd, ns=nd;
    const T *B_0, *B_1, *B_2, *B_3;
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 0, 0, B, &B_0 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 0, 1, B, &B_1 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 1, 0, B, &B_2 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 1, 1, B, &B_3 );

    ms=md, ks=kd, ns=nd;
    T *C_0, *C_1, *C_2, *C_3;
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 0, 0, C, &C_0 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 0, 1, C, &C_1 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 1, 0, C, &C_2 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 1, 1, C, &C_3 );

    ms=ms/2, ks=ks/2, ns=ns/2;

    // M0 = (1 * A_0 + 1 * A_3) * (1 * B_0 + 1 * B_3);  C_0 += 1 * M0;  C_3 += 1 * M0;
    stra_matrix_view<T,2> Av0({ms, ks}, {const_cast<T*>(A_0), const_cast<T*>(A_3)}, {1, 1}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv0({ks, ns}, {const_cast<T*>(B_0), const_cast<T*>(B_3)}, {1, 1}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv0({ms, ns}, {const_cast<T*>(C_0), const_cast<T*>(C_3)}, {1, 1}, {rs_C, cs_C});
    straprim_naive(comm, cfg, alpha, Av0, Bv0, beta, Cv0);
    comm.barrier();


    //std::cout << "A:" << std::endl;
    //stra_printmat( Av );
    //std::cout << "B:" << std::endl;
    //stra_printmat( Bv );
    //std::cout << "C:" << std::endl;
    //stra_printmat( Cv );

    // M1 = (1 * A_2 + 1 * A_3) * (1 * B_0);  C_2 += 1 * M1;  C_3 += -1 * M1;
    stra_matrix_view<T,2> Av1({ms, ks}, {const_cast<T*>(A_2), const_cast<T*>(A_3)}, {1, 1}, {rs_A, cs_A});
    stra_matrix_view<T,1> Bv1({ks, ns}, {const_cast<T*>(B_0)}, {1}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv1({ms, ns}, {const_cast<T*>(C_2), const_cast<T*>(C_3)}, {1, -1}, {rs_C, cs_C});
    straprim_naive(comm, cfg, alpha, Av1, Bv1, beta, Cv1);
    comm.barrier();

    //std::cout << "A:" << std::endl;
    //stra_printmat( Av );
    //std::cout << "B:" << std::endl;
    //stra_printmat( Bv );
    //std::cout << "C:" << std::endl;
    //stra_printmat( Cv );


    // M2 = (1 * A_0) * (1 * B_1 + -1 * B_3);  C_1 += 1 * M2;  C_3 += 1 * M2;
    stra_matrix_view<T,1> Av2({ms, ks}, {const_cast<T*>(A_0)}, {1}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv2({ks, ns}, {const_cast<T*>(B_1), const_cast<T*>(B_3)}, {1, -1}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv2({ms, ns}, {const_cast<T*>(C_1), const_cast<T*>(C_3)}, {1, 1}, {rs_C, cs_C});
    straprim_naive(comm, cfg, alpha, Av2, Bv2, beta, Cv2);
    comm.barrier();

    // M3 = (1 * A_3) * (-1 * B_0 + 1 * B_2);  C_0 += 1 * M3;  C_2 += 1 * M3;
    stra_matrix_view<T,1> Av3({ms, ks}, {const_cast<T*>(A_3)}, {1}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv3({ks, ns}, {const_cast<T*>(B_0), const_cast<T*>(B_2)}, {-1, 1}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv3({ms, ns}, {const_cast<T*>(C_0), const_cast<T*>(C_2)}, {1, 1}, {rs_C, cs_C});
    straprim_naive(comm, cfg, alpha, Av3, Bv3, beta, Cv3);
    comm.barrier();

    // M4 = (1 * A_0 + 1 * A_1) * (1 * B_3);  C_0 += -1 * M4;  C_1 += 1 * M4;
    stra_matrix_view<T,2> Av4({ms, ks}, {const_cast<T*>(A_0), const_cast<T*>(A_1)}, {1, 1}, {rs_A, cs_A});
    stra_matrix_view<T,1> Bv4({ks, ns}, {const_cast<T*>(B_3)}, {1}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv4({ms, ns}, {const_cast<T*>(C_0), const_cast<T*>(C_1)}, {-1, 1}, {rs_C, cs_C});
    straprim_naive(comm, cfg, alpha, Av4, Bv4, beta, Cv4);
    comm.barrier();

    // M5 = (-1 * A_0 + 1 * A_2) * (1 * B_0 + 1 * B_1);  C_3 += 1 * M5;
    stra_matrix_view<T,2> Av5({ms, ks}, {const_cast<T*>(A_0), const_cast<T*>(A_2)}, {-1, 1}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv5({ks, ns}, {const_cast<T*>(B_0), const_cast<T*>(B_1)}, {1, 1}, {rs_B, cs_B});
    stra_matrix_view<T,1> Cv5({ms, ns}, {const_cast<T*>(C_3)}, {1}, {rs_C, cs_C});
    straprim_naive(comm, cfg, alpha, Av5, Bv5, beta, Cv5);
    comm.barrier();

    // M6 = (1 * A_1 + -1 * A_3) * (1 * B_2 + 1 * B_3);  C_0 += 1 * M6;
    stra_matrix_view<T,2> Av6({ms, ks}, {const_cast<T*>(A_1), const_cast<T*>(A_3)}, {1, -1}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv6({ks, ns}, {const_cast<T*>(B_2), const_cast<T*>(B_3)}, {1, 1}, {rs_B, cs_B});
    stra_matrix_view<T,1> Cv6({ms, ns}, {const_cast<T*>(C_0)}, {1}, {rs_C, cs_C});
    straprim_naive(comm, cfg, alpha, Av6, Bv6, beta, Cv6);
    comm.barrier();


}

#define FOREACH_TYPE(T) \
template void stra_mult_naive(const communicator& comm, const config& cfg, \
                        len_type m, len_type n, len_type k, \
                        T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A, \
                                 bool conj_B, const T* B, stride_type rs_B, stride_type cs_B, \
                        T  beta, bool conj_C,       T* C, stride_type rs_C, stride_type cs_C);
#include "configs/foreach_type.h"



template <typename T>
void stra_mult_ab(const communicator& comm, const config& cfg,
          len_type m, len_type n, len_type k,
          T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
                   bool conj_B, const T* B, stride_type rs_B, stride_type cs_B,
          T  beta, bool conj_C,       T* C, stride_type rs_C, stride_type cs_C)
{
    //std::cout << "Enter stra_internal/3m/stra_mult_ab\n" << std::endl;
    TBLIS_ASSERT(!conj_A && !conj_B && !conj_C);

    const bool row_major = cfg.gemm_row_major.value<T>();

    if ((row_major ? rs_C : cs_C) == 1)
    {
        /*
         * Compute C^T = B^T * A^T instead
         */
        std::swap(m, n);
        std::swap(A, B);
        std::swap(rs_A, cs_B);
        std::swap(rs_B, cs_A);
        std::swap(rs_C, cs_C);
    }


    StraGotoGEMM stra_gemm;

    int nt = comm.num_threads();
    gemm_thread_config tc = make_gemm_thread_config<T>(cfg, nt, m, n, k);
    step<0>(stra_gemm).distribute = tc.jc_nt;
    step<3>(stra_gemm).distribute = tc.ic_nt;
    step<5>(stra_gemm).distribute = tc.jr_nt;
    step<6>(stra_gemm).distribute = tc.ir_nt;

    ////DEBUGGING....
    //matrix_view<T> Av({m, k}, const_cast<T*>(A), {rs_A, cs_A});
    //matrix_view<T> Bv({k, n}, const_cast<T*>(B), {rs_B, cs_B});
    //matrix_view<T> Cv({m, n},                C , {rs_C, cs_C});


    len_type ms, ks, ns;
    len_type md, kd, nd;
    len_type mr, kr, nr;

    mr = m % ( 2 ), kr = k % ( 2 ), nr = n % ( 2 );
    md = m - mr, kd = k - kr, nd = n - nr;

    ms=md, ks=kd, ns=nd;
    const T *A_0, *A_1, *A_2, *A_3;
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 0, 0, A, &A_0 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 0, 1, A, &A_1 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 1, 0, A, &A_2 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 1, 1, A, &A_3 );

    ms=md, ks=kd, ns=nd;
    const T *B_0, *B_1, *B_2, *B_3;
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 0, 0, B, &B_0 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 0, 1, B, &B_1 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 1, 0, B, &B_2 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 1, 1, B, &B_3 );

    ms=md, ks=kd, ns=nd;
    T *C_0, *C_1, *C_2, *C_3;
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 0, 0, C, &C_0 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 0, 1, C, &C_1 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 1, 0, C, &C_2 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 1, 1, C, &C_3 );

    ms=ms/2, ks=ks/2, ns=ns/2;

    // M0 = (1 * A_0 + 1 * A_3) * (1 * B_0 + 1 * B_3);  C_0 += 1 * M0;  C_3 += 1 * M0;
    stra_matrix_view<T,2> Av0({ms, ks}, {const_cast<T*>(A_0), const_cast<T*>(A_3)}, {1, 1}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv0({ks, ns}, {const_cast<T*>(B_0), const_cast<T*>(B_3)}, {1, 1}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv0({ms, ns}, {const_cast<T*>(C_0), const_cast<T*>(C_3)}, {1, 1}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av0, Bv0, beta, Cv0);
    comm.barrier();


    //std::cout << "A:" << std::endl;
    //stra_printmat( Av );
    //std::cout << "B:" << std::endl;
    //stra_printmat( Bv );
    //std::cout << "C:" << std::endl;
    //stra_printmat( Cv );

    // M1 = (1 * A_2 + 1 * A_3) * (1 * B_0);  C_2 += 1 * M1;  C_3 += -1 * M1;
    stra_matrix_view<T,2> Av1({ms, ks}, {const_cast<T*>(A_2), const_cast<T*>(A_3)}, {1, 1}, {rs_A, cs_A});
    stra_matrix_view<T,1> Bv1({ks, ns}, {const_cast<T*>(B_0)}, {1}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv1({ms, ns}, {const_cast<T*>(C_2), const_cast<T*>(C_3)}, {1, -1}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av1, Bv1, beta, Cv1);
    comm.barrier();


    // M2 = (1 * A_0) * (1 * B_1 + -1 * B_3);  C_1 += 1 * M2;  C_3 += 1 * M2;
    stra_matrix_view<T,1> Av2({ms, ks}, {const_cast<T*>(A_0)}, {1}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv2({ks, ns}, {const_cast<T*>(B_1), const_cast<T*>(B_3)}, {1, -1}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv2({ms, ns}, {const_cast<T*>(C_1), const_cast<T*>(C_3)}, {1, 1}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av2, Bv2, beta, Cv2);
    comm.barrier();

    // M3 = (1 * A_3) * (-1 * B_0 + 1 * B_2);  C_0 += 1 * M3;  C_2 += 1 * M3;
    stra_matrix_view<T,1> Av3({ms, ks}, {const_cast<T*>(A_3)}, {1}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv3({ks, ns}, {const_cast<T*>(B_0), const_cast<T*>(B_2)}, {-1, 1}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv3({ms, ns}, {const_cast<T*>(C_0), const_cast<T*>(C_2)}, {1, 1}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av3, Bv3, beta, Cv3);
    comm.barrier();

    // M4 = (1 * A_0 + 1 * A_1) * (1 * B_3);  C_0 += -1 * M4;  C_1 += 1 * M4;
    stra_matrix_view<T,2> Av4({ms, ks}, {const_cast<T*>(A_0), const_cast<T*>(A_1)}, {1, 1}, {rs_A, cs_A});
    stra_matrix_view<T,1> Bv4({ks, ns}, {const_cast<T*>(B_3)}, {1}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv4({ms, ns}, {const_cast<T*>(C_0), const_cast<T*>(C_1)}, {-1, 1}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av4, Bv4, beta, Cv4);
    comm.barrier();

    // M5 = (-1 * A_0 + 1 * A_2) * (1 * B_0 + 1 * B_1);  C_3 += 1 * M5;
    stra_matrix_view<T,2> Av5({ms, ks}, {const_cast<T*>(A_0), const_cast<T*>(A_2)}, {-1, 1}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv5({ks, ns}, {const_cast<T*>(B_0), const_cast<T*>(B_1)}, {1, 1}, {rs_B, cs_B});
    stra_matrix_view<T,1> Cv5({ms, ns}, {const_cast<T*>(C_3)}, {1}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av5, Bv5, beta, Cv5);
    comm.barrier();

    // M6 = (1 * A_1 + -1 * A_3) * (1 * B_2 + 1 * B_3);  C_0 += 1 * M6;
    stra_matrix_view<T,2> Av6({ms, ks}, {const_cast<T*>(A_1), const_cast<T*>(A_3)}, {1, -1}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv6({ks, ns}, {const_cast<T*>(B_2), const_cast<T*>(B_3)}, {1, 1}, {rs_B, cs_B});
    stra_matrix_view<T,1> Cv6({ms, ns}, {const_cast<T*>(C_0)}, {1}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av6, Bv6, beta, Cv6);
    comm.barrier();


}

#define FOREACH_TYPE(T) \
template void stra_mult_ab(const communicator& comm, const config& cfg, \
                        len_type m, len_type n, len_type k, \
                        T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A, \
                                 bool conj_B, const T* B, stride_type rs_B, stride_type cs_B, \
                        T  beta, bool conj_C,       T* C, stride_type rs_C, stride_type cs_C);
#include "configs/foreach_type.h"



}
}
