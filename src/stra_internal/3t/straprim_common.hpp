#ifndef _STRAPRIM_3T_COMMON_HPP_
#define _STRAPRIM_3T_COMMON_HPP_


#include "tensor2matrix.hpp"

namespace tblis
{
namespace internal
{

template<typename T>
void tblis_printmat(matrix_view<T>& M)
{
    for (int i = 0; i < M.length(0); i++) {
        for (int j = 0; j < M.length(1); j++) {
            std::cout << (M.data())[i*M.stride(0)+j*M.stride(1)] << " ";
        }
        std::cout << std::endl;
    }
}

extern MemoryPool BuffersForA, BuffersForB, BuffersForScatter;

extern MemoryPool BuffersForS, BuffersForT, BuffersForM;


using StraTensorGEMM = stra_partition_gemm_nc<
                         stra_partition_gemm_kc<
                           stra_matrify_and_pack_b<BuffersForB,
                             stra_partition_gemm_mc<
                               stra_matrify_and_pack_a<BuffersForA,
                                 stra_matrify_c<BuffersForScatter,
                                   stra_partition_gemm_nr<
                                     stra_partition_gemm_mr<
                                       stra_gemm_micro_kernel>>>>>>>>;

using StraMatGotoGEMM = stra_partition_gemm_nc<
                          stra_partition_gemm_kc<
                            stra_pack_b<BuffersForB,
                              stra_partition_gemm_mc<
                                stra_pack_a<BuffersForA,
                                  stra_partition_gemm_nr<
                                    stra_partition_gemm_mr<
                                      stra_gemm_micro_kernel>>>>>>>;

using StraTensorGEMM_TTM = stra_partition_gemm_nc<
                             stra_partition_gemm_kc<
                               stra_matrify_and_pack_b<BuffersForB,
                                 stra_partition_gemm_mc<
                                   stra_matrify_and_pack_a<BuffersForA,
                                       stra_partition_gemm_nr<
                                         stra_partition_gemm_mr<
                                           stra_gemm_micro_kernel>>>>>>>;



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


template <typename T, unsigned NA, unsigned NB, unsigned NC>
void straprim_naive2(const communicator& comm, const config& cfg,
        T alpha, stra_tensor_view<T, NA>& A, stra_tensor_view<T, NB>& B, T beta, stra_tensor_view<T, NC>& C)
{
    len_type m = C.length(0);
    len_type n = C.length(1);
    len_type k = A.length(1);

    const len_type MR = cfg.gemm_mr.def<T>();
    const len_type NR = cfg.gemm_nr.def<T>();
    const len_type KR = cfg.gemm_kr.def<T>();
    const len_type ME = cfg.gemm_mr.extent<T>();
    const len_type NE = cfg.gemm_nr.extent<T>();
    const len_type KE = cfg.gemm_kr.extent<T>();

    len_type m_ext = ceil_div(m, MR)*ME;
    len_type n_ext = ceil_div(n, NR)*NE;
    len_type k_ext = ceil_div(k, KR)*KE;

    //StraTensorGEMM stra_gemm;
    //int nt = comm.num_threads();
    //auto tc = make_gemm_thread_config<T>(cfg, nt, ms, ns, ks);
    //step<0>(stra_gemm).distribute = tc.jc_nt;
    //step<4>(stra_gemm).distribute = tc.ic_nt;
    //step<8>(stra_gemm).distribute = tc.jr_nt;
    //step<9>(stra_gemm).distribute = tc.ir_nt;


    StraMatGotoGEMM stra_gemm;
    int nt = comm.num_threads();
    gemm_thread_config tc = make_gemm_thread_config<T>(cfg, nt, m, n, k);
    step<0>(stra_gemm).distribute = tc.jc_nt;
    step<3>(stra_gemm).distribute = tc.ic_nt;
    step<5>(stra_gemm).distribute = tc.jr_nt;
    step<6>(stra_gemm).distribute = tc.ir_nt;

    //A -> A_sum
    MemoryPool::Block buffer_s;
    T* s_ptr;
    if (comm.master())
    {
        buffer_s = BuffersForS.allocate<T>( m_ext * k );
        s_ptr = buffer_s.get<T>();
    }
    comm.broadcast(s_ptr);

    matrix_view<T> Sv({m, k}, s_ptr, {1,m_ext});  // m should be 4/8's multiple for alignment?


    add(comm, cfg, T(1.0), A, T(0.0), Sv);

    //std::cout << "Sv: ";
    //tblis_printmat( Sv );

    //B -> B_sum
    MemoryPool::Block buffer_t;
    T* t_ptr; 
    if (comm.master())
    {
        buffer_t = BuffersForT.allocate<T>( k_ext * n );
        t_ptr = buffer_t.get<T>();
    }
    comm.broadcast(t_ptr);
    matrix_view<T> Tv({k, n}, t_ptr, {1,k_ext});

    add(comm, cfg, T(1.0), B, T(0.0), Tv);

    //std::cout << "Tv: ";
    //tblis_printmat( Tv );

    MemoryPool::Block buffer_m;
    T* m_ptr;
    if (comm.master())
    {
        buffer_m = BuffersForM.allocate<T>( m_ext * n );
        m_ptr = buffer_m.get<T>();
    }
    comm.broadcast(m_ptr);

    matrix_view<T> Mv({m, n}, m_ptr, {1,m_ext});

    stra_gemm(comm, cfg, alpha, Sv, Tv, T(0), Mv);

    //std::cout << "Mv: ";
    //tblis_printmat( Mv );

    add(comm, cfg, T(1.0), Mv, T(1.0), C);
}


template <typename T, unsigned NA, unsigned NB, unsigned NC>
void straprim_ab2(const communicator& comm, const config& cfg,
        T alpha, stra_tensor_view<T, NA>& A, stra_tensor_view<T, NB>& B, T beta, stra_tensor_view<T, NC>& C)
{
    len_type m = C.length(0);
    len_type n = C.length(1);
    len_type k = A.length(1);

    const len_type MR = cfg.gemm_mr.def<T>();
    const len_type NR = cfg.gemm_nr.def<T>();
    const len_type KR = cfg.gemm_kr.def<T>();
    const len_type ME = cfg.gemm_mr.extent<T>();
    const len_type NE = cfg.gemm_nr.extent<T>();
    const len_type KE = cfg.gemm_kr.extent<T>();

    len_type m_ext = ceil_div(m, MR)*ME;
    len_type n_ext = ceil_div(n, NR)*NE;
    len_type k_ext = ceil_div(k, KR)*KE;

    StraTensorGEMM_TTM stra_gemm_ttm;
    int nt = comm.num_threads();
    auto tc = make_gemm_thread_config<T>(cfg, nt, A.length(0), B.length(1), A.length(1));
    step<0>(stra_gemm_ttm).distribute = tc.jc_nt;
    step<4>(stra_gemm_ttm).distribute = tc.ic_nt;
    step<7>(stra_gemm_ttm).distribute = tc.jr_nt;
    step<8>(stra_gemm_ttm).distribute = tc.ir_nt;

    MemoryPool::Block buffer_m;
    T* m_ptr;
    if (comm.master())
    {
        buffer_m = BuffersForM.allocate<T>( m_ext * n );
        m_ptr = buffer_m.get<T>();
    }
    comm.broadcast(m_ptr);

    matrix_view<T> Mv({m, n}, m_ptr, {1,m_ext});

    stra_gemm_ttm( comm, cfg, alpha, A, B, T(0), Mv );

    //std::cout << "Mv: ";
    //tblis_printmat( Mv );

    add(comm, cfg, T(1.0), Mv, T(1.0), C);
}

template <typename T, unsigned NA, unsigned NB, unsigned NC>
void straprim_naive(const communicator& comm, const config& cfg, 
                        const std::vector<len_type>& len_AB,
                        const std::vector<len_type>& len_AC,
                        const std::vector<len_type>& len_BC,
                        T alpha,
                        const std::array<T*,NA>& A_list,
                        const std::array<T,NA>& A_coeff,
                        const std::vector<stride_type>& stride_A_AB,
                        const std::vector<stride_type>& stride_A_AC,
                        const std::array<T*,NB>& B_list,
                        const std::array<T,NB>& B_coeff,
                        const std::vector<stride_type>& stride_B_AB,
                        const std::vector<stride_type>& stride_B_BC,
                        T  beta,
                        std::array<T*,NC>& C_list,
                        const std::array<T,NC>& C_coeff,
                        const std::vector<stride_type>& stride_C_AC,
                        const std::vector<stride_type>& stride_C_BC)
{

    tensor<T> ar, br, cr;
    T* ptrs_local[3];
    T** ptrs = &ptrs_local[0];

    if (comm.master())
    {
        ar.reset(len_AC+len_AB);
        br.reset(len_AB+len_BC);
        cr.reset(len_AC+len_BC);
        ptrs[0] = ar.data();
        ptrs[1] = br.data();
        ptrs[2] = cr.data();
    }

    comm.broadcast(ptrs);

    tensor_view<T> arv(len_AC+len_AB, ptrs[0]);
    tensor_view<T> brv(len_AB+len_BC, ptrs[1]);
    tensor_view<T> crv(len_AC+len_BC, ptrs[2]);

    matrix_view<T> am, bm, cm;
    matricize<T>(arv, am, static_cast<unsigned>(len_AC.size()));
    matricize<T>(brv, bm, static_cast<unsigned>(len_AB.size()));
    matricize<T>(crv, cm, static_cast<unsigned>(len_AC.size()));



    for (unsigned idx = 0; idx < NA; idx++) {
        if (idx==0) {
            add(comm, cfg, {}, {}, arv.lengths(),
          A_coeff[0], false,   A_list[0], {}, stride_A_AC+stride_A_AB,
                T(0), false,  arv.data(), {},           arv.strides());
        } else {
            add(comm, cfg, {}, {}, arv.lengths(),
        A_coeff[idx], false, A_list[idx], {}, stride_A_AC+stride_A_AB,
                T(1), false,  arv.data(), {},           arv.strides());
        }
    }


    for (unsigned idx = 0; idx < NB; idx++) {
        if (idx==0) {
            add(comm, cfg, {}, {}, brv.lengths(),
          B_coeff[0], false,   B_list[0], {}, stride_B_AB+stride_B_BC,
                T(0), false,  brv.data(), {},           brv.strides());
        } else {
            add(comm, cfg, {}, {}, brv.lengths(),
        B_coeff[idx], false, B_list[idx], {}, stride_B_AB+stride_B_BC,
                T(1), false,  brv.data(), {},           brv.strides());
        }
    }


    //std::cout << "am: " << std::endl;
    //stra_printmat( am );
    //std::cout << "bm: " << std::endl;
    //stra_printmat( bm );
    //std::cout << "cm: " << std::endl;
    //stra_printmat( cm );

    //std::cout << "Before stra_mult\n" << std::endl;
    //stra_mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
    mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
              alpha, false, am.data(), am.stride(0), am.stride(1),
                     false, bm.data(), bm.stride(0), bm.stride(1),
               T(0), false, cm.data(), cm.stride(0), cm.stride(1));
    //std::cout << "After stra_mult\n" << std::endl;

    //std::cout << "am: " << std::endl;
    //stra_printmat( am );
    //std::cout << "bm: " << std::endl;
    //stra_printmat( bm );
    //std::cout << "cm: " << std::endl;
    //stra_printmat( cm );


    //C_sum (matrix) -> C0, C1 (tensor)
    for (unsigned idx = 0; idx < NC; idx++) {
        add(comm, cfg, {}, {}, crv.lengths(),
    C_coeff[idx], false,  crv.data(), {},            crv.strides(),
            T(1), false, C_list[idx], {}, stride_C_AC+stride_C_BC);
            //beta, false, C_list[idx], {}, stride_C_AC+stride_C_BC);
    }

}

template <typename T, unsigned NA, unsigned NB, unsigned NC>
void straprim_ab(const communicator& comm, const config& cfg, 
                        const std::vector<len_type>& len_AC,
                        const std::vector<len_type>& len_BC,
                        T alpha,
                        stra_tensor_view<T,NA>& Av,
                        stra_tensor_view<T,NB>& Bv,
                        T  beta,
                        std::array<T*,NC>& C_list,
                        const std::array<T,NC>& C_coeff,
                        const std::vector<stride_type>& stride_C_AC,
                        const std::vector<stride_type>& stride_C_BC)
{

    tensor<T> cr;
    T* ptrs;
    if (comm.master())
    {
        cr.reset(len_AC+len_BC);
        ptrs = cr.data();
    }
    comm.broadcast(ptrs);
    tensor_view<T> crv(len_AC+len_BC, ptrs);
    matrix_view<T> cm;
    matricize<T>(crv, cm, static_cast<unsigned>(len_AC.size()));

    //tensor_matrix<T> mt(my_len_AC,
    //                    my_len_BC,
    //                    crv.data(),
    //                    ,
    //                    );


    StraTensorGEMM_TTM stra_gemm_ttm;
    int nt = comm.num_threads();
    auto tc = make_gemm_thread_config<T>(cfg, nt, Av.length(0), Bv.length(1), Av.length(1));
    step<0>(stra_gemm_ttm).distribute = tc.jc_nt;
    step<4>(stra_gemm_ttm).distribute = tc.ic_nt;
    step<7>(stra_gemm_ttm).distribute = tc.jr_nt;
    step<8>(stra_gemm_ttm).distribute = tc.ir_nt;

    stra_gemm_ttm( comm, cfg, alpha, Av, Bv, T(0), cm );

    //std::cout << "am: " << std::endl;
    //stra_printmat( am );
    //std::cout << "bm: " << std::endl;
    //stra_printmat( bm );
    //std::cout << "cm: " << std::endl;
    //stra_printmat( cm );

    //std::cout << "Before stra_mult\n" << std::endl;
    //stra_mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
    //mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
    //          alpha, false, am.data(), am.stride(0), am.stride(1),
    //                 false, bm.data(), bm.stride(0), bm.stride(1),
    //           T(0), false, cm.data(), cm.stride(0), cm.stride(1));
    //std::cout << "After stra_mult\n" << std::endl;

    //std::cout << "am: " << std::endl;
    //stra_printmat( am );
    //std::cout << "bm: " << std::endl;
    //stra_printmat( bm );
    //std::cout << "cm: " << std::endl;
    //stra_printmat( cm );


    //C_sum (matrix) -> C0, C1 (tensor)
    for (unsigned idx = 0; idx < NC; idx++) {
        add(comm, cfg, {}, {}, crv.lengths(),
    C_coeff[idx], false,  crv.data(), {},            crv.strides(),
            T(1), false, C_list[idx], {}, stride_C_AC+stride_C_BC);
            //beta, false, C_list[idx], {}, stride_C_AC+stride_C_BC);
    }

}

}
}

#endif

