#ifndef _STRAPRIM_3M_COMMON_HPP_
#define _STRAPRIM_3M_COMMON_HPP_

template <typename T, unsigned NA, unsigned NB, unsigned NC>
void straprim_naive(const communicator& comm, const config& cfg,
        T alpha, stra_matrix_view<T, NA>& A, stra_matrix_view<T, NB>& B, T beta, stra_matrix_view<T, NC>& C)
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

    //std::cout << "MR: " << MR << ";NR: " << NR << "; KR: " << KR << std::endl;
    //std::cout << "ME: " << MR << ";NE: " << NR << "; KE: " << KR << std::endl;

    len_type m_ext = ceil_div(m, MR)*ME;
    len_type n_ext = ceil_div(n, NR)*NE;
    len_type k_ext = ceil_div(k, KR)*KE;

    //std::cout << "m: " << m << ";n: " << n << "; k: " << k << std::endl;
    ////m_ext = m; n_ext = n; k_ext = k;

    StraGotoGEMM stra_gemm;

    int nt = comm.num_threads();
    gemm_thread_config tc = make_gemm_thread_config<T>(cfg, nt, m, n, k);
    step<0>(stra_gemm).distribute = tc.jc_nt;
    step<3>(stra_gemm).distribute = tc.ic_nt;
    step<5>(stra_gemm).distribute = tc.jr_nt;
    step<6>(stra_gemm).distribute = tc.ir_nt;

//    std::cout << "A.length(0): " << A.length(0) << "; A.length(1): " << A.length(1) << std::endl;
//    std::cout << "A.stride(0): " << A.stride(0) << "; A.stride(1): " << A.stride(1) << std::endl;

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


    for (unsigned idx = 0; idx < NA; idx++) {
        if (idx == 0) {
          add(comm, cfg, m, k,
              A.coeff(0), false,   A.data(0),  A.stride(0),  A.stride(1),
                    T(0), false,   Sv.data(), Sv.stride(0), Sv.stride(1));
        } else {
            add(comm, cfg, m, k,
             A.coeff(idx), false,  A.data(idx),  A.stride(0),  A.stride(1),
                     T(1), false,    Sv.data(), Sv.stride(0), Sv.stride(1));
        }
    }


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

    for (unsigned idx = 0; idx < NB; idx++) {
        if (idx == 0) {
            add(comm, cfg, k, n,
                B.coeff(0), false,   B.data(0),  B.stride(0),  B.stride(1),
                      T(0), false,   Tv.data(), Tv.stride(0), Tv.stride(1));
        } else {
            add(comm, cfg, k, n,
              B.coeff(idx), false, B.data(idx),  B.stride(0),  B.stride(1),
                      T(1), false,   Tv.data(), Tv.stride(0), Tv.stride(1));
        }
    }


    MemoryPool::Block buffer_m;
    T* m_ptr;
    if (comm.master())
    {
        buffer_m = BuffersForM.allocate<T>( m_ext * n );
        m_ptr = buffer_m.get<T>();
    }
    comm.broadcast(m_ptr);

    matrix_view<T> Mv({m, n}, m_ptr, {1,m_ext});

    //std::cout << "alpha: " << alpha << "; beta: " << beta << std::endl;
    //A_sum (Sv) * B_sum (Tv) -> C_sum (Mv)
    stra_gemm(comm, cfg, alpha, Sv, Tv, T(0), Mv);
    //gemm(comm, cfg, alpha, Sv, Tv, beta, Mv);

    //std::cout << "Sv:" << std::endl;
    //stra_printmat( Sv );
    //std::cout << "Tv:" << std::endl;
    //stra_printmat( Tv );
    //std::cout << "Mv:" << std::endl;
    //stra_printmat( Mv );


    //C_sum -> C0, C1

    for (unsigned idx = 0; idx < NC; idx++) {
            add(comm, cfg, m, n,
             C.coeff(idx), false,   Mv.data(), Mv.stride(0), Mv.stride(1),
                     T(1), false, C.data(idx),  C.stride(0),  C.stride(1));
    }


}

template <typename T, unsigned NA, unsigned NB, unsigned NC>
void straprim_ab(const communicator& comm, const config& cfg,
        T alpha, stra_matrix_view<T, NA>& A, stra_matrix_view<T, NB>& B, T beta, stra_matrix_view<T, NC>& C)
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

    //std::cout << "MR: " << MR << ";NR: " << NR << "; KR: " << KR << std::endl;
    //std::cout << "ME: " << MR << ";NE: " << NR << "; KE: " << KR << std::endl;

    len_type m_ext = ceil_div(m, MR)*ME;
    len_type n_ext = ceil_div(n, NR)*NE;
    len_type k_ext = ceil_div(k, KR)*KE;

    //std::cout << "m: " << m << ";n: " << n << "; k: " << k << std::endl;
    ////m_ext = m; n_ext = n; k_ext = k;

    StraGotoGEMM stra_gemm;

    int nt = comm.num_threads();
    gemm_thread_config tc = make_gemm_thread_config<T>(cfg, nt, m, n, k);
    step<0>(stra_gemm).distribute = tc.jc_nt;
    step<3>(stra_gemm).distribute = tc.ic_nt;
    step<5>(stra_gemm).distribute = tc.jr_nt;
    step<6>(stra_gemm).distribute = tc.ir_nt;

//    std::cout << "A.length(0): " << A.length(0) << "; A.length(1): " << A.length(1) << std::endl;
//    std::cout << "A.stride(0): " << A.stride(0) << "; A.stride(1): " << A.stride(1) << std::endl;


    MemoryPool::Block buffer_m;
    T* m_ptr;
    if (comm.master())
    {
        buffer_m = BuffersForM.allocate<T>( m_ext * n );
        m_ptr = buffer_m.get<T>();
    }
    comm.broadcast(m_ptr);


    matrix_view<T> Mv({m, n}, m_ptr, {1,m_ext});


    //std::cout << "alpha: " << alpha << "; beta: " << beta << std::endl;
    //A_sum (Sv) * B_sum (Tv) -> C_sum (Mv)
    stra_gemm(comm, cfg, alpha, A, B, T(0), Mv);
    //gemm(comm, cfg, alpha, Sv, Tv, beta, Mv);

    //std::cout << "Sv:" << std::endl;
    //stra_printmat( Sv );
    //std::cout << "Tv:" << std::endl;
    //stra_printmat( Tv );
    //std::cout << "Mv:" << std::endl;
    //stra_printmat( Mv );

    //C_sum -> C0, C1

    for (unsigned idx = 0; idx < NC; idx++) {
            add(comm, cfg, m, n,
             C.coeff(idx), false,   Mv.data(), Mv.stride(0), Mv.stride(1),
                     T(1), false, C.data(idx),  C.stride(0),  C.stride(1));
    }


}

#endif
