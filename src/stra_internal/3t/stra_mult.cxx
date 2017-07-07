#include "stra_mult.hpp"

#include "util/gemm_thread.hpp"
#include "util/tensor.hpp"

#include "matrix/stra_tensor_view.hpp"

#include "stra_nodes/stra_matrify.hpp"
#include "stra_nodes/stra_partm.hpp"
#include "stra_nodes/stra_gemm_ukr.hpp"

#include "internal/1t/add.hpp"
#include "stra_internal/3m/stra_mult.hpp"
//#include "internal/3m/mult.hpp"

#define PRINT_VECTOR( name ) \
    std::cout << #name << std::endl; \
    for (auto &elem : name) \
    { \
        std::cout << elem << " "; \
    } \
    std::cout << std::endl;


#include "straprim_common.hpp"



namespace tblis
{
namespace internal
{

//impl_t impl = BLIS_BASED;

extern MemoryPool BuffersForA, BuffersForB, BuffersForScatter;
//MemoryPool BuffersForScatter(4096);

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


template <typename T>
void stra_contract_blas(const communicator& comm, const config& cfg,
                        const std::vector<len_type>& len_AB,
                        const std::vector<len_type>& len_AC,
                        const std::vector<len_type>& len_BC,
                        T alpha, const T* A,
                        const std::vector<stride_type>& stride_A_AB,
                        const std::vector<stride_type>& stride_A_AC,
                                 const T* B,
                        const std::vector<stride_type>& stride_B_AB,
                        const std::vector<stride_type>& stride_B_BC,
                        T  beta,       T* C,
                        const std::vector<stride_type>& stride_C_AC,
                        const std::vector<stride_type>& stride_C_BC)
{
    //std::cout << "Enter TTDT\n" << std::endl;
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

    //PRINT_VECTOR( len_AB )
    //PRINT_VECTOR( len_AC )
    //PRINT_VECTOR( len_BC )
    //PRINT_VECTOR( stride_A_AB )
    //PRINT_VECTOR( stride_A_AC )
    //PRINT_VECTOR( stride_B_AB )
    //PRINT_VECTOR( stride_B_BC )
    //PRINT_VECTOR( stride_C_AC )
    //PRINT_VECTOR( stride_C_BC )



    ////Print A, arv
    //tensor_matrix<T> at( len_AC, len_AB, const_cast<T*>(A), stride_A_AC, stride_A_AB );
    //std::cout << "A: " << std::endl;
    //print_tensor_matrix( at );


    //add(comm, cfg, {}, {}, arv.lengths(),
    //    T(1), false,          A, {}, stride_A_AC+stride_A_AB,
    //    T(0), false, arv.data(), {},           arv.strides());

    tensor_matrix<T> at(len_AC,
                        len_AB,
                        const_cast<T*>(A),
                        stride_A_AC,
                        stride_A_AB);
    add(comm, cfg, T(1), at, T(0), am );


    //std::cout << "at: " << std::endl;
    //print_tensor_matrix( at );
    //std::cout << "am: " << std::endl;
    //tblis_printmat( am );


    //PRINT_VECTOR( arv.lengths() )
    //PRINT_VECTOR( arv.strides() )
    ////Print A, arv
    //std::cout << "arv: " << std::endl;
    //for (unsigned i = 0; i < at.length(0); i++)
    //{
    //    for (unsigned j = 0; j < at.length(1); j++)
    //    {
    //        std::cout << (arv.data())[ i + j * at.length(0) ] << " ";
    //    }
    //    std::cout << std::endl;
    //}


    //add(comm, cfg, {}, {}, brv.lengths(),
    //    T(1), false,          B, {}, stride_B_AB+stride_B_BC,
    //    T(0), false, brv.data(), {},           brv.strides());

    tensor_matrix<T> bt(len_AB,
                        len_BC,
                        const_cast<T*>(B),
                        stride_B_AB,
                        stride_B_BC);
    add(comm, cfg, T(1), bt, T(0), bm );


    //stra_mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
    stra_mult_naive(comm, cfg, cm.length(0), cm.length(1), am.length(1),
              alpha, false, am.data(), am.stride(0), am.stride(1),
                     false, bm.data(), bm.stride(0), bm.stride(1),
               T(0), false, cm.data(), cm.stride(0), cm.stride(1));

    //add(comm, cfg, {}, {}, crv.lengths(),
    //    T(1), false, crv.data(), {},            crv.strides(),
    //    beta, false,          C, {}, stride_C_AC+stride_C_BC);

    tensor_matrix<T> ct(len_AC,
                        len_BC,
                        const_cast<T*>(C),
                        stride_C_AC,
                        stride_C_BC);
    add(comm, cfg, T(1), cm, beta, ct );

}

template <typename T>
void stra_contract_ref(const communicator& comm, const config& cfg,
                       const std::vector<len_type>& len_AB,
                       const std::vector<len_type>& len_AC,
                       const std::vector<len_type>& len_BC,
                       T alpha, const T* A,
                       const std::vector<stride_type>& stride_A_AB,
                       const std::vector<stride_type>& stride_A_AC,
                                const T* B,
                       const std::vector<stride_type>& stride_B_AB,
                       const std::vector<stride_type>& stride_B_BC,
                       T  beta,       T* C,
                       const std::vector<stride_type>& stride_C_AC,
                       const std::vector<stride_type>& stride_C_BC)
{
    (void)cfg;

    MArray::viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
    MArray::viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    MArray::viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);
    len_type m = stl_ext::prod(len_AC);
    len_type n = stl_ext::prod(len_BC);

    len_type m_min, m_max, n_min, n_max;
    std::tie(m_min, m_max, std::ignore,
             n_min, n_max, std::ignore) = comm.distribute_over_threads_2d(m, n);

    const T* A0 = A;
    const T* B0 = B;
          T* C0 = C;

    iter_AC.position(m_min, A0, C0);

    for (len_type i = m_min;i < m_max;i++)
    {
        iter_AC.next(A0, C0);

        A = A0;
        B = B0;
        C = C0;

        iter_BC.position(n_min, B, C);

        for (len_type j = n_min;j < n_max;j++)
        {
            iter_BC.next(B, C);

            T temp = T();

            while (iter_AB.next(A, B))
            {
                temp += (*A)*(*B);
            }
            temp *= alpha;

            if (beta == T(0))
            {
                *C = temp;
            }
            else
            {
                *C = temp + beta*(*C);
            }
        }
    }
}

//template <>
void stra_divide_vector(
          std::vector<len_type>& vec,
          int divisor
          )
{
    //for (auto &elem : vec) {
    //    elem /= divisor;
    //}

    int last_idx = vec.size() - 1;
    vec[ last_idx ] /= divisor;
}

//stra_acquire_tpart( sub_len_AB, sub_len_AC, stride_A_AB, stride_A_AC, 2, 2, 0, 0, A, &A_0 );
template <typename T>
void stra_acquire_tpart(
          const std::vector<len_type>& len_m,
          const std::vector<len_type>& len_n,
          const std::vector<stride_type>& rs,
          const std::vector<stride_type>& cs,
          int x, int y, int s, int t,
          T* srcM, T** dstM
          )
{
    //*dstM = &srcM[ ( m / x * s ) * rs + ( n / y * t ) * cs ]; //src( m/x*i, n/y*j )
    len_type offset = 0;
    //for (int i = 0; i < len_m.size(); i++ ) {
    //    offset += ( len_m[i] / x * s ) * rs[i];
    //}
    //for (int i = 0; i < len_n.size(); i++ ) {
    //    offset += ( len_n[i] / y * t ) * cs[i];
    //}
    int last_idx = len_m.size() - 1;
    offset += ( len_m[ last_idx ] / x * s ) * rs[ last_idx ];
    last_idx = len_n.size() - 1;
    offset += ( len_n[ last_idx ] / y * t ) * cs[ last_idx ];
    *dstM = &srcM[ offset ]; //src( m/x*i, n/y*j )
}

template<typename T>
void print_tensor_matrix( tensor_matrix<T> C ) {
    len_type m = C.length(0);
    len_type n = C.length(1);
    stride_type* scat_buffer = (stride_type*)malloc( sizeof(stride_type) * (m + n) );
    stride_type* rs_c = scat_buffer;
    stride_type* cs_c = scat_buffer + m;
    // Generate rs_c, cs_c;
    C.fill_scatter(0, rs_c);
    C.fill_scatter(1, cs_c);
    //for ( len_AC: stride_C_AC )
    //    for ( len_BC : stride_C_BC )
    //        printf ....
    //m = MULT( len_AC )
    //n = MULT( len_BC )
    for (len_type i = 0;i < m;i++)
    {
        for (len_type j = 0;j < n;j++)
        {
            std::cout << (C.data())[rs_c[i] + cs_c[j]]  << " ";
        }
        std::cout << std::endl;
    }
    free(scat_buffer);
}

template <typename T>
void stra_contract_blis(const communicator& comm, const config& cfg,
                        const std::vector<len_type>& len_AB,
                        const std::vector<len_type>& len_AC,
                        const std::vector<len_type>& len_BC,
                        T alpha, const T* A,
                        const std::vector<stride_type>& stride_A_AB,
                        const std::vector<stride_type>& stride_A_AC,
                                 const T* B,
                        const std::vector<stride_type>& stride_B_AB,
                        const std::vector<stride_type>& stride_B_BC,
                        T  beta,       T* C,
                        const std::vector<stride_type>& stride_C_AC,
                        const std::vector<stride_type>& stride_C_BC)
{
    //std::cout << "Enter stra_internal/3t/stra_mult/stra_contract_blis\n" << std::endl;

    //PRINT_VECTOR( len_AB )
    //PRINT_VECTOR( len_AC )
    //PRINT_VECTOR( len_BC )
    //PRINT_VECTOR( stride_A_AB )
    //PRINT_VECTOR( stride_A_AC )
    //PRINT_VECTOR( stride_B_AB )
    //PRINT_VECTOR( stride_B_BC )
    //PRINT_VECTOR( stride_C_AC )
    //PRINT_VECTOR( stride_C_BC )


    auto reorder_AC = detail::sort_by_stride(stride_C_AC, stride_A_AC);
    auto reorder_BC = detail::sort_by_stride(stride_C_BC, stride_B_BC);
    auto reorder_AB = detail::sort_by_stride(stride_A_AB, stride_B_AB);

    //PRINT_VECTOR( reorder_AC )
    //PRINT_VECTOR( reorder_BC )
    //PRINT_VECTOR( reorder_AB )

    auto my_len_AC = stl_ext::permuted(len_AC, reorder_AC);
    auto my_len_AB = stl_ext::permuted(len_AB, reorder_AB);
    auto my_len_BC = stl_ext::permuted(len_BC, reorder_BC);


    auto my_stride_A_AC = stl_ext::permuted(stride_A_AC, reorder_AC);
    auto my_stride_A_AB = stl_ext::permuted(stride_A_AB, reorder_AB);

    auto my_stride_B_AB = stl_ext::permuted(stride_B_AB, reorder_AB);
    auto my_stride_B_BC = stl_ext::permuted(stride_B_BC, reorder_BC);

    auto my_stride_C_AC = stl_ext::permuted(stride_C_AC, reorder_AC);
    auto my_stride_C_BC = stl_ext::permuted(stride_C_BC, reorder_BC);



    tensor_matrix<T> at(stl_ext::permuted(len_AC, reorder_AC),
                        stl_ext::permuted(len_AB, reorder_AB),
                        const_cast<T*>(A),
                        stl_ext::permuted(stride_A_AC, reorder_AC),
                        stl_ext::permuted(stride_A_AB, reorder_AB));

    tensor_matrix<T> bt(stl_ext::permuted(len_AB, reorder_AB),
                        stl_ext::permuted(len_BC, reorder_BC),
                        const_cast<T*>(B),
                        stl_ext::permuted(stride_B_AB, reorder_AB),
                        stl_ext::permuted(stride_B_BC, reorder_BC));

    tensor_matrix<T> ct(stl_ext::permuted(len_AC, reorder_AC),
                        stl_ext::permuted(len_BC, reorder_BC),
                        C,
                        stl_ext::permuted(stride_C_AC, reorder_AC),
                        stl_ext::permuted(stride_C_BC, reorder_BC));



    const bool row_major = cfg.gemm_row_major.value<T>();

    //if (ct.stride(!row_major) == 1)
    //{
    //    /*
    //     * Compute C^T = B^T * A^T instead
    //     */
    //    at.swap(bt);
    //    at.transpose();
    //    bt.transpose();
    //    ct.transpose();
    //}


    len_type m = ct.length(0);
    len_type n = ct.length(1);
    len_type k = at.length(1);

    len_type md, kd, nd;
    len_type mr, kr, nr;

    mr = m % ( 2 ), kr = k % ( 2 ), nr = n % ( 2 );
    md = m - mr, kd = k - kr, nd = n - nr;

    //ms=md, ks=kd, ns=nd;
    //ms=ms/2, ks=ks/2, ns=ns/2;

    const len_type ms=m/2, ks=k/2, ns=n/2;


    StraTensorGEMM stra_gemm;
    int nt = comm.num_threads();
    auto tc = make_gemm_thread_config<T>(cfg, nt, ms, ns, ks);
    step<0>(stra_gemm).distribute = tc.jc_nt;
    step<4>(stra_gemm).distribute = tc.ic_nt;
    step<8>(stra_gemm).distribute = tc.jr_nt;
    step<9>(stra_gemm).distribute = tc.ir_nt;

    //stra_gemm(comm, cfg, alpha, at, bt, beta, ct);

    
    const std::array<unsigned,2> A_divisor={2,2};
    const std::array<unsigned,2> B_divisor={2,2};
    const std::array<unsigned,2> C_divisor={2,2};


    // M0 = (1 * A_0 + 1 * A_3) * (1 * B_0 + 1 * B_3);  C_0 += 1 * M0;  C_3 += 1 * M0;
    std::array<unsigned, 2> A0_subid = {0, 3};
    std::array<T,2> A0_coeff_list = {1, 1};
    stra_tensor_view<T,2> Av0(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A0_subid, A0_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B0_subid = {0, 3};
    std::array<T,2> B0_coeff_list = {1, 1};
    stra_tensor_view<T,2> Bv0(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B0_subid, B0_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C0_subid = {0, 3};
    std::array<T,2> C0_coeff_list = {1, 1};
    stra_tensor_view<T,2> Cv0(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C0_subid, C0_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv0.stride(!row_major) == 1)
    {
        Av0.transpose();
        Bv0.transpose();
        Cv0.transpose();
        stra_gemm(comm, cfg, alpha, Bv0, Av0, beta, Cv0);
    } else {
        stra_gemm(comm, cfg, alpha, Av0, Bv0, beta, Cv0);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_B:" << std::endl;
    //print_tensor_matrix( bt );


    //std::cout << "Finish M0" << std::endl;

    // M1 = (1 * A_2 + 1 * A_3) * (1 * B_0);  C_2 += 1 * M1;  C_3 += -1 * M1;
    std::array<unsigned, 2> A1_subid = {2, 3};
    std::array<T,2> A1_coeff_list = {1, 1};
    stra_tensor_view<T,2> Av1(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A1_subid, A1_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 1> B1_subid = {0};
    std::array<T,1> B1_coeff_list = {1};
    stra_tensor_view<T,1> Bv1(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B1_subid, B1_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C1_subid = {2, 3};
    std::array<T,2> C1_coeff_list = {1, -1};
    stra_tensor_view<T,2> Cv1(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C1_subid, C1_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv1.stride(!row_major) == 1)
    {
        //std::cout << "transposed" << std::endl;
        Av1.transpose();
        Bv1.transpose();
        Cv1.transpose();
        stra_gemm(comm, cfg, alpha, Bv1, Av1, beta, Cv1);
    } else {
        //std::cout << "non-transposed" << std::endl;
        stra_gemm(comm, cfg, alpha, Av1, Bv1, beta, Cv1);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M1:" << std::endl;
    //print_tensor_matrix( ct );

    //std::cout << "Finish M1" << std::endl;


    // M2 = (1 * A_0) * (1 * B_1 + -1 * B_3);  C_1 += 1 * M2;  C_3 += 1 * M2;
    std::array<unsigned, 1> A2_subid = {0};
    std::array<T,1> A2_coeff_list = {1};
    stra_tensor_view<T,1> Av2(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A2_subid, A2_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B2_subid = {1, 3};
    std::array<T,2> B2_coeff_list = {1, -1};
    stra_tensor_view<T,2> Bv2(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B2_subid, B2_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C2_subid = {1, 3};
    std::array<T,2> C2_coeff_list = {1, 1};
    stra_tensor_view<T,2> Cv2(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C2_subid, C2_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv2.stride(!row_major) == 1)
    {
        Av2.transpose();
        Bv2.transpose();
        Cv2.transpose();
        stra_gemm(comm, cfg, alpha, Bv2, Av2, beta, Cv2);
    } else {
        stra_gemm(comm, cfg, alpha, Av2, Bv2, beta, Cv2);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M2:" << std::endl;
    //print_tensor_matrix( ct );

    // M3 = (1 * A_3) * (-1 * B_0 + 1 * B_2);  C_0 += 1 * M3;  C_2 += 1 * M3;
    std::array<unsigned, 1> A3_subid = {3};
    std::array<T,1> A3_coeff_list = {1};
    stra_tensor_view<T,1> Av3(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A3_subid, A3_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B3_subid = {0, 2};
    std::array<T,2> B3_coeff_list = {-1, 1};
    stra_tensor_view<T,2> Bv3(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B3_subid, B3_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C3_subid = {0, 2};
    std::array<T,2> C3_coeff_list = {1, 1};
    stra_tensor_view<T,2> Cv3(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C3_subid, C3_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv3.stride(!row_major) == 1)
    {
        Av3.transpose();
        Bv3.transpose();
        Cv3.transpose();
        stra_gemm(comm, cfg, alpha, Bv3, Av3, beta, Cv3);
    } else {
        stra_gemm(comm, cfg, alpha, Av3, Bv3, beta, Cv3);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M3:" << std::endl;
    //print_tensor_matrix( ct );

    // M4 = (1 * A_0 + 1 * A_1) * (1 * B_3);  C_0 += -1 * M4;  C_1 += 1 * M4;
    std::array<unsigned, 2> A4_subid = {0, 1};
    std::array<T,2> A4_coeff_list = {1, 1};
    stra_tensor_view<T,2> Av4(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A4_subid, A4_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 1> B4_subid = {3};
    std::array<T,1> B4_coeff_list = {1};
    stra_tensor_view<T,1> Bv4(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B4_subid, B4_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C4_subid = {0, 1};
    std::array<T,2> C4_coeff_list = {-1, 1};
    stra_tensor_view<T,2> Cv4(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C4_subid, C4_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv4.stride(!row_major) == 1)
    {
        Av4.transpose();
        Bv4.transpose();
        Cv4.transpose();
        stra_gemm(comm, cfg, alpha, Bv4, Av4, beta, Cv4);
    } else {
        stra_gemm(comm, cfg, alpha, Av4, Bv4, beta, Cv4);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M4:" << std::endl;
    //print_tensor_matrix( ct );

    // M5 = (-1 * A_0 + 1 * A_2) * (1 * B_0 + 1 * B_1);  C_3 += 1 * M5;
    std::array<unsigned, 2> A5_subid = {0, 2};
    std::array<T,2> A5_coeff_list = {-1, 1};
    stra_tensor_view<T,2> Av5(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A5_subid, A5_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B5_subid = {0, 1};
    std::array<T,2> B5_coeff_list = {1, 1};
    stra_tensor_view<T,2> Bv5(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B5_subid, B5_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 1> C5_subid = {3};
    std::array<T,1> C5_coeff_list = {1};
    stra_tensor_view<T,1> Cv5(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C5_subid, C5_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv5.stride(!row_major) == 1)
    {
        Av5.transpose();
        Bv5.transpose();
        Cv5.transpose();
        stra_gemm(comm, cfg, alpha, Bv5, Av5, beta, Cv5);
    } else {
        stra_gemm(comm, cfg, alpha, Av5, Bv5, beta, Cv5);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M5:" << std::endl;
    //print_tensor_matrix( ct );

    // M6 = (1 * A_1 + -1 * A_3) * (1 * B_2 + 1 * B_3);  C_0 += 1 * M6;
    std::array<unsigned, 2> A6_subid = {1, 3};
    std::array<T,2> A6_coeff_list = {1, -1};
    stra_tensor_view<T,2> Av6(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A6_subid, A6_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B6_subid = {2, 3};
    std::array<T,2> B6_coeff_list = {1, 1};
    stra_tensor_view<T,2> Bv6(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B6_subid, B6_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 1> C6_subid = {0};
    std::array<T,1> C6_coeff_list = {1};
    stra_tensor_view<T,1> Cv6(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C6_subid, C6_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv6.stride(!row_major) == 1)
    {
        Av6.transpose();
        Bv6.transpose();
        Cv6.transpose();
        stra_gemm(comm, cfg, alpha, Bv6, Av6, beta, Cv6);
    } else {
        stra_gemm(comm, cfg, alpha, Av6, Bv6, beta, Cv6);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M6:" << std::endl;
    //print_tensor_matrix( ct );


    //std::cout << "stra_internal/stra_mult_A:" << std::endl;
    //print_tensor_matrix( at );

    //std::cout << "stra_internal/stra_mult_B:" << std::endl;
    //print_tensor_matrix( bt );

    //std::cout << "stra_internal/stra_mult_M6:" << std::endl;
    //print_tensor_matrix( ct );

}

#define INSTANTIATE_CONTRACT_BLIS(T) \
template void stra_contract_blis(const communicator& comm, const config& cfg, \
                                 const std::vector<len_type>& len_AB, \
                                 const std::vector<len_type>& len_AC, \
                                 const std::vector<len_type>& len_BC, \
                                 T alpha, const T* A, \
                                 const std::vector<stride_type>& stride_A_AB, \
                                 const std::vector<stride_type>& stride_A_AC, \
                                          const T* B, \
                                 const std::vector<stride_type>& stride_B_AB, \
                                 const std::vector<stride_type>& stride_B_BC, \
                                 T  beta,       T* C, \
                                 const std::vector<stride_type>& stride_C_AC, \
                                 const std::vector<stride_type>& stride_C_BC);

INSTANTIATE_CONTRACT_BLIS(float);
INSTANTIATE_CONTRACT_BLIS(double);
INSTANTIATE_CONTRACT_BLIS(scomplex);
INSTANTIATE_CONTRACT_BLIS(dcomplex);



template <typename T>
void stra_contract_blis_naive(const communicator& comm, const config& cfg,
                              const std::vector<len_type>& len_AB,
                              const std::vector<len_type>& len_AC,
                              const std::vector<len_type>& len_BC,
                              T alpha, const T* A,
                              const std::vector<stride_type>& stride_A_AB,
                              const std::vector<stride_type>& stride_A_AC,
                                       const T* B,
                              const std::vector<stride_type>& stride_B_AB,
                              const std::vector<stride_type>& stride_B_BC,
                              T  beta,       T* C,
                              const std::vector<stride_type>& stride_C_AC,
                              const std::vector<stride_type>& stride_C_BC)
{
    //std::cout << "Enter stra_internal/3t/stra_mult/stra_contract_blis_naive\n" << std::endl;

    //PRINT_VECTOR( len_AB )
    //PRINT_VECTOR( len_AC )
    //PRINT_VECTOR( len_BC )
    //PRINT_VECTOR( stride_A_AB )
    //PRINT_VECTOR( stride_A_AC )
    //PRINT_VECTOR( stride_B_AB )
    //PRINT_VECTOR( stride_B_BC )
    //PRINT_VECTOR( stride_C_AC )
    //PRINT_VECTOR( stride_C_BC )


    auto reorder_AC = detail::sort_by_stride(stride_C_AC, stride_A_AC);
    auto reorder_BC = detail::sort_by_stride(stride_C_BC, stride_B_BC);
    auto reorder_AB = detail::sort_by_stride(stride_A_AB, stride_B_AB);

    //PRINT_VECTOR( reorder_AC )
    //PRINT_VECTOR( reorder_BC )
    //PRINT_VECTOR( reorder_AB )

    auto my_len_AC = stl_ext::permuted(len_AC, reorder_AC);
    auto my_len_AB = stl_ext::permuted(len_AB, reorder_AB);
    auto my_len_BC = stl_ext::permuted(len_BC, reorder_BC);


    auto my_stride_A_AC = stl_ext::permuted(stride_A_AC, reorder_AC);
    auto my_stride_A_AB = stl_ext::permuted(stride_A_AB, reorder_AB);

    auto my_stride_B_AB = stl_ext::permuted(stride_B_AB, reorder_AB);
    auto my_stride_B_BC = stl_ext::permuted(stride_B_BC, reorder_BC);

    auto my_stride_C_AC = stl_ext::permuted(stride_C_AC, reorder_AC);
    auto my_stride_C_BC = stl_ext::permuted(stride_C_BC, reorder_BC);


    const bool row_major = cfg.gemm_row_major.value<T>();

    tensor_matrix<T> at(stl_ext::permuted(len_AC, reorder_AC),
                        stl_ext::permuted(len_AB, reorder_AB),
                        const_cast<T*>(A),
                        stl_ext::permuted(stride_A_AC, reorder_AC),
                        stl_ext::permuted(stride_A_AB, reorder_AB));

    tensor_matrix<T> bt(stl_ext::permuted(len_AB, reorder_AB),
                        stl_ext::permuted(len_BC, reorder_BC),
                        const_cast<T*>(B),
                        stl_ext::permuted(stride_B_AB, reorder_AB),
                        stl_ext::permuted(stride_B_BC, reorder_BC));

    tensor_matrix<T> ct(stl_ext::permuted(len_AC, reorder_AC),
                        stl_ext::permuted(len_BC, reorder_BC),
                        C,
                        stl_ext::permuted(stride_C_AC, reorder_AC),
                        stl_ext::permuted(stride_C_BC, reorder_BC));


    //if (ct.stride(!row_major) == 1)
    //{
    //    //Compute C^T = B^T * A^T instead
    //    at.swap(bt);
    //    at.transpose();
    //    bt.transpose();
    //    ct.transpose();
    //}


    len_type m = ct.length(0);
    len_type n = ct.length(1);
    len_type k = at.length(1);

    len_type md, kd, nd;
    len_type mr, kr, nr;

    mr = m % ( 2 ), kr = k % ( 2 ), nr = n % ( 2 );
    md = m - mr, kd = k - kr, nd = n - nr;

    //ms=md, ks=kd, ns=nd;
    //ms=ms/2, ks=ks/2, ns=ns/2;

    const len_type ms=m/2, ks=k/2, ns=n/2;


    std::vector<len_type> my_sub_len_AB;
    std::vector<len_type> my_sub_len_AC;
    std::vector<len_type> my_sub_len_BC;

    my_sub_len_AB = my_len_AB;
    my_sub_len_AC = my_len_AC;
    my_sub_len_BC = my_len_BC;

    //ms=md, ks=kd, ns=nd;
    const T *A_0, *A_1, *A_2, *A_3;
    stra_acquire_tpart( my_len_AC, my_len_AB, my_stride_A_AC, my_stride_A_AB, 2, 2, 0, 0, A, &A_0 );
    stra_acquire_tpart( my_len_AC, my_len_AB, my_stride_A_AC, my_stride_A_AB, 2, 2, 0, 1, A, &A_1 );
    stra_acquire_tpart( my_len_AC, my_len_AB, my_stride_A_AC, my_stride_A_AB, 2, 2, 1, 0, A, &A_2 );
    stra_acquire_tpart( my_len_AC, my_len_AB, my_stride_A_AC, my_stride_A_AB, 2, 2, 1, 1, A, &A_3 );

    //ms=md, ks=kd, ns=nd;
    const T *B_0, *B_1, *B_2, *B_3;
    stra_acquire_tpart( my_len_AB, my_len_BC, my_stride_B_AB, my_stride_B_BC, 2, 2, 0, 0, B, &B_0 );
    stra_acquire_tpart( my_len_AB, my_len_BC, my_stride_B_AB, my_stride_B_BC, 2, 2, 0, 1, B, &B_1 );
    stra_acquire_tpart( my_len_AB, my_len_BC, my_stride_B_AB, my_stride_B_BC, 2, 2, 1, 0, B, &B_2 );
    stra_acquire_tpart( my_len_AB, my_len_BC, my_stride_B_AB, my_stride_B_BC, 2, 2, 1, 1, B, &B_3 );

    //ms=md, ks=kd, ns=nd;
    T *C_0, *C_1, *C_2, *C_3;
    stra_acquire_tpart( my_len_AC, my_len_BC, my_stride_C_AC, my_stride_C_BC, 2, 2, 0, 0, C, &C_0 );
    stra_acquire_tpart( my_len_AC, my_len_BC, my_stride_C_AC, my_stride_C_BC, 2, 2, 0, 1, C, &C_1 );
    stra_acquire_tpart( my_len_AC, my_len_BC, my_stride_C_AC, my_stride_C_BC, 2, 2, 1, 0, C, &C_2 );
    stra_acquire_tpart( my_len_AC, my_len_BC, my_stride_C_AC, my_stride_C_BC, 2, 2, 1, 1, C, &C_3 );

    //len_AB, len_AC, len_BC % 2
    stra_divide_vector( my_sub_len_AB, 2 );
    stra_divide_vector( my_sub_len_AC, 2 );
    stra_divide_vector( my_sub_len_BC, 2 );


    //StraTensorGEMM stra_gemm;
    //int nt = comm.num_threads();
    //auto tc = make_gemm_thread_config<T>(cfg, nt, ms, ns, ks);
    //step<0>(stra_gemm).distribute = tc.jc_nt;
    //step<4>(stra_gemm).distribute = tc.ic_nt;
    //step<8>(stra_gemm).distribute = tc.jr_nt;
    //step<9>(stra_gemm).distribute = tc.ir_nt;

    //stra_gemm(comm, cfg, alpha, at, bt, beta, ct);

    const std::array<unsigned,2> A_divisor={2,2};
    const std::array<unsigned,2> B_divisor={2,2};
    const std::array<unsigned,2> C_divisor={2,2};

    // M0 = (1 * A_0 + 1 * A_3) * (1 * B_0 + 1 * B_3);  C_0 += 1 * M0;  C_3 += 1 * M0;
    std::array<unsigned, 2> A0_subid = {0, 3};
    std::array<T,2> A0_coeff_list = {1, 1};
    stra_tensor_view<T,2> Av0(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A0_subid, A0_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B0_subid = {0, 3};
    std::array<T,2> B0_coeff_list = {1, 1};
    stra_tensor_view<T,2> Bv0(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B0_subid, B0_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C0_subid = {0, 3};
    std::array<T,2> C0_coeff_list = {1, 1};
    stra_tensor_view<T,2> Cv0(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C0_subid, C0_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv0.stride(!row_major) == 1)
    {
        Av0.transpose();
        Bv0.transpose();
        Cv0.transpose();
        straprim_naive2(comm, cfg, alpha, Bv0, Av0, beta, Cv0);
    } else {
        straprim_naive2(comm, cfg, alpha, Av0, Bv0, beta, Cv0);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M0:" << std::endl;
    //print_tensor_matrix( ct );

    // M1 = (1 * A_2 + 1 * A_3) * (1 * B_0);  C_2 += 1 * M1;  C_3 += -1 * M1;
    std::array<unsigned, 2> A1_subid = {2, 3};
    std::array<T,2> A1_coeff_list = {1, 1};
    stra_tensor_view<T,2> Av1(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A1_subid, A1_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 1> B1_subid = {0};
    std::array<T,1> B1_coeff_list = {1};
    stra_tensor_view<T,1> Bv1(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B1_subid, B1_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C1_subid = {2, 3};
    std::array<T,2> C1_coeff_list = {1, -1};
    stra_tensor_view<T,2> Cv1(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C1_subid, C1_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv1.stride(!row_major) == 1)
    {
        Av1.transpose();
        Bv1.transpose();
        Cv1.transpose();
        straprim_naive2(comm, cfg, alpha, Bv1, Av1, beta, Cv1);
    } else {
        straprim_naive2(comm, cfg, alpha, Av1, Bv1, beta, Cv1);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M1:" << std::endl;
    //print_tensor_matrix( ct );

    // M2 = (1 * A_0) * (1 * B_1 + -1 * B_3);  C_1 += 1 * M2;  C_3 += 1 * M2;
    std::array<unsigned, 1> A2_subid = {0};
    std::array<T,1> A2_coeff_list = {1};
    stra_tensor_view<T,1> Av2(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A2_subid, A2_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B2_subid = {1, 3};
    std::array<T,2> B2_coeff_list = {1, -1};
    stra_tensor_view<T,2> Bv2(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B2_subid, B2_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C2_subid = {1, 3};
    std::array<T,2> C2_coeff_list = {1, 1};
    stra_tensor_view<T,2> Cv2(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C2_subid, C2_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv2.stride(!row_major) == 1)
    {
        Av2.transpose();
        Bv2.transpose();
        Cv2.transpose();
        straprim_naive2(comm, cfg, alpha, Bv2, Av2, beta, Cv2);
    } else {
        straprim_naive2(comm, cfg, alpha, Av2, Bv2, beta, Cv2);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M2:" << std::endl;
    //print_tensor_matrix( ct );

    // M3 = (1 * A_3) * (-1 * B_0 + 1 * B_2);  C_0 += 1 * M3;  C_2 += 1 * M3;
    std::array<unsigned, 1> A3_subid = {3};
    std::array<T,1> A3_coeff_list = {1};
    stra_tensor_view<T,1> Av3(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A3_subid, A3_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B3_subid = {0, 2};
    std::array<T,2> B3_coeff_list = {-1, 1};
    stra_tensor_view<T,2> Bv3(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B3_subid, B3_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C3_subid = {0, 2};
    std::array<T,2> C3_coeff_list = {1, 1};
    stra_tensor_view<T,2> Cv3(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C3_subid, C3_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv3.stride(!row_major) == 1)
    {
        Av3.transpose();
        Bv3.transpose();
        Cv3.transpose();
        straprim_naive2(comm, cfg, alpha, Bv3, Av3, beta, Cv3);
    } else {
        straprim_naive2(comm, cfg, alpha, Av3, Bv3, beta, Cv3);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M3:" << std::endl;
    //print_tensor_matrix( ct );

    // M4 = (1 * A_0 + 1 * A_1) * (1 * B_3);  C_0 += -1 * M4;  C_1 += 1 * M4;
    std::array<unsigned, 2> A4_subid = {0, 1};
    std::array<T,2> A4_coeff_list = {1, 1};
    stra_tensor_view<T,2> Av4(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A4_subid, A4_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 1> B4_subid = {3};
    std::array<T,1> B4_coeff_list = {1};
    stra_tensor_view<T,1> Bv4(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B4_subid, B4_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C4_subid = {0, 1};
    std::array<T,2> C4_coeff_list = {-1, 1};
    stra_tensor_view<T,2> Cv4(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C4_subid, C4_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv4.stride(!row_major) == 1)
    {
        Av4.transpose();
        Bv4.transpose();
        Cv4.transpose();
        straprim_naive2(comm, cfg, alpha, Bv4, Av4, beta, Cv4);
    } else {
        straprim_naive2(comm, cfg, alpha, Av4, Bv4, beta, Cv4);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M4:" << std::endl;
    //print_tensor_matrix( ct );

    // M5 = (-1 * A_0 + 1 * A_2) * (1 * B_0 + 1 * B_1);  C_3 += 1 * M5;
    std::array<unsigned, 2> A5_subid = {0, 2};
    std::array<T,2> A5_coeff_list = {-1, 1};
    stra_tensor_view<T,2> Av5(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A5_subid, A5_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B5_subid = {0, 1};
    std::array<T,2> B5_coeff_list = {1, 1};
    stra_tensor_view<T,2> Bv5(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B5_subid, B5_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 1> C5_subid = {3};
    std::array<T,1> C5_coeff_list = {1};
    stra_tensor_view<T,1> Cv5(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C5_subid, C5_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv5.stride(!row_major) == 1)
    {
        Av5.transpose();
        Bv5.transpose();
        Cv5.transpose();
        straprim_naive2(comm, cfg, alpha, Bv5, Av5, beta, Cv5);
    } else {
        straprim_naive2(comm, cfg, alpha, Av5, Bv5, beta, Cv5);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M5:" << std::endl;
    //print_tensor_matrix( ct );

    // M6 = (1 * A_1 + -1 * A_3) * (1 * B_2 + 1 * B_3);  C_0 += 1 * M6;
    std::array<unsigned, 2> A6_subid = {1, 3};
    std::array<T,2> A6_coeff_list = {1, -1};
    stra_tensor_view<T,2> Av6(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A6_subid, A6_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B6_subid = {2, 3};
    std::array<T,2> B6_coeff_list = {1, 1};
    stra_tensor_view<T,2> Bv6(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B6_subid, B6_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 1> C6_subid = {0};
    std::array<T,1> C6_coeff_list = {1};
    stra_tensor_view<T,1> Cv6(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C6_subid, C6_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv6.stride(!row_major) == 1)
    {
        Av6.transpose();
        Bv6.transpose();
        Cv6.transpose();
        straprim_naive2(comm, cfg, alpha, Bv6, Av6, beta, Cv6);
    } else {
        straprim_naive2(comm, cfg, alpha, Av6, Bv6, beta, Cv6);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M6:" << std::endl;
    //print_tensor_matrix( ct );

    




    //std::cout << "stra_internal/stra_mult_A:" << std::endl;
    //print_tensor_matrix( at );

    //std::cout << "stra_internal/stra_mult_B:" << std::endl;
    //print_tensor_matrix( bt );

    //std::cout << "stra_internal/stra_mult_M6:" << std::endl;
    //print_tensor_matrix( ct );

}

#define INSTANTIATE_CONTRACT_BLIS_NAIVE(T) \
template void stra_contract_blis_naive(const communicator& comm, const config& cfg, \
                                 const std::vector<len_type>& len_AB, \
                                 const std::vector<len_type>& len_AC, \
                                 const std::vector<len_type>& len_BC, \
                                 T alpha, const T* A, \
                                 const std::vector<stride_type>& stride_A_AB, \
                                 const std::vector<stride_type>& stride_A_AC, \
                                          const T* B, \
                                 const std::vector<stride_type>& stride_B_AB, \
                                 const std::vector<stride_type>& stride_B_BC, \
                                 T  beta,       T* C, \
                                 const std::vector<stride_type>& stride_C_AC, \
                                 const std::vector<stride_type>& stride_C_BC);

INSTANTIATE_CONTRACT_BLIS_NAIVE(float);
INSTANTIATE_CONTRACT_BLIS_NAIVE(double);
INSTANTIATE_CONTRACT_BLIS_NAIVE(scomplex);
INSTANTIATE_CONTRACT_BLIS_NAIVE(dcomplex);



template <typename T>
void stra_contract_blis_ab(const communicator& comm, const config& cfg,
                              const std::vector<len_type>& len_AB,
                              const std::vector<len_type>& len_AC,
                              const std::vector<len_type>& len_BC,
                              T alpha, const T* A,
                              const std::vector<stride_type>& stride_A_AB,
                              const std::vector<stride_type>& stride_A_AC,
                                       const T* B,
                              const std::vector<stride_type>& stride_B_AB,
                              const std::vector<stride_type>& stride_B_BC,
                              T  beta,       T* C,
                              const std::vector<stride_type>& stride_C_AC,
                              const std::vector<stride_type>& stride_C_BC)
{
    //std::cout << "Enter stra_internal/3t/stra_mult/stra_contract_blis_ab\n" << std::endl;

    //PRINT_VECTOR( len_AB )
    //PRINT_VECTOR( len_AC )
    //PRINT_VECTOR( len_BC )
    //PRINT_VECTOR( stride_A_AB )
    //PRINT_VECTOR( stride_A_AC )
    //PRINT_VECTOR( stride_B_AB )
    //PRINT_VECTOR( stride_B_BC )
    //PRINT_VECTOR( stride_C_AC )
    //PRINT_VECTOR( stride_C_BC )


    auto reorder_AC = detail::sort_by_stride(stride_C_AC, stride_A_AC);
    auto reorder_BC = detail::sort_by_stride(stride_C_BC, stride_B_BC);
    auto reorder_AB = detail::sort_by_stride(stride_A_AB, stride_B_AB);

    //PRINT_VECTOR( reorder_AC )
    //PRINT_VECTOR( reorder_BC )
    //PRINT_VECTOR( reorder_AB )

    auto my_len_AC = stl_ext::permuted(len_AC, reorder_AC);
    auto my_len_AB = stl_ext::permuted(len_AB, reorder_AB);
    auto my_len_BC = stl_ext::permuted(len_BC, reorder_BC);


    auto my_stride_A_AC = stl_ext::permuted(stride_A_AC, reorder_AC);
    auto my_stride_A_AB = stl_ext::permuted(stride_A_AB, reorder_AB);

    auto my_stride_B_AB = stl_ext::permuted(stride_B_AB, reorder_AB);
    auto my_stride_B_BC = stl_ext::permuted(stride_B_BC, reorder_BC);

    auto my_stride_C_AC = stl_ext::permuted(stride_C_AC, reorder_AC);
    auto my_stride_C_BC = stl_ext::permuted(stride_C_BC, reorder_BC);



    tensor_matrix<T> at(stl_ext::permuted(len_AC, reorder_AC),
                        stl_ext::permuted(len_AB, reorder_AB),
                        const_cast<T*>(A),
                        stl_ext::permuted(stride_A_AC, reorder_AC),
                        stl_ext::permuted(stride_A_AB, reorder_AB));

    tensor_matrix<T> bt(stl_ext::permuted(len_AB, reorder_AB),
                        stl_ext::permuted(len_BC, reorder_BC),
                        const_cast<T*>(B),
                        stl_ext::permuted(stride_B_AB, reorder_AB),
                        stl_ext::permuted(stride_B_BC, reorder_BC));

    tensor_matrix<T> ct(stl_ext::permuted(len_AC, reorder_AC),
                        stl_ext::permuted(len_BC, reorder_BC),
                        C,
                        stl_ext::permuted(stride_C_AC, reorder_AC),
                        stl_ext::permuted(stride_C_BC, reorder_BC));



    const bool row_major = cfg.gemm_row_major.value<T>();

    //if (ct.stride(!row_major) == 1)
    //{
    //    //Compute C^T = B^T * A^T instead
    //    at.swap(bt);
    //    at.transpose();
    //    bt.transpose();
    //    ct.transpose();
    //}

    StraTensorGEMM stra_gemm;

    len_type m = ct.length(0);
    len_type n = ct.length(1);
    len_type k = at.length(1);

    int nt = comm.num_threads();
    auto tc = make_gemm_thread_config<T>(cfg, nt, m, n, k);
    step<0>(stra_gemm).distribute = tc.jc_nt;
    step<4>(stra_gemm).distribute = tc.ic_nt;
    step<8>(stra_gemm).distribute = tc.jr_nt;
    step<9>(stra_gemm).distribute = tc.ir_nt;

    //stra_gemm(comm, cfg, alpha, at, bt, beta, ct);



    len_type md, kd, nd;
    len_type mr, kr, nr;

    mr = m % ( 2 ), kr = k % ( 2 ), nr = n % ( 2 );
    md = m - mr, kd = k - kr, nd = n - nr;

    //ms=md, ks=kd, ns=nd;
    //ms=ms/2, ks=ks/2, ns=ns/2;

    const len_type ms=m/2, ks=k/2, ns=n/2;


    std::vector<len_type> my_sub_len_AB;
    std::vector<len_type> my_sub_len_AC;
    std::vector<len_type> my_sub_len_BC;

    my_sub_len_AB = my_len_AB;
    my_sub_len_AC = my_len_AC;
    my_sub_len_BC = my_len_BC;

    //ms=md, ks=kd, ns=nd;
    T *C_0, *C_1, *C_2, *C_3;
    stra_acquire_tpart( my_sub_len_AC, my_sub_len_BC, my_stride_C_AC, my_stride_C_BC, 2, 2, 0, 0, C, &C_0 );
    stra_acquire_tpart( my_sub_len_AC, my_sub_len_BC, my_stride_C_AC, my_stride_C_BC, 2, 2, 0, 1, C, &C_1 );
    stra_acquire_tpart( my_sub_len_AC, my_sub_len_BC, my_stride_C_AC, my_stride_C_BC, 2, 2, 1, 0, C, &C_2 );
    stra_acquire_tpart( my_sub_len_AC, my_sub_len_BC, my_stride_C_AC, my_stride_C_BC, 2, 2, 1, 1, C, &C_3 );
    
    //len_AB, len_AC, len_BC % 2
    stra_divide_vector( my_sub_len_AB, 2 );
    stra_divide_vector( my_sub_len_AC, 2 );
    stra_divide_vector( my_sub_len_BC, 2 );


    const std::array<unsigned,2> A_divisor={2,2};
    const std::array<unsigned,2> B_divisor={2,2};
    const std::array<unsigned,2> C_divisor={2,2};

    // M0 = (1 * A_0 + 1 * A_3) * (1 * B_0 + 1 * B_3);  C_0 += 1 * M0;  C_3 += 1 * M0;
    std::array<unsigned, 2> A0_subid = {0, 3};
    std::array<T,2> A0_coeff_list = {1, 1};
    stra_tensor_view<T,2> Av0(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A0_subid, A0_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B0_subid = {0, 3};
    std::array<T,2> B0_coeff_list = {1, 1};
    stra_tensor_view<T,2> Bv0(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B0_subid, B0_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C0_subid = {0, 3};
    std::array<T,2> C0_coeff_list = {1, 1};
    stra_tensor_view<T,2> Cv0(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C0_subid, C0_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv0.stride(!row_major) == 1)
    {
        Av0.transpose();
        Bv0.transpose();
        Cv0.transpose();
        straprim_ab2(comm, cfg, alpha, Bv0, Av0, beta, Cv0);
    } else {
        straprim_ab2(comm, cfg, alpha, Av0, Bv0, beta, Cv0);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M0:" << std::endl;
    //print_tensor_matrix( ct );

    // M1 = (1 * A_2 + 1 * A_3) * (1 * B_0);  C_2 += 1 * M1;  C_3 += -1 * M1;
    std::array<unsigned, 2> A1_subid = {2, 3};
    std::array<T,2> A1_coeff_list = {1, 1};
    stra_tensor_view<T,2> Av1(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A1_subid, A1_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 1> B1_subid = {0};
    std::array<T,1> B1_coeff_list = {1};
    stra_tensor_view<T,1> Bv1(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B1_subid, B1_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C1_subid = {2, 3};
    std::array<T,2> C1_coeff_list = {1, -1};
    stra_tensor_view<T,2> Cv1(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C1_subid, C1_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv1.stride(!row_major) == 1)
    {
        Av1.transpose();
        Bv1.transpose();
        Cv1.transpose();
        straprim_ab2(comm, cfg, alpha, Bv1, Av1, beta, Cv1);
    } else {
        straprim_ab2(comm, cfg, alpha, Av1, Bv1, beta, Cv1);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M1:" << std::endl;
    //print_tensor_matrix( ct );

    // M2 = (1 * A_0) * (1 * B_1 + -1 * B_3);  C_1 += 1 * M2;  C_3 += 1 * M2;
    std::array<unsigned, 1> A2_subid = {0};
    std::array<T,1> A2_coeff_list = {1};
    stra_tensor_view<T,1> Av2(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A2_subid, A2_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B2_subid = {1, 3};
    std::array<T,2> B2_coeff_list = {1, -1};
    stra_tensor_view<T,2> Bv2(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B2_subid, B2_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C2_subid = {1, 3};
    std::array<T,2> C2_coeff_list = {1, 1};
    stra_tensor_view<T,2> Cv2(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C2_subid, C2_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv2.stride(!row_major) == 1)
    {
        Av2.transpose();
        Bv2.transpose();
        Cv2.transpose();
        straprim_ab2(comm, cfg, alpha, Bv2, Av2, beta, Cv2);
    } else {
        straprim_ab2(comm, cfg, alpha, Av2, Bv2, beta, Cv2);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M2:" << std::endl;
    //print_tensor_matrix( ct );

    // M3 = (1 * A_3) * (-1 * B_0 + 1 * B_2);  C_0 += 1 * M3;  C_2 += 1 * M3;
    std::array<unsigned, 1> A3_subid = {3};
    std::array<T,1> A3_coeff_list = {1};
    stra_tensor_view<T,1> Av3(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A3_subid, A3_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B3_subid = {0, 2};
    std::array<T,2> B3_coeff_list = {-1, 1};
    stra_tensor_view<T,2> Bv3(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B3_subid, B3_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C3_subid = {0, 2};
    std::array<T,2> C3_coeff_list = {1, 1};
    stra_tensor_view<T,2> Cv3(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C3_subid, C3_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv3.stride(!row_major) == 1)
    {
        Av3.transpose();
        Bv3.transpose();
        Cv3.transpose();
        straprim_ab2(comm, cfg, alpha, Bv3, Av3, beta, Cv3);
    } else {
        straprim_ab2(comm, cfg, alpha, Av3, Bv3, beta, Cv3);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M3:" << std::endl;
    //print_tensor_matrix( ct );

    // M4 = (1 * A_0 + 1 * A_1) * (1 * B_3);  C_0 += -1 * M4;  C_1 += 1 * M4;
    std::array<unsigned, 2> A4_subid = {0, 1};
    std::array<T,2> A4_coeff_list = {1, 1};
    stra_tensor_view<T,2> Av4(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A4_subid, A4_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 1> B4_subid = {3};
    std::array<T,1> B4_coeff_list = {1};
    stra_tensor_view<T,1> Bv4(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B4_subid, B4_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C4_subid = {0, 1};
    std::array<T,2> C4_coeff_list = {-1, 1};
    stra_tensor_view<T,2> Cv4(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C4_subid, C4_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv4.stride(!row_major) == 1)
    {
        Av4.transpose();
        Bv4.transpose();
        Cv4.transpose();
        straprim_ab2(comm, cfg, alpha, Bv4, Av4, beta, Cv4);
    } else {
        straprim_ab2(comm, cfg, alpha, Av4, Bv4, beta, Cv4);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M4:" << std::endl;
    //print_tensor_matrix( ct );

    // M5 = (-1 * A_0 + 1 * A_2) * (1 * B_0 + 1 * B_1);  C_3 += 1 * M5;
    std::array<unsigned, 2> A5_subid = {0, 2};
    std::array<T,2> A5_coeff_list = {-1, 1};
    stra_tensor_view<T,2> Av5(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A5_subid, A5_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B5_subid = {0, 1};
    std::array<T,2> B5_coeff_list = {1, 1};
    stra_tensor_view<T,2> Bv5(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B5_subid, B5_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 1> C5_subid = {3};
    std::array<T,1> C5_coeff_list = {1};
    stra_tensor_view<T,1> Cv5(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C5_subid, C5_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv5.stride(!row_major) == 1)
    {
        Av5.transpose();
        Bv5.transpose();
        Cv5.transpose();
        straprim_ab2(comm, cfg, alpha, Bv5, Av5, beta, Cv5);
    } else {
        straprim_ab2(comm, cfg, alpha, Av5, Bv5, beta, Cv5);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M5:" << std::endl;
    //print_tensor_matrix( ct );

    // M6 = (1 * A_1 + -1 * A_3) * (1 * B_2 + 1 * B_3);  C_0 += 1 * M6;
    std::array<unsigned, 2> A6_subid = {1, 3};
    std::array<T,2> A6_coeff_list = {1, -1};
    stra_tensor_view<T,2> Av6(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A6_subid, A6_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B6_subid = {2, 3};
    std::array<T,2> B6_coeff_list = {1, 1};
    stra_tensor_view<T,2> Bv6(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B6_subid, B6_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 1> C6_subid = {0};
    std::array<T,1> C6_coeff_list = {1};
    stra_tensor_view<T,1> Cv6(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C6_subid, C6_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv6.stride(!row_major) == 1)
    {
        Av6.transpose();
        Bv6.transpose();
        Cv6.transpose();
        straprim_ab2(comm, cfg, alpha, Bv6, Av6, beta, Cv6);
    } else {
        straprim_ab2(comm, cfg, alpha, Av6, Bv6, beta, Cv6);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M6:" << std::endl;
    //print_tensor_matrix( ct );




    //std::cout << "stra_internal/stra_mult_A:" << std::endl;
    //print_tensor_matrix( at );

    //std::cout << "stra_internal/stra_mult_B:" << std::endl;
    //print_tensor_matrix( bt );

    //std::cout << "stra_internal/stra_mult_M6:" << std::endl;
    //print_tensor_matrix( ct );

}

#define INSTANTIATE_CONTRACT_BLIS_AB(T) \
template void stra_contract_blis_ab(const communicator& comm, const config& cfg, \
                                 const std::vector<len_type>& len_AB, \
                                 const std::vector<len_type>& len_AC, \
                                 const std::vector<len_type>& len_BC, \
                                 T alpha, const T* A, \
                                 const std::vector<stride_type>& stride_A_AB, \
                                 const std::vector<stride_type>& stride_A_AC, \
                                          const T* B, \
                                 const std::vector<stride_type>& stride_B_AB, \
                                 const std::vector<stride_type>& stride_B_BC, \
                                 T  beta,       T* C, \
                                 const std::vector<stride_type>& stride_C_AC, \
                                 const std::vector<stride_type>& stride_C_BC);

INSTANTIATE_CONTRACT_BLIS_AB(float);
INSTANTIATE_CONTRACT_BLIS_AB(double);
INSTANTIATE_CONTRACT_BLIS_AB(scomplex);
INSTANTIATE_CONTRACT_BLIS_AB(dcomplex);




template <typename T>
void stra_mult_blas(const communicator& comm, const config& cfg,
                    const std::vector<len_type>& len_A,
                    const std::vector<len_type>& len_B,
                    const std::vector<len_type>& len_C,
                    const std::vector<len_type>& len_AB,
                    const std::vector<len_type>& len_AC,
                    const std::vector<len_type>& len_BC,
                    const std::vector<len_type>& len_ABC,
                    T alpha, const T* A,
                    const std::vector<stride_type>& stride_A_A,
                    const std::vector<stride_type>& stride_A_AB,
                    const std::vector<stride_type>& stride_A_AC,
                    const std::vector<stride_type>& stride_A_ABC,
                             const T* B,
                    const std::vector<stride_type>& stride_B_B,
                    const std::vector<stride_type>& stride_B_AB,
                    const std::vector<stride_type>& stride_B_BC,
                    const std::vector<stride_type>& stride_B_ABC,
                    T  beta,       T* C,
                    const std::vector<stride_type>& stride_C_C,
                    const std::vector<stride_type>& stride_C_AC,
                    const std::vector<stride_type>& stride_C_BC,
                    const std::vector<stride_type>& stride_C_ABC)
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

    MArray::viterator<3> it(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

    while (it.next(A, B, C))
    {
        add(comm, cfg, len_A, {}, arv.lengths(),
            T(1), false,          A, stride_A_A, stride_A_AC+stride_A_AB,
            T(0), false, arv.data(),         {},           arv.strides());

        add(comm, cfg, len_B, {}, brv.lengths(),
            T(1), false,          B, stride_B_B, stride_B_AB+stride_B_BC,
            T(0), false, brv.data(),         {},           brv.strides());

        stra_mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
                  alpha, false, am.data(), am.stride(0), am.stride(1),
                         false, bm.data(), bm.stride(0), bm.stride(1),
                   T(0), false, cm.data(), cm.stride(0), cm.stride(1));

        add(comm, cfg, {}, len_C, crv.lengths(),
            T(1), false, crv.data(),         {},            crv.strides(),
            beta, false,          C, stride_C_C, stride_C_AC+stride_C_BC);
    }
}

template <typename T>
void stra_mult_ref(const communicator& comm, const config& cfg,
                   const std::vector<len_type>& len_A,
                   const std::vector<len_type>& len_B,
                   const std::vector<len_type>& len_C,
                   const std::vector<len_type>& len_AB,
                   const std::vector<len_type>& len_AC,
                   const std::vector<len_type>& len_BC,
                   const std::vector<len_type>& len_ABC,
                   T alpha, const T* A,
                   const std::vector<stride_type>& stride_A_A,
                   const std::vector<stride_type>& stride_A_AB,
                   const std::vector<stride_type>& stride_A_AC,
                   const std::vector<stride_type>& stride_A_ABC,
                            const T* B,
                   const std::vector<stride_type>& stride_B_B,
                   const std::vector<stride_type>& stride_B_AB,
                   const std::vector<stride_type>& stride_B_BC,
                   const std::vector<stride_type>& stride_B_ABC,
                   T  beta,       T* C,
                   const std::vector<stride_type>& stride_C_C,
                   const std::vector<stride_type>& stride_C_AC,
                   const std::vector<stride_type>& stride_C_BC,
                   const std::vector<stride_type>& stride_C_ABC)
{
    (void)cfg;

    MArray::viterator<1> iter_A(len_A, stride_A_A);
    MArray::viterator<1> iter_B(len_B, stride_B_B);
    MArray::viterator<1> iter_C(len_C, stride_C_C);
    MArray::viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
    MArray::viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    MArray::viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);
    MArray::viterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);
    len_type n = stl_ext::prod(len_ABC);

    len_type n_min, n_max;
    std::tie(n_min, n_max, std::ignore) = comm.distribute_over_threads(n);

    iter_ABC.position(n_min, A, B, C);

    for (len_type i = n_min;i < n_max;i++)
    {
        iter_ABC.next(A, B, C);

        while (iter_AC.next(A, C))
        {
            while (iter_BC.next(B, C))
            {
                T temp = T();

                while (iter_AB.next(A, B))
                {
                    T temp_A = T();
                    while (iter_A.next(A))
                    {
                        temp_A += *A;
                    }

                    T temp_B = T();
                    while (iter_B.next(B))
                    {
                        temp_B += *B;
                    }

                    temp += temp_A*temp_B;
                }

                temp *= alpha;

                if (beta == T(0))
                {
                    while (iter_C.next(C))
                    {
                        *C = temp;
                    }
                }
                else
                {
                    while (iter_C.next(C))
                    {
                        *C = temp + beta*(*C);
                    }
                }
            }
        }
    }
}

template <typename T>
void stra_outer_prod_blas(const communicator& comm, const config& cfg,
                          const std::vector<len_type>& len_AC,
                          const std::vector<len_type>& len_BC,
                          T alpha, const T* A,
                          const std::vector<stride_type>& stride_A_AC,
                                   const T* B,
                          const std::vector<stride_type>& stride_B_BC,
                          T  beta,       T* C,
                          const std::vector<stride_type>& stride_C_AC,
                          const std::vector<stride_type>& stride_C_BC)
{
    tensor<T> ar, br, cr;
    T* ptrs_local[3];
    T** ptrs = &ptrs_local[0];

    if (comm.master())
    {
        ar.reset(len_AC);
        br.reset(len_BC);
        cr.reset(len_AC+len_BC);
        ptrs[0] = ar.data();
        ptrs[1] = br.data();
        ptrs[2] = cr.data();
    }

    comm.broadcast(ptrs);

    tensor_view<T> arv(len_AC, ptrs[0]);
    tensor_view<T> brv(len_BC, ptrs[1]);
    tensor_view<T> crv(len_AC+len_BC, ptrs[2]);

    matrix_view<T> am, bm, cm;
    matricize<T>(arv, am, static_cast<unsigned>(len_AC.size()));
    matricize<T>(brv, bm, 0);
    matricize<T>(crv, cm, static_cast<unsigned>(len_AC.size()));

    add(comm, cfg, {}, {}, arv.lengths(),
        T(1), false,          A, {},   stride_A_AC,
        T(0), false, arv.data(), {}, arv.strides());

    add(comm, cfg, {}, {}, brv.lengths(),
        T(1), false,          B, {},   stride_B_BC,
        T(0), false, brv.data(), {}, brv.strides());

    stra_mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
             alpha, false, am.data(), am.stride(0), am.stride(1),
                    false, bm.data(), bm.stride(0), bm.stride(1),
              T(0), false, cm.data(), cm.stride(0), cm.stride(1));

    add(comm, cfg, {}, {}, crv.lengths(),
        T(1), false, crv.data(), {},            crv.strides(),
        beta, false,          C, {}, stride_C_AC+stride_C_BC);
}

template <typename T>
void stra_outer_prod_ref(const communicator& comm, const config& cfg,
                         const std::vector<len_type>& len_AC,
                         const std::vector<len_type>& len_BC,
                         T alpha, const T* A,
                         const std::vector<stride_type>& stride_A_AC,
                                  const T* B,
                         const std::vector<stride_type>& stride_B_BC,
                         T  beta,       T* C,
                         const std::vector<stride_type>& stride_C_AC,
                         const std::vector<stride_type>& stride_C_BC)
{
    (void)cfg;

    MArray::viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    MArray::viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);
    len_type m = stl_ext::prod(len_AC);
    len_type n = stl_ext::prod(len_BC);

    len_type m_min, m_max, n_min, n_max;
    std::tie(m_min, m_max, std::ignore,
             n_min, n_max, std::ignore) = comm.distribute_over_threads_2d(m, n);

    const T* A0 = A;
    const T* B0 = B;
          T* C0 = C;

    iter_AC.position(m_min, A0, C0);

    for (len_type i = m_min;i < m_max;i++)
    {
        iter_AC.next(A0, C0);

        A = A0;
        B = B0;
        C = C0;

        iter_BC.position(n_min, B, C);

        if (beta == T(0))
        {
            for (len_type j = n_min;j < n_max;j++)
            {
                iter_BC.next(B, C);
                *C = alpha*(*A)*(*B);
            }
        }
        else
        {
            for (len_type j = n_min;j < n_max;j++)
            {
                iter_BC.next(B, C);
                *C = alpha*(*A)*(*B) + beta*(*C);
            }
        }
    }
}

template <typename T>
void stra_weight_blas(const communicator& comm, const config& cfg,
                      const std::vector<len_type>& len_AC,
                      const std::vector<len_type>& len_BC,
                      const std::vector<len_type>& len_ABC,
                      T alpha, const T* A,
                      const std::vector<stride_type>& stride_A_AC,
                      const std::vector<stride_type>& stride_A_ABC,
                               const T* B,
                      const std::vector<stride_type>& stride_B_BC,
                      const std::vector<stride_type>& stride_B_ABC,
                      T  beta,       T* C,
                      const std::vector<stride_type>& stride_C_AC,
                      const std::vector<stride_type>& stride_C_BC,
                      const std::vector<stride_type>& stride_C_ABC)
{
    tensor<T> ar, br, cr;
    T* ptrs_local[3];
    T** ptrs = &ptrs_local[0];

    if (comm.master())
    {
        ar.reset(len_AC);
        br.reset(len_BC);
        cr.reset(len_AC+len_BC);
        ptrs[0] = ar.data();
        ptrs[1] = br.data();
        ptrs[2] = cr.data();
    }

    comm.broadcast(ptrs);

    tensor_view<T> arv(len_AC, ptrs[0]);
    tensor_view<T> brv(len_BC, ptrs[1]);
    tensor_view<T> crv(len_AC+len_BC, ptrs[2]);

    matrix_view<T> am, bm, cm;
    matricize<T>(arv, am, static_cast<unsigned>(len_AC.size()));
    matricize<T>(brv, bm, 0);
    matricize<T>(crv, cm, static_cast<unsigned>(len_AC.size()));

    MArray::viterator<3> it(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

    while (it.next(A, B, C))
    {
        add(comm, cfg, {}, {}, arv.lengths(),
            T(1), false,          A, {},   stride_A_AC,
            T(0), false, arv.data(), {}, arv.strides());

        add(comm, cfg, {}, {}, brv.lengths(),
            T(1), false,          B, {},   stride_B_BC,
            T(0), false, brv.data(), {}, brv.strides());

        stra_mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
                  alpha, false, am.data(), am.stride(0), am.stride(1),
                         false, bm.data(), bm.stride(0), bm.stride(1),
                   T(0), false, cm.data(), cm.stride(0), cm.stride(1));

        add(comm, cfg, {}, {}, crv.lengths(),
            T(1), false, crv.data(), {},            crv.strides(),
            beta, false,          C, {}, stride_C_AC+stride_C_BC);
    }
}

template <typename T>
void stra_weight_ref(const communicator& comm, const config& cfg,
                     const std::vector<len_type>& len_AC,
                     const std::vector<len_type>& len_BC,
                     const std::vector<len_type>& len_ABC,
                     T alpha, const T* A,
                     const std::vector<stride_type>& stride_A_AC,
                     const std::vector<stride_type>& stride_A_ABC,
                              const T* B,
                     const std::vector<stride_type>& stride_B_BC,
                     const std::vector<stride_type>& stride_B_ABC,
                     T  beta,       T* C,
                     const std::vector<stride_type>& stride_C_AC,
                     const std::vector<stride_type>& stride_C_BC,
                     const std::vector<stride_type>& stride_C_ABC)
{
    (void)cfg;

    MArray::viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    MArray::viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);
    MArray::viterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);
    len_type n = stl_ext::prod(len_ABC);

    len_type n_min, n_max;
    std::tie(n_min, n_max, std::ignore) = comm.distribute_over_threads(n);

    iter_ABC.position(n_min, A, B, C);

    for (len_type i = n_min;i < n_max;i++)
    {
        iter_ABC.next(A, B, C);

        while (iter_AC.next(A, C))
        {
            if (beta == T(0))
            {
                while (iter_BC.next(B, C))
                {
                    *C = alpha*(*A)*(*B);
                }
            }
            else
            {
                while (iter_BC.next(B, C))
                {
                    *C = alpha*(*A)*(*B) + beta*(*C);
                }
            }
        }
    }
}

template <typename T>
void stra_mult(const communicator& comm, const config& cfg,
               const std::vector<len_type>& len_A,
               const std::vector<len_type>& len_B,
               const std::vector<len_type>& len_C,
               const std::vector<len_type>& len_AB,
               const std::vector<len_type>& len_AC,
               const std::vector<len_type>& len_BC,
               const std::vector<len_type>& len_ABC,
               T alpha, bool conj_A, const T* A,
               const std::vector<stride_type>& stride_A_A,
               const std::vector<stride_type>& stride_A_AB,
               const std::vector<stride_type>& stride_A_AC,
               const std::vector<stride_type>& stride_A_ABC,
                        bool conj_B, const T* B,
               const std::vector<stride_type>& stride_B_B,
               const std::vector<stride_type>& stride_B_AB,
               const std::vector<stride_type>& stride_B_BC,
               const std::vector<stride_type>& stride_B_ABC,
               T  beta, bool conj_C,       T* C,
               const std::vector<stride_type>& stride_C_C,
               const std::vector<stride_type>& stride_C_AC,
               const std::vector<stride_type>& stride_C_BC,
               const std::vector<stride_type>& stride_C_ABC)
{
    TBLIS_ASSERT(!conj_A && !conj_B && !conj_C);

    if (len_A.empty() && len_B.empty() && len_C.empty() &&
        (len_AB.empty() || len_ABC.empty()))
    {
        if (len_AB.empty())
        {
            if (len_ABC.empty())
            {
                if (impl == REFERENCE)
                {
                    //std::cout << "stra_outer_prod_ref" << std::endl;
                    stra_outer_prod_ref(comm, cfg, len_AC, len_BC,
                                        alpha, A, stride_A_AC,
                                               B, stride_B_BC,
                                         beta, C, stride_C_AC, stride_C_BC);
                }
                else
                {
                    //std::cout << "stra_outer_prod_blas" << std::endl;
                    stra_outer_prod_blas(comm, cfg, len_AC, len_BC,
                                         alpha, A, stride_A_AC,
                                                B, stride_B_BC,
                                          beta, C, stride_C_AC, stride_C_BC);
                }
            }
            else
            {
                if (impl == REFERENCE)
                {
                    //std::cout << "stra_weight_ref" << std::endl;
                    stra_weight_ref(comm, cfg, len_AC, len_BC, len_ABC,
                                   alpha, A, stride_A_AC, stride_A_ABC,
                                          B, stride_B_BC, stride_B_ABC,
                                    beta, C, stride_C_AC, stride_C_BC, stride_C_ABC);
                }
                else
                {
                    //std::cout << "stra_weight_blas" << std::endl;
                    stra_weight_blas(comm, cfg, len_AC, len_BC, len_ABC,
                                    alpha, A, stride_A_AC, stride_A_ABC,
                                           B, stride_B_BC, stride_B_ABC,
                                     beta, C, stride_C_AC, stride_C_BC, stride_C_ABC);
                }
            }
        }
        else
        {
            if (impl == REFERENCE)
            {
                //std::cout << "stra_contract_ref" << std::endl;
                stra_contract_ref(comm, cfg, len_AB, len_AC, len_BC,
                                 alpha, A, stride_A_AB, stride_A_AC,
                                        B, stride_B_AB, stride_B_BC,
                                  beta, C, stride_C_AC, stride_C_BC);
            }
            else if (impl == BLAS_BASED)
            {
                //std::cout << "stra_contract_blas" << std::endl;
                stra_contract_blas(comm, cfg, len_AB, len_AC, len_BC,
                                  alpha, A, stride_A_AB, stride_A_AC,
                                         B, stride_B_AB, stride_B_BC,
                                   beta, C, stride_C_AC, stride_C_BC);
            }
            else if (impl == BLIS_BASED)
            {
                //std::cout << "stra_contract_blis" << std::endl;
                stra_contract_blis(comm, cfg, len_AB, len_AC, len_BC,
                                  alpha, A, stride_A_AB, stride_A_AC,
                                         B, stride_B_AB, stride_B_BC,
                                   beta, C, stride_C_AC, stride_C_BC);
            }
            else if (impl == STRA_NAIVE)
            {
                //std::cout << "stra_contract_blis_naive" << std::endl;
                stra_contract_blis_naive(comm, cfg, len_AB, len_AC, len_BC,
                                  alpha, A, stride_A_AB, stride_A_AC,
                                         B, stride_B_AB, stride_B_BC,
                                   beta, C, stride_C_AC, stride_C_BC);
            }
            else // impl == STRA_AB
            {
                //std::cout << "stra_contract_blis_ab" << std::endl;
                stra_contract_blis_ab(comm, cfg, len_AB, len_AC, len_BC,
                                  alpha, A, stride_A_AB, stride_A_AC,
                                         B, stride_B_AB, stride_B_BC,
                                   beta, C, stride_C_AC, stride_C_BC);
            }

        }
    }
    else
    {
        if (impl == REFERENCE)
        {
            //std::cout << "stra_mult_ref" << std::endl;
            stra_mult_ref(comm, cfg, len_A, len_B, len_C,
                        len_AB, len_AC, len_BC, len_ABC,
                        alpha, A, stride_A_A, stride_A_AB,
                                  stride_A_AC, stride_A_ABC,
                               B, stride_B_B, stride_B_AB,
                                  stride_B_BC, stride_B_ABC,
                         beta, C, stride_C_C, stride_C_AC,
                                  stride_C_BC, stride_C_ABC);
        }
        else
        {
            //std::cout << "stra_mult_blas" << std::endl;
            stra_mult_blas(comm, cfg, len_A, len_B, len_C,
                         len_AB, len_AC, len_BC, len_ABC,
                         alpha, A, stride_A_A, stride_A_AB,
                                   stride_A_AC, stride_A_ABC,
                                B, stride_B_B, stride_B_AB,
                                   stride_B_BC, stride_B_ABC,
                          beta, C, stride_C_C, stride_C_AC,
                                   stride_C_BC, stride_C_ABC);
        }
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void stra_mult(const communicator& comm, const config& cfg, \
                        const std::vector<len_type>& len_A, \
                        const std::vector<len_type>& len_B, \
                        const std::vector<len_type>& len_C, \
                        const std::vector<len_type>& len_AB, \
                        const std::vector<len_type>& len_AC, \
                        const std::vector<len_type>& len_BC, \
                        const std::vector<len_type>& len_ABC, \
                        T alpha, bool conj_A, const T* A, \
                        const std::vector<stride_type>& stride_A_A, \
                        const std::vector<stride_type>& stride_A_AB, \
                        const std::vector<stride_type>& stride_A_AC, \
                        const std::vector<stride_type>& stride_A_ABC, \
                                 bool conj_B, const T* B, \
                        const std::vector<stride_type>& stride_B_B, \
                        const std::vector<stride_type>& stride_B_AB, \
                        const std::vector<stride_type>& stride_B_BC, \
                        const std::vector<stride_type>& stride_B_ABC, \
                        T  beta, bool conj_C,       T* C, \
                        const std::vector<stride_type>& stride_C_C, \
                        const std::vector<stride_type>& stride_C_AC, \
                        const std::vector<stride_type>& stride_C_BC, \
                        const std::vector<stride_type>& stride_C_ABC);
#include "configs/foreach_type.h"



}
}
