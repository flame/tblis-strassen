#include "stra_mult_2level.hpp"

#include "util/gemm_thread.hpp"
#include "util/tensor.hpp"

#include "matrix/stra_tensor_view.hpp"

#include "stra_nodes/stra_matrify.hpp"
#include "stra_nodes/stra_partm.hpp"
#include "stra_nodes/stra_gemm_ukr.hpp"

#include "internal/1t/add.hpp"
#include "stra_internal/3m/stra_mult_2level.hpp"

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
void stra_contract_2level_blas(const communicator& comm, const config& cfg,
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

    add(comm, cfg, {}, {}, arv.lengths(),
        T(1), false,          A, {}, stride_A_AC+stride_A_AB,
        T(0), false, arv.data(), {},           arv.strides());

    add(comm, cfg, {}, {}, brv.lengths(),
        T(1), false,          B, {}, stride_B_AB+stride_B_BC,
        T(0), false, brv.data(), {},           brv.strides());

    stra_mult_2level(comm, cfg, cm.length(0), cm.length(1), am.length(1),
              alpha, false, am.data(), am.stride(0), am.stride(1),
                     false, bm.data(), bm.stride(0), bm.stride(1),
               T(0), false, cm.data(), cm.stride(0), cm.stride(1));

    add(comm, cfg, {}, {}, crv.lengths(),
        T(1), false, crv.data(), {},            crv.strides(),
        beta, false,          C, {}, stride_C_AC+stride_C_BC);
}

template <typename T>
void stra_contract_2level_ref(const communicator& comm, const config& cfg,
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

template <typename T>
void stra_divide_vector(
          std::vector<T>& vec,
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
void stra_contract_2level_blis(const communicator& comm, const config& cfg,
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
    //std::cout << "Enter stra_internal/3t/stra_mult_2level/stra_contract_2level_blis\n" << std::endl;

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
;
    auto my_stride_B_AB = stl_ext::permuted(stride_B_AB, reorder_AB);
    auto my_stride_B_BC = stl_ext::permuted(stride_B_BC, reorder_BC);
;
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

    //const len_type ms=m/2, ks=k/2, ns=n/2;

    const std::array<unsigned,2> A_divisor={4,4};
    const std::array<unsigned,2> B_divisor={4,4};
    const std::array<unsigned,2> C_divisor={4,4};

    //stra_gemm(comm, cfg, alpha, at, bt, beta, ct);

    // M0 = (1.0 * A_0_0 + 1.0 * A_0_3 + 1.0 * A_3_0 + 1.0 * A_3_3) * (1.0 * B_0_0 + 1.0 * B_0_3 + 1.0 * B_3_0 + 1.0 * B_3_3);  C_0_0 += 1.0 * M0;  C_0_3 += 1.0 * M0;  C_3_0 += 1.0 * M0;  C_3_3 += 1.0 * M0;
    std::array<unsigned, 4> A0_subid = {0, 5, 10, 15};
    std::array<T,4> A0_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av0(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A0_subid, A0_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B0_subid = {0, 5, 10, 15};
    std::array<T,4> B0_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv0(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B0_subid, B0_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C0_subid = {0, 5, 10, 15};
    std::array<T,4> C0_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv0(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C0_subid, C0_coeff_list, my_stride_C_AC, my_stride_C_BC);
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
    //std::cout << "stra_internal/stra_mult_M0:" << std::endl;
    //print_tensor_matrix( ct );

    // M1 = (1.0 * A_0_2 + 1.0 * A_0_3 + 1.0 * A_3_2 + 1.0 * A_3_3) * (1.0 * B_0_0 + 1.0 * B_3_0);  C_0_2 += 1.0 * M1;  C_0_3 += -1.0 * M1;  C_3_2 += 1.0 * M1;  C_3_3 += -1.0 * M1;
    std::array<unsigned, 4> A1_subid = {4, 5, 14, 15};
    std::array<T,4> A1_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av1(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A1_subid, A1_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B1_subid = {0, 10};
    std::array<T,2> B1_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv1(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B1_subid, B1_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C1_subid = {4, 5, 14, 15};
    std::array<T,4> C1_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Cv1(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C1_subid, C1_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv1.stride(!row_major) == 1)
    {
        Av1.transpose();
        Bv1.transpose();
        Cv1.transpose();
        stra_gemm(comm, cfg, alpha, Bv1, Av1, beta, Cv1);
    } else {
        stra_gemm(comm, cfg, alpha, Av1, Bv1, beta, Cv1);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M1:" << std::endl;
    //print_tensor_matrix( ct );

    // M2 = (1.0 * A_0_0 + 1.0 * A_3_0) * (1.0 * B_0_1 + -1.0 * B_0_3 + 1.0 * B_3_1 + -1.0 * B_3_3);  C_0_1 += 1.0 * M2;  C_0_3 += 1.0 * M2;  C_3_1 += 1.0 * M2;  C_3_3 += 1.0 * M2;
    std::array<unsigned, 2> A2_subid = {0, 10};
    std::array<T,2> A2_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av2(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A2_subid, A2_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B2_subid = {1, 5, 11, 15};
    std::array<T,4> B2_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Bv2(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B2_subid, B2_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C2_subid = {1, 5, 11, 15};
    std::array<T,4> C2_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv2(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C2_subid, C2_coeff_list, my_stride_C_AC, my_stride_C_BC);
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

    // M3 = (1.0 * A_0_3 + 1.0 * A_3_3) * (-1.0 * B_0_0 + 1.0 * B_0_2 + -1.0 * B_3_0 + 1.0 * B_3_2);  C_0_0 += 1.0 * M3;  C_0_2 += 1.0 * M3;  C_3_0 += 1.0 * M3;  C_3_2 += 1.0 * M3;
    std::array<unsigned, 2> A3_subid = {5, 15};
    std::array<T,2> A3_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av3(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A3_subid, A3_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B3_subid = {0, 4, 10, 14};
    std::array<T,4> B3_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Bv3(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B3_subid, B3_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C3_subid = {0, 4, 10, 14};
    std::array<T,4> C3_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv3(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C3_subid, C3_coeff_list, my_stride_C_AC, my_stride_C_BC);
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

    // M4 = (1.0 * A_0_0 + 1.0 * A_0_1 + 1.0 * A_3_0 + 1.0 * A_3_1) * (1.0 * B_0_3 + 1.0 * B_3_3);  C_0_0 += -1.0 * M4;  C_0_1 += 1.0 * M4;  C_3_0 += -1.0 * M4;  C_3_1 += 1.0 * M4;
    std::array<unsigned, 4> A4_subid = {0, 1, 10, 11};
    std::array<T,4> A4_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av4(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A4_subid, A4_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B4_subid = {5, 15};
    std::array<T,2> B4_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv4(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B4_subid, B4_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C4_subid = {0, 1, 10, 11};
    std::array<T,4> C4_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Cv4(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C4_subid, C4_coeff_list, my_stride_C_AC, my_stride_C_BC);
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

    // M5 = (-1.0 * A_0_0 + 1.0 * A_0_2 + -1.0 * A_3_0 + 1.0 * A_3_2) * (1.0 * B_0_0 + 1.0 * B_0_1 + 1.0 * B_3_0 + 1.0 * B_3_1);  C_0_3 += 1.0 * M5;  C_3_3 += 1.0 * M5;
    std::array<unsigned, 4> A5_subid = {0, 4, 10, 14};
    std::array<T,4> A5_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Av5(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A5_subid, A5_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B5_subid = {0, 1, 10, 11};
    std::array<T,4> B5_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv5(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B5_subid, B5_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C5_subid = {5, 15};
    std::array<T,2> C5_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv5(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C5_subid, C5_coeff_list, my_stride_C_AC, my_stride_C_BC);
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

    // M6 = (1.0 * A_0_1 + -1.0 * A_0_3 + 1.0 * A_3_1 + -1.0 * A_3_3) * (1.0 * B_0_2 + 1.0 * B_0_3 + 1.0 * B_3_2 + 1.0 * B_3_3);  C_0_0 += 1.0 * M6;  C_3_0 += 1.0 * M6;
    std::array<unsigned, 4> A6_subid = {1, 5, 11, 15};
    std::array<T,4> A6_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Av6(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A6_subid, A6_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B6_subid = {4, 5, 14, 15};
    std::array<T,4> B6_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv6(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B6_subid, B6_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C6_subid = {0, 10};
    std::array<T,2> C6_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv6(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C6_subid, C6_coeff_list, my_stride_C_AC, my_stride_C_BC);
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

    // M7 = (1.0 * A_2_0 + 1.0 * A_2_3 + 1.0 * A_3_0 + 1.0 * A_3_3) * (1.0 * B_0_0 + 1.0 * B_0_3);  C_2_0 += 1.0 * M7;  C_2_3 += 1.0 * M7;  C_3_0 += -1.0 * M7;  C_3_3 += -1.0 * M7;
    std::array<unsigned, 4> A7_subid = {8, 13, 10, 15};
    std::array<T,4> A7_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av7(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A7_subid, A7_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B7_subid = {0, 5};
    std::array<T,2> B7_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv7(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B7_subid, B7_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C7_subid = {8, 13, 10, 15};
    std::array<T,4> C7_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Cv7(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C7_subid, C7_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv7.stride(!row_major) == 1)
    {
        Av7.transpose();
        Bv7.transpose();
        Cv7.transpose();
        stra_gemm(comm, cfg, alpha, Bv7, Av7, beta, Cv7);
    } else {
        stra_gemm(comm, cfg, alpha, Av7, Bv7, beta, Cv7);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M7:" << std::endl;
    //print_tensor_matrix( ct );

    // M8 = (1.0 * A_2_2 + 1.0 * A_2_3 + 1.0 * A_3_2 + 1.0 * A_3_3) * (1.0 * B_0_0);  C_2_2 += 1.0 * M8;  C_2_3 += -1.0 * M8;  C_3_2 += -1.0 * M8;  C_3_3 += 1.0 * M8;
    std::array<unsigned, 4> A8_subid = {12, 13, 14, 15};
    std::array<T,4> A8_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av8(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A8_subid, A8_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 1> B8_subid = {0};
    std::array<T,1> B8_coeff_list = {1.0};
    stra_tensor_view<T,1> Bv8(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B8_subid, B8_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C8_subid = {12, 13, 14, 15};
    std::array<T,4> C8_coeff_list = {1.0, -1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Cv8(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C8_subid, C8_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv8.stride(!row_major) == 1)
    {
        Av8.transpose();
        Bv8.transpose();
        Cv8.transpose();
        stra_gemm(comm, cfg, alpha, Bv8, Av8, beta, Cv8);
    } else {
        stra_gemm(comm, cfg, alpha, Av8, Bv8, beta, Cv8);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M8:" << std::endl;
    //print_tensor_matrix( ct );

    // M9 = (1.0 * A_2_0 + 1.0 * A_3_0) * (1.0 * B_0_1 + -1.0 * B_0_3);  C_2_1 += 1.0 * M9;  C_2_3 += 1.0 * M9;  C_3_1 += -1.0 * M9;  C_3_3 += -1.0 * M9;
    std::array<unsigned, 2> A9_subid = {8, 10};
    std::array<T,2> A9_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av9(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A9_subid, A9_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B9_subid = {1, 5};
    std::array<T,2> B9_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Bv9(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B9_subid, B9_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C9_subid = {9, 13, 11, 15};
    std::array<T,4> C9_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Cv9(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C9_subid, C9_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv9.stride(!row_major) == 1)
    {
        Av9.transpose();
        Bv9.transpose();
        Cv9.transpose();
        stra_gemm(comm, cfg, alpha, Bv9, Av9, beta, Cv9);
    } else {
        stra_gemm(comm, cfg, alpha, Av9, Bv9, beta, Cv9);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M9:" << std::endl;
    //print_tensor_matrix( ct );

    // M10 = (1.0 * A_2_3 + 1.0 * A_3_3) * (-1.0 * B_0_0 + 1.0 * B_0_2);  C_2_0 += 1.0 * M10;  C_2_2 += 1.0 * M10;  C_3_0 += -1.0 * M10;  C_3_2 += -1.0 * M10;
    std::array<unsigned, 2> A10_subid = {13, 15};
    std::array<T,2> A10_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av10(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A10_subid, A10_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B10_subid = {0, 4};
    std::array<T,2> B10_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Bv10(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B10_subid, B10_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C10_subid = {8, 12, 10, 14};
    std::array<T,4> C10_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Cv10(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C10_subid, C10_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv10.stride(!row_major) == 1)
    {
        Av10.transpose();
        Bv10.transpose();
        Cv10.transpose();
        stra_gemm(comm, cfg, alpha, Bv10, Av10, beta, Cv10);
    } else {
        stra_gemm(comm, cfg, alpha, Av10, Bv10, beta, Cv10);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M10:" << std::endl;
    //print_tensor_matrix( ct );

    // M11 = (1.0 * A_2_0 + 1.0 * A_2_1 + 1.0 * A_3_0 + 1.0 * A_3_1) * (1.0 * B_0_3);  C_2_0 += -1.0 * M11;  C_2_1 += 1.0 * M11;  C_3_0 += 1.0 * M11;  C_3_1 += -1.0 * M11;
    std::array<unsigned, 4> A11_subid = {8, 9, 10, 11};
    std::array<T,4> A11_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av11(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A11_subid, A11_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 1> B11_subid = {5};
    std::array<T,1> B11_coeff_list = {1.0};
    stra_tensor_view<T,1> Bv11(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B11_subid, B11_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C11_subid = {8, 9, 10, 11};
    std::array<T,4> C11_coeff_list = {-1.0, 1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Cv11(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C11_subid, C11_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv11.stride(!row_major) == 1)
    {
        Av11.transpose();
        Bv11.transpose();
        Cv11.transpose();
        stra_gemm(comm, cfg, alpha, Bv11, Av11, beta, Cv11);
    } else {
        stra_gemm(comm, cfg, alpha, Av11, Bv11, beta, Cv11);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M11:" << std::endl;
    //print_tensor_matrix( ct );

    // M12 = (-1.0 * A_2_0 + 1.0 * A_2_2 + -1.0 * A_3_0 + 1.0 * A_3_2) * (1.0 * B_0_0 + 1.0 * B_0_1);  C_2_3 += 1.0 * M12;  C_3_3 += -1.0 * M12;
    std::array<unsigned, 4> A12_subid = {8, 12, 10, 14};
    std::array<T,4> A12_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Av12(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A12_subid, A12_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B12_subid = {0, 1};
    std::array<T,2> B12_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv12(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B12_subid, B12_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C12_subid = {13, 15};
    std::array<T,2> C12_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Cv12(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C12_subid, C12_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv12.stride(!row_major) == 1)
    {
        Av12.transpose();
        Bv12.transpose();
        Cv12.transpose();
        stra_gemm(comm, cfg, alpha, Bv12, Av12, beta, Cv12);
    } else {
        stra_gemm(comm, cfg, alpha, Av12, Bv12, beta, Cv12);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M12:" << std::endl;
    //print_tensor_matrix( ct );

    // M13 = (1.0 * A_2_1 + -1.0 * A_2_3 + 1.0 * A_3_1 + -1.0 * A_3_3) * (1.0 * B_0_2 + 1.0 * B_0_3);  C_2_0 += 1.0 * M13;  C_3_0 += -1.0 * M13;
    std::array<unsigned, 4> A13_subid = {9, 13, 11, 15};
    std::array<T,4> A13_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Av13(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A13_subid, A13_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B13_subid = {4, 5};
    std::array<T,2> B13_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv13(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B13_subid, B13_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C13_subid = {8, 10};
    std::array<T,2> C13_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Cv13(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C13_subid, C13_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv13.stride(!row_major) == 1)
    {
        Av13.transpose();
        Bv13.transpose();
        Cv13.transpose();
        stra_gemm(comm, cfg, alpha, Bv13, Av13, beta, Cv13);
    } else {
        stra_gemm(comm, cfg, alpha, Av13, Bv13, beta, Cv13);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M13:" << std::endl;
    //print_tensor_matrix( ct );

    // M14 = (1.0 * A_0_0 + 1.0 * A_0_3) * (1.0 * B_1_0 + 1.0 * B_1_3 + -1.0 * B_3_0 + -1.0 * B_3_3);  C_1_0 += 1.0 * M14;  C_1_3 += 1.0 * M14;  C_3_0 += 1.0 * M14;  C_3_3 += 1.0 * M14;
    std::array<unsigned, 2> A14_subid = {0, 5};
    std::array<T,2> A14_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av14(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A14_subid, A14_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B14_subid = {2, 7, 10, 15};
    std::array<T,4> B14_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Bv14(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B14_subid, B14_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C14_subid = {2, 7, 10, 15};
    std::array<T,4> C14_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv14(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C14_subid, C14_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv14.stride(!row_major) == 1)
    {
        Av14.transpose();
        Bv14.transpose();
        Cv14.transpose();
        stra_gemm(comm, cfg, alpha, Bv14, Av14, beta, Cv14);
    } else {
        stra_gemm(comm, cfg, alpha, Av14, Bv14, beta, Cv14);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M14:" << std::endl;
    //print_tensor_matrix( ct );

    // M15 = (1.0 * A_0_2 + 1.0 * A_0_3) * (1.0 * B_1_0 + -1.0 * B_3_0);  C_1_2 += 1.0 * M15;  C_1_3 += -1.0 * M15;  C_3_2 += 1.0 * M15;  C_3_3 += -1.0 * M15;
    std::array<unsigned, 2> A15_subid = {4, 5};
    std::array<T,2> A15_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av15(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A15_subid, A15_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B15_subid = {2, 10};
    std::array<T,2> B15_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Bv15(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B15_subid, B15_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C15_subid = {6, 7, 14, 15};
    std::array<T,4> C15_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Cv15(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C15_subid, C15_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv15.stride(!row_major) == 1)
    {
        Av15.transpose();
        Bv15.transpose();
        Cv15.transpose();
        stra_gemm(comm, cfg, alpha, Bv15, Av15, beta, Cv15);
    } else {
        stra_gemm(comm, cfg, alpha, Av15, Bv15, beta, Cv15);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M15:" << std::endl;
    //print_tensor_matrix( ct );

    // M16 = (1.0 * A_0_0) * (1.0 * B_1_1 + -1.0 * B_1_3 + -1.0 * B_3_1 + 1.0 * B_3_3);  C_1_1 += 1.0 * M16;  C_1_3 += 1.0 * M16;  C_3_1 += 1.0 * M16;  C_3_3 += 1.0 * M16;
    std::array<unsigned, 1> A16_subid = {0};
    std::array<T,1> A16_coeff_list = {1.0};
    stra_tensor_view<T,1> Av16(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A16_subid, A16_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B16_subid = {3, 7, 11, 15};
    std::array<T,4> B16_coeff_list = {1.0, -1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Bv16(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B16_subid, B16_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C16_subid = {3, 7, 11, 15};
    std::array<T,4> C16_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv16(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C16_subid, C16_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv16.stride(!row_major) == 1)
    {
        Av16.transpose();
        Bv16.transpose();
        Cv16.transpose();
        stra_gemm(comm, cfg, alpha, Bv16, Av16, beta, Cv16);
    } else {
        stra_gemm(comm, cfg, alpha, Av16, Bv16, beta, Cv16);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M16:" << std::endl;
    //print_tensor_matrix( ct );

    // M17 = (1.0 * A_0_3) * (-1.0 * B_1_0 + 1.0 * B_1_2 + 1.0 * B_3_0 + -1.0 * B_3_2);  C_1_0 += 1.0 * M17;  C_1_2 += 1.0 * M17;  C_3_0 += 1.0 * M17;  C_3_2 += 1.0 * M17;
    std::array<unsigned, 1> A17_subid = {5};
    std::array<T,1> A17_coeff_list = {1.0};
    stra_tensor_view<T,1> Av17(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A17_subid, A17_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B17_subid = {2, 6, 10, 14};
    std::array<T,4> B17_coeff_list = {-1.0, 1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Bv17(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B17_subid, B17_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C17_subid = {2, 6, 10, 14};
    std::array<T,4> C17_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv17(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C17_subid, C17_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv17.stride(!row_major) == 1)
    {
        Av17.transpose();
        Bv17.transpose();
        Cv17.transpose();
        stra_gemm(comm, cfg, alpha, Bv17, Av17, beta, Cv17);
    } else {
        stra_gemm(comm, cfg, alpha, Av17, Bv17, beta, Cv17);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M17:" << std::endl;
    //print_tensor_matrix( ct );

    // M18 = (1.0 * A_0_0 + 1.0 * A_0_1) * (1.0 * B_1_3 + -1.0 * B_3_3);  C_1_0 += -1.0 * M18;  C_1_1 += 1.0 * M18;  C_3_0 += -1.0 * M18;  C_3_1 += 1.0 * M18;
    std::array<unsigned, 2> A18_subid = {0, 1};
    std::array<T,2> A18_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av18(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A18_subid, A18_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B18_subid = {7, 15};
    std::array<T,2> B18_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Bv18(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B18_subid, B18_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C18_subid = {2, 3, 10, 11};
    std::array<T,4> C18_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Cv18(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C18_subid, C18_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv18.stride(!row_major) == 1)
    {
        Av18.transpose();
        Bv18.transpose();
        Cv18.transpose();
        stra_gemm(comm, cfg, alpha, Bv18, Av18, beta, Cv18);
    } else {
        stra_gemm(comm, cfg, alpha, Av18, Bv18, beta, Cv18);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M18:" << std::endl;
    //print_tensor_matrix( ct );

    // M19 = (-1.0 * A_0_0 + 1.0 * A_0_2) * (1.0 * B_1_0 + 1.0 * B_1_1 + -1.0 * B_3_0 + -1.0 * B_3_1);  C_1_3 += 1.0 * M19;  C_3_3 += 1.0 * M19;
    std::array<unsigned, 2> A19_subid = {0, 4};
    std::array<T,2> A19_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Av19(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A19_subid, A19_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B19_subid = {2, 3, 10, 11};
    std::array<T,4> B19_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Bv19(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B19_subid, B19_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C19_subid = {7, 15};
    std::array<T,2> C19_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv19(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C19_subid, C19_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv19.stride(!row_major) == 1)
    {
        Av19.transpose();
        Bv19.transpose();
        Cv19.transpose();
        stra_gemm(comm, cfg, alpha, Bv19, Av19, beta, Cv19);
    } else {
        stra_gemm(comm, cfg, alpha, Av19, Bv19, beta, Cv19);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M19:" << std::endl;
    //print_tensor_matrix( ct );

    // M20 = (1.0 * A_0_1 + -1.0 * A_0_3) * (1.0 * B_1_2 + 1.0 * B_1_3 + -1.0 * B_3_2 + -1.0 * B_3_3);  C_1_0 += 1.0 * M20;  C_3_0 += 1.0 * M20;
    std::array<unsigned, 2> A20_subid = {1, 5};
    std::array<T,2> A20_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Av20(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A20_subid, A20_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B20_subid = {6, 7, 14, 15};
    std::array<T,4> B20_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Bv20(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B20_subid, B20_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C20_subid = {2, 10};
    std::array<T,2> C20_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv20(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C20_subid, C20_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv20.stride(!row_major) == 1)
    {
        Av20.transpose();
        Bv20.transpose();
        Cv20.transpose();
        stra_gemm(comm, cfg, alpha, Bv20, Av20, beta, Cv20);
    } else {
        stra_gemm(comm, cfg, alpha, Av20, Bv20, beta, Cv20);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M20:" << std::endl;
    //print_tensor_matrix( ct );

    // M21 = (1.0 * A_3_0 + 1.0 * A_3_3) * (-1.0 * B_0_0 + -1.0 * B_0_3 + 1.0 * B_2_0 + 1.0 * B_2_3);  C_0_0 += 1.0 * M21;  C_0_3 += 1.0 * M21;  C_2_0 += 1.0 * M21;  C_2_3 += 1.0 * M21;
    std::array<unsigned, 2> A21_subid = {10, 15};
    std::array<T,2> A21_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av21(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A21_subid, A21_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B21_subid = {0, 5, 8, 13};
    std::array<T,4> B21_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv21(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B21_subid, B21_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C21_subid = {0, 5, 8, 13};
    std::array<T,4> C21_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv21(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C21_subid, C21_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv21.stride(!row_major) == 1)
    {
        Av21.transpose();
        Bv21.transpose();
        Cv21.transpose();
        stra_gemm(comm, cfg, alpha, Bv21, Av21, beta, Cv21);
    } else {
        stra_gemm(comm, cfg, alpha, Av21, Bv21, beta, Cv21);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M21:" << std::endl;
    //print_tensor_matrix( ct );

    // M22 = (1.0 * A_3_2 + 1.0 * A_3_3) * (-1.0 * B_0_0 + 1.0 * B_2_0);  C_0_2 += 1.0 * M22;  C_0_3 += -1.0 * M22;  C_2_2 += 1.0 * M22;  C_2_3 += -1.0 * M22;
    std::array<unsigned, 2> A22_subid = {14, 15};
    std::array<T,2> A22_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av22(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A22_subid, A22_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B22_subid = {0, 8};
    std::array<T,2> B22_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Bv22(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B22_subid, B22_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C22_subid = {4, 5, 12, 13};
    std::array<T,4> C22_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Cv22(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C22_subid, C22_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv22.stride(!row_major) == 1)
    {
        Av22.transpose();
        Bv22.transpose();
        Cv22.transpose();
        stra_gemm(comm, cfg, alpha, Bv22, Av22, beta, Cv22);
    } else {
        stra_gemm(comm, cfg, alpha, Av22, Bv22, beta, Cv22);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M22:" << std::endl;
    //print_tensor_matrix( ct );

    // M23 = (1.0 * A_3_0) * (-1.0 * B_0_1 + 1.0 * B_0_3 + 1.0 * B_2_1 + -1.0 * B_2_3);  C_0_1 += 1.0 * M23;  C_0_3 += 1.0 * M23;  C_2_1 += 1.0 * M23;  C_2_3 += 1.0 * M23;
    std::array<unsigned, 1> A23_subid = {10};
    std::array<T,1> A23_coeff_list = {1.0};
    stra_tensor_view<T,1> Av23(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A23_subid, A23_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B23_subid = {1, 5, 9, 13};
    std::array<T,4> B23_coeff_list = {-1.0, 1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Bv23(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B23_subid, B23_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C23_subid = {1, 5, 9, 13};
    std::array<T,4> C23_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv23(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C23_subid, C23_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv23.stride(!row_major) == 1)
    {
        Av23.transpose();
        Bv23.transpose();
        Cv23.transpose();
        stra_gemm(comm, cfg, alpha, Bv23, Av23, beta, Cv23);
    } else {
        stra_gemm(comm, cfg, alpha, Av23, Bv23, beta, Cv23);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M23:" << std::endl;
    //print_tensor_matrix( ct );

    // M24 = (1.0 * A_3_3) * (1.0 * B_0_0 + -1.0 * B_0_2 + -1.0 * B_2_0 + 1.0 * B_2_2);  C_0_0 += 1.0 * M24;  C_0_2 += 1.0 * M24;  C_2_0 += 1.0 * M24;  C_2_2 += 1.0 * M24;
    std::array<unsigned, 1> A24_subid = {15};
    std::array<T,1> A24_coeff_list = {1.0};
    stra_tensor_view<T,1> Av24(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A24_subid, A24_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B24_subid = {0, 4, 8, 12};
    std::array<T,4> B24_coeff_list = {1.0, -1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Bv24(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B24_subid, B24_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C24_subid = {0, 4, 8, 12};
    std::array<T,4> C24_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv24(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C24_subid, C24_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv24.stride(!row_major) == 1)
    {
        Av24.transpose();
        Bv24.transpose();
        Cv24.transpose();
        stra_gemm(comm, cfg, alpha, Bv24, Av24, beta, Cv24);
    } else {
        stra_gemm(comm, cfg, alpha, Av24, Bv24, beta, Cv24);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M24:" << std::endl;
    //print_tensor_matrix( ct );

    // M25 = (1.0 * A_3_0 + 1.0 * A_3_1) * (-1.0 * B_0_3 + 1.0 * B_2_3);  C_0_0 += -1.0 * M25;  C_0_1 += 1.0 * M25;  C_2_0 += -1.0 * M25;  C_2_1 += 1.0 * M25;
    std::array<unsigned, 2> A25_subid = {10, 11};
    std::array<T,2> A25_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av25(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A25_subid, A25_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B25_subid = {5, 13};
    std::array<T,2> B25_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Bv25(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B25_subid, B25_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C25_subid = {0, 1, 8, 9};
    std::array<T,4> C25_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Cv25(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C25_subid, C25_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv25.stride(!row_major) == 1)
    {
        Av25.transpose();
        Bv25.transpose();
        Cv25.transpose();
        stra_gemm(comm, cfg, alpha, Bv25, Av25, beta, Cv25);
    } else {
        stra_gemm(comm, cfg, alpha, Av25, Bv25, beta, Cv25);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M25:" << std::endl;
    //print_tensor_matrix( ct );

    // M26 = (-1.0 * A_3_0 + 1.0 * A_3_2) * (-1.0 * B_0_0 + -1.0 * B_0_1 + 1.0 * B_2_0 + 1.0 * B_2_1);  C_0_3 += 1.0 * M26;  C_2_3 += 1.0 * M26;
    std::array<unsigned, 2> A26_subid = {10, 14};
    std::array<T,2> A26_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Av26(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A26_subid, A26_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B26_subid = {0, 1, 8, 9};
    std::array<T,4> B26_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv26(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B26_subid, B26_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C26_subid = {5, 13};
    std::array<T,2> C26_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv26(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C26_subid, C26_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv26.stride(!row_major) == 1)
    {
        Av26.transpose();
        Bv26.transpose();
        Cv26.transpose();
        stra_gemm(comm, cfg, alpha, Bv26, Av26, beta, Cv26);
    } else {
        stra_gemm(comm, cfg, alpha, Av26, Bv26, beta, Cv26);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M26:" << std::endl;
    //print_tensor_matrix( ct );

    // M27 = (1.0 * A_3_1 + -1.0 * A_3_3) * (-1.0 * B_0_2 + -1.0 * B_0_3 + 1.0 * B_2_2 + 1.0 * B_2_3);  C_0_0 += 1.0 * M27;  C_2_0 += 1.0 * M27;
    std::array<unsigned, 2> A27_subid = {11, 15};
    std::array<T,2> A27_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Av27(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A27_subid, A27_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B27_subid = {4, 5, 12, 13};
    std::array<T,4> B27_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv27(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B27_subid, B27_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C27_subid = {0, 8};
    std::array<T,2> C27_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv27(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C27_subid, C27_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv27.stride(!row_major) == 1)
    {
        Av27.transpose();
        Bv27.transpose();
        Cv27.transpose();
        stra_gemm(comm, cfg, alpha, Bv27, Av27, beta, Cv27);
    } else {
        stra_gemm(comm, cfg, alpha, Av27, Bv27, beta, Cv27);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M27:" << std::endl;
    //print_tensor_matrix( ct );

    // M28 = (1.0 * A_0_0 + 1.0 * A_0_3 + 1.0 * A_1_0 + 1.0 * A_1_3) * (1.0 * B_3_0 + 1.0 * B_3_3);  C_0_0 += -1.0 * M28;  C_0_3 += -1.0 * M28;  C_1_0 += 1.0 * M28;  C_1_3 += 1.0 * M28;
    std::array<unsigned, 4> A28_subid = {0, 5, 2, 7};
    std::array<T,4> A28_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av28(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A28_subid, A28_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B28_subid = {10, 15};
    std::array<T,2> B28_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv28(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B28_subid, B28_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C28_subid = {0, 5, 2, 7};
    std::array<T,4> C28_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv28(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C28_subid, C28_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv28.stride(!row_major) == 1)
    {
        Av28.transpose();
        Bv28.transpose();
        Cv28.transpose();
        stra_gemm(comm, cfg, alpha, Bv28, Av28, beta, Cv28);
    } else {
        stra_gemm(comm, cfg, alpha, Av28, Bv28, beta, Cv28);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M28:" << std::endl;
    //print_tensor_matrix( ct );

    // M29 = (1.0 * A_0_2 + 1.0 * A_0_3 + 1.0 * A_1_2 + 1.0 * A_1_3) * (1.0 * B_3_0);  C_0_2 += -1.0 * M29;  C_0_3 += 1.0 * M29;  C_1_2 += 1.0 * M29;  C_1_3 += -1.0 * M29;
    std::array<unsigned, 4> A29_subid = {4, 5, 6, 7};
    std::array<T,4> A29_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av29(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A29_subid, A29_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 1> B29_subid = {10};
    std::array<T,1> B29_coeff_list = {1.0};
    stra_tensor_view<T,1> Bv29(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B29_subid, B29_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C29_subid = {4, 5, 6, 7};
    std::array<T,4> C29_coeff_list = {-1.0, 1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Cv29(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C29_subid, C29_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv29.stride(!row_major) == 1)
    {
        Av29.transpose();
        Bv29.transpose();
        Cv29.transpose();
        stra_gemm(comm, cfg, alpha, Bv29, Av29, beta, Cv29);
    } else {
        stra_gemm(comm, cfg, alpha, Av29, Bv29, beta, Cv29);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M29:" << std::endl;
    //print_tensor_matrix( ct );

    // M30 = (1.0 * A_0_0 + 1.0 * A_1_0) * (1.0 * B_3_1 + -1.0 * B_3_3);  C_0_1 += -1.0 * M30;  C_0_3 += -1.0 * M30;  C_1_1 += 1.0 * M30;  C_1_3 += 1.0 * M30;
    std::array<unsigned, 2> A30_subid = {0, 2};
    std::array<T,2> A30_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av30(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A30_subid, A30_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B30_subid = {11, 15};
    std::array<T,2> B30_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Bv30(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B30_subid, B30_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C30_subid = {1, 5, 3, 7};
    std::array<T,4> C30_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv30(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C30_subid, C30_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv30.stride(!row_major) == 1)
    {
        Av30.transpose();
        Bv30.transpose();
        Cv30.transpose();
        stra_gemm(comm, cfg, alpha, Bv30, Av30, beta, Cv30);
    } else {
        stra_gemm(comm, cfg, alpha, Av30, Bv30, beta, Cv30);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M30:" << std::endl;
    //print_tensor_matrix( ct );

    // M31 = (1.0 * A_0_3 + 1.0 * A_1_3) * (-1.0 * B_3_0 + 1.0 * B_3_2);  C_0_0 += -1.0 * M31;  C_0_2 += -1.0 * M31;  C_1_0 += 1.0 * M31;  C_1_2 += 1.0 * M31;
    std::array<unsigned, 2> A31_subid = {5, 7};
    std::array<T,2> A31_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av31(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A31_subid, A31_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B31_subid = {10, 14};
    std::array<T,2> B31_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Bv31(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B31_subid, B31_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C31_subid = {0, 4, 2, 6};
    std::array<T,4> C31_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv31(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C31_subid, C31_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv31.stride(!row_major) == 1)
    {
        Av31.transpose();
        Bv31.transpose();
        Cv31.transpose();
        stra_gemm(comm, cfg, alpha, Bv31, Av31, beta, Cv31);
    } else {
        stra_gemm(comm, cfg, alpha, Av31, Bv31, beta, Cv31);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M31:" << std::endl;
    //print_tensor_matrix( ct );

    // M32 = (1.0 * A_0_0 + 1.0 * A_0_1 + 1.0 * A_1_0 + 1.0 * A_1_1) * (1.0 * B_3_3);  C_0_0 += 1.0 * M32;  C_0_1 += -1.0 * M32;  C_1_0 += -1.0 * M32;  C_1_1 += 1.0 * M32;
    std::array<unsigned, 4> A32_subid = {0, 1, 2, 3};
    std::array<T,4> A32_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av32(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A32_subid, A32_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 1> B32_subid = {15};
    std::array<T,1> B32_coeff_list = {1.0};
    stra_tensor_view<T,1> Bv32(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B32_subid, B32_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C32_subid = {0, 1, 2, 3};
    std::array<T,4> C32_coeff_list = {1.0, -1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Cv32(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C32_subid, C32_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv32.stride(!row_major) == 1)
    {
        Av32.transpose();
        Bv32.transpose();
        Cv32.transpose();
        stra_gemm(comm, cfg, alpha, Bv32, Av32, beta, Cv32);
    } else {
        stra_gemm(comm, cfg, alpha, Av32, Bv32, beta, Cv32);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M32:" << std::endl;
    //print_tensor_matrix( ct );

    // M33 = (-1.0 * A_0_0 + 1.0 * A_0_2 + -1.0 * A_1_0 + 1.0 * A_1_2) * (1.0 * B_3_0 + 1.0 * B_3_1);  C_0_3 += -1.0 * M33;  C_1_3 += 1.0 * M33;
    std::array<unsigned, 4> A33_subid = {0, 4, 2, 6};
    std::array<T,4> A33_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Av33(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A33_subid, A33_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B33_subid = {10, 11};
    std::array<T,2> B33_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv33(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B33_subid, B33_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C33_subid = {5, 7};
    std::array<T,2> C33_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Cv33(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C33_subid, C33_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv33.stride(!row_major) == 1)
    {
        Av33.transpose();
        Bv33.transpose();
        Cv33.transpose();
        stra_gemm(comm, cfg, alpha, Bv33, Av33, beta, Cv33);
    } else {
        stra_gemm(comm, cfg, alpha, Av33, Bv33, beta, Cv33);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M33:" << std::endl;
    //print_tensor_matrix( ct );

    // M34 = (1.0 * A_0_1 + -1.0 * A_0_3 + 1.0 * A_1_1 + -1.0 * A_1_3) * (1.0 * B_3_2 + 1.0 * B_3_3);  C_0_0 += -1.0 * M34;  C_1_0 += 1.0 * M34;
    std::array<unsigned, 4> A34_subid = {1, 5, 3, 7};
    std::array<T,4> A34_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Av34(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A34_subid, A34_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B34_subid = {14, 15};
    std::array<T,2> B34_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv34(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B34_subid, B34_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C34_subid = {0, 2};
    std::array<T,2> C34_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Cv34(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C34_subid, C34_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv34.stride(!row_major) == 1)
    {
        Av34.transpose();
        Bv34.transpose();
        Cv34.transpose();
        stra_gemm(comm, cfg, alpha, Bv34, Av34, beta, Cv34);
    } else {
        stra_gemm(comm, cfg, alpha, Av34, Bv34, beta, Cv34);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M34:" << std::endl;
    //print_tensor_matrix( ct );

    // M35 = (-1.0 * A_0_0 + -1.0 * A_0_3 + 1.0 * A_2_0 + 1.0 * A_2_3) * (1.0 * B_0_0 + 1.0 * B_0_3 + 1.0 * B_1_0 + 1.0 * B_1_3);  C_3_0 += 1.0 * M35;  C_3_3 += 1.0 * M35;
    std::array<unsigned, 4> A35_subid = {0, 5, 8, 13};
    std::array<T,4> A35_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av35(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A35_subid, A35_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B35_subid = {0, 5, 2, 7};
    std::array<T,4> B35_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv35(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B35_subid, B35_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C35_subid = {10, 15};
    std::array<T,2> C35_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv35(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C35_subid, C35_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv35.stride(!row_major) == 1)
    {
        Av35.transpose();
        Bv35.transpose();
        Cv35.transpose();
        stra_gemm(comm, cfg, alpha, Bv35, Av35, beta, Cv35);
    } else {
        stra_gemm(comm, cfg, alpha, Av35, Bv35, beta, Cv35);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M35:" << std::endl;
    //print_tensor_matrix( ct );

    // M36 = (-1.0 * A_0_2 + -1.0 * A_0_3 + 1.0 * A_2_2 + 1.0 * A_2_3) * (1.0 * B_0_0 + 1.0 * B_1_0);  C_3_2 += 1.0 * M36;  C_3_3 += -1.0 * M36;
    std::array<unsigned, 4> A36_subid = {4, 5, 12, 13};
    std::array<T,4> A36_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av36(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A36_subid, A36_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B36_subid = {0, 2};
    std::array<T,2> B36_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv36(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B36_subid, B36_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C36_subid = {14, 15};
    std::array<T,2> C36_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Cv36(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C36_subid, C36_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv36.stride(!row_major) == 1)
    {
        Av36.transpose();
        Bv36.transpose();
        Cv36.transpose();
        stra_gemm(comm, cfg, alpha, Bv36, Av36, beta, Cv36);
    } else {
        stra_gemm(comm, cfg, alpha, Av36, Bv36, beta, Cv36);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M36:" << std::endl;
    //print_tensor_matrix( ct );

    // M37 = (-1.0 * A_0_0 + 1.0 * A_2_0) * (1.0 * B_0_1 + -1.0 * B_0_3 + 1.0 * B_1_1 + -1.0 * B_1_3);  C_3_1 += 1.0 * M37;  C_3_3 += 1.0 * M37;
    std::array<unsigned, 2> A37_subid = {0, 8};
    std::array<T,2> A37_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Av37(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A37_subid, A37_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B37_subid = {1, 5, 3, 7};
    std::array<T,4> B37_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Bv37(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B37_subid, B37_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C37_subid = {11, 15};
    std::array<T,2> C37_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv37(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C37_subid, C37_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv37.stride(!row_major) == 1)
    {
        Av37.transpose();
        Bv37.transpose();
        Cv37.transpose();
        stra_gemm(comm, cfg, alpha, Bv37, Av37, beta, Cv37);
    } else {
        stra_gemm(comm, cfg, alpha, Av37, Bv37, beta, Cv37);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M37:" << std::endl;
    //print_tensor_matrix( ct );

    // M38 = (-1.0 * A_0_3 + 1.0 * A_2_3) * (-1.0 * B_0_0 + 1.0 * B_0_2 + -1.0 * B_1_0 + 1.0 * B_1_2);  C_3_0 += 1.0 * M38;  C_3_2 += 1.0 * M38;
    std::array<unsigned, 2> A38_subid = {5, 13};
    std::array<T,2> A38_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Av38(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A38_subid, A38_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B38_subid = {0, 4, 2, 6};
    std::array<T,4> B38_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Bv38(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B38_subid, B38_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C38_subid = {10, 14};
    std::array<T,2> C38_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv38(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C38_subid, C38_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv38.stride(!row_major) == 1)
    {
        Av38.transpose();
        Bv38.transpose();
        Cv38.transpose();
        stra_gemm(comm, cfg, alpha, Bv38, Av38, beta, Cv38);
    } else {
        stra_gemm(comm, cfg, alpha, Av38, Bv38, beta, Cv38);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M38:" << std::endl;
    //print_tensor_matrix( ct );

    // M39 = (-1.0 * A_0_0 + -1.0 * A_0_1 + 1.0 * A_2_0 + 1.0 * A_2_1) * (1.0 * B_0_3 + 1.0 * B_1_3);  C_3_0 += -1.0 * M39;  C_3_1 += 1.0 * M39;
    std::array<unsigned, 4> A39_subid = {0, 1, 8, 9};
    std::array<T,4> A39_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av39(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A39_subid, A39_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B39_subid = {5, 7};
    std::array<T,2> B39_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv39(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B39_subid, B39_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C39_subid = {10, 11};
    std::array<T,2> C39_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Cv39(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C39_subid, C39_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv39.stride(!row_major) == 1)
    {
        Av39.transpose();
        Bv39.transpose();
        Cv39.transpose();
        stra_gemm(comm, cfg, alpha, Bv39, Av39, beta, Cv39);
    } else {
        stra_gemm(comm, cfg, alpha, Av39, Bv39, beta, Cv39);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M39:" << std::endl;
    //print_tensor_matrix( ct );

    // M40 = (1.0 * A_0_0 + -1.0 * A_0_2 + -1.0 * A_2_0 + 1.0 * A_2_2) * (1.0 * B_0_0 + 1.0 * B_0_1 + 1.0 * B_1_0 + 1.0 * B_1_1);  C_3_3 += 1.0 * M40;
    std::array<unsigned, 4> A40_subid = {0, 4, 8, 12};
    std::array<T,4> A40_coeff_list = {1.0, -1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Av40(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A40_subid, A40_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B40_subid = {0, 1, 2, 3};
    std::array<T,4> B40_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv40(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B40_subid, B40_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 1> C40_subid = {15};
    std::array<T,1> C40_coeff_list = {1.0};
    stra_tensor_view<T,1> Cv40(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C40_subid, C40_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv40.stride(!row_major) == 1)
    {
        Av40.transpose();
        Bv40.transpose();
        Cv40.transpose();
        stra_gemm(comm, cfg, alpha, Bv40, Av40, beta, Cv40);
    } else {
        stra_gemm(comm, cfg, alpha, Av40, Bv40, beta, Cv40);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M40:" << std::endl;
    //print_tensor_matrix( ct );

    // M41 = (-1.0 * A_0_1 + 1.0 * A_0_3 + 1.0 * A_2_1 + -1.0 * A_2_3) * (1.0 * B_0_2 + 1.0 * B_0_3 + 1.0 * B_1_2 + 1.0 * B_1_3);  C_3_0 += 1.0 * M41;
    std::array<unsigned, 4> A41_subid = {1, 5, 9, 13};
    std::array<T,4> A41_coeff_list = {-1.0, 1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Av41(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A41_subid, A41_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B41_subid = {4, 5, 6, 7};
    std::array<T,4> B41_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv41(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B41_subid, B41_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 1> C41_subid = {10};
    std::array<T,1> C41_coeff_list = {1.0};
    stra_tensor_view<T,1> Cv41(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C41_subid, C41_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv41.stride(!row_major) == 1)
    {
        Av41.transpose();
        Bv41.transpose();
        Cv41.transpose();
        stra_gemm(comm, cfg, alpha, Bv41, Av41, beta, Cv41);
    } else {
        stra_gemm(comm, cfg, alpha, Av41, Bv41, beta, Cv41);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M41:" << std::endl;
    //print_tensor_matrix( ct );

    // M42 = (1.0 * A_1_0 + 1.0 * A_1_3 + -1.0 * A_3_0 + -1.0 * A_3_3) * (1.0 * B_2_0 + 1.0 * B_2_3 + 1.0 * B_3_0 + 1.0 * B_3_3);  C_0_0 += 1.0 * M42;  C_0_3 += 1.0 * M42;
    std::array<unsigned, 4> A42_subid = {2, 7, 10, 15};
    std::array<T,4> A42_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Av42(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A42_subid, A42_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B42_subid = {8, 13, 10, 15};
    std::array<T,4> B42_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv42(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B42_subid, B42_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C42_subid = {0, 5};
    std::array<T,2> C42_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv42(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C42_subid, C42_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv42.stride(!row_major) == 1)
    {
        Av42.transpose();
        Bv42.transpose();
        Cv42.transpose();
        stra_gemm(comm, cfg, alpha, Bv42, Av42, beta, Cv42);
    } else {
        stra_gemm(comm, cfg, alpha, Av42, Bv42, beta, Cv42);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M42:" << std::endl;
    //print_tensor_matrix( ct );

    // M43 = (1.0 * A_1_2 + 1.0 * A_1_3 + -1.0 * A_3_2 + -1.0 * A_3_3) * (1.0 * B_2_0 + 1.0 * B_3_0);  C_0_2 += 1.0 * M43;  C_0_3 += -1.0 * M43;
    std::array<unsigned, 4> A43_subid = {6, 7, 14, 15};
    std::array<T,4> A43_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Av43(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A43_subid, A43_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B43_subid = {8, 10};
    std::array<T,2> B43_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv43(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B43_subid, B43_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C43_subid = {4, 5};
    std::array<T,2> C43_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Cv43(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C43_subid, C43_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv43.stride(!row_major) == 1)
    {
        Av43.transpose();
        Bv43.transpose();
        Cv43.transpose();
        stra_gemm(comm, cfg, alpha, Bv43, Av43, beta, Cv43);
    } else {
        stra_gemm(comm, cfg, alpha, Av43, Bv43, beta, Cv43);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M43:" << std::endl;
    //print_tensor_matrix( ct );

    // M44 = (1.0 * A_1_0 + -1.0 * A_3_0) * (1.0 * B_2_1 + -1.0 * B_2_3 + 1.0 * B_3_1 + -1.0 * B_3_3);  C_0_1 += 1.0 * M44;  C_0_3 += 1.0 * M44;
    std::array<unsigned, 2> A44_subid = {2, 10};
    std::array<T,2> A44_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Av44(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A44_subid, A44_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B44_subid = {9, 13, 11, 15};
    std::array<T,4> B44_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Bv44(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B44_subid, B44_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C44_subid = {1, 5};
    std::array<T,2> C44_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv44(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C44_subid, C44_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv44.stride(!row_major) == 1)
    {
        Av44.transpose();
        Bv44.transpose();
        Cv44.transpose();
        stra_gemm(comm, cfg, alpha, Bv44, Av44, beta, Cv44);
    } else {
        stra_gemm(comm, cfg, alpha, Av44, Bv44, beta, Cv44);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M44:" << std::endl;
    //print_tensor_matrix( ct );

    // M45 = (1.0 * A_1_3 + -1.0 * A_3_3) * (-1.0 * B_2_0 + 1.0 * B_2_2 + -1.0 * B_3_0 + 1.0 * B_3_2);  C_0_0 += 1.0 * M45;  C_0_2 += 1.0 * M45;
    std::array<unsigned, 2> A45_subid = {7, 15};
    std::array<T,2> A45_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Av45(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A45_subid, A45_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B45_subid = {8, 12, 10, 14};
    std::array<T,4> B45_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Bv45(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B45_subid, B45_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C45_subid = {0, 4};
    std::array<T,2> C45_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv45(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C45_subid, C45_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv45.stride(!row_major) == 1)
    {
        Av45.transpose();
        Bv45.transpose();
        Cv45.transpose();
        stra_gemm(comm, cfg, alpha, Bv45, Av45, beta, Cv45);
    } else {
        stra_gemm(comm, cfg, alpha, Av45, Bv45, beta, Cv45);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M45:" << std::endl;
    //print_tensor_matrix( ct );

    // M46 = (1.0 * A_1_0 + 1.0 * A_1_1 + -1.0 * A_3_0 + -1.0 * A_3_1) * (1.0 * B_2_3 + 1.0 * B_3_3);  C_0_0 += -1.0 * M46;  C_0_1 += 1.0 * M46;
    std::array<unsigned, 4> A46_subid = {2, 3, 10, 11};
    std::array<T,4> A46_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Av46(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A46_subid, A46_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B46_subid = {13, 15};
    std::array<T,2> B46_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv46(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B46_subid, B46_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C46_subid = {0, 1};
    std::array<T,2> C46_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Cv46(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C46_subid, C46_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv46.stride(!row_major) == 1)
    {
        Av46.transpose();
        Bv46.transpose();
        Cv46.transpose();
        stra_gemm(comm, cfg, alpha, Bv46, Av46, beta, Cv46);
    } else {
        stra_gemm(comm, cfg, alpha, Av46, Bv46, beta, Cv46);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M46:" << std::endl;
    //print_tensor_matrix( ct );

    // M47 = (-1.0 * A_1_0 + 1.0 * A_1_2 + 1.0 * A_3_0 + -1.0 * A_3_2) * (1.0 * B_2_0 + 1.0 * B_2_1 + 1.0 * B_3_0 + 1.0 * B_3_1);  C_0_3 += 1.0 * M47;
    std::array<unsigned, 4> A47_subid = {2, 6, 10, 14};
    std::array<T,4> A47_coeff_list = {-1.0, 1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Av47(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A47_subid, A47_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B47_subid = {8, 9, 10, 11};
    std::array<T,4> B47_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv47(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B47_subid, B47_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 1> C47_subid = {5};
    std::array<T,1> C47_coeff_list = {1.0};
    stra_tensor_view<T,1> Cv47(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C47_subid, C47_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv47.stride(!row_major) == 1)
    {
        Av47.transpose();
        Bv47.transpose();
        Cv47.transpose();
        stra_gemm(comm, cfg, alpha, Bv47, Av47, beta, Cv47);
    } else {
        stra_gemm(comm, cfg, alpha, Av47, Bv47, beta, Cv47);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M47:" << std::endl;
    //print_tensor_matrix( ct );

    // M48 = (1.0 * A_1_1 + -1.0 * A_1_3 + -1.0 * A_3_1 + 1.0 * A_3_3) * (1.0 * B_2_2 + 1.0 * B_2_3 + 1.0 * B_3_2 + 1.0 * B_3_3);  C_0_0 += 1.0 * M48;
    std::array<unsigned, 4> A48_subid = {3, 7, 11, 15};
    std::array<T,4> A48_coeff_list = {1.0, -1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Av48(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A48_subid, A48_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B48_subid = {12, 13, 14, 15};
    std::array<T,4> B48_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv48(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B48_subid, B48_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 1> C48_subid = {0};
    std::array<T,1> C48_coeff_list = {1.0};
    stra_tensor_view<T,1> Cv48(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C48_subid, C48_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv48.stride(!row_major) == 1)
    {
        Av48.transpose();
        Bv48.transpose();
        Cv48.transpose();
        stra_gemm(comm, cfg, alpha, Bv48, Av48, beta, Cv48);
    } else {
        stra_gemm(comm, cfg, alpha, Av48, Bv48, beta, Cv48);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M48:" << std::endl;
    //print_tensor_matrix( ct );


    //std::cout << "stra_internal/stra_mult_2level_A:" << std::endl;
    //print_tensor_matrix( at );

    //std::cout << "stra_internal/stra_mult_2level_B:" << std::endl;
    //print_tensor_matrix( bt );

    //std::cout << "stra_internal/stra_mult_2level_M6:" << std::endl;
    //print_tensor_matrix( ct );

}

#define INSTANTIATE_CONTRACT_BLIS(T) \
template void stra_contract_2level_blis(const communicator& comm, const config& cfg, \
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
void stra_contract_2level_blis_naive(const communicator& comm, const config& cfg,
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
    //std::cout << "Enter stra_internal/3t/stra_mult_2level/stra_contract_2level_blis_naive\n" << std::endl;

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
;
    auto my_stride_B_AB = stl_ext::permuted(stride_B_AB, reorder_AB);
    auto my_stride_B_BC = stl_ext::permuted(stride_B_BC, reorder_BC);
;
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

    //StraTensorGEMM stra_gemm;

    //len_type m = ct.length(0);
    //len_type n = ct.length(1);
    //len_type k = at.length(1);

    //int nt = comm.num_threads();
    //auto tc = make_gemm_thread_config<T>(cfg, nt, m, n, k);
    //step<0>(stra_gemm).distribute = tc.jc_nt;
    //step<4>(stra_gemm).distribute = tc.ic_nt;
    //step<8>(stra_gemm).distribute = tc.jr_nt;
    //step<9>(stra_gemm).distribute = tc.ir_nt;

    //const len_type ms=m/2, ks=k/2, ns=n/2;

    const std::array<unsigned,2> A_divisor={4,4};
    const std::array<unsigned,2> B_divisor={4,4};
    const std::array<unsigned,2> C_divisor={4,4};

    // M0 = (1.0 * A_0_0 + 1.0 * A_0_3 + 1.0 * A_3_0 + 1.0 * A_3_3) * (1.0 * B_0_0 + 1.0 * B_0_3 + 1.0 * B_3_0 + 1.0 * B_3_3);  C_0_0 += 1.0 * M0;  C_0_3 += 1.0 * M0;  C_3_0 += 1.0 * M0;  C_3_3 += 1.0 * M0;
    std::array<unsigned, 4> A0_subid = {0, 5, 10, 15};
    std::array<T,4> A0_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av0(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A0_subid, A0_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B0_subid = {0, 5, 10, 15};
    std::array<T,4> B0_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv0(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B0_subid, B0_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C0_subid = {0, 5, 10, 15};
    std::array<T,4> C0_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv0(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C0_subid, C0_coeff_list, my_stride_C_AC, my_stride_C_BC);
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

    // M1 = (1.0 * A_0_2 + 1.0 * A_0_3 + 1.0 * A_3_2 + 1.0 * A_3_3) * (1.0 * B_0_0 + 1.0 * B_3_0);  C_0_2 += 1.0 * M1;  C_0_3 += -1.0 * M1;  C_3_2 += 1.0 * M1;  C_3_3 += -1.0 * M1;
    std::array<unsigned, 4> A1_subid = {4, 5, 14, 15};
    std::array<T,4> A1_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av1(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A1_subid, A1_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B1_subid = {0, 10};
    std::array<T,2> B1_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv1(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B1_subid, B1_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C1_subid = {4, 5, 14, 15};
    std::array<T,4> C1_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Cv1(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C1_subid, C1_coeff_list, my_stride_C_AC, my_stride_C_BC);
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

    // M2 = (1.0 * A_0_0 + 1.0 * A_3_0) * (1.0 * B_0_1 + -1.0 * B_0_3 + 1.0 * B_3_1 + -1.0 * B_3_3);  C_0_1 += 1.0 * M2;  C_0_3 += 1.0 * M2;  C_3_1 += 1.0 * M2;  C_3_3 += 1.0 * M2;
    std::array<unsigned, 2> A2_subid = {0, 10};
    std::array<T,2> A2_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av2(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A2_subid, A2_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B2_subid = {1, 5, 11, 15};
    std::array<T,4> B2_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Bv2(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B2_subid, B2_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C2_subid = {1, 5, 11, 15};
    std::array<T,4> C2_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv2(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C2_subid, C2_coeff_list, my_stride_C_AC, my_stride_C_BC);
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

    // M3 = (1.0 * A_0_3 + 1.0 * A_3_3) * (-1.0 * B_0_0 + 1.0 * B_0_2 + -1.0 * B_3_0 + 1.0 * B_3_2);  C_0_0 += 1.0 * M3;  C_0_2 += 1.0 * M3;  C_3_0 += 1.0 * M3;  C_3_2 += 1.0 * M3;
    std::array<unsigned, 2> A3_subid = {5, 15};
    std::array<T,2> A3_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av3(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A3_subid, A3_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B3_subid = {0, 4, 10, 14};
    std::array<T,4> B3_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Bv3(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B3_subid, B3_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C3_subid = {0, 4, 10, 14};
    std::array<T,4> C3_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv3(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C3_subid, C3_coeff_list, my_stride_C_AC, my_stride_C_BC);
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

    // M4 = (1.0 * A_0_0 + 1.0 * A_0_1 + 1.0 * A_3_0 + 1.0 * A_3_1) * (1.0 * B_0_3 + 1.0 * B_3_3);  C_0_0 += -1.0 * M4;  C_0_1 += 1.0 * M4;  C_3_0 += -1.0 * M4;  C_3_1 += 1.0 * M4;
    std::array<unsigned, 4> A4_subid = {0, 1, 10, 11};
    std::array<T,4> A4_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av4(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A4_subid, A4_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B4_subid = {5, 15};
    std::array<T,2> B4_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv4(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B4_subid, B4_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C4_subid = {0, 1, 10, 11};
    std::array<T,4> C4_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Cv4(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C4_subid, C4_coeff_list, my_stride_C_AC, my_stride_C_BC);
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

    // M5 = (-1.0 * A_0_0 + 1.0 * A_0_2 + -1.0 * A_3_0 + 1.0 * A_3_2) * (1.0 * B_0_0 + 1.0 * B_0_1 + 1.0 * B_3_0 + 1.0 * B_3_1);  C_0_3 += 1.0 * M5;  C_3_3 += 1.0 * M5;
    std::array<unsigned, 4> A5_subid = {0, 4, 10, 14};
    std::array<T,4> A5_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Av5(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A5_subid, A5_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B5_subid = {0, 1, 10, 11};
    std::array<T,4> B5_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv5(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B5_subid, B5_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C5_subid = {5, 15};
    std::array<T,2> C5_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv5(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C5_subid, C5_coeff_list, my_stride_C_AC, my_stride_C_BC);
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

    // M6 = (1.0 * A_0_1 + -1.0 * A_0_3 + 1.0 * A_3_1 + -1.0 * A_3_3) * (1.0 * B_0_2 + 1.0 * B_0_3 + 1.0 * B_3_2 + 1.0 * B_3_3);  C_0_0 += 1.0 * M6;  C_3_0 += 1.0 * M6;
    std::array<unsigned, 4> A6_subid = {1, 5, 11, 15};
    std::array<T,4> A6_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Av6(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A6_subid, A6_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B6_subid = {4, 5, 14, 15};
    std::array<T,4> B6_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv6(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B6_subid, B6_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C6_subid = {0, 10};
    std::array<T,2> C6_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv6(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C6_subid, C6_coeff_list, my_stride_C_AC, my_stride_C_BC);
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

    // M7 = (1.0 * A_2_0 + 1.0 * A_2_3 + 1.0 * A_3_0 + 1.0 * A_3_3) * (1.0 * B_0_0 + 1.0 * B_0_3);  C_2_0 += 1.0 * M7;  C_2_3 += 1.0 * M7;  C_3_0 += -1.0 * M7;  C_3_3 += -1.0 * M7;
    std::array<unsigned, 4> A7_subid = {8, 13, 10, 15};
    std::array<T,4> A7_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av7(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A7_subid, A7_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B7_subid = {0, 5};
    std::array<T,2> B7_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv7(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B7_subid, B7_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C7_subid = {8, 13, 10, 15};
    std::array<T,4> C7_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Cv7(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C7_subid, C7_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv7.stride(!row_major) == 1)
    {
        Av7.transpose();
        Bv7.transpose();
        Cv7.transpose();
        straprim_naive2(comm, cfg, alpha, Bv7, Av7, beta, Cv7);
    } else {
        straprim_naive2(comm, cfg, alpha, Av7, Bv7, beta, Cv7);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M7:" << std::endl;
    //print_tensor_matrix( ct );

    // M8 = (1.0 * A_2_2 + 1.0 * A_2_3 + 1.0 * A_3_2 + 1.0 * A_3_3) * (1.0 * B_0_0);  C_2_2 += 1.0 * M8;  C_2_3 += -1.0 * M8;  C_3_2 += -1.0 * M8;  C_3_3 += 1.0 * M8;
    std::array<unsigned, 4> A8_subid = {12, 13, 14, 15};
    std::array<T,4> A8_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av8(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A8_subid, A8_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 1> B8_subid = {0};
    std::array<T,1> B8_coeff_list = {1.0};
    stra_tensor_view<T,1> Bv8(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B8_subid, B8_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C8_subid = {12, 13, 14, 15};
    std::array<T,4> C8_coeff_list = {1.0, -1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Cv8(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C8_subid, C8_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv8.stride(!row_major) == 1)
    {
        Av8.transpose();
        Bv8.transpose();
        Cv8.transpose();
        straprim_naive2(comm, cfg, alpha, Bv8, Av8, beta, Cv8);
    } else {
        straprim_naive2(comm, cfg, alpha, Av8, Bv8, beta, Cv8);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M8:" << std::endl;
    //print_tensor_matrix( ct );

    // M9 = (1.0 * A_2_0 + 1.0 * A_3_0) * (1.0 * B_0_1 + -1.0 * B_0_3);  C_2_1 += 1.0 * M9;  C_2_3 += 1.0 * M9;  C_3_1 += -1.0 * M9;  C_3_3 += -1.0 * M9;
    std::array<unsigned, 2> A9_subid = {8, 10};
    std::array<T,2> A9_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av9(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A9_subid, A9_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B9_subid = {1, 5};
    std::array<T,2> B9_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Bv9(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B9_subid, B9_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C9_subid = {9, 13, 11, 15};
    std::array<T,4> C9_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Cv9(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C9_subid, C9_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv9.stride(!row_major) == 1)
    {
        Av9.transpose();
        Bv9.transpose();
        Cv9.transpose();
        straprim_naive2(comm, cfg, alpha, Bv9, Av9, beta, Cv9);
    } else {
        straprim_naive2(comm, cfg, alpha, Av9, Bv9, beta, Cv9);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M9:" << std::endl;
    //print_tensor_matrix( ct );

    // M10 = (1.0 * A_2_3 + 1.0 * A_3_3) * (-1.0 * B_0_0 + 1.0 * B_0_2);  C_2_0 += 1.0 * M10;  C_2_2 += 1.0 * M10;  C_3_0 += -1.0 * M10;  C_3_2 += -1.0 * M10;
    std::array<unsigned, 2> A10_subid = {13, 15};
    std::array<T,2> A10_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av10(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A10_subid, A10_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B10_subid = {0, 4};
    std::array<T,2> B10_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Bv10(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B10_subid, B10_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C10_subid = {8, 12, 10, 14};
    std::array<T,4> C10_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Cv10(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C10_subid, C10_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv10.stride(!row_major) == 1)
    {
        Av10.transpose();
        Bv10.transpose();
        Cv10.transpose();
        straprim_naive2(comm, cfg, alpha, Bv10, Av10, beta, Cv10);
    } else {
        straprim_naive2(comm, cfg, alpha, Av10, Bv10, beta, Cv10);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M10:" << std::endl;
    //print_tensor_matrix( ct );

    // M11 = (1.0 * A_2_0 + 1.0 * A_2_1 + 1.0 * A_3_0 + 1.0 * A_3_1) * (1.0 * B_0_3);  C_2_0 += -1.0 * M11;  C_2_1 += 1.0 * M11;  C_3_0 += 1.0 * M11;  C_3_1 += -1.0 * M11;
    std::array<unsigned, 4> A11_subid = {8, 9, 10, 11};
    std::array<T,4> A11_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av11(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A11_subid, A11_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 1> B11_subid = {5};
    std::array<T,1> B11_coeff_list = {1.0};
    stra_tensor_view<T,1> Bv11(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B11_subid, B11_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C11_subid = {8, 9, 10, 11};
    std::array<T,4> C11_coeff_list = {-1.0, 1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Cv11(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C11_subid, C11_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv11.stride(!row_major) == 1)
    {
        Av11.transpose();
        Bv11.transpose();
        Cv11.transpose();
        straprim_naive2(comm, cfg, alpha, Bv11, Av11, beta, Cv11);
    } else {
        straprim_naive2(comm, cfg, alpha, Av11, Bv11, beta, Cv11);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M11:" << std::endl;
    //print_tensor_matrix( ct );

    // M12 = (-1.0 * A_2_0 + 1.0 * A_2_2 + -1.0 * A_3_0 + 1.0 * A_3_2) * (1.0 * B_0_0 + 1.0 * B_0_1);  C_2_3 += 1.0 * M12;  C_3_3 += -1.0 * M12;
    std::array<unsigned, 4> A12_subid = {8, 12, 10, 14};
    std::array<T,4> A12_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Av12(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A12_subid, A12_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B12_subid = {0, 1};
    std::array<T,2> B12_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv12(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B12_subid, B12_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C12_subid = {13, 15};
    std::array<T,2> C12_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Cv12(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C12_subid, C12_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv12.stride(!row_major) == 1)
    {
        Av12.transpose();
        Bv12.transpose();
        Cv12.transpose();
        straprim_naive2(comm, cfg, alpha, Bv12, Av12, beta, Cv12);
    } else {
        straprim_naive2(comm, cfg, alpha, Av12, Bv12, beta, Cv12);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M12:" << std::endl;
    //print_tensor_matrix( ct );

    // M13 = (1.0 * A_2_1 + -1.0 * A_2_3 + 1.0 * A_3_1 + -1.0 * A_3_3) * (1.0 * B_0_2 + 1.0 * B_0_3);  C_2_0 += 1.0 * M13;  C_3_0 += -1.0 * M13;
    std::array<unsigned, 4> A13_subid = {9, 13, 11, 15};
    std::array<T,4> A13_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Av13(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A13_subid, A13_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B13_subid = {4, 5};
    std::array<T,2> B13_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv13(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B13_subid, B13_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C13_subid = {8, 10};
    std::array<T,2> C13_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Cv13(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C13_subid, C13_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv13.stride(!row_major) == 1)
    {
        Av13.transpose();
        Bv13.transpose();
        Cv13.transpose();
        straprim_naive2(comm, cfg, alpha, Bv13, Av13, beta, Cv13);
    } else {
        straprim_naive2(comm, cfg, alpha, Av13, Bv13, beta, Cv13);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M13:" << std::endl;
    //print_tensor_matrix( ct );

    // M14 = (1.0 * A_0_0 + 1.0 * A_0_3) * (1.0 * B_1_0 + 1.0 * B_1_3 + -1.0 * B_3_0 + -1.0 * B_3_3);  C_1_0 += 1.0 * M14;  C_1_3 += 1.0 * M14;  C_3_0 += 1.0 * M14;  C_3_3 += 1.0 * M14;
    std::array<unsigned, 2> A14_subid = {0, 5};
    std::array<T,2> A14_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av14(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A14_subid, A14_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B14_subid = {2, 7, 10, 15};
    std::array<T,4> B14_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Bv14(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B14_subid, B14_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C14_subid = {2, 7, 10, 15};
    std::array<T,4> C14_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv14(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C14_subid, C14_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv14.stride(!row_major) == 1)
    {
        Av14.transpose();
        Bv14.transpose();
        Cv14.transpose();
        straprim_naive2(comm, cfg, alpha, Bv14, Av14, beta, Cv14);
    } else {
        straprim_naive2(comm, cfg, alpha, Av14, Bv14, beta, Cv14);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M14:" << std::endl;
    //print_tensor_matrix( ct );

    // M15 = (1.0 * A_0_2 + 1.0 * A_0_3) * (1.0 * B_1_0 + -1.0 * B_3_0);  C_1_2 += 1.0 * M15;  C_1_3 += -1.0 * M15;  C_3_2 += 1.0 * M15;  C_3_3 += -1.0 * M15;
    std::array<unsigned, 2> A15_subid = {4, 5};
    std::array<T,2> A15_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av15(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A15_subid, A15_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B15_subid = {2, 10};
    std::array<T,2> B15_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Bv15(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B15_subid, B15_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C15_subid = {6, 7, 14, 15};
    std::array<T,4> C15_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Cv15(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C15_subid, C15_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv15.stride(!row_major) == 1)
    {
        Av15.transpose();
        Bv15.transpose();
        Cv15.transpose();
        straprim_naive2(comm, cfg, alpha, Bv15, Av15, beta, Cv15);
    } else {
        straprim_naive2(comm, cfg, alpha, Av15, Bv15, beta, Cv15);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M15:" << std::endl;
    //print_tensor_matrix( ct );

    // M16 = (1.0 * A_0_0) * (1.0 * B_1_1 + -1.0 * B_1_3 + -1.0 * B_3_1 + 1.0 * B_3_3);  C_1_1 += 1.0 * M16;  C_1_3 += 1.0 * M16;  C_3_1 += 1.0 * M16;  C_3_3 += 1.0 * M16;
    std::array<unsigned, 1> A16_subid = {0};
    std::array<T,1> A16_coeff_list = {1.0};
    stra_tensor_view<T,1> Av16(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A16_subid, A16_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B16_subid = {3, 7, 11, 15};
    std::array<T,4> B16_coeff_list = {1.0, -1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Bv16(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B16_subid, B16_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C16_subid = {3, 7, 11, 15};
    std::array<T,4> C16_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv16(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C16_subid, C16_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv16.stride(!row_major) == 1)
    {
        Av16.transpose();
        Bv16.transpose();
        Cv16.transpose();
        straprim_naive2(comm, cfg, alpha, Bv16, Av16, beta, Cv16);
    } else {
        straprim_naive2(comm, cfg, alpha, Av16, Bv16, beta, Cv16);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M16:" << std::endl;
    //print_tensor_matrix( ct );

    // M17 = (1.0 * A_0_3) * (-1.0 * B_1_0 + 1.0 * B_1_2 + 1.0 * B_3_0 + -1.0 * B_3_2);  C_1_0 += 1.0 * M17;  C_1_2 += 1.0 * M17;  C_3_0 += 1.0 * M17;  C_3_2 += 1.0 * M17;
    std::array<unsigned, 1> A17_subid = {5};
    std::array<T,1> A17_coeff_list = {1.0};
    stra_tensor_view<T,1> Av17(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A17_subid, A17_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B17_subid = {2, 6, 10, 14};
    std::array<T,4> B17_coeff_list = {-1.0, 1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Bv17(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B17_subid, B17_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C17_subid = {2, 6, 10, 14};
    std::array<T,4> C17_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv17(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C17_subid, C17_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv17.stride(!row_major) == 1)
    {
        Av17.transpose();
        Bv17.transpose();
        Cv17.transpose();
        straprim_naive2(comm, cfg, alpha, Bv17, Av17, beta, Cv17);
    } else {
        straprim_naive2(comm, cfg, alpha, Av17, Bv17, beta, Cv17);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M17:" << std::endl;
    //print_tensor_matrix( ct );

    // M18 = (1.0 * A_0_0 + 1.0 * A_0_1) * (1.0 * B_1_3 + -1.0 * B_3_3);  C_1_0 += -1.0 * M18;  C_1_1 += 1.0 * M18;  C_3_0 += -1.0 * M18;  C_3_1 += 1.0 * M18;
    std::array<unsigned, 2> A18_subid = {0, 1};
    std::array<T,2> A18_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av18(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A18_subid, A18_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B18_subid = {7, 15};
    std::array<T,2> B18_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Bv18(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B18_subid, B18_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C18_subid = {2, 3, 10, 11};
    std::array<T,4> C18_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Cv18(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C18_subid, C18_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv18.stride(!row_major) == 1)
    {
        Av18.transpose();
        Bv18.transpose();
        Cv18.transpose();
        straprim_naive2(comm, cfg, alpha, Bv18, Av18, beta, Cv18);
    } else {
        straprim_naive2(comm, cfg, alpha, Av18, Bv18, beta, Cv18);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M18:" << std::endl;
    //print_tensor_matrix( ct );

    // M19 = (-1.0 * A_0_0 + 1.0 * A_0_2) * (1.0 * B_1_0 + 1.0 * B_1_1 + -1.0 * B_3_0 + -1.0 * B_3_1);  C_1_3 += 1.0 * M19;  C_3_3 += 1.0 * M19;
    std::array<unsigned, 2> A19_subid = {0, 4};
    std::array<T,2> A19_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Av19(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A19_subid, A19_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B19_subid = {2, 3, 10, 11};
    std::array<T,4> B19_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Bv19(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B19_subid, B19_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C19_subid = {7, 15};
    std::array<T,2> C19_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv19(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C19_subid, C19_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv19.stride(!row_major) == 1)
    {
        Av19.transpose();
        Bv19.transpose();
        Cv19.transpose();
        straprim_naive2(comm, cfg, alpha, Bv19, Av19, beta, Cv19);
    } else {
        straprim_naive2(comm, cfg, alpha, Av19, Bv19, beta, Cv19);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M19:" << std::endl;
    //print_tensor_matrix( ct );

    // M20 = (1.0 * A_0_1 + -1.0 * A_0_3) * (1.0 * B_1_2 + 1.0 * B_1_3 + -1.0 * B_3_2 + -1.0 * B_3_3);  C_1_0 += 1.0 * M20;  C_3_0 += 1.0 * M20;
    std::array<unsigned, 2> A20_subid = {1, 5};
    std::array<T,2> A20_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Av20(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A20_subid, A20_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B20_subid = {6, 7, 14, 15};
    std::array<T,4> B20_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Bv20(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B20_subid, B20_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C20_subid = {2, 10};
    std::array<T,2> C20_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv20(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C20_subid, C20_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv20.stride(!row_major) == 1)
    {
        Av20.transpose();
        Bv20.transpose();
        Cv20.transpose();
        straprim_naive2(comm, cfg, alpha, Bv20, Av20, beta, Cv20);
    } else {
        straprim_naive2(comm, cfg, alpha, Av20, Bv20, beta, Cv20);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M20:" << std::endl;
    //print_tensor_matrix( ct );

    // M21 = (1.0 * A_3_0 + 1.0 * A_3_3) * (-1.0 * B_0_0 + -1.0 * B_0_3 + 1.0 * B_2_0 + 1.0 * B_2_3);  C_0_0 += 1.0 * M21;  C_0_3 += 1.0 * M21;  C_2_0 += 1.0 * M21;  C_2_3 += 1.0 * M21;
    std::array<unsigned, 2> A21_subid = {10, 15};
    std::array<T,2> A21_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av21(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A21_subid, A21_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B21_subid = {0, 5, 8, 13};
    std::array<T,4> B21_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv21(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B21_subid, B21_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C21_subid = {0, 5, 8, 13};
    std::array<T,4> C21_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv21(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C21_subid, C21_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv21.stride(!row_major) == 1)
    {
        Av21.transpose();
        Bv21.transpose();
        Cv21.transpose();
        straprim_naive2(comm, cfg, alpha, Bv21, Av21, beta, Cv21);
    } else {
        straprim_naive2(comm, cfg, alpha, Av21, Bv21, beta, Cv21);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M21:" << std::endl;
    //print_tensor_matrix( ct );

    // M22 = (1.0 * A_3_2 + 1.0 * A_3_3) * (-1.0 * B_0_0 + 1.0 * B_2_0);  C_0_2 += 1.0 * M22;  C_0_3 += -1.0 * M22;  C_2_2 += 1.0 * M22;  C_2_3 += -1.0 * M22;
    std::array<unsigned, 2> A22_subid = {14, 15};
    std::array<T,2> A22_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av22(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A22_subid, A22_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B22_subid = {0, 8};
    std::array<T,2> B22_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Bv22(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B22_subid, B22_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C22_subid = {4, 5, 12, 13};
    std::array<T,4> C22_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Cv22(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C22_subid, C22_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv22.stride(!row_major) == 1)
    {
        Av22.transpose();
        Bv22.transpose();
        Cv22.transpose();
        straprim_naive2(comm, cfg, alpha, Bv22, Av22, beta, Cv22);
    } else {
        straprim_naive2(comm, cfg, alpha, Av22, Bv22, beta, Cv22);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M22:" << std::endl;
    //print_tensor_matrix( ct );

    // M23 = (1.0 * A_3_0) * (-1.0 * B_0_1 + 1.0 * B_0_3 + 1.0 * B_2_1 + -1.0 * B_2_3);  C_0_1 += 1.0 * M23;  C_0_3 += 1.0 * M23;  C_2_1 += 1.0 * M23;  C_2_3 += 1.0 * M23;
    std::array<unsigned, 1> A23_subid = {10};
    std::array<T,1> A23_coeff_list = {1.0};
    stra_tensor_view<T,1> Av23(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A23_subid, A23_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B23_subid = {1, 5, 9, 13};
    std::array<T,4> B23_coeff_list = {-1.0, 1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Bv23(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B23_subid, B23_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C23_subid = {1, 5, 9, 13};
    std::array<T,4> C23_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv23(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C23_subid, C23_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv23.stride(!row_major) == 1)
    {
        Av23.transpose();
        Bv23.transpose();
        Cv23.transpose();
        straprim_naive2(comm, cfg, alpha, Bv23, Av23, beta, Cv23);
    } else {
        straprim_naive2(comm, cfg, alpha, Av23, Bv23, beta, Cv23);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M23:" << std::endl;
    //print_tensor_matrix( ct );

    // M24 = (1.0 * A_3_3) * (1.0 * B_0_0 + -1.0 * B_0_2 + -1.0 * B_2_0 + 1.0 * B_2_2);  C_0_0 += 1.0 * M24;  C_0_2 += 1.0 * M24;  C_2_0 += 1.0 * M24;  C_2_2 += 1.0 * M24;
    std::array<unsigned, 1> A24_subid = {15};
    std::array<T,1> A24_coeff_list = {1.0};
    stra_tensor_view<T,1> Av24(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A24_subid, A24_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B24_subid = {0, 4, 8, 12};
    std::array<T,4> B24_coeff_list = {1.0, -1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Bv24(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B24_subid, B24_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C24_subid = {0, 4, 8, 12};
    std::array<T,4> C24_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv24(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C24_subid, C24_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv24.stride(!row_major) == 1)
    {
        Av24.transpose();
        Bv24.transpose();
        Cv24.transpose();
        straprim_naive2(comm, cfg, alpha, Bv24, Av24, beta, Cv24);
    } else {
        straprim_naive2(comm, cfg, alpha, Av24, Bv24, beta, Cv24);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M24:" << std::endl;
    //print_tensor_matrix( ct );

    // M25 = (1.0 * A_3_0 + 1.0 * A_3_1) * (-1.0 * B_0_3 + 1.0 * B_2_3);  C_0_0 += -1.0 * M25;  C_0_1 += 1.0 * M25;  C_2_0 += -1.0 * M25;  C_2_1 += 1.0 * M25;
    std::array<unsigned, 2> A25_subid = {10, 11};
    std::array<T,2> A25_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av25(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A25_subid, A25_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B25_subid = {5, 13};
    std::array<T,2> B25_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Bv25(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B25_subid, B25_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C25_subid = {0, 1, 8, 9};
    std::array<T,4> C25_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Cv25(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C25_subid, C25_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv25.stride(!row_major) == 1)
    {
        Av25.transpose();
        Bv25.transpose();
        Cv25.transpose();
        straprim_naive2(comm, cfg, alpha, Bv25, Av25, beta, Cv25);
    } else {
        straprim_naive2(comm, cfg, alpha, Av25, Bv25, beta, Cv25);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M25:" << std::endl;
    //print_tensor_matrix( ct );

    // M26 = (-1.0 * A_3_0 + 1.0 * A_3_2) * (-1.0 * B_0_0 + -1.0 * B_0_1 + 1.0 * B_2_0 + 1.0 * B_2_1);  C_0_3 += 1.0 * M26;  C_2_3 += 1.0 * M26;
    std::array<unsigned, 2> A26_subid = {10, 14};
    std::array<T,2> A26_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Av26(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A26_subid, A26_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B26_subid = {0, 1, 8, 9};
    std::array<T,4> B26_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv26(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B26_subid, B26_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C26_subid = {5, 13};
    std::array<T,2> C26_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv26(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C26_subid, C26_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv26.stride(!row_major) == 1)
    {
        Av26.transpose();
        Bv26.transpose();
        Cv26.transpose();
        straprim_naive2(comm, cfg, alpha, Bv26, Av26, beta, Cv26);
    } else {
        straprim_naive2(comm, cfg, alpha, Av26, Bv26, beta, Cv26);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M26:" << std::endl;
    //print_tensor_matrix( ct );

    // M27 = (1.0 * A_3_1 + -1.0 * A_3_3) * (-1.0 * B_0_2 + -1.0 * B_0_3 + 1.0 * B_2_2 + 1.0 * B_2_3);  C_0_0 += 1.0 * M27;  C_2_0 += 1.0 * M27;
    std::array<unsigned, 2> A27_subid = {11, 15};
    std::array<T,2> A27_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Av27(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A27_subid, A27_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B27_subid = {4, 5, 12, 13};
    std::array<T,4> B27_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv27(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B27_subid, B27_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C27_subid = {0, 8};
    std::array<T,2> C27_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv27(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C27_subid, C27_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv27.stride(!row_major) == 1)
    {
        Av27.transpose();
        Bv27.transpose();
        Cv27.transpose();
        straprim_naive2(comm, cfg, alpha, Bv27, Av27, beta, Cv27);
    } else {
        straprim_naive2(comm, cfg, alpha, Av27, Bv27, beta, Cv27);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M27:" << std::endl;
    //print_tensor_matrix( ct );

    // M28 = (1.0 * A_0_0 + 1.0 * A_0_3 + 1.0 * A_1_0 + 1.0 * A_1_3) * (1.0 * B_3_0 + 1.0 * B_3_3);  C_0_0 += -1.0 * M28;  C_0_3 += -1.0 * M28;  C_1_0 += 1.0 * M28;  C_1_3 += 1.0 * M28;
    std::array<unsigned, 4> A28_subid = {0, 5, 2, 7};
    std::array<T,4> A28_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av28(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A28_subid, A28_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B28_subid = {10, 15};
    std::array<T,2> B28_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv28(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B28_subid, B28_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C28_subid = {0, 5, 2, 7};
    std::array<T,4> C28_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv28(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C28_subid, C28_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv28.stride(!row_major) == 1)
    {
        Av28.transpose();
        Bv28.transpose();
        Cv28.transpose();
        straprim_naive2(comm, cfg, alpha, Bv28, Av28, beta, Cv28);
    } else {
        straprim_naive2(comm, cfg, alpha, Av28, Bv28, beta, Cv28);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M28:" << std::endl;
    //print_tensor_matrix( ct );

    // M29 = (1.0 * A_0_2 + 1.0 * A_0_3 + 1.0 * A_1_2 + 1.0 * A_1_3) * (1.0 * B_3_0);  C_0_2 += -1.0 * M29;  C_0_3 += 1.0 * M29;  C_1_2 += 1.0 * M29;  C_1_3 += -1.0 * M29;
    std::array<unsigned, 4> A29_subid = {4, 5, 6, 7};
    std::array<T,4> A29_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av29(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A29_subid, A29_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 1> B29_subid = {10};
    std::array<T,1> B29_coeff_list = {1.0};
    stra_tensor_view<T,1> Bv29(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B29_subid, B29_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C29_subid = {4, 5, 6, 7};
    std::array<T,4> C29_coeff_list = {-1.0, 1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Cv29(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C29_subid, C29_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv29.stride(!row_major) == 1)
    {
        Av29.transpose();
        Bv29.transpose();
        Cv29.transpose();
        straprim_naive2(comm, cfg, alpha, Bv29, Av29, beta, Cv29);
    } else {
        straprim_naive2(comm, cfg, alpha, Av29, Bv29, beta, Cv29);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M29:" << std::endl;
    //print_tensor_matrix( ct );

    // M30 = (1.0 * A_0_0 + 1.0 * A_1_0) * (1.0 * B_3_1 + -1.0 * B_3_3);  C_0_1 += -1.0 * M30;  C_0_3 += -1.0 * M30;  C_1_1 += 1.0 * M30;  C_1_3 += 1.0 * M30;
    std::array<unsigned, 2> A30_subid = {0, 2};
    std::array<T,2> A30_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av30(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A30_subid, A30_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B30_subid = {11, 15};
    std::array<T,2> B30_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Bv30(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B30_subid, B30_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C30_subid = {1, 5, 3, 7};
    std::array<T,4> C30_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv30(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C30_subid, C30_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv30.stride(!row_major) == 1)
    {
        Av30.transpose();
        Bv30.transpose();
        Cv30.transpose();
        straprim_naive2(comm, cfg, alpha, Bv30, Av30, beta, Cv30);
    } else {
        straprim_naive2(comm, cfg, alpha, Av30, Bv30, beta, Cv30);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M30:" << std::endl;
    //print_tensor_matrix( ct );

    // M31 = (1.0 * A_0_3 + 1.0 * A_1_3) * (-1.0 * B_3_0 + 1.0 * B_3_2);  C_0_0 += -1.0 * M31;  C_0_2 += -1.0 * M31;  C_1_0 += 1.0 * M31;  C_1_2 += 1.0 * M31;
    std::array<unsigned, 2> A31_subid = {5, 7};
    std::array<T,2> A31_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av31(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A31_subid, A31_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B31_subid = {10, 14};
    std::array<T,2> B31_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Bv31(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B31_subid, B31_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C31_subid = {0, 4, 2, 6};
    std::array<T,4> C31_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv31(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C31_subid, C31_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv31.stride(!row_major) == 1)
    {
        Av31.transpose();
        Bv31.transpose();
        Cv31.transpose();
        straprim_naive2(comm, cfg, alpha, Bv31, Av31, beta, Cv31);
    } else {
        straprim_naive2(comm, cfg, alpha, Av31, Bv31, beta, Cv31);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M31:" << std::endl;
    //print_tensor_matrix( ct );

    // M32 = (1.0 * A_0_0 + 1.0 * A_0_1 + 1.0 * A_1_0 + 1.0 * A_1_1) * (1.0 * B_3_3);  C_0_0 += 1.0 * M32;  C_0_1 += -1.0 * M32;  C_1_0 += -1.0 * M32;  C_1_1 += 1.0 * M32;
    std::array<unsigned, 4> A32_subid = {0, 1, 2, 3};
    std::array<T,4> A32_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av32(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A32_subid, A32_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 1> B32_subid = {15};
    std::array<T,1> B32_coeff_list = {1.0};
    stra_tensor_view<T,1> Bv32(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B32_subid, B32_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C32_subid = {0, 1, 2, 3};
    std::array<T,4> C32_coeff_list = {1.0, -1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Cv32(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C32_subid, C32_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv32.stride(!row_major) == 1)
    {
        Av32.transpose();
        Bv32.transpose();
        Cv32.transpose();
        straprim_naive2(comm, cfg, alpha, Bv32, Av32, beta, Cv32);
    } else {
        straprim_naive2(comm, cfg, alpha, Av32, Bv32, beta, Cv32);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M32:" << std::endl;
    //print_tensor_matrix( ct );

    // M33 = (-1.0 * A_0_0 + 1.0 * A_0_2 + -1.0 * A_1_0 + 1.0 * A_1_2) * (1.0 * B_3_0 + 1.0 * B_3_1);  C_0_3 += -1.0 * M33;  C_1_3 += 1.0 * M33;
    std::array<unsigned, 4> A33_subid = {0, 4, 2, 6};
    std::array<T,4> A33_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Av33(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A33_subid, A33_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B33_subid = {10, 11};
    std::array<T,2> B33_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv33(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B33_subid, B33_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C33_subid = {5, 7};
    std::array<T,2> C33_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Cv33(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C33_subid, C33_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv33.stride(!row_major) == 1)
    {
        Av33.transpose();
        Bv33.transpose();
        Cv33.transpose();
        straprim_naive2(comm, cfg, alpha, Bv33, Av33, beta, Cv33);
    } else {
        straprim_naive2(comm, cfg, alpha, Av33, Bv33, beta, Cv33);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M33:" << std::endl;
    //print_tensor_matrix( ct );

    // M34 = (1.0 * A_0_1 + -1.0 * A_0_3 + 1.0 * A_1_1 + -1.0 * A_1_3) * (1.0 * B_3_2 + 1.0 * B_3_3);  C_0_0 += -1.0 * M34;  C_1_0 += 1.0 * M34;
    std::array<unsigned, 4> A34_subid = {1, 5, 3, 7};
    std::array<T,4> A34_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Av34(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A34_subid, A34_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B34_subid = {14, 15};
    std::array<T,2> B34_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv34(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B34_subid, B34_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C34_subid = {0, 2};
    std::array<T,2> C34_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Cv34(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C34_subid, C34_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv34.stride(!row_major) == 1)
    {
        Av34.transpose();
        Bv34.transpose();
        Cv34.transpose();
        straprim_naive2(comm, cfg, alpha, Bv34, Av34, beta, Cv34);
    } else {
        straprim_naive2(comm, cfg, alpha, Av34, Bv34, beta, Cv34);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M34:" << std::endl;
    //print_tensor_matrix( ct );

    // M35 = (-1.0 * A_0_0 + -1.0 * A_0_3 + 1.0 * A_2_0 + 1.0 * A_2_3) * (1.0 * B_0_0 + 1.0 * B_0_3 + 1.0 * B_1_0 + 1.0 * B_1_3);  C_3_0 += 1.0 * M35;  C_3_3 += 1.0 * M35;
    std::array<unsigned, 4> A35_subid = {0, 5, 8, 13};
    std::array<T,4> A35_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av35(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A35_subid, A35_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B35_subid = {0, 5, 2, 7};
    std::array<T,4> B35_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv35(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B35_subid, B35_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C35_subid = {10, 15};
    std::array<T,2> C35_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv35(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C35_subid, C35_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv35.stride(!row_major) == 1)
    {
        Av35.transpose();
        Bv35.transpose();
        Cv35.transpose();
        straprim_naive2(comm, cfg, alpha, Bv35, Av35, beta, Cv35);
    } else {
        straprim_naive2(comm, cfg, alpha, Av35, Bv35, beta, Cv35);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M35:" << std::endl;
    //print_tensor_matrix( ct );

    // M36 = (-1.0 * A_0_2 + -1.0 * A_0_3 + 1.0 * A_2_2 + 1.0 * A_2_3) * (1.0 * B_0_0 + 1.0 * B_1_0);  C_3_2 += 1.0 * M36;  C_3_3 += -1.0 * M36;
    std::array<unsigned, 4> A36_subid = {4, 5, 12, 13};
    std::array<T,4> A36_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av36(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A36_subid, A36_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B36_subid = {0, 2};
    std::array<T,2> B36_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv36(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B36_subid, B36_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C36_subid = {14, 15};
    std::array<T,2> C36_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Cv36(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C36_subid, C36_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv36.stride(!row_major) == 1)
    {
        Av36.transpose();
        Bv36.transpose();
        Cv36.transpose();
        straprim_naive2(comm, cfg, alpha, Bv36, Av36, beta, Cv36);
    } else {
        straprim_naive2(comm, cfg, alpha, Av36, Bv36, beta, Cv36);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M36:" << std::endl;
    //print_tensor_matrix( ct );

    // M37 = (-1.0 * A_0_0 + 1.0 * A_2_0) * (1.0 * B_0_1 + -1.0 * B_0_3 + 1.0 * B_1_1 + -1.0 * B_1_3);  C_3_1 += 1.0 * M37;  C_3_3 += 1.0 * M37;
    std::array<unsigned, 2> A37_subid = {0, 8};
    std::array<T,2> A37_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Av37(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A37_subid, A37_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B37_subid = {1, 5, 3, 7};
    std::array<T,4> B37_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Bv37(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B37_subid, B37_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C37_subid = {11, 15};
    std::array<T,2> C37_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv37(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C37_subid, C37_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv37.stride(!row_major) == 1)
    {
        Av37.transpose();
        Bv37.transpose();
        Cv37.transpose();
        straprim_naive2(comm, cfg, alpha, Bv37, Av37, beta, Cv37);
    } else {
        straprim_naive2(comm, cfg, alpha, Av37, Bv37, beta, Cv37);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M37:" << std::endl;
    //print_tensor_matrix( ct );

    // M38 = (-1.0 * A_0_3 + 1.0 * A_2_3) * (-1.0 * B_0_0 + 1.0 * B_0_2 + -1.0 * B_1_0 + 1.0 * B_1_2);  C_3_0 += 1.0 * M38;  C_3_2 += 1.0 * M38;
    std::array<unsigned, 2> A38_subid = {5, 13};
    std::array<T,2> A38_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Av38(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A38_subid, A38_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B38_subid = {0, 4, 2, 6};
    std::array<T,4> B38_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Bv38(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B38_subid, B38_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C38_subid = {10, 14};
    std::array<T,2> C38_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv38(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C38_subid, C38_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv38.stride(!row_major) == 1)
    {
        Av38.transpose();
        Bv38.transpose();
        Cv38.transpose();
        straprim_naive2(comm, cfg, alpha, Bv38, Av38, beta, Cv38);
    } else {
        straprim_naive2(comm, cfg, alpha, Av38, Bv38, beta, Cv38);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M38:" << std::endl;
    //print_tensor_matrix( ct );

    // M39 = (-1.0 * A_0_0 + -1.0 * A_0_1 + 1.0 * A_2_0 + 1.0 * A_2_1) * (1.0 * B_0_3 + 1.0 * B_1_3);  C_3_0 += -1.0 * M39;  C_3_1 += 1.0 * M39;
    std::array<unsigned, 4> A39_subid = {0, 1, 8, 9};
    std::array<T,4> A39_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av39(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A39_subid, A39_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B39_subid = {5, 7};
    std::array<T,2> B39_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv39(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B39_subid, B39_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C39_subid = {10, 11};
    std::array<T,2> C39_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Cv39(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C39_subid, C39_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv39.stride(!row_major) == 1)
    {
        Av39.transpose();
        Bv39.transpose();
        Cv39.transpose();
        straprim_naive2(comm, cfg, alpha, Bv39, Av39, beta, Cv39);
    } else {
        straprim_naive2(comm, cfg, alpha, Av39, Bv39, beta, Cv39);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M39:" << std::endl;
    //print_tensor_matrix( ct );

    // M40 = (1.0 * A_0_0 + -1.0 * A_0_2 + -1.0 * A_2_0 + 1.0 * A_2_2) * (1.0 * B_0_0 + 1.0 * B_0_1 + 1.0 * B_1_0 + 1.0 * B_1_1);  C_3_3 += 1.0 * M40;
    std::array<unsigned, 4> A40_subid = {0, 4, 8, 12};
    std::array<T,4> A40_coeff_list = {1.0, -1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Av40(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A40_subid, A40_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B40_subid = {0, 1, 2, 3};
    std::array<T,4> B40_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv40(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B40_subid, B40_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 1> C40_subid = {15};
    std::array<T,1> C40_coeff_list = {1.0};
    stra_tensor_view<T,1> Cv40(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C40_subid, C40_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv40.stride(!row_major) == 1)
    {
        Av40.transpose();
        Bv40.transpose();
        Cv40.transpose();
        straprim_naive2(comm, cfg, alpha, Bv40, Av40, beta, Cv40);
    } else {
        straprim_naive2(comm, cfg, alpha, Av40, Bv40, beta, Cv40);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M40:" << std::endl;
    //print_tensor_matrix( ct );

    // M41 = (-1.0 * A_0_1 + 1.0 * A_0_3 + 1.0 * A_2_1 + -1.0 * A_2_3) * (1.0 * B_0_2 + 1.0 * B_0_3 + 1.0 * B_1_2 + 1.0 * B_1_3);  C_3_0 += 1.0 * M41;
    std::array<unsigned, 4> A41_subid = {1, 5, 9, 13};
    std::array<T,4> A41_coeff_list = {-1.0, 1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Av41(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A41_subid, A41_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B41_subid = {4, 5, 6, 7};
    std::array<T,4> B41_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv41(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B41_subid, B41_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 1> C41_subid = {10};
    std::array<T,1> C41_coeff_list = {1.0};
    stra_tensor_view<T,1> Cv41(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C41_subid, C41_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv41.stride(!row_major) == 1)
    {
        Av41.transpose();
        Bv41.transpose();
        Cv41.transpose();
        straprim_naive2(comm, cfg, alpha, Bv41, Av41, beta, Cv41);
    } else {
        straprim_naive2(comm, cfg, alpha, Av41, Bv41, beta, Cv41);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M41:" << std::endl;
    //print_tensor_matrix( ct );

    // M42 = (1.0 * A_1_0 + 1.0 * A_1_3 + -1.0 * A_3_0 + -1.0 * A_3_3) * (1.0 * B_2_0 + 1.0 * B_2_3 + 1.0 * B_3_0 + 1.0 * B_3_3);  C_0_0 += 1.0 * M42;  C_0_3 += 1.0 * M42;
    std::array<unsigned, 4> A42_subid = {2, 7, 10, 15};
    std::array<T,4> A42_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Av42(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A42_subid, A42_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B42_subid = {8, 13, 10, 15};
    std::array<T,4> B42_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv42(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B42_subid, B42_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C42_subid = {0, 5};
    std::array<T,2> C42_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv42(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C42_subid, C42_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv42.stride(!row_major) == 1)
    {
        Av42.transpose();
        Bv42.transpose();
        Cv42.transpose();
        straprim_naive2(comm, cfg, alpha, Bv42, Av42, beta, Cv42);
    } else {
        straprim_naive2(comm, cfg, alpha, Av42, Bv42, beta, Cv42);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M42:" << std::endl;
    //print_tensor_matrix( ct );

    // M43 = (1.0 * A_1_2 + 1.0 * A_1_3 + -1.0 * A_3_2 + -1.0 * A_3_3) * (1.0 * B_2_0 + 1.0 * B_3_0);  C_0_2 += 1.0 * M43;  C_0_3 += -1.0 * M43;
    std::array<unsigned, 4> A43_subid = {6, 7, 14, 15};
    std::array<T,4> A43_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Av43(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A43_subid, A43_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B43_subid = {8, 10};
    std::array<T,2> B43_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv43(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B43_subid, B43_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C43_subid = {4, 5};
    std::array<T,2> C43_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Cv43(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C43_subid, C43_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv43.stride(!row_major) == 1)
    {
        Av43.transpose();
        Bv43.transpose();
        Cv43.transpose();
        straprim_naive2(comm, cfg, alpha, Bv43, Av43, beta, Cv43);
    } else {
        straprim_naive2(comm, cfg, alpha, Av43, Bv43, beta, Cv43);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M43:" << std::endl;
    //print_tensor_matrix( ct );

    // M44 = (1.0 * A_1_0 + -1.0 * A_3_0) * (1.0 * B_2_1 + -1.0 * B_2_3 + 1.0 * B_3_1 + -1.0 * B_3_3);  C_0_1 += 1.0 * M44;  C_0_3 += 1.0 * M44;
    std::array<unsigned, 2> A44_subid = {2, 10};
    std::array<T,2> A44_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Av44(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A44_subid, A44_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B44_subid = {9, 13, 11, 15};
    std::array<T,4> B44_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Bv44(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B44_subid, B44_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C44_subid = {1, 5};
    std::array<T,2> C44_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv44(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C44_subid, C44_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv44.stride(!row_major) == 1)
    {
        Av44.transpose();
        Bv44.transpose();
        Cv44.transpose();
        straprim_naive2(comm, cfg, alpha, Bv44, Av44, beta, Cv44);
    } else {
        straprim_naive2(comm, cfg, alpha, Av44, Bv44, beta, Cv44);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M44:" << std::endl;
    //print_tensor_matrix( ct );

    // M45 = (1.0 * A_1_3 + -1.0 * A_3_3) * (-1.0 * B_2_0 + 1.0 * B_2_2 + -1.0 * B_3_0 + 1.0 * B_3_2);  C_0_0 += 1.0 * M45;  C_0_2 += 1.0 * M45;
    std::array<unsigned, 2> A45_subid = {7, 15};
    std::array<T,2> A45_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Av45(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A45_subid, A45_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B45_subid = {8, 12, 10, 14};
    std::array<T,4> B45_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Bv45(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B45_subid, B45_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C45_subid = {0, 4};
    std::array<T,2> C45_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv45(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C45_subid, C45_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv45.stride(!row_major) == 1)
    {
        Av45.transpose();
        Bv45.transpose();
        Cv45.transpose();
        straprim_naive2(comm, cfg, alpha, Bv45, Av45, beta, Cv45);
    } else {
        straprim_naive2(comm, cfg, alpha, Av45, Bv45, beta, Cv45);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M45:" << std::endl;
    //print_tensor_matrix( ct );

    // M46 = (1.0 * A_1_0 + 1.0 * A_1_1 + -1.0 * A_3_0 + -1.0 * A_3_1) * (1.0 * B_2_3 + 1.0 * B_3_3);  C_0_0 += -1.0 * M46;  C_0_1 += 1.0 * M46;
    std::array<unsigned, 4> A46_subid = {2, 3, 10, 11};
    std::array<T,4> A46_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Av46(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A46_subid, A46_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B46_subid = {13, 15};
    std::array<T,2> B46_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv46(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B46_subid, B46_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C46_subid = {0, 1};
    std::array<T,2> C46_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Cv46(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C46_subid, C46_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv46.stride(!row_major) == 1)
    {
        Av46.transpose();
        Bv46.transpose();
        Cv46.transpose();
        straprim_naive2(comm, cfg, alpha, Bv46, Av46, beta, Cv46);
    } else {
        straprim_naive2(comm, cfg, alpha, Av46, Bv46, beta, Cv46);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M46:" << std::endl;
    //print_tensor_matrix( ct );

    // M47 = (-1.0 * A_1_0 + 1.0 * A_1_2 + 1.0 * A_3_0 + -1.0 * A_3_2) * (1.0 * B_2_0 + 1.0 * B_2_1 + 1.0 * B_3_0 + 1.0 * B_3_1);  C_0_3 += 1.0 * M47;
    std::array<unsigned, 4> A47_subid = {2, 6, 10, 14};
    std::array<T,4> A47_coeff_list = {-1.0, 1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Av47(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A47_subid, A47_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B47_subid = {8, 9, 10, 11};
    std::array<T,4> B47_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv47(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B47_subid, B47_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 1> C47_subid = {5};
    std::array<T,1> C47_coeff_list = {1.0};
    stra_tensor_view<T,1> Cv47(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C47_subid, C47_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv47.stride(!row_major) == 1)
    {
        Av47.transpose();
        Bv47.transpose();
        Cv47.transpose();
        straprim_naive2(comm, cfg, alpha, Bv47, Av47, beta, Cv47);
    } else {
        straprim_naive2(comm, cfg, alpha, Av47, Bv47, beta, Cv47);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M47:" << std::endl;
    //print_tensor_matrix( ct );

    // M48 = (1.0 * A_1_1 + -1.0 * A_1_3 + -1.0 * A_3_1 + 1.0 * A_3_3) * (1.0 * B_2_2 + 1.0 * B_2_3 + 1.0 * B_3_2 + 1.0 * B_3_3);  C_0_0 += 1.0 * M48;
    std::array<unsigned, 4> A48_subid = {3, 7, 11, 15};
    std::array<T,4> A48_coeff_list = {1.0, -1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Av48(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A48_subid, A48_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B48_subid = {12, 13, 14, 15};
    std::array<T,4> B48_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv48(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B48_subid, B48_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 1> C48_subid = {0};
    std::array<T,1> C48_coeff_list = {1.0};
    stra_tensor_view<T,1> Cv48(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C48_subid, C48_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv48.stride(!row_major) == 1)
    {
        Av48.transpose();
        Bv48.transpose();
        Cv48.transpose();
        straprim_naive2(comm, cfg, alpha, Bv48, Av48, beta, Cv48);
    } else {
        straprim_naive2(comm, cfg, alpha, Av48, Bv48, beta, Cv48);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M48:" << std::endl;
    //print_tensor_matrix( ct );



    //std::cout << "stra_internal/stra_mult_2level_A:" << std::endl;
    //print_tensor_matrix( at );

    //std::cout << "stra_internal/stra_mult_2level_B:" << std::endl;
    //print_tensor_matrix( bt );

    //std::cout << "stra_internal/stra_mult_2level_M6:" << std::endl;
    //print_tensor_matrix( ct );

}

#define INSTANTIATE_CONTRACT_BLIS_NAIVE(T) \
template void stra_contract_2level_blis_naive(const communicator& comm, const config& cfg, \
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
void stra_contract_2level_blis_ab(const communicator& comm, const config& cfg,
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
    //std::cout << "Enter stra_internal/3t/stra_mult_2level/stra_contract_2level_blis_ab\n" << std::endl;

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
;
    auto my_stride_B_AB = stl_ext::permuted(stride_B_AB, reorder_AB);
    auto my_stride_B_BC = stl_ext::permuted(stride_B_BC, reorder_BC);
;
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

    //StraTensorGEMM stra_gemm;

    //len_type m = ct.length(0);
    //len_type n = ct.length(1);
    //len_type k = at.length(1);

    //int nt = comm.num_threads();
    //auto tc = make_gemm_thread_config<T>(cfg, nt, m, n, k);
    //step<0>(stra_gemm).distribute = tc.jc_nt;
    //step<4>(stra_gemm).distribute = tc.ic_nt;
    //step<8>(stra_gemm).distribute = tc.jr_nt;
    //step<9>(stra_gemm).distribute = tc.ir_nt;

    //const len_type ms=m/2, ks=k/2, ns=n/2;

    const std::array<unsigned,2> A_divisor={4,4};
    const std::array<unsigned,2> B_divisor={4,4};
    const std::array<unsigned,2> C_divisor={4,4};

    // M0 = (1.0 * A_0_0 + 1.0 * A_0_3 + 1.0 * A_3_0 + 1.0 * A_3_3) * (1.0 * B_0_0 + 1.0 * B_0_3 + 1.0 * B_3_0 + 1.0 * B_3_3);  C_0_0 += 1.0 * M0;  C_0_3 += 1.0 * M0;  C_3_0 += 1.0 * M0;  C_3_3 += 1.0 * M0;
    std::array<unsigned, 4> A0_subid = {0, 5, 10, 15};
    std::array<T,4> A0_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av0(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A0_subid, A0_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B0_subid = {0, 5, 10, 15};
    std::array<T,4> B0_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv0(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B0_subid, B0_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C0_subid = {0, 5, 10, 15};
    std::array<T,4> C0_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv0(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C0_subid, C0_coeff_list, my_stride_C_AC, my_stride_C_BC);
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

    // M1 = (1.0 * A_0_2 + 1.0 * A_0_3 + 1.0 * A_3_2 + 1.0 * A_3_3) * (1.0 * B_0_0 + 1.0 * B_3_0);  C_0_2 += 1.0 * M1;  C_0_3 += -1.0 * M1;  C_3_2 += 1.0 * M1;  C_3_3 += -1.0 * M1;
    std::array<unsigned, 4> A1_subid = {4, 5, 14, 15};
    std::array<T,4> A1_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av1(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A1_subid, A1_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B1_subid = {0, 10};
    std::array<T,2> B1_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv1(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B1_subid, B1_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C1_subid = {4, 5, 14, 15};
    std::array<T,4> C1_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Cv1(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C1_subid, C1_coeff_list, my_stride_C_AC, my_stride_C_BC);
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

    // M2 = (1.0 * A_0_0 + 1.0 * A_3_0) * (1.0 * B_0_1 + -1.0 * B_0_3 + 1.0 * B_3_1 + -1.0 * B_3_3);  C_0_1 += 1.0 * M2;  C_0_3 += 1.0 * M2;  C_3_1 += 1.0 * M2;  C_3_3 += 1.0 * M2;
    std::array<unsigned, 2> A2_subid = {0, 10};
    std::array<T,2> A2_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av2(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A2_subid, A2_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B2_subid = {1, 5, 11, 15};
    std::array<T,4> B2_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Bv2(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B2_subid, B2_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C2_subid = {1, 5, 11, 15};
    std::array<T,4> C2_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv2(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C2_subid, C2_coeff_list, my_stride_C_AC, my_stride_C_BC);
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

    // M3 = (1.0 * A_0_3 + 1.0 * A_3_3) * (-1.0 * B_0_0 + 1.0 * B_0_2 + -1.0 * B_3_0 + 1.0 * B_3_2);  C_0_0 += 1.0 * M3;  C_0_2 += 1.0 * M3;  C_3_0 += 1.0 * M3;  C_3_2 += 1.0 * M3;
    std::array<unsigned, 2> A3_subid = {5, 15};
    std::array<T,2> A3_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av3(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A3_subid, A3_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B3_subid = {0, 4, 10, 14};
    std::array<T,4> B3_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Bv3(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B3_subid, B3_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C3_subid = {0, 4, 10, 14};
    std::array<T,4> C3_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv3(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C3_subid, C3_coeff_list, my_stride_C_AC, my_stride_C_BC);
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

    // M4 = (1.0 * A_0_0 + 1.0 * A_0_1 + 1.0 * A_3_0 + 1.0 * A_3_1) * (1.0 * B_0_3 + 1.0 * B_3_3);  C_0_0 += -1.0 * M4;  C_0_1 += 1.0 * M4;  C_3_0 += -1.0 * M4;  C_3_1 += 1.0 * M4;
    std::array<unsigned, 4> A4_subid = {0, 1, 10, 11};
    std::array<T,4> A4_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av4(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A4_subid, A4_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B4_subid = {5, 15};
    std::array<T,2> B4_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv4(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B4_subid, B4_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C4_subid = {0, 1, 10, 11};
    std::array<T,4> C4_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Cv4(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C4_subid, C4_coeff_list, my_stride_C_AC, my_stride_C_BC);
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

    // M5 = (-1.0 * A_0_0 + 1.0 * A_0_2 + -1.0 * A_3_0 + 1.0 * A_3_2) * (1.0 * B_0_0 + 1.0 * B_0_1 + 1.0 * B_3_0 + 1.0 * B_3_1);  C_0_3 += 1.0 * M5;  C_3_3 += 1.0 * M5;
    std::array<unsigned, 4> A5_subid = {0, 4, 10, 14};
    std::array<T,4> A5_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Av5(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A5_subid, A5_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B5_subid = {0, 1, 10, 11};
    std::array<T,4> B5_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv5(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B5_subid, B5_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C5_subid = {5, 15};
    std::array<T,2> C5_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv5(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C5_subid, C5_coeff_list, my_stride_C_AC, my_stride_C_BC);
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

    // M6 = (1.0 * A_0_1 + -1.0 * A_0_3 + 1.0 * A_3_1 + -1.0 * A_3_3) * (1.0 * B_0_2 + 1.0 * B_0_3 + 1.0 * B_3_2 + 1.0 * B_3_3);  C_0_0 += 1.0 * M6;  C_3_0 += 1.0 * M6;
    std::array<unsigned, 4> A6_subid = {1, 5, 11, 15};
    std::array<T,4> A6_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Av6(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A6_subid, A6_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B6_subid = {4, 5, 14, 15};
    std::array<T,4> B6_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv6(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B6_subid, B6_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C6_subid = {0, 10};
    std::array<T,2> C6_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv6(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C6_subid, C6_coeff_list, my_stride_C_AC, my_stride_C_BC);
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

    // M7 = (1.0 * A_2_0 + 1.0 * A_2_3 + 1.0 * A_3_0 + 1.0 * A_3_3) * (1.0 * B_0_0 + 1.0 * B_0_3);  C_2_0 += 1.0 * M7;  C_2_3 += 1.0 * M7;  C_3_0 += -1.0 * M7;  C_3_3 += -1.0 * M7;
    std::array<unsigned, 4> A7_subid = {8, 13, 10, 15};
    std::array<T,4> A7_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av7(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A7_subid, A7_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B7_subid = {0, 5};
    std::array<T,2> B7_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv7(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B7_subid, B7_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C7_subid = {8, 13, 10, 15};
    std::array<T,4> C7_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Cv7(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C7_subid, C7_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv7.stride(!row_major) == 1)
    {
        Av7.transpose();
        Bv7.transpose();
        Cv7.transpose();
        straprim_ab2(comm, cfg, alpha, Bv7, Av7, beta, Cv7);
    } else {
        straprim_ab2(comm, cfg, alpha, Av7, Bv7, beta, Cv7);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M7:" << std::endl;
    //print_tensor_matrix( ct );

    // M8 = (1.0 * A_2_2 + 1.0 * A_2_3 + 1.0 * A_3_2 + 1.0 * A_3_3) * (1.0 * B_0_0);  C_2_2 += 1.0 * M8;  C_2_3 += -1.0 * M8;  C_3_2 += -1.0 * M8;  C_3_3 += 1.0 * M8;
    std::array<unsigned, 4> A8_subid = {12, 13, 14, 15};
    std::array<T,4> A8_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av8(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A8_subid, A8_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 1> B8_subid = {0};
    std::array<T,1> B8_coeff_list = {1.0};
    stra_tensor_view<T,1> Bv8(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B8_subid, B8_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C8_subid = {12, 13, 14, 15};
    std::array<T,4> C8_coeff_list = {1.0, -1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Cv8(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C8_subid, C8_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv8.stride(!row_major) == 1)
    {
        Av8.transpose();
        Bv8.transpose();
        Cv8.transpose();
        straprim_ab2(comm, cfg, alpha, Bv8, Av8, beta, Cv8);
    } else {
        straprim_ab2(comm, cfg, alpha, Av8, Bv8, beta, Cv8);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M8:" << std::endl;
    //print_tensor_matrix( ct );

    // M9 = (1.0 * A_2_0 + 1.0 * A_3_0) * (1.0 * B_0_1 + -1.0 * B_0_3);  C_2_1 += 1.0 * M9;  C_2_3 += 1.0 * M9;  C_3_1 += -1.0 * M9;  C_3_3 += -1.0 * M9;
    std::array<unsigned, 2> A9_subid = {8, 10};
    std::array<T,2> A9_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av9(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A9_subid, A9_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B9_subid = {1, 5};
    std::array<T,2> B9_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Bv9(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B9_subid, B9_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C9_subid = {9, 13, 11, 15};
    std::array<T,4> C9_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Cv9(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C9_subid, C9_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv9.stride(!row_major) == 1)
    {
        Av9.transpose();
        Bv9.transpose();
        Cv9.transpose();
        straprim_ab2(comm, cfg, alpha, Bv9, Av9, beta, Cv9);
    } else {
        straprim_ab2(comm, cfg, alpha, Av9, Bv9, beta, Cv9);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M9:" << std::endl;
    //print_tensor_matrix( ct );

    // M10 = (1.0 * A_2_3 + 1.0 * A_3_3) * (-1.0 * B_0_0 + 1.0 * B_0_2);  C_2_0 += 1.0 * M10;  C_2_2 += 1.0 * M10;  C_3_0 += -1.0 * M10;  C_3_2 += -1.0 * M10;
    std::array<unsigned, 2> A10_subid = {13, 15};
    std::array<T,2> A10_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av10(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A10_subid, A10_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B10_subid = {0, 4};
    std::array<T,2> B10_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Bv10(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B10_subid, B10_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C10_subid = {8, 12, 10, 14};
    std::array<T,4> C10_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Cv10(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C10_subid, C10_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv10.stride(!row_major) == 1)
    {
        Av10.transpose();
        Bv10.transpose();
        Cv10.transpose();
        straprim_ab2(comm, cfg, alpha, Bv10, Av10, beta, Cv10);
    } else {
        straprim_ab2(comm, cfg, alpha, Av10, Bv10, beta, Cv10);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M10:" << std::endl;
    //print_tensor_matrix( ct );

    // M11 = (1.0 * A_2_0 + 1.0 * A_2_1 + 1.0 * A_3_0 + 1.0 * A_3_1) * (1.0 * B_0_3);  C_2_0 += -1.0 * M11;  C_2_1 += 1.0 * M11;  C_3_0 += 1.0 * M11;  C_3_1 += -1.0 * M11;
    std::array<unsigned, 4> A11_subid = {8, 9, 10, 11};
    std::array<T,4> A11_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av11(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A11_subid, A11_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 1> B11_subid = {5};
    std::array<T,1> B11_coeff_list = {1.0};
    stra_tensor_view<T,1> Bv11(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B11_subid, B11_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C11_subid = {8, 9, 10, 11};
    std::array<T,4> C11_coeff_list = {-1.0, 1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Cv11(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C11_subid, C11_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv11.stride(!row_major) == 1)
    {
        Av11.transpose();
        Bv11.transpose();
        Cv11.transpose();
        straprim_ab2(comm, cfg, alpha, Bv11, Av11, beta, Cv11);
    } else {
        straprim_ab2(comm, cfg, alpha, Av11, Bv11, beta, Cv11);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M11:" << std::endl;
    //print_tensor_matrix( ct );

    // M12 = (-1.0 * A_2_0 + 1.0 * A_2_2 + -1.0 * A_3_0 + 1.0 * A_3_2) * (1.0 * B_0_0 + 1.0 * B_0_1);  C_2_3 += 1.0 * M12;  C_3_3 += -1.0 * M12;
    std::array<unsigned, 4> A12_subid = {8, 12, 10, 14};
    std::array<T,4> A12_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Av12(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A12_subid, A12_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B12_subid = {0, 1};
    std::array<T,2> B12_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv12(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B12_subid, B12_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C12_subid = {13, 15};
    std::array<T,2> C12_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Cv12(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C12_subid, C12_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv12.stride(!row_major) == 1)
    {
        Av12.transpose();
        Bv12.transpose();
        Cv12.transpose();
        straprim_ab2(comm, cfg, alpha, Bv12, Av12, beta, Cv12);
    } else {
        straprim_ab2(comm, cfg, alpha, Av12, Bv12, beta, Cv12);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M12:" << std::endl;
    //print_tensor_matrix( ct );

    // M13 = (1.0 * A_2_1 + -1.0 * A_2_3 + 1.0 * A_3_1 + -1.0 * A_3_3) * (1.0 * B_0_2 + 1.0 * B_0_3);  C_2_0 += 1.0 * M13;  C_3_0 += -1.0 * M13;
    std::array<unsigned, 4> A13_subid = {9, 13, 11, 15};
    std::array<T,4> A13_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Av13(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A13_subid, A13_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B13_subid = {4, 5};
    std::array<T,2> B13_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv13(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B13_subid, B13_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C13_subid = {8, 10};
    std::array<T,2> C13_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Cv13(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C13_subid, C13_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv13.stride(!row_major) == 1)
    {
        Av13.transpose();
        Bv13.transpose();
        Cv13.transpose();
        straprim_ab2(comm, cfg, alpha, Bv13, Av13, beta, Cv13);
    } else {
        straprim_ab2(comm, cfg, alpha, Av13, Bv13, beta, Cv13);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M13:" << std::endl;
    //print_tensor_matrix( ct );

    // M14 = (1.0 * A_0_0 + 1.0 * A_0_3) * (1.0 * B_1_0 + 1.0 * B_1_3 + -1.0 * B_3_0 + -1.0 * B_3_3);  C_1_0 += 1.0 * M14;  C_1_3 += 1.0 * M14;  C_3_0 += 1.0 * M14;  C_3_3 += 1.0 * M14;
    std::array<unsigned, 2> A14_subid = {0, 5};
    std::array<T,2> A14_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av14(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A14_subid, A14_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B14_subid = {2, 7, 10, 15};
    std::array<T,4> B14_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Bv14(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B14_subid, B14_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C14_subid = {2, 7, 10, 15};
    std::array<T,4> C14_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv14(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C14_subid, C14_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv14.stride(!row_major) == 1)
    {
        Av14.transpose();
        Bv14.transpose();
        Cv14.transpose();
        straprim_ab2(comm, cfg, alpha, Bv14, Av14, beta, Cv14);
    } else {
        straprim_ab2(comm, cfg, alpha, Av14, Bv14, beta, Cv14);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M14:" << std::endl;
    //print_tensor_matrix( ct );

    // M15 = (1.0 * A_0_2 + 1.0 * A_0_3) * (1.0 * B_1_0 + -1.0 * B_3_0);  C_1_2 += 1.0 * M15;  C_1_3 += -1.0 * M15;  C_3_2 += 1.0 * M15;  C_3_3 += -1.0 * M15;
    std::array<unsigned, 2> A15_subid = {4, 5};
    std::array<T,2> A15_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av15(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A15_subid, A15_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B15_subid = {2, 10};
    std::array<T,2> B15_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Bv15(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B15_subid, B15_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C15_subid = {6, 7, 14, 15};
    std::array<T,4> C15_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Cv15(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C15_subid, C15_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv15.stride(!row_major) == 1)
    {
        Av15.transpose();
        Bv15.transpose();
        Cv15.transpose();
        straprim_ab2(comm, cfg, alpha, Bv15, Av15, beta, Cv15);
    } else {
        straprim_ab2(comm, cfg, alpha, Av15, Bv15, beta, Cv15);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M15:" << std::endl;
    //print_tensor_matrix( ct );

    // M16 = (1.0 * A_0_0) * (1.0 * B_1_1 + -1.0 * B_1_3 + -1.0 * B_3_1 + 1.0 * B_3_3);  C_1_1 += 1.0 * M16;  C_1_3 += 1.0 * M16;  C_3_1 += 1.0 * M16;  C_3_3 += 1.0 * M16;
    std::array<unsigned, 1> A16_subid = {0};
    std::array<T,1> A16_coeff_list = {1.0};
    stra_tensor_view<T,1> Av16(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A16_subid, A16_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B16_subid = {3, 7, 11, 15};
    std::array<T,4> B16_coeff_list = {1.0, -1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Bv16(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B16_subid, B16_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C16_subid = {3, 7, 11, 15};
    std::array<T,4> C16_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv16(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C16_subid, C16_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv16.stride(!row_major) == 1)
    {
        Av16.transpose();
        Bv16.transpose();
        Cv16.transpose();
        straprim_ab2(comm, cfg, alpha, Bv16, Av16, beta, Cv16);
    } else {
        straprim_ab2(comm, cfg, alpha, Av16, Bv16, beta, Cv16);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M16:" << std::endl;
    //print_tensor_matrix( ct );

    // M17 = (1.0 * A_0_3) * (-1.0 * B_1_0 + 1.0 * B_1_2 + 1.0 * B_3_0 + -1.0 * B_3_2);  C_1_0 += 1.0 * M17;  C_1_2 += 1.0 * M17;  C_3_0 += 1.0 * M17;  C_3_2 += 1.0 * M17;
    std::array<unsigned, 1> A17_subid = {5};
    std::array<T,1> A17_coeff_list = {1.0};
    stra_tensor_view<T,1> Av17(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A17_subid, A17_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B17_subid = {2, 6, 10, 14};
    std::array<T,4> B17_coeff_list = {-1.0, 1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Bv17(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B17_subid, B17_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C17_subid = {2, 6, 10, 14};
    std::array<T,4> C17_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv17(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C17_subid, C17_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv17.stride(!row_major) == 1)
    {
        Av17.transpose();
        Bv17.transpose();
        Cv17.transpose();
        straprim_ab2(comm, cfg, alpha, Bv17, Av17, beta, Cv17);
    } else {
        straprim_ab2(comm, cfg, alpha, Av17, Bv17, beta, Cv17);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M17:" << std::endl;
    //print_tensor_matrix( ct );

    // M18 = (1.0 * A_0_0 + 1.0 * A_0_1) * (1.0 * B_1_3 + -1.0 * B_3_3);  C_1_0 += -1.0 * M18;  C_1_1 += 1.0 * M18;  C_3_0 += -1.0 * M18;  C_3_1 += 1.0 * M18;
    std::array<unsigned, 2> A18_subid = {0, 1};
    std::array<T,2> A18_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av18(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A18_subid, A18_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B18_subid = {7, 15};
    std::array<T,2> B18_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Bv18(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B18_subid, B18_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C18_subid = {2, 3, 10, 11};
    std::array<T,4> C18_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Cv18(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C18_subid, C18_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv18.stride(!row_major) == 1)
    {
        Av18.transpose();
        Bv18.transpose();
        Cv18.transpose();
        straprim_ab2(comm, cfg, alpha, Bv18, Av18, beta, Cv18);
    } else {
        straprim_ab2(comm, cfg, alpha, Av18, Bv18, beta, Cv18);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M18:" << std::endl;
    //print_tensor_matrix( ct );

    // M19 = (-1.0 * A_0_0 + 1.0 * A_0_2) * (1.0 * B_1_0 + 1.0 * B_1_1 + -1.0 * B_3_0 + -1.0 * B_3_1);  C_1_3 += 1.0 * M19;  C_3_3 += 1.0 * M19;
    std::array<unsigned, 2> A19_subid = {0, 4};
    std::array<T,2> A19_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Av19(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A19_subid, A19_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B19_subid = {2, 3, 10, 11};
    std::array<T,4> B19_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Bv19(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B19_subid, B19_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C19_subid = {7, 15};
    std::array<T,2> C19_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv19(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C19_subid, C19_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv19.stride(!row_major) == 1)
    {
        Av19.transpose();
        Bv19.transpose();
        Cv19.transpose();
        straprim_ab2(comm, cfg, alpha, Bv19, Av19, beta, Cv19);
    } else {
        straprim_ab2(comm, cfg, alpha, Av19, Bv19, beta, Cv19);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M19:" << std::endl;
    //print_tensor_matrix( ct );

    // M20 = (1.0 * A_0_1 + -1.0 * A_0_3) * (1.0 * B_1_2 + 1.0 * B_1_3 + -1.0 * B_3_2 + -1.0 * B_3_3);  C_1_0 += 1.0 * M20;  C_3_0 += 1.0 * M20;
    std::array<unsigned, 2> A20_subid = {1, 5};
    std::array<T,2> A20_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Av20(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A20_subid, A20_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B20_subid = {6, 7, 14, 15};
    std::array<T,4> B20_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Bv20(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B20_subid, B20_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C20_subid = {2, 10};
    std::array<T,2> C20_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv20(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C20_subid, C20_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv20.stride(!row_major) == 1)
    {
        Av20.transpose();
        Bv20.transpose();
        Cv20.transpose();
        straprim_ab2(comm, cfg, alpha, Bv20, Av20, beta, Cv20);
    } else {
        straprim_ab2(comm, cfg, alpha, Av20, Bv20, beta, Cv20);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M20:" << std::endl;
    //print_tensor_matrix( ct );

    // M21 = (1.0 * A_3_0 + 1.0 * A_3_3) * (-1.0 * B_0_0 + -1.0 * B_0_3 + 1.0 * B_2_0 + 1.0 * B_2_3);  C_0_0 += 1.0 * M21;  C_0_3 += 1.0 * M21;  C_2_0 += 1.0 * M21;  C_2_3 += 1.0 * M21;
    std::array<unsigned, 2> A21_subid = {10, 15};
    std::array<T,2> A21_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av21(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A21_subid, A21_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B21_subid = {0, 5, 8, 13};
    std::array<T,4> B21_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv21(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B21_subid, B21_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C21_subid = {0, 5, 8, 13};
    std::array<T,4> C21_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv21(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C21_subid, C21_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv21.stride(!row_major) == 1)
    {
        Av21.transpose();
        Bv21.transpose();
        Cv21.transpose();
        straprim_ab2(comm, cfg, alpha, Bv21, Av21, beta, Cv21);
    } else {
        straprim_ab2(comm, cfg, alpha, Av21, Bv21, beta, Cv21);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M21:" << std::endl;
    //print_tensor_matrix( ct );

    // M22 = (1.0 * A_3_2 + 1.0 * A_3_3) * (-1.0 * B_0_0 + 1.0 * B_2_0);  C_0_2 += 1.0 * M22;  C_0_3 += -1.0 * M22;  C_2_2 += 1.0 * M22;  C_2_3 += -1.0 * M22;
    std::array<unsigned, 2> A22_subid = {14, 15};
    std::array<T,2> A22_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av22(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A22_subid, A22_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B22_subid = {0, 8};
    std::array<T,2> B22_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Bv22(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B22_subid, B22_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C22_subid = {4, 5, 12, 13};
    std::array<T,4> C22_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Cv22(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C22_subid, C22_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv22.stride(!row_major) == 1)
    {
        Av22.transpose();
        Bv22.transpose();
        Cv22.transpose();
        straprim_ab2(comm, cfg, alpha, Bv22, Av22, beta, Cv22);
    } else {
        straprim_ab2(comm, cfg, alpha, Av22, Bv22, beta, Cv22);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M22:" << std::endl;
    //print_tensor_matrix( ct );

    // M23 = (1.0 * A_3_0) * (-1.0 * B_0_1 + 1.0 * B_0_3 + 1.0 * B_2_1 + -1.0 * B_2_3);  C_0_1 += 1.0 * M23;  C_0_3 += 1.0 * M23;  C_2_1 += 1.0 * M23;  C_2_3 += 1.0 * M23;
    std::array<unsigned, 1> A23_subid = {10};
    std::array<T,1> A23_coeff_list = {1.0};
    stra_tensor_view<T,1> Av23(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A23_subid, A23_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B23_subid = {1, 5, 9, 13};
    std::array<T,4> B23_coeff_list = {-1.0, 1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Bv23(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B23_subid, B23_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C23_subid = {1, 5, 9, 13};
    std::array<T,4> C23_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv23(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C23_subid, C23_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv23.stride(!row_major) == 1)
    {
        Av23.transpose();
        Bv23.transpose();
        Cv23.transpose();
        straprim_ab2(comm, cfg, alpha, Bv23, Av23, beta, Cv23);
    } else {
        straprim_ab2(comm, cfg, alpha, Av23, Bv23, beta, Cv23);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M23:" << std::endl;
    //print_tensor_matrix( ct );

    // M24 = (1.0 * A_3_3) * (1.0 * B_0_0 + -1.0 * B_0_2 + -1.0 * B_2_0 + 1.0 * B_2_2);  C_0_0 += 1.0 * M24;  C_0_2 += 1.0 * M24;  C_2_0 += 1.0 * M24;  C_2_2 += 1.0 * M24;
    std::array<unsigned, 1> A24_subid = {15};
    std::array<T,1> A24_coeff_list = {1.0};
    stra_tensor_view<T,1> Av24(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A24_subid, A24_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B24_subid = {0, 4, 8, 12};
    std::array<T,4> B24_coeff_list = {1.0, -1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Bv24(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B24_subid, B24_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C24_subid = {0, 4, 8, 12};
    std::array<T,4> C24_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv24(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C24_subid, C24_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv24.stride(!row_major) == 1)
    {
        Av24.transpose();
        Bv24.transpose();
        Cv24.transpose();
        straprim_ab2(comm, cfg, alpha, Bv24, Av24, beta, Cv24);
    } else {
        straprim_ab2(comm, cfg, alpha, Av24, Bv24, beta, Cv24);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M24:" << std::endl;
    //print_tensor_matrix( ct );

    // M25 = (1.0 * A_3_0 + 1.0 * A_3_1) * (-1.0 * B_0_3 + 1.0 * B_2_3);  C_0_0 += -1.0 * M25;  C_0_1 += 1.0 * M25;  C_2_0 += -1.0 * M25;  C_2_1 += 1.0 * M25;
    std::array<unsigned, 2> A25_subid = {10, 11};
    std::array<T,2> A25_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av25(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A25_subid, A25_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B25_subid = {5, 13};
    std::array<T,2> B25_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Bv25(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B25_subid, B25_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C25_subid = {0, 1, 8, 9};
    std::array<T,4> C25_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Cv25(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C25_subid, C25_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv25.stride(!row_major) == 1)
    {
        Av25.transpose();
        Bv25.transpose();
        Cv25.transpose();
        straprim_ab2(comm, cfg, alpha, Bv25, Av25, beta, Cv25);
    } else {
        straprim_ab2(comm, cfg, alpha, Av25, Bv25, beta, Cv25);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M25:" << std::endl;
    //print_tensor_matrix( ct );

    // M26 = (-1.0 * A_3_0 + 1.0 * A_3_2) * (-1.0 * B_0_0 + -1.0 * B_0_1 + 1.0 * B_2_0 + 1.0 * B_2_1);  C_0_3 += 1.0 * M26;  C_2_3 += 1.0 * M26;
    std::array<unsigned, 2> A26_subid = {10, 14};
    std::array<T,2> A26_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Av26(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A26_subid, A26_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B26_subid = {0, 1, 8, 9};
    std::array<T,4> B26_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv26(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B26_subid, B26_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C26_subid = {5, 13};
    std::array<T,2> C26_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv26(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C26_subid, C26_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv26.stride(!row_major) == 1)
    {
        Av26.transpose();
        Bv26.transpose();
        Cv26.transpose();
        straprim_ab2(comm, cfg, alpha, Bv26, Av26, beta, Cv26);
    } else {
        straprim_ab2(comm, cfg, alpha, Av26, Bv26, beta, Cv26);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M26:" << std::endl;
    //print_tensor_matrix( ct );

    // M27 = (1.0 * A_3_1 + -1.0 * A_3_3) * (-1.0 * B_0_2 + -1.0 * B_0_3 + 1.0 * B_2_2 + 1.0 * B_2_3);  C_0_0 += 1.0 * M27;  C_2_0 += 1.0 * M27;
    std::array<unsigned, 2> A27_subid = {11, 15};
    std::array<T,2> A27_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Av27(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A27_subid, A27_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B27_subid = {4, 5, 12, 13};
    std::array<T,4> B27_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv27(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B27_subid, B27_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C27_subid = {0, 8};
    std::array<T,2> C27_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv27(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C27_subid, C27_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv27.stride(!row_major) == 1)
    {
        Av27.transpose();
        Bv27.transpose();
        Cv27.transpose();
        straprim_ab2(comm, cfg, alpha, Bv27, Av27, beta, Cv27);
    } else {
        straprim_ab2(comm, cfg, alpha, Av27, Bv27, beta, Cv27);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M27:" << std::endl;
    //print_tensor_matrix( ct );

    // M28 = (1.0 * A_0_0 + 1.0 * A_0_3 + 1.0 * A_1_0 + 1.0 * A_1_3) * (1.0 * B_3_0 + 1.0 * B_3_3);  C_0_0 += -1.0 * M28;  C_0_3 += -1.0 * M28;  C_1_0 += 1.0 * M28;  C_1_3 += 1.0 * M28;
    std::array<unsigned, 4> A28_subid = {0, 5, 2, 7};
    std::array<T,4> A28_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av28(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A28_subid, A28_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B28_subid = {10, 15};
    std::array<T,2> B28_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv28(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B28_subid, B28_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C28_subid = {0, 5, 2, 7};
    std::array<T,4> C28_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv28(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C28_subid, C28_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv28.stride(!row_major) == 1)
    {
        Av28.transpose();
        Bv28.transpose();
        Cv28.transpose();
        straprim_ab2(comm, cfg, alpha, Bv28, Av28, beta, Cv28);
    } else {
        straprim_ab2(comm, cfg, alpha, Av28, Bv28, beta, Cv28);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M28:" << std::endl;
    //print_tensor_matrix( ct );

    // M29 = (1.0 * A_0_2 + 1.0 * A_0_3 + 1.0 * A_1_2 + 1.0 * A_1_3) * (1.0 * B_3_0);  C_0_2 += -1.0 * M29;  C_0_3 += 1.0 * M29;  C_1_2 += 1.0 * M29;  C_1_3 += -1.0 * M29;
    std::array<unsigned, 4> A29_subid = {4, 5, 6, 7};
    std::array<T,4> A29_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av29(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A29_subid, A29_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 1> B29_subid = {10};
    std::array<T,1> B29_coeff_list = {1.0};
    stra_tensor_view<T,1> Bv29(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B29_subid, B29_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C29_subid = {4, 5, 6, 7};
    std::array<T,4> C29_coeff_list = {-1.0, 1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Cv29(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C29_subid, C29_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv29.stride(!row_major) == 1)
    {
        Av29.transpose();
        Bv29.transpose();
        Cv29.transpose();
        straprim_ab2(comm, cfg, alpha, Bv29, Av29, beta, Cv29);
    } else {
        straprim_ab2(comm, cfg, alpha, Av29, Bv29, beta, Cv29);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M29:" << std::endl;
    //print_tensor_matrix( ct );

    // M30 = (1.0 * A_0_0 + 1.0 * A_1_0) * (1.0 * B_3_1 + -1.0 * B_3_3);  C_0_1 += -1.0 * M30;  C_0_3 += -1.0 * M30;  C_1_1 += 1.0 * M30;  C_1_3 += 1.0 * M30;
    std::array<unsigned, 2> A30_subid = {0, 2};
    std::array<T,2> A30_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av30(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A30_subid, A30_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B30_subid = {11, 15};
    std::array<T,2> B30_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Bv30(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B30_subid, B30_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C30_subid = {1, 5, 3, 7};
    std::array<T,4> C30_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv30(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C30_subid, C30_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv30.stride(!row_major) == 1)
    {
        Av30.transpose();
        Bv30.transpose();
        Cv30.transpose();
        straprim_ab2(comm, cfg, alpha, Bv30, Av30, beta, Cv30);
    } else {
        straprim_ab2(comm, cfg, alpha, Av30, Bv30, beta, Cv30);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M30:" << std::endl;
    //print_tensor_matrix( ct );

    // M31 = (1.0 * A_0_3 + 1.0 * A_1_3) * (-1.0 * B_3_0 + 1.0 * B_3_2);  C_0_0 += -1.0 * M31;  C_0_2 += -1.0 * M31;  C_1_0 += 1.0 * M31;  C_1_2 += 1.0 * M31;
    std::array<unsigned, 2> A31_subid = {5, 7};
    std::array<T,2> A31_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Av31(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A31_subid, A31_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B31_subid = {10, 14};
    std::array<T,2> B31_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Bv31(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B31_subid, B31_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C31_subid = {0, 4, 2, 6};
    std::array<T,4> C31_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Cv31(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C31_subid, C31_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv31.stride(!row_major) == 1)
    {
        Av31.transpose();
        Bv31.transpose();
        Cv31.transpose();
        straprim_ab2(comm, cfg, alpha, Bv31, Av31, beta, Cv31);
    } else {
        straprim_ab2(comm, cfg, alpha, Av31, Bv31, beta, Cv31);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M31:" << std::endl;
    //print_tensor_matrix( ct );

    // M32 = (1.0 * A_0_0 + 1.0 * A_0_1 + 1.0 * A_1_0 + 1.0 * A_1_1) * (1.0 * B_3_3);  C_0_0 += 1.0 * M32;  C_0_1 += -1.0 * M32;  C_1_0 += -1.0 * M32;  C_1_1 += 1.0 * M32;
    std::array<unsigned, 4> A32_subid = {0, 1, 2, 3};
    std::array<T,4> A32_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av32(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A32_subid, A32_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 1> B32_subid = {15};
    std::array<T,1> B32_coeff_list = {1.0};
    stra_tensor_view<T,1> Bv32(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B32_subid, B32_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 4> C32_subid = {0, 1, 2, 3};
    std::array<T,4> C32_coeff_list = {1.0, -1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Cv32(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C32_subid, C32_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv32.stride(!row_major) == 1)
    {
        Av32.transpose();
        Bv32.transpose();
        Cv32.transpose();
        straprim_ab2(comm, cfg, alpha, Bv32, Av32, beta, Cv32);
    } else {
        straprim_ab2(comm, cfg, alpha, Av32, Bv32, beta, Cv32);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M32:" << std::endl;
    //print_tensor_matrix( ct );

    // M33 = (-1.0 * A_0_0 + 1.0 * A_0_2 + -1.0 * A_1_0 + 1.0 * A_1_2) * (1.0 * B_3_0 + 1.0 * B_3_1);  C_0_3 += -1.0 * M33;  C_1_3 += 1.0 * M33;
    std::array<unsigned, 4> A33_subid = {0, 4, 2, 6};
    std::array<T,4> A33_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Av33(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A33_subid, A33_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B33_subid = {10, 11};
    std::array<T,2> B33_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv33(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B33_subid, B33_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C33_subid = {5, 7};
    std::array<T,2> C33_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Cv33(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C33_subid, C33_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv33.stride(!row_major) == 1)
    {
        Av33.transpose();
        Bv33.transpose();
        Cv33.transpose();
        straprim_ab2(comm, cfg, alpha, Bv33, Av33, beta, Cv33);
    } else {
        straprim_ab2(comm, cfg, alpha, Av33, Bv33, beta, Cv33);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M33:" << std::endl;
    //print_tensor_matrix( ct );

    // M34 = (1.0 * A_0_1 + -1.0 * A_0_3 + 1.0 * A_1_1 + -1.0 * A_1_3) * (1.0 * B_3_2 + 1.0 * B_3_3);  C_0_0 += -1.0 * M34;  C_1_0 += 1.0 * M34;
    std::array<unsigned, 4> A34_subid = {1, 5, 3, 7};
    std::array<T,4> A34_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Av34(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A34_subid, A34_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B34_subid = {14, 15};
    std::array<T,2> B34_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv34(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B34_subid, B34_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C34_subid = {0, 2};
    std::array<T,2> C34_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Cv34(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C34_subid, C34_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv34.stride(!row_major) == 1)
    {
        Av34.transpose();
        Bv34.transpose();
        Cv34.transpose();
        straprim_ab2(comm, cfg, alpha, Bv34, Av34, beta, Cv34);
    } else {
        straprim_ab2(comm, cfg, alpha, Av34, Bv34, beta, Cv34);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M34:" << std::endl;
    //print_tensor_matrix( ct );

    // M35 = (-1.0 * A_0_0 + -1.0 * A_0_3 + 1.0 * A_2_0 + 1.0 * A_2_3) * (1.0 * B_0_0 + 1.0 * B_0_3 + 1.0 * B_1_0 + 1.0 * B_1_3);  C_3_0 += 1.0 * M35;  C_3_3 += 1.0 * M35;
    std::array<unsigned, 4> A35_subid = {0, 5, 8, 13};
    std::array<T,4> A35_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av35(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A35_subid, A35_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B35_subid = {0, 5, 2, 7};
    std::array<T,4> B35_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv35(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B35_subid, B35_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C35_subid = {10, 15};
    std::array<T,2> C35_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv35(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C35_subid, C35_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv35.stride(!row_major) == 1)
    {
        Av35.transpose();
        Bv35.transpose();
        Cv35.transpose();
        straprim_ab2(comm, cfg, alpha, Bv35, Av35, beta, Cv35);
    } else {
        straprim_ab2(comm, cfg, alpha, Av35, Bv35, beta, Cv35);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M35:" << std::endl;
    //print_tensor_matrix( ct );

    // M36 = (-1.0 * A_0_2 + -1.0 * A_0_3 + 1.0 * A_2_2 + 1.0 * A_2_3) * (1.0 * B_0_0 + 1.0 * B_1_0);  C_3_2 += 1.0 * M36;  C_3_3 += -1.0 * M36;
    std::array<unsigned, 4> A36_subid = {4, 5, 12, 13};
    std::array<T,4> A36_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av36(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A36_subid, A36_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B36_subid = {0, 2};
    std::array<T,2> B36_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv36(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B36_subid, B36_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C36_subid = {14, 15};
    std::array<T,2> C36_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Cv36(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C36_subid, C36_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv36.stride(!row_major) == 1)
    {
        Av36.transpose();
        Bv36.transpose();
        Cv36.transpose();
        straprim_ab2(comm, cfg, alpha, Bv36, Av36, beta, Cv36);
    } else {
        straprim_ab2(comm, cfg, alpha, Av36, Bv36, beta, Cv36);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M36:" << std::endl;
    //print_tensor_matrix( ct );

    // M37 = (-1.0 * A_0_0 + 1.0 * A_2_0) * (1.0 * B_0_1 + -1.0 * B_0_3 + 1.0 * B_1_1 + -1.0 * B_1_3);  C_3_1 += 1.0 * M37;  C_3_3 += 1.0 * M37;
    std::array<unsigned, 2> A37_subid = {0, 8};
    std::array<T,2> A37_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Av37(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A37_subid, A37_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B37_subid = {1, 5, 3, 7};
    std::array<T,4> B37_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Bv37(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B37_subid, B37_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C37_subid = {11, 15};
    std::array<T,2> C37_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv37(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C37_subid, C37_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv37.stride(!row_major) == 1)
    {
        Av37.transpose();
        Bv37.transpose();
        Cv37.transpose();
        straprim_ab2(comm, cfg, alpha, Bv37, Av37, beta, Cv37);
    } else {
        straprim_ab2(comm, cfg, alpha, Av37, Bv37, beta, Cv37);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M37:" << std::endl;
    //print_tensor_matrix( ct );

    // M38 = (-1.0 * A_0_3 + 1.0 * A_2_3) * (-1.0 * B_0_0 + 1.0 * B_0_2 + -1.0 * B_1_0 + 1.0 * B_1_2);  C_3_0 += 1.0 * M38;  C_3_2 += 1.0 * M38;
    std::array<unsigned, 2> A38_subid = {5, 13};
    std::array<T,2> A38_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Av38(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A38_subid, A38_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B38_subid = {0, 4, 2, 6};
    std::array<T,4> B38_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Bv38(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B38_subid, B38_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C38_subid = {10, 14};
    std::array<T,2> C38_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv38(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C38_subid, C38_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv38.stride(!row_major) == 1)
    {
        Av38.transpose();
        Bv38.transpose();
        Cv38.transpose();
        straprim_ab2(comm, cfg, alpha, Bv38, Av38, beta, Cv38);
    } else {
        straprim_ab2(comm, cfg, alpha, Av38, Bv38, beta, Cv38);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M38:" << std::endl;
    //print_tensor_matrix( ct );

    // M39 = (-1.0 * A_0_0 + -1.0 * A_0_1 + 1.0 * A_2_0 + 1.0 * A_2_1) * (1.0 * B_0_3 + 1.0 * B_1_3);  C_3_0 += -1.0 * M39;  C_3_1 += 1.0 * M39;
    std::array<unsigned, 4> A39_subid = {0, 1, 8, 9};
    std::array<T,4> A39_coeff_list = {-1.0, -1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Av39(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A39_subid, A39_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B39_subid = {5, 7};
    std::array<T,2> B39_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv39(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B39_subid, B39_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C39_subid = {10, 11};
    std::array<T,2> C39_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Cv39(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C39_subid, C39_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv39.stride(!row_major) == 1)
    {
        Av39.transpose();
        Bv39.transpose();
        Cv39.transpose();
        straprim_ab2(comm, cfg, alpha, Bv39, Av39, beta, Cv39);
    } else {
        straprim_ab2(comm, cfg, alpha, Av39, Bv39, beta, Cv39);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M39:" << std::endl;
    //print_tensor_matrix( ct );

    // M40 = (1.0 * A_0_0 + -1.0 * A_0_2 + -1.0 * A_2_0 + 1.0 * A_2_2) * (1.0 * B_0_0 + 1.0 * B_0_1 + 1.0 * B_1_0 + 1.0 * B_1_1);  C_3_3 += 1.0 * M40;
    std::array<unsigned, 4> A40_subid = {0, 4, 8, 12};
    std::array<T,4> A40_coeff_list = {1.0, -1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Av40(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A40_subid, A40_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B40_subid = {0, 1, 2, 3};
    std::array<T,4> B40_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv40(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B40_subid, B40_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 1> C40_subid = {15};
    std::array<T,1> C40_coeff_list = {1.0};
    stra_tensor_view<T,1> Cv40(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C40_subid, C40_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv40.stride(!row_major) == 1)
    {
        Av40.transpose();
        Bv40.transpose();
        Cv40.transpose();
        straprim_ab2(comm, cfg, alpha, Bv40, Av40, beta, Cv40);
    } else {
        straprim_ab2(comm, cfg, alpha, Av40, Bv40, beta, Cv40);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M40:" << std::endl;
    //print_tensor_matrix( ct );

    // M41 = (-1.0 * A_0_1 + 1.0 * A_0_3 + 1.0 * A_2_1 + -1.0 * A_2_3) * (1.0 * B_0_2 + 1.0 * B_0_3 + 1.0 * B_1_2 + 1.0 * B_1_3);  C_3_0 += 1.0 * M41;
    std::array<unsigned, 4> A41_subid = {1, 5, 9, 13};
    std::array<T,4> A41_coeff_list = {-1.0, 1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Av41(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A41_subid, A41_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B41_subid = {4, 5, 6, 7};
    std::array<T,4> B41_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv41(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B41_subid, B41_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 1> C41_subid = {10};
    std::array<T,1> C41_coeff_list = {1.0};
    stra_tensor_view<T,1> Cv41(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C41_subid, C41_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv41.stride(!row_major) == 1)
    {
        Av41.transpose();
        Bv41.transpose();
        Cv41.transpose();
        straprim_ab2(comm, cfg, alpha, Bv41, Av41, beta, Cv41);
    } else {
        straprim_ab2(comm, cfg, alpha, Av41, Bv41, beta, Cv41);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M41:" << std::endl;
    //print_tensor_matrix( ct );

    // M42 = (1.0 * A_1_0 + 1.0 * A_1_3 + -1.0 * A_3_0 + -1.0 * A_3_3) * (1.0 * B_2_0 + 1.0 * B_2_3 + 1.0 * B_3_0 + 1.0 * B_3_3);  C_0_0 += 1.0 * M42;  C_0_3 += 1.0 * M42;
    std::array<unsigned, 4> A42_subid = {2, 7, 10, 15};
    std::array<T,4> A42_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Av42(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A42_subid, A42_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B42_subid = {8, 13, 10, 15};
    std::array<T,4> B42_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv42(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B42_subid, B42_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C42_subid = {0, 5};
    std::array<T,2> C42_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv42(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C42_subid, C42_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv42.stride(!row_major) == 1)
    {
        Av42.transpose();
        Bv42.transpose();
        Cv42.transpose();
        straprim_ab2(comm, cfg, alpha, Bv42, Av42, beta, Cv42);
    } else {
        straprim_ab2(comm, cfg, alpha, Av42, Bv42, beta, Cv42);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M42:" << std::endl;
    //print_tensor_matrix( ct );

    // M43 = (1.0 * A_1_2 + 1.0 * A_1_3 + -1.0 * A_3_2 + -1.0 * A_3_3) * (1.0 * B_2_0 + 1.0 * B_3_0);  C_0_2 += 1.0 * M43;  C_0_3 += -1.0 * M43;
    std::array<unsigned, 4> A43_subid = {6, 7, 14, 15};
    std::array<T,4> A43_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Av43(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A43_subid, A43_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B43_subid = {8, 10};
    std::array<T,2> B43_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv43(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B43_subid, B43_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C43_subid = {4, 5};
    std::array<T,2> C43_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Cv43(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C43_subid, C43_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv43.stride(!row_major) == 1)
    {
        Av43.transpose();
        Bv43.transpose();
        Cv43.transpose();
        straprim_ab2(comm, cfg, alpha, Bv43, Av43, beta, Cv43);
    } else {
        straprim_ab2(comm, cfg, alpha, Av43, Bv43, beta, Cv43);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M43:" << std::endl;
    //print_tensor_matrix( ct );

    // M44 = (1.0 * A_1_0 + -1.0 * A_3_0) * (1.0 * B_2_1 + -1.0 * B_2_3 + 1.0 * B_3_1 + -1.0 * B_3_3);  C_0_1 += 1.0 * M44;  C_0_3 += 1.0 * M44;
    std::array<unsigned, 2> A44_subid = {2, 10};
    std::array<T,2> A44_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Av44(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A44_subid, A44_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B44_subid = {9, 13, 11, 15};
    std::array<T,4> B44_coeff_list = {1.0, -1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Bv44(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B44_subid, B44_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C44_subid = {1, 5};
    std::array<T,2> C44_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv44(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C44_subid, C44_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv44.stride(!row_major) == 1)
    {
        Av44.transpose();
        Bv44.transpose();
        Cv44.transpose();
        straprim_ab2(comm, cfg, alpha, Bv44, Av44, beta, Cv44);
    } else {
        straprim_ab2(comm, cfg, alpha, Av44, Bv44, beta, Cv44);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M44:" << std::endl;
    //print_tensor_matrix( ct );

    // M45 = (1.0 * A_1_3 + -1.0 * A_3_3) * (-1.0 * B_2_0 + 1.0 * B_2_2 + -1.0 * B_3_0 + 1.0 * B_3_2);  C_0_0 += 1.0 * M45;  C_0_2 += 1.0 * M45;
    std::array<unsigned, 2> A45_subid = {7, 15};
    std::array<T,2> A45_coeff_list = {1.0, -1.0};
    stra_tensor_view<T,2> Av45(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A45_subid, A45_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B45_subid = {8, 12, 10, 14};
    std::array<T,4> B45_coeff_list = {-1.0, 1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Bv45(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B45_subid, B45_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C45_subid = {0, 4};
    std::array<T,2> C45_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Cv45(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C45_subid, C45_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv45.stride(!row_major) == 1)
    {
        Av45.transpose();
        Bv45.transpose();
        Cv45.transpose();
        straprim_ab2(comm, cfg, alpha, Bv45, Av45, beta, Cv45);
    } else {
        straprim_ab2(comm, cfg, alpha, Av45, Bv45, beta, Cv45);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M45:" << std::endl;
    //print_tensor_matrix( ct );

    // M46 = (1.0 * A_1_0 + 1.0 * A_1_1 + -1.0 * A_3_0 + -1.0 * A_3_1) * (1.0 * B_2_3 + 1.0 * B_3_3);  C_0_0 += -1.0 * M46;  C_0_1 += 1.0 * M46;
    std::array<unsigned, 4> A46_subid = {2, 3, 10, 11};
    std::array<T,4> A46_coeff_list = {1.0, 1.0, -1.0, -1.0};
    stra_tensor_view<T,4> Av46(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A46_subid, A46_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 2> B46_subid = {13, 15};
    std::array<T,2> B46_coeff_list = {1.0, 1.0};
    stra_tensor_view<T,2> Bv46(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B46_subid, B46_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 2> C46_subid = {0, 1};
    std::array<T,2> C46_coeff_list = {-1.0, 1.0};
    stra_tensor_view<T,2> Cv46(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C46_subid, C46_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv46.stride(!row_major) == 1)
    {
        Av46.transpose();
        Bv46.transpose();
        Cv46.transpose();
        straprim_ab2(comm, cfg, alpha, Bv46, Av46, beta, Cv46);
    } else {
        straprim_ab2(comm, cfg, alpha, Av46, Bv46, beta, Cv46);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M46:" << std::endl;
    //print_tensor_matrix( ct );

    // M47 = (-1.0 * A_1_0 + 1.0 * A_1_2 + 1.0 * A_3_0 + -1.0 * A_3_2) * (1.0 * B_2_0 + 1.0 * B_2_1 + 1.0 * B_3_0 + 1.0 * B_3_1);  C_0_3 += 1.0 * M47;
    std::array<unsigned, 4> A47_subid = {2, 6, 10, 14};
    std::array<T,4> A47_coeff_list = {-1.0, 1.0, 1.0, -1.0};
    stra_tensor_view<T,4> Av47(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A47_subid, A47_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B47_subid = {8, 9, 10, 11};
    std::array<T,4> B47_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv47(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B47_subid, B47_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 1> C47_subid = {5};
    std::array<T,1> C47_coeff_list = {1.0};
    stra_tensor_view<T,1> Cv47(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C47_subid, C47_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv47.stride(!row_major) == 1)
    {
        Av47.transpose();
        Bv47.transpose();
        Cv47.transpose();
        straprim_ab2(comm, cfg, alpha, Bv47, Av47, beta, Cv47);
    } else {
        straprim_ab2(comm, cfg, alpha, Av47, Bv47, beta, Cv47);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M47:" << std::endl;
    //print_tensor_matrix( ct );

    // M48 = (1.0 * A_1_1 + -1.0 * A_1_3 + -1.0 * A_3_1 + 1.0 * A_3_3) * (1.0 * B_2_2 + 1.0 * B_2_3 + 1.0 * B_3_2 + 1.0 * B_3_3);  C_0_0 += 1.0 * M48;
    std::array<unsigned, 4> A48_subid = {3, 7, 11, 15};
    std::array<T,4> A48_coeff_list = {1.0, -1.0, -1.0, 1.0};
    stra_tensor_view<T,4> Av48(my_len_AC, my_len_AB, A_divisor, const_cast<T*>(A), A48_subid, A48_coeff_list, my_stride_A_AC, my_stride_A_AB);
    std::array<unsigned, 4> B48_subid = {12, 13, 14, 15};
    std::array<T,4> B48_coeff_list = {1.0, 1.0, 1.0, 1.0};
    stra_tensor_view<T,4> Bv48(my_len_AB, my_len_BC, B_divisor, const_cast<T*>(B), B48_subid, B48_coeff_list, my_stride_B_AB, my_stride_B_BC);
    std::array<unsigned, 1> C48_subid = {0};
    std::array<T,1> C48_coeff_list = {1.0};
    stra_tensor_view<T,1> Cv48(my_len_AC, my_len_BC, C_divisor, const_cast<T*>(C), C48_subid, C48_coeff_list, my_stride_C_AC, my_stride_C_BC);
    if (Cv48.stride(!row_major) == 1)
    {
        Av48.transpose();
        Bv48.transpose();
        Cv48.transpose();
        straprim_ab2(comm, cfg, alpha, Bv48, Av48, beta, Cv48);
    } else {
        straprim_ab2(comm, cfg, alpha, Av48, Bv48, beta, Cv48);
    }
    comm.barrier();
    //std::cout << "stra_internal/stra_mult_M48:" << std::endl;
    //print_tensor_matrix( ct );



    //std::cout << "stra_internal/stra_mult_2level_A:" << std::endl;
    //print_tensor_matrix( at );

    //std::cout << "stra_internal/stra_mult_2level_B:" << std::endl;
    //print_tensor_matrix( bt );

    //std::cout << "stra_internal/stra_mult_2level_M6:" << std::endl;
    //print_tensor_matrix( ct );

}

#define INSTANTIATE_CONTRACT_BLIS_AB(T) \
template void stra_contract_2level_blis_ab(const communicator& comm, const config& cfg, \
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
void stra_mult_2level_blas(const communicator& comm, const config& cfg,
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

        stra_mult_2level(comm, cfg, cm.length(0), cm.length(1), am.length(1),
                  alpha, false, am.data(), am.stride(0), am.stride(1),
                         false, bm.data(), bm.stride(0), bm.stride(1),
                   T(0), false, cm.data(), cm.stride(0), cm.stride(1));

        add(comm, cfg, {}, len_C, crv.lengths(),
            T(1), false, crv.data(),         {},            crv.strides(),
            beta, false,          C, stride_C_C, stride_C_AC+stride_C_BC);
    }
}

template <typename T>
void stra_mult_2level_ref(const communicator& comm, const config& cfg,
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

    stra_mult_2level(comm, cfg, cm.length(0), cm.length(1), am.length(1),
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

        stra_mult_2level(comm, cfg, cm.length(0), cm.length(1), am.length(1),
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
void stra_mult_2level(const communicator& comm, const config& cfg,
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
                //std::cout << "stra_contract_2level_ref" << std::endl;
                stra_contract_2level_ref(comm, cfg, len_AB, len_AC, len_BC,
                                 alpha, A, stride_A_AB, stride_A_AC,
                                        B, stride_B_AB, stride_B_BC,
                                  beta, C, stride_C_AC, stride_C_BC);
            }
            else if (impl == BLAS_BASED)
            {
                //std::cout << "stra_contract_2level_blas" << std::endl;
                stra_contract_2level_blas(comm, cfg, len_AB, len_AC, len_BC,
                                  alpha, A, stride_A_AB, stride_A_AC,
                                         B, stride_B_AB, stride_B_BC,
                                   beta, C, stride_C_AC, stride_C_BC);
            }
            else if (impl == BLIS_BASED)
            {
                //std::cout << "stra_contract_2level_blis" << std::endl;
                stra_contract_2level_blis(comm, cfg, len_AB, len_AC, len_BC,
                                  alpha, A, stride_A_AB, stride_A_AC,
                                         B, stride_B_AB, stride_B_BC,
                                   beta, C, stride_C_AC, stride_C_BC);
            }
            else if (impl == STRA_NAIVE)
            {
                //std::cout << "stra_contract_2level_blis_naive" << std::endl;
                stra_contract_2level_blis_naive(comm, cfg, len_AB, len_AC, len_BC,
                                  alpha, A, stride_A_AB, stride_A_AC,
                                         B, stride_B_AB, stride_B_BC,
                                   beta, C, stride_C_AC, stride_C_BC);
            }
            else // impl == STRA_AB
            {
                //std::cout << "stra_contract_2level_blis_ab" << std::endl;
                stra_contract_2level_blis_ab(comm, cfg, len_AB, len_AC, len_BC,
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
            std::cout << "stra_mult_2level_ref" << std::endl;
            stra_mult_2level_ref(comm, cfg, len_A, len_B, len_C,
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
            std::cout << "stra_mult_2level_blas" << std::endl;
            stra_mult_2level_blas(comm, cfg, len_A, len_B, len_C,
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
template void stra_mult_2level(const communicator& comm, const config& cfg, \
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
