#ifndef _TBLIS_STRA_NODES_PACKM_HPP_
#define _TBLIS_STRA_NODES_PACKM_HPP_

#include "util/thread.h"
#include "util/basic_types.h"

#include "memory/alignment.hpp"
#include "memory/memory_pool.hpp"

#include "matrix/scatter_matrix.hpp"
#include "matrix/block_scatter_matrix.hpp"
#include "matrix/stra_matrix_view.hpp"


#include "matrix/stra_block_scatter_matrix.hpp"

#include "configs/configs.hpp"

#include "iface/1m/reduce.h"

#define TBLIS_MAX_UNROLL 8

namespace tblis
{

template<bool Trans, typename T, unsigned N>
bool check_all_rs_a_nonzero_same( stra_block_scatter_matrix<T,N>& A )
{
    for (unsigned idx = 0; idx < N; idx++) {
        if ( A.stride(idx, Trans) == 0 || A.stride(idx, Trans) != A.stride(0, Trans) ) {
            return false;
        }
    }
    return true;
}


template <typename T, int Mat>
struct stra_pack_row_panel
{
    static constexpr bool Trans = Mat == matrix_constants::MAT_B;

    void operator()(const communicator& comm, const config& cfg,
                    matrix_view<T>& A, matrix_view<T>& Ap) const
    {
        const len_type MR = (!Trans ? cfg.gemm_mr.def<T>()
                                    : cfg.gemm_nr.def<T>());
        const len_type ME = (!Trans ? cfg.gemm_mr.extent<T>()
                                    : cfg.gemm_nr.extent<T>());
        const len_type KR = cfg.gemm_kr.def<T>();

        len_type m_a = A.length( Trans);
        len_type k_a = A.length(!Trans);
        stride_type rs_a = A.stride( Trans);
        stride_type cs_a = A.stride(!Trans);
        const T* p_a = A.data();
        T* p_ap = Ap.data();

        len_type m_first, m_last, k_first, k_last;
        std::tie(m_first, m_last, std::ignore,
                 k_first, k_last, std::ignore) =
            comm.distribute_over_threads_2d(m_a, k_a, MR, KR);

        p_a += m_first*rs_a + k_first*cs_a;
        p_ap += (m_first/MR)*ME*k_a + k_first*ME;

        /*
        comm.barrier();
        T norm = T();
        for (len_type i = 0;i < m_a;i++)
        {
            for (len_type j = 0;j < k_a;j++)
            {
                norm += norm2(A.data()[i*rs_a + j*cs_a]);
            }
        }
        printf("%d/%d in %d/%d: %s before: %.15f\n",
               comm.thread_num(), comm.num_threads(),
               comm.gang_num(), comm.num_gangs(),
               (Trans ? "B" : "A"), sqrt(norm));
        comm.barrier();
        */

        for (len_type off_m = m_first;off_m < m_last;off_m += MR)
        {
            len_type m = std::min(MR, m_last-off_m);
            len_type k = k_last-k_first;

            if (!Trans)
                cfg.pack_nn_mr_ukr.call<T>(m, k, p_a, rs_a, cs_a, p_ap);
            else
                cfg.pack_nn_nr_ukr.call<T>(m, k, p_a, rs_a, cs_a, p_ap);

            p_a += m*rs_a;
            p_ap += ME*k_a;
        }

        /*
        comm.barrier();
        norm = T();
        for (len_type i = 0;i < m_a;i++)
        {
            for (len_type j = 0;j < k_a;j++)
            {
                norm += norm2(Ap.data()[(i/MR)*ME*k_a + (i%MR) + j*ME]);
            }
        }
        printf("%d/%d in %d/%d: %s after: %.15f\n",
               comm.thread_num(), comm.num_threads(),
               comm.gang_num(), comm.num_gangs(),
               (Trans ? "B" : "A"), sqrt(norm));
        comm.barrier();
        */
    }

    template<unsigned N>
    void operator()(const communicator& comm, const config& cfg,
                    stra_matrix_view<T,N>& A, matrix_view<T>& Ap) const
    {
        const len_type MR = (!Trans ? cfg.gemm_mr.def<T>()
                                    : cfg.gemm_nr.def<T>());
        const len_type ME = (!Trans ? cfg.gemm_mr.extent<T>()
                                    : cfg.gemm_nr.extent<T>());
        const len_type KR = cfg.gemm_kr.def<T>();

        len_type m_a = A.length( Trans);
        len_type k_a = A.length(!Trans);
        stride_type rs_a = A.stride( Trans);
        stride_type cs_a = A.stride(!Trans);
        //const T* p_a = A.data();
        //std::array<T*,N> p_a_list = A.data_list();

        auto a_list = A.data_list();
        const T* p_a_list[N];
        for (unsigned idx = 0; idx < N; idx++)
        {
            p_a_list[idx] = a_list[idx];
        }

        auto my_coeff_list = A.coeff_list();
        //const T coeff_list[N]=;
        T coeff_list[N];
        for (unsigned idx = 0; idx < N; idx++)
        {
            coeff_list[idx] = my_coeff_list[idx];
        }

        T* p_ap = Ap.data();

        len_type m_first, m_last, k_first, k_last;
        std::tie(m_first, m_last, std::ignore,
                 k_first, k_last, std::ignore) =
            comm.distribute_over_threads_2d(m_a, k_a, MR, KR);

        //for (auto &p_a : p_a_list)
        for (int i = 0; i < N; i++) {
            p_a_list[i] += m_first*rs_a + k_first*cs_a;
        }
        //p_a += m_first*rs_a + k_first*cs_a;

        p_ap += (m_first/MR)*ME*k_a + k_first*ME;

        /*
        comm.barrier();
        T norm = T();
        for (len_type i = 0;i < m_a;i++)
        {
            for (len_type j = 0;j < k_a;j++)
            {
                norm += norm2(A.data()[i*rs_a + j*cs_a]);
            }
        }
        printf("%d/%d in %d/%d: %s before: %.15f\n",
               comm.thread_num(), comm.num_threads(),
               comm.gang_num(), comm.num_gangs(),
               (Trans ? "B" : "A"), sqrt(norm));
        comm.barrier();
        */

        for (len_type off_m = m_first;off_m < m_last;off_m += MR)
        {
            len_type m = std::min(MR, m_last-off_m);
            len_type k = k_last-k_first;


            const T* p_a_list2[N];
            for (unsigned i = 0; i < N; i++)
            {
                p_a_list2[i] = p_a_list[i];
            }

            if (!Trans)
            {
                //Pack A
                //cfg.stra_pack_nn_mr_ukr.call<T>(m, k, N, p_a_list2, coeff_list, rs_a, cs_a, p_ap);

                //std::cout << "N:" << N << std::endl;
                if (N == 1) {
                    cfg.pack_nn_mr_ukr.call<T>(m, k, p_a_list2[0], rs_a, cs_a, p_ap);
                } else if ( N == 2 ) {
                    cfg.stra_pack_two_nn_mr_ukr.call<T>(m, k, N, p_a_list2, coeff_list, rs_a, cs_a, p_ap);
                } else if ( N == 4 ) {
                    cfg.stra_pack_four_nn_mr_ukr.call<T>(m, k, N, p_a_list2, coeff_list, rs_a, cs_a, p_ap);
                } else {
                    std::cout << "stra_pack_nn_mr_ukr: N is not 1,2,4, N: " << N << std::endl;
                    cfg.stra_pack_nn_mr_ukr.call<T>(m, k, N, p_a_list2, coeff_list, rs_a, cs_a, p_ap);
                }
            }
            else
            {
                //Pack B
                //cfg.stra_pack_nn_nr_ukr.call<T>(m, k, N, p_a_list2, coeff_list, rs_a, cs_a, p_ap);

                if (N == 1) {
                    cfg.pack_nn_nr_ukr.call<T>(m, k, p_a_list2[0], rs_a, cs_a, p_ap);
                } else if ( N == 2 ) {
                    cfg.stra_pack_two_nn_nr_ukr.call<T>(m, k, N, p_a_list2, coeff_list, rs_a, cs_a, p_ap);
                } else if ( N == 4 ) {
                    cfg.stra_pack_four_nn_nr_ukr.call<T>(m, k, N, p_a_list2, coeff_list, rs_a, cs_a, p_ap);
                } else {
                    std::cout << "stra_pack_nn_nr_ukr: N is not 1,2,4, N: " << N << std::endl;
                    cfg.stra_pack_nn_nr_ukr.call<T>(m, k, N, p_a_list2, coeff_list, rs_a, cs_a, p_ap);
                }
            }



            //if (m == 4) {

            //    std::cout << "Packing" << std::endl;
            //    std::cout << "m:" << m << ";k:" << k << ";rs_a:" << rs_a << ";cs_a:" << cs_a << std::endl;

            //    std::cout << "A0:\n";
            //    for (int i = 0; i < k; i++)
            //    {
            //        for (int j = 0; j < m; j++)
            //        {
            //            std::cout << p_a_list[0][i*cs_a+j*rs_a] << " ";
            //        }
            //        std::cout << std::endl;
            //    }

            //    std::cout << "A1:\n";
            //    for (int i = 0; i < k; i++)
            //    {
            //        for (int j = 0; j < m; j++)
            //        {
            //            std::cout << p_a_list[1][i*cs_a+j*rs_a] << " ";
            //        }
            //        std::cout << std::endl;
            //    }

            //    std::cout << "A_packed:\n";
            //    for (int i = 0; i < k; i++)
            //    {
            //        for (int j = 0; j < m; j++)
            //        {
            //            std::cout << p_ap[i*4+j] << " ";
            //        }
            //        std::cout << std::endl;
            //    }
            //}


            //for (auto &p_a : p_a_list) 
            for (int i = 0; i < N; i++) {
                p_a_list[i] += m*rs_a;
            }
            //p_a += m*rs_a;
            p_ap += ME*k_a;
        }

        /*
        comm.barrier();
        norm = T();
        for (len_type i = 0;i < m_a;i++)
        {
            for (len_type j = 0;j < k_a;j++)
            {
                norm += norm2(Ap.data()[(i/MR)*ME*k_a + (i%MR) + j*ME]);
            }
        }
        printf("%d/%d in %d/%d: %s after: %.15f\n",
               comm.thread_num(), comm.num_threads(),
               comm.gang_num(), comm.num_gangs(),
               (Trans ? "B" : "A"), sqrt(norm));
        comm.barrier();
        */
    }

    void operator()(const communicator& comm, const config& cfg,
                    const_scatter_matrix_view<T>& A, matrix_view<T>& Ap) const
    {
        const len_type MR = (!Trans ? cfg.gemm_mr.def<T>()
                                    : cfg.gemm_nr.def<T>());
        const len_type ME = (!Trans ? cfg.gemm_mr.extent<T>()
                                    : cfg.gemm_nr.extent<T>());
        const len_type KR = cfg.gemm_kr.def<T>();

        len_type m_a = A.length( Trans);
        len_type k_a = A.length(!Trans);
        stride_type rs_a = A.stride( Trans);
        stride_type cs_a = A.stride(!Trans);
        const stride_type* rscat_a = A.scatter( Trans);
        const stride_type* cscat_a = A.scatter(!Trans);
        const T* p_a = A.data();
        T* p_ap = Ap.data();

        len_type m_first, m_last, k_first, k_last;
        std::tie(m_first, m_last, std::ignore,
                 k_first, k_last, std::ignore) =
            comm.distribute_over_threads_2d(m_a, k_a, MR, KR);

        p_a += m_first*rs_a + k_first*cs_a;
        rscat_a += m_first;
        cscat_a += k_first;
        p_ap += m_first*k_a + k_first*ME;

        for (len_type off_m = m_first;off_m < m_last;off_m += MR)
        {
            len_type m = std::min(MR, m_last-off_m);
            len_type k = k_last-k_first;

            if (rs_a == 0 && cs_a == 0)
            {
                if (!Trans)
                    cfg.pack_ss_mr_ukr.call<T>(m, k, p_a, rscat_a, cscat_a, p_ap);
                else
                    cfg.pack_ss_nr_ukr.call<T>(m, k, p_a, rscat_a, cscat_a, p_ap);

                rscat_a += m;
            }
            else if (rs_a == 0)
            {
                if (!Trans)
                    cfg.pack_sn_mr_ukr.call<T>(m, k, p_a, rscat_a, cs_a, p_ap);
                else
                    cfg.pack_sn_nr_ukr.call<T>(m, k, p_a, rscat_a, cs_a, p_ap);

                rscat_a += m;
            }
            else if (cs_a == 0)
            {
                if (!Trans)
                    cfg.pack_ns_mr_ukr.call<T>(m, k, p_a, rs_a, cscat_a, p_ap);
                else
                    cfg.pack_ns_nr_ukr.call<T>(m, k, p_a, rs_a, cscat_a, p_ap);

                p_a += m*rs_a;
            }
            else
            {
                if (!Trans)
                    cfg.pack_nn_mr_ukr.call<T>(m, k, p_a, rs_a, cs_a, p_ap);
                else
                    cfg.pack_nn_nr_ukr.call<T>(m, k, p_a, rs_a, cs_a, p_ap);

                p_a += m*rs_a;
            }

            p_ap += ME*k_a;
        }
    }


    // Should I use block_scatter_matrix<T>& A, or block_scatter_matrix<T> A??
    template<unsigned N>
    void operator()(const communicator& comm, const config& cfg,
                    stra_block_scatter_matrix<T,N> A, matrix_view<T>& Ap) const
    {
        const len_type MR = (!Trans ? cfg.gemm_mr.def<T>()
                                    : cfg.gemm_nr.def<T>());
        const len_type ME = (!Trans ? cfg.gemm_mr.extent<T>()
                                    : cfg.gemm_nr.extent<T>());
        const len_type KR = cfg.gemm_kr.def<T>();

        //std::cout << "MR: " << MR << ";ME: " << ME << ";KR: " << KR << std::endl;

        TBLIS_ASSERT(A.block_size(0) == (!Trans ? MR : KR));
        TBLIS_ASSERT(A.block_size(1) == (!Trans ? KR : MR));


        //std::cout << "A.block_size(0): " << A.block_size(0) << ";A.block_size(1): " << A.block_size(1) << std::endl;


        len_type m_a = A.length( Trans);
        len_type k_a = A.length(!Trans);
        T* p_ap = Ap.data();


        //std::cout << "m_a: " << m_a << ";k_a: " << k_a << std::endl;

        len_type m_first, m_last, k_first, k_last;
        std::tie(m_first, m_last, std::ignore,
                 k_first, k_last, std::ignore) =
            comm.distribute_over_threads_2d(m_a, k_a, MR, KR);

        p_ap += m_first*k_a + k_first*ME;


        //std::cout << "m_first: " << m_first << ";m_last: " << m_last << ";k_first: " << k_first << ";k_last: " << k_last << std::endl;

        /*
        comm.barrier();
        T norm = T();
        for (len_type i = 0;i < m_a;i++)
        {
            for (len_type j = 0;j < k_a;j++)
            {
                norm += norm2(A.raw_data()[A.scatter(Trans)[i] + A.scatter(!Trans)[j]]);
            }
        }
        printf("%d/%d in %d/%d: %d-%d/%d %d-%d/%d\n",
               comm.thread_num(), comm.num_threads(),
               comm.gang_num(), comm.num_gangs(),
               m_first, m_last, m_a, k_first, k_last, k_a);
        printf("%d/%d in %d/%d: %s before: %.15f\n",
               comm.thread_num(), comm.num_threads(),
               comm.gang_num(), comm.num_gangs(),
               (Trans ? "B" : "A"), sqrt(norm));
        comm.barrier();
        */

        len_type off_m = m_first;


        //std::cout << "off_m: " << off_m << std::endl;
        //exit( 0 );

        A.length(Trans, MR);
        A.shift(Trans, off_m);

        const T* p_a = A.raw_data();

        //auto a_list = A.raw_data();
        //const T* p_a_list[N];
        //for (unsigned idx = 0; idx < N; idx++)
        //{
        //    p_a_list[idx] = a_list[idx];
        //}

        auto my_coeff_list = A.coeff_list();
        T coeff_list[N];
        for (unsigned idx = 0; idx < N; idx++)
        {
            coeff_list[idx] = my_coeff_list[idx];
        }

        //const stride_type* cscat_a = A.scatter(!Trans) + k_first;
        //const stride_type* cbs_a = A.block_scatter(!Trans) + k_first/KR;
        const stride_type* cscat_a[N];
        const stride_type* cbs_a[N];
        for (unsigned idx = 0; idx < N; idx++)
        {
            cscat_a[idx] =  A.scatter(idx, !Trans) + k_first;
            cbs_a[idx] = A.block_scatter(idx, !Trans) + k_first/KR;
        }

        while (off_m < m_last)
        {
            //stride_type rs_a = A.stride(Trans);
            stride_type rs_a = A.stride(0,Trans);

            bool is_all_rs_a_nonzero_same = check_all_rs_a_nonzero_same<Trans>( A ) ;
            //bool is_all_rs_a_nonzero_same = true;

            //const stride_type* rscat_a = A.scatter(Trans);
            //////const stride_type* rscat_a = A.scatter(0,Trans);
            const stride_type* rscat_a[N];
            for (unsigned idx = 0; idx < N; idx++)
            {
                rscat_a[idx] = A.scatter(idx,Trans);
            }

            len_type m = std::min(MR, m_last-off_m);
            len_type k = k_last-k_first;

            //std::cout << "p_a_list[0]:" << p_a_list[0] << std::endl;

            //if (rs_a == 0)
            if ( !is_all_rs_a_nonzero_same )
            {
                //std::cout << "not is_all_rs_a_nonzero_same" << std::endl;
                //printf("%d/%d in %d/%d: sb\n",
                //       comm.thread_num(), comm.num_threads(),
                //       comm.gang_num(), comm.num_gangs());
                if (!Trans)
                {
                    ////cfg.pack_sb_mr_ukr.call<T>(m, k, p_a, rscat_a, cscat_a, cbs_a, p_ap);
                    ////cfg.pack_sb_mr_ukr.call<T>(m, k, p_a, rscat_a[0], cscat_a[0], cbs_a[0], p_ap);
                    //cfg.stra_pack_sb_mr_ukr.call<T>(m, k, N, p_a, coeff_list, rscat_a, cscat_a, cbs_a, p_ap);
                    ////cfg.stra_pack_sb_mr_ukr.call<T,N>(m, k, p_a, rscat_a, cscat_a, cbs_a, p_ap);


                    //std::cout << "Branch3" << std::endl;

                    if (N == 1) {
                        cfg.pack_sb_mr_ukr.call<T>(m, k, p_a, rscat_a[0], cscat_a[0], cbs_a[0], p_ap);
                    } else if ( N == 2 ) {
                        cfg.stra_pack_two_sb_mr_ukr.call<T>(m, k, N, p_a, coeff_list, rscat_a, cscat_a, cbs_a, p_ap);
                    } else if ( N == 4 ) {
                        cfg.stra_pack_four_sb_mr_ukr.call<T>(m, k, N, p_a, coeff_list, rscat_a, cscat_a, cbs_a, p_ap);
                    } else {
                        std::cout << "N is not 1,2,4" << std::endl;
                        cfg.stra_pack_sb_mr_ukr.call<T>(m, k, N, p_a, coeff_list, rscat_a, cscat_a, cbs_a, p_ap);
                    }

                }
                else
                {
                    ////cfg.pack_sb_nr_ukr.call<T>(m, k, p_a, rscat_a, cscat_a, cbs_a, p_ap);
                    ////cfg.pack_sb_nr_ukr.call<T>(m, k, p_a, rscat_a[0], cscat_a[0], cbs_a[0], p_ap);
                    //cfg.stra_pack_sb_nr_ukr.call<T>(m, k, N, p_a, coeff_list, rscat_a, cscat_a, cbs_a, p_ap);
                    ////cfg.stra_pack_sb_nr_ukr.call<T,N>(m, k, p_a, rscat_a, cscat_a, cbs_a, p_ap);


                    //std::cout << "Branch4" << std::endl;

                    if (N == 1) {
                        cfg.pack_sb_nr_ukr.call<T>(m, k, p_a, rscat_a[0], cscat_a[0], cbs_a[0], p_ap);
                    } else if ( N == 2 ) {
                        cfg.stra_pack_two_sb_nr_ukr.call<T>(m, k, N, p_a, coeff_list, rscat_a, cscat_a, cbs_a, p_ap);
                    } else if ( N == 4 ) {
                        cfg.stra_pack_four_sb_nr_ukr.call<T>(m, k, N, p_a, coeff_list, rscat_a, cscat_a, cbs_a, p_ap);
                    } else {
                        std::cout << "N is not 1,2,4" << std::endl;
                        cfg.stra_pack_sb_nr_ukr.call<T>(m, k, N, p_a, coeff_list, rscat_a, cscat_a, cbs_a, p_ap);
                    }

                }
            }
            else
            {
                //std::cout << "is_all_rs_a_nonzero_same" << std::endl;
                const T* p_a_list2[N];
                const stride_type* cscat_a2[N];
                const stride_type* cbs_a2[N];
                for (unsigned idx = 0; idx < N; idx++)
                {
                    p_a_list2[idx] = p_a + rscat_a[idx][0];
                    cscat_a2[idx]  = cscat_a[idx];
                    cbs_a2[idx]    = cbs_a[idx];
                }

                //printf("%d/%d in %d/%d: nb\n",
                //       comm.thread_num(), comm.num_threads(),
                //       comm.gang_num(), comm.num_gangs());
                if (!Trans)
                {
                    ////cfg.pack_nb_mr_ukr.call<T>(m, k, p_a+rscat_a[0], rs_a, cscat_a, cbs_a, p_ap);
                    ////cfg.pack_nb_mr_ukr.call<T>(m, k, p_a_list[0]+rscat_a[0][0], rs_a, cscat_a[0], cbs_a[0], p_ap);
                    //cfg.stra_pack_nb_mr_ukr.call<T>(m, k, N, p_a_list2, coeff_list, rs_a, cscat_a2, cbs_a2, p_ap);
                    ////rscat_a[idx][0]


                    //std::cout << "Branch1" << std::endl;

                    //std::cout << "p_a_list0:" << p_a[ rscat_a[0][0] + cscat_a[0][0] ] << std::endl;
                    //std::cout << "p_a_list1:" << p_a[ rscat_a[1][0] + cscat_a[1][0] ] << std::endl;

                    if (N == 1) {
                        cfg.pack_nb_mr_ukr.call<T>(m, k, p_a_list2[0], rs_a, cscat_a2[0], cbs_a2[0], p_ap);
                    } else if ( N == 2 ) {
                        cfg.stra_pack_two_nb_mr_ukr.call<T>(m, k, N, p_a_list2, coeff_list, rs_a, cscat_a2, cbs_a2, p_ap);
                    } else if ( N == 4 ) {
                        cfg.stra_pack_four_nb_mr_ukr.call<T>(m, k, N, p_a_list2, coeff_list, rs_a, cscat_a2, cbs_a2, p_ap);
                    } else {
                        std::cout << "N is not 1,2,4" << std::endl;
                        cfg.stra_pack_nb_mr_ukr.call<T>(m, k, N, p_a_list2, coeff_list, rs_a, cscat_a2, cbs_a2, p_ap);
                    }


                }
                else
                {
                    ////cfg.pack_nb_nr_ukr.call<T>(m, k, p_a+rscat_a[0], rs_a, cscat_a, cbs_a, p_ap);
                    ////cfg.pack_nb_nr_ukr.call<T>(m, k, p_a+rscat_a[0][0], rs_a, cscat_a[0], cbs_a[0], p_ap);
                    //cfg.stra_pack_nb_nr_ukr.call<T>(m, k, N, p_a, coeff_list, rs_a, cscat_a2, cbs_a2, p_ap);

                    //std::cout << "Branch2" << std::endl;
                    //std::cout << "p_a_list0:" << p_a[ rscat_a[0][0] + cscat_a[0][0] ] << std::endl;
                    //std::cout << "p_a_list1:" << p_a[ rscat_a[1][0] + cscat_a[1][0] ] << std::endl;


                    
                    if (N == 1) {
                        cfg.pack_nb_nr_ukr.call<T>(m, k, p_a_list2[0], rs_a, cscat_a2[0], cbs_a2[0], p_ap);
                    } else if ( N == 2 ) {
                        cfg.stra_pack_two_nb_nr_ukr.call<T>(m, k, N, p_a_list2, coeff_list, rs_a, cscat_a2, cbs_a2, p_ap);
                    } else if ( N == 4 ) {
                        cfg.stra_pack_four_nb_nr_ukr.call<T>(m, k, N, p_a_list2, coeff_list, rs_a, cscat_a2, cbs_a2, p_ap);
                    } else {
                        std::cout << "N is not 1,2,4" << std::endl;
                        cfg.stra_pack_nb_nr_ukr.call<T>(m, k, N, p_a_list2, coeff_list, rs_a, cscat_a2, cbs_a2, p_ap);
                    }

                }
            }


            //if (m == 4) {

                //std::cout << "A[0]:\n";
                //for (int i = 0; i < k; i++)
                //{
                //    for (int j = 0; j < m; j++)
                //    {
                //        std::cout << p_a_list[0][i*4+j] << " ";
                //    }
                //    std::cout << std::endl;
                //}

                //if ( N == 2 )
                //{
                //    std::cout << "A[1]:\n";
                //    for (int i = 0; i < k; i++)
                //    {
                //        for (int j = 0; j < m; j++)
                //        {
                //            std::cout << p_a_list[1][i*4+j] << " ";
                //        }
                //        std::cout << std::endl;
                //    }
                //}

                ////std::cout << "p_a[0]: " << p_a[0]  << std::endl;
                //std::cout << "A_packed:\n";
                //for (int i = 0; i < k; i++)
                //{
                //    for (int j = 0; j < m; j++)
                //    {
                //        std::cout << p_ap[i*4+j] << " ";
                //    }
                //    std::cout << std::endl;
                //}
            //}


            /*
            T nrm1 = T();
            T nrm2 = T();
            for (len_type i = 0;i < m;i++)
            {
                for (len_type j = 0;j < k;j++)
                {
                    nrm1 += norm2(p_ap[i + j*ME]);
                    if (rs_a == 0)
                        nrm1 += norm2(p_a[rscat_a[i] + cscat_a[j]]);
                    else
                        nrm2 += norm2(p_a[i*rs_a + cscat_a[j]]);
                }
            }
            printf("%d/%d in %d/%d: %s sub: %.15f %.15f\n",
                   comm.thread_num(), comm.num_threads(),
                   comm.gang_num(), comm.num_gangs(),
                   (Trans ? "B" : "A"), sqrt(nrm1), sqrt(nrm2));
            */

            p_ap += ME*k_a;
            A.shift_block(Trans, 1);
            off_m += MR;
        }

        A.shift(Trans, -off_m);
        A.length(Trans, m_a);

        /*
        comm.barrier();
        norm = T();
        for (len_type i = 0;i < m_a;i++)
        {
            for (len_type j = 0;j < k_a;j++)
            {
                norm += norm2(Ap.data()[(i/MR)*ME*k_a + (i%MR) + j*ME]);
            }
        }
        printf("%d/%d in %d/%d: %s after: %.15f\n",
               comm.thread_num(), comm.num_threads(),
               comm.gang_num(), comm.num_gangs(),
               (Trans ? "B" : "A"), sqrt(norm));
        comm.barrier();
        */
    }

};

template <typename Pack, int Mat> struct stra_pack_and_run;

template <typename Pack>
struct stra_pack_and_run<Pack, matrix_constants::MAT_A>
{
    template <typename Run, typename T, typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixP>
    stra_pack_and_run(Run& run, const communicator& comm, const config& cfg,
                 T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C, MatrixP& P)
    {
        Pack()(comm, cfg, A, P);
        comm.barrier();
        run(comm, cfg, alpha, P, B, beta, C);
        comm.barrier();
    }
};

template <typename Pack>
struct stra_pack_and_run<Pack, matrix_constants::MAT_B>
{
    template <typename Run, typename T, typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixP>
    stra_pack_and_run(Run& run, const communicator& comm, const config& cfg,
                 T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C, MatrixP& P)
    {
        Pack()(comm, cfg, B, P);
        comm.barrier();
        run(comm, cfg, alpha, A, P, beta, C);
        comm.barrier();
    }
};

template <int Mat, MemoryPool& Pool, typename Child>
struct stra_pack
{
    Child child;
    MemoryPool::Block pack_buffer;
    void* pack_ptr = nullptr;

    template <typename T, typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(const communicator& comm, const config& cfg,
                    T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        using namespace matrix_constants;

        //std::cout << "Enter stra_pack" << std::endl;


        constexpr bool Trans = (Mat == MAT_B);
        const len_type MR = (!Trans ? cfg.gemm_mr.def<T>()
                                    : cfg.gemm_nr.def<T>());
        const len_type ME = (!Trans ? cfg.gemm_mr.extent<T>()
                                    : cfg.gemm_nr.extent<T>());

        len_type m_p = ceil_div(!Trans ? A.length(0) : B.length(1), MR)*ME;
        len_type k_p =         (!Trans ? A.length(1) : B.length(0));

        if (!pack_ptr)
        {
            if (comm.master())
            {
                pack_buffer = Pool.allocate<T>(m_p*k_p+std::max(m_p,k_p)*TBLIS_MAX_UNROLL);
                pack_ptr = pack_buffer.get();
            }

            comm.broadcast(pack_ptr);
        }

        matrix_view<T> P({!Trans ? m_p : k_p,
                          !Trans ? k_p : m_p},
                          static_cast<T*>(pack_ptr),
                          {!Trans? k_p :  1,
                          !Trans?   1 : k_p});

        typedef stra_pack_row_panel<T, Mat> Pack;
        stra_pack_and_run<Pack, Mat>(child, comm, cfg, alpha, A, B, beta, C, P);
    }
};

template <MemoryPool& Pool, typename Child>
using stra_pack_a = stra_pack<matrix_constants::MAT_A, Pool, Child>;

template <MemoryPool& Pool, typename Child>
using stra_pack_b = stra_pack<matrix_constants::MAT_B, Pool, Child>;

}

#endif
