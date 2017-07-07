
    template<typename T, int Mat, unsigned N>
    void tensor2matrix(const communicator& comm, const config& cfg,
                    stra_tensor_view<T,N> A, matrix_view<T>& Ap) const
                    //stra_block_scatter_matrix<T,N> A, matrix_view<T>& Ap) const
    {
        // -1: before, allocate buffers for Ap.
        //0. Some constants extraction. Note that stra_tensor_matrix contains offset information
        //1. Computing stra_block_scatter/scatter vector, formatting stra_block_scatter_matrix
        //constexpr bool Trans = (Mat == MAT_B);
        constexpr bool Trans = 0;
        using namespace matrix_constants;
        const len_type MR = (Mat == MAT_A ? cfg.gemm_mr.def<T>() : Mat == MAT_B ? cfg.gemm_kr.def<T>() : cfg.gemm_mr.def<T>());
        const len_type KR = (Mat == MAT_A ? cfg.gemm_kr.def<T>() : Mat == MAT_B ? cfg.gemm_nr.def<T>() : cfg.gemm_nr.def<T>());

        len_type m = At.length(0);
        len_type k = At.length(1);
        //stride_type* scat_buffer = (stride_type*)malloc( sizeof(stride_type) * (m + k) * 2 * N );


        allocate....

        stride_type* rscat_a = scat_buffer;
        stride_type* cscat_a = scat_buffer + m;
        stride_type* rbs_a   = cscat_a + k;
        stride_type* cbs_a   = rbs_a + m;
        // Generate rs_c, cs_c;

        for (unsigned idx=0; idx < stra_size(At); idx++)
        {
            const unsigned offset = idx*2*(m+n);
            
            At.fill_block_scatter(idx, 0, parent.rscat+offset, MB, parent.rbs+offset);
            At.fill_block_scatter(idx, 1, parent.cscat+offset, NB, parent.cbs+offset);
        }
        auto buf = At.data();
        auto coeff = At.coeff_list();
        stra_block_scatter_matrix<T, N> A(At.length(0), At.length(1), buf, coeff,
                rscat_a, MB, rbs_a,
                cscat_a, NB, cbs_a);

        //2. copying/adding tensor to matrix / matrix to tensor

        //std::cout << "MR: " << MR << ";ME: " << ME << ";KR: " << KR << std::endl;

        //TBLIS_ASSERT(A.block_size(0) == (!Trans ? MR : KR));
        //TBLIS_ASSERT(A.block_size(1) == (!Trans ? KR : MR));

        //const len_type MR = A.block_size(0);
        //const len_type KR = A.block_size(1);
        //std::cout << "MR: " << MR << "; KR:" << KR << std::endl;

        //std::cout << "A.block_size(0): " << A.block_size(0) << ";A.block_size(1): " << A.block_size(1) << std::endl;

        len_type m_a = A.length( 0 );
        len_type k_a = A.length( 1 );

        T* p_ap = Ap.data();

        //std::cout << "m_a: " << m_a << ";k_a: " << k_a << std::endl;

        len_type m_first, m_last, k_first, k_last;
        std::tie(m_first, m_last, std::ignore,
                 k_first, k_last, std::ignore) =
            comm.distribute_over_threads_2d(m_a, k_a, MR, KR);

        //p_ap += m_first*k_a + k_first*ME; //(m_first, k_first)

        len_type rs_ap = Ap.stride(0), cs_ap = Ap.stride(1);
        p_ap += m_first*rs_ap + k_first*cs_ap; //(m_first, k_first)

        //std::cout << "m_first: " << m_first << ";m_last: " << m_last << ";k_first: " << k_first << ";k_last: " << k_last << std::endl;

        len_type off_m = m_first;

        //std::cout << "off_m: " << off_m << std::endl;
        //exit( 0 );

        //A.length(0, MR);
        A.shift(0, off_m);

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

        //while (off_m < m_last)
        //{
        //    len_type off_k = k_first;
        //    const stride_type* rscat_a[N];
        //    for (unsigned idx = 0; idx < N; idx++)
        //    {
        //        rscat_a[idx] = A.scatter(idx,0);
        //    }

        //    //len_type m = std::min(MR, m_last-off_m);
        //    //len_type k = k_last-k_first;

        //    T* p_ap_tmp = p_ap;
 
        //    while (off_k < k_last)
        //    {
        //        for (len_type p = 0;p < KR;p++) //replace KR with k
        //        {   
        //            for (len_type mr = 0;mr < MR;mr++) //replace MR with m
        //            {   
        //                p_ap_tmp[mr + cs_ap*p] = 0;//p_a[rscat_a[mr] + cscat_a[p]];
        //                for (unsigned idx = 0; idx < N; idx++)
        //                {
        //                    p_ap_tmp[mr + cs_ap*p] += coeff_list[idx] * p_a[rscat_a[idx][mr] + cscat_a[idx][p]];
        //                }

        //            }
        //            //for (len_type mr = m;mr < MR;mr++)
        //            //{   
        //            //    p_ap[mr + ME*p] = T();
        //            //}
        //        }
        //        p_ap_tmp += cs_ap*KR;
        //        A.shift_block(1, 1);
        //        off_k += KR;
        //    }
        //    p_ap += rs_ap*MR;
        //    A.shift_block(0, 1);
        //    off_m += MR;
        //}


        while (off_m < m_last)
        {
            //stride_type rs_a = A.stride(Trans);
            stride_type rs_a = A.stride(0,0);

            bool is_all_rs_a_nonzero_same = check_all_rs_a_nonzero_same<Trans>( A ) ;
            //bool is_all_rs_a_nonzero_same = true;

            //const stride_type* rscat_a = A.scatter(Trans);
            //////const stride_type* rscat_a = A.scatter(0,Trans);
            const stride_type* rscat_a[N];
            for (unsigned idx = 0; idx < N; idx++)
            {
                rscat_a[idx] = A.scatter(idx,0);
            }

            len_type m = std::min(MR, m_last-off_m);
            len_type k = k_last-k_first;

            //std::cout << "p_a_list[0]:" << p_a_list[0] << std::endl;

            ////if (rs_a == 0)
            //if ( !is_all_rs_a_nonzero_same )
            {
                //std::cout << "not is_all_rs_a_nonzero_same" << std::endl;
                //printf("%d/%d in %d/%d: sb\n",
                //       comm.thread_num(), comm.num_threads(),
                //       comm.gang_num(), comm.num_gangs());
                {


                    //add_sb_mr_ukr.call<T>(m, k, p_a, rscat_a[0], cscat_a[0], cbs_a[0], p_ap);

                for (len_type p = 0;p < k;p++) //replace KR with k
                {   
                    for (len_type mr = 0;mr < m;mr++) //replace MR with m
                    {   
                        p_ap_tmp[mr + cs_ap*p] = 0;//p_a[rscat_a[mr] + cscat_a[p]];
                        for (unsigned idx = 0; idx < N; idx++)
                        {
                            p_ap_tmp[mr + cs_ap*p] += coeff_list[idx] * p_a[rscat_a[idx][mr] + cscat_a[idx][p]];
                        }

                    }
                    for (len_type mr = m;mr < MR;mr++)
                    {   
                        p_ap[mr + cs_ap*p] = T();
                    }
                }


                    //if (N == 1) {
                    //    cfg.pack_sb_mr_ukr.call<T>(m, k, p_a, rscat_a[0], cscat_a[0], cbs_a[0], p_ap);

                    //} else if ( N == 2 ) {
                    //    cfg.stra_pack_two_sb_mr_ukr.call<T>(m, k, N, p_a, coeff_list, rscat_a, cscat_a, cbs_a, p_ap);
                    //} else if ( N == 4 ) {
                    //    cfg.stra_pack_four_sb_mr_ukr.call<T>(m, k, N, p_a, coeff_list, rscat_a, cscat_a, cbs_a, p_ap);
                    //} else {
                    //    std::cout << "N is not 1,2,4" << std::endl;
                    //    cfg.stra_pack_sb_mr_ukr.call<T>(m, k, N, p_a, coeff_list, rscat_a, cscat_a, cbs_a, p_ap);
                    //}


                }
                
            }
            //else
            //{
            //    //std::cout << "is_all_rs_a_nonzero_same" << std::endl;
            //    const T* p_a_list2[N];
            //    const stride_type* cscat_a2[N];
            //    const stride_type* cbs_a2[N];
            //    for (unsigned idx = 0; idx < N; idx++)
            //    {
            //        p_a_list2[idx] = p_a + rscat_a[idx][0];
            //        cscat_a2[idx]  = cscat_a[idx];
            //        cbs_a2[idx]    = cbs_a[idx];
            //    }

            //    //printf("%d/%d in %d/%d: nb\n",
            //    //       comm.thread_num(), comm.num_threads(),
            //    //       comm.gang_num(), comm.num_gangs());
            //    {
            //        ////cfg.pack_nb_mr_ukr.call<T>(m, k, p_a+rscat_a[0], rs_a, cscat_a, cbs_a, p_ap);
            //        ////cfg.pack_nb_mr_ukr.call<T>(m, k, p_a_list[0]+rscat_a[0][0], rs_a, cscat_a[0], cbs_a[0], p_ap);
            //        //cfg.stra_pack_nb_mr_ukr.call<T>(m, k, N, p_a_list2, coeff_list, rs_a, cscat_a2, cbs_a2, p_ap);
            //        ////rscat_a[idx][0]


            //        //std::cout << "p_a_list0:" << p_a[ rscat_a[0][0] + cscat_a[0][0] ] << std::endl;
            //        //std::cout << "p_a_list1:" << p_a[ rscat_a[1][0] + cscat_a[1][0] ] << std::endl;

            //        if (N == 1) {
            //            cfg.pack_nb_mr_ukr.call<T>(m, k, p_a_list2[0], rs_a, cscat_a2[0], cbs_a2[0], p_ap);
            //        } else if ( N == 2 ) {
            //            cfg.stra_pack_two_nb_mr_ukr.call<T>(m, k, N, p_a_list2, coeff_list, rs_a, cscat_a2, cbs_a2, p_ap);
            //        } else if ( N == 4 ) {
            //            cfg.stra_pack_four_nb_mr_ukr.call<T>(m, k, N, p_a_list2, coeff_list, rs_a, cscat_a2, cbs_a2, p_ap);
            //        } else {
            //            std::cout << "N is not 1,2,4" << std::endl;
            //            cfg.stra_pack_nb_mr_ukr.call<T>(m, k, N, p_a_list2, coeff_list, rs_a, cscat_a2, cbs_a2, p_ap);
            //        }


            //    }
            //    
            //}


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

            //p_ap += ME*k_a;
            p_ap += cs_ap*k_a;
            A.shift_block(0, 1);
            off_m += MR;
        }

        A.shift(0, -off_m);
        A.length(0, m_a);

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

