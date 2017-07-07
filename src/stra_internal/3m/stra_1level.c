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

