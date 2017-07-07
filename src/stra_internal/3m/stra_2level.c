    len_type ms, ks, ns;
    len_type md, kd, nd;
    len_type mr, kr, nr;

    mr = m % ( 4 ), kr = k % ( 4 ), nr = n % ( 4 );
    md = m - mr, kd = k - kr, nd = n - nr;

    ms=md, ks=kd, ns=nd;
    const T *A_0, *A_1, *A_2, *A_3;
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 0, 0, A, &A_0 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 0, 1, A, &A_1 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 1, 0, A, &A_2 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 1, 1, A, &A_3 );
    ms=ms/2, ks=ks/2, ns=ns/2;
    const T *A_0_0, *A_0_1, *A_0_2, *A_0_3;
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 0, 0, A_0, &A_0_0 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 0, 1, A_0, &A_0_1 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 1, 0, A_0, &A_0_2 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 1, 1, A_0, &A_0_3 );
    const T *A_1_0, *A_1_1, *A_1_2, *A_1_3;
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 0, 0, A_1, &A_1_0 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 0, 1, A_1, &A_1_1 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 1, 0, A_1, &A_1_2 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 1, 1, A_1, &A_1_3 );
    const T *A_2_0, *A_2_1, *A_2_2, *A_2_3;
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 0, 0, A_2, &A_2_0 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 0, 1, A_2, &A_2_1 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 1, 0, A_2, &A_2_2 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 1, 1, A_2, &A_2_3 );
    const T *A_3_0, *A_3_1, *A_3_2, *A_3_3;
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 0, 0, A_3, &A_3_0 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 0, 1, A_3, &A_3_1 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 1, 0, A_3, &A_3_2 );
    stra_acquire_mpart( ms, ks, rs_A, cs_A, 2, 2, 1, 1, A_3, &A_3_3 );

    ms=md, ks=kd, ns=nd;
    const T *B_0, *B_1, *B_2, *B_3;
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 0, 0, B, &B_0 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 0, 1, B, &B_1 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 1, 0, B, &B_2 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 1, 1, B, &B_3 );
    ms=ms/2, ks=ks/2, ns=ns/2;
    const T *B_0_0, *B_0_1, *B_0_2, *B_0_3;
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 0, 0, B_0, &B_0_0 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 0, 1, B_0, &B_0_1 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 1, 0, B_0, &B_0_2 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 1, 1, B_0, &B_0_3 );
    const T *B_1_0, *B_1_1, *B_1_2, *B_1_3;
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 0, 0, B_1, &B_1_0 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 0, 1, B_1, &B_1_1 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 1, 0, B_1, &B_1_2 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 1, 1, B_1, &B_1_3 );
    const T *B_2_0, *B_2_1, *B_2_2, *B_2_3;
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 0, 0, B_2, &B_2_0 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 0, 1, B_2, &B_2_1 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 1, 0, B_2, &B_2_2 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 1, 1, B_2, &B_2_3 );
    const T *B_3_0, *B_3_1, *B_3_2, *B_3_3;
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 0, 0, B_3, &B_3_0 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 0, 1, B_3, &B_3_1 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 1, 0, B_3, &B_3_2 );
    stra_acquire_mpart( ks, ns, rs_B, cs_B, 2, 2, 1, 1, B_3, &B_3_3 );

    ms=md, ks=kd, ns=nd;
    T *C_0, *C_1, *C_2, *C_3;
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 0, 0, C, &C_0 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 0, 1, C, &C_1 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 1, 0, C, &C_2 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 1, 1, C, &C_3 );
    ms=ms/2, ks=ks/2, ns=ns/2;
    T *C_0_0, *C_0_1, *C_0_2, *C_0_3;
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 0, 0, C_0, &C_0_0 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 0, 1, C_0, &C_0_1 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 1, 0, C_0, &C_0_2 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 1, 1, C_0, &C_0_3 );
    T *C_1_0, *C_1_1, *C_1_2, *C_1_3;
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 0, 0, C_1, &C_1_0 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 0, 1, C_1, &C_1_1 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 1, 0, C_1, &C_1_2 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 1, 1, C_1, &C_1_3 );
    T *C_2_0, *C_2_1, *C_2_2, *C_2_3;
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 0, 0, C_2, &C_2_0 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 0, 1, C_2, &C_2_1 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 1, 0, C_2, &C_2_2 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 1, 1, C_2, &C_2_3 );
    T *C_3_0, *C_3_1, *C_3_2, *C_3_3;
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 0, 0, C_3, &C_3_0 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 0, 1, C_3, &C_3_1 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 1, 0, C_3, &C_3_2 );
    stra_acquire_mpart( ms, ns, rs_C, cs_C, 2, 2, 1, 1, C_3, &C_3_3 );

    ms=ms/2, ks=ks/2, ns=ns/2;


    // M0 = (1.0 * A_0_0 + 1.0 * A_0_3 + 1.0 * A_3_0 + 1.0 * A_3_3) * (1.0 * B_0_0 + 1.0 * B_0_3 + 1.0 * B_3_0 + 1.0 * B_3_3);  C_0_0 += 1.0 * M0;  C_0_3 += 1.0 * M0;  C_3_0 += 1.0 * M0;  C_3_3 += 1.0 * M0;
    stra_matrix_view<T,4> Av0({ms, ks}, {const_cast<T*>(A_0_0), const_cast<T*>(A_0_3), const_cast<T*>(A_3_0), const_cast<T*>(A_3_3)}, {1.0, 1.0, 1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv0({ks, ns}, {const_cast<T*>(B_0_0), const_cast<T*>(B_0_3), const_cast<T*>(B_3_0), const_cast<T*>(B_3_3)}, {1.0, 1.0, 1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv0({ms, ns}, {const_cast<T*>(C_0_0), const_cast<T*>(C_0_3), const_cast<T*>(C_3_0), const_cast<T*>(C_3_3)}, {1.0, 1.0, 1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av0, Bv0, beta, Cv0);
    comm.barrier();

    // M1 = (1.0 * A_0_2 + 1.0 * A_0_3 + 1.0 * A_3_2 + 1.0 * A_3_3) * (1.0 * B_0_0 + 1.0 * B_3_0);  C_0_2 += 1.0 * M1;  C_0_3 += -1.0 * M1;  C_3_2 += 1.0 * M1;  C_3_3 += -1.0 * M1;
    stra_matrix_view<T,4> Av1({ms, ks}, {const_cast<T*>(A_0_2), const_cast<T*>(A_0_3), const_cast<T*>(A_3_2), const_cast<T*>(A_3_3)}, {1.0, 1.0, 1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv1({ks, ns}, {const_cast<T*>(B_0_0), const_cast<T*>(B_3_0)}, {1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv1({ms, ns}, {const_cast<T*>(C_0_2), const_cast<T*>(C_0_3), const_cast<T*>(C_3_2), const_cast<T*>(C_3_3)}, {1.0, -1.0, 1.0, -1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av1, Bv1, beta, Cv1);
    comm.barrier();

    // M2 = (1.0 * A_0_0 + 1.0 * A_3_0) * (1.0 * B_0_1 + -1.0 * B_0_3 + 1.0 * B_3_1 + -1.0 * B_3_3);  C_0_1 += 1.0 * M2;  C_0_3 += 1.0 * M2;  C_3_1 += 1.0 * M2;  C_3_3 += 1.0 * M2;
    stra_matrix_view<T,2> Av2({ms, ks}, {const_cast<T*>(A_0_0), const_cast<T*>(A_3_0)}, {1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv2({ks, ns}, {const_cast<T*>(B_0_1), const_cast<T*>(B_0_3), const_cast<T*>(B_3_1), const_cast<T*>(B_3_3)}, {1.0, -1.0, 1.0, -1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv2({ms, ns}, {const_cast<T*>(C_0_1), const_cast<T*>(C_0_3), const_cast<T*>(C_3_1), const_cast<T*>(C_3_3)}, {1.0, 1.0, 1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av2, Bv2, beta, Cv2);
    comm.barrier();

    // M3 = (1.0 * A_0_3 + 1.0 * A_3_3) * (-1.0 * B_0_0 + 1.0 * B_0_2 + -1.0 * B_3_0 + 1.0 * B_3_2);  C_0_0 += 1.0 * M3;  C_0_2 += 1.0 * M3;  C_3_0 += 1.0 * M3;  C_3_2 += 1.0 * M3;
    stra_matrix_view<T,2> Av3({ms, ks}, {const_cast<T*>(A_0_3), const_cast<T*>(A_3_3)}, {1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv3({ks, ns}, {const_cast<T*>(B_0_0), const_cast<T*>(B_0_2), const_cast<T*>(B_3_0), const_cast<T*>(B_3_2)}, {-1.0, 1.0, -1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv3({ms, ns}, {const_cast<T*>(C_0_0), const_cast<T*>(C_0_2), const_cast<T*>(C_3_0), const_cast<T*>(C_3_2)}, {1.0, 1.0, 1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av3, Bv3, beta, Cv3);
    comm.barrier();

    // M4 = (1.0 * A_0_0 + 1.0 * A_0_1 + 1.0 * A_3_0 + 1.0 * A_3_1) * (1.0 * B_0_3 + 1.0 * B_3_3);  C_0_0 += -1.0 * M4;  C_0_1 += 1.0 * M4;  C_3_0 += -1.0 * M4;  C_3_1 += 1.0 * M4;
    stra_matrix_view<T,4> Av4({ms, ks}, {const_cast<T*>(A_0_0), const_cast<T*>(A_0_1), const_cast<T*>(A_3_0), const_cast<T*>(A_3_1)}, {1.0, 1.0, 1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv4({ks, ns}, {const_cast<T*>(B_0_3), const_cast<T*>(B_3_3)}, {1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv4({ms, ns}, {const_cast<T*>(C_0_0), const_cast<T*>(C_0_1), const_cast<T*>(C_3_0), const_cast<T*>(C_3_1)}, {-1.0, 1.0, -1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av4, Bv4, beta, Cv4);
    comm.barrier();

    // M5 = (-1.0 * A_0_0 + 1.0 * A_0_2 + -1.0 * A_3_0 + 1.0 * A_3_2) * (1.0 * B_0_0 + 1.0 * B_0_1 + 1.0 * B_3_0 + 1.0 * B_3_1);  C_0_3 += 1.0 * M5;  C_3_3 += 1.0 * M5;
    stra_matrix_view<T,4> Av5({ms, ks}, {const_cast<T*>(A_0_0), const_cast<T*>(A_0_2), const_cast<T*>(A_3_0), const_cast<T*>(A_3_2)}, {-1.0, 1.0, -1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv5({ks, ns}, {const_cast<T*>(B_0_0), const_cast<T*>(B_0_1), const_cast<T*>(B_3_0), const_cast<T*>(B_3_1)}, {1.0, 1.0, 1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv5({ms, ns}, {const_cast<T*>(C_0_3), const_cast<T*>(C_3_3)}, {1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av5, Bv5, beta, Cv5);
    comm.barrier();

    // M6 = (1.0 * A_0_1 + -1.0 * A_0_3 + 1.0 * A_3_1 + -1.0 * A_3_3) * (1.0 * B_0_2 + 1.0 * B_0_3 + 1.0 * B_3_2 + 1.0 * B_3_3);  C_0_0 += 1.0 * M6;  C_3_0 += 1.0 * M6;
    stra_matrix_view<T,4> Av6({ms, ks}, {const_cast<T*>(A_0_1), const_cast<T*>(A_0_3), const_cast<T*>(A_3_1), const_cast<T*>(A_3_3)}, {1.0, -1.0, 1.0, -1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv6({ks, ns}, {const_cast<T*>(B_0_2), const_cast<T*>(B_0_3), const_cast<T*>(B_3_2), const_cast<T*>(B_3_3)}, {1.0, 1.0, 1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv6({ms, ns}, {const_cast<T*>(C_0_0), const_cast<T*>(C_3_0)}, {1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av6, Bv6, beta, Cv6);
    comm.barrier();

    // M7 = (1.0 * A_2_0 + 1.0 * A_2_3 + 1.0 * A_3_0 + 1.0 * A_3_3) * (1.0 * B_0_0 + 1.0 * B_0_3);  C_2_0 += 1.0 * M7;  C_2_3 += 1.0 * M7;  C_3_0 += -1.0 * M7;  C_3_3 += -1.0 * M7;
    stra_matrix_view<T,4> Av7({ms, ks}, {const_cast<T*>(A_2_0), const_cast<T*>(A_2_3), const_cast<T*>(A_3_0), const_cast<T*>(A_3_3)}, {1.0, 1.0, 1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv7({ks, ns}, {const_cast<T*>(B_0_0), const_cast<T*>(B_0_3)}, {1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv7({ms, ns}, {const_cast<T*>(C_2_0), const_cast<T*>(C_2_3), const_cast<T*>(C_3_0), const_cast<T*>(C_3_3)}, {1.0, 1.0, -1.0, -1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av7, Bv7, beta, Cv7);
    comm.barrier();

    // M8 = (1.0 * A_2_2 + 1.0 * A_2_3 + 1.0 * A_3_2 + 1.0 * A_3_3) * (1.0 * B_0_0);  C_2_2 += 1.0 * M8;  C_2_3 += -1.0 * M8;  C_3_2 += -1.0 * M8;  C_3_3 += 1.0 * M8;
    stra_matrix_view<T,4> Av8({ms, ks}, {const_cast<T*>(A_2_2), const_cast<T*>(A_2_3), const_cast<T*>(A_3_2), const_cast<T*>(A_3_3)}, {1.0, 1.0, 1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,1> Bv8({ks, ns}, {const_cast<T*>(B_0_0)}, {1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv8({ms, ns}, {const_cast<T*>(C_2_2), const_cast<T*>(C_2_3), const_cast<T*>(C_3_2), const_cast<T*>(C_3_3)}, {1.0, -1.0, -1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av8, Bv8, beta, Cv8);
    comm.barrier();

    // M9 = (1.0 * A_2_0 + 1.0 * A_3_0) * (1.0 * B_0_1 + -1.0 * B_0_3);  C_2_1 += 1.0 * M9;  C_2_3 += 1.0 * M9;  C_3_1 += -1.0 * M9;  C_3_3 += -1.0 * M9;
    stra_matrix_view<T,2> Av9({ms, ks}, {const_cast<T*>(A_2_0), const_cast<T*>(A_3_0)}, {1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv9({ks, ns}, {const_cast<T*>(B_0_1), const_cast<T*>(B_0_3)}, {1.0, -1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv9({ms, ns}, {const_cast<T*>(C_2_1), const_cast<T*>(C_2_3), const_cast<T*>(C_3_1), const_cast<T*>(C_3_3)}, {1.0, 1.0, -1.0, -1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av9, Bv9, beta, Cv9);
    comm.barrier();

    // M10 = (1.0 * A_2_3 + 1.0 * A_3_3) * (-1.0 * B_0_0 + 1.0 * B_0_2);  C_2_0 += 1.0 * M10;  C_2_2 += 1.0 * M10;  C_3_0 += -1.0 * M10;  C_3_2 += -1.0 * M10;
    stra_matrix_view<T,2> Av10({ms, ks}, {const_cast<T*>(A_2_3), const_cast<T*>(A_3_3)}, {1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv10({ks, ns}, {const_cast<T*>(B_0_0), const_cast<T*>(B_0_2)}, {-1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv10({ms, ns}, {const_cast<T*>(C_2_0), const_cast<T*>(C_2_2), const_cast<T*>(C_3_0), const_cast<T*>(C_3_2)}, {1.0, 1.0, -1.0, -1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av10, Bv10, beta, Cv10);
    comm.barrier();

    // M11 = (1.0 * A_2_0 + 1.0 * A_2_1 + 1.0 * A_3_0 + 1.0 * A_3_1) * (1.0 * B_0_3);  C_2_0 += -1.0 * M11;  C_2_1 += 1.0 * M11;  C_3_0 += 1.0 * M11;  C_3_1 += -1.0 * M11;
    stra_matrix_view<T,4> Av11({ms, ks}, {const_cast<T*>(A_2_0), const_cast<T*>(A_2_1), const_cast<T*>(A_3_0), const_cast<T*>(A_3_1)}, {1.0, 1.0, 1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,1> Bv11({ks, ns}, {const_cast<T*>(B_0_3)}, {1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv11({ms, ns}, {const_cast<T*>(C_2_0), const_cast<T*>(C_2_1), const_cast<T*>(C_3_0), const_cast<T*>(C_3_1)}, {-1.0, 1.0, 1.0, -1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av11, Bv11, beta, Cv11);
    comm.barrier();

    // M12 = (-1.0 * A_2_0 + 1.0 * A_2_2 + -1.0 * A_3_0 + 1.0 * A_3_2) * (1.0 * B_0_0 + 1.0 * B_0_1);  C_2_3 += 1.0 * M12;  C_3_3 += -1.0 * M12;
    stra_matrix_view<T,4> Av12({ms, ks}, {const_cast<T*>(A_2_0), const_cast<T*>(A_2_2), const_cast<T*>(A_3_0), const_cast<T*>(A_3_2)}, {-1.0, 1.0, -1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv12({ks, ns}, {const_cast<T*>(B_0_0), const_cast<T*>(B_0_1)}, {1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv12({ms, ns}, {const_cast<T*>(C_2_3), const_cast<T*>(C_3_3)}, {1.0, -1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av12, Bv12, beta, Cv12);
    comm.barrier();

    // M13 = (1.0 * A_2_1 + -1.0 * A_2_3 + 1.0 * A_3_1 + -1.0 * A_3_3) * (1.0 * B_0_2 + 1.0 * B_0_3);  C_2_0 += 1.0 * M13;  C_3_0 += -1.0 * M13;
    stra_matrix_view<T,4> Av13({ms, ks}, {const_cast<T*>(A_2_1), const_cast<T*>(A_2_3), const_cast<T*>(A_3_1), const_cast<T*>(A_3_3)}, {1.0, -1.0, 1.0, -1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv13({ks, ns}, {const_cast<T*>(B_0_2), const_cast<T*>(B_0_3)}, {1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv13({ms, ns}, {const_cast<T*>(C_2_0), const_cast<T*>(C_3_0)}, {1.0, -1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av13, Bv13, beta, Cv13);
    comm.barrier();

    // M14 = (1.0 * A_0_0 + 1.0 * A_0_3) * (1.0 * B_1_0 + 1.0 * B_1_3 + -1.0 * B_3_0 + -1.0 * B_3_3);  C_1_0 += 1.0 * M14;  C_1_3 += 1.0 * M14;  C_3_0 += 1.0 * M14;  C_3_3 += 1.0 * M14;
    stra_matrix_view<T,2> Av14({ms, ks}, {const_cast<T*>(A_0_0), const_cast<T*>(A_0_3)}, {1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv14({ks, ns}, {const_cast<T*>(B_1_0), const_cast<T*>(B_1_3), const_cast<T*>(B_3_0), const_cast<T*>(B_3_3)}, {1.0, 1.0, -1.0, -1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv14({ms, ns}, {const_cast<T*>(C_1_0), const_cast<T*>(C_1_3), const_cast<T*>(C_3_0), const_cast<T*>(C_3_3)}, {1.0, 1.0, 1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av14, Bv14, beta, Cv14);
    comm.barrier();

    // M15 = (1.0 * A_0_2 + 1.0 * A_0_3) * (1.0 * B_1_0 + -1.0 * B_3_0);  C_1_2 += 1.0 * M15;  C_1_3 += -1.0 * M15;  C_3_2 += 1.0 * M15;  C_3_3 += -1.0 * M15;
    stra_matrix_view<T,2> Av15({ms, ks}, {const_cast<T*>(A_0_2), const_cast<T*>(A_0_3)}, {1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv15({ks, ns}, {const_cast<T*>(B_1_0), const_cast<T*>(B_3_0)}, {1.0, -1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv15({ms, ns}, {const_cast<T*>(C_1_2), const_cast<T*>(C_1_3), const_cast<T*>(C_3_2), const_cast<T*>(C_3_3)}, {1.0, -1.0, 1.0, -1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av15, Bv15, beta, Cv15);
    comm.barrier();

    // M16 = (1.0 * A_0_0) * (1.0 * B_1_1 + -1.0 * B_1_3 + -1.0 * B_3_1 + 1.0 * B_3_3);  C_1_1 += 1.0 * M16;  C_1_3 += 1.0 * M16;  C_3_1 += 1.0 * M16;  C_3_3 += 1.0 * M16;
    stra_matrix_view<T,1> Av16({ms, ks}, {const_cast<T*>(A_0_0)}, {1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv16({ks, ns}, {const_cast<T*>(B_1_1), const_cast<T*>(B_1_3), const_cast<T*>(B_3_1), const_cast<T*>(B_3_3)}, {1.0, -1.0, -1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv16({ms, ns}, {const_cast<T*>(C_1_1), const_cast<T*>(C_1_3), const_cast<T*>(C_3_1), const_cast<T*>(C_3_3)}, {1.0, 1.0, 1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av16, Bv16, beta, Cv16);
    comm.barrier();

    // M17 = (1.0 * A_0_3) * (-1.0 * B_1_0 + 1.0 * B_1_2 + 1.0 * B_3_0 + -1.0 * B_3_2);  C_1_0 += 1.0 * M17;  C_1_2 += 1.0 * M17;  C_3_0 += 1.0 * M17;  C_3_2 += 1.0 * M17;
    stra_matrix_view<T,1> Av17({ms, ks}, {const_cast<T*>(A_0_3)}, {1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv17({ks, ns}, {const_cast<T*>(B_1_0), const_cast<T*>(B_1_2), const_cast<T*>(B_3_0), const_cast<T*>(B_3_2)}, {-1.0, 1.0, 1.0, -1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv17({ms, ns}, {const_cast<T*>(C_1_0), const_cast<T*>(C_1_2), const_cast<T*>(C_3_0), const_cast<T*>(C_3_2)}, {1.0, 1.0, 1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av17, Bv17, beta, Cv17);
    comm.barrier();

    // M18 = (1.0 * A_0_0 + 1.0 * A_0_1) * (1.0 * B_1_3 + -1.0 * B_3_3);  C_1_0 += -1.0 * M18;  C_1_1 += 1.0 * M18;  C_3_0 += -1.0 * M18;  C_3_1 += 1.0 * M18;
    stra_matrix_view<T,2> Av18({ms, ks}, {const_cast<T*>(A_0_0), const_cast<T*>(A_0_1)}, {1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv18({ks, ns}, {const_cast<T*>(B_1_3), const_cast<T*>(B_3_3)}, {1.0, -1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv18({ms, ns}, {const_cast<T*>(C_1_0), const_cast<T*>(C_1_1), const_cast<T*>(C_3_0), const_cast<T*>(C_3_1)}, {-1.0, 1.0, -1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av18, Bv18, beta, Cv18);
    comm.barrier();

    // M19 = (-1.0 * A_0_0 + 1.0 * A_0_2) * (1.0 * B_1_0 + 1.0 * B_1_1 + -1.0 * B_3_0 + -1.0 * B_3_1);  C_1_3 += 1.0 * M19;  C_3_3 += 1.0 * M19;
    stra_matrix_view<T,2> Av19({ms, ks}, {const_cast<T*>(A_0_0), const_cast<T*>(A_0_2)}, {-1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv19({ks, ns}, {const_cast<T*>(B_1_0), const_cast<T*>(B_1_1), const_cast<T*>(B_3_0), const_cast<T*>(B_3_1)}, {1.0, 1.0, -1.0, -1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv19({ms, ns}, {const_cast<T*>(C_1_3), const_cast<T*>(C_3_3)}, {1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av19, Bv19, beta, Cv19);
    comm.barrier();

    // M20 = (1.0 * A_0_1 + -1.0 * A_0_3) * (1.0 * B_1_2 + 1.0 * B_1_3 + -1.0 * B_3_2 + -1.0 * B_3_3);  C_1_0 += 1.0 * M20;  C_3_0 += 1.0 * M20;
    stra_matrix_view<T,2> Av20({ms, ks}, {const_cast<T*>(A_0_1), const_cast<T*>(A_0_3)}, {1.0, -1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv20({ks, ns}, {const_cast<T*>(B_1_2), const_cast<T*>(B_1_3), const_cast<T*>(B_3_2), const_cast<T*>(B_3_3)}, {1.0, 1.0, -1.0, -1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv20({ms, ns}, {const_cast<T*>(C_1_0), const_cast<T*>(C_3_0)}, {1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av20, Bv20, beta, Cv20);
    comm.barrier();

    // M21 = (1.0 * A_3_0 + 1.0 * A_3_3) * (-1.0 * B_0_0 + -1.0 * B_0_3 + 1.0 * B_2_0 + 1.0 * B_2_3);  C_0_0 += 1.0 * M21;  C_0_3 += 1.0 * M21;  C_2_0 += 1.0 * M21;  C_2_3 += 1.0 * M21;
    stra_matrix_view<T,2> Av21({ms, ks}, {const_cast<T*>(A_3_0), const_cast<T*>(A_3_3)}, {1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv21({ks, ns}, {const_cast<T*>(B_0_0), const_cast<T*>(B_0_3), const_cast<T*>(B_2_0), const_cast<T*>(B_2_3)}, {-1.0, -1.0, 1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv21({ms, ns}, {const_cast<T*>(C_0_0), const_cast<T*>(C_0_3), const_cast<T*>(C_2_0), const_cast<T*>(C_2_3)}, {1.0, 1.0, 1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av21, Bv21, beta, Cv21);
    comm.barrier();

    // M22 = (1.0 * A_3_2 + 1.0 * A_3_3) * (-1.0 * B_0_0 + 1.0 * B_2_0);  C_0_2 += 1.0 * M22;  C_0_3 += -1.0 * M22;  C_2_2 += 1.0 * M22;  C_2_3 += -1.0 * M22;
    stra_matrix_view<T,2> Av22({ms, ks}, {const_cast<T*>(A_3_2), const_cast<T*>(A_3_3)}, {1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv22({ks, ns}, {const_cast<T*>(B_0_0), const_cast<T*>(B_2_0)}, {-1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv22({ms, ns}, {const_cast<T*>(C_0_2), const_cast<T*>(C_0_3), const_cast<T*>(C_2_2), const_cast<T*>(C_2_3)}, {1.0, -1.0, 1.0, -1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av22, Bv22, beta, Cv22);
    comm.barrier();

    // M23 = (1.0 * A_3_0) * (-1.0 * B_0_1 + 1.0 * B_0_3 + 1.0 * B_2_1 + -1.0 * B_2_3);  C_0_1 += 1.0 * M23;  C_0_3 += 1.0 * M23;  C_2_1 += 1.0 * M23;  C_2_3 += 1.0 * M23;
    stra_matrix_view<T,1> Av23({ms, ks}, {const_cast<T*>(A_3_0)}, {1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv23({ks, ns}, {const_cast<T*>(B_0_1), const_cast<T*>(B_0_3), const_cast<T*>(B_2_1), const_cast<T*>(B_2_3)}, {-1.0, 1.0, 1.0, -1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv23({ms, ns}, {const_cast<T*>(C_0_1), const_cast<T*>(C_0_3), const_cast<T*>(C_2_1), const_cast<T*>(C_2_3)}, {1.0, 1.0, 1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av23, Bv23, beta, Cv23);
    comm.barrier();

    // M24 = (1.0 * A_3_3) * (1.0 * B_0_0 + -1.0 * B_0_2 + -1.0 * B_2_0 + 1.0 * B_2_2);  C_0_0 += 1.0 * M24;  C_0_2 += 1.0 * M24;  C_2_0 += 1.0 * M24;  C_2_2 += 1.0 * M24;
    stra_matrix_view<T,1> Av24({ms, ks}, {const_cast<T*>(A_3_3)}, {1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv24({ks, ns}, {const_cast<T*>(B_0_0), const_cast<T*>(B_0_2), const_cast<T*>(B_2_0), const_cast<T*>(B_2_2)}, {1.0, -1.0, -1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv24({ms, ns}, {const_cast<T*>(C_0_0), const_cast<T*>(C_0_2), const_cast<T*>(C_2_0), const_cast<T*>(C_2_2)}, {1.0, 1.0, 1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av24, Bv24, beta, Cv24);
    comm.barrier();

    // M25 = (1.0 * A_3_0 + 1.0 * A_3_1) * (-1.0 * B_0_3 + 1.0 * B_2_3);  C_0_0 += -1.0 * M25;  C_0_1 += 1.0 * M25;  C_2_0 += -1.0 * M25;  C_2_1 += 1.0 * M25;
    stra_matrix_view<T,2> Av25({ms, ks}, {const_cast<T*>(A_3_0), const_cast<T*>(A_3_1)}, {1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv25({ks, ns}, {const_cast<T*>(B_0_3), const_cast<T*>(B_2_3)}, {-1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv25({ms, ns}, {const_cast<T*>(C_0_0), const_cast<T*>(C_0_1), const_cast<T*>(C_2_0), const_cast<T*>(C_2_1)}, {-1.0, 1.0, -1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av25, Bv25, beta, Cv25);
    comm.barrier();

    // M26 = (-1.0 * A_3_0 + 1.0 * A_3_2) * (-1.0 * B_0_0 + -1.0 * B_0_1 + 1.0 * B_2_0 + 1.0 * B_2_1);  C_0_3 += 1.0 * M26;  C_2_3 += 1.0 * M26;
    stra_matrix_view<T,2> Av26({ms, ks}, {const_cast<T*>(A_3_0), const_cast<T*>(A_3_2)}, {-1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv26({ks, ns}, {const_cast<T*>(B_0_0), const_cast<T*>(B_0_1), const_cast<T*>(B_2_0), const_cast<T*>(B_2_1)}, {-1.0, -1.0, 1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv26({ms, ns}, {const_cast<T*>(C_0_3), const_cast<T*>(C_2_3)}, {1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av26, Bv26, beta, Cv26);
    comm.barrier();

    // M27 = (1.0 * A_3_1 + -1.0 * A_3_3) * (-1.0 * B_0_2 + -1.0 * B_0_3 + 1.0 * B_2_2 + 1.0 * B_2_3);  C_0_0 += 1.0 * M27;  C_2_0 += 1.0 * M27;
    stra_matrix_view<T,2> Av27({ms, ks}, {const_cast<T*>(A_3_1), const_cast<T*>(A_3_3)}, {1.0, -1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv27({ks, ns}, {const_cast<T*>(B_0_2), const_cast<T*>(B_0_3), const_cast<T*>(B_2_2), const_cast<T*>(B_2_3)}, {-1.0, -1.0, 1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv27({ms, ns}, {const_cast<T*>(C_0_0), const_cast<T*>(C_2_0)}, {1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av27, Bv27, beta, Cv27);
    comm.barrier();

    // M28 = (1.0 * A_0_0 + 1.0 * A_0_3 + 1.0 * A_1_0 + 1.0 * A_1_3) * (1.0 * B_3_0 + 1.0 * B_3_3);  C_0_0 += -1.0 * M28;  C_0_3 += -1.0 * M28;  C_1_0 += 1.0 * M28;  C_1_3 += 1.0 * M28;
    stra_matrix_view<T,4> Av28({ms, ks}, {const_cast<T*>(A_0_0), const_cast<T*>(A_0_3), const_cast<T*>(A_1_0), const_cast<T*>(A_1_3)}, {1.0, 1.0, 1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv28({ks, ns}, {const_cast<T*>(B_3_0), const_cast<T*>(B_3_3)}, {1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv28({ms, ns}, {const_cast<T*>(C_0_0), const_cast<T*>(C_0_3), const_cast<T*>(C_1_0), const_cast<T*>(C_1_3)}, {-1.0, -1.0, 1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av28, Bv28, beta, Cv28);
    comm.barrier();

    // M29 = (1.0 * A_0_2 + 1.0 * A_0_3 + 1.0 * A_1_2 + 1.0 * A_1_3) * (1.0 * B_3_0);  C_0_2 += -1.0 * M29;  C_0_3 += 1.0 * M29;  C_1_2 += 1.0 * M29;  C_1_3 += -1.0 * M29;
    stra_matrix_view<T,4> Av29({ms, ks}, {const_cast<T*>(A_0_2), const_cast<T*>(A_0_3), const_cast<T*>(A_1_2), const_cast<T*>(A_1_3)}, {1.0, 1.0, 1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,1> Bv29({ks, ns}, {const_cast<T*>(B_3_0)}, {1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv29({ms, ns}, {const_cast<T*>(C_0_2), const_cast<T*>(C_0_3), const_cast<T*>(C_1_2), const_cast<T*>(C_1_3)}, {-1.0, 1.0, 1.0, -1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av29, Bv29, beta, Cv29);
    comm.barrier();

    // M30 = (1.0 * A_0_0 + 1.0 * A_1_0) * (1.0 * B_3_1 + -1.0 * B_3_3);  C_0_1 += -1.0 * M30;  C_0_3 += -1.0 * M30;  C_1_1 += 1.0 * M30;  C_1_3 += 1.0 * M30;
    stra_matrix_view<T,2> Av30({ms, ks}, {const_cast<T*>(A_0_0), const_cast<T*>(A_1_0)}, {1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv30({ks, ns}, {const_cast<T*>(B_3_1), const_cast<T*>(B_3_3)}, {1.0, -1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv30({ms, ns}, {const_cast<T*>(C_0_1), const_cast<T*>(C_0_3), const_cast<T*>(C_1_1), const_cast<T*>(C_1_3)}, {-1.0, -1.0, 1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av30, Bv30, beta, Cv30);
    comm.barrier();

    // M31 = (1.0 * A_0_3 + 1.0 * A_1_3) * (-1.0 * B_3_0 + 1.0 * B_3_2);  C_0_0 += -1.0 * M31;  C_0_2 += -1.0 * M31;  C_1_0 += 1.0 * M31;  C_1_2 += 1.0 * M31;
    stra_matrix_view<T,2> Av31({ms, ks}, {const_cast<T*>(A_0_3), const_cast<T*>(A_1_3)}, {1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv31({ks, ns}, {const_cast<T*>(B_3_0), const_cast<T*>(B_3_2)}, {-1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv31({ms, ns}, {const_cast<T*>(C_0_0), const_cast<T*>(C_0_2), const_cast<T*>(C_1_0), const_cast<T*>(C_1_2)}, {-1.0, -1.0, 1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av31, Bv31, beta, Cv31);
    comm.barrier();

    // M32 = (1.0 * A_0_0 + 1.0 * A_0_1 + 1.0 * A_1_0 + 1.0 * A_1_1) * (1.0 * B_3_3);  C_0_0 += 1.0 * M32;  C_0_1 += -1.0 * M32;  C_1_0 += -1.0 * M32;  C_1_1 += 1.0 * M32;
    stra_matrix_view<T,4> Av32({ms, ks}, {const_cast<T*>(A_0_0), const_cast<T*>(A_0_1), const_cast<T*>(A_1_0), const_cast<T*>(A_1_1)}, {1.0, 1.0, 1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,1> Bv32({ks, ns}, {const_cast<T*>(B_3_3)}, {1.0}, {rs_B, cs_B});
    stra_matrix_view<T,4> Cv32({ms, ns}, {const_cast<T*>(C_0_0), const_cast<T*>(C_0_1), const_cast<T*>(C_1_0), const_cast<T*>(C_1_1)}, {1.0, -1.0, -1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av32, Bv32, beta, Cv32);
    comm.barrier();

    // M33 = (-1.0 * A_0_0 + 1.0 * A_0_2 + -1.0 * A_1_0 + 1.0 * A_1_2) * (1.0 * B_3_0 + 1.0 * B_3_1);  C_0_3 += -1.0 * M33;  C_1_3 += 1.0 * M33;
    stra_matrix_view<T,4> Av33({ms, ks}, {const_cast<T*>(A_0_0), const_cast<T*>(A_0_2), const_cast<T*>(A_1_0), const_cast<T*>(A_1_2)}, {-1.0, 1.0, -1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv33({ks, ns}, {const_cast<T*>(B_3_0), const_cast<T*>(B_3_1)}, {1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv33({ms, ns}, {const_cast<T*>(C_0_3), const_cast<T*>(C_1_3)}, {-1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av33, Bv33, beta, Cv33);
    comm.barrier();

    // M34 = (1.0 * A_0_1 + -1.0 * A_0_3 + 1.0 * A_1_1 + -1.0 * A_1_3) * (1.0 * B_3_2 + 1.0 * B_3_3);  C_0_0 += -1.0 * M34;  C_1_0 += 1.0 * M34;
    stra_matrix_view<T,4> Av34({ms, ks}, {const_cast<T*>(A_0_1), const_cast<T*>(A_0_3), const_cast<T*>(A_1_1), const_cast<T*>(A_1_3)}, {1.0, -1.0, 1.0, -1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv34({ks, ns}, {const_cast<T*>(B_3_2), const_cast<T*>(B_3_3)}, {1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv34({ms, ns}, {const_cast<T*>(C_0_0), const_cast<T*>(C_1_0)}, {-1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av34, Bv34, beta, Cv34);
    comm.barrier();

    // M35 = (-1.0 * A_0_0 + -1.0 * A_0_3 + 1.0 * A_2_0 + 1.0 * A_2_3) * (1.0 * B_0_0 + 1.0 * B_0_3 + 1.0 * B_1_0 + 1.0 * B_1_3);  C_3_0 += 1.0 * M35;  C_3_3 += 1.0 * M35;
    stra_matrix_view<T,4> Av35({ms, ks}, {const_cast<T*>(A_0_0), const_cast<T*>(A_0_3), const_cast<T*>(A_2_0), const_cast<T*>(A_2_3)}, {-1.0, -1.0, 1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv35({ks, ns}, {const_cast<T*>(B_0_0), const_cast<T*>(B_0_3), const_cast<T*>(B_1_0), const_cast<T*>(B_1_3)}, {1.0, 1.0, 1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv35({ms, ns}, {const_cast<T*>(C_3_0), const_cast<T*>(C_3_3)}, {1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av35, Bv35, beta, Cv35);
    comm.barrier();

    // M36 = (-1.0 * A_0_2 + -1.0 * A_0_3 + 1.0 * A_2_2 + 1.0 * A_2_3) * (1.0 * B_0_0 + 1.0 * B_1_0);  C_3_2 += 1.0 * M36;  C_3_3 += -1.0 * M36;
    stra_matrix_view<T,4> Av36({ms, ks}, {const_cast<T*>(A_0_2), const_cast<T*>(A_0_3), const_cast<T*>(A_2_2), const_cast<T*>(A_2_3)}, {-1.0, -1.0, 1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv36({ks, ns}, {const_cast<T*>(B_0_0), const_cast<T*>(B_1_0)}, {1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv36({ms, ns}, {const_cast<T*>(C_3_2), const_cast<T*>(C_3_3)}, {1.0, -1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av36, Bv36, beta, Cv36);
    comm.barrier();

    // M37 = (-1.0 * A_0_0 + 1.0 * A_2_0) * (1.0 * B_0_1 + -1.0 * B_0_3 + 1.0 * B_1_1 + -1.0 * B_1_3);  C_3_1 += 1.0 * M37;  C_3_3 += 1.0 * M37;
    stra_matrix_view<T,2> Av37({ms, ks}, {const_cast<T*>(A_0_0), const_cast<T*>(A_2_0)}, {-1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv37({ks, ns}, {const_cast<T*>(B_0_1), const_cast<T*>(B_0_3), const_cast<T*>(B_1_1), const_cast<T*>(B_1_3)}, {1.0, -1.0, 1.0, -1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv37({ms, ns}, {const_cast<T*>(C_3_1), const_cast<T*>(C_3_3)}, {1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av37, Bv37, beta, Cv37);
    comm.barrier();

    // M38 = (-1.0 * A_0_3 + 1.0 * A_2_3) * (-1.0 * B_0_0 + 1.0 * B_0_2 + -1.0 * B_1_0 + 1.0 * B_1_2);  C_3_0 += 1.0 * M38;  C_3_2 += 1.0 * M38;
    stra_matrix_view<T,2> Av38({ms, ks}, {const_cast<T*>(A_0_3), const_cast<T*>(A_2_3)}, {-1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv38({ks, ns}, {const_cast<T*>(B_0_0), const_cast<T*>(B_0_2), const_cast<T*>(B_1_0), const_cast<T*>(B_1_2)}, {-1.0, 1.0, -1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv38({ms, ns}, {const_cast<T*>(C_3_0), const_cast<T*>(C_3_2)}, {1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av38, Bv38, beta, Cv38);
    comm.barrier();

    // M39 = (-1.0 * A_0_0 + -1.0 * A_0_1 + 1.0 * A_2_0 + 1.0 * A_2_1) * (1.0 * B_0_3 + 1.0 * B_1_3);  C_3_0 += -1.0 * M39;  C_3_1 += 1.0 * M39;
    stra_matrix_view<T,4> Av39({ms, ks}, {const_cast<T*>(A_0_0), const_cast<T*>(A_0_1), const_cast<T*>(A_2_0), const_cast<T*>(A_2_1)}, {-1.0, -1.0, 1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv39({ks, ns}, {const_cast<T*>(B_0_3), const_cast<T*>(B_1_3)}, {1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv39({ms, ns}, {const_cast<T*>(C_3_0), const_cast<T*>(C_3_1)}, {-1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av39, Bv39, beta, Cv39);
    comm.barrier();

    // M40 = (1.0 * A_0_0 + -1.0 * A_0_2 + -1.0 * A_2_0 + 1.0 * A_2_2) * (1.0 * B_0_0 + 1.0 * B_0_1 + 1.0 * B_1_0 + 1.0 * B_1_1);  C_3_3 += 1.0 * M40;
    stra_matrix_view<T,4> Av40({ms, ks}, {const_cast<T*>(A_0_0), const_cast<T*>(A_0_2), const_cast<T*>(A_2_0), const_cast<T*>(A_2_2)}, {1.0, -1.0, -1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv40({ks, ns}, {const_cast<T*>(B_0_0), const_cast<T*>(B_0_1), const_cast<T*>(B_1_0), const_cast<T*>(B_1_1)}, {1.0, 1.0, 1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,1> Cv40({ms, ns}, {const_cast<T*>(C_3_3)}, {1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av40, Bv40, beta, Cv40);
    comm.barrier();

    // M41 = (-1.0 * A_0_1 + 1.0 * A_0_3 + 1.0 * A_2_1 + -1.0 * A_2_3) * (1.0 * B_0_2 + 1.0 * B_0_3 + 1.0 * B_1_2 + 1.0 * B_1_3);  C_3_0 += 1.0 * M41;
    stra_matrix_view<T,4> Av41({ms, ks}, {const_cast<T*>(A_0_1), const_cast<T*>(A_0_3), const_cast<T*>(A_2_1), const_cast<T*>(A_2_3)}, {-1.0, 1.0, 1.0, -1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv41({ks, ns}, {const_cast<T*>(B_0_2), const_cast<T*>(B_0_3), const_cast<T*>(B_1_2), const_cast<T*>(B_1_3)}, {1.0, 1.0, 1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,1> Cv41({ms, ns}, {const_cast<T*>(C_3_0)}, {1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av41, Bv41, beta, Cv41);
    comm.barrier();

    // M42 = (1.0 * A_1_0 + 1.0 * A_1_3 + -1.0 * A_3_0 + -1.0 * A_3_3) * (1.0 * B_2_0 + 1.0 * B_2_3 + 1.0 * B_3_0 + 1.0 * B_3_3);  C_0_0 += 1.0 * M42;  C_0_3 += 1.0 * M42;
    stra_matrix_view<T,4> Av42({ms, ks}, {const_cast<T*>(A_1_0), const_cast<T*>(A_1_3), const_cast<T*>(A_3_0), const_cast<T*>(A_3_3)}, {1.0, 1.0, -1.0, -1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv42({ks, ns}, {const_cast<T*>(B_2_0), const_cast<T*>(B_2_3), const_cast<T*>(B_3_0), const_cast<T*>(B_3_3)}, {1.0, 1.0, 1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv42({ms, ns}, {const_cast<T*>(C_0_0), const_cast<T*>(C_0_3)}, {1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av42, Bv42, beta, Cv42);
    comm.barrier();

    // M43 = (1.0 * A_1_2 + 1.0 * A_1_3 + -1.0 * A_3_2 + -1.0 * A_3_3) * (1.0 * B_2_0 + 1.0 * B_3_0);  C_0_2 += 1.0 * M43;  C_0_3 += -1.0 * M43;
    stra_matrix_view<T,4> Av43({ms, ks}, {const_cast<T*>(A_1_2), const_cast<T*>(A_1_3), const_cast<T*>(A_3_2), const_cast<T*>(A_3_3)}, {1.0, 1.0, -1.0, -1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv43({ks, ns}, {const_cast<T*>(B_2_0), const_cast<T*>(B_3_0)}, {1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv43({ms, ns}, {const_cast<T*>(C_0_2), const_cast<T*>(C_0_3)}, {1.0, -1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av43, Bv43, beta, Cv43);
    comm.barrier();

    // M44 = (1.0 * A_1_0 + -1.0 * A_3_0) * (1.0 * B_2_1 + -1.0 * B_2_3 + 1.0 * B_3_1 + -1.0 * B_3_3);  C_0_1 += 1.0 * M44;  C_0_3 += 1.0 * M44;
    stra_matrix_view<T,2> Av44({ms, ks}, {const_cast<T*>(A_1_0), const_cast<T*>(A_3_0)}, {1.0, -1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv44({ks, ns}, {const_cast<T*>(B_2_1), const_cast<T*>(B_2_3), const_cast<T*>(B_3_1), const_cast<T*>(B_3_3)}, {1.0, -1.0, 1.0, -1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv44({ms, ns}, {const_cast<T*>(C_0_1), const_cast<T*>(C_0_3)}, {1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av44, Bv44, beta, Cv44);
    comm.barrier();

    // M45 = (1.0 * A_1_3 + -1.0 * A_3_3) * (-1.0 * B_2_0 + 1.0 * B_2_2 + -1.0 * B_3_0 + 1.0 * B_3_2);  C_0_0 += 1.0 * M45;  C_0_2 += 1.0 * M45;
    stra_matrix_view<T,2> Av45({ms, ks}, {const_cast<T*>(A_1_3), const_cast<T*>(A_3_3)}, {1.0, -1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv45({ks, ns}, {const_cast<T*>(B_2_0), const_cast<T*>(B_2_2), const_cast<T*>(B_3_0), const_cast<T*>(B_3_2)}, {-1.0, 1.0, -1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv45({ms, ns}, {const_cast<T*>(C_0_0), const_cast<T*>(C_0_2)}, {1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av45, Bv45, beta, Cv45);
    comm.barrier();

    // M46 = (1.0 * A_1_0 + 1.0 * A_1_1 + -1.0 * A_3_0 + -1.0 * A_3_1) * (1.0 * B_2_3 + 1.0 * B_3_3);  C_0_0 += -1.0 * M46;  C_0_1 += 1.0 * M46;
    stra_matrix_view<T,4> Av46({ms, ks}, {const_cast<T*>(A_1_0), const_cast<T*>(A_1_1), const_cast<T*>(A_3_0), const_cast<T*>(A_3_1)}, {1.0, 1.0, -1.0, -1.0}, {rs_A, cs_A});
    stra_matrix_view<T,2> Bv46({ks, ns}, {const_cast<T*>(B_2_3), const_cast<T*>(B_3_3)}, {1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,2> Cv46({ms, ns}, {const_cast<T*>(C_0_0), const_cast<T*>(C_0_1)}, {-1.0, 1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av46, Bv46, beta, Cv46);
    comm.barrier();

    // M47 = (-1.0 * A_1_0 + 1.0 * A_1_2 + 1.0 * A_3_0 + -1.0 * A_3_2) * (1.0 * B_2_0 + 1.0 * B_2_1 + 1.0 * B_3_0 + 1.0 * B_3_1);  C_0_3 += 1.0 * M47;
    stra_matrix_view<T,4> Av47({ms, ks}, {const_cast<T*>(A_1_0), const_cast<T*>(A_1_2), const_cast<T*>(A_3_0), const_cast<T*>(A_3_2)}, {-1.0, 1.0, 1.0, -1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv47({ks, ns}, {const_cast<T*>(B_2_0), const_cast<T*>(B_2_1), const_cast<T*>(B_3_0), const_cast<T*>(B_3_1)}, {1.0, 1.0, 1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,1> Cv47({ms, ns}, {const_cast<T*>(C_0_3)}, {1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av47, Bv47, beta, Cv47);
    comm.barrier();

    // M48 = (1.0 * A_1_1 + -1.0 * A_1_3 + -1.0 * A_3_1 + 1.0 * A_3_3) * (1.0 * B_2_2 + 1.0 * B_2_3 + 1.0 * B_3_2 + 1.0 * B_3_3);  C_0_0 += 1.0 * M48;
    stra_matrix_view<T,4> Av48({ms, ks}, {const_cast<T*>(A_1_1), const_cast<T*>(A_1_3), const_cast<T*>(A_3_1), const_cast<T*>(A_3_3)}, {1.0, -1.0, -1.0, 1.0}, {rs_A, cs_A});
    stra_matrix_view<T,4> Bv48({ks, ns}, {const_cast<T*>(B_2_2), const_cast<T*>(B_2_3), const_cast<T*>(B_3_2), const_cast<T*>(B_3_3)}, {1.0, 1.0, 1.0, 1.0}, {rs_B, cs_B});
    stra_matrix_view<T,1> Cv48({ms, ns}, {const_cast<T*>(C_0_0)}, {1.0}, {rs_C, cs_C});
    straprim_ab(comm, cfg, alpha, Av48, Bv48, beta, Cv48);
    comm.barrier();

