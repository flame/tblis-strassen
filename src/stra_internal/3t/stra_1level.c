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

