#ifndef _TBLIS_STRA_IFACE_3M_MULT_H_
#define _TBLIS_STRA_IFACE_3M_MULT_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void stra_tblis_matrix_mult(const tblis_comm* comm, const tblis_config* cfg,
                            const tblis_matrix* A, const tblis_matrix* B,
                            tblis_matrix* C);

void stra_tblis_matrix_mult_naive(const tblis_comm* comm, const tblis_config* cfg,
                                  const tblis_matrix* A, const tblis_matrix* B,
                                  tblis_matrix* C);

void stra_tblis_matrix_mult_ab(const tblis_comm* comm, const tblis_config* cfg,
                                  const tblis_matrix* A, const tblis_matrix* B,
                                  tblis_matrix* C);

#ifdef __cplusplus

}

template <typename T>
void stra_mult(T alpha, const_matrix_view<T> A, const_matrix_view<T> B,
               T beta, matrix_view<T> C)
{
    tblis_matrix A_s(alpha, A);
    tblis_matrix B_s(B);
    tblis_matrix C_s(beta, C);

    stra_tblis_matrix_mult(nullptr, nullptr, &A_s, &B_s, &C_s);
}

template <typename T>
void stra_mult(single_t,
               T alpha, const_matrix_view<T> A, const_matrix_view<T> B,
               T beta, matrix_view<T> C)
{
    tblis_matrix A_s(alpha, A);
    tblis_matrix B_s(B);
    tblis_matrix C_s(beta, C);

    stra_tblis_matrix_mult(tblis_single, nullptr, &A_s, &B_s, &C_s);
}

template <typename T>
void stra_mult(const communicator& comm,
          T alpha, const_matrix_view<T> A, const_matrix_view<T> B,
          T beta, matrix_view<T> C)
{
    tblis_matrix A_s(alpha, A);
    tblis_matrix B_s(B);
    tblis_matrix C_s(beta, C);

    stra_tblis_matrix_mult(comm, nullptr, &A_s, &B_s, &C_s);
}

template <typename T>
void stra_mult_naive(T alpha, const_matrix_view<T> A, const_matrix_view<T> B,
               T beta, matrix_view<T> C)
{
    tblis_matrix A_s(alpha, A);
    tblis_matrix B_s(B);
    tblis_matrix C_s(beta, C);

    stra_tblis_matrix_mult_naive(nullptr, nullptr, &A_s, &B_s, &C_s);
}

template <typename T>
void stra_mult_naive(single_t,
               T alpha, const_matrix_view<T> A, const_matrix_view<T> B,
               T beta, matrix_view<T> C)
{
    tblis_matrix A_s(alpha, A);
    tblis_matrix B_s(B);
    tblis_matrix C_s(beta, C);

    stra_tblis_matrix_mult_naive(tblis_single, nullptr, &A_s, &B_s, &C_s);
}

template <typename T>
void stra_mult_naive(const communicator& comm,
          T alpha, const_matrix_view<T> A, const_matrix_view<T> B,
          T beta, matrix_view<T> C)
{
    tblis_matrix A_s(alpha, A);
    tblis_matrix B_s(B);
    tblis_matrix C_s(beta, C);

    stra_tblis_matrix_mult_naive(comm, nullptr, &A_s, &B_s, &C_s);
}


template <typename T>
void stra_mult_ab(T alpha, const_matrix_view<T> A, const_matrix_view<T> B,
               T beta, matrix_view<T> C)
{
    tblis_matrix A_s(alpha, A);
    tblis_matrix B_s(B);
    tblis_matrix C_s(beta, C);

    stra_tblis_matrix_mult_ab(nullptr, nullptr, &A_s, &B_s, &C_s);
}

template <typename T>
void stra_mult_ab(single_t,
               T alpha, const_matrix_view<T> A, const_matrix_view<T> B,
               T beta, matrix_view<T> C)
{
    tblis_matrix A_s(alpha, A);
    tblis_matrix B_s(B);
    tblis_matrix C_s(beta, C);

    stra_tblis_matrix_mult_ab(tblis_single, nullptr, &A_s, &B_s, &C_s);
}

template <typename T>
void stra_mult_ab(const communicator& comm,
          T alpha, const_matrix_view<T> A, const_matrix_view<T> B,
          T beta, matrix_view<T> C)
{
    tblis_matrix A_s(alpha, A);
    tblis_matrix B_s(B);
    tblis_matrix C_s(beta, C);

    stra_tblis_matrix_mult_ab(comm, nullptr, &A_s, &B_s, &C_s);
}

}

#endif

#endif
