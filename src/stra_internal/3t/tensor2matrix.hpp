#ifndef _TENSOR2MATRIX_HPP_
#define _TENSOR2MATRIX_HPP_

#include "util/basic_types.h"
#include "util/macros.h"
#include "util/thread.h"

#include "matrix/tensor_matrix.hpp"
#include "matrix/stra_tensor_view.hpp"

namespace tblis
{
namespace internal
{

template <bool Forward, typename T>
void add_internal(const communicator& comm,
                  T alpha, tensor_matrix<T>& A,
                  T  beta,   matrix_view<T>& B)
{
    TBLIS_ASSERT(A.length(0) == B.length(0));
    TBLIS_ASSERT(A.length(1) == B.length(1));

    constexpr len_type MB = 8;
    constexpr len_type NB = 8;

    len_type m = A.length(0);
    len_type n = A.length(1);
    stride_type rs_B = B.stride(0);
    stride_type cs_B = B.stride(1);

    std::vector<stride_type> rscat_A(m);
    std::vector<stride_type> cscat_A(n);
    std::vector<stride_type> rbs_A(m);
    std::vector<stride_type> cbs_A(n);

    A.fill_block_scatter(0, rscat_A.data(), MB, rbs_A.data());
    A.fill_block_scatter(1, cscat_A.data(), NB, cbs_A.data());

    len_type m_min, m_max, n_min, n_max;
    std::tie(m_min, m_max, std::ignore,
             n_min, n_max, std::ignore) =
        comm.distribute_over_threads_2d(m, n, MB, NB);

    stride_type* rscat_A_;
    stride_type* rbs_A_;
    stride_type* cscat_A_;
    stride_type* cbs_A_;

    if (rs_B == 1)
    {
        for (len_type j = n_min;j < n_max;j += NB)
        {
            len_type n_loc = std::min(NB, n_max-j);

            rscat_A_ = rscat_A.data() + m_min;
            rbs_A_ = rbs_A.data() + m_min/MB;
            cscat_A_ = cscat_A.data() + j;
            cbs_A_ = cbs_A.data() + j/NB;

            T* p_B = B.data() + m_min + cs_B*j;

            stride_type cs_A = *cbs_A_;

            T* p_A = A.data();

            TBLIS_SPECIAL_CASE(n_loc == NB,

            if (cs_A)
            {
                p_A += *cscat_A_;

                TBLIS_SPECIAL_CASE(cs_A == 1,

                for (len_type i = m_min;i < m_max;i += MB)
                {
                    len_type m_loc = std::min(MB, m_max-i);

                    stride_type rs_A = *rbs_A_;
                    stride_type off_A = *rscat_A_;

                    TBLIS_SPECIAL_CASE(m_loc == MB,

                    if (rs_A)
                    {
                        TBLIS_SPECIAL_CASE(rs_A == 1,

                        for (len_type jr = 0;jr < n_loc;jr++)
                        {
                            for (len_type ir = 0;ir < m_loc;ir++)
                            {
                                if (Forward)
                                {
                                    p_B[ir + cs_B*jr] =
                                        alpha * p_A[rs_A*ir + off_A + cs_A*jr] +
                                        beta * p_B[ir + cs_B*jr];
                                }
                                else
                                {
                                    p_A[rs_A*ir + off_A + cs_A*jr] =
                                        alpha * p_A[rs_A*ir + off_A + cs_A*jr] +
                                        beta * p_B[ir + cs_B*jr];
                                }
                            }
                        }

                        )
                    }
                    else
                    {
                        for (len_type jr = 0;jr < n_loc;jr++)
                        {
                            for (len_type ir = 0;ir < m_loc;ir++)
                            {
                                if (Forward)
                                {
                                    p_B[ir + cs_B*jr] =
                                        alpha * p_A[rscat_A_[ir] + cs_A*jr] +
                                        beta * p_B[ir + cs_B*jr];
                                }
                                else
                                {
                                    p_A[rscat_A_[ir] + cs_A*jr] =
                                        alpha * p_A[rscat_A_[ir] + cs_A*jr] +
                                        beta * p_B[ir + cs_B*jr];
                                }
                            }
                        }
                    }

                    )

                    rscat_A_ += MB;
                    rbs_A_++;

                    p_B += MB;
                }

                )
            }
            else
            {
                for (len_type i = m_min;i < m_max;i += MB)
                {
                    len_type m_loc = std::min(MB, m_max-i);

                    stride_type rs_A = *rbs_A_;
                    stride_type off_A = *rscat_A_;

                    TBLIS_SPECIAL_CASE(m_loc == MB,

                    if (rs_A)
                    {
                        TBLIS_SPECIAL_CASE(rs_A == 1,

                        for (len_type jr = 0;jr < n_loc;jr++)
                        {
                            for (len_type ir = 0;ir < m_loc;ir++)
                            {
                                if (Forward)
                                {
                                    p_B[ir + cs_B*jr] =
                                        alpha * p_A[rs_A*ir + off_A + cscat_A_[jr]] +
                                        beta * p_B[ir + cs_B*jr];
                                }
                                else
                                {
                                    p_A[rs_A*ir + off_A + cscat_A_[jr]] =
                                        alpha * p_A[rs_A*ir + off_A + cscat_A_[jr]] +
                                        beta * p_B[ir + cs_B*jr];
                                }
                            }
                        }

                        )
                    }
                    else
                    {
                        for (len_type jr = 0;jr < n_loc;jr++)
                        {
                            for (len_type ir = 0;ir < m_loc;ir++)
                            {
                                if (Forward)
                                {
                                    p_B[ir + cs_B*jr] =
                                        alpha * p_A[rscat_A_[ir] + cscat_A_[jr]] +
                                        beta * p_B[ir + cs_B*jr];
                                }
                                else
                                {
                                    p_A[rscat_A_[ir] + cscat_A_[jr]] =
                                        alpha * p_A[rscat_A_[ir] + cscat_A_[jr]] +
                                        beta * p_B[ir + cs_B*jr];
                                }
                            }
                        }
                    }

                    )

                    rscat_A_ += MB;
                    rbs_A_++;

                    p_B += MB;
                }
            }

            )
        }
    }
    else if (cs_B == 1)
    {
        for (len_type i = m_min;i < m_max;i += MB)
        {
            len_type m_loc = std::min(MB, m_max-i);

            rscat_A_ = rscat_A.data() + i;
            rbs_A_ = rbs_A.data() + i/MB;
            cscat_A_ = cscat_A.data() + n_min;
            cbs_A_ = cbs_A.data() + n_min/NB;

            T* p_B = B.data() + rs_B*i + n_min;

            stride_type rs_A = *rbs_A_;

            T* p_A = A.data();

            TBLIS_SPECIAL_CASE(m_loc == MB,

            if (rs_A)
            {
                p_A += *rscat_A_;

                TBLIS_SPECIAL_CASE(rs_A == 1,

                for (len_type j = n_min;j < n_max;j += NB)
                {
                    len_type n_loc = std::min(NB, n_max-j);

                    stride_type cs_A = *cbs_A_;
                    stride_type off_A = *cscat_A_;

                    TBLIS_SPECIAL_CASE(n_loc == NB,

                    if (cs_A)
                    {
                        TBLIS_SPECIAL_CASE(cs_A == 1,

                        for (len_type ir = 0;ir < m_loc;ir++)
                        {
                            for (len_type jr = 0;jr < n_loc;jr++)
                            {
                                if (Forward)
                                {
                                    p_B[rs_B*ir + jr] =
                                        alpha * p_A[rs_A*ir + cs_A*jr + off_A] +
                                        beta * p_B[rs_B*ir + jr];
                                }
                                else
                                {
                                    p_A[rs_A*ir + cs_A*jr + off_A] =
                                        alpha * p_A[rs_A*ir + cs_A*jr + off_A] +
                                        beta * p_B[rs_B*ir + jr];
                                }
                            }
                        }

                        )
                    }
                    else
                    {
                        for (len_type ir = 0;ir < m_loc;ir++)
                        {
                            for (len_type jr = 0;jr < n_loc;jr++)
                            {
                                if (Forward)
                                {
                                    p_B[rs_B*ir + jr] =
                                        alpha * p_A[rs_A*ir + cscat_A_[jr]] +
                                        beta * p_B[rs_B*ir + jr];
                                }
                                else
                                {
                                    p_A[rs_A*ir + cscat_A_[jr]] =
                                        alpha * p_A[rs_A*ir + cscat_A_[jr]] +
                                        beta * p_B[rs_B*ir + jr];
                                }
                            }
                        }
                    }

                    )

                    cscat_A_ += NB;
                    cbs_A_++;

                    p_B += NB;
                }

                )
            }
            else
            {
                for (len_type j = n_min;j < n_max;j += NB)
                {
                    len_type n_loc = std::min(NB, n_max-j);

                    stride_type cs_A = *cbs_A_;
                    stride_type off_A = *cscat_A_;

                    TBLIS_SPECIAL_CASE(n_loc == NB,

                    if (cs_A)
                    {
                        TBLIS_SPECIAL_CASE(cs_A == 1,

                        for (len_type ir = 0;ir < m_loc;ir++)
                        {
                            for (len_type jr = 0;jr < n_loc;jr++)
                            {
                                if (Forward)
                                {
                                    p_B[rs_B*ir + jr] =
                                        alpha * p_A[rscat_A[ir] + cs_A*jr + off_A] +
                                        beta * p_B[rs_B*ir + jr];
                                }
                                else
                                {
                                    p_A[rscat_A[ir] + cs_A*jr + off_A] =
                                        alpha * p_A[rscat_A[ir] + cs_A*jr + off_A] +
                                        beta * p_B[rs_B*ir + jr];
                                }
                            }
                        }

                        )
                    }
                    else
                    {
                        for (len_type ir = 0;ir < m_loc;ir++)
                        {
                            for (len_type jr = 0;jr < n_loc;jr++)
                            {
                                if (Forward)
                                {
                                    p_B[rs_B*ir + jr] =
                                        alpha * p_A[rscat_A[ir] + cscat_A_[jr]] +
                                        beta * p_B[rs_B*ir + jr];
                                }
                                else
                                {
                                    p_A[rscat_A[ir] + cscat_A_[jr]] =
                                        alpha * p_A[rscat_A[ir] + cscat_A_[jr]] +
                                        beta * p_B[rs_B*ir + jr];
                                }
                            }
                        }
                    }

                    )

                    cscat_A_ += NB;
                    cbs_A_++;

                    p_B += NB;
                }
            }

            )
        }
    }
    else
    {
        abort();
    }
}

template <typename T>
void add(const communicator& comm, const config&, T alpha, tensor_matrix<T>& A, T beta, matrix_view<T>& B)
{
    add_internal<true>(comm, alpha, A, beta, B);
}

template <typename T>
void add(const communicator& comm, const config&, T alpha, matrix_view<T>& A, T beta, tensor_matrix<T>& B)
{
    add_internal<false>(comm, beta, B, alpha, A);
}

template <bool Forward, typename T, unsigned N>
void add_internal(const communicator& comm,
                  T alpha, stra_tensor_view<T, N>& A,
                  T  beta,      matrix_view<T   >& B)
{
    TBLIS_ASSERT(A.length(0) == B.length(0));
    TBLIS_ASSERT(A.length(1) == B.length(1));

    constexpr len_type MB = 8;
    constexpr len_type NB = 8;

    T coeffs[N];
    for (unsigned k = 0;k < N;k++)
        coeffs[k] = (Forward ? alpha : beta)*A.coeff(k);

    len_type m = A.length(0);
    len_type n = A.length(1);
    stride_type rs_B = B.stride(0);
    stride_type cs_B = B.stride(1);

    std::vector<stride_type> rscat_A[N];
    std::vector<stride_type> cscat_A[N];
    std::vector<stride_type> rbs_A[N];
    std::vector<stride_type> cbs_A[N];

    for (unsigned k = 0;k < N;k++)
    {
        rscat_A[k].resize(m);
        cscat_A[k].resize(n);
        rbs_A[k].resize(m);
        cbs_A[k].resize(n);
        A.fill_block_scatter(k, 0, rscat_A[k].data(), MB, rbs_A[k].data());
        A.fill_block_scatter(k, 1, cscat_A[k].data(), NB, cbs_A[k].data());
    }

    len_type m_min, m_max, n_min, n_max;
    std::tie(m_min, m_max, std::ignore,
             n_min, n_max, std::ignore) =
        comm.distribute_over_threads_2d(m, n, MB, NB);

    const stride_type* rscat_A_[N];
    const stride_type* rbs_A_[N];
    const stride_type* cscat_A_[N];
    const stride_type* cbs_A_[N];

    if (rs_B == 1)
    {
        for (len_type j = n_min;j < n_max;j += NB)
        {
            len_type n_loc = std::min(NB, n_max-j);

            for (unsigned k = 0;k < N;k++)
            {
                rscat_A_[k] = rscat_A[k].data() + m_min;
                rbs_A_[k] = rbs_A[k].data() + m_min/MB;
                cscat_A_[k] = cscat_A[k].data() + j;
                cbs_A_[k] = cbs_A[k].data() + j/NB;
            }

            T* p_B = B.data() + m_min + cs_B*j;

            stride_type cs_A = *cbs_A_[0];

            bool n_strided = true;
            for (unsigned k = 0;k < N;k++) n_strided = n_strided && *cbs_A_[k];

            T* p_A[N];
            for (unsigned k = 0;k < N;k++) p_A[k] = A.data();

            TBLIS_SPECIAL_CASE(n_loc == NB,

            if (n_strided)
            {
                for (unsigned k = 0;k < N;k++) p_A[k] += *cscat_A_[k];

                TBLIS_SPECIAL_CASE(cs_A == 1,

                for (len_type i = m_min;i < m_max;i += MB)
                {
                    len_type m_loc = std::min(MB, m_max-i);

                    stride_type rs_A = *rbs_A_[0];
                    stride_type off_A[N];
                    for (unsigned k = 0;k < N;k++) off_A[k] = *rscat_A_[k];

                    bool m_strided = true;
                    for (unsigned k = 0;k < N;k++) m_strided = m_strided && *rbs_A_[k];

                    TBLIS_SPECIAL_CASE(m_loc == MB,

                    if (m_strided)
                    {
                        TBLIS_SPECIAL_CASE(rs_A == 1,

                        for (len_type jr = 0;jr < n_loc;jr++)
                        {
                            for (len_type ir = 0;ir < m_loc;ir++)
                            {
                                if (Forward)
                                {
                                    p_B[ir + cs_B*jr] *= beta;
                                    for (unsigned k = 0;k < N;k++)
                                        p_B[ir + cs_B*jr] +=
                                            coeffs[k] * p_A[k][rs_A*ir + off_A[k] + cs_A*jr];
                                }
                                else
                                {
                                    for (unsigned k = 0;k < N;k++)
                                        p_A[k][rs_A*ir + off_A[k] + cs_A*jr] =
                                            alpha * p_A[k][rs_A*ir + off_A[k] + cs_A*jr] +
                                            coeffs[k] * p_B[ir + cs_B*jr];
                                }
                            }
                        }

                        )
                    }
                    else
                    {
                        for (len_type jr = 0;jr < n_loc;jr++)
                        {
                            for (len_type ir = 0;ir < m_loc;ir++)
                            {
                                if (Forward)
                                {
                                    p_B[ir + cs_B*jr] *= beta;
                                    for (unsigned k = 0;k < N;k++)
                                        p_B[ir + cs_B*jr] +=
                                            coeffs[k] * p_A[k][rscat_A_[k][ir] + cs_A*jr];
                                }
                                else
                                {
                                    for (unsigned k = 0;k < N;k++)
                                        p_A[k][rscat_A_[k][ir] + cs_A*jr] =
                                            alpha * p_A[k][rscat_A_[k][ir] + cs_A*jr] +
                                            coeffs[k] * p_B[ir + cs_B*jr];
                                }
                            }
                        }
                    }

                    )

                    for (unsigned k = 0;k < N;k++)
                    {
                        rscat_A_[k] += MB;
                        rbs_A_[k]++;
                    }

                    p_B += MB;
                }

                )
            }
            else
            {
                for (len_type i = m_min;i < m_max;i += MB)
                {
                    len_type m_loc = std::min(MB, m_max-i);

                    stride_type rs_A = *rbs_A_[0];
                    stride_type off_A[N];
                    for (unsigned k = 0;k < N;k++) off_A[k] = *rscat_A_[k];

                    bool m_strided = true;
                    for (unsigned k = 0;k < N;k++) m_strided = m_strided && *rbs_A_[k];

                    TBLIS_SPECIAL_CASE(m_loc == MB,

                    if (m_strided)
                    {
                        TBLIS_SPECIAL_CASE(rs_A == 1,

                        for (len_type jr = 0;jr < n_loc;jr++)
                        {
                            for (len_type ir = 0;ir < m_loc;ir++)
                            {
                                if (Forward)
                                {
                                    p_B[ir + cs_B*jr] *= beta;
                                    for (unsigned k = 0;k < N;k++)
                                        p_B[ir + cs_B*jr] +=
                                            coeffs[k] * p_A[k][rs_A*ir + off_A[k] + cscat_A_[k][jr]];
                                }
                                else
                                {
                                    for (unsigned k = 0;k < N;k++)
                                        p_A[k][rs_A*ir + off_A[k] + cscat_A_[k][jr]] =
                                            alpha * p_A[k][rs_A*ir + off_A[k] + cscat_A_[k][jr]] +
                                            coeffs[k] * p_B[ir + cs_B*jr];
                                }
                            }
                        }

                        )
                    }
                    else
                    {
                        for (len_type jr = 0;jr < n_loc;jr++)
                        {
                            for (len_type ir = 0;ir < m_loc;ir++)
                            {
                                if (Forward)
                                {
                                    p_B[ir + cs_B*jr] *= beta;
                                    for (unsigned k = 0;k < N;k++)
                                        p_B[ir + cs_B*jr] +=
                                            coeffs[k] * p_A[k][rscat_A_[k][ir] + cscat_A_[k][jr]];
                                }
                                else
                                {
                                    for (unsigned k = 0;k < N;k++)
                                        p_A[k][rscat_A_[k][ir] + cscat_A_[k][jr]] =
                                            alpha * p_A[k][rscat_A_[k][ir] + cscat_A_[k][jr]] +
                                            coeffs[k] * p_B[ir + cs_B*jr];
                                }
                            }
                        }
                    }

                    )

                    for (unsigned k = 0;k < N;k++)
                    {
                        rscat_A_[k] += MB;
                        rbs_A_[k]++;
                    }

                    p_B += MB;
                }
            }

            )
        }
    }
    else if (cs_B == 1)
    {
        for (len_type i = m_min;i < m_max;i += MB)
        {
            len_type m_loc = std::min(MB, m_max-i);

            for (unsigned k = 0;k < N;k++)
            {
                rscat_A_[k] = rscat_A[k].data() + i;
                rbs_A_[k] = rbs_A[k].data() + i/MB;
                cscat_A_[k] = cscat_A[k].data() + n_min;
                cbs_A_[k] = cbs_A[k].data() + n_min/NB;
            }

            T* p_B = B.data() + rs_B*i + n_min;

            stride_type rs_A = *rbs_A_[0];

            bool m_strided = true;
            for (unsigned k = 0;k < N;k++) m_strided = m_strided && *rbs_A_[k];

            T* p_A[N];
            for (unsigned k = 0;k < N;k++) p_A[k] = A.data();

            TBLIS_SPECIAL_CASE(m_loc == MB,

            if (m_strided)
            {
                for (unsigned k = 0;k < N;k++) p_A[k] += *rscat_A_[k];

                TBLIS_SPECIAL_CASE(rs_A == 1,

                for (len_type j = n_min;j < n_max;j += NB)
                {
                    len_type n_loc = std::min(NB, n_max-j);

                    stride_type cs_A = *cbs_A_[0];
                    stride_type off_A[N];
                    for (unsigned k = 0;k < N;k++) off_A[k] = *cscat_A_[k];

                    bool n_strided = true;
                    for (unsigned k = 0;k < N;k++) n_strided = n_strided && *cbs_A_[k];

                    TBLIS_SPECIAL_CASE(n_loc == NB,

                    if (n_strided)
                    {
                        TBLIS_SPECIAL_CASE(cs_A == 1,

                        for (len_type ir = 0;ir < m_loc;ir++)
                        {
                            for (len_type jr = 0;jr < n_loc;jr++)
                            {
                                if (Forward)
                                {
                                    p_B[rs_B*ir + jr] *= beta;
                                    for (unsigned k = 0;k < N;k++)
                                        p_B[rs_B*ir + jr] +=
                                            coeffs[k] * p_A[k][rs_A*ir + cs_A*jr + off_A[k]];
                                }
                                else
                                {
                                    for (unsigned k = 0;k < N;k++)
                                        p_A[k][rs_A*ir + cs_A*jr + off_A[k]] =
                                            alpha * p_A[k][rs_A*ir + cs_A*jr + off_A[k]] +
                                            coeffs[k] * p_B[rs_B*ir + jr];
                                }
                            }
                        }

                        )
                    }
                    else
                    {
                        for (len_type ir = 0;ir < m_loc;ir++)
                        {
                            for (len_type jr = 0;jr < n_loc;jr++)
                            {
                                if (Forward)
                                {
                                    p_B[rs_B*ir + jr] *= beta;
                                    for (unsigned k = 0;k < N;k++)
                                        p_B[rs_B*ir + jr] +=
                                            coeffs[k] * p_A[k][rs_A*ir + cscat_A_[k][jr]];
                                }
                                else
                                {
                                    for (unsigned k = 0;k < N;k++)
                                        p_A[k][rs_A*ir + cscat_A_[k][jr]] =
                                            alpha * p_A[k][rs_A*ir + cscat_A_[k][jr]] +
                                            coeffs[k] * p_B[rs_B*ir + jr];
                                }
                            }
                        }
                    }

                    )

                    for (unsigned k = 0;k < N;k++)
                    {
                        cscat_A_[k] += NB;
                        cbs_A_[k]++;
                    }

                    p_B += NB;
                }

                )
            }
            else
            {
                for (len_type j = n_min;j < n_max;j += NB)
                {
                    len_type n_loc = std::min(NB, n_max-j);

                    stride_type cs_A = *cbs_A_[0];
                    stride_type off_A[N];
                    for (unsigned k = 0;k < N;k++) off_A[k] = *cscat_A_[k];

                    bool n_strided = true;
                    for (unsigned k = 0;k < N;k++) n_strided = n_strided && *cbs_A_[k];

                    TBLIS_SPECIAL_CASE(n_loc == NB,

                    if (n_strided)
                    {
                        TBLIS_SPECIAL_CASE(cs_A == 1,

                        for (len_type ir = 0;ir < m_loc;ir++)
                        {
                            for (len_type jr = 0;jr < n_loc;jr++)
                            {
                                if (Forward)
                                {
                                    p_B[rs_B*ir + jr] *= beta;
                                    for (unsigned k = 0;k < N;k++)
                                        p_B[rs_B*ir + jr] +=
                                            coeffs[k] * p_A[k][rscat_A[k][ir] + cs_A*jr + off_A[k]];
                                }
                                else
                                {
                                    for (unsigned k = 0;k < N;k++)
                                        p_A[k][rscat_A[k][ir] + cs_A*jr + off_A[k]] =
                                            alpha * p_A[k][rscat_A[k][ir] + cs_A*jr + off_A[k]] +
                                            coeffs[k] * p_B[rs_B*ir + jr];
                                }
                            }
                        }

                        )
                    }
                    else
                    {
                        for (len_type ir = 0;ir < m_loc;ir++)
                        {
                            for (len_type jr = 0;jr < n_loc;jr++)
                            {
                                if (Forward)
                                {
                                    p_B[rs_B*ir + jr] *= beta;
                                    for (unsigned k = 0;k < N;k++)
                                        p_B[rs_B*ir + jr] +=
                                            coeffs[k] * p_A[k][rscat_A[k][ir] + cscat_A_[k][jr]];
                                }
                                else
                                {
                                    for (unsigned k = 0;k < N;k++)
                                        p_A[k][rscat_A[k][ir] + cscat_A_[k][jr]] =
                                            alpha * p_A[k][rscat_A[k][ir] + cscat_A_[k][jr]] +
                                            coeffs[k] * p_B[rs_B*ir + jr];
                                }
                            }
                        }
                    }

                    )

                    for (unsigned k = 0;k < N;k++)
                    {
                        cscat_A_[k] += NB;
                        cbs_A_[k]++;
                    }

                    p_B += NB;
                }
            }

            )
        }
    }
    else
    {
        abort();
    }
}

template <typename T, unsigned N>
void add(const communicator& comm, const config&, T alpha, stra_tensor_view<T,N>& A, T beta, matrix_view<T>& B)
{
    add_internal<true>(comm, alpha, A, beta, B);
}

template <typename T, unsigned N>
void add(const communicator& comm, const config&, T alpha, matrix_view<T>& A, T beta, stra_tensor_view<T,N>& B)
{
    add_internal<false>(comm, beta, B, alpha, A);
}

}
}

#endif
