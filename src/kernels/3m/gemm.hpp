#ifndef _TBLIS_KERNELS_3M_GEMM_HPP_
#define _TBLIS_KERNELS_3M_GEMM_HPP_

#include "util/basic_types.h"
#include <type_traits>

namespace tblis
{

#define EXTERN_STRA_GEMM_UKR_FOUR(T, name) \
extern void name(tblis::stride_type k, \
                 const T* alpha, \
                 const T* a, const T* b, \
                 const T* beta, \
                 unsigned N, T** c_list, const T* coeff_list, tblis::stride_type rs_c, \
                       tblis::stride_type cs_c);

template <typename T>
using stra_gemm_ukr_four_t =
void (*)(stride_type k,
        const T* alpha,
        const T* a, const T* b,
        const T* beta,
        unsigned N, T** c_list, const T* coeff_list, stride_type rs_c, stride_type cs_c);

template <typename Config, typename T>
void stra_gemm_ukr_four_def(stride_type k,
                  const T* TBLIS_RESTRICT alpha,
                  const T* TBLIS_RESTRICT p_a, const T* TBLIS_RESTRICT p_b,
                  const T* TBLIS_RESTRICT beta,
                  unsigned N, T** TBLIS_RESTRICT p_c_list, const T* coeff_list, stride_type rs_c, stride_type cs_c)
{
    std::cout << "WARNING: Enter stra_gemm_ukr_four_def\n" << std::endl;
    constexpr len_type MR = Config::template gemm_mr<T>::def;
    constexpr len_type NR = Config::template gemm_nr<T>::def;

    TBLIS_ASSERT(strcmp(Config::name, "reference") == 0);

    T p_ab[MR*NR] __attribute__((aligned(64))) = {};

    while (k --> 0)
    {
        for (int j = 0;j < NR;j++)
        {
            for (int i = 0;i < MR;i++)
            {
                p_ab[i + MR*j] += p_a[i] * p_b[j];
            }
        }

        p_a += MR;
        p_b += NR;
    }

    if (*beta == T(0))
    {
        for (len_type j = 0;j < NR;j++)
        {
            for (len_type i = 0;i < MR;i++)
            {
                //p_c[i*rs_c + j*cs_c] = (*alpha)*p_ab[i + MR*j];
                //#pragma unroll
                //for (unsigned idx = 0; idx < N; idx++) {
                //    p_c_list[idx][i*rs_c + j*cs_c] = coeff_list[idx] * (*alpha) * p_ab[i + MR*j];
                //}

                p_c_list[0][i*rs_c + j*cs_c] = coeff_list[0] * (*alpha) * p_ab[i + MR*j];
                p_c_list[1][i*rs_c + j*cs_c] = coeff_list[1] * (*alpha) * p_ab[i + MR*j];
                p_c_list[2][i*rs_c + j*cs_c] = coeff_list[2] * (*alpha) * p_ab[i + MR*j];
                p_c_list[3][i*rs_c + j*cs_c] = coeff_list[3] * (*alpha) * p_ab[i + MR*j];

            }
        }
    }
    else
    {
        for (len_type j = 0;j < NR;j++)
        {
            for (len_type i = 0;i < MR;i++)
            {
                //p_c[i*rs_c + j*cs_c] = (*alpha)*p_ab[i + MR*j] +
                //                       (*beta)*p_c[i*rs_c + j*cs_c];
                //#pragma unroll
                //for (unsigned idx = 0; idx < N; idx++) {
                //    p_c_list[idx][i*rs_c + j*cs_c] = coeff_list[idx] * (*alpha) * p_ab[i + MR*j] +
                //                                     (*beta)*p_c_list[idx][i*rs_c + j*cs_c];
                //}
                p_c_list[0][i*rs_c + j*cs_c] = coeff_list[0] * (*alpha) * p_ab[i + MR*j] +
                    (*beta)*p_c_list[0][i*rs_c + j*cs_c];
                p_c_list[1][i*rs_c + j*cs_c] = coeff_list[1] * (*alpha) * p_ab[i + MR*j] +
                    (*beta)*p_c_list[1][i*rs_c + j*cs_c];
                p_c_list[2][i*rs_c + j*cs_c] = coeff_list[2] * (*alpha) * p_ab[i + MR*j] +
                    (*beta)*p_c_list[2][i*rs_c + j*cs_c];
                p_c_list[3][i*rs_c + j*cs_c] = coeff_list[3] * (*alpha) * p_ab[i + MR*j] +
                    (*beta)*p_c_list[3][i*rs_c + j*cs_c];

            }
        }
    }
}

#define EXTERN_STRA_GEMM_UKR_TWO(T, name) \
extern void name(tblis::stride_type k, \
                 const T* alpha, \
                 const T* a, const T* b, \
                 const T* beta, \
                 unsigned N, T** c_list, const T* coeff_list, tblis::stride_type rs_c, \
                       tblis::stride_type cs_c);

template <typename T>
using stra_gemm_ukr_two_t =
void (*)(stride_type k,
        const T* alpha,
        const T* a, const T* b,
        const T* beta,
        unsigned N, T** c_list, const T* coeff_list, stride_type rs_c, stride_type cs_c);

template <typename Config, typename T>
void stra_gemm_ukr_two_def(stride_type k,
                  const T* TBLIS_RESTRICT alpha,
                  const T* TBLIS_RESTRICT p_a, const T* TBLIS_RESTRICT p_b,
                  const T* TBLIS_RESTRICT beta,
                  unsigned N, T** TBLIS_RESTRICT p_c_list, const T* coeff_list, stride_type rs_c, stride_type cs_c)
{
    std::cout << "WARNING: Enter stra_gemm_ukr_two_def\n" << std::endl;
    constexpr len_type MR = Config::template gemm_mr<T>::def;
    constexpr len_type NR = Config::template gemm_nr<T>::def;

    TBLIS_ASSERT(strcmp(Config::name, "reference") == 0);

    T p_ab[MR*NR] __attribute__((aligned(64))) = {};

    while (k --> 0)
    {
        for (int j = 0;j < NR;j++)
        {
            for (int i = 0;i < MR;i++)
            {
                p_ab[i + MR*j] += p_a[i] * p_b[j];
            }
        }

        p_a += MR;
        p_b += NR;
    }

    if (*beta == T(0))
    {
        for (len_type j = 0;j < NR;j++)
        {
            for (len_type i = 0;i < MR;i++)
            {
                //p_c[i*rs_c + j*cs_c] = (*alpha)*p_ab[i + MR*j];
                //#pragma unroll
                //for (unsigned idx = 0; idx < N; idx++) {
                //    p_c_list[idx][i*rs_c + j*cs_c] = coeff_list[idx] * (*alpha) * p_ab[i + MR*j];
                //}

                p_c_list[0][i*rs_c + j*cs_c] = coeff_list[0] * (*alpha) * p_ab[i + MR*j];
                p_c_list[1][i*rs_c + j*cs_c] = coeff_list[1] * (*alpha) * p_ab[i + MR*j];
            }
        }
    }
    else
    {
        for (len_type j = 0;j < NR;j++)
        {
            for (len_type i = 0;i < MR;i++)
            {
                //p_c[i*rs_c + j*cs_c] = (*alpha)*p_ab[i + MR*j] +
                //                       (*beta)*p_c[i*rs_c + j*cs_c];
                //#pragma unroll
                //for (unsigned idx = 0; idx < N; idx++) {
                //    p_c_list[idx][i*rs_c + j*cs_c] = coeff_list[idx] * (*alpha) * p_ab[i + MR*j] +
                //                                     (*beta)*p_c_list[idx][i*rs_c + j*cs_c];
                //}
                p_c_list[0][i*rs_c + j*cs_c] = coeff_list[0] * (*alpha) * p_ab[i + MR*j] +
                    (*beta)*p_c_list[0][i*rs_c + j*cs_c];
                p_c_list[1][i*rs_c + j*cs_c] = coeff_list[1] * (*alpha) * p_ab[i + MR*j] +
                    (*beta)*p_c_list[1][i*rs_c + j*cs_c];
            }
        }
    }
}


#define EXTERN_STRA_GEMM_UKR(T, name) \
extern void name(tblis::stride_type k, \
                 const T* alpha, \
                 const T* a, const T* b, \
                 const T* beta, \
                 unsigned N, T** c_list, const T* coeff_list, tblis::stride_type rs_c, \
                       tblis::stride_type cs_c);

template <typename T>
using stra_gemm_ukr_t =
void (*)(stride_type k,
        const T* alpha,
        const T* a, const T* b,
        const T* beta,
        unsigned N, T** c_list, const T* coeff_list, stride_type rs_c, stride_type cs_c);

template <typename Config, typename T>
void stra_gemm_ukr_def(stride_type k,
                  const T* TBLIS_RESTRICT alpha,
                  const T* TBLIS_RESTRICT p_a, const T* TBLIS_RESTRICT p_b,
                  const T* TBLIS_RESTRICT beta,
                  unsigned N, T** TBLIS_RESTRICT p_c_list, const T* coeff_list, stride_type rs_c, stride_type cs_c)
{

    std::cout << "WARNING: Enter stra_gemm_ukr_def\n" << std::endl;
    constexpr len_type MR = Config::template gemm_mr<T>::def;
    constexpr len_type NR = Config::template gemm_nr<T>::def;

    TBLIS_ASSERT(strcmp(Config::name, "reference") == 0);

    T p_ab[MR*NR] __attribute__((aligned(64))) = {};

    while (k --> 0)
    {
        for (int j = 0;j < NR;j++)
        {
            for (int i = 0;i < MR;i++)
            {
                p_ab[i + MR*j] += p_a[i] * p_b[j];
            }
        }

        p_a += MR;
        p_b += NR;
    }

    if (*beta == T(0))
    {
        for (len_type j = 0;j < NR;j++)
        {
            for (len_type i = 0;i < MR;i++)
            {
                //p_c[i*rs_c + j*cs_c] = (*alpha)*p_ab[i + MR*j];
                //#pragma unroll
                for (unsigned idx = 0; idx < N; idx++) {
                    p_c_list[idx][i*rs_c + j*cs_c] = coeff_list[idx] * (*alpha) * p_ab[i + MR*j];
                }
            }
        }
    }
    else
    {
        for (len_type j = 0;j < NR;j++)
        {
            for (len_type i = 0;i < MR;i++)
            {
                //p_c[i*rs_c + j*cs_c] = (*alpha)*p_ab[i + MR*j] +
                //                       (*beta)*p_c[i*rs_c + j*cs_c];
                //#pragma unroll
                for (unsigned idx = 0; idx < N; idx++) {
                    p_c_list[idx][i*rs_c + j*cs_c] = coeff_list[idx] * (*alpha) * p_ab[i + MR*j] +
                                                     (*beta)*p_c_list[idx][i*rs_c + j*cs_c];
                }
            }
        }
    }
}

#define EXTERN_GEMM_UKR(T, name) \
extern void name(tblis::stride_type k, \
                 const T* alpha, \
                 const T* a, const T* b, \
                 const T* beta, \
                 T* c, tblis::stride_type rs_c, \
                       tblis::stride_type cs_c);

template <typename T>
using gemm_ukr_t =
void (*)(stride_type k,
        const T* alpha,
        const T* a, const T* b,
        const T* beta,
        T* c, stride_type rs_c, stride_type cs_c);

template <typename Config, typename T>
void gemm_ukr_def(stride_type k,
                  const T* TBLIS_RESTRICT alpha,
                  const T* TBLIS_RESTRICT p_a, const T* TBLIS_RESTRICT p_b,
                  const T* TBLIS_RESTRICT beta,
                  T* TBLIS_RESTRICT p_c, stride_type rs_c, stride_type cs_c)
{
    constexpr len_type MR = Config::template gemm_mr<T>::def;
    constexpr len_type NR = Config::template gemm_nr<T>::def;

    TBLIS_ASSERT(strcmp(Config::name, "reference") == 0);

    T p_ab[MR*NR] __attribute__((aligned(64))) = {};

    while (k --> 0)
    {
        for (int j = 0;j < NR;j++)
        {
            for (int i = 0;i < MR;i++)
            {
                p_ab[i + MR*j] += p_a[i] * p_b[j];
            }
        }

        p_a += MR;
        p_b += NR;
    }

    if (*beta == T(0))
    {
        for (len_type j = 0;j < NR;j++)
        {
            for (len_type i = 0;i < MR;i++)
            {
                p_c[i*rs_c + j*cs_c] = (*alpha)*p_ab[i + MR*j];
            }
        }
    }
    else
    {
        for (len_type j = 0;j < NR;j++)
        {
            for (len_type i = 0;i < MR;i++)
            {
                p_c[i*rs_c + j*cs_c] = (*alpha)*p_ab[i + MR*j] +
                                       (*beta)*p_c[i*rs_c + j*cs_c];
            }
        }
    }
}

#define EXTERN_PACK_NN_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const T* p_a, tblis::stride_type rs_a, \
                               tblis::stride_type cs_a, \
                 T* p_ap);

template <typename T>
using pack_nn_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, stride_type rs_a, stride_type cs_a,
         T* p_ap);


////////////////////////////////////////////////////////////////////////
#define EXTERN_STRA_PACK_NN_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const unsigned N, const T** p_a_list, const T* coeff_list, tblis::stride_type rs_a, \
                               tblis::stride_type cs_a, \
                 T* p_ap);

template <typename T>
using stra_pack_nn_ukr_t =
void (*)(len_type m, len_type k,
         const unsigned N, const T** p_a_list, const T* coeff_list, stride_type rs_a, stride_type cs_a,
         T* p_ap);

template <typename T>
using stra_pack_two_nn_ukr_t =
void (*)(len_type m, len_type k,
         const unsigned N, const T** p_a_list, const T* coeff_list, stride_type rs_a, stride_type cs_a,
         T* p_ap);

template <typename T>
using stra_pack_four_nn_ukr_t =
void (*)(len_type m, len_type k,
         const unsigned N, const T** p_a_list, const T* coeff_list, stride_type rs_a, stride_type cs_a,
         T* p_ap);


////////////////////////////////////////////////////////////////////////

#define EXTERN_PACK_SN_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const T* p_a, const tblis::stride_type* rscat_a, \
                               tblis::stride_type cs_a, \
                 T* p_ap);

template <typename T>
using pack_sn_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, const stride_type* rscat_a, stride_type cs_a,
         T* p_ap);

#define EXTERN_PACK_NS_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const T* p_a, tblis::stride_type rs_a, \
                               const tblis::stride_type* cscat_a, \
                 T* p_ap);

template <typename T>
using pack_ns_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, stride_type rs_a, const stride_type* cscat_a,
         T* p_ap);

#define EXTERN_PACK_SS_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const T* p_a, const tblis::stride_type* rscat_a, \
                               const tblis::stride_type* cscat_a, \
                 T* p_ap);

template <typename T>
using pack_ss_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, const stride_type* rscat_a, const stride_type* cscat_a,
         T* p_ap);

#define EXTERN_PACK_NB_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const T* p_a, tblis::stride_type rs_a, \
                               const tblis::stride_type* cscat_a, \
                               const tblis::stride_type* cbs_a, \
                 T* p_ap);

template <typename T>
using pack_nb_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, stride_type rs_a, const stride_type* cscat_a,
         const stride_type* cbs_a,
         T* p_ap);


////////////////////////////////////////////////////////////////////////
#define EXTERN_STRA_PACK_NB_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const unsigned N, const T* p_a, const T* coeff_list, tblis::stride_type rs_a, \
                               const tblis::stride_type** cscat_a, \
                               const tblis::stride_type** cbs_a, \
                 T* p_ap);

template <typename T>
using stra_pack_nb_ukr_t =
void (*)(len_type m, len_type k,
         const unsigned N, const T** p_a_list, const T* coeff_list, stride_type rs_a, const stride_type** cscat_a,
         const stride_type** cbs_a,
         T* p_ap);

template <typename T>
using stra_pack_two_nb_ukr_t =
void (*)(len_type m, len_type k,
         const unsigned N, const T** p_a_list, const T* coeff_list, stride_type rs_a, const stride_type** cscat_a,
         const stride_type** cbs_a,
         T* p_ap);

template <typename T>
using stra_pack_four_nb_ukr_t =
void (*)(len_type m, len_type k,
         const unsigned N, const T** p_a_list, const T* coeff_list, stride_type rs_a, const stride_type** cscat_a,
         const stride_type** cbs_a,
         T* p_ap);

////////////////////////////////////////////////////////////////////////


#define EXTERN_PACK_SB_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const T* p_a, const tblis::stride_type* rscat_a, \
                               const tblis::stride_type* cscat_a, \
                               const tblis::stride_type* cbs_a, \
                 T* p_ap);

template <typename T>
using pack_sb_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, const stride_type* rscat_a, const stride_type* cscat_a,
         const stride_type* cbs_a,
         T* p_ap);


////////////////////////////////////////////////////////////////////////
#define EXTERN_STRA_PACK_SB_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const unsigned N, const T* p_a, const T* coeff_list, const tblis::stride_type** rscat_a, \
                               const tblis::stride_type** cscat_a, \
                               const tblis::stride_type** cbs_a, \
                 T* p_ap);

template <typename T>
using stra_pack_sb_ukr_t =
void (*)(len_type m, len_type k,
         const unsigned N, const T* p_a, const T* coeff_list, const stride_type** rscat_a, const stride_type** cscat_a,
         const stride_type** cbs_a,
         T* p_ap);

template <typename T>
using stra_pack_two_sb_ukr_t =
void (*)(len_type m, len_type k,
         const unsigned N, const T* p_a, const T* coeff_list, const stride_type** rscat_a, const stride_type** cscat_a,
         const stride_type** cbs_a,
         T* p_ap);

template <typename T>
using stra_pack_four_sb_ukr_t =
void (*)(len_type m, len_type k,
         const unsigned N, const T* p_a, const T* coeff_list, const stride_type** rscat_a, const stride_type** cscat_a,
         const stride_type** cbs_a,
         T* p_ap);
////////////////////////////////////////////////////////////////////////




template <typename Config, typename T, int Mat>
void stra_pack_two_nn_ukr_def(len_type m, len_type k,
                     const unsigned N, const T** TBLIS_RESTRICT p_a_list, const T* TBLIS_RESTRICT coeff_list, stride_type rs_a, stride_type cs_a,
                     T* TBLIS_RESTRICT p_ap)
{
    const T coeff0 = coeff_list[0], coeff1 = coeff_list[1];
    const T *p_a0 = p_a_list[0], *p_a1 = p_a_list[1];
    
    //std::cout << "Enter stra_pack_two_nn_ukr_def\n" << std::endl;
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);
    constexpr len_type KR = Config::template gemm_kr<T>::def;

    if (m == MR && rs_a == 1)
    {
        len_type p = 0;
        for (;p < k-KR;p += KR)
        {
            for (len_type kr = 0;kr < KR;kr++)
            {
                for (len_type mr = 0;mr < MR;mr++)
                {
                    ////p_ap[mr + ME*kr] = p_a[mr + cs_a*kr];
                    //p_ap[mr + ME*kr] = 0;
                    ////#pragma unroll
                    //for (unsigned idx = 0; idx < N; idx++) {
                    //    p_ap[mr + ME*kr] += coeff_list[idx] * p_a_list[idx][mr + cs_a*kr];
                    //}

                    //p_ap[mr + ME*kr]  = coeff_list[0] * p_a_list[0][mr + cs_a*kr] +  coeff_list[1] * p_a_list[1][mr + cs_a*kr];
                    p_ap[mr + ME*kr]  = coeff0 * p_a0[mr + cs_a*kr] +  coeff1 * p_a1[mr + cs_a*kr];
                }
            }
            ////p_a += cs_a*KR;
            //for (unsigned idx = 0; idx < N; idx++) {
            //    p_a_list[idx] += cs_a*KR;
            //}
            ////for (auto &p_a : p_a_list) {
            ////    p_a += cs_a*KR;
            ////}

            //p_a_list[0] += cs_a*KR;
            //p_a_list[1] += cs_a*KR;
            p_a0 += cs_a*KR;
            p_a1 += cs_a*KR;

            p_ap += ME*KR;
        }

        for (len_type kr = 0;kr < k-p;kr++)
        {
            for (len_type mr = 0;mr < MR;mr++)
            {
                ////p_ap[mr + ME*kr] = p_a[mr + cs_a*kr];
                //p_ap[mr + ME*kr] = 0;
                ////#pragma unroll
                //for (unsigned idx = 0; idx < N; idx++) {
                //    p_ap[mr + ME*kr] += coeff_list[idx] * p_a_list[idx][mr + cs_a*kr];
                //}

                //p_ap[mr + ME*kr] = coeff_list[0] * p_a_list[0][mr + cs_a*kr] + coeff_list[1] * p_a_list[1][mr + cs_a*kr];
                p_ap[mr + ME*kr] = coeff0 * p_a0[mr + cs_a*kr] + coeff1 * p_a1[mr + cs_a*kr];

            }
        }
    }
    else if (m == MR && cs_a == 1)
    {
        //std::cout << "KR: " << KR << ";k: " << k << std::endl;

        len_type p = 0;
        for (;p < k-KR;p += KR)
        {
            for (len_type kr = 0;kr < KR;kr++)
            {
                for (len_type mr = 0;mr < MR;mr++)
                {
                    ////p_ap[mr + ME*kr] = p_a[rs_a*mr + kr];
                    //p_ap[mr + ME*kr] = 0;
                    ////#pragma unroll
                    //for (unsigned idx = 0; idx < N; idx++) {
                    //    p_ap[mr + ME*kr] += coeff_list[idx] * p_a_list[idx][rs_a*mr + kr];
                    //    //std::cout << coeff_list[i] << "," << p_a_list[i][rs_a*mr + kr] << ";";
                    //}

                    //p_ap[mr + ME*kr] = coeff_list[0] * p_a_list[0][rs_a*mr + kr] + coeff_list[1] * p_a_list[1][rs_a*mr + kr];
                    p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + kr] + coeff1 * p_a1[rs_a*mr + kr];

                    //std::cout << "p_ap[" << mr << "][" << kr << "]=" << p_ap[mr + ME*kr] << std::endl;
                }
            }

            //p_a += KR;
            //for (auto &p_a : p_a_list) {
            //    p_a += KR;
            //}
            //for (unsigned idx = 0; idx < N; idx++) {
            //    p_a_list[idx] += KR;
            //}

            //p_a_list[0] += KR;
            //p_a_list[1] += KR;
            p_a0 += KR;
            p_a1 += KR;

            p_ap += ME*KR;
        }

        for (len_type kr = 0;kr < k-p;kr++)
        {
            for (len_type mr = 0;mr < MR;mr++)
            {
                ////p_ap[mr + ME*kr] = p_a[rs_a*mr + kr];
                //p_ap[mr + ME*kr] = 0;
                ////#pragma unroll
                //for (unsigned idx = 0; idx < N; idx++) {
                //    p_ap[mr + ME*kr] += coeff_list[idx] * p_a_list[idx][rs_a*mr + kr];
                //}

                //p_ap[mr + ME*kr] = coeff_list[0] * p_a_list[0][rs_a*mr + kr] + coeff_list[1] * p_a_list[1][rs_a*mr + kr];
                p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + kr] + coeff1 * p_a1[rs_a*mr + kr];
            }
        }
    }
    else
    {
        for (len_type p = 0;p < k;p++)
        {
            for (len_type mr = 0;mr < m;mr++)
            {
                ////p_ap[mr + ME*p] = p_a[rs_a*mr + cs_a*p];
                //p_ap[mr + ME*p] = 0;
                ////#pragma unroll
                //for (unsigned idx = 0; idx < N; idx++) {
                //    p_ap[mr + ME*p] += coeff_list[idx] * p_a_list[idx][rs_a*mr + cs_a*p];
                //}

                //p_ap[mr + ME*p] = coeff_list[0] * p_a_list[0][rs_a*mr + cs_a*p] + coeff_list[1] * p_a_list[1][rs_a*mr + cs_a*p];
                p_ap[mr + ME*p] = coeff0 * p_a0[rs_a*mr + cs_a*p] + coeff1 * p_a1[rs_a*mr + cs_a*p];
            }

            for (len_type mr = m;mr < MR;mr++)
            {
                p_ap[mr + ME*p] = T();
            }
        }
    }
}

template <typename Config, typename T, int Mat>
void stra_pack_four_nn_ukr_def(len_type m, len_type k,
                     const unsigned N, const T** TBLIS_RESTRICT p_a_list, const T* TBLIS_RESTRICT coeff_list, stride_type rs_a, stride_type cs_a,
                     T* TBLIS_RESTRICT p_ap)
{
    const T coeff0 = coeff_list[0], coeff1 = coeff_list[1], coeff2 = coeff_list[2], coeff3 = coeff_list[3];
    const T *p_a0 = p_a_list[0], *p_a1 = p_a_list[1], *p_a2 = p_a_list[2], *p_a3 = p_a_list[3];
    
    //std::cout << "Enter stra_pack_four_nn_ukr_def\n" << std::endl;
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);
    constexpr len_type KR = Config::template gemm_kr<T>::def;

    if (m == MR && rs_a == 1)
    {
        len_type p = 0;
        for (;p < k-KR;p += KR)
        {
            for (len_type kr = 0;kr < KR;kr++)
            {
                for (len_type mr = 0;mr < MR;mr++)
                {
                    ////p_ap[mr + ME*kr] = p_a[mr + cs_a*kr];
                    //p_ap[mr + ME*kr] = 0;
                    ////#pragma unroll
                    //for (unsigned idx = 0; idx < N; idx++) {
                    //    p_ap[mr + ME*kr] += coeff_list[idx] * p_a_list[idx][mr + cs_a*kr];
                    //}

                    //p_ap[mr + ME*kr]  = coeff_list[0] * p_a_list[0][mr + cs_a*kr] +  coeff_list[1] * p_a_list[1][mr + cs_a*kr];
                    p_ap[mr + ME*kr]  = coeff0 * p_a0[mr + cs_a*kr] +  coeff1 * p_a1[mr + cs_a*kr]
                                      + coeff2 * p_a2[mr + cs_a*kr] +  coeff3 * p_a3[mr + cs_a*kr];
                }
            }
            ////p_a += cs_a*KR;
            //for (unsigned idx = 0; idx < N; idx++) {
            //    p_a_list[idx] += cs_a*KR;
            //}
            ////for (auto &p_a : p_a_list) {
            ////    p_a += cs_a*KR;
            ////}

            //p_a_list[0] += cs_a*KR;
            //p_a_list[1] += cs_a*KR;
            p_a0 += cs_a*KR;
            p_a1 += cs_a*KR;
            p_a2 += cs_a*KR;
            p_a3 += cs_a*KR;

            p_ap += ME*KR;
        }

        for (len_type kr = 0;kr < k-p;kr++)
        {
            for (len_type mr = 0;mr < MR;mr++)
            {
                ////p_ap[mr + ME*kr] = p_a[mr + cs_a*kr];
                //p_ap[mr + ME*kr] = 0;
                ////#pragma unroll
                //for (unsigned idx = 0; idx < N; idx++) {
                //    p_ap[mr + ME*kr] += coeff_list[idx] * p_a_list[idx][mr + cs_a*kr];
                //}

                //p_ap[mr + ME*kr] = coeff_list[0] * p_a_list[0][mr + cs_a*kr] + coeff_list[1] * p_a_list[1][mr + cs_a*kr];
                p_ap[mr + ME*kr] = coeff0 * p_a0[mr + cs_a*kr] + coeff1 * p_a1[mr + cs_a*kr]
                                 + coeff2 * p_a2[mr + cs_a*kr] + coeff3 * p_a3[mr + cs_a*kr];

            }
        }
    }
    else if (m == MR && cs_a == 1)
    {
        //std::cout << "KR: " << KR << ";k: " << k << std::endl;

        len_type p = 0;
        for (;p < k-KR;p += KR)
        {
            for (len_type kr = 0;kr < KR;kr++)
            {
                for (len_type mr = 0;mr < MR;mr++)
                {
                    ////p_ap[mr + ME*kr] = p_a[rs_a*mr + kr];
                    //p_ap[mr + ME*kr] = 0;
                    ////#pragma unroll
                    //for (unsigned idx = 0; idx < N; idx++) {
                    //    p_ap[mr + ME*kr] += coeff_list[idx] * p_a_list[idx][rs_a*mr + kr];
                    //    //std::cout << coeff_list[i] << "," << p_a_list[i][rs_a*mr + kr] << ";";
                    //}

                    //p_ap[mr + ME*kr] = coeff_list[0] * p_a_list[0][rs_a*mr + kr] + coeff_list[1] * p_a_list[1][rs_a*mr + kr];
                    p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + kr] + coeff1 * p_a1[rs_a*mr + kr]
                                     + coeff2 * p_a2[rs_a*mr + kr] + coeff3 * p_a3[rs_a*mr + kr];

                    //std::cout << "p_ap[" << mr << "][" << kr << "]=" << p_ap[mr + ME*kr] << std::endl;
                }
            }

            //p_a += KR;
            //for (auto &p_a : p_a_list) {
            //    p_a += KR;
            //}
            //for (unsigned idx = 0; idx < N; idx++) {
            //    p_a_list[idx] += KR;
            //}

            //p_a_list[0] += KR;
            //p_a_list[1] += KR;
            p_a0 += KR;
            p_a1 += KR;
            p_a2 += KR;
            p_a3 += KR;

            p_ap += ME*KR;
        }

        for (len_type kr = 0;kr < k-p;kr++)
        {
            for (len_type mr = 0;mr < MR;mr++)
            {
                ////p_ap[mr + ME*kr] = p_a[rs_a*mr + kr];
                //p_ap[mr + ME*kr] = 0;
                ////#pragma unroll
                //for (unsigned idx = 0; idx < N; idx++) {
                //    p_ap[mr + ME*kr] += coeff_list[idx] * p_a_list[idx][rs_a*mr + kr];
                //}

                //p_ap[mr + ME*kr] = coeff_list[0] * p_a_list[0][rs_a*mr + kr] + coeff_list[1] * p_a_list[1][rs_a*mr + kr];
                p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + kr] + coeff1 * p_a1[rs_a*mr + kr]
                                 + coeff2 * p_a2[rs_a*mr + kr] + coeff3 * p_a3[rs_a*mr + kr];
            }
        }
    }
    else
    {
        for (len_type p = 0;p < k;p++)
        {
            for (len_type mr = 0;mr < m;mr++)
            {
                ////p_ap[mr + ME*p] = p_a[rs_a*mr + cs_a*p];
                //p_ap[mr + ME*p] = 0;
                ////#pragma unroll
                //for (unsigned idx = 0; idx < N; idx++) {
                //    p_ap[mr + ME*p] += coeff_list[idx] * p_a_list[idx][rs_a*mr + cs_a*p];
                //}

                //p_ap[mr + ME*p] = coeff_list[0] * p_a_list[0][rs_a*mr + cs_a*p] + coeff_list[1] * p_a_list[1][rs_a*mr + cs_a*p];
                p_ap[mr + ME*p] = coeff0 * p_a0[rs_a*mr + cs_a*p] + coeff1 * p_a1[rs_a*mr + cs_a*p]
                                + coeff2 * p_a2[rs_a*mr + cs_a*p] + coeff3 * p_a3[rs_a*mr + cs_a*p];
            }

            for (len_type mr = m;mr < MR;mr++)
            {
                p_ap[mr + ME*p] = T();
            }
        }
    }
}


template <typename Config, typename T, int Mat>
void stra_pack_nn_ukr_def(len_type m, len_type k,
                     const unsigned N, const T** TBLIS_RESTRICT p_a_list, const T* TBLIS_RESTRICT coeff_list, stride_type rs_a, stride_type cs_a,
                     T* TBLIS_RESTRICT p_ap)
{
    //std::cout << "Enter stra_pack_nn_ukr_def\n" << std::endl;
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);
    constexpr len_type KR = Config::template gemm_kr<T>::def;

    if (m == MR && rs_a == 1)
    {
        len_type p = 0;
        for (;p < k-KR;p += KR)
        {
            for (len_type kr = 0;kr < KR;kr++)
            {
                for (len_type mr = 0;mr < MR;mr++)
                {

                    //p_ap[mr + ME*kr] = p_a[mr + cs_a*kr];
                    p_ap[mr + ME*kr] = 0;
                    //#pragma unroll
                    for (unsigned idx = 0; idx < N; idx++) {
                        p_ap[mr + ME*kr] += coeff_list[idx] * p_a_list[idx][mr + cs_a*kr];
                    }


                }
            }

            //p_a += cs_a*KR;
            for (unsigned idx = 0; idx < N; idx++) {
                p_a_list[idx] += cs_a*KR;
            }
            //for (auto &p_a : p_a_list) {
            //    p_a += cs_a*KR;
            //}
            
            p_ap += ME*KR;
        }

        for (len_type kr = 0;kr < k-p;kr++)
        {
            for (len_type mr = 0;mr < MR;mr++)
            {
                //p_ap[mr + ME*kr] = p_a[mr + cs_a*kr];
                p_ap[mr + ME*kr] = 0;
                //#pragma unroll
                for (unsigned idx = 0; idx < N; idx++) {
                    p_ap[mr + ME*kr] += coeff_list[idx] * p_a_list[idx][mr + cs_a*kr];
                }

            }
        }
    }
    else if (m == MR && cs_a == 1)
    {
        //std::cout << "KR: " << KR << ";k: " << k << std::endl;

        len_type p = 0;
        for (;p < k-KR;p += KR)
        {
            for (len_type kr = 0;kr < KR;kr++)
            {
                for (len_type mr = 0;mr < MR;mr++)
                {
                    //p_ap[mr + ME*kr] = p_a[rs_a*mr + kr];
                    p_ap[mr + ME*kr] = 0;
                    //#pragma unroll
                    for (unsigned idx = 0; idx < N; idx++) {
                        p_ap[mr + ME*kr] += coeff_list[idx] * p_a_list[idx][rs_a*mr + kr];
                        //std::cout << coeff_list[i] << "," << p_a_list[i][rs_a*mr + kr] << ";";
                    }

                    //std::cout << "p_ap[" << mr << "][" << kr << "]=" << p_ap[mr + ME*kr] << std::endl;

                }
            }

            //p_a += KR;
            //for (auto &p_a : p_a_list) {
            //    p_a += KR;
            //}
            for (unsigned idx = 0; idx < N; idx++) {
                p_a_list[idx] += KR;
            }

            p_ap += ME*KR;
        }

        for (len_type kr = 0;kr < k-p;kr++)
        {
            for (len_type mr = 0;mr < MR;mr++)
            {
                //p_ap[mr + ME*kr] = p_a[rs_a*mr + kr];
                p_ap[mr + ME*kr] = 0;
                //#pragma unroll
                for (unsigned idx = 0; idx < N; idx++) {
                    p_ap[mr + ME*kr] += coeff_list[idx] * p_a_list[idx][rs_a*mr + kr];
                }

            }
        }
    }
    else
    {
        for (len_type p = 0;p < k;p++)
        {
            for (len_type mr = 0;mr < m;mr++)
            {
                //p_ap[mr + ME*p] = p_a[rs_a*mr + cs_a*p];
                p_ap[mr + ME*p] = 0;
                //#pragma unroll
                for (unsigned idx = 0; idx < N; idx++) {
                    p_ap[mr + ME*p] += coeff_list[idx] * p_a_list[idx][rs_a*mr + cs_a*p];
                }

            }

            for (len_type mr = m;mr < MR;mr++)
            {
                p_ap[mr + ME*p] = T();
            }
        }
    }
}

template <typename Config, typename T, int Mat>
void pack_nn_ukr_def(len_type m, len_type k,
                     const T* TBLIS_RESTRICT p_a, stride_type rs_a, stride_type cs_a,
                     T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);
    constexpr len_type KR = Config::template gemm_kr<T>::def;

    if (m == MR && rs_a == 1)
    {
        len_type p = 0;
        for (;p < k-KR;p += KR)
        {
            for (len_type kr = 0;kr < KR;kr++)
            {
                for (len_type mr = 0;mr < MR;mr++)
                {
                    p_ap[mr + ME*kr] = p_a[mr + cs_a*kr];
                }
            }

            p_a += cs_a*KR;
            p_ap += ME*KR;
        }

        for (len_type kr = 0;kr < k-p;kr++)
        {
            for (len_type mr = 0;mr < MR;mr++)
            {
                p_ap[mr + ME*kr] = p_a[mr + cs_a*kr];
            }
        }
    }
    else if (m == MR && cs_a == 1)
    {
        len_type p = 0;
        for (;p < k-KR;p += KR)
        {
            for (len_type kr = 0;kr < KR;kr++)
            {
                for (len_type mr = 0;mr < MR;mr++)
                {
                    p_ap[mr + ME*kr] = p_a[rs_a*mr + kr];
                }
            }

            p_a += KR;
            p_ap += ME*KR;
        }

        for (len_type kr = 0;kr < k-p;kr++)
        {
            for (len_type mr = 0;mr < MR;mr++)
            {
                p_ap[mr + ME*kr] = p_a[rs_a*mr + kr];
            }
        }
    }
    else
    {
        for (len_type p = 0;p < k;p++)
        {
            for (len_type mr = 0;mr < m;mr++)
            {
                p_ap[mr + ME*p] = p_a[rs_a*mr + cs_a*p];
            }

            for (len_type mr = m;mr < MR;mr++)
            {
                p_ap[mr + ME*p] = T();
            }
        }
    }
}

template <typename Config, typename T, int Mat>
void pack_sn_ukr_def(len_type m, len_type k,
                     const T* TBLIS_RESTRICT p_a,
                     const stride_type* TBLIS_RESTRICT rscat_a, stride_type cs_a,
                     T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);

    for (len_type p = 0;p < k;p++)
    {
        for (len_type mr = 0;mr < m;mr++)
        {
            p_ap[mr + ME*p] = p_a[rscat_a[mr] + cs_a*p];
        }

        for (len_type mr = m;mr < MR;mr++)
        {
            p_ap[mr + ME*p] = T();
        }
    }
}

template <typename Config, typename T, int Mat>
void pack_ns_ukr_def(len_type m, len_type k,
                     const T* TBLIS_RESTRICT p_a,
                     stride_type rs_a, const stride_type* TBLIS_RESTRICT cscat_a,
                     T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);

    for (len_type p = 0;p < k;p++)
    {
        for (len_type mr = 0;mr < m;mr++)
        {
            p_ap[mr + ME*p] = p_a[rs_a*mr + cscat_a[p]];
        }

        for (len_type mr = m;mr < MR;mr++)
        {
            p_ap[mr + ME*p] = T();
        }
    }
}

template <typename Config, typename T, int Mat>
void pack_ss_ukr_def(len_type m, len_type k,
                     const T* TBLIS_RESTRICT p_a,
                     const stride_type* TBLIS_RESTRICT rscat_a,
                     const stride_type* TBLIS_RESTRICT cscat_a,
                     T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);

    for (len_type p = 0;p < k;p++)
    {
        for (len_type mr = 0;mr < m;mr++)
        {
            p_ap[mr + ME*p] = p_a[rscat_a[mr] + cscat_a[p]];
        }

        for (len_type mr = m;mr < MR;mr++)
        {
            p_ap[mr + ME*p] = T();
        }
    }
}

template <typename Config, typename T, int Mat>
void pack_nb_ukr_def(len_type m, len_type k,
                     const T* TBLIS_RESTRICT p_a,
                     stride_type rs_a, const stride_type* TBLIS_RESTRICT cscat_a,
                     const stride_type* TBLIS_RESTRICT cbs_a,
                     T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);
    constexpr len_type KR = Config::template gemm_kr<T>::def;

    if (m == MR && rs_a == 1)
    {
        len_type p = 0;
        for (;p < k-KR;p += KR)
        {
            //len_type k_loc = std::min(KR, k-p);
            stride_type cs_a = *cbs_a;
            stride_type off_a = *cscat_a;
            if (cs_a)
            {
                for (len_type kr = 0;kr < KR;kr++)  //k_loc -> KR
                {
                    for (len_type mr = 0;mr < MR;mr++)  // m -> MR
                    {
                        p_ap[mr + ME*kr] = p_a[mr + cs_a*kr + off_a];
                    }
                }
            }
            else
            {
                for (len_type kr = 0;kr < KR;kr++)
                {
                    for (len_type mr = 0;mr < MR;mr++)
                    {
                        p_ap[mr + ME*kr] = p_a[mr + cscat_a[kr]];
                    }
                }
            }
            p_ap += ME*KR;
            cscat_a += KR;
            cbs_a++;
        }
        {
            stride_type cs_a = *cbs_a;
            stride_type off_a = *cscat_a;
            if (cs_a)
            {
                for (len_type kr = 0;kr < k-p;kr++)  //k_loc -> KR
                {
                    for (len_type mr = 0;mr < MR;mr++)  // m -> MR
                    {
                        p_ap[mr + ME*kr] = p_a[mr + cs_a*kr + off_a];
                    }
                }
            }
            else
            {
                for (len_type kr = 0;kr < k-p;kr++)
                {
                    for (len_type mr = 0;mr < MR;mr++)
                    {
                        p_ap[mr + ME*kr] = p_a[mr + cscat_a[kr]];
                    }
                }
            }
        }
    }
    else // m != MR, or rs_a != 1: Do we need to separte the case where [m != MR AND rs_a == 1]?
    {
        len_type p = 0;
        for (;p < k-KR;p += KR)
        {
            //len_type k_loc = std::min(KR, k-p);
            stride_type cs_a = *cbs_a;
            stride_type off_a = *cscat_a;

            if (cs_a)
            {
                if (m == MR && cs_a == 1)
                {
                    for (len_type kr = 0;kr < KR;kr++)  //k_loc -> KR
                    {
                        for (len_type mr = 0;mr < MR;mr++)  // m -> MR
                        {
                            p_ap[mr + ME*kr] = p_a[rs_a*mr + kr + off_a];  // cs_a/rs_a can be 1;
                        }
                    }
                }
                else if (m == MR)
                {
                    for (len_type kr = 0;kr < KR;kr++)  //k_loc -> KR
                    {
                        for (len_type mr = 0;mr < MR;mr++)  // m -> MR
                        {
                            p_ap[mr + ME*kr] = p_a[rs_a*mr + cs_a*kr + off_a];
                        }
                    }
                }
                else
                {
                    for (len_type kr = 0;kr < KR;kr++)  //k_loc -> KR
                    {
                        for (len_type mr = 0;mr < m;mr++)  // m -> MR
                        {
                            p_ap[mr + ME*kr] = p_a[rs_a*mr + cs_a*kr + off_a];
                        }
                        for (len_type mr = m;mr < MR;mr++)
                        {
                            p_ap[mr + ME*kr] = T();
                        }
                    }
                }
            }
            else // cs_a == 0
            {
                if (m == MR)
                {
                    for (len_type kr = 0;kr < KR;kr++)
                    {
                        for (len_type mr = 0;mr < MR;mr++)
                        {
                            p_ap[mr + ME*kr] = p_a[rs_a*mr + cscat_a[kr]];
                        }
                    }
                }
                else // m != MR
                {
                    for (len_type kr = 0;kr < KR;kr++)
                    {
                        for (len_type mr = 0;mr < m;mr++)
                        {
                            p_ap[mr + ME*kr] = p_a[rs_a*mr + cscat_a[kr]];
                        }

                        for (len_type mr = m;mr < MR;mr++)
                        {
                            p_ap[mr + ME*kr] = T();
                        }
                    }
                }
            }

            p_ap += ME*KR;
            cscat_a += KR;
            cbs_a++;
        }

        { //k_left: k%KR
            stride_type cs_a = *cbs_a;
            stride_type off_a = *cscat_a;

            if (cs_a)
            {
                if (m == MR && cs_a == 1)
                {
                    for (len_type kr = 0;kr < k-p;kr++)  //k_loc -> KR
                    {
                        for (len_type mr = 0;mr < MR;mr++)  // m -> MR
                        {
                            p_ap[mr + ME*kr] = p_a[rs_a*mr + kr + off_a];  // cs_a/rs_a can be 1;
                        }
                    }
                }
                else if (m == MR)
                {
                    for (len_type kr = 0;kr < k-p;kr++)  //k_loc -> KR
                    {
                        for (len_type mr = 0;mr < MR;mr++)  // m -> MR
                        {
                            p_ap[mr + ME*kr] = p_a[rs_a*mr + cs_a*kr + off_a];
                        }
                    }
                }
                else
                {
                    for (len_type kr = 0;kr < k-p;kr++)  //k_loc -> KR
                    {
                        for (len_type mr = 0;mr < m;mr++)  // m -> MR
                        {
                            p_ap[mr + ME*kr] = p_a[rs_a*mr + cs_a*kr + off_a];
                        }
                        for (len_type mr = m;mr < MR;mr++)
                        {
                            p_ap[mr + ME*kr] = T();
                        }
                    }
                }
            }
            else // cs_a == 0
            {
                if (m == MR)
                {
                    for (len_type kr = 0;kr < k-p;kr++)
                    {
                        for (len_type mr = 0;mr < MR;mr++)
                        {
                            p_ap[mr + ME*kr] = p_a[rs_a*mr + cscat_a[kr]];
                        }
                    }
                }
                else // m != MR
                {
                    for (len_type kr = 0;kr < k-p;kr++)
                    {
                        for (len_type mr = 0;mr < m;mr++)
                        {
                            p_ap[mr + ME*kr] = p_a[rs_a*mr + cscat_a[kr]];
                        }

                        for (len_type mr = m;mr < MR;mr++)
                        {
                            p_ap[mr + ME*kr] = T();
                        }
                    }
                }
            }
        }

    }

}

////template<unsigned N>
//bool check_all_cs_a_nonzero_same( const stride_type** cbs_a, unsigned N )
//{                                                                          
//    for (unsigned idx = 0; idx < N; idx++) {                               
//        if ( cbs_a[idx][0] == 0 || cbs_a[idx][0] != cbs_a[0][0] ) {
//            return false;                                                  
//        }   
//    }   
//    return true;                                                           
//}   

template <typename Config, typename T, int Mat>
void stra_pack_two_nb_ukr_def(len_type m, len_type k,
                     const unsigned N, const T** TBLIS_RESTRICT p_a_list, const T* coeff_list,
                     stride_type rs_a, const stride_type** TBLIS_RESTRICT cscat_a,
                     const stride_type** TBLIS_RESTRICT cbs_a,
                     T* TBLIS_RESTRICT p_ap)
{
    //std::cout << "Enter stra_pack_nb_ukr_def\n" << std::endl;
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);
    constexpr len_type KR = Config::template gemm_kr<T>::def;

    const T coeff0 = coeff_list[0], coeff1 = coeff_list[1];
    const T *p_a0 = p_a_list[0], *p_a1 = p_a_list[1];
    const stride_type *cscat_a0 = cscat_a[0], *cscat_a1 = cscat_a[1];
    const stride_type *cbs_a0 = cbs_a[0], *cbs_a1 = cbs_a[1];


    if (m == MR && rs_a == 1)
    {
        len_type p = 0;
        for (;p < k-KR;p += KR)
        {
            const stride_type cs_a = *cbs_a0;
            const stride_type off_a0 = *cscat_a0;
            const stride_type off_a1 = *cscat_a1;

            bool is_all_cs_a_nonzero_same = true;
            if ( cbs_a0[0] == 0 || cbs_a1[0] == 0 || cbs_a1[0] != cbs_a0[0] ) {
                is_all_cs_a_nonzero_same = false;
            }

            if ( is_all_cs_a_nonzero_same )
            {
                for (len_type kr = 0;kr < KR;kr++)  //k_loc -> KR
                {
                    for (len_type mr = 0;mr < MR;mr++)  // m -> MR
                    {
                        //p_ap[mr + ME*kr] = p_a[mr + cs_a*kr + off_a];
                        p_ap[mr + ME*kr] = coeff0 * p_a0[mr + cs_a*kr + off_a0] + coeff1 * p_a1[mr + cs_a*kr + off_a1];
                    }
                }
            }
            else
            {
                for (len_type kr = 0;kr < KR;kr++)
                {
                    for (len_type mr = 0;mr < MR;mr++)
                    {
                        //p_ap[mr + ME*kr] = p_a[mr + cscat_a[kr]];
                        p_ap[mr + ME*kr] = coeff0 * p_a0[mr + cscat_a0[kr]] + coeff1 * p_a1[mr + cscat_a1[kr]];
                    }
                }
            }
            p_ap += ME*KR;

            cscat_a0 += KR;
            cscat_a1 += KR;
            cbs_a0 += 1;
            cbs_a1 += 1;
        }
        {
            const stride_type cs_a = *cbs_a0;
            const stride_type off_a0 = *cscat_a0;
            const stride_type off_a1 = *cscat_a1;

            bool is_all_cs_a_nonzero_same = true;
            if ( cbs_a0[0] == 0 || cbs_a1[0] == 0 || cbs_a1[0] != cbs_a0[0] ) {
                is_all_cs_a_nonzero_same = false;
            }

            if ( is_all_cs_a_nonzero_same )
            {
                for (len_type kr = 0;kr < k-p;kr++)  //k_loc -> KR
                {
                    for (len_type mr = 0;mr < MR;mr++)  // m -> MR
                    {
                        p_ap[mr + ME*kr] = coeff0 * p_a0[mr + cs_a*kr + off_a0] + coeff1 * p_a1[mr + cs_a*kr + off_a1];
                    }
                }
            }
            else
            {
                for (len_type kr = 0;kr < k-p;kr++)
                {
                    for (len_type mr = 0;mr < MR;mr++)
                    {
                        p_ap[mr + ME*kr] = coeff0 * p_a0[mr + cscat_a0[kr]] + coeff1 * p_a1[mr + cscat_a1[kr]];
                    }
                }
            }
        }
    }
    else // m != MR, or rs_a != 1: Do we need to separte the case where [m != MR AND rs_a == 1]?
    {
        len_type p = 0;
        for (;p < k-KR;p += KR)
        {
            const stride_type cs_a = *cbs_a0;
            const stride_type off_a0 = *cscat_a0;
            const stride_type off_a1 = *cscat_a1;

            bool is_all_cs_a_nonzero_same = true;

            if ( cbs_a0[0] == 0 || cbs_a1[0] == 0 || cbs_a1[0] != cbs_a0[0] ) {
                is_all_cs_a_nonzero_same = false;
            }
            if ( is_all_cs_a_nonzero_same )
            {

                if (m == MR && cs_a == 1)
                {
                    for (len_type kr = 0;kr < KR;kr++)  //k_loc -> KR
                    {
                        for (len_type mr = 0;mr < MR;mr++)  // m -> MR
                        {
                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + kr + off_a0] + coeff1 * p_a1[rs_a*mr + kr + off_a1];
                        }
                    }
                }
                else if (m == MR)
                {
                    for (len_type kr = 0;kr < KR;kr++)  //k_loc -> KR
                    {
                        for (len_type mr = 0;mr < MR;mr++)  // m -> MR
                        {
                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + cs_a*kr + off_a0] + coeff1 * p_a1[rs_a*mr + cs_a*kr + off_a1];
                        }
                    }
                }
                else // m != MR
                {
                    for (len_type kr = 0;kr < KR;kr++)  //k_loc -> KR
                    {
                        for (len_type mr = 0;mr < m;mr++)  // m -> MR
                        {
                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + cs_a*kr + off_a0] + coeff1 * p_a1[rs_a*mr + cs_a*kr + off_a1];
                        }
                        for (len_type mr = m;mr < MR;mr++)
                        {
                            p_ap[mr + ME*kr] = T();
                        }
                    }
                }
            }
            else // cs_a == 0
            {
                if (m == MR)
                {
                    for (len_type kr = 0;kr < KR;kr++)
                    {
                        for (len_type mr = 0;mr < MR;mr++)
                        {
                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + cscat_a0[kr]] + coeff1 * p_a1[rs_a*mr + cscat_a1[kr]];
                        }
                    }
                }
                else // m != MR
                {
                    for (len_type kr = 0;kr < KR;kr++)
                    {
                        for (len_type mr = 0;mr < m;mr++)
                        {

                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + cscat_a0[kr]] + coeff1 * p_a1[rs_a*mr + cscat_a1[kr]];
                        }

                        for (len_type mr = m;mr < MR;mr++)
                        {
                            p_ap[mr + ME*kr] = T();
                        }
                    }
                }
            }

            p_ap += ME*KR;
            cscat_a0 += KR;
            cscat_a1 += KR;
            cbs_a0 += 1;
            cbs_a1 += 1;
        }

        { //k_left: k%KR
            const stride_type cs_a = *cbs_a0;
            const stride_type off_a0 = *cscat_a0;
            const stride_type off_a1 = *cscat_a1;

            bool is_all_cs_a_nonzero_same = true;

            if ( cbs_a0[0] == 0 || cbs_a1[0] == 0 || cbs_a1[0] != cbs_a0[0] ) {
                is_all_cs_a_nonzero_same = false;
            }
 
            if ( is_all_cs_a_nonzero_same )
            {
                if (m == MR && cs_a == 1)
                {
                    for (len_type kr = 0;kr < k-p;kr++)  //k_loc -> KR
                    {
                        for (len_type mr = 0;mr < MR;mr++)  // m -> MR
                        {
                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + kr + off_a0] + coeff1 * p_a1[rs_a*mr + kr + off_a1];
                        }
                    }
                }
                else if (m == MR)
                {
                    for (len_type kr = 0;kr < k-p;kr++)  //k_loc -> KR
                    {
                        for (len_type mr = 0;mr < MR;mr++)  // m -> MR
                        {
                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + cs_a*kr + off_a0] + coeff1 * p_a1[rs_a*mr + cs_a*kr + off_a1];
                        }
                    }
                }
                else
                {
                    for (len_type kr = 0;kr < k-p;kr++)  //k_loc -> KR
                    {
                        for (len_type mr = 0;mr < m;mr++)  // m -> MR
                        {
                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + cs_a*kr + off_a0] + coeff1 * p_a1[rs_a*mr + cs_a*kr + off_a1];
                        }
                        for (len_type mr = m;mr < MR;mr++)
                        {
                            p_ap[mr + ME*kr] = T();
                        }
                    }
                }
            }
            else // cs_a == 0
            {
                if (m == MR)
                {
                    for (len_type kr = 0;kr < k-p;kr++)
                    {
                        for (len_type mr = 0;mr < MR;mr++)
                        {
                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + cscat_a0[kr]] + coeff1 * p_a1[rs_a*mr + cscat_a1[kr]];
                        }
                    }
                }
                else // m != MR
                {
                    for (len_type kr = 0;kr < k-p;kr++)
                    {
                        for (len_type mr = 0;mr < m;mr++)
                        {
                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + cscat_a0[kr]] + coeff1 * p_a1[rs_a*mr + cscat_a1[kr]];
                        }

                        for (len_type mr = m;mr < MR;mr++)
                        {
                            p_ap[mr + ME*kr] = T();
                        }
                    }
                }
            }
        }
    }


}


template <typename Config, typename T, int Mat>
void stra_pack_four_nb_ukr_def(len_type m, len_type k,
                     const unsigned N, const T** TBLIS_RESTRICT p_a_list, const T* coeff_list,
                     stride_type rs_a, const stride_type** TBLIS_RESTRICT cscat_a,
                     const stride_type** TBLIS_RESTRICT cbs_a,
                     T* TBLIS_RESTRICT p_ap)
{
    //std::cout << "Enter stra_pack_four_nb_ukr_def\n" << std::endl;
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);
    constexpr len_type KR = Config::template gemm_kr<T>::def;




    const T coeff0 = coeff_list[0], coeff1 = coeff_list[1], coeff2 = coeff_list[2], coeff3 = coeff_list[3];
    const T *p_a0 = p_a_list[0], *p_a1 = p_a_list[1], *p_a2 = p_a_list[2], *p_a3 = p_a_list[3];
    const stride_type *cscat_a0 = cscat_a[0], *cscat_a1 = cscat_a[1], *cscat_a2 = cscat_a[2], *cscat_a3 = cscat_a[3];
    const stride_type *cbs_a0 = cbs_a[0], *cbs_a1 = cbs_a[1], *cbs_a2 = cbs_a[2], *cbs_a3 = cbs_a[3];


    if (m == MR && rs_a == 1)
    {
        len_type p = 0;
        for (;p < k-KR;p += KR)
        {
            const stride_type cs_a = *cbs_a0;
            const stride_type off_a0 = *cscat_a0, off_a1 = *cscat_a1, off_a2 = *cscat_a2, off_a3 = *cscat_a3;

            bool is_all_cs_a_nonzero_same = true;

            if ( cbs_a0[0] == 0 || cbs_a1[0] == 0 || cbs_a2[0] == 0 || cbs_a3[0] == 0
                    || cbs_a1[0] != cbs_a0[0] || cbs_a2[0] != cbs_a0[0] || cbs_a3[0] != cbs_a0[0] ) {
                is_all_cs_a_nonzero_same = false;
            }


            if ( is_all_cs_a_nonzero_same )
            {
                for (len_type kr = 0;kr < KR;kr++)  //k_loc -> KR
                {
                    for (len_type mr = 0;mr < MR;mr++)  // m -> MR
                    {
                        //p_ap[mr + ME*kr] = p_a[mr + cs_a*kr + off_a];
                        p_ap[mr + ME*kr] = coeff0 * p_a0[mr + cs_a*kr + off_a0] + coeff1 * p_a1[mr + cs_a*kr + off_a1]
                            + coeff2 * p_a2[mr + cs_a*kr + off_a2] + coeff3 * p_a3[mr + cs_a*kr + off_a3];

                    }
                }
            }
            else
            {
                for (len_type kr = 0;kr < KR;kr++)
                {
                    for (len_type mr = 0;mr < MR;mr++)
                    {
                        //p_ap[mr + ME*kr] = p_a[mr + cscat_a[kr]];
                        p_ap[mr + ME*kr] = coeff0 * p_a0[mr + cscat_a0[kr]] + coeff1 * p_a1[mr + cscat_a1[kr]]
                            + coeff2 * p_a2[mr + cscat_a2[kr]] + coeff3 * p_a3[mr + cscat_a3[kr]];

                    }
                }
            }
            p_ap += ME*KR;


            cscat_a0 += KR;
            cscat_a1 += KR;
            cscat_a2 += KR;
            cscat_a3 += KR;
            cbs_a0 += 1;
            cbs_a1 += 1;
            cbs_a2 += 1;
            cbs_a3 += 1;

        }
        {
            const stride_type cs_a = *cbs_a0;
            const stride_type off_a0 = *cscat_a0, off_a1 = *cscat_a1, off_a2 = *cscat_a2, off_a3 = *cscat_a3;

            bool is_all_cs_a_nonzero_same = true;
            if ( cbs_a0[0] == 0 || cbs_a1[0] == 0 || cbs_a2[0] == 0 || cbs_a3[0] == 0
                    || cbs_a1[0] != cbs_a0[0] || cbs_a2[0] != cbs_a0[0] || cbs_a3[0] != cbs_a0[0] ) {
                is_all_cs_a_nonzero_same = false;
            }


            if ( is_all_cs_a_nonzero_same )
            {
                for (len_type kr = 0;kr < k-p;kr++)  //k_loc -> KR
                {
                    for (len_type mr = 0;mr < MR;mr++)  // m -> MR
                    {
                        p_ap[mr + ME*kr] = coeff0 * p_a0[mr + cs_a*kr + off_a0] + coeff1 * p_a1[mr + cs_a*kr + off_a1]
                            + coeff2 * p_a2[mr + cs_a*kr + off_a2] + coeff3 * p_a3[mr + cs_a*kr + off_a3];
                    }
                }
            }
            else
            {
                for (len_type kr = 0;kr < k-p;kr++)
                {
                    for (len_type mr = 0;mr < MR;mr++)
                    {
                        p_ap[mr + ME*kr] = coeff0 * p_a0[mr + cscat_a0[kr]] + coeff1 * p_a1[mr + cscat_a1[kr]]
                            + coeff2 * p_a2[mr + cscat_a2[kr]] + coeff3 * p_a3[mr + cscat_a3[kr]];

                    }
                }
            }
        }
    }
    else // m != MR, or rs_a != 1: Do we need to separte the case where [m != MR AND rs_a == 1]?
    {
        len_type p = 0;
        for (;p < k-KR;p += KR)
        {
            const stride_type cs_a = *cbs_a0;
            const stride_type off_a0 = *cscat_a0, off_a1 = *cscat_a1, off_a2 = *cscat_a2, off_a3 = *cscat_a3;

            bool is_all_cs_a_nonzero_same = true;

            if ( cbs_a0[0] == 0 || cbs_a1[0] == 0 || cbs_a1[0] != cbs_a0[0] ) {
                is_all_cs_a_nonzero_same = false;
            }
            if ( is_all_cs_a_nonzero_same )
            {

                if (m == MR && cs_a == 1)
                {
                    for (len_type kr = 0;kr < KR;kr++)  //k_loc -> KR
                    {
                        for (len_type mr = 0;mr < MR;mr++)  // m -> MR
                        {
                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + kr + off_a0] + coeff1 * p_a1[rs_a*mr + kr + off_a1]
                                + coeff2 * p_a2[rs_a*mr + kr + off_a2] + coeff3 * p_a3[rs_a*mr + kr + off_a3];


                        }
                    }
                }
                else if (m == MR)
                {
                    for (len_type kr = 0;kr < KR;kr++)  //k_loc -> KR
                    {
                        for (len_type mr = 0;mr < MR;mr++)  // m -> MR
                        {
                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + cs_a*kr + off_a0] + coeff1 * p_a1[rs_a*mr + cs_a*kr + off_a1]
                                + coeff2 * p_a2[rs_a*mr + cs_a*kr + off_a2] + coeff3 * p_a3[rs_a*mr + cs_a*kr + off_a3];

                        }
                    }
                }
                else // m != MR
                {
                    for (len_type kr = 0;kr < KR;kr++)  //k_loc -> KR
                    {
                        for (len_type mr = 0;mr < m;mr++)  // m -> MR
                        {
                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + cs_a*kr + off_a0] + coeff1 * p_a1[rs_a*mr + cs_a*kr + off_a1]
                                + coeff2 * p_a2[rs_a*mr + cs_a*kr + off_a2] + coeff3 * p_a3[rs_a*mr + cs_a*kr + off_a3];


                        }
                        for (len_type mr = m;mr < MR;mr++)
                        {
                            p_ap[mr + ME*kr] = T();
                        }
                    }
                }
            }
            else // cs_a == 0
            {
                if (m == MR)
                {
                    for (len_type kr = 0;kr < KR;kr++)
                    {
                        for (len_type mr = 0;mr < MR;mr++)
                        {
                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + cscat_a0[kr]] + coeff1 * p_a1[rs_a*mr + cscat_a1[kr]]
                                + coeff2 * p_a2[rs_a*mr + cscat_a2[kr]] + coeff3 * p_a3[rs_a*mr + cscat_a3[kr]];

                        }
                    }
                }
                else // m != MR
                {
                    for (len_type kr = 0;kr < KR;kr++)
                    {
                        for (len_type mr = 0;mr < m;mr++)
                        {

                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + cscat_a0[kr]] + coeff1 * p_a1[rs_a*mr + cscat_a1[kr]]
                                + coeff2 * p_a2[rs_a*mr + cscat_a2[kr]] + coeff3 * p_a3[rs_a*mr + cscat_a3[kr]];



                        }

                        for (len_type mr = m;mr < MR;mr++)
                        {
                            p_ap[mr + ME*kr] = T();
                        }
                    }
                }
            }

            p_ap += ME*KR;

            cscat_a0 += KR;
            cscat_a1 += KR;
            cscat_a2 += KR;
            cscat_a3 += KR;
            cbs_a0 += 1;
            cbs_a1 += 1;
            cbs_a2 += 1;
            cbs_a3 += 1;
        }

        { //k_left: k%KR
            const stride_type cs_a = *cbs_a0;
            const stride_type off_a0 = *cscat_a0, off_a1 = *cscat_a1, off_a2 = *cscat_a2, off_a3 = *cscat_a3;

            bool is_all_cs_a_nonzero_same = true;
 
            if ( is_all_cs_a_nonzero_same )
            {
                if (m == MR && cs_a == 1)
                {
                    for (len_type kr = 0;kr < k-p;kr++)  //k_loc -> KR
                    {
                        for (len_type mr = 0;mr < MR;mr++)  // m -> MR
                        {
                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + kr + off_a0] + coeff1 * p_a1[rs_a*mr + kr + off_a1]
                                + coeff2 * p_a2[rs_a*mr + kr + off_a2] + coeff3 * p_a3[rs_a*mr + kr + off_a3];

                        }
                    }
                }
                else if (m == MR)
                {
                    for (len_type kr = 0;kr < k-p;kr++)  //k_loc -> KR
                    {
                        for (len_type mr = 0;mr < MR;mr++)  // m -> MR
                        {
                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + cs_a*kr + off_a0] + coeff1 * p_a1[rs_a*mr + cs_a*kr + off_a1]
                                + coeff2 * p_a2[rs_a*mr + cs_a*kr + off_a2] + coeff3 * p_a3[rs_a*mr + cs_a*kr + off_a3];
                        }
                    }
                }
                else // m != MR
                {
                    for (len_type kr = 0;kr < k-p;kr++)  //k_loc -> KR
                    {
                        for (len_type mr = 0;mr < m;mr++)  // m -> MR
                        {
                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + cs_a*kr + off_a0] + coeff1 * p_a1[rs_a*mr + cs_a*kr + off_a1]
                                + coeff2 * p_a2[rs_a*mr + cs_a*kr + off_a2] + coeff3 * p_a3[rs_a*mr + cs_a*kr + off_a3];

                        }
                        for (len_type mr = m;mr < MR;mr++)
                        {
                            p_ap[mr + ME*kr] = T();
                        }
                    }
                }
            }
            else // cs_a == 0
            {
                if (m == MR)
                {
                    for (len_type kr = 0;kr < k-p;kr++)
                    {
                        for (len_type mr = 0;mr < MR;mr++)
                        {
                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + cscat_a0[kr]] + coeff1 * p_a1[rs_a*mr + cscat_a1[kr]]
                                + coeff2 * p_a2[rs_a*mr + cscat_a2[kr]] + coeff3 * p_a3[rs_a*mr + cscat_a3[kr]];
                        }
                    }
                }
                else // m != MR
                {
                    for (len_type kr = 0;kr < k-p;kr++)
                    {
                        for (len_type mr = 0;mr < m;mr++)
                        {
                            p_ap[mr + ME*kr] = coeff0 * p_a0[rs_a*mr + cscat_a0[kr]] + coeff1 * p_a1[rs_a*mr + cscat_a1[kr]]
                                + coeff2 * p_a2[rs_a*mr + cscat_a2[kr]] + coeff3 * p_a3[rs_a*mr + cscat_a3[kr]];
                        }

                        for (len_type mr = m;mr < MR;mr++)
                        {
                            p_ap[mr + ME*kr] = T();
                        }
                    }
                }
            }
        }
    }
}



template <typename Config, typename T, int Mat>
void stra_pack_nb_ukr_def(len_type m, len_type k,
                     const unsigned N, const T** TBLIS_RESTRICT p_a_list, const T* coeff_list,
                     stride_type rs_a, const stride_type** TBLIS_RESTRICT cscat_a,
                     const stride_type** TBLIS_RESTRICT cbs_a,
                     T* TBLIS_RESTRICT p_ap)
{
    //std::cout << "Enter stra_pack_nb_ukr_def\n" << std::endl;
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);
    constexpr len_type KR = Config::template gemm_kr<T>::def;

    for (len_type p = 0;p < k;p += KR)
    {
        len_type k_loc = std::min(KR, k-p);
        //stride_type cs_a = *cbs_a;
        stride_type cs_a = *cbs_a[0];
        //stride_type off_a = *cscat_a[0];

        //stride_type cs_a[N];
        //for (unsigned idx = 0; idx < N; idx++) {
        //    cs_a[idx] = *cbs_a[idx];
        //}
        //stride_type off_a = *cscat_a[0];
        stride_type off_a[N];
        for (unsigned idx = 0; idx < N; idx++) {
            //off_a[idx] = cscat_a[idx][0];
            off_a[idx] = *cscat_a[idx];
        }

        //std::cout << "cs_a: " << cs_a << "; rs_a: " << rs_a << "; off_a: " << off_a << std::endl;
        //std::cout << "*cscat_a[0]: " << *cscat_a[0] << "; *cscat_a[1]: " << *cscat_a[1] << std::endl;

        //std::cout << "p_a_list[1][0]: " << p_a_list[1][0] << std::endl;


        //bool is_all_cs_a_nonzero_same = check_all_cs_a_nonzero_same( cbs_a, N );
        bool is_all_cs_a_nonzero_same = true;
        for (unsigned idx = 0; idx < N; idx++) {
            if ( cbs_a[idx][0] == 0 || cbs_a[idx][0] != cbs_a[0][0] ) {
                is_all_cs_a_nonzero_same = false;
                break;
            }
        }

        //if (cs_a)
        if ( is_all_cs_a_nonzero_same )
        {
            //std::cout << "is_all_cs_a_nonzero_same" << std::endl;
            for (len_type kr = 0;kr < k_loc;kr++)
            {
                for (len_type mr = 0;mr < m;mr++)
                {
                    //Gather pattern
                    //p_ap[mr + ME*kr] = p_a[rs_a*mr + cs_a*kr + off_a];

                    p_ap[mr + ME*kr] = 0;
                    //std::cout << "[";
                    //#pragma unroll
                    for (unsigned idx = 0; idx < N; idx++) {
                        //p_ap[mr + ME*kr] += coeff_list[idx] * p_a_list[idx][rs_a*mr + cs_a*kr + off_a];
                        p_ap[mr + ME*kr] += coeff_list[idx] * p_a_list[idx][rs_a*mr + cs_a*kr + off_a[idx]];
                        //std::cout << p_a_list[idx][rs_a*mr + cs_a*kr + off_a[idx]] << ",";
                    }

                    //std::cout << p_ap[mr + ME*kr] << "] ";
                }

                //std::cout << std::endl;

                for (len_type mr = m;mr < MR;mr++)
                {
                    p_ap[mr + ME*kr] = T();
                }
            }
        }
        else
        {

            //std::cout << "not is_all_cs_a_nonzero_same" << std::endl;
            for (len_type kr = 0;kr < k_loc;kr++)
            {
                for (len_type mr = 0;mr < m;mr++)
                {

                    //p_ap[mr + ME*kr] = p_a[rs_a*mr + cscat_a[kr]];

                    p_ap[mr + ME*kr] = 0;
                    //std::cout << "[";
                    //#pragma unroll
                    for (unsigned idx = 0; idx < N; idx++) {
                        p_ap[mr + ME*kr] += coeff_list[idx] * p_a_list[idx][rs_a*mr + cscat_a[idx][kr]];

                        //std::cout << "(rs_a: " << rs_a << "; mr: " << mr << "; kr: " << kr << "; cscat_a[" << idx << "][" << kr << "]:" << cscat_a[idx][kr] << "):";
                        

                        ////std::cout << p_a_list[idx][rscat_a[idx][mr] + cscat_a[idx][p]] << ",";
                        //std::cout <<p_a_list[idx][rs_a*mr + cscat_a[idx][kr]] << ", ";
                    }

                    //std::cout << p_ap[mr + ME*kr] << "] ";

                }

                //std::cout << std::endl;

                for (len_type mr = m;mr < MR;mr++)
                {
                    p_ap[mr + ME*kr] = T();
                }
            }
        }

        p_ap += ME*KR;

        //cscat_a += KR;
        //cbs_a++;
        for (unsigned idx = 0; idx < N; idx++) {
            cscat_a[idx] += KR;
            cbs_a[idx] += 1;
        } // Next block
    }
}

template <typename Config, typename T, int Mat>
void pack_sb_ukr_def(len_type m, len_type k,
                     const T* TBLIS_RESTRICT p_a,
                     const stride_type* TBLIS_RESTRICT rscat_a,
                     const stride_type* TBLIS_RESTRICT cscat_a,
                     const stride_type* TBLIS_RESTRICT cbs_a,
                     T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);

    (void)cbs_a;

    for (len_type p = 0;p < k;p++)
    {
        for (len_type mr = 0;mr < m;mr++)
        {
            p_ap[mr + ME*p] = p_a[rscat_a[mr] + cscat_a[p]];
        }

        for (len_type mr = m;mr < MR;mr++)
        {
            p_ap[mr + ME*p] = T();
        }
    }
}

template <typename Config, typename T, int Mat>
void stra_pack_two_sb_ukr_def(len_type m, len_type k,
                              const unsigned N, const T* TBLIS_RESTRICT p_a, const T* coeff_list,
                              const stride_type** TBLIS_RESTRICT rscat_a,
                              const stride_type** TBLIS_RESTRICT cscat_a,
                              const stride_type** TBLIS_RESTRICT cbs_a,
                              T* TBLIS_RESTRICT p_ap)
{
    //std::cout << "Enter stra_pack_sb_ukr_def\n" << std::endl;
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);

    const T coeff0 = coeff_list[0], coeff1 = coeff_list[1];
    const T *p_a0 = p_a, *p_a1 = p_a;
    const stride_type *rscat_a0 = rscat_a[0], *rscat_a1 = rscat_a[1];
    const stride_type *cscat_a0 = cscat_a[0], *cscat_a1 = cscat_a[1];

    (void)cbs_a;

    for (len_type p = 0;p < k;p++)
    {
        for (len_type mr = 0;mr < m;mr++)
        {

            ////p_ap[mr + ME*p] = p_a[rscat_a[mr] + cscat_a[p]];
            //p_ap[mr + ME*p] = 0;
            ////std::cout << "[";
            ////#pragma unroll
            //for (unsigned idx = 0; idx < N; idx++)
            //{
            //    p_ap[mr + ME*p] += coeff_list[idx] * p_a_list[idx][rscat_a[idx][mr] + cscat_a[idx][p]];

            //    //std::cout << p_a_list[idx][rscat_a[idx][mr] + cscat_a[idx][p]] << ",";
            //}
            ////std::cout << p_ap[mr + ME*p] << "] ";

            //p_ap[mr + ME*p] = coeff_list[0] * p_a_list[0][rscat_a[0][mr] + cscat_a[0][p]] + coeff_list[1] * p_a_list[1][rscat_a[1][mr] + cscat_a[1][p]];
            p_ap[mr + ME*p] = coeff0 * p_a0[rscat_a0[mr] + cscat_a0[p]] + coeff1 * p_a1[rscat_a1[mr] + cscat_a1[p]];
        }

        //std::cout << std::endl;

        for (len_type mr = m;mr < MR;mr++)
        {
            p_ap[mr + ME*p] = T();
        }
    }
}

template <typename Config, typename T, int Mat>
void stra_pack_four_sb_ukr_def(len_type m, len_type k,
                              const unsigned N, const T* TBLIS_RESTRICT p_a, const T* coeff_list,
                              const stride_type** TBLIS_RESTRICT rscat_a,
                              const stride_type** TBLIS_RESTRICT cscat_a,
                              const stride_type** TBLIS_RESTRICT cbs_a,
                              T* TBLIS_RESTRICT p_ap)
{
    //std::cout << "Enter stra_pack_four_sb_ukr_def\n" << std::endl;
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);

    const T coeff0 = coeff_list[0], coeff1 = coeff_list[1], coeff2 = coeff_list[2], coeff3 = coeff_list[3];
    const T *p_a0 = p_a, *p_a1 = p_a, *p_a2 = p_a, *p_a3 = p_a;
    const stride_type *rscat_a0 = rscat_a[0], *rscat_a1 = rscat_a[1], *rscat_a2 = rscat_a[2], *rscat_a3 = rscat_a[3];
    const stride_type *cscat_a0 = cscat_a[0], *cscat_a1 = cscat_a[1], *cscat_a2 = cscat_a[2], *cscat_a3 = cscat_a[3];

    (void)cbs_a;

    for (len_type p = 0;p < k;p++)
    {
        for (len_type mr = 0;mr < m;mr++)
        {

            ////p_ap[mr + ME*p] = p_a[rscat_a[mr] + cscat_a[p]];
            //p_ap[mr + ME*p] = 0;
            ////std::cout << "[";
            ////#pragma unroll
            //for (unsigned idx = 0; idx < N; idx++)
            //{
            //    p_ap[mr + ME*p] += coeff_list[idx] * p_a_list[idx][rscat_a[idx][mr] + cscat_a[idx][p]];

            //    //std::cout << p_a_list[idx][rscat_a[idx][mr] + cscat_a[idx][p]] << ",";
            //}
            ////std::cout << p_ap[mr + ME*p] << "] ";

            //p_ap[mr + ME*p] = coeff_list[0] * p_a_list[0][rscat_a[0][mr] + cscat_a[0][p]] + coeff_list[1] * p_a_list[1][rscat_a[1][mr] + cscat_a[1][p]];
            //p_ap[mr + ME*p] = coeff0 * p_a0[rscat_a0[mr] + cscat_a0[p]] + coeff1 * p_a1[rscat_a1[mr] + cscat_a1[p]];
            p_ap[mr + ME*p] = coeff0 * p_a0[rscat_a0[mr] + cscat_a0[p]] + coeff1 * p_a1[rscat_a1[mr] + cscat_a1[p]]
                            + coeff2 * p_a2[rscat_a2[mr] + cscat_a2[p]] + coeff3 * p_a3[rscat_a3[mr] + cscat_a3[p]];
        }

        //std::cout << std::endl;

        for (len_type mr = m;mr < MR;mr++)
        {
            p_ap[mr + ME*p] = T();
        }
    }
}

template <typename Config, typename T, int Mat>
void stra_pack_sb_ukr_def(len_type m, len_type k,
                     const unsigned N, const T* TBLIS_RESTRICT p_a, const T* coeff_list,
                     const stride_type** TBLIS_RESTRICT rscat_a,
                     const stride_type** TBLIS_RESTRICT cscat_a,
                     const stride_type** TBLIS_RESTRICT cbs_a,
                     T* TBLIS_RESTRICT p_ap)
{
    //std::cout << "Enter stra_pack_sb_ukr_def\n" << std::endl;
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);

    (void)cbs_a;

    for (len_type p = 0;p < k;p++)
    {
        for (len_type mr = 0;mr < m;mr++)
        {
            //p_ap[mr + ME*p] = p_a[rscat_a[mr] + cscat_a[p]];
            p_ap[mr + ME*p] = 0;
            //std::cout << "[";
            //#pragma unroll
            for (unsigned idx = 0; idx < N; idx++)
            {
                p_ap[mr + ME*p] += coeff_list[idx] * p_a[rscat_a[idx][mr] + cscat_a[idx][p]];

                //std::cout << p_a_list[idx][rscat_a[idx][mr] + cscat_a[idx][p]] << ",";
            }
            //std::cout << p_ap[mr + ME*p] << "] ";

        }

        //std::cout << std::endl;

        for (len_type mr = m;mr < MR;mr++)
        {
            p_ap[mr + ME*p] = T();
        }
    }
}

}

#endif
