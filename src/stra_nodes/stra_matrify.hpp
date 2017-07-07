#ifndef _TBLIS_STRA_NODES_MATRIFY_HPP_
#define _TBLIS_STRA_NODES_MATRIFY_HPP_

#include "util/basic_types.h"
#include "util/thread.h"

#include "matrix/tensor_matrix.hpp"

//#include "nodes/packm.hpp"
#include "stra_nodes/stra_packm.hpp"

#include "configs/configs.hpp"

namespace tblis
{

namespace detail
{
    extern MemoryPool BuffersForScatter;
}

/*
template <typename MatrixA>
void block_scatter(const communicator& comm, MatrixA& A,
                   stride_type* rscat, len_type MB, stride_type* rbs,
                   stride_type* cscat, len_type NB, stride_type* cbs)
{
    len_type m = A.length(0);
    len_type n = A.length(1);

    len_type first, last;
    std::tie(first, last, std::ignore) = comm.distribute_over_threads(m, MB);

    A.length(0, last-first);
    A.shift(0, first);
    A.fill_block_scatter(0, rscat+first, MB, rbs+first/MB);
    A.shift(0, -first);
    A.length(0, m);

    std::tie(first, last, std::ignore) = comm.distribute_over_threads(n, NB);

    A.length(1, last-first);
    A.shift(1, first);
    A.fill_block_scatter(1, cscat+first, NB, cbs+first/NB);
    A.shift(1, -first);
    A.length(1, n);

    comm.barrier();
}
*/



//template<int Mat> struct stra_size;
//
//template <> struct stra_size<matrix_constants::MAT_A>
//{
//    template <typename T, unsigned N, typename MatrixB, typename MatrixC>
//    constexpr unsigned stra_size(stra_tensor_view<T,N>& A, MatrixB& B, MatrixC& C)
//    {
//        return N;
//    }
//}

template <typename T, unsigned N>
constexpr unsigned stra_size(stra_tensor_view<T,N>& A)
{
    return N;
}

template <typename MatrixA>
//constexpr unsigned stra_size(const MatrixA &,...)
constexpr unsigned stra_size(MatrixA &,...)
{
    return 1;
}



template <int Mat> struct stra_matrify_and_run;

template <> struct stra_matrify_and_run<matrix_constants::MAT_A>
{
    template <typename T, typename Parent, typename MatrixA, typename MatrixB, typename MatrixC>
    stra_matrify_and_run(Parent& parent, const communicator& comm, const config& cfg,
                    T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        const len_type MB = cfg.gemm_mr.def<T>();
        const len_type NB = cfg.gemm_kr.def<T>();

        const len_type m = A.length(0);
        const len_type n = A.length(1);
        parent.cscat = parent.rscat+m;
        parent.rbs   = parent.cscat+n;
        parent.cbs   = parent.rbs+m;

        //block_scatter(comm, A, parent.rscat, MB, parent.rbs,
        //                       parent.cscat, NB, parent.cbs);

        //std::cout << "Enter stra_matrify_and_run: MAT_A" << std::endl;
        //std::cout << "stra_size(A): " << stra_size(A) << std::endl;

        for (unsigned idx=0; idx < stra_size(A); idx++)
        {
            const unsigned offset = idx*2*(m+n);
            
            A.fill_block_scatter(idx, 0, parent.rscat+offset, MB, parent.rbs+offset);
            A.fill_block_scatter(idx, 1, parent.cscat+offset, NB, parent.cbs+offset);

            //std::cout << "idx: " << idx << std::endl;
            //std::cout << "A.length(0):" << A.length(0) << std::endl;
            //std::cout << "A.length(1):" << A.length(1) << std::endl;
            //std::cout << "A:rscat:" << std::endl;
            //for (unsigned i = 0; i < A.length(0); i++) {
            //    std::cout << parent.rscat[offset+i] << " ";
            //}
            //std::cout << std::endl;
            //std::cout << "A:cscat:" << std::endl;
            //for (unsigned i = 0; i < A.length(1); i++) {
            //    std::cout << parent.cscat[offset+i] << " ";
            //}
            //std::cout << std::endl;

            //std::cout << "A:rbs:" << std::endl;
            //for (unsigned i = 0; i < A.length(0); i++) {
            //    std::cout << parent.rbs[offset+i] << " ";
            //}
            //std::cout << std::endl;
            //std::cout << "A:cbs:" << std::endl;
            //for (unsigned i = 0; i < A.length(1); i++) {
            //    std::cout << parent.cbs[offset+i] << " ";
            //}
            //std::cout << std::endl;

        }

        //std::cout << "Exit stra_matrify_and_run: MAT_A" << std::endl;

        //rbs -> 0, cbs -> 0: scatter matrix format

        // Change A to M view (block scatter matrix view)
        ////////////////////TO BE MODIFIED: A.data(0)
        auto buf = A.data();
        auto coeff = A.coeff_list();
        stra_block_scatter_matrix<T, stra_size(A) > M(A.length(0), A.length(1), buf, coeff,
                                                      parent.rscat, MB, parent.rbs,
                                                      parent.cscat, NB, parent.cbs);

        parent.child(comm, cfg, alpha, M, B, beta, C);
    }
};

template <> struct stra_matrify_and_run<matrix_constants::MAT_B>
{
    template <typename T, typename Parent, typename MatrixA, typename MatrixB, typename MatrixC>
    stra_matrify_and_run(Parent& parent, const communicator& comm, const config& cfg,
                    T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {

        //std::cout << "Enter stra_matrify_and_run: MAT_B" << std::endl;
        //std::cout << "stra_size(B): " << stra_size(B) << std::endl;

        const len_type MB = cfg.gemm_kr.def<T>();
        const len_type NB = cfg.gemm_nr.def<T>();

        const len_type m = B.length(0);
        const len_type n = B.length(1);
        parent.cscat = parent.rscat+m;
        parent.rbs   = parent.cscat+n;
        parent.cbs   = parent.rbs+m;

        //block_scatter(comm, B, parent.rscat, MB, parent.rbs,
        //                       parent.cscat, NB, parent.cbs);


        for (unsigned idx=0; idx < stra_size(B); idx++)
        {
            const unsigned offset = idx*2*(B.length(0)+B.length(1));
            B.fill_block_scatter(idx, 0, parent.rscat+offset, MB, parent.rbs+offset);
            B.fill_block_scatter(idx, 1, parent.cscat+offset, NB, parent.cbs+offset);

            //std::cout << std::endl;
            //std::cout << "idx: " << idx << std::endl;
            //std::cout << "B.length(0):" << B.length(0) << std::endl;
            //std::cout << "B.length(1):" << B.length(1) << std::endl;
            //std::cout << "B:rscat:" << std::endl;
            //for (unsigned i = 0; i < B.length(0); i++) {
            //    std::cout << parent.rscat[offset+i] << " ";
            //}
            //std::cout << std::endl;
            //std::cout << "B:cscat:" << std::endl;
            //for (unsigned i = 0; i < B.length(1); i++) {
            //    std::cout << parent.cscat[offset+i] << " ";
            //}
            //std::cout << std::endl;

            //std::cout << "B:rbs:" << std::endl;
            //for (unsigned i = 0; i < B.length(0); i++) {
            //    std::cout << parent.rbs[offset+i] << " ";
            //}
            //std::cout << std::endl;
            //std::cout << "B:cbs:" << std::endl;
            //for (unsigned i = 0; i < B.length(1); i++) {
            //    std::cout << parent.cbs[offset+i] << " ";
            //}
            //std::cout << std::endl;
        }

        //std::cout << "Exit stra_matrify_and_run: MAT_B" << std::endl;



        // Change B to M view (block scatter matrix view)
        ////////////////////TO BE MODIFIED: B.data(0)
        //std::array<T*,stra_size(B)> buf = B.data_list();
        auto buf = B.data();
        auto coeff = B.coeff_list();
        stra_block_scatter_matrix<T, stra_size(B)> M(B.length(0), B.length(1), buf, coeff,
                                                     parent.rscat, MB, parent.rbs,
                                                     parent.cscat, NB, parent.cbs);

        //stra_block_scatter_matrix<T> M.... ->  B(2 buffers)

        
        //exit( 0 );

        parent.child(comm, cfg, alpha, A, M, beta, C);
    }
};

template <> struct stra_matrify_and_run<matrix_constants::MAT_C>
{
    template <typename T, typename Parent, typename MatrixA, typename MatrixB, typename MatrixC>
    stra_matrify_and_run(Parent& parent, const communicator& comm, const config& cfg,
                    T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        const len_type MB = cfg.gemm_mr.def<T>();
        const len_type NB = cfg.gemm_nr.def<T>();

        const len_type m = C.length(0);
        const len_type n = C.length(1);
        parent.cscat = parent.rscat+m;
        parent.rbs   = parent.cscat+n;
        parent.cbs   = parent.rbs+m;

        //block_scatter(comm, C, parent.rscat, MB, parent.rbs,
        //                       parent.cscat, NB, parent.cbs);

        for (unsigned idx=0; idx < stra_size(C); idx++)
        {
            const unsigned offset = idx*2*(C.length(0)+C.length(1));
            C.fill_block_scatter(idx, 0, parent.rscat+offset, MB, parent.rbs+offset);
            C.fill_block_scatter(idx, 1, parent.cscat+offset, NB, parent.cbs+offset);


            //std::cout << "idx: " << idx << std::endl;
            //std::cout << "C.length(0):" << C.length(0) << std::endl;
            //std::cout << "C.length(1):" << C.length(1) << std::endl;
            //std::cout << "C:rscat:" << std::endl;
            //for (unsigned i = 0; i < C.length(0); i++) {
            //    std::cout << parent.rscat[offset+i] << " ";
            //}
            //std::cout << std::endl;
            //std::cout << "C:cscat:" << std::endl;
            //for (unsigned i = 0; i < C.length(1); i++) {
            //    std::cout << parent.cscat[offset+i] << " ";
            //}
            //std::cout << std::endl;

            //std::cout << "C:rbs:" << std::endl;
            //for (unsigned i = 0; i < C.length(0); i++) {
            //    std::cout << parent.rbs[offset+i] << " ";
            //}
            //std::cout << std::endl;
            //std::cout << "C:cbs:" << std::endl;
            //for (unsigned i = 0; i < C.length(1); i++) {
            //    std::cout << parent.cbs[offset+i] << " ";
            //}
            //std::cout << std::endl;
        }

        //std::cout << "C:all:" << std::endl;
        //for (unsigned i = 0; i < stra_size(C)*2*(C.length(0) + C.length(1)); i++) {
        //        std::cout << parent.rscat[i] << " ";
        //}
        //std::cout << std::endl;


        // Change C to M view (block scatter matrix view)
        ////////////////////TO BE MODIFIED: C.data(0)

        auto buf = C.data();
        auto coeff = C.coeff_list();
        stra_block_scatter_matrix<T, stra_size(C)> M(C.length(0), C.length(1), buf, coeff,
                                  parent.rscat, MB, parent.rbs,
                                  parent.cscat, NB, parent.cbs);

        parent.child(comm, cfg, alpha, A, B, beta, M);
    }
};


template <int Mat, MemoryPool& Pool, typename Child>
struct stra_matrify
{
    Child child;
    MemoryPool::Block scat_buffer;
    stride_type* rscat = nullptr;
    stride_type* cscat = nullptr;
    stride_type* rbs = nullptr;
    stride_type* cbs = nullptr;


    template <typename T, typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(const communicator& comm, const config& cfg,
                    T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        using namespace matrix_constants;

        len_type m = (Mat == MAT_A ? A.length(0) : Mat == MAT_B ? B.length(0) : C.length(0));
        len_type n = (Mat == MAT_A ? A.length(1) : Mat == MAT_B ? B.length(1) : C.length(1));

        const unsigned NN = (Mat == MAT_A ? stra_size(A) : Mat == MAT_B ? stra_size(B) : stra_size(C));

        if (!rscat)
        {
            if (comm.master())
            {
                //if ( Mat == MAT_C ) std::cout << "MAT_C: m: " << m << "; n: " << n << std::endl;
                scat_buffer = Pool.allocate<stride_type>( (2*m + 2*n) * NN );
                rscat = scat_buffer.get<stride_type>();
            }

            comm.broadcast(rscat);

            //cscat = rscat+m;
            //rbs = cscat+n;
            //cbs = rbs+m;
        }

        stra_matrify_and_run<Mat>(*this, comm, cfg, alpha, A, B, beta, C);
    }
};

template <MemoryPool& Pool, typename Child>
using stra_matrify_a = stra_matrify<matrix_constants::MAT_A, Pool, Child>;

template <MemoryPool& Pool, typename Child>
using stra_matrify_b = stra_matrify<matrix_constants::MAT_B, Pool, Child>;

template <MemoryPool& Pool, typename Child>
using stra_matrify_c = stra_matrify<matrix_constants::MAT_C, Pool, Child>;

template <int Mat, MemoryPool& Pool, typename Child>
struct stra_matrify_and_pack : stra_matrify<Mat, Pool, stra_pack<Mat, Pool, Child>>
{
    typedef stra_matrify<Mat, Pool, stra_pack<Mat, Pool, Child>> Sib;

    using Sib::child;
    using Sib::rscat;
    using Sib::cscat;
    using Sib::rbs;
    using Sib::cbs;

    template <typename T, typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(const communicator& comm, const config& cfg,
                    T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        using namespace matrix_constants;

        const len_type MR = (Mat == MAT_B ? cfg.gemm_kr.def<T>()
                                          : cfg.gemm_mr.def<T>());
        const len_type NR = (Mat == MAT_A ? cfg.gemm_kr.def<T>()
                                          : cfg.gemm_nr.def<T>());

        len_type m = (Mat == MAT_A ? A.length(0) : Mat == MAT_B ? B.length(0) : C.length(0));
        len_type n = (Mat == MAT_A ? A.length(1) : Mat == MAT_B ? B.length(1) : C.length(1));
        m = round_up(m, MR);
        n = round_up(n, NR);

        //std::cout << "Enter matrify" << std::endl;

        auto& pack_buffer = child.pack_buffer;
        auto& pack_ptr = child.pack_ptr;

        if (!pack_ptr)
        {
            if (comm.master())
            {
                const unsigned NN = (Mat == MAT_A ? stra_size(A) : Mat == MAT_B ? stra_size(B) : stra_size(C));
                len_type scatter_size = size_as_type<stride_type,T>( 2 * (m + n) * NN );
                pack_buffer = Pool.allocate<T>(m*n + std::max(m,n)*TBLIS_MAX_UNROLL + scatter_size);
                pack_ptr = pack_buffer.get();
            }

            comm.broadcast(pack_ptr);

            rscat = convert_and_align<T,stride_type>(static_cast<T*>(pack_ptr) + m*n);
            cscat = rscat+m;
            rbs = cscat+n;
            cbs = rbs+m;
        }

        Sib::operator()(comm, cfg, alpha, A, B, beta, C);
    }
};

template <MemoryPool& Pool, typename Child>
using stra_matrify_and_pack_a = stra_matrify_and_pack<matrix_constants::MAT_A, Pool, Child>;

template <MemoryPool& Pool, typename Child>
using stra_matrify_and_pack_b = stra_matrify_and_pack<matrix_constants::MAT_B, Pool, Child>;

}

#endif
