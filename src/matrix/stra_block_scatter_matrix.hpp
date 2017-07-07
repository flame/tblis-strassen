#ifndef _TBLIS_STRA_BLOCK_SCATTER_MATRIX_HPP_
#define _TBLIS_STRA_BLOCK_SCATTER_MATRIX_HPP_

#include "util/basic_types.h"

namespace tblis
{

template <typename T, unsigned N>
class stra_block_scatter_matrix
{
    public:
        typedef size_t size_type;
        typedef const stride_type* scatter_type;
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef const T& const_reference;

    protected:
        //std::array<pointer, N> data_;
        pointer data_;
        std::array<value_type, N> coeff_;
        std::array<len_type, 2> len_;
        std::array<std::array<scatter_type, 2>, N> block_scatter_;
        std::array<std::array<scatter_type, 2>, N> scatter_;
        std::array<len_type, 2> block_size_;

    public:
        stra_block_scatter_matrix()
        {
            reset();
        }

        stra_block_scatter_matrix(const stra_block_scatter_matrix&) = default;

        stra_block_scatter_matrix(len_type m, len_type n, 
                                  pointer p,
                                  std::array<T,N>& coeff,
                                  std::array<scatter_type,N>& rscat, len_type MB, std::array<scatter_type,N>& rbs,
                                  std::array<scatter_type,N>& cscat, len_type NB, std::array<scatter_type,N>& cbs)
        {
            reset(m, n, p, coeff, rscat, MB, rbs, cscat, NB, cbs);
        }

        stra_block_scatter_matrix(len_type m, len_type n, 
                                  pointer p,
                                  std::array<T,N>& coeff,
                                  scatter_type rscat, len_type MB, scatter_type rbs,
                                  scatter_type cscat, len_type NB, scatter_type cbs)
        {
            reset(m, n, p, coeff, rscat, MB, rbs, cscat, NB, cbs);
        }

        stra_block_scatter_matrix& operator=(const stra_block_scatter_matrix&) = delete;

        void reset()
        {
            data_ = nullptr;
            len_[0] = 0;
            len_[1] = 0;
            block_scatter_[0] = nullptr;
            block_scatter_[1] = nullptr;
            scatter_[0] = nullptr;
            scatter_[1] = nullptr;
        }

        void reset(const stra_block_scatter_matrix& other)
        {
            data_ = other.data_;
            len_[0] = other.len_[0];
            len_[1] = other.len_[1];
            block_scatter_[0] = other.block_scatter_[0];
            block_scatter_[1] = other.block_scatter_[1];
            scatter_[0] = other.scatter_[0];
            scatter_[1] = other.scatter_[1];
        }

        void reset(len_type m, len_type n,
                   pointer p, std::array<T,N>& coeff,
                   std::array<scatter_type,N>& rscat, len_type MB, std::array<scatter_type,N>& rbs,
                   std::array<scatter_type,N>& cscat, len_type NB, std::array<scatter_type,N>& cbs)
        {
            data_ = p;
            coeff_ = coeff;
            len_[0] = m;
            len_[1] = n;

            for (unsigned idx = 0; idx < N; idx++)
            {
                block_scatter_[idx][0] = rbs[idx];
                block_scatter_[idx][1] = cbs[idx];
                scatter_[idx][0] = rscat[idx];
                scatter_[idx][1] = cscat[idx];


            }

            block_size_[0] = MB;
            block_size_[1] = NB;

            for (unsigned idx = 0; idx < N; idx++)
            {
                for (len_type i = 0;i < m;i += MB)
                {
                    stride_type s = (m-i) > 1 ? rscat[idx][i+1]-rscat[idx][i] : 1;
                    for (len_type j = i+1;j+1 < std::min(i+MB,m);j++)
                    {
                        if (rscat[idx][j+1]-rscat[idx][j] != s) s = 0;
                    }
                    TBLIS_ASSERT(s == -1 || s == rbs[i/MB]);
                }

                for (len_type i = 0;i < n;i += NB)
                {
                    stride_type s = (n-i) > 1 ? cscat[idx][i+1]-cscat[idx][i] : 1;
                    for (len_type j = i+1;j+1 < std::min(i+NB,n);j++)
                    {
                        if (cscat[idx][j+1]-cscat[idx][j] != s) s = 0;
                    }
                    TBLIS_ASSERT(s == -1 || s == cbs[i/NB]);
                }

            }

        }

        void reset(len_type m, len_type n,
                   pointer p,
                   std::array<T,N>& coeff,
                   scatter_type rscat, len_type MB, scatter_type rbs,
                   scatter_type cscat, len_type NB, scatter_type cbs)
        {
            data_ = p;
            coeff_ = coeff;
            len_[0] = m;
            len_[1] = n;

            for (unsigned idx = 0; idx < N; idx++)
            {
                const unsigned offset = idx*2*(m+n);
                block_scatter_[idx][0] = &(rbs[offset]);
                block_scatter_[idx][1] = &(cbs[offset]);
                scatter_[idx][0] = &(rscat[offset]);
                scatter_[idx][1] = &(cscat[offset]);

                //std::cout << "cbs[" << offset << "]:" << cbs[offset] << std::endl;

                //std::cout << "block_scatter_[" << idx <<  "][1]: " <<  *block_scatter_[idx][1] << std::endl;

                //std::cout << "idx: " << idx << std::endl;
                //std::cout << "m:" << m << std::endl;
                //std::cout << "n:" << n << std::endl;
                //std::cout << "C:rscat:" << std::endl; 
                //for (unsigned i = 0; i < m; i++) {
                //        std::cout << rscat[offset+i] << " ";
                //    }
                //std::cout << std::endl; 
                //std::cout << "C:cscat:" << std::endl; 
                //for (unsigned i = 0; i < n; i++) {
                //        std::cout << cscat[offset+i] << " ";
                //    }
                //std::cout << std::endl;
                //
                //    std::cout << "C:rbs:" << std::endl;
                //for (unsigned i = 0; i < m; i++) {
                //        std::cout << rbs[offset+i] << " ";
                //    }
                //std::cout << std::endl;
                //std::cout << "C:cbs:" << std::endl;
                //for (unsigned i = 0; i < n; i++) {
                //        std::cout << cbs[offset+i] << " ";
                //    }
                //std::cout << std::endl;


            }






            block_size_[0] = MB;
            block_size_[1] = NB;

            for (unsigned idx = 0; idx < N; idx++)
            {
                const unsigned offset = idx*2*(m+n);
                for (len_type i = 0;i < m;i += MB)
                {
                    stride_type s = (m-i) > 1 ? rscat[offset+i+1]-rscat[offset+i] : 1;
                    for (len_type j = i+1;j+1 < std::min(i+MB,m);j++)
                    {
                        if (rscat[offset+j+1]-rscat[offset+j] != s) s = 0;
                    }
                    TBLIS_ASSERT(s == -1 || s == rbs[offset+i/MB]);
                }

                for (len_type i = 0;i < n;i += NB)
                {
                    stride_type s = (n-i) > 1 ? cscat[offset+i+1]-cscat[offset+i] : 1;
                    for (len_type j = i+1;j+1 < std::min(i+NB,n);j++)
                    {
                        if (cscat[offset+j+1]-cscat[offset+j] != s) s = 0;
                    }
                    TBLIS_ASSERT(s == -1 || s == cbs[offset+i/NB]);
                }

            }

        }

        len_type block_size(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return block_size_[dim];
        }

        len_type length(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return len_[dim];
        }

        len_type length(unsigned dim, len_type m)
        {
            TBLIS_ASSERT(dim < 2);
            std::swap(m, len_[dim]);
            return m;
        }

        stride_type stride(unsigned idx, unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return *block_scatter_[idx][dim];
        }

        scatter_type scatter(unsigned idx, unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return scatter_[idx][dim];
        }

        scatter_type block_scatter(unsigned idx, unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return block_scatter_[idx][dim];
        }

        void shift(unsigned dim, len_type n)
        {
            TBLIS_ASSERT(dim < 2);
            for(unsigned idx = 0; idx < N; idx++) {
                scatter_[idx][dim] += n;
                block_scatter_[idx][dim] += ceil_div(n, block_size_[dim]);
            }
        }

        void shift_down(unsigned dim)
        {
            shift(dim, length(dim));
        }

        void shift_up(unsigned dim)
        {
            shift(dim, -length(dim));
        }

        void shift_block(unsigned dim, len_type n)
        {
            TBLIS_ASSERT(dim < 2);
            for(unsigned idx = 0; idx < N; idx++) {
                scatter_[idx][dim] += n*block_size_[dim];
                block_scatter_[idx][dim] += n;
            }
        }

        //std::array<T*,N> raw_data_list() {
        //    return data_;
        //}

        ////std::array<T*,N> data_list() {
        ////    return data_;
        ////}

        std::array<T,N> coeff_list() {
            return coeff_;
        }

        pointer data(unsigned idx)
        {
            return data_ + (stride(idx, 0) == 0 ? 0 : *scatter_[idx][0])
                         + (stride(idx, 1) == 0 ? 0 : *scatter_[idx][1]);
        }

        const_pointer data(unsigned idx) const
        {
            return const_cast<stra_block_scatter_matrix&>(*this).data(idx);
        }

        pointer raw_data() { return data_; }

        const_pointer raw_data() const { return data_; }
};

}

#endif
