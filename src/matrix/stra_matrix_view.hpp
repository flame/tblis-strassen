#ifndef _TBLIS_STRA_MATRIX_VIEW_HPP_
#define _TBLIS_STRA_MATRIX_VIEW_HPP_

#include "util/basic_types.h"

namespace tblis
{

template <typename T, unsigned N>
class stra_matrix_view
{
    protected:
        unsigned stra_size_;
        std::array<T*,N> data_;
        std::array<T,N> coeff_;
        std::array<len_type,2> len_;
        std::array<stride_type,2> stride_;

    public:
        stra_matrix_view(const std::array<len_type,2>& len,
                const std::array<T*,N>& ptr,
                const std::array<T,N>& coeff,
                const std::array<stride_type,2>& stride)
            : data_(ptr), len_(len), coeff_(coeff), stride_(stride), stra_size_(N) {}

        unsigned stra_size() {
            return stra_size_;
        }

        //Not sure if the following is correct...
        std::array<T*,N> data_list() {
            return data_;
        }

        std::array<T,N> coeff_list() {
            return coeff_;
        }

        T* data(unsigned idx) {
            return data_[idx];
        }

        T coeff(unsigned idx) {
            return coeff_[idx];
        }

        len_type length(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return len_[dim];
        }

        len_type length(unsigned dim, len_type len)
        {
            TBLIS_ASSERT(dim < 2);
            //len_[dim] = len;
            std::swap(len, len_[dim]);
            return len;
        }

        stride_type stride(unsigned dim)
        {
            TBLIS_ASSERT(dim < 2);
            return stride_[dim]; 
        }

        void stride(unsigned dim, len_type str)
        {
            TBLIS_ASSERT(dim < 2);
            stride_[dim] = str;
        }

        void shift(unsigned dim, len_type amount)
        {
            TBLIS_ASSERT(dim < 2);
            for(unsigned idx = 0; idx < N; idx++) {
                data_[idx] += amount*stride_[dim];
            }
        }
};

}

#endif
