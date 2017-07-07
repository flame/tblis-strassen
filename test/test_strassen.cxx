#include <cstdlib>
#include <algorithm>
#include <limits>
#include <stdint.h>
#include <iostream>
#include <random>
#include <numeric>
#include <getopt.h>
#include <sstream>
#include <type_traits>
#include <iomanip>
#include <functional>
#include <set>
#include <map>

#include "tblis.h"
#include "util/time.hpp"
#include "util/tensor.hpp"
#include "util/random.hpp"

#include "omp.h"

#include "internal/3t/mult.hpp"

using namespace std;
using namespace tblis;
using namespace stl_ext;

#define PRINT_VECTOR( name ) \
    std::cout << #name << ": " ; \
for (auto &elem : name) \
{ \
    std::cout << elem << " "; \
} \
std::cout << std::endl;

char* get_arg(char **begin, char **end, const std::string &option){
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end){
        return *itr;
    }
    return 0;
}

map<char, len_type> parse_lengths(const string &len_str)
{
    std::string delimiter = ";";
    string s(len_str);
    map<char, len_type> lengths;
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);

        size_t colon1 = token.find(':');
        if (colon1 != 1) {
            std::cout << "Error" << std::endl;
            exit(0);
        }
        char ch = token[0];
        len_type mylen = stol( token.substr(2) );
        lengths[ch] = mylen;
        s.erase(0, pos + delimiter.length());
    }
    return lengths;
}

template<typename T>
void trashCache( T *trash1, T *trash2, int nTotal){
    for(int i = 0; i < nTotal; i ++)
        trash1[i] += 0.99 * trash2[i];
}

template<typename Kernel, typename ...Args>
double run_kernel(len_type R, const Kernel & kernel, Args &&...args)
{
    double *trash1, *trash2;
    int nTotal = 1024*1024*100;
    trash1 = (double*)malloc(sizeof(double)*nTotal);
    trash2 = (double*)malloc(sizeof(double)*nTotal);

    double dt = numeric_limits<double>::max();
    for (len_type r = 0;r < R;r++)
    {
        trashCache(trash1, trash2, nTotal);
        double t0 = omp_get_wtime();
        kernel(args...);
        double t1 = omp_get_wtime();
        dt = min(dt, t1-t0);
    }
    free( trash1 );
    free( trash2 );

    return dt;
}

template<typename T>
T vec_prod( std::vector<T> &vec )
{
    T prod = 1;
    for( auto &elem : vec )
    {
        prod *= elem;
    }
    return prod;
}

#define TOLERANCE 10E-1

template<typename T>
T compare_tensor(T *C, T *C_ref, len_type total_size)
{
    T norm2_error = 0.0, norm2_ref = 0.0;
    for (len_type i=0; i < total_size; i++)
    {
        norm2_error += ( ( C[i]-C_ref[i] ) * ( C[i]-C_ref[i] ) );
        norm2_ref   += C_ref[i] * C_ref[i];
        //if ( fabs( C[i]-C_ref[i] ) > TOLERANCE ) 
        //{
        //    std::cout << "Wrong Result!: C[" << i << "]=" << C[i] << "; C_ref[" << i << "]=" << C_ref[i] << std::endl;
        //    break;
        //}
    }
    if ( norm2_ref == 0.0 ) norm2_ref = 1;
    //printf( "norm2_error: %lf; norm2_ref: %lf\n", norm2_error, norm2_ref );
    return  norm2_error / norm2_ref;
}

template<typename T, int N=3>
struct random_contraction
{
    len_type R;
    random_contraction(len_type R,
                       len_type m,
                       len_type n,
                       len_type k,
                       int level,
                       int impl)
    : R(R)
    {
        operator()(m, n, k, level, impl);
    }

    void operator()(len_type m, len_type n, len_type k, int level, int impl) const
    {
        for (int i = 0;i < N;i++)
        {
            vector<len_type> len_m =
                random_product_constrained_sequence<len_type, ROUND_NEAREST>(random_number(1, 3), m);
            vector<len_type> len_n =
                random_product_constrained_sequence<len_type, ROUND_NEAREST>(random_number(1, 3), n);
            vector<len_type> len_k =
                random_product_constrained_sequence<len_type, ROUND_NEAREST>(random_number(1, 3), k);

            vector<label_type> idx_A, idx_B, idx_C;
            vector<len_type> len_A, len_B, len_C;
            char idx = 'a';

            map<char,len_type> lengths;

            stride_type tm = 1;
            for (len_type len : len_m)
            {
                idx_A.push_back(idx);
                len_A.push_back(len);
                idx_C.push_back(idx);
                len_C.push_back(len);
                lengths[idx] = len;
                idx++;
                tm *= len;
            }

            stride_type tn = 1;
            for (len_type len : len_n)
            {
                idx_B.push_back(idx);
                len_B.push_back(len);
                idx_C.push_back(idx);
                len_C.push_back(len);
                lengths[idx] = len;
                idx++;
                tn *= len;
            }

            stride_type tk = 1;
            for (len_type len : len_k)
            {
                idx_A.push_back(idx);
                len_A.push_back(len);
                idx_B.push_back(idx);
                len_B.push_back(len);
                lengths[idx] = len;
                idx++;
                tk *= len;
            }

            vector<unsigned> reorder_A = range<unsigned>(len_A.size());
            vector<unsigned> reorder_B = range<unsigned>(len_B.size());
            vector<unsigned> reorder_C = range<unsigned>(len_C.size());

            random_shuffle(reorder_A.begin(), reorder_A.end());
            random_shuffle(reorder_B.begin(), reorder_B.end());
            random_shuffle(reorder_C.begin(), reorder_C.end());

            permute(idx_A, reorder_A);
            permute(len_A, reorder_A);
            permute(idx_B, reorder_B);
            permute(len_B, reorder_B);
            permute(idx_C, reorder_C);
            permute(len_C, reorder_C);

            tensor<T> A(len_A, 1.0);
            tensor<T> B(len_B, 2.0);
            tensor<T> C(len_C, 0.0);
            tensor<T> C_ref(len_C, 0.0);

            //Initialize the Tensor elements
            //srand48 (time(NULL));
            for (int i = 0; i < vec_prod( len_m ) * vec_prod( len_k ); i++) {
                (A.data())[i] = (double)(drand48());
                //(A.data())[i] = (double)(1.0);
            }
            for (int i = 0; i < vec_prod( len_k ) * vec_prod( len_n ); i++) {
                (B.data())[i] = (double)(drand48());
                //(B.data())[i] = (double)(1.0);
            }

            double gflops = 2*tm*tn*tk*1e-9;

            switch ( impl ) {
                case 1:
                    tblis::internal::impl = tblis::internal::BLIS_BASED;
                    break;
                case 2:
                    tblis::internal::impl = tblis::internal::STRA_AB;
                    break;
                case 3:
                    tblis::internal::impl = tblis::internal::STRA_NAIVE;
                    break;
                default:
                    std::cout << "No such Implementation!" << std::endl;
                    break;
            }

            double dt, dt2;
            dt = run_kernel(R,
            [&]
            {   mult(T(1), A, idx_A.data(),
                           B, idx_B.data(),
                     T(1), C_ref, idx_C.data());
            });

            if ( level == 0 ) {
                dt2 = run_kernel(R,
                [&]
                {   mult(T(1), A, idx_A.data(),
                               B, idx_B.data(),
                         T(1), C, idx_C.data());
                });
            } else if ( level == 1 ) {
                dt2 = run_kernel(R,
                [&]
                {   stra_mult(T(1), A, idx_A.data(),
                               B, idx_B.data(),
                         T(1), C, idx_C.data());
                });
            } else if ( level == 2 ) {
                dt2 = run_kernel(R,
                [&]
                {   stra_mult_2level(T(1), A, idx_A.data(),
                               B, idx_B.data(),
                         T(1), C, idx_C.data());
                });
            } else {
                printf( "More than 2-level is not supported\n" );
            }

            len_type total_size = vec_prod( len_m ) * vec_prod( len_n );
            double norm2_error = compare_tensor( C.data(), C_ref.data(), total_size );

            printf("%5ld %5ld %5ld\t %10.3e %10.3e %8.3e\t %7.2lf %7.2lf\t %5.2lf\n", tm, tn, tk, dt, dt2, gflops, gflops / dt, gflops / dt2, norm2_error);

            fflush(stdout);
        }
    }
};

template<typename T>
struct regular_contraction
{
    len_type R;
    vector<label_type> idx_A, idx_B, idx_C;

    regular_contraction(len_type R,
                        const vector<label_type> &idx_A,
                        const vector<label_type> &idx_B,
                        const vector<label_type> &idx_C,
                        const map<char,len_type> &lengths,
                        int level,
                        int impl)
    : R(R), idx_A(idx_A), idx_B(idx_B), idx_C(idx_C)
    {
        operator()( lengths, level, impl );
    }

    void operator()(const map<char, len_type> &lengths, int level, int impl) const
    {
        vector<len_type> len_A, len_B, len_C;

        stride_type ntot = 1;
        for (auto & p : lengths) ntot *= p.second;


        for (char c : idx_A)
        {
            len_A.push_back(lengths.at(c));
        }

        for (char c : idx_B)
        {
            len_B.push_back(lengths.at(c));
        }

        for (char c : idx_C)
        {
            len_C.push_back(lengths.at(c));
        }

        auto idx_AB = stl_ext::intersection(idx_A, idx_B);
        auto len_AB = stl_ext::select_from(len_A, idx_A, idx_AB);
        auto idx_AC = stl_ext::intersection(idx_A, idx_C);
        auto len_AC = stl_ext::select_from(len_A, idx_A, idx_AC);
        auto idx_BC = stl_ext::intersection(idx_B, idx_C);
        auto len_BC = stl_ext::select_from(len_B, idx_B, idx_BC);

        auto pm = vec_prod( len_AC );
        auto pn = vec_prod( len_BC );
        auto pk = vec_prod( len_AB );


        tensor<T> A(len_A);
        tensor<T> B(len_B);
        tensor<T> C(len_C);
        tensor<T> C_ref(len_C);

        //Initialize the Tensor elements
        //srand48 (time(NULL));
        for (int i = 0; i < vec_prod( len_A ); i++) {
            (A.data())[i] = (double)(drand48());
        }
        for (int i = 0; i < vec_prod( len_B ); i++) {
            (B.data())[i] = (double)(drand48());
        }

        double gflops = 2*ntot*1e-9;

        switch ( impl ) {
            case 1: 
                tblis::internal::impl = tblis::internal::BLIS_BASED;
                break;
            case 2:
                tblis::internal::impl = tblis::internal::STRA_AB;
                break;
            case 3:
                tblis::internal::impl = tblis::internal::STRA_NAIVE;
                break;
            default:
                std::cout << "No such Implementation!" << std::endl;
                break;
        }

        double dt, dt2;
        dt = run_kernel(R,
        [&]
        {
            mult(T(1), A, idx_A.data(),
                       B, idx_B.data(),
                 T(1), C_ref, idx_C.data());
        });

        if ( level == 0 ) {
            dt2 = run_kernel(R,
            [&]
            {   mult(T(1), A, idx_A.data(),
                           B, idx_B.data(),
                     T(1), C, idx_C.data());
            });
        } else if ( level == 1 ) {
            dt2 = run_kernel(R,
            [&]
            {   stra_mult(T(1), A, idx_A.data(),
                           B, idx_B.data(),
                     T(1), C, idx_C.data());
            });
        } else if ( level == 2 ) {
            dt2 = run_kernel(R,
            [&]
            {   stra_mult_2level(T(1), A, idx_A.data(),
                           B, idx_B.data(),
                     T(1), C, idx_C.data());
            });
        } else {
            printf( "More than 2-level is not supported\n" );
        }

        len_type total_size = vec_prod( len_C );
        double norm2_error = compare_tensor( C.data(), C_ref.data(), total_size );

        printf("%5ld %5ld %5ld\t %10.3e %10.3e %8.3e\t %7.2lf %7.2lf\t %5.2lf\n", pm, pn, pk, dt, dt2, gflops, gflops / dt, gflops / dt2, norm2_error);

        //printf("%5ld %5ld %5ld\t%5ld %5ld %5ld\t%5.2lf %5.2lf %5.2lf\n", m, n, k, tm, tn, tk, gflops / dt, gflops / dt2, norm2_error);


        fflush(stdout);
    }
};

int main(int argc, char** argv)
{
    int L, level, impl;
    len_type m, n, k;
    len_type niter;
    int const in_num  = argc;
    char ** input_str = argv;

    if (get_arg(input_str, input_str+in_num, "-niter")){
        niter = atoi(get_arg(input_str, input_str+in_num, "-niter"));
        if (niter < 0) niter = 3;
    } else {
        niter = 3;
    }

    if (get_arg(input_str, input_str+in_num, "-m")){
        m = atoi(get_arg(input_str, input_str+in_num, "-m"));
        if (m < 0) m = 256;
    } else {
        m = 256;
    }

    if (get_arg(input_str, input_str+in_num, "-n")){
        n = atoi(get_arg(input_str, input_str+in_num, "-n"));
        if (n < 0) n = 256;
    } else {
        n = 256;
    }

    if (get_arg(input_str, input_str+in_num, "-k")){
        k = atoi(get_arg(input_str, input_str+in_num, "-k"));
        if (k < 0) k = 256;
    } else {
        k = 256;
    }

    if (get_arg(input_str, input_str+in_num, "-impl")){
        impl = atoi(get_arg(input_str, input_str+in_num, "-impl"));
        if (impl <= 0 || impl > 3) impl = 1;
    } else {
        impl = 1;
    }

    if (get_arg(input_str, input_str+in_num, "-level")){
        level = atoi(get_arg(input_str, input_str+in_num, "-level"));
        if (level < 0 || level > 3) {
            level = 0;
        }
        if (level == 0) {
            impl = 1;
        }
    } else {
        level = 1;
    }

    char* filename;
    if (get_arg(input_str, input_str+in_num, "-file")){
        filename = get_arg(input_str, input_str+in_num, "-file");
    } else {
        filename = nullptr;
    }

    // Using the default seed if not specify seed, or specify the seed any number except -1 (-1 will lead to seed bounded to timing).
    // This can guarantee all implementations run on the same sythesized dataset.
    time_t seed;
    if (get_arg(input_str, input_str+in_num, "-seed")){
        seed = atoi(get_arg(input_str, input_str+in_num, "-seed"));
        if ( seed != -1 ) {
            rand_engine.seed(seed);
        }
    } else {
        seed = time(nullptr);
        rand_engine.seed(seed);
    }

    if ( filename == nullptr ) {
        random_contraction<double>( niter, m, n, k, level, impl );
    } else {
        freopen( filename, "r", stdin );

        string line;
        while (getline(cin, line) && !line.empty())
        {
            if (line[0] == '#') continue;
            istringstream iss(line);
            {
                string idxABC_;
                string idx_A_, idx_B_, idx_C_;

                iss >> idx_C_;
                iss >> idx_A_;
                iss >> idx_B_;

                std::set<char>labels;
                for (char c : idx_A_) labels.insert(c);
                for (char c : idx_B_) labels.insert(c);
                for (char c : idx_C_) labels.insert(c);
                string symboland;
                iss >> symboland;

                map<char, len_type> lengths;
                string length_str;
                iss >> length_str;
                lengths = parse_lengths( length_str );
                vector<label_type> idx_A(idx_A_.begin(), idx_A_.end());
                vector<label_type> idx_B(idx_B_.begin(), idx_B_.end());
                vector<label_type> idx_C(idx_C_.begin(), idx_C_.end());

                regular_contraction<double>(niter, idx_A, idx_B, idx_C, lengths, level, impl);

            }
        }
    }
    return 0;
}

