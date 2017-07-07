#import abc_gen

import sys

from common import is_one, is_negone, is_nonzero, write_line, write_break, transpose, printmat, contain_nontrivial

def write_header_start( myfile ):
    myfile.write( \
'''\
#ifndef BLISLAB_DFMM_KERNEL_H
#define BLISLAB_DFMM_KERNEL_H

#include "bl_config.h"

#include <stdio.h>
#include <immintrin.h> // AVX


// Allow C++ users to include this header file in their source code. However,
// we make the extern "C" conditional on whether we're using a C++ compiler,
// since regular C compilers don't understand the extern "C" construct.
#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned long long dim_t;

typedef union {
    __m256d v;
    __m256i u;
    double d[ 4 ];
} v4df_t;


typedef union {
    __m128i v;
    int d[ 4 ];
} v4li_t;

struct aux_s {
    double *b_next;
    float  *b_next_s;
    char   *flag;
    int    pc;
    int    m;
    int    n;
};
typedef struct aux_s aux_t;

void bl_dgemm_asm_8x4( int k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data );

void bl_dgemm_asm_8x4_mulstrassen( int k,
                                   double *a,
                                   double *b,
                                   unsigned long long len_c, unsigned long long ldc,
                                   double **c_list, double *alpha_list,
                                   aux_t* data );

static void (*bl_micro_kernel) (
        int k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data
        ) = {
        BL_MICRO_KERNEL
};

''')


def write_header_end( myfile ):
    myfile.write( \
'''\
// End extern "C" construct block.
#ifdef __cplusplus
}
#endif

#endif
''')


def generate_kernel_header( myfile, nonzero_coeffs, index ):
    nnz = len( nonzero_coeffs )
    #write_line( myfile, 1, 'a' )
    add = 'inline void bl_dgemm_micro_kernel_stra_abc%d( int k, double *a, double *b, ' % index
    add += 'unsigned long long ldc, '
    add += ', '.join( ['double *c%d' % ( i ) for i in range( nnz )] )
    if ( contain_nontrivial( nonzero_coeffs ) ):
        add += ', double *alpha_list'
    add += ', aux_t *aux );'
    write_line(myfile, 0, add)
    #write_break( myfile )


def write_prefetch_assembly( myfile, nonzero_coeffs ):
    for j, coeff in enumerate(nonzero_coeffs):
        myfile.write( \
'''\
    "movq                %{0}, %%{2}               \\n\\t" // load address of c{1}
    "leaq   (%%{2},%%rdi,2), %%{3}               \\n\\t" // load address of c{1} + 2 * ldc;
    "prefetcht0   3 * 8(%%{2})                   \\n\\t" // prefetch c{1} + 0 * ldc
    "prefetcht0   3 * 8(%%{2},%%rdi)             \\n\\t" // prefetch c{1} + 1 * ldc
    "prefetcht0   3 * 8(%%{3})                   \\n\\t" // prefetch c{1} + 2 * ldc
    "prefetcht0   3 * 8(%%{3},%%rdi)             \\n\\t" // prefetch c{1} + 3 * ldc
'''.format( str(j+6), str(j), get_reg(), get_reg() ) )



#Round Robin way to get the register
def get_reg( avoid_reg = '' ):
    get_reg.counter += 1
    res_reg = get_reg.reg_pool[ get_reg.counter % len(get_reg.reg_pool) ]
    if ( res_reg == avoid_reg ):
        get_reg.counter += 1
        res_reg = get_reg.reg_pool[ get_reg.counter % len(get_reg.reg_pool) ]
    return res_reg

get_reg.counter = -1
get_reg.reg_pool = [ 'rcx', 'rdx', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14' ]
#get_reg.reg_pool = [ 'rcx', 'rdx', 'rsi', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14' ]
# rdi, rax, rbx, r15, already occupied.
# (rcx, rdx, rsi, r8, r9, r10, r11, r12, r13, r14): register allocation algorithm



#Round Robin way to get the AVX 256-bit register
def get_avx_reg( avoid_reg = '' ):
    get_avx_reg.counter += 1
    res_reg = get_avx_reg.avx_reg_pool[ get_avx_reg.counter % len(get_avx_reg.avx_reg_pool) ]
    if( res_reg == avoid_reg ):
        get_avx_reg.counter += 1
        res_reg = get_avx_reg.avx_reg_pool[ get_avx_reg.counter % len(get_avx_reg.avx_reg_pool) ]
    return res_reg
        
get_avx_reg.counter = -1
get_avx_reg.avx_reg_pool = [ 'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5', 'ymm6', 'ymm7' ]


def write_updatec_assembly( myfile, nonzero_coeffs ):
    nnz = len( nonzero_coeffs )
    if contain_nontrivial( nonzero_coeffs ):
        write_line( myfile, 1, '"movq         %{0}, %%rax                      \\n\\t" // load address of alpha_list'.format(nnz+6) )

    for j, coeff in enumerate(nonzero_coeffs):
        if is_one(coeff) or is_negone(coeff):

            if is_one(coeff):
                update_avx = 'vaddpd'
                update_op = '+'
            elif is_negone(coeff):
                update_avx = 'vsubpd'
                update_op = '-'

            myfile.write( \
'''\
    "movq                   %{2}, %%{4}            \\n\\t" // load address of c
    "                                            \\n\\t"
    "vmovapd    0 * 32(%%{4}),  %%{5}           \\n\\t" // {5} = c{3}( 0:3, 0 )
    "{0}            %%ymm9,  %%{5},  %%{5}  \\n\\t" // {5} {1}= ymm9
    "vmovapd           %%{5},  0(%%{4})         \\n\\t" // c{3}( 0:3, 0 ) = {5}
    "vmovapd    1 * 32(%%{4}),  %%{6}           \\n\\t" // {6} = c{3}( 4:7, 0 )
    "{0}            %%ymm8,  %%{6},  %%{6}  \\n\\t" // {6} {1}= ymm8
    "vmovapd           %%{6},  32(%%{4})        \\n\\t" // c{3}( 4:7, 0 ) = {6}
    "addq              %%rdi,   %%{4}            \\n\\t"
    "vmovapd    0 * 32(%%{4}),  %%{7}           \\n\\t" // {7} = c{3}( 0:3, 1 )
    "{0}            %%ymm11, %%{7},  %%{7}  \\n\\t" // {7} {1}= ymm11
    "vmovapd           %%{7},  0(%%{4})         \\n\\t" // c{3}( 0:3, 1 ) = {7}
    "vmovapd    1 * 32(%%{4}),  %%{8}           \\n\\t" // {8} = c{3}( 4:7, 1 )
    "{0}            %%ymm10, %%{8},  %%{8}  \\n\\t" // {8} {1}= ymm10
    "vmovapd           %%{8},  32(%%{4})        \\n\\t" // c{3}( 4:7, 1 ) = {8}
    "addq              %%rdi,   %%{4}            \\n\\t"
    "vmovapd    0 * 32(%%{4}),  %%{9}           \\n\\t" // {9} = c{3}( 0:3, 2 )
    "{0}            %%ymm13, %%{9},  %%{9}  \\n\\t" // {9} {1}= ymm13
    "vmovapd           %%{9},  0(%%{4})         \\n\\t" // c{3}( 0:3, 2 ) = {9}
    "vmovapd    1 * 32(%%{4}),  %%{10}           \\n\\t" // {10} = c{3}( 4:7, 2 )
    "{0}            %%ymm12, %%{10},  %%{10}  \\n\\t" // {10} {1}= ymm12
    "vmovapd           %%{10},  32(%%{4})        \\n\\t" // c{3}( 4:7, 2 ) = {10}
    "addq              %%rdi,   %%{4}            \\n\\t"
    "vmovapd    0 * 32(%%{4}),  %%{11}           \\n\\t" // {11} = c{3}( 0:3, 3 )
    "{0}            %%ymm15, %%{11},  %%{11}  \\n\\t" // {11} {1}= ymm15
    "vmovapd           %%{11},  0(%%{4})         \\n\\t" // c{3}( 0:3, 3 ) = {11}
    "vmovapd    1 * 32(%%{4}),  %%{12}           \\n\\t" // {12} = c{3}( 4:7, 3 )
    "{0}            %%ymm14, %%{12},  %%{12}  \\n\\t" // {12} {1}= ymm14
    "vmovapd           %%{12}, 32(%%{4})         \\n\\t" // c{3}( 4:7, 3 ) = {12}
'''.format( update_avx, update_op, str(j+6), str(j), get_reg(), get_avx_reg(), get_avx_reg(), get_avx_reg(), get_avx_reg(), get_avx_reg(), get_avx_reg(), get_avx_reg(), get_avx_reg() ) )

            if contain_nontrivial( nonzero_coeffs ):
                write_line( myfile, 1, '"addq              $1 * 8,  %%rax            \\n\\t" // alpha_list += 8' )

        else:
            #print "coeff not 1 / -1!"
            alpha_avx_reg = get_avx_reg()
            myfile.write( \
'''\
    "                                            \\n\\t"
	"vbroadcastsd    (%%rax), %%{3}             \\n\\t" // load alpha_list[ i ] and duplicate
    "movq                   %{0}, %%{2}            \\n\\t" // load address of c
    "                                            \\n\\t"
    "vmovapd    0 * 32(%%{2}),  %%{4}           \\n\\t" // {4} = c{1}( 0:3, 0 )
	"vmulpd            %%{3},  %%ymm9,  %%{5}  \\n\\t" // scale by alpha, {5} = {3}( alpha ) * ymm9( c{1}( 0:3, 0 ) )
    "vaddpd            %%{4},  %%{5},  %%{4}  \\n\\t" // {4} += {5}
    "vmovapd           %%{4},  0(%%{2})         \\n\\t" // c{1}( 0:3, 0 ) = {4}
    "vmovapd    1 * 32(%%{2}),  %%{6}           \\n\\t" // {6} = c{1}( 4:7, 0 )
	"vmulpd            %%{3},  %%ymm8,  %%{7}  \\n\\t" // scale by alpha, {7} = {3}( alpha ) * ymm8( c{1}( 4:7, 0 ) )
    "vaddpd            %%{6},  %%{7},  %%{6}  \\n\\t" // {6} += {7}
    "vmovapd           %%{6},  32(%%{2})        \\n\\t" // c{1}( 4:7, 0 ) = {6}
    "addq              %%rdi,   %%{2}            \\n\\t"
    "vmovapd    0 * 32(%%{2}),  %%{8}           \\n\\t" // {8} = c{1}( 0:3, 1 )
	"vmulpd            %%{3},  %%ymm11,  %%{9}  \\n\\t" // scale by alpha, {5} = {3}( alpha ) * ymm11( c{1}( 0:3, 1 ) )
    "vaddpd            %%{8}, %%{9},  %%{8}  \\n\\t" // {8} += {7}
    "vmovapd           %%{8},  0(%%{2})         \\n\\t" // c{1}( 0:3, 1 ) = {8}
    "vmovapd    1 * 32(%%{2}),  %%{10}           \\n\\t" // {10} = c{1}( 4:7, 1 )
	"vmulpd            %%{3},  %%ymm10,  %%{11}  \\n\\t" // scale by alpha, {5} = {3}( alpha ) * ymm10( c{1}( 4:7, 1 ) )
    "vaddpd            %%{10}, %%{11},  %%{10}  \\n\\t" // {10} += {9}
    "vmovapd           %%{10},  32(%%{2})        \\n\\t" // c{1}( 4:7, 1 ) = {10}
    "addq              %%rdi,   %%{2}            \\n\\t"
    "vmovapd    0 * 32(%%{2}),  %%{12}           \\n\\t" // {12} = c{1}( 0:3, 2 )
	"vmulpd            %%{3},  %%ymm13,  %%{13}  \\n\\t" // scale by alpha, {5} = {3}( alpha ) * ymm13( c{1}( 0:3, 2 ) )
    "vaddpd            %%{12}, %%{13},  %%{12}  \\n\\t" // {12} += {11}
    "vmovapd           %%{12},  0(%%{2})         \\n\\t" // c{1}( 0:3, 2 ) = {12}
    "vmovapd    1 * 32(%%{2}),  %%{14}           \\n\\t" // {14} = c{1}( 4:7, 2 )
	"vmulpd            %%{3},  %%ymm12,  %%{15}  \\n\\t" // scale by alpha, {5} = {3}( alpha ) * ymm12( c{1}( 4:7, 2 ) )
    "vaddpd            %%{14}, %%{15},  %%{14}  \\n\\t" // {14} += {13}
    "vmovapd           %%{14},  32(%%{2})        \\n\\t" // c{1}( 4:7, 2 ) = {14}
    "addq              %%rdi,   %%{2}            \\n\\t"
    "vmovapd    0 * 32(%%{2}),  %%{16}           \\n\\t" // {16} = c{1}( 0:3, 3 )
	"vmulpd            %%{3},  %%ymm15,  %%{17}  \\n\\t" // scale by alpha, {5} = {3}( alpha ) * ymm15( c{1}( 0:3, 3 ) )
    "vaddpd            %%{16}, %%{17},  %%{16}  \\n\\t" // {16} += {15}
    "vmovapd           %%{16},  0(%%{2})         \\n\\t" // c{1}( 0:3, 3 ) = {16}
    "vmovapd    1 * 32(%%{2}),  %%{18}           \\n\\t" // {18} = c{1}( 4:7, 3 )
	"vmulpd            %%{3},  %%ymm14,  %%{19}  \\n\\t" // scale by alpha, {5} = {3}( alpha ) * ymm14( c{1}( 4:7, 3 ) )
    "vaddpd            %%{18}, %%{19},  %%{18}  \\n\\t" // {18} +={17}
    "vmovapd           %%{18}, 32(%%{2})         \\n\\t" // c{1}( 4:7, 3 ) = {18}
    "addq              $1 * 8,  %%rax            \\n\\t" // alpha_list += 8
    "                                            \\n\\t"
'''.format( str(j+6), str(j), get_reg(), alpha_avx_reg, get_avx_reg( alpha_avx_reg ), get_avx_reg( alpha_avx_reg ), get_avx_reg( alpha_avx_reg ), get_avx_reg( alpha_avx_reg ), get_avx_reg( alpha_avx_reg ), get_avx_reg( alpha_avx_reg ), get_avx_reg( alpha_avx_reg ), get_avx_reg( alpha_avx_reg ), get_avx_reg( alpha_avx_reg ), get_avx_reg( alpha_avx_reg ), get_avx_reg( alpha_avx_reg ), get_avx_reg( alpha_avx_reg ), get_avx_reg( alpha_avx_reg ), get_avx_reg( alpha_avx_reg ), get_avx_reg( alpha_avx_reg ), get_avx_reg( alpha_avx_reg ) ) )

def write_common_rankk_macro_assembly( myfile ):
    myfile.write( \
 '''\
#define STRINGIFY(...) #__VA_ARGS__

#define RANKK_UPDATE( NUM ) \\
    "vxorpd    %%ymm8,  %%ymm8,  %%ymm8          \\n\\t" /* set ymm8 to 0                   ( v )  */ \\
    "vxorpd    %%ymm9,  %%ymm9,  %%ymm9          \\n\\t" \\
    "vxorpd    %%ymm10, %%ymm10, %%ymm10         \\n\\t" \\
    "vxorpd    %%ymm11, %%ymm11, %%ymm11         \\n\\t" \\
	"vxorpd    %%ymm12, %%ymm12, %%ymm12         \\n\\t" \\
	"vxorpd    %%ymm13, %%ymm13, %%ymm13         \\n\\t" \\
	"vxorpd    %%ymm14, %%ymm14, %%ymm14         \\n\\t" \\
	"vxorpd    %%ymm15, %%ymm15, %%ymm15         \\n\\t" \\
	"                                            \\n\\t" \\
	"movq      %0, %%rsi                         \\n\\t" /*  i = k_iter;                     ( v ) */ \\
	"testq  %%rsi, %%rsi                         \\n\\t" /*  check i via logical AND.        ( v ) */ \\
	"je     .DCONSIDKLEFT"STRINGIFY(NUM)"                       \\n\\t" /*  if i == 0, jump to code that    ( v ) */ \\
	"                                            \\n\\t" /*  contains the k_left loop. */ \\
	"                                            \\n\\t" \\
	".DLOOPKITER"STRINGIFY(NUM)":                               \\n\\t" /*  MAIN LOOP */ \\
	"                                            \\n\\t" \\
	"addq         $4 * 4 * 8,  %%r15             \\n\\t" /*  b_next += 4*4 (unroll x nr)     ( v ) */ \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  iteration 0 */ \\
	"vmovapd   1 * 32(%%rax),  %%ymm1            \\n\\t" /*  preload a47 for iter 0 */ \\
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \\n\\t" /*  ymm6 ( c_tmp0 ) = ymm0 ( a03 ) * ymm2( b0 ) */ \\
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \\n\\t" /*  ymm4 ( b0x3_0 ) */ \\
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \\n\\t" /*  ymm7 ( c_tmp1 ) = ymm0 ( a03 ) * ymm3( b0x5 ) */ \\
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \\n\\t" /*  ymm5 ( b0x3_1 ) */ \\
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \\n\\t" /*  ymm15 ( c_03_0 ) += ymm6( c_tmp0 ) */ \\
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \\n\\t" /*  ymm13 ( c_03_1 ) += ymm7( c_tmp1 ) */ \\
	"                                            \\n\\t" \\
	"prefetcht0  16 * 32(%%rax)                  \\n\\t" /*  prefetch a03 for iter 1 */ \\
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \\n\\t" \\
	"vmovapd   1 * 32(%%rbx),  %%ymm2            \\n\\t" /*  preload b for iter 1 */ \\
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \\n\\t" \\
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \\n\\t" \\
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \\n\\t" \\
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \\n\\t" \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \\n\\t" \\
	"vmovapd   2 * 32(%%rax),  %%ymm0            \\n\\t" /*  preload a03 for iter 1 */ \\
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \\n\\t" \\
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \\n\\t" \\
	"prefetcht0   0 * 32(%%r15)                  \\n\\t" /*  prefetch b_next[0*4] */ \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \\n\\t" \\
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \\n\\t" \\
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  iteration 1 */ \\
	"vmovapd   3 * 32(%%rax),  %%ymm1            \\n\\t" /*  preload a47 for iter 1 */ \\
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \\n\\t" \\
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \\n\\t" \\
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \\n\\t" \\
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \\n\\t" \\
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \\n\\t" \\
	"                                            \\n\\t" \\
	"prefetcht0  18 * 32(%%rax)                  \\n\\t" /*  prefetch a for iter 9  ( ? ) */ \\
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \\n\\t" \\
	"vmovapd   2 * 32(%%rbx),  %%ymm2            \\n\\t" /*  preload b for iter 2 */ \\
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \\n\\t" \\
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \\n\\t" \\
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \\n\\t" \\
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \\n\\t" \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \\n\\t" \\
	"vmovapd   4 * 32(%%rax),  %%ymm0            \\n\\t" /*  preload a03 for iter 2 */ \\
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \\n\\t" \\
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \\n\\t" \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \\n\\t" \\
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \\n\\t" \\
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  iteration 2 */ \\
	"vmovapd   5 * 32(%%rax),  %%ymm1            \\n\\t" /*  preload a47 for iter 2 */ \\
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \\n\\t" \\
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \\n\\t" \\
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \\n\\t" \\
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \\n\\t" \\
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \\n\\t" \\
	"                                            \\n\\t" \\
	"prefetcht0  20 * 32(%%rax)                  \\n\\t" /*  prefetch a for iter 10 ( ? ) */ \\
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \\n\\t" \\
	"vmovapd   3 * 32(%%rbx),  %%ymm2            \\n\\t" /*  preload b for iter 3 */ \\
	"addq         $4 * 4 * 8,  %%rbx             \\n\\t" /*  b += 4*4 (unroll x nr) */ \\
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \\n\\t" \\
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \\n\\t" \\
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \\n\\t" \\
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \\n\\t" \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \\n\\t" \\
	"vmovapd   6 * 32(%%rax),  %%ymm0            \\n\\t" /*  preload a03 for iter 3 */ \\
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \\n\\t" \\
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \\n\\t" \\
	"prefetcht0   2 * 32(%%r15)                  \\n\\t" /*  prefetch b_next[2*4] */ \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \\n\\t" \\
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \\n\\t" \\
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  iteration 3 */ \\
	"vmovapd   7 * 32(%%rax),  %%ymm1            \\n\\t" /*  preload a47 for iter 3 */ \\
	"addq         $4 * 8 * 8,  %%rax             \\n\\t" /*  a += 4*8 (unroll x mr) */ \\
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \\n\\t" \\
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \\n\\t" \\
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \\n\\t" \\
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \\n\\t" \\
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \\n\\t" \\
	"                                            \\n\\t" \\
	"prefetcht0  14 * 32(%%rax)                  \\n\\t" /*  prefetch a for iter 11 ( ? ) */ \\
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \\n\\t" \\
	"vmovapd   0 * 32(%%rbx),  %%ymm2            \\n\\t" /*  preload b for iter 4 */ \\
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \\n\\t" \\
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \\n\\t" \\
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \\n\\t" \\
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \\n\\t" \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \\n\\t" \\
	"vmovapd   0 * 32(%%rax),  %%ymm0            \\n\\t" /*  preload a03 for iter 4 */ \\
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \\n\\t" \\
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \\n\\t" \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \\n\\t" \\
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \\n\\t" \\
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \\n\\t" \\
	"                                            \\n\\t" \\
	"decq   %%rsi                                \\n\\t" /*  i -= 1; */ \\
	"jne    .DLOOPKITER"STRINGIFY(NUM)"                         \\n\\t" /*  iterate again if i != 0. */ \\
	"                                            \\n\\t" \\
	".DCONSIDKLEFT"STRINGIFY(NUM)":                             \\n\\t" \\
	"                                            \\n\\t" \\
	"movq      %1, %%rsi                         \\n\\t" /*  i = k_left; */ \\
	"testq  %%rsi, %%rsi                         \\n\\t" /*  check i via logical AND. */ \\
	"je     .DPOSTACCUM"STRINGIFY(NUM)"                        \\n\\t" /*  if i == 0, we're done; jump to end. */ \\
	"                                            \\n\\t" /*  else, we prepare to enter k_left loop. */ \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" \\
	".DLOOPKLEFT"STRINGIFY(NUM)":                               \\n\\t" /*  EDGE LOOP */ \\
	"                                            \\n\\t" \\
	"vmovapd   1 * 32(%%rax),  %%ymm1            \\n\\t" /*  preload a47 */ \\
	"addq         $8 * 1 * 8,  %%rax             \\n\\t" /*  a += 8 (1 x mr) */ \\
	"vmulpd           %%ymm0,  %%ymm2, %%ymm6    \\n\\t" \\
	"vperm2f128 $0x3, %%ymm2,  %%ymm2, %%ymm4    \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm3, %%ymm7    \\n\\t" \\
	"vperm2f128 $0x3, %%ymm3,  %%ymm3, %%ymm5    \\n\\t" \\
	"vaddpd           %%ymm15, %%ymm6, %%ymm15   \\n\\t" \\
	"vaddpd           %%ymm13, %%ymm7, %%ymm13   \\n\\t" \\
	"                                            \\n\\t" \\
	"prefetcht0  14 * 32(%%rax)                  \\n\\t" /*  prefetch a03 for iter 7 later ( ? ) */ \\
	"vmulpd           %%ymm1,  %%ymm2, %%ymm6    \\n\\t" \\
	"vmovapd   1 * 32(%%rbx),  %%ymm2            \\n\\t" \\
	"addq         $4 * 1 * 8,  %%rbx             \\n\\t" /*  b += 4 (1 x nr) */ \\
	"vmulpd           %%ymm1,  %%ymm3, %%ymm7    \\n\\t" \\
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \\n\\t" \\
	"vaddpd           %%ymm14, %%ymm6, %%ymm14   \\n\\t" \\
	"vaddpd           %%ymm12, %%ymm7, %%ymm12   \\n\\t" \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm4, %%ymm6    \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm5, %%ymm7    \\n\\t" \\
	"vmovapd   0 * 32(%%rax),  %%ymm0            \\n\\t" \\
	"vaddpd           %%ymm11, %%ymm6, %%ymm11   \\n\\t" \\
	"vaddpd           %%ymm9,  %%ymm7, %%ymm9    \\n\\t" \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm4, %%ymm6    \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm5, %%ymm7    \\n\\t" \\
	"vaddpd           %%ymm10, %%ymm6, %%ymm10   \\n\\t" \\
	"vaddpd           %%ymm8,  %%ymm7, %%ymm8    \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" \\
	"decq   %%rsi                                \\n\\t" /*  i -= 1; */ \\
	"jne    .DLOOPKLEFT"STRINGIFY(NUM)"                         \\n\\t" /*  iterate again if i != 0. */ \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" \\
	".DPOSTACCUM"STRINGIFY(NUM)":                               \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  ymm15:  ymm13:  ymm11:  ymm9: */ \\
	"                                            \\n\\t" /*  ( ab00  ( ab01  ( ab02  ( ab03 */ \\
	"                                            \\n\\t" /*    ab11    ab10    ab13    ab12 */ \\
	"                                            \\n\\t" /*    ab22    ab23    ab20    ab21 */ \\
	"                                            \\n\\t" /*    ab33 )  ab32 )  ab31 )  ab30 ) */ \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  ymm14:  ymm12:  ymm10:  ymm8: */ \\
	"                                            \\n\\t" /*  ( ab40  ( ab41  ( ab42  ( ab43 */ \\
	"                                            \\n\\t" /*    ab51    ab50    ab53    ab52 */ \\
	"                                            \\n\\t" /*    ab62    ab63    ab60    ab61 */ \\
	"                                            \\n\\t" /*    ab73 )  ab72 )  ab71 )  ab70 ) */ \\
	"                                            \\n\\t" \\
	"vmovapd          %%ymm15, %%ymm7            \\n\\t" \\
	"vshufpd    $0xa, %%ymm15, %%ymm13, %%ymm15  \\n\\t" \\
	"vshufpd    $0xa, %%ymm13, %%ymm7,  %%ymm13  \\n\\t" \\
	"                                            \\n\\t" \\
	"vmovapd          %%ymm11, %%ymm7            \\n\\t" \\
	"vshufpd    $0xa, %%ymm11, %%ymm9,  %%ymm11  \\n\\t" \\
	"vshufpd    $0xa, %%ymm9,  %%ymm7,  %%ymm9   \\n\\t" \\
	"                                            \\n\\t" \\
	"vmovapd          %%ymm14, %%ymm7            \\n\\t" \\
	"vshufpd    $0xa, %%ymm14, %%ymm12, %%ymm14  \\n\\t" \\
	"vshufpd    $0xa, %%ymm12, %%ymm7,  %%ymm12  \\n\\t" \\
	"                                            \\n\\t" \\
	"vmovapd          %%ymm10, %%ymm7            \\n\\t" \\
	"vshufpd    $0xa, %%ymm10, %%ymm8,  %%ymm10  \\n\\t" \\
	"vshufpd    $0xa, %%ymm8,  %%ymm7,  %%ymm8   \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  ymm15:  ymm13:  ymm11:  ymm9: */ \\
	"                                            \\n\\t" /*  ( ab01  ( ab00  ( ab03  ( ab02 */ \\
	"                                            \\n\\t" /*    ab11    ab10    ab13    ab12 */ \\
	"                                            \\n\\t" /*    ab23    ab22    ab21    ab20 */ \\
	"                                            \\n\\t" /*    ab33 )  ab32 )  ab31 )  ab30 ) */ \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  ymm14:  ymm12:  ymm10:  ymm8: */ \\
	"                                            \\n\\t" /*  ( ab41  ( ab40  ( ab43  ( ab42 */ \\
	"                                            \\n\\t" /*    ab51    ab50    ab53    ab52 */ \\
	"                                            \\n\\t" /*    ab63    ab62    ab61    ab60 */ \\
	"                                            \\n\\t" /*    ab73 )  ab72 )  ab71 )  ab70 ) */ \\
	"                                            \\n\\t" \\
	"vmovapd           %%ymm15, %%ymm7           \\n\\t" \\
	"vperm2f128 $0x30, %%ymm15, %%ymm11, %%ymm15 \\n\\t" \\
	"vperm2f128 $0x12, %%ymm7,  %%ymm11, %%ymm11 \\n\\t" \\
	"                                            \\n\\t" \\
	"vmovapd           %%ymm13, %%ymm7           \\n\\t" \\
	"vperm2f128 $0x30, %%ymm13, %%ymm9,  %%ymm13 \\n\\t" \\
	"vperm2f128 $0x12, %%ymm7,  %%ymm9,  %%ymm9  \\n\\t" \\
	"                                            \\n\\t" \\
	"vmovapd           %%ymm14, %%ymm7           \\n\\t" \\
	"vperm2f128 $0x30, %%ymm14, %%ymm10, %%ymm14 \\n\\t" \\
	"vperm2f128 $0x12, %%ymm7,  %%ymm10, %%ymm10 \\n\\t" \\
	"                                            \\n\\t" \\
	"vmovapd           %%ymm12, %%ymm7           \\n\\t" \\
	"vperm2f128 $0x30, %%ymm12, %%ymm8,  %%ymm12 \\n\\t" \\
	"vperm2f128 $0x12, %%ymm7,  %%ymm8,  %%ymm8  \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  ymm9:   ymm11:  ymm13:  ymm15: */ \\
	"                                            \\n\\t" /*  ( ab00  ( ab01  ( ab02  ( ab03 */ \\
	"                                            \\n\\t" /*    ab10    ab11    ab12    ab13 */ \\
	"                                            \\n\\t" /*    ab20    ab21    ab22    ab23 */ \\
	"                                            \\n\\t" /*    ab30 )  ab31 )  ab32 )  ab33 ) */ \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  ymm8:   ymm10:  ymm12:  ymm14: */ \\
	"                                            \\n\\t" /*  ( ab40  ( ab41  ( ab42  ( ab43 */ \\
	"                                            \\n\\t" /*    ab50    ab51    ab52    ab53 */ \\
	"                                            \\n\\t" /*    ab60    ab61    ab62    ab63 */ \\
	"                                            \\n\\t" /*    ab70 )  ab71 )  ab72 )  ab73 ) */
''' )

def macro_rankk_xor0_assembly( myfile ):
    myfile.write( \
'''\
#define RANKK_XOR0 \\
    "vxorpd    %%ymm8,  %%ymm8,  %%ymm8          \\n\\t" /* set ymm8 to 0                   ( v )  */ \\
    "vxorpd    %%ymm9,  %%ymm9,  %%ymm9          \\n\\t" \\
    "vxorpd    %%ymm10, %%ymm10, %%ymm10         \\n\\t" \\
    "vxorpd    %%ymm11, %%ymm11, %%ymm11         \\n\\t" \\
	"vxorpd    %%ymm12, %%ymm12, %%ymm12         \\n\\t" \\
	"vxorpd    %%ymm13, %%ymm13, %%ymm13         \\n\\t" \\
	"vxorpd    %%ymm14, %%ymm14, %%ymm14         \\n\\t" \\
	"vxorpd    %%ymm15, %%ymm15, %%ymm15         \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" \\
	"movq      %0, %%rsi                         \\n\\t" /*  i = k_iter;                     ( v ) */ \\
	"testq  %%rsi, %%rsi                         \\n\\t" /*  check i via logical AND.        ( v ) */ \\
''' )

def macro_rankk_loopkiter_assembly( myfile ):
    myfile.write( \
'''\
#define RANKK_LOOPKITER \\
	"addq         $4 * 4 * 8,  %%r15             \\n\\t" /*  b_next += 4*4 (unroll x nr)     ( v ) */ \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  iteration 0 */ \\
	"vmovapd   1 * 32(%%rax),  %%ymm1            \\n\\t" /*  preload a47 for iter 0 */ \\
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \\n\\t" /*  ymm6 ( c_tmp0 ) = ymm0 ( a03 ) * ymm2( b0 ) */ \\
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \\n\\t" /*  ymm4 ( b0x3_0 ) */ \\
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \\n\\t" /*  ymm7 ( c_tmp1 ) = ymm0 ( a03 ) * ymm3( b0x5 ) */ \\
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \\n\\t" /*  ymm5 ( b0x3_1 ) */ \\
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \\n\\t" /*  ymm15 ( c_03_0 ) += ymm6( c_tmp0 ) */ \\
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \\n\\t" /*  ymm13 ( c_03_1 ) += ymm7( c_tmp1 ) */ \\
	"                                            \\n\\t" \\
	"prefetcht0  16 * 32(%%rax)                  \\n\\t" /*  prefetch a03 for iter 1 */ \\
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \\n\\t" \\
	"vmovapd   1 * 32(%%rbx),  %%ymm2            \\n\\t" /*  preload b for iter 1 */ \\
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \\n\\t" \\
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \\n\\t" \\
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \\n\\t" \\
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \\n\\t" \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \\n\\t" \\
	"vmovapd   2 * 32(%%rax),  %%ymm0            \\n\\t" /*  preload a03 for iter 1 */ \\
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \\n\\t" \\
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \\n\\t" \\
	"prefetcht0   0 * 32(%%r15)                  \\n\\t" /*  prefetch b_next[0*4] */ \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \\n\\t" \\
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \\n\\t" \\
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  iteration 1 */ \\
	"vmovapd   3 * 32(%%rax),  %%ymm1            \\n\\t" /*  preload a47 for iter 1 */ \\
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \\n\\t" \\
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \\n\\t" \\
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \\n\\t" \\
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \\n\\t" \\
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \\n\\t" \\
	"                                            \\n\\t" \\
	"prefetcht0  18 * 32(%%rax)                  \\n\\t" /*  prefetch a for iter 9  ( ? ) */ \\
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \\n\\t" \\
	"vmovapd   2 * 32(%%rbx),  %%ymm2            \\n\\t" /*  preload b for iter 2 */ \\
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \\n\\t" \\
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \\n\\t" \\
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \\n\\t" \\
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \\n\\t" \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \\n\\t" \\
	"vmovapd   4 * 32(%%rax),  %%ymm0            \\n\\t" /*  preload a03 for iter 2 */ \\
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \\n\\t" \\
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \\n\\t" \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \\n\\t" \\
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \\n\\t" \\
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  iteration 2 */ \\
	"vmovapd   5 * 32(%%rax),  %%ymm1            \\n\\t" /*  preload a47 for iter 2 */ \\
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \\n\\t" \\
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \\n\\t" \\
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \\n\\t" \\
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \\n\\t" \\
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \\n\\t" \\
	"                                            \\n\\t" \\
	"prefetcht0  20 * 32(%%rax)                  \\n\\t" /*  prefetch a for iter 10 ( ? ) */ \\
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \\n\\t" \\
	"vmovapd   3 * 32(%%rbx),  %%ymm2            \\n\\t" /*  preload b for iter 3 */ \\
	"addq         $4 * 4 * 8,  %%rbx             \\n\\t" /*  b += 4*4 (unroll x nr) */ \\
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \\n\\t" \\
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \\n\\t" \\
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \\n\\t" \\
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \\n\\t" \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \\n\\t" \\
	"vmovapd   6 * 32(%%rax),  %%ymm0            \\n\\t" /*  preload a03 for iter 3 */ \\
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \\n\\t" \\
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \\n\\t" \\
	"prefetcht0   2 * 32(%%r15)                  \\n\\t" /*  prefetch b_next[2*4] */ \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \\n\\t" \\
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \\n\\t" \\
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  iteration 3 */ \\
	"vmovapd   7 * 32(%%rax),  %%ymm1            \\n\\t" /*  preload a47 for iter 3 */ \\
	"addq         $4 * 8 * 8,  %%rax             \\n\\t" /*  a += 4*8 (unroll x mr) */ \\
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \\n\\t" \\
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \\n\\t" \\
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \\n\\t" \\
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \\n\\t" \\
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \\n\\t" \\
	"                                            \\n\\t" \\
	"prefetcht0  14 * 32(%%rax)                  \\n\\t" /*  prefetch a for iter 11 ( ? ) */ \\
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \\n\\t" \\
	"vmovapd   0 * 32(%%rbx),  %%ymm2            \\n\\t" /*  preload b for iter 4 */ \\
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \\n\\t" \\
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \\n\\t" \\
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \\n\\t" \\
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \\n\\t" \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \\n\\t" \\
	"vmovapd   0 * 32(%%rax),  %%ymm0            \\n\\t" /*  preload a03 for iter 4 */ \\
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \\n\\t" \\
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \\n\\t" \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \\n\\t" \\
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \\n\\t" \\
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" \\
	"decq   %%rsi                                \\n\\t" /*  i -= 1; */ \\
''' )

def macro_rankk_loopkleft_assembly( myfile ):
    myfile.write( \
'''\
#define RANKK_LOOPKLEFT \\
	"vmovapd   1 * 32(%%rax),  %%ymm1            \\n\\t" /*  preload a47 */ \\
	"addq         $8 * 1 * 8,  %%rax             \\n\\t" /*  a += 8 (1 x mr) */ \\
	"vmulpd           %%ymm0,  %%ymm2, %%ymm6    \\n\\t" \\
	"vperm2f128 $0x3, %%ymm2,  %%ymm2, %%ymm4    \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm3, %%ymm7    \\n\\t" \\
	"vperm2f128 $0x3, %%ymm3,  %%ymm3, %%ymm5    \\n\\t" \\
	"vaddpd           %%ymm15, %%ymm6, %%ymm15   \\n\\t" \\
	"vaddpd           %%ymm13, %%ymm7, %%ymm13   \\n\\t" \\
	"                                            \\n\\t" \\
	"prefetcht0  14 * 32(%%rax)                  \\n\\t" /*  prefetch a03 for iter 7 later ( ? ) */ \\
	"vmulpd           %%ymm1,  %%ymm2, %%ymm6    \\n\\t" \\
	"vmovapd   1 * 32(%%rbx),  %%ymm2            \\n\\t" \\
	"addq         $4 * 1 * 8,  %%rbx             \\n\\t" /*  b += 4 (1 x nr) */ \\
	"vmulpd           %%ymm1,  %%ymm3, %%ymm7    \\n\\t" \\
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \\n\\t" \\
	"vaddpd           %%ymm14, %%ymm6, %%ymm14   \\n\\t" \\
	"vaddpd           %%ymm12, %%ymm7, %%ymm12   \\n\\t" \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm4, %%ymm6    \\n\\t" \\
	"vmulpd           %%ymm0,  %%ymm5, %%ymm7    \\n\\t" \\
	"vmovapd   0 * 32(%%rax),  %%ymm0            \\n\\t" \\
	"vaddpd           %%ymm11, %%ymm6, %%ymm11   \\n\\t" \\
	"vaddpd           %%ymm9,  %%ymm7, %%ymm9    \\n\\t" \\
	"                                            \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm4, %%ymm6    \\n\\t" \\
	"vmulpd           %%ymm1,  %%ymm5, %%ymm7    \\n\\t" \\
	"vaddpd           %%ymm10, %%ymm6, %%ymm10   \\n\\t" \\
	"vaddpd           %%ymm8,  %%ymm7, %%ymm8    \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" \\
	"decq   %%rsi                                \\n\\t" /*  i -= 1; */ \\
''' )

def macro_rankk_postaccum_assembly( myfile ):
    myfile.write( \
'''\
#define RANKK_POSTACCUM \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  ymm15:  ymm13:  ymm11:  ymm9: */ \\
	"                                            \\n\\t" /*  ( ab00  ( ab01  ( ab02  ( ab03 */ \\
	"                                            \\n\\t" /*    ab11    ab10    ab13    ab12 */ \\
	"                                            \\n\\t" /*    ab22    ab23    ab20    ab21 */ \\
	"                                            \\n\\t" /*    ab33 )  ab32 )  ab31 )  ab30 ) */ \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  ymm14:  ymm12:  ymm10:  ymm8: */ \\
	"                                            \\n\\t" /*  ( ab40  ( ab41  ( ab42  ( ab43 */ \\
	"                                            \\n\\t" /*    ab51    ab50    ab53    ab52 */ \\
	"                                            \\n\\t" /*    ab62    ab63    ab60    ab61 */ \\
	"                                            \\n\\t" /*    ab73 )  ab72 )  ab71 )  ab70 ) */ \\
	"                                            \\n\\t" \\
	"vmovapd          %%ymm15, %%ymm7            \\n\\t" \\
	"vshufpd    $0xa, %%ymm15, %%ymm13, %%ymm15  \\n\\t" \\
	"vshufpd    $0xa, %%ymm13, %%ymm7,  %%ymm13  \\n\\t" \\
	"                                            \\n\\t" \\
	"vmovapd          %%ymm11, %%ymm7            \\n\\t" \\
	"vshufpd    $0xa, %%ymm11, %%ymm9,  %%ymm11  \\n\\t" \\
	"vshufpd    $0xa, %%ymm9,  %%ymm7,  %%ymm9   \\n\\t" \\
	"                                            \\n\\t" \\
	"vmovapd          %%ymm14, %%ymm7            \\n\\t" \\
	"vshufpd    $0xa, %%ymm14, %%ymm12, %%ymm14  \\n\\t" \\
	"vshufpd    $0xa, %%ymm12, %%ymm7,  %%ymm12  \\n\\t" \\
	"                                            \\n\\t" \\
	"vmovapd          %%ymm10, %%ymm7            \\n\\t" \\
	"vshufpd    $0xa, %%ymm10, %%ymm8,  %%ymm10  \\n\\t" \\
	"vshufpd    $0xa, %%ymm8,  %%ymm7,  %%ymm8   \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  ymm15:  ymm13:  ymm11:  ymm9: */ \\
	"                                            \\n\\t" /*  ( ab01  ( ab00  ( ab03  ( ab02 */ \\
	"                                            \\n\\t" /*    ab11    ab10    ab13    ab12 */ \\
	"                                            \\n\\t" /*    ab23    ab22    ab21    ab20 */ \\
	"                                            \\n\\t" /*    ab33 )  ab32 )  ab31 )  ab30 ) */ \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  ymm14:  ymm12:  ymm10:  ymm8: */ \\
	"                                            \\n\\t" /*  ( ab41  ( ab40  ( ab43  ( ab42 */ \\
	"                                            \\n\\t" /*    ab51    ab50    ab53    ab52 */ \\
	"                                            \\n\\t" /*    ab63    ab62    ab61    ab60 */ \\
	"                                            \\n\\t" /*    ab73 )  ab72 )  ab71 )  ab70 ) */ \\
	"                                            \\n\\t" \\
	"vmovapd           %%ymm15, %%ymm7           \\n\\t" \\
	"vperm2f128 $0x30, %%ymm15, %%ymm11, %%ymm15 \\n\\t" \\
	"vperm2f128 $0x12, %%ymm7,  %%ymm11, %%ymm11 \\n\\t" \\
	"                                            \\n\\t" \\
	"vmovapd           %%ymm13, %%ymm7           \\n\\t" \\
	"vperm2f128 $0x30, %%ymm13, %%ymm9,  %%ymm13 \\n\\t" \\
	"vperm2f128 $0x12, %%ymm7,  %%ymm9,  %%ymm9  \\n\\t" \\
	"                                            \\n\\t" \\
	"vmovapd           %%ymm14, %%ymm7           \\n\\t" \\
	"vperm2f128 $0x30, %%ymm14, %%ymm10, %%ymm14 \\n\\t" \\
	"vperm2f128 $0x12, %%ymm7,  %%ymm10, %%ymm10 \\n\\t" \\
	"                                            \\n\\t" \\
	"vmovapd           %%ymm12, %%ymm7           \\n\\t" \\
	"vperm2f128 $0x30, %%ymm12, %%ymm8,  %%ymm12 \\n\\t" \\
	"vperm2f128 $0x12, %%ymm7,  %%ymm8,  %%ymm8  \\n\\t" \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  ymm9:   ymm11:  ymm13:  ymm15: */ \\
	"                                            \\n\\t" /*  ( ab00  ( ab01  ( ab02  ( ab03 */ \\
	"                                            \\n\\t" /*    ab10    ab11    ab12    ab13 */ \\
	"                                            \\n\\t" /*    ab20    ab21    ab22    ab23 */ \\
	"                                            \\n\\t" /*    ab30 )  ab31 )  ab32 )  ab33 ) */ \\
	"                                            \\n\\t" \\
	"                                            \\n\\t" /*  ymm8:   ymm10:  ymm12:  ymm14: */ \\
	"                                            \\n\\t" /*  ( ab40  ( ab41  ( ab42  ( ab43 */ \\
	"                                            \\n\\t" /*    ab50    ab51    ab52    ab53 */ \\
	"                                            \\n\\t" /*    ab60    ab61    ab62    ab63 */ \\
	"                                            \\n\\t" /*    ab70 )  ab71 )  ab72 )  ab73 ) */
''' )

def write_common_simple_rankk_assembly( myfile, index ):
    myfile.write( \
 '''\
    RANKK_XOR0
	"je     .DCONSIDKLEFT{0}                       \\n\\t" // if i == 0, jump to code that    ( v )
	"                                            \\n\\t" // contains the k_left loop.
	".DLOOPKITER{0}:                               \\n\\t" // MAIN LOOP
	"                                            \\n\\t"
    RANKK_LOOPKITER
	"jne    .DLOOPKITER{0}                         \\n\\t" // iterate again if i != 0.
	".DCONSIDKLEFT{0}:                             \\n\\t"
	"movq      %1, %%rsi                         \\n\\t" // i = k_left;
	"testq  %%rsi, %%rsi                         \\n\\t" // check i via logical AND.
	"je     .DPOSTACCUM{0}                         \\n\\t" // if i == 0, we're done; jump to end.
	"                                            \\n\\t" // else, we prepare to enter k_left loop.
	".DLOOPKLEFT{0}:                               \\n\\t" // EDGE LOOP
	"                                            \\n\\t"
    RANKK_LOOPKLEFT
	"jne    .DLOOPKLEFT{0}                         \\n\\t" // iterate again if i != 0.
	"                                            \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t"
	".DPOSTACCUM{0}:                               \\n\\t"
	"                                            \\n\\t"
    RANKK_POSTACCUM
'''.format( index ) )

def write_common_rankk_assembly( myfile, index ):
    #write_line( myfile, 1, )
    myfile.write( \
 '''\
    "                                            \\n\\t"
    "vxorpd    %%ymm8,  %%ymm8,  %%ymm8          \\n\\t" // set ymm8 to 0                   ( v )
    "vxorpd    %%ymm9,  %%ymm9,  %%ymm9          \\n\\t"
    "vxorpd    %%ymm10, %%ymm10, %%ymm10         \\n\\t"
    "vxorpd    %%ymm11, %%ymm11, %%ymm11         \\n\\t"
	"vxorpd    %%ymm12, %%ymm12, %%ymm12         \\n\\t"
	"vxorpd    %%ymm13, %%ymm13, %%ymm13         \\n\\t"
	"vxorpd    %%ymm14, %%ymm14, %%ymm14         \\n\\t"
	"vxorpd    %%ymm15, %%ymm15, %%ymm15         \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t"
	"movq      %0, %%rsi                         \\n\\t" // i = k_iter;                     ( v )
	"testq  %%rsi, %%rsi                         \\n\\t" // check i via logical AND.        ( v )
	"je     .DCONSIDKLEFT{0}                       \\n\\t" // if i == 0, jump to code that    ( v )
	"                                            \\n\\t" // contains the k_left loop.
	"                                            \\n\\t"
	"                                            \\n\\t"
	".DLOOPKITER{0}:                               \\n\\t" // MAIN LOOP
	"                                            \\n\\t"
	"addq         $4 * 4 * 8,  %%r15             \\n\\t" // b_next += 4*4 (unroll x nr)     ( v )
	"                                            \\n\\t"
	"                                            \\n\\t" // iteration 0
	"vmovapd   1 * 32(%%rax),  %%ymm1            \\n\\t" // preload a47 for iter 0
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \\n\\t" // ymm6 ( c_tmp0 ) = ymm0 ( a03 ) * ymm2( b0 )
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \\n\\t" // ymm4 ( b0x3_0 )
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \\n\\t" // ymm7 ( c_tmp1 ) = ymm0 ( a03 ) * ymm3( b0x5 )
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \\n\\t" // ymm5 ( b0x3_1 )
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \\n\\t" // ymm15 ( c_03_0 ) += ymm6( c_tmp0 )
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \\n\\t" // ymm13 ( c_03_1 ) += ymm7( c_tmp1 )
	"                                            \\n\\t"
	"prefetcht0  16 * 32(%%rax)                  \\n\\t" // prefetch a03 for iter 1
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \\n\\t"
	"vmovapd   1 * 32(%%rbx),  %%ymm2            \\n\\t" // preload b for iter 1
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \\n\\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \\n\\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \\n\\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \\n\\t"
	"                                            \\n\\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \\n\\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \\n\\t"
	"vmovapd   2 * 32(%%rax),  %%ymm0            \\n\\t" // preload a03 for iter 1
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \\n\\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \\n\\t"
	"prefetcht0   0 * 32(%%r15)                  \\n\\t" // prefetch b_next[0*4]
	"                                            \\n\\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \\n\\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \\n\\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \\n\\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t" // iteration 1
	"vmovapd   3 * 32(%%rax),  %%ymm1            \\n\\t" // preload a47 for iter 1
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \\n\\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \\n\\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \\n\\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \\n\\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \\n\\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \\n\\t"
	"                                            \\n\\t"
	"prefetcht0  18 * 32(%%rax)                  \\n\\t" // prefetch a for iter 9  ( ? )
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \\n\\t"
	"vmovapd   2 * 32(%%rbx),  %%ymm2            \\n\\t" // preload b for iter 2
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \\n\\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \\n\\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \\n\\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \\n\\t"
	"                                            \\n\\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \\n\\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \\n\\t"
	"vmovapd   4 * 32(%%rax),  %%ymm0            \\n\\t" // preload a03 for iter 2
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \\n\\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \\n\\t"
	"                                            \\n\\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \\n\\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \\n\\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \\n\\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t" // iteration 2
	"vmovapd   5 * 32(%%rax),  %%ymm1            \\n\\t" // preload a47 for iter 2
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \\n\\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \\n\\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \\n\\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \\n\\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \\n\\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \\n\\t"
	"                                            \\n\\t"
	"prefetcht0  20 * 32(%%rax)                  \\n\\t" // prefetch a for iter 10 ( ? )
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \\n\\t"
	"vmovapd   3 * 32(%%rbx),  %%ymm2            \\n\\t" // preload b for iter 3
	"addq         $4 * 4 * 8,  %%rbx             \\n\\t" // b += 4*4 (unroll x nr)
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \\n\\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \\n\\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \\n\\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \\n\\t"
	"                                            \\n\\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \\n\\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \\n\\t"
	"vmovapd   6 * 32(%%rax),  %%ymm0            \\n\\t" // preload a03 for iter 3
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \\n\\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \\n\\t"
	"prefetcht0   2 * 32(%%r15)                  \\n\\t" // prefetch b_next[2*4]
	"                                            \\n\\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \\n\\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \\n\\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \\n\\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t" // iteration 3
	"vmovapd   7 * 32(%%rax),  %%ymm1            \\n\\t" // preload a47 for iter 3
	"addq         $4 * 8 * 8,  %%rax             \\n\\t" // a += 4*8 (unroll x mr)
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \\n\\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \\n\\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \\n\\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \\n\\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \\n\\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \\n\\t"
	"                                            \\n\\t"
	"prefetcht0  14 * 32(%%rax)                  \\n\\t" // prefetch a for iter 11 ( ? )
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \\n\\t"
	"vmovapd   0 * 32(%%rbx),  %%ymm2            \\n\\t" // preload b for iter 4
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \\n\\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \\n\\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \\n\\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \\n\\t"
	"                                            \\n\\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \\n\\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \\n\\t"
	"vmovapd   0 * 32(%%rax),  %%ymm0            \\n\\t" // preload a03 for iter 4
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \\n\\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \\n\\t"
	"                                            \\n\\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \\n\\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \\n\\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \\n\\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t"
	"decq   %%rsi                                \\n\\t" // i -= 1;
	"jne    .DLOOPKITER{0}                         \\n\\t" // iterate again if i != 0.
	"                                            \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t"
	".DCONSIDKLEFT{0}:                             \\n\\t"
	"                                            \\n\\t"
	"movq      %1, %%rsi                         \\n\\t" // i = k_left;
	"testq  %%rsi, %%rsi                         \\n\\t" // check i via logical AND.
	"je     .DPOSTACCUM{0}                         \\n\\t" // if i == 0, we're done; jump to end.
	"                                            \\n\\t" // else, we prepare to enter k_left loop.
	"                                            \\n\\t"
	"                                            \\n\\t"
	".DLOOPKLEFT{0}:                               \\n\\t" // EDGE LOOP
	"                                            \\n\\t"
	"vmovapd   1 * 32(%%rax),  %%ymm1            \\n\\t" // preload a47
	"addq         $8 * 1 * 8,  %%rax             \\n\\t" // a += 8 (1 x mr)
	"vmulpd           %%ymm0,  %%ymm2, %%ymm6    \\n\\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2, %%ymm4    \\n\\t"
	"vmulpd           %%ymm0,  %%ymm3, %%ymm7    \\n\\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3, %%ymm5    \\n\\t"
	"vaddpd           %%ymm15, %%ymm6, %%ymm15   \\n\\t"
	"vaddpd           %%ymm13, %%ymm7, %%ymm13   \\n\\t"
	"                                            \\n\\t"
	"prefetcht0  14 * 32(%%rax)                  \\n\\t" // prefetch a03 for iter 7 later ( ? )
	"vmulpd           %%ymm1,  %%ymm2, %%ymm6    \\n\\t"
	"vmovapd   1 * 32(%%rbx),  %%ymm2            \\n\\t"
	"addq         $4 * 1 * 8,  %%rbx             \\n\\t" // b += 4 (1 x nr)
	"vmulpd           %%ymm1,  %%ymm3, %%ymm7    \\n\\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \\n\\t"
	"vaddpd           %%ymm14, %%ymm6, %%ymm14   \\n\\t"
	"vaddpd           %%ymm12, %%ymm7, %%ymm12   \\n\\t"
	"                                            \\n\\t"
	"vmulpd           %%ymm0,  %%ymm4, %%ymm6    \\n\\t"
	"vmulpd           %%ymm0,  %%ymm5, %%ymm7    \\n\\t"
	"vmovapd   0 * 32(%%rax),  %%ymm0            \\n\\t"
	"vaddpd           %%ymm11, %%ymm6, %%ymm11   \\n\\t"
	"vaddpd           %%ymm9,  %%ymm7, %%ymm9    \\n\\t"
	"                                            \\n\\t"
	"vmulpd           %%ymm1,  %%ymm4, %%ymm6    \\n\\t"
	"vmulpd           %%ymm1,  %%ymm5, %%ymm7    \\n\\t"
	"vaddpd           %%ymm10, %%ymm6, %%ymm10   \\n\\t"
	"vaddpd           %%ymm8,  %%ymm7, %%ymm8    \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t"
	"decq   %%rsi                                \\n\\t" // i -= 1;
	"jne    .DLOOPKLEFT{0}                         \\n\\t" // iterate again if i != 0.
	"                                            \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t"
	".DPOSTACCUM{0}:                               \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t" // ymm15:  ymm13:  ymm11:  ymm9:
	"                                            \\n\\t" // ( ab00  ( ab01  ( ab02  ( ab03
	"                                            \\n\\t" //   ab11    ab10    ab13    ab12
	"                                            \\n\\t" //   ab22    ab23    ab20    ab21
	"                                            \\n\\t" //   ab33 )  ab32 )  ab31 )  ab30 )
	"                                            \\n\\t"
	"                                            \\n\\t" // ymm14:  ymm12:  ymm10:  ymm8:
	"                                            \\n\\t" // ( ab40  ( ab41  ( ab42  ( ab43
	"                                            \\n\\t" //   ab51    ab50    ab53    ab52
	"                                            \\n\\t" //   ab62    ab63    ab60    ab61
	"                                            \\n\\t" //   ab73 )  ab72 )  ab71 )  ab70 )
	"                                            \\n\\t"
	"vmovapd          %%ymm15, %%ymm7            \\n\\t"
	"vshufpd    $0xa, %%ymm15, %%ymm13, %%ymm15  \\n\\t"
	"vshufpd    $0xa, %%ymm13, %%ymm7,  %%ymm13  \\n\\t"
	"                                            \\n\\t"
	"vmovapd          %%ymm11, %%ymm7            \\n\\t"
	"vshufpd    $0xa, %%ymm11, %%ymm9,  %%ymm11  \\n\\t"
	"vshufpd    $0xa, %%ymm9,  %%ymm7,  %%ymm9   \\n\\t"
	"                                            \\n\\t"
	"vmovapd          %%ymm14, %%ymm7            \\n\\t"
	"vshufpd    $0xa, %%ymm14, %%ymm12, %%ymm14  \\n\\t"
	"vshufpd    $0xa, %%ymm12, %%ymm7,  %%ymm12  \\n\\t"
	"                                            \\n\\t"
	"vmovapd          %%ymm10, %%ymm7            \\n\\t"
	"vshufpd    $0xa, %%ymm10, %%ymm8,  %%ymm10  \\n\\t"
	"vshufpd    $0xa, %%ymm8,  %%ymm7,  %%ymm8   \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t" // ymm15:  ymm13:  ymm11:  ymm9:
	"                                            \\n\\t" // ( ab01  ( ab00  ( ab03  ( ab02
	"                                            \\n\\t" //   ab11    ab10    ab13    ab12
	"                                            \\n\\t" //   ab23    ab22    ab21    ab20
	"                                            \\n\\t" //   ab33 )  ab32 )  ab31 )  ab30 )
	"                                            \\n\\t"
	"                                            \\n\\t" // ymm14:  ymm12:  ymm10:  ymm8:
	"                                            \\n\\t" // ( ab41  ( ab40  ( ab43  ( ab42
	"                                            \\n\\t" //   ab51    ab50    ab53    ab52
	"                                            \\n\\t" //   ab63    ab62    ab61    ab60
	"                                            \\n\\t" //   ab73 )  ab72 )  ab71 )  ab70 )
	"                                            \\n\\t"
	"vmovapd           %%ymm15, %%ymm7           \\n\\t"
	"vperm2f128 $0x30, %%ymm15, %%ymm11, %%ymm15 \\n\\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm11, %%ymm11 \\n\\t"
	"                                            \\n\\t"
	"vmovapd           %%ymm13, %%ymm7           \\n\\t"
	"vperm2f128 $0x30, %%ymm13, %%ymm9,  %%ymm13 \\n\\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm9,  %%ymm9  \\n\\t"
	"                                            \\n\\t"
	"vmovapd           %%ymm14, %%ymm7           \\n\\t"
	"vperm2f128 $0x30, %%ymm14, %%ymm10, %%ymm14 \\n\\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm10, %%ymm10 \\n\\t"
	"                                            \\n\\t"
	"vmovapd           %%ymm12, %%ymm7           \\n\\t"
	"vperm2f128 $0x30, %%ymm12, %%ymm8,  %%ymm12 \\n\\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm8,  %%ymm8  \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t" // ymm9:   ymm11:  ymm13:  ymm15:
	"                                            \\n\\t" // ( ab00  ( ab01  ( ab02  ( ab03
	"                                            \\n\\t" //   ab10    ab11    ab12    ab13
	"                                            \\n\\t" //   ab20    ab21    ab22    ab23
	"                                            \\n\\t" //   ab30 )  ab31 )  ab32 )  ab33 )
	"                                            \\n\\t"
	"                                            \\n\\t" // ymm8:   ymm10:  ymm12:  ymm14:
	"                                            \\n\\t" // ( ab40  ( ab41  ( ab42  ( ab43
	"                                            \\n\\t" //   ab50    ab51    ab52    ab53
	"                                            \\n\\t" //   ab60    ab61    ab62    ab63
	"                                            \\n\\t" //   ab70 )  ab71 )  ab72 )  ab73 )
	"                                            \\n\\t"
'''.format( index ) )

def write_common_start_assembly( myfile ):
    write_line( myfile, 1, 'unsigned long long k_iter = (unsigned long long)k / 4;' )
    write_line( myfile, 1, 'unsigned long long k_left = (unsigned long long)k % 4;' )
    write_line( myfile, 1, '__asm__ volatile' )
    write_line( myfile, 1, '(' )
    write_line( myfile, 1, 'INITIALIZE' )


def macro_initialize_assembly( myfile ):
    myfile.write( \
'''\
#define INITIALIZE \\
    "movq                %2, %%rax               \\n\\t" /* load address of a. */ \\
    "movq                %3, %%rbx               \\n\\t" /* load address of b. */ \\
    "movq                %4, %%r15               \\n\\t" /* load address of b_next. */ \\
    "addq          $-4 * 64, %%r15               \\n\\t" \\
    "                                            \\n\\t" \\
    "vmovapd   0 * 32(%%rax), %%ymm0             \\n\\t" /* initialize loop by pre-loading */ \\
    "vmovapd   0 * 32(%%rbx), %%ymm2             \\n\\t" /* elements of a and b. */ \\
    "vpermilpd  $0x5, %%ymm2, %%ymm3             \\n\\t" \\
    "                                            \\n\\t" \\
    "movq                %5, %%rdi               \\n\\t" /* load ldc */ \\
    "leaq        (,%%rdi,8), %%rdi               \\n\\t" /* ldc * sizeof(double) */ \\
    "                                            \\n\\t" \\
''' )


def write_common_end_assembly( myfile, nnz, index, has_nontrivial=False ):
    write_line( myfile, 1, '"                                            \\n\\t"' )
    write_line( myfile, 1, '".DDONE{0}:                                    \\n\\t"'.format( index ) )
    write_line( myfile, 1, '"                                            \\n\\t"' )
    write_line( myfile, 1, ': // output operands (none)' )
    write_line( myfile, 1, ': // input operands' )
    write_line( myfile, 1, '  "m" (k_iter),      // 0' )
    write_line( myfile, 1, '  "m" (k_left),      // 1' )
    write_line( myfile, 1, '  "m" (a),           // 2' )
    write_line( myfile, 1, '  "m" (b),           // 3' )
    write_line( myfile, 1, '  "m" (aux->b_next), // 4' )
    write_line( myfile, 1, '  "m" (ldc),         // 5' )

    add = ''
    add += '\n    '.join( [ '  "m" (c%d)         // %d' % ( i, i+6 ) for i in range( nnz ) ] )
    if has_nontrivial: 
        add += '\n      "m" (alpha_list)        // %d' % (nnz + 6) 
    #write_line( myfile, 1, '  "m" (c)            // 6' )
    write_line( myfile, 1, add )

    write_line( myfile, 1, ': // register clobber list' )
    write_line( myfile, 1, '  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",' )
    write_line( myfile, 1, '  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",' )
    write_line( myfile, 1, '  "xmm0", "xmm1", "xmm2", "xmm3",' )
    write_line( myfile, 1, '  "xmm4", "xmm5", "xmm6", "xmm7",' )
    write_line( myfile, 1, '  "xmm8", "xmm9", "xmm10", "xmm11",' )
    write_line( myfile, 1, '  "xmm12", "xmm13", "xmm14", "xmm15",' )
    write_line( myfile, 1, '  "memory"' )
    write_line( myfile, 1, ');' )


def generate_micro_kernel( myfile, nonzero_coeffs, index ):
    nnz = len( nonzero_coeffs )
    #write_line( myfile, 1, 'a' )
    add = 'inline void bl_dgemm_micro_kernel_stra_abc%d( int k, double *a, double *b, unsigned long long ldc, ' % index
    add += ', '.join( ['double *c%d' % ( i ) for i in range( nnz )] )
    if ( contain_nontrivial( nonzero_coeffs ) ):
        add += ', double *alpha_list'
    add += ', aux_t *aux ) {'
    write_line(myfile, 0, add)

    write_common_start_assembly( myfile )

    write_prefetch_assembly( myfile, nonzero_coeffs )

    write_line( myfile, 1, 'RANKK_UPDATE( %d )' % index )
    #write_common_rankk_assembly( myfile, index )
    #write_common_simple_rankk_assembly( myfile, index )

    write_updatec_assembly( myfile, nonzero_coeffs )
    
    write_common_end_assembly( myfile, nnz, index, contain_nontrivial( nonzero_coeffs ) )

    write_line( myfile, 0, '}' )

    #write_break( myfile )


def main():
    nonzero_coeffs = [ 1, -1 ]
    myfile = open( "a.c", 'w' )
    i = 3
    generate_micro_kernel( myfile, nonzero_coeffs, i )

if __name__ == '__main__':
    main()


