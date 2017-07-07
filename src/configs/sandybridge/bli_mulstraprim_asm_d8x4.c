
#include "blis.h"

//#define MONITORS

void bli_dmulstraprim_asm_8x4( dim_t k,
                               dim_t len,
                               double* restrict *alpha_list,
                               double* restrict a,
                               double* restrict b,
                               double* restrict *beta_list,
                               double* restrict *c_list, inc_t rs_c, inc_t cs_c,
                               auxinfo_t* data )
{
    //unsigned long long ldc,
    //inc_t ldcA = cs_cA;
    //inc_t ldcB = cs_cB;

    //assert ( cs_cA == cs_cB );
    //inc_t ldc  = cs_c;

    inc_t ldc;
    if (rs_c == 1) {
        ldc = cs_c;
    } else {
        ldc = rs_c;
    }



    //#pragma unroll
    //for ( dim_t ii = 0; ii < STRA_LIST_LEN; ii++ ) {
    //    if ( c_list[ ii ] == NULL ) {
    //        printf( "c_list[ %lu ] == NULL\n", ii );
    //        printf( "a:alpha_list[ %lu ] = %lf\n", ii, *alpha_list[ ii ] );
    //    }
    //        printf( "b:alpha_list[ %lu ] = %lf\n", ii, *alpha_list[ ii ] );
    //}


    /*
    printf( "a: \n" );
    for ( dim_t jj = 0; jj < 8; jj++ ) {
        for ( dim_t ii = 0; ii < k; ii++ ) {
            printf( "%lf,", a[8*ii+jj] );
        }
        printf( "\n" );
    }

    printf( "b: \n" );
    for ( dim_t ii = 0; ii < k; ii++ ) {
        for ( dim_t jj = 0; jj < 4; jj++ ) {
            printf( "%lf,", b[4*ii+jj] );
        }
        printf( "\n" );
    }

    printf( "ldc = %lu\n", ldc );
    #pragma unroll
    for ( dim_t ii = 0; ii < STRA_LIST_LEN; ii++ ) {
        printf( "c_list[%lu]: \n", ii );
        printf( "%lf, %lf, %lf, %lf\n", c_list[ii][0], c_list[ii][ ldc + 0], c_list[ii][ ldc * 2 + 0], c_list[ii][ ldc * 3 + 0] );
        printf( "%lf, %lf, %lf, %lf\n", c_list[ii][1], c_list[ii][ ldc + 1], c_list[ii][ ldc * 2 + 1], c_list[ii][ ldc * 3 + 1] );
        printf( "%lf, %lf, %lf, %lf\n", c_list[ii][2], c_list[ii][ ldc + 2], c_list[ii][ ldc * 2 + 2], c_list[ii][ ldc * 3 + 2] );
        printf( "%lf, %lf, %lf, %lf\n", c_list[ii][3], c_list[ii][ ldc + 3], c_list[ii][ ldc * 2 + 3], c_list[ii][ ldc * 3 + 3] );
        printf( "%lf, %lf, %lf, %lf\n", c_list[ii][4], c_list[ii][ ldc + 4], c_list[ii][ ldc * 2 + 4], c_list[ii][ ldc * 3 + 4] );
        printf( "%lf, %lf, %lf, %lf\n", c_list[ii][5], c_list[ii][ ldc + 5], c_list[ii][ ldc * 2 + 5], c_list[ii][ ldc * 3 + 5] );
        printf( "%lf, %lf, %lf, %lf\n", c_list[ii][6], c_list[ii][ ldc + 6], c_list[ii][ ldc * 2 + 6], c_list[ii][ ldc * 3 + 6] );
        printf( "%lf, %lf, %lf, %lf\n", c_list[ii][7], c_list[ii][ ldc + 7], c_list[ii][ ldc * 2 + 7], c_list[ii][ ldc * 3 + 7] );
        printf( "\n" );
    }
    printf( "\n" );
    */

	void*   b_next = bli_auxinfo_next_b( data );

    dim_t k_iter = k / 4;
    dim_t k_left = k % 4;

    //printf( "%ld\n", last );
#ifdef MONITORS
    int toph, topl, both, botl, midl, midh, mid2l, mid2h;
#endif
 

	__asm__ volatile
	(
	"                                            \n\t"
#ifdef MONITORS
    "rdtsc                                       \n\t"
    "mov %%eax,  %18                            \n\t" // eax -> topl
    "mov %%edx,  %19                            \n\t" // eax -> toph
#endif
	"                                            \n\t"
    "movq                %2, %%rax               \n\t" // load address of a.              ( v )
    "movq                %3, %%rbx               \n\t" // load address of b.              ( v )
    "movq               %17, %%r15               \n\t" // load address of b_next.         ( v )
    "addq          $-4 * 64, %%r15               \n\t" //                                 ( ? )
    "                                            \n\t"
    "vmovapd   0 * 32(%%rax), %%ymm0             \n\t" // initialize loop by pre-loading
    "vmovapd   0 * 32(%%rbx), %%ymm2             \n\t" // elements of a and b.
    "vpermilpd  $0x5, %%ymm2, %%ymm3             \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "movq               %16, %%rdi               \n\t" // load ldc
    "leaq        (,%%rdi,8), %%rdi               \n\t" // ldc * sizeof(double)
    "                                            \n\t"
    "                                            \n\t"
  "movq               %12, %%rcx               \n\t" // load address of c_list[ 0 ]
//	"testq  %%rcx, %%rcx                         \n\t" // check rcx via logical AND.      ( v )
//	"je     .DSPECIALHANDLETEST1                 \n\t" // if rcx == 0, jump to code that  ( v )
    "leaq   (%%rcx,%%rdi,2), %%r12               \n\t" // load address of c_list[ 0 ] + 2 * ldc;
    "prefetcht0   3 * 8(%%rcx)                   \n\t" // prefetch c_list[ 0 ] + 0 * ldc
    "prefetcht0   3 * 8(%%rcx,%%rdi)             \n\t" // prefetch c_list[ 0 ] + 1 * ldc
    "prefetcht0   3 * 8(%%r12)                   \n\t" // prefetch c_list[ 0 ] + 2 * ldc
    "prefetcht0   3 * 8(%%r12,%%rdi)             \n\t" // prefetch c_list[ 0 ] + 3 * ldc
    "                                            \n\t"
    "                                            \n\t"
  "movq               %13, %%rdx               \n\t" // load address of c_list[ 1 ]
	"testq  %%rdx, %%rdx                         \n\t" // check rdx via logical AND.      ( v )
//	"je     .DSPECIALHANDLETEST2                 \n\t" // if rcx == 0, jump to code that  ( v )
	"je     .DC1NULL                             \n\t" // if rdx == 0, jump to code that  ( v )
    "leaq   (%%rdx,%%rdi,2), %%r11               \n\t" // load address of c_list[ 1 ] + 2 * ldc;
    "prefetcht0   3 * 8(%%rdx)                   \n\t" // prefetch c_list[ 1 ] + 0 * ldc
    "prefetcht0   3 * 8(%%rdx,%%rdi)             \n\t" // prefetch c_list[ 1 ] + 1 * ldc
    "prefetcht0   3 * 8(%%r11)                   \n\t" // prefetch c_list[ 1 ] + 2 * ldc
    "prefetcht0   3 * 8(%%r11,%%rdi)             \n\t" // prefetch c_list[ 1 ] + 3 * ldc
    "                                            \n\t"
    ".DC1NULL:                                   \n\t" // if C1 == NULL, code to jump
    "                                            \n\t"
  "movq               %14, %%r14               \n\t" // load address of c_list[ 2 ]
	"testq  %%r14, %%r14                         \n\t" // check r14 via logical AND.      ( v )
//	"je     .DSPECIALHANDLETEST3                 \n\t" // if rcx == 0, jump to code that  ( v )
	"je     .DC2NULL                             \n\t" // if r14 == 0, jump to code that  ( v )
    "leaq   (%%r14,%%rdi,2), %%r10               \n\t" // load address of c_list[ 2 ] + 2 * ldc;
    "prefetcht0   3 * 8(%%r14)                   \n\t" // prefetch c_list[ 2 ] + 0 * ldc
    "prefetcht0   3 * 8(%%r14,%%rdi)             \n\t" // prefetch c_list[ 2 ] + 1 * ldc
    "prefetcht0   3 * 8(%%r10)                   \n\t" // prefetch c_list[ 2 ] + 2 * ldc
    "prefetcht0   3 * 8(%%r10,%%rdi)             \n\t" // prefetch c_list[ 2 ] + 3 * ldc
    "                                            \n\t"
    ".DC2NULL:                                   \n\t" // if C2 == NULL, code to jump
    "                                            \n\t"
  "movq               %15, %%r13               \n\t" // load address of c_list[ 3 ]
	"testq  %%r13, %%r13                         \n\t" // check rcx via logical AND.      ( v )
//	"je     .DSPECIALHANDLETEST4                 \n\t" // if rcx == 0, jump to code that  ( v )
	"je     .DC3NULL                             \n\t" // if rcx == 0, jump to code that  ( v )
    "leaq   (%%r13,%%rdi,2), %%rsi               \n\t" // load address of c_list[ 3 ] + 2 * ldc;
    "prefetcht0   3 * 8(%%r13)                   \n\t" // prefetch c_list[ 3 ] + 0 * ldc
    "prefetcht0   3 * 8(%%r13,%%rdi)             \n\t" // prefetch c_list[ 3 ] + 1 * ldc
    "prefetcht0   3 * 8(%%rsi)                   \n\t" // prefetch c_list[ 3 ] + 2 * ldc
    "prefetcht0   3 * 8(%%rsi,%%rdi)             \n\t" // prefetch c_list[ 3 ] + 3 * ldc
    "                                            \n\t"
    ".DC3NULL:                                   \n\t" // if C3 == NULL, code to jump
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
	"vxorpd    %%ymm8,  %%ymm8,  %%ymm8          \n\t" // set ymm8 to 0                   ( v )
	"vxorpd    %%ymm9,  %%ymm9,  %%ymm9          \n\t"
	"vxorpd    %%ymm10, %%ymm10, %%ymm10         \n\t"
	"vxorpd    %%ymm11, %%ymm11, %%ymm11         \n\t"
	"vxorpd    %%ymm12, %%ymm12, %%ymm12         \n\t"
	"vxorpd    %%ymm13, %%ymm13, %%ymm13         \n\t"
	"vxorpd    %%ymm14, %%ymm14, %%ymm14         \n\t"
	"vxorpd    %%ymm15, %%ymm15, %%ymm15         \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq      %0, %%rsi                         \n\t" // i = k_iter;                     ( v )
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.        ( v )
	"je     .DCONSIDKLEFT                        \n\t" // if i == 0, jump to code that    ( v )
	"                                            \n\t" // contains the k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".DLOOPKITER:                                \n\t" // MAIN LOOP
	"                                            \n\t"
	"addq         $4 * 4 * 8,  %%r15             \n\t" // b_next += 4*4 (unroll x nr)     ( v )
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"vmovapd   1 * 32(%%rax),  %%ymm1            \n\t" // preload a47 for iter 0
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t" // ymm6 ( c_tmp0 ) = ymm0 ( a03 ) * ymm2( b0 )
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t" // ymm4 ( b0x3_0 )
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t" // ymm7 ( c_tmp1 ) = ymm0 ( a03 ) * ymm3( b0x5 )
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t" // ymm5 ( b0x3_1 )
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t" // ymm15 ( c_03_0 ) += ymm6( c_tmp0 )
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t" // ymm13 ( c_03_1 ) += ymm7( c_tmp1 )
	"                                            \n\t"
	"prefetcht0  16 * 32(%%rax)                  \n\t" // prefetch a03 for iter 1
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   1 * 32(%%rbx),  %%ymm2            \n\t" // preload b for iter 1
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   2 * 32(%%rax),  %%ymm0            \n\t" // preload a03 for iter 1
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \n\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"prefetcht0   0 * 32(%%r15)                  \n\t" // prefetch b_next[0*4]
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \n\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 1
	"vmovapd   3 * 32(%%rax),  %%ymm1            \n\t" // preload a47 for iter 1
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	"prefetcht0  18 * 32(%%rax)                  \n\t" // prefetch a for iter 9  ( ? )
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   2 * 32(%%rbx),  %%ymm2            \n\t" // preload b for iter 2 
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   4 * 32(%%rax),  %%ymm0            \n\t" // preload a03 for iter 2
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \n\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \n\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 2
	"vmovapd   5 * 32(%%rax),  %%ymm1            \n\t" // preload a47 for iter 2
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	"prefetcht0  20 * 32(%%rax)                  \n\t" // prefetch a for iter 10 ( ? )
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   3 * 32(%%rbx),  %%ymm2            \n\t" // preload b for iter 3
	"addq         $4 * 4 * 8,  %%rbx             \n\t" // b += 4*4 (unroll x nr)
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   6 * 32(%%rax),  %%ymm0            \n\t" // preload a03 for iter 3
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \n\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"prefetcht0   2 * 32(%%r15)                  \n\t" // prefetch b_next[2*4]
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \n\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 3
	"vmovapd   7 * 32(%%rax),  %%ymm1            \n\t" // preload a47 for iter 3
	"addq         $4 * 8 * 8,  %%rax             \n\t" // a += 4*8 (unroll x mr)
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	"prefetcht0  14 * 32(%%rax)                  \n\t" // prefetch a for iter 11 ( ? )
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   0 * 32(%%rbx),  %%ymm2            \n\t" // preload b for iter 4
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   0 * 32(%%rax),  %%ymm0            \n\t" // preload a03 for iter 4
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \n\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \n\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .DLOOPKITER                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DCONSIDKLEFT:                              \n\t"
	"                                            \n\t"
	"movq      %1, %%rsi                         \n\t" // i = k_left;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .DPOSTACCUM                          \n\t" // if i == 0, we're done; jump to end.
	"                                            \n\t" // else, we prepare to enter k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".DLOOPKLEFT:                                \n\t" // EDGE LOOP
	"                                            \n\t"
	"vmovapd   1 * 32(%%rax),  %%ymm1            \n\t" // preload a47 
	"addq         $8 * 1 * 8,  %%rax             \n\t" // a += 8 (1 x mr)
	"vmulpd           %%ymm0,  %%ymm2, %%ymm6    \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2, %%ymm4    \n\t"
	"vmulpd           %%ymm0,  %%ymm3, %%ymm7    \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3, %%ymm5    \n\t"
	"vaddpd           %%ymm15, %%ymm6, %%ymm15   \n\t"
	"vaddpd           %%ymm13, %%ymm7, %%ymm13   \n\t"
	"                                            \n\t"
	"prefetcht0  14 * 32(%%rax)                  \n\t" // prefetch a03 for iter 7 later ( ? )
	"vmulpd           %%ymm1,  %%ymm2, %%ymm6    \n\t"
	"vmovapd   1 * 32(%%rbx),  %%ymm2            \n\t"
	"addq         $4 * 1 * 8,  %%rbx             \n\t" // b += 4 (1 x nr)
	"vmulpd           %%ymm1,  %%ymm3, %%ymm7    \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6, %%ymm14   \n\t"
	"vaddpd           %%ymm12, %%ymm7, %%ymm12   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4, %%ymm6    \n\t"
	"vmulpd           %%ymm0,  %%ymm5, %%ymm7    \n\t"
	"vmovapd   0 * 32(%%rax),  %%ymm0            \n\t"
	"vaddpd           %%ymm11, %%ymm6, %%ymm11   \n\t"
	"vaddpd           %%ymm9,  %%ymm7, %%ymm9    \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4, %%ymm6    \n\t"
	"vmulpd           %%ymm1,  %%ymm5, %%ymm7    \n\t"
	"vaddpd           %%ymm10, %%ymm6, %%ymm10   \n\t"
	"vaddpd           %%ymm8,  %%ymm7, %%ymm8    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .DLOOPKLEFT                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DPOSTACCUM:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
	"                                            \n\t" // ( ab00  ( ab01  ( ab02  ( ab03
	"                                            \n\t" //   ab11    ab10    ab13    ab12  
	"                                            \n\t" //   ab22    ab23    ab20    ab21
	"                                            \n\t" //   ab33 )  ab32 )  ab31 )  ab30 )
	"                                            \n\t"
	"                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
	"                                            \n\t" // ( ab40  ( ab41  ( ab42  ( ab43
	"                                            \n\t" //   ab51    ab50    ab53    ab52  
	"                                            \n\t" //   ab62    ab63    ab60    ab61
	"                                            \n\t" //   ab73 )  ab72 )  ab71 )  ab70 )
	"                                            \n\t"
	"vmovapd          %%ymm15, %%ymm7            \n\t"
	"vshufpd    $0xa, %%ymm15, %%ymm13, %%ymm15  \n\t"
	"vshufpd    $0xa, %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	"vmovapd          %%ymm11, %%ymm7            \n\t"
	"vshufpd    $0xa, %%ymm11, %%ymm9,  %%ymm11  \n\t"
	"vshufpd    $0xa, %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmovapd          %%ymm14, %%ymm7            \n\t"
	"vshufpd    $0xa, %%ymm14, %%ymm12, %%ymm14  \n\t"
	"vshufpd    $0xa, %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmovapd          %%ymm10, %%ymm7            \n\t"
	"vshufpd    $0xa, %%ymm10, %%ymm8,  %%ymm10  \n\t"
	"vshufpd    $0xa, %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
	"                                            \n\t" // ( ab01  ( ab00  ( ab03  ( ab02
	"                                            \n\t" //   ab11    ab10    ab13    ab12  
	"                                            \n\t" //   ab23    ab22    ab21    ab20
	"                                            \n\t" //   ab33 )  ab32 )  ab31 )  ab30 )
	"                                            \n\t"
	"                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
	"                                            \n\t" // ( ab41  ( ab40  ( ab43  ( ab42
	"                                            \n\t" //   ab51    ab50    ab53    ab52  
	"                                            \n\t" //   ab63    ab62    ab61    ab60
	"                                            \n\t" //   ab73 )  ab72 )  ab71 )  ab70 )
	"                                            \n\t"
	"vmovapd           %%ymm15, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm15, %%ymm11, %%ymm15 \n\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm11, %%ymm11 \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm13, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm13, %%ymm9,  %%ymm13 \n\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm9,  %%ymm9  \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm14, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm14, %%ymm10, %%ymm14 \n\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm10, %%ymm10 \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm12, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm12, %%ymm8,  %%ymm12 \n\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm8,  %%ymm8  \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm9:   ymm11:  ymm13:  ymm15:
	"                                            \n\t" // ( ab00  ( ab01  ( ab02  ( ab03
	"                                            \n\t" //   ab10    ab11    ab12    ab13  
	"                                            \n\t" //   ab20    ab21    ab22    ab23
	"                                            \n\t" //   ab30 )  ab31 )  ab32 )  ab33 )
	"                                            \n\t"
	"                                            \n\t" // ymm8:   ymm10:  ymm12:  ymm14:
	"                                            \n\t" // ( ab40  ( ab41  ( ab42  ( ab43
	"                                            \n\t" //   ab50    ab51    ab52    ab53  
	"                                            \n\t" //   ab60    ab61    ab62    ab63
	"                                            \n\t" //   ab70 )  ab71 )  ab72 )  ab73 )
	"                                            \n\t"
	"                                            \n\t"
    "movq         %4, %%rax                      \n\t" // load address of alpha_list[ 0 ]
	"movq         %5, %%rbx                      \n\t" // load address of alpha_list[ 1 ]
	"movq         %6, %%r12                      \n\t" // load address of alpha_list[ 2 ]
	"movq         %7, %%r11                      \n\t" // load address of alpha_list[ 3 ]
	"vbroadcastsd    (%%rax), %%ymm7             \n\t" // load alpha_list[ 0 ] and duplicate
	"vbroadcastsd    (%%rbx), %%ymm6             \n\t" // load alpha_list[ 1 ] and duplicate
	"vbroadcastsd    (%%r12), %%ymm5             \n\t" // load alpha_list[ 2 ] and duplicate
	"vbroadcastsd    (%%r11), %%ymm4             \n\t" // load alpha_list[ 3 ] and duplicate
	"                                            \n\t"
    "movq         %8, %%rax                      \n\t" // load address of beta1
//	"movq         %9, %%rbx                      \n\t" // load address of beta2
	"vbroadcastsd    (%%rax), %%ymm3             \n\t" // load beta_list[ 0 ] and duplicate
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
////////////////////////////Not sure if we need the following...
//    "movq                   %8, %%rcx            \n\t" // load address of c
//    "movq                   %9, %%rdx               \n\t" // load address of cB
//	"movq                  %10, %%rdi            \n\t" // load  ldc
//	"leaq           (,%%rdi,8), %%rdi            \n\t" // rsi = ldc * sizeof(double)
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vxorpd    %%ymm0,  %%ymm0,  %%ymm0          \n\t" // set ymm0 to zero.
	"vucomisd  %%xmm0,  %%xmm3                   \n\t" // set ZF if beta == 0.
	//"je      .DBETAZERO                          \n\t" // if ZF = 1, jump to beta == 0 case
	//"                                            \n\t"
	//"andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
	//"andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
	"je     .DCOLSTORBZ                          \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
////	"vmulpd            %%ymm0,  %%ymm5,  %%ymm0  \n\t" // scale by beta1
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd    0 * 32(%%rcx),  %%ymm0           \n\t" // ymm0 = c_list[0]( 0:3, 0 )
	"vmulpd            %%ymm7,  %%ymm9,  %%ymm1  \n\t" // scale by alpha1, ymm1 = ymm7( alpha1 ) * ymm9( ab0_3:0 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%rcx)    \n\t" // and store back to memory: c_list[0]( 0:3, 0 )
	"vmovapd    1 * 32(%%rcx),  %%ymm3           \n\t" // ymm3 = c_list[0]( 4:7, 0 )
	"vmulpd            %%ymm7,  %%ymm8,  %%ymm2  \n\t" // scale by alpha1, ymm2 = ymm7( alpha1 ) * ymm8( ab4_7:0 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%rcx)    \n\t" // and store back to memory: c_list[0]( 4:7, 0 )
	"addq              %%rdi,   %%rcx            \n\t"
    "                                            \n\t"
	"vmovapd    0 * 32(%%rcx),  %%ymm0           \n\t" // ymm0 = c_list[0]( 0:3, 0 )
	"vmulpd            %%ymm7,  %%ymm11, %%ymm1  \n\t" // scale by alpha1, ymm1 = ymm7( alpha1 ) * ymm11( ab0_3:1 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%rcx)    \n\t" // and store back to memory: c_list[0]( 0:3, 0 )
	"vmovapd    1 * 32(%%rcx),  %%ymm3           \n\t" // ymm3 = c_list[0]( 4:7, 0 )
	"vmulpd            %%ymm7,  %%ymm10, %%ymm2  \n\t" // scale by alpha1, ymm2 = ymm7( alpha1 ) * ymm10( ab4_7:1 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%rcx)    \n\t" // and store back to memory: c_list[0]( 4:7, 0 )
	"addq              %%rdi,   %%rcx            \n\t"
    "                                            \n\t"
	"vmovapd    0 * 32(%%rcx),  %%ymm0           \n\t" // ymm0 = c_list[0]( 0:3, 0 )
	"vmulpd            %%ymm7,  %%ymm13, %%ymm1  \n\t" // scale by alpha1, ymm1 = ymm7( alpha1 ) * ymm13( ab0_3:1 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%rcx)    \n\t" // and store back to memory: c_list[0]( 0:3, 0 )
	"vmovapd    1 * 32(%%rcx),  %%ymm3           \n\t" // ymm3 = c_list[0]( 4:7, 0 )
	"vmulpd            %%ymm7,  %%ymm12, %%ymm2  \n\t" // scale by alpha1, ymm2 = ymm7( alpha1 ) * ymm12( ab4_7:1 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%rcx)    \n\t" // and store back to memory: c_list[0]( 4:7, 0 )
	"addq              %%rdi,   %%rcx            \n\t"
    "                                            \n\t"
	"vmovapd    0 * 32(%%rcx),  %%ymm0           \n\t" // ymm0 = c_list[0]( 0:3, 0 )
	"vmulpd            %%ymm7,  %%ymm15, %%ymm1  \n\t" // scale by alpha1, ymm1 = ymm7( alpha1 ) * ymm15( ab0_3:1 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%rcx)    \n\t" // and store back to memory: c_list[0]( 0:3, 0 )
	"vmovapd    1 * 32(%%rcx),  %%ymm3           \n\t" // ymm3 = c_list[0]( 4:7, 0 )
	"vmulpd            %%ymm7,  %%ymm14, %%ymm2  \n\t" // scale by alpha1, ymm2 = ymm7( alpha1 ) * ymm14( ab4_7:1 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%rcx)    \n\t" // and store back to memory: c_list[0]( 4:7, 0 )
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
	"testq  %%rdx, %%rdx                         \n\t" // check rdx via logical AND.      ( v )
	"je     .DC1NULL_R1                          \n\t" // if rdx == 0, jump to code that  ( v )
    "                                            \n\t"
	"vmovapd    0 * 32(%%rdx),  %%ymm0           \n\t" // ymm0 = c_list[1]( 0:3, 0 )
	"vmulpd            %%ymm6,  %%ymm9,  %%ymm1  \n\t" // scale by alpha2, ymm1 = ymm6( alpha2 ) * ymm9( ab0_3:0 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 0:3, 0 )
	"vmovapd    1 * 32(%%rdx),  %%ymm3           \n\t" // ymm3 = c_list[1]( 4:7, 0 )
	"vmulpd            %%ymm6,  %%ymm8,  %%ymm2  \n\t" // scale by alpha2, ymm2 = ymm6( alpha2 ) * ymm8( ab4_7:0 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 4:7, 0 )
	"addq              %%rdi,   %%rdx            \n\t"
    "                                            \n\t"
	"vmovapd    0 * 32(%%rdx),  %%ymm0           \n\t" // ymm0 = c_list[1]( 0:3, 0 )
	"vmulpd            %%ymm6,  %%ymm11, %%ymm1  \n\t" // scale by alpha2, ymm1 = ymm6( alpha2 ) * ymm11( ab0_3:1 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 0:3, 0 )
	"vmovapd    1 * 32(%%rdx),  %%ymm3           \n\t" // ymm3 = c_list[1]( 4:7, 0 )
	"vmulpd            %%ymm6,  %%ymm10, %%ymm2  \n\t" // scale by alpha2, ymm2 = ymm6( alpha2 ) * ymm10( ab4_7:1 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 4:7, 0 )
	"addq              %%rdi,   %%rdx            \n\t"
    "                                            \n\t"
	"vmovapd    0 * 32(%%rdx),  %%ymm0           \n\t" // ymm0 = c_list[1]( 0:3, 0 )
	"vmulpd            %%ymm6,  %%ymm13, %%ymm1  \n\t" // scale by alpha2, ymm1 = ymm6( alpha2 ) * ymm13( ab0_3:1 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 0:3, 0 )
	"vmovapd    1 * 32(%%rdx),  %%ymm3           \n\t" // ymm3 = c_list[1]( 4:7, 0 )
	"vmulpd            %%ymm6,  %%ymm12, %%ymm2  \n\t" // scale by alpha2, ymm2 = ymm6( alpha2 ) * ymm12( ab4_7:1 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 4:7, 0 )
	"addq              %%rdi,   %%rdx            \n\t"
    "                                            \n\t"
	"vmovapd    0 * 32(%%rdx),  %%ymm0           \n\t" // ymm0 = c_list[1]( 0:3, 0 )
	"vmulpd            %%ymm6,  %%ymm15, %%ymm1  \n\t" // scale by alpha2, ymm1 = ymm6( alpha2 ) * ymm15( ab0_3:1 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 0:3, 0 )
	"vmovapd    1 * 32(%%rdx),  %%ymm3           \n\t" // ymm3 = c_list[1]( 4:7, 0 )
	"vmulpd            %%ymm6,  %%ymm14, %%ymm2  \n\t" // scale by alpha2, ymm2 = ymm6( alpha2 ) * ymm14( ab4_7:1 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 4:7, 0 )
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".DC1NULL_R1:                                \n\t" // if C1 == NULL, code to jump
	"testq  %%r14, %%r14                         \n\t" // check r14 via logical AND.      ( v )
	"je     .DC2NULL_R1                          \n\t" // if r14 == 0, jump to code that  ( v )
    "                                            \n\t"
	"vmovapd    0 * 32(%%r14),  %%ymm0           \n\t" // ymm0 = c_list[2]( 0:3, 0 )
	"vmulpd            %%ymm5,  %%ymm9,  %%ymm1  \n\t" // scale by alpha3, ymm1 = ymm5( alpha3 ) * ymm9( ab0_3:0 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%r14)    \n\t" // and store back to memory: c_list[2]( 0:3, 0 )
	"vmovapd    1 * 32(%%r14),  %%ymm3           \n\t" // ymm3 = c_list[2]( 4:7, 0 )
	"vmulpd            %%ymm5,  %%ymm8,  %%ymm2  \n\t" // scale by alpha3, ymm2 = ymm5( alpha3 ) * ymm8( ab4_7:0 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%r14)    \n\t" // and store back to memory: c_list[2]( 4:7, 0 )
	"addq              %%rdi,   %%r14            \n\t"
    "                                            \n\t"
	"vmovapd    0 * 32(%%r14),  %%ymm0           \n\t" // ymm0 = c_list[2]( 0:3, 0 )
	"vmulpd            %%ymm5,  %%ymm11, %%ymm1  \n\t" // scale by alpha3, ymm1 = ymm5( alpha3 ) * ymm11( ab0_3:1 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%r14)    \n\t" // and store back to memory: c_list[2]( 0:3, 0 )
	"vmovapd    1 * 32(%%r14),  %%ymm3           \n\t" // ymm3 = c_list[2]( 4:7, 0 )
	"vmulpd            %%ymm5,  %%ymm10, %%ymm2  \n\t" // scale by alpha3, ymm2 = ymm5( alpha3 ) * ymm10( ab4_7:1 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%r14)    \n\t" // and store back to memory: c_list[2]( 4:7, 0 )
	"addq              %%rdi,   %%r14            \n\t"
    "                                            \n\t"
	"vmovapd    0 * 32(%%r14),  %%ymm0           \n\t" // ymm0 = c_list[2]( 0:3, 0 )
	"vmulpd            %%ymm5,  %%ymm13, %%ymm1  \n\t" // scale by alpha3, ymm1 = ymm5( alpha3 ) * ymm13( ab0_3:1 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%r14)    \n\t" // and store back to memory: c_list[2]( 0:3, 0 )
	"vmovapd    1 * 32(%%r14),  %%ymm3           \n\t" // ymm3 = c_list[2]( 4:7, 0 )
	"vmulpd            %%ymm5,  %%ymm12, %%ymm2  \n\t" // scale by alpha3, ymm2 = ymm5( alpha3 ) * ymm12( ab4_7:1 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%r14)    \n\t" // and store back to memory: c_list[2]( 4:7, 0 )
	"addq              %%rdi,   %%r14            \n\t"
    "                                            \n\t"
	"vmovapd    0 * 32(%%r14),  %%ymm0           \n\t" // ymm0 = c_list[2]( 0:3, 0 )
	"vmulpd            %%ymm5,  %%ymm15, %%ymm1  \n\t" // scale by alpha3, ymm1 = ymm5( alpha3 ) * ymm15( ab0_3:1 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%r14)    \n\t" // and store back to memory: c_list[2]( 0:3, 0 )
	"vmovapd    1 * 32(%%r14),  %%ymm3           \n\t" // ymm3 = c_list[2]( 4:7, 0 )
	"vmulpd            %%ymm5,  %%ymm14, %%ymm2  \n\t" // scale by alpha3, ymm2 = ymm5( alpha3 ) * ymm14( ab4_7:1 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%r14)    \n\t" // and store back to memory: c_list[2]( 4:7, 0 )
    "                                            \n\t"
    "                                            \n\t"
    ".DC2NULL_R1:                                \n\t" // if C2 == NULL, code to jump
	"testq  %%r13, %%r13                         \n\t" // check rcx via logical AND.      ( v )
	"je     .DC3NULL_R1                          \n\t" // if rcx == 0, jump to code that  ( v )
    "                                            \n\t"
	"vmovapd    0 * 32(%%r13),  %%ymm0           \n\t" // ymm0 = c_list[3]( 0:3, 0 )
	"vmulpd            %%ymm4,  %%ymm9,  %%ymm1  \n\t" // scale by alpha4, ymm1 = ymm4( alpha4 ) * ymm9( ab0_3:0 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%r13)    \n\t" // and store back to memory: c_list[3]( 0:3, 0 )
	"vmovapd    1 * 32(%%r13),  %%ymm3           \n\t" // ymm3 = c_list[3]( 4:7, 0 )
	"vmulpd            %%ymm4,  %%ymm8,  %%ymm2  \n\t" // scale by alpha4, ymm2 = ymm4( alpha4 ) * ymm8( ab4_7:0 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%r13)    \n\t" // and store back to memory: c_list[3]( 4:7, 0 )
	"addq              %%rdi,   %%r13            \n\t"
    "                                            \n\t"
	"vmovapd    0 * 32(%%r13),  %%ymm0           \n\t" // ymm0 = c_list[3]( 0:3, 0 )
	"vmulpd            %%ymm4,  %%ymm11, %%ymm1  \n\t" // scale by alpha4, ymm1 = ymm4( alpha4 ) * ymm11( ab0_3:1 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%r13)    \n\t" // and store back to memory: c_list[3]( 0:3, 0 )
	"vmovapd    1 * 32(%%r13),  %%ymm3           \n\t" // ymm3 = c_list[3]( 4:7, 0 )
	"vmulpd            %%ymm4,  %%ymm10, %%ymm2  \n\t" // scale by alpha4, ymm2 = ymm4( alpha4 ) * ymm10( ab4_7:1 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%r13)    \n\t" // and store back to memory: c_list[3]( 4:7, 0 )
	"addq              %%rdi,   %%r13            \n\t"
    "                                            \n\t"
	"vmovapd    0 * 32(%%r13),  %%ymm0           \n\t" // ymm0 = c_list[3]( 0:3, 0 )
	"vmulpd            %%ymm4,  %%ymm13, %%ymm1  \n\t" // scale by alpha4, ymm1 = ymm4( alpha4 ) * ymm13( ab0_3:1 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%r13)    \n\t" // and store back to memory: c_list[3]( 0:3, 0 )
	"vmovapd    1 * 32(%%r13),  %%ymm3           \n\t" // ymm3 = c_list[3]( 4:7, 0 )
	"vmulpd            %%ymm4,  %%ymm12, %%ymm2  \n\t" // scale by alpha4, ymm2 = ymm4( alpha4 ) * ymm12( ab4_7:1 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%r13)    \n\t" // and store back to memory: c_list[3]( 4:7, 0 )
	"addq              %%rdi,   %%r13            \n\t"
    "                                            \n\t"
	"vmovapd    0 * 32(%%r13),  %%ymm0           \n\t" // ymm0 = c_list[3]( 0:3, 0 )
	"vmulpd            %%ymm4,  %%ymm15, %%ymm1  \n\t" // scale by alpha4, ymm1 = ymm4( alpha4 ) * ymm15( ab0_3:1 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%r13)    \n\t" // and store back to memory: c_list[3]( 0:3, 0 )
	"vmovapd    1 * 32(%%r13),  %%ymm3           \n\t" // ymm3 = c_list[3]( 4:7, 0 )
	"vmulpd            %%ymm4,  %%ymm14, %%ymm2  \n\t" // scale by alpha4, ymm2 = ymm4( alpha4 ) * ymm14( ab4_7:1 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%r13)    \n\t" // and store back to memory: c_list[3]( 4:7, 0 )
    "                                            \n\t"
    ".DC3NULL_R1:                                \n\t" // if C3 == NULL, code to jump
	"                                            \n\t"
	"jmp    .DDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
//	".STOREBACK:                                 \n\t"
//	"                                            \n\t"
//	"movq                  %12, %%rcx            \n\t" // load address of c
//	"movq                  %16, %%rdi            \n\t" // load address of ldc
//	"leaq           (,%%rdi,8), %%rdi            \n\t" // rsi = ldc * sizeof(double)
//	"                                            \n\t"
//	"vmovapd           %%ymm9,   0(%%rcx)         \n\t" // C_c( 0, 0:3 ) = ymm9
//	"vmovapd           %%ymm8,  32(%%rcx)         \n\t" // C_c( 1, 0:3 ) = ymm8
//	"addq              %%rdi,   %%rcx            \n\t"
//	"vmovapd           %%ymm11,  0(%%rcx)         \n\t" // C_c( 2, 0:3 ) = ymm11
//	"vmovapd           %%ymm10, 32(%%rcx)         \n\t" // C_c( 3, 0:3 ) = ymm10
//	"addq              %%rdi,   %%rcx            \n\t"
//	"vmovapd           %%ymm13,  0(%%rcx)         \n\t" // C_c( 4, 0:3 ) = ymm13
//	"vmovapd           %%ymm12, 32(%%rcx)         \n\t" // C_c( 5, 0:3 ) = ymm12
//	"addq              %%rdi,   %%rcx            \n\t"
//	"vmovapd           %%ymm15,  0(%%rcx)         \n\t" // C_c( 6, 0:3 ) = ymm15
//	"vmovapd           %%ymm14, 32(%%rcx)         \n\t" // C_c( 7, 0:3 ) = ymm14
//	"                                            \n\t"
//	"jmp    .DDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DCOLSTORBZ:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmulpd            %%ymm7,  %%ymm9,  %%ymm1  \n\t" // scale by alpha1, ymm1 = ymm7( alpha1 ) * ymm9( ab0_3:0 )
	"vmovapd           %%ymm1,  0 * 32(%%rcx)    \n\t" // and store back to memory: c_list[0]( 0:3, 0 )
	"vmulpd            %%ymm7,  %%ymm8,  %%ymm2  \n\t" // scale by alpha1, ymm2 = ymm7( alpha1 ) * ymm8( ab4_7:0 )
	"vmovapd           %%ymm2,  1 * 32(%%rcx)    \n\t" // and store back to memory: c_list[0]( 4:7, 0 )
	"addq              %%rdi,   %%rcx            \n\t"
    "                                            \n\t"
	"vmulpd            %%ymm7,  %%ymm11, %%ymm1  \n\t" // scale by alpha1, ymm1 = ymm7( alpha1 ) * ymm11( ab0_3:1 )
	"vmovapd           %%ymm1,  0 * 32(%%rcx)    \n\t" // and store back to memory: c_list[0]( 0:3, 0 )
	"vmulpd            %%ymm7,  %%ymm10, %%ymm2  \n\t" // scale by alpha1, ymm2 = ymm7( alpha1 ) * ymm10( ab4_7:1 )
	"vmovapd           %%ymm2,  1 * 32(%%rcx)    \n\t" // and store back to memory: c_list[0]( 4:7, 0 )
	"addq              %%rdi,   %%rcx            \n\t"
    "                                            \n\t"
	"vmulpd            %%ymm7,  %%ymm13, %%ymm1  \n\t" // scale by alpha1, ymm1 = ymm7( alpha1 ) * ymm13( ab0_3:1 )
	"vmovapd           %%ymm1,  0 * 32(%%rcx)    \n\t" // and store back to memory: c_list[0]( 0:3, 0 )
	"vmulpd            %%ymm7,  %%ymm12, %%ymm2  \n\t" // scale by alpha1, ymm2 = ymm7( alpha1 ) * ymm12( ab4_7:1 )
	"vmovapd           %%ymm2,  1 * 32(%%rcx)    \n\t" // and store back to memory: c_list[0]( 4:7, 0 )
	"addq              %%rdi,   %%rcx            \n\t"
    "                                            \n\t"
	"vmulpd            %%ymm7,  %%ymm15, %%ymm1  \n\t" // scale by alpha1, ymm1 = ymm7( alpha1 ) * ymm15( ab0_3:1 )
	"vmovapd           %%ymm1,  0 * 32(%%rcx)    \n\t" // and store back to memory: c_list[0]( 0:3, 0 )
	"vmulpd            %%ymm7,  %%ymm14, %%ymm2  \n\t" // scale by alpha1, ymm2 = ymm7( alpha1 ) * ymm14( ab4_7:1 )
	"vmovapd           %%ymm2,  1 * 32(%%rcx)    \n\t" // and store back to memory: c_list[0]( 4:7, 0 )
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
	"testq  %%rdx, %%rdx                         \n\t" // check rdx via logical AND.      ( v )
	"je     .DC1NULL_R2                          \n\t" // if rdx == 0, jump to code that  ( v )
    "                                            \n\t"
	"vmulpd            %%ymm6,  %%ymm9,  %%ymm1  \n\t" // scale by alpha2, ymm1 = ymm6( alpha2 ) * ymm9( ab0_3:0 )
	"vmovapd           %%ymm1,  0 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 0:3, 0 )
	"vmulpd            %%ymm6,  %%ymm8,  %%ymm2  \n\t" // scale by alpha2, ymm2 = ymm6( alpha2 ) * ymm8( ab4_7:0 )
	"vmovapd           %%ymm2,  1 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 4:7, 0 )
	"addq              %%rdi,   %%rdx            \n\t"
    "                                            \n\t"
	"vmulpd            %%ymm6,  %%ymm11, %%ymm1  \n\t" // scale by alpha2, ymm1 = ymm6( alpha2 ) * ymm11( ab0_3:1 )
	"vmovapd           %%ymm1,  0 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 0:3, 0 )
	"vmulpd            %%ymm6,  %%ymm10, %%ymm2  \n\t" // scale by alpha2, ymm2 = ymm6( alpha2 ) * ymm10( ab4_7:1 )
	"vmovapd           %%ymm2,  1 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 4:7, 0 )
	"addq              %%rdi,   %%rdx            \n\t"
    "                                            \n\t"
	"vmulpd            %%ymm6,  %%ymm13, %%ymm1  \n\t" // scale by alpha2, ymm1 = ymm6( alpha2 ) * ymm13( ab0_3:1 )
	"vmovapd           %%ymm1,  0 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 0:3, 0 )
	"vmulpd            %%ymm6,  %%ymm12, %%ymm2  \n\t" // scale by alpha2, ymm2 = ymm6( alpha2 ) * ymm12( ab4_7:1 )
	"vmovapd           %%ymm2,  1 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 4:7, 0 )
	"addq              %%rdi,   %%rdx            \n\t"
    "                                            \n\t"
	"vmulpd            %%ymm6,  %%ymm15, %%ymm1  \n\t" // scale by alpha2, ymm1 = ymm6( alpha2 ) * ymm15( ab0_3:1 )
	"vmovapd           %%ymm1,  0 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 0:3, 0 )
	"vmulpd            %%ymm6,  %%ymm14, %%ymm2  \n\t" // scale by alpha2, ymm2 = ymm6( alpha2 ) * ymm14( ab4_7:1 )
	"vmovapd           %%ymm2,  1 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 4:7, 0 )
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".DC1NULL_R2:                                \n\t" // if C1 == NULL, code to jump
	"testq  %%r14, %%r14                         \n\t" // check r14 via logical AND.      ( v )
	"je     .DC2NULL_R2                          \n\t" // if r14 == 0, jump to code that  ( v )
    "                                            \n\t"
	"vmulpd            %%ymm5,  %%ymm9,  %%ymm1  \n\t" // scale by alpha3, ymm1 = ymm5( alpha3 ) * ymm9( ab0_3:0 )
	"vmovapd           %%ymm1,  0 * 32(%%r14)    \n\t" // and store back to memory: c_list[2]( 0:3, 0 )
	"vmulpd            %%ymm5,  %%ymm8,  %%ymm2  \n\t" // scale by alpha3, ymm2 = ymm5( alpha3 ) * ymm8( ab4_7:0 )
	"vmovapd           %%ymm2,  1 * 32(%%r14)    \n\t" // and store back to memory: c_list[2]( 4:7, 0 )
	"addq              %%rdi,   %%r14            \n\t"
    "                                            \n\t"
	"vmulpd            %%ymm5,  %%ymm11, %%ymm1  \n\t" // scale by alpha3, ymm1 = ymm5( alpha3 ) * ymm11( ab0_3:1 )
	"vmovapd           %%ymm1,  0 * 32(%%r14)    \n\t" // and store back to memory: c_list[2]( 0:3, 0 )
	"vmulpd            %%ymm5,  %%ymm10, %%ymm2  \n\t" // scale by alpha3, ymm2 = ymm5( alpha3 ) * ymm10( ab4_7:1 )
	"vmovapd           %%ymm2,  1 * 32(%%r14)    \n\t" // and store back to memory: c_list[2]( 4:7, 0 )
	"addq              %%rdi,   %%r14            \n\t"
    "                                            \n\t"
	"vmulpd            %%ymm5,  %%ymm13, %%ymm1  \n\t" // scale by alpha3, ymm1 = ymm5( alpha3 ) * ymm13( ab0_3:1 )
	"vmovapd           %%ymm1,  0 * 32(%%r14)    \n\t" // and store back to memory: c_list[2]( 0:3, 0 )
	"vmulpd            %%ymm5,  %%ymm12, %%ymm2  \n\t" // scale by alpha3, ymm2 = ymm5( alpha3 ) * ymm12( ab4_7:1 )
	"vmovapd           %%ymm2,  1 * 32(%%r14)    \n\t" // and store back to memory: c_list[2]( 4:7, 0 )
	"addq              %%rdi,   %%r14            \n\t"
    "                                            \n\t"
	"vmulpd            %%ymm5,  %%ymm15, %%ymm1  \n\t" // scale by alpha3, ymm1 = ymm5( alpha3 ) * ymm15( ab0_3:1 )
	"vmovapd           %%ymm1,  0 * 32(%%r14)    \n\t" // and store back to memory: c_list[2]( 0:3, 0 )
	"vmulpd            %%ymm5,  %%ymm14, %%ymm2  \n\t" // scale by alpha3, ymm2 = ymm5( alpha3 ) * ymm14( ab4_7:1 )
	"vmovapd           %%ymm2,  1 * 32(%%r14)    \n\t" // and store back to memory: c_list[2]( 4:7, 0 )
    "                                            \n\t"
    "                                            \n\t"
    ".DC2NULL_R2:                                \n\t" // if C2 == NULL, code to jump
	"testq  %%r13, %%r13                         \n\t" // check rcx via logical AND.      ( v )
	"je     .DC3NULL_R2                          \n\t" // if rcx == 0, jump to code that  ( v )
    "                                            \n\t"
	"vmulpd            %%ymm4,  %%ymm9,  %%ymm1  \n\t" // scale by alpha4, ymm1 = ymm4( alpha4 ) * ymm9( ab0_3:0 )
	"vmovapd           %%ymm1,  0 * 32(%%r13)    \n\t" // and store back to memory: c_list[3]( 0:3, 0 )
	"vmulpd            %%ymm4,  %%ymm8,  %%ymm2  \n\t" // scale by alpha4, ymm2 = ymm4( alpha4 ) * ymm8( ab4_7:0 )
	"vmovapd           %%ymm2,  1 * 32(%%r13)    \n\t" // and store back to memory: c_list[3]( 4:7, 0 )
	"addq              %%rdi,   %%r13            \n\t"
    "                                            \n\t"
	"vmulpd            %%ymm4,  %%ymm11, %%ymm1  \n\t" // scale by alpha4, ymm1 = ymm4( alpha4 ) * ymm11( ab0_3:1 )
	"vmovapd           %%ymm1,  0 * 32(%%r13)    \n\t" // and store back to memory: c_list[3]( 0:3, 0 )
	"vmulpd            %%ymm4,  %%ymm10, %%ymm2  \n\t" // scale by alpha4, ymm2 = ymm4( alpha4 ) * ymm10( ab4_7:1 )
	"vmovapd           %%ymm2,  1 * 32(%%r13)    \n\t" // and store back to memory: c_list[3]( 4:7, 0 )
	"addq              %%rdi,   %%r13            \n\t"
    "                                            \n\t"
	"vmulpd            %%ymm4,  %%ymm13, %%ymm1  \n\t" // scale by alpha4, ymm1 = ymm4( alpha4 ) * ymm13( ab0_3:1 )
	"vmovapd           %%ymm1,  0 * 32(%%r13)    \n\t" // and store back to memory: c_list[3]( 0:3, 0 )
	"vmulpd            %%ymm4,  %%ymm12, %%ymm2  \n\t" // scale by alpha4, ymm2 = ymm4( alpha4 ) * ymm12( ab4_7:1 )
	"vmovapd           %%ymm2,  1 * 32(%%r13)    \n\t" // and store back to memory: c_list[3]( 4:7, 0 )
	"addq              %%rdi,   %%r13            \n\t"
    "                                            \n\t"
	"vmulpd            %%ymm4,  %%ymm15, %%ymm1  \n\t" // scale by alpha4, ymm1 = ymm4( alpha4 ) * ymm15( ab0_3:1 )
	"vmovapd           %%ymm1,  0 * 32(%%r13)    \n\t" // and store back to memory: c_list[3]( 0:3, 0 )
	"vmulpd            %%ymm4,  %%ymm14, %%ymm2  \n\t" // scale by alpha4, ymm2 = ymm4( alpha4 ) * ymm14( ab4_7:1 )
	"vmovapd           %%ymm2,  1 * 32(%%r13)    \n\t" // and store back to memory: c_list[3]( 4:7, 0 )
    "                                            \n\t"
    ".DC3NULL_R2:                                \n\t" // if C3 == NULL, code to jump
	"                                            \n\t"

	"                                            \n\t"
//	"                                            \n\t"
	"                                            \n\t"
//	"jmp    .DDONE                               \n\t" // jump to end.
//	"                                            \n\t"
//	".DSPECIALHANDLETEST1:                       \n\t" // if i == 0, jump to code that    ( v )
//    "movq              %%rcx, %16                \n\t" // store ldc
//	"                                            \n\t"
//    ".DSPECIALHANDLETEST2:                       \n\t" // if i == 0, jump to code that    ( v )
//    "movq              %%rdx, %16                \n\t" // store ldc
//	"                                            \n\t"
//    ".DSPECIALHANDLETEST3:                       \n\t" // if i == 0, jump to code that    ( v )
//    "movq              %%r14, %16                \n\t" // store ldc
//	"                                            \n\t"
//	"                                            \n\t"
//    ".DSPECIALHANDLETEST4:                       \n\t" // if i == 0, jump to code that    ( v )
//    "movq              %%r13, %16                \n\t" // store ldc
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DDONE:                                     \n\t"
	"                                            \n\t"
#ifdef MONITORS
    "rdtsc                                       \n\t"
    "mov %%eax, %20                             \n\t" // eax -> botl
    "mov %%edx, %21                             \n\t" // edx -> both
#endif
	"                                            \n\t"
	: // output operands (none)
	: // input operands
	  "m" (k_iter),             // 0
	  "m" (k_left),             // 1
	  "m" (a),                  // 2
	  "m" (b),                  // 3
	  "m" (alpha_list[0]),      // 4
	  "m" (alpha_list[1]),      // 5
	  "m" (alpha_list[2]),      // 6
	  "m" (alpha_list[3]),      // 7
	  "m" (beta_list[0]),       // 8
	  "m" (beta_list[1]),       // 9
	  "m" (beta_list[2]),       // 10
	  "m" (beta_list[3]),       // 11
	  "m" (c_list[0]),          // 12
	  "m" (c_list[1]),          // 13
	  "m" (c_list[2]),          // 14
	  "m" (c_list[3]),          // 15
      "m" (ldc),                // 16
	  "m" (b_next) 
#ifdef MONITORS
          ,                     // 17
	  "m" (topl),               // 18
	  "m" (toph),               // 19
	  "m" (botl),               // 20
	  "m" (both)                // 21
#endif
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx",  "rdi", "rsi",
      "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	);

#ifdef MONITORS
    dim_t top = ((dim_t)toph << 32) | topl;
    //dim_t mid = ((dim_t)midh << 32) | midl;
    //dim_t mid2 = ((dim_t)mid2h << 32) | mid2l;
    dim_t bot = ((dim_t)both << 32) | botl;
    //printf("setup =\t%u\tmain loop =\t%u\tcleanup=\t%u\ttotal=\t%u\n", mid - top, mid2 - mid, bot - mid2, bot - top);
    printf("total=\t%lu\n", bot - top);
#endif


    //printf( "ldc = %lu\n", ldc );
    /*
    #pragma unroll
    for ( dim_t ii = 0; ii < STRA_LIST_LEN; ii++ ) {
        printf( "c_list[%lu]: \n", ii );
        printf( "%lf, %lf, %lf, %lf\n", c_list[ii][0], c_list[ii][ ldc + 0], c_list[ii][ ldc * 2 + 0], c_list[ii][ ldc * 3 + 0] );
        printf( "%lf, %lf, %lf, %lf\n", c_list[ii][1], c_list[ii][ ldc + 1], c_list[ii][ ldc * 2 + 1], c_list[ii][ ldc * 3 + 1] );
        printf( "%lf, %lf, %lf, %lf\n", c_list[ii][2], c_list[ii][ ldc + 2], c_list[ii][ ldc * 2 + 2], c_list[ii][ ldc * 3 + 2] );
        printf( "%lf, %lf, %lf, %lf\n", c_list[ii][3], c_list[ii][ ldc + 3], c_list[ii][ ldc * 2 + 3], c_list[ii][ ldc * 3 + 3] );
        printf( "%lf, %lf, %lf, %lf\n", c_list[ii][4], c_list[ii][ ldc + 4], c_list[ii][ ldc * 2 + 4], c_list[ii][ ldc * 3 + 4] );
        printf( "%lf, %lf, %lf, %lf\n", c_list[ii][5], c_list[ii][ ldc + 5], c_list[ii][ ldc * 2 + 5], c_list[ii][ ldc * 3 + 5] );
        printf( "%lf, %lf, %lf, %lf\n", c_list[ii][6], c_list[ii][ ldc + 6], c_list[ii][ ldc * 2 + 6], c_list[ii][ ldc * 3 + 6] );
        printf( "%lf, %lf, %lf, %lf\n", c_list[ii][7], c_list[ii][ ldc + 7], c_list[ii][ ldc * 2 + 7], c_list[ii][ ldc * 3 + 7] );
        printf( "\n" );
    }
    printf( "\n" );
    */


}

