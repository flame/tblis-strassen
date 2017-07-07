#include "blis.h"

void bli_dstra_four_asm_8x4(
                        dim_t k,
                        double* restrict alpha,
                        double* restrict a,
                        double* restrict b,
                        double* restrict beta,
                        unsigned N, 
                        double **c_list,
                        double *coeff_list,
                        inc_t rs_c, inc_t cs_c,
                        auxinfo_t* restrict data,
                        cntx_t*    restrict cntx
                       )
{
    //printf( "dstra_asm: rs_c: %lu, cs_c: %lu\n", rs_c, cs_c );

    //if ( N == 1 ) {
    //    printf( "coeff_list[0]: %lf\n", coeff_list[0] );
    //} else {
    //    printf( "coeff_list[0]: %lf\n", coeff_list[0] );
    //    printf( "coeff_list[1]: %lf\n", coeff_list[1] );
    //}

    //printf("enter stra_asm\n");
    //void*   a_next = bli_auxinfo_next_a( data );
    void*   b_next = bli_auxinfo_next_b( data );

    uint64_t k_iter = k / 4;
    uint64_t k_left = k % 4;

    uint64_t len_c = N;

	__asm__ volatile
	(
	"                                            \n\t"
	"                                            \n\t"
    "movq                %[a], %%rax             \n\t" // load address of a.              ( v )
    "movq                %[b], %%rbx             \n\t" // load address of b.              ( v )
    "movq                %[b_next], %%r15        \n\t" // load address of b_next.         ( v )
    "addq          $-4 * 64, %%r15               \n\t" //                                 ( ? )
    "                                            \n\t"
    "vmovapd   0 * 32(%%rax), %%ymm0             \n\t" // initialize loop by pre-loading
    "vmovapd   0 * 32(%%rbx), %%ymm2             \n\t" // elements of a and b.
    "vpermilpd  $0x5, %%ymm2, %%ymm3             \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "movq                %[cs_c], %%rdi          \n\t" // load cs_c
    "leaq        (,%%rdi,8), %%rdi               \n\t" // cs_c * sizeof(double)
    "                                            \n\t"
    "movq                %[c_list], %%rcx        \n\t" // load address of c_list[ 0 ]
    "                                            \n\t"
	"movq      %[len_c], %%rsi                   \n\t" // i = len;                        ( v )
    "                                            \n\t"
    ".DPREFETCHLOOP:                             \n\t"
    "                                            \n\t"
	"movq       0 * 8(%%rcx),  %%rdx             \n\t" // load address of c_list[ i ]: rdx = c_list[ i ] ( address )
    "                                            \n\t"
	//"testq  %%rdx, %%rdx                         \n\t" // check rdx via logical AND.      ( v )
	//"je     .DC1NULL                             \n\t" // if rdx == 0, jump to code that  ( v )
    "leaq   (%%rdx,%%rdi,2), %%r11               \n\t" // load address of c_list[ i ] + 2 * cs_c;
    "prefetcht0   3 * 8(%%rdx)                   \n\t" // prefetch c_list[ i ] + 0 * cs_c
    "prefetcht0   3 * 8(%%rdx,%%rdi)             \n\t" // prefetch c_list[ i ] + 1 * cs_c
    "prefetcht0   3 * 8(%%r11)                   \n\t" // prefetch c_list[ i ] + 2 * cs_c
    "prefetcht0   3 * 8(%%r11,%%rdi)             \n\t" // prefetch c_list[ i ] + 3 * cs_c
    "                                            \n\t"
    //".DC1NULL:                                   \n\t" // if C1 == NULL, code to jump
    "                                            \n\t"
	"addq              $1 * 8,  %%rcx            \n\t" // c_list += 8
    "                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .DPREFETCHLOOP                       \n\t" // iterate again if i != 0.
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
	"movq      %[k_iter], %%rsi                  \n\t" // i = k_iter;                     ( v )
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
	"movq      %[k_left], %%rsi                  \n\t" // i = k_left;
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
    "movq         %[c_list], %%rcx               \n\t" // load address of c_list[ 0 ]
	"                                            \n\t"
    "                                            \n\t"
    "movq         %[rs_c], %%rsi                 \n\t" // load rs_c
    "                                            \n\t"
    "leaq        (,%%rsi,8), %%rsi               \n\t" // rsi = rs_c * sizeof(double)
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // determine if
    "                                            \n\t" //    c    % 32 == 0, AND
    "                                            \n\t" //  8*cs_c % 32 == 0, AND
    "                                            \n\t" //    rs_c      == 1
    "                                            \n\t" // ie: aligned, ldim aligned, and
    "                                            \n\t" // column-stored
    "                                            \n\t"
    "cmpq       $8, %%rsi                        \n\t" // set ZF if (8*rs_c) == 8.
    "sete           %%bl                         \n\t" // bl = ( ZF == 1 ? 1 : 0 );
	"movq       0 * 8(%%rcx),  %%r11             \n\t" // r11 = c_list[ 0 ] ( address )
    "testq     $31, %%r11                        \n\t" // set ZF if c_list[ 0 ] & 32 is zero.
    "setz           %%bh                         \n\t" // bh = ( ZF == 0 ? 1 : 0 );
	"movq       1 * 8(%%rcx),  %%r8              \n\t" // r8 = c_list[ 1 ] ( address )
    "testq     $31, %%r8                         \n\t" // set ZF if c_list[ 1 ] & 32 is zero.
    "setz           %%ah                         \n\t" // ah = ( ZF == 0 ? 1 : 0 );
	"movq       2 * 8(%%rcx),  %%r9              \n\t" // r9 = c_list[ 2 ] ( address )
    "testq     $31, %%r9                         \n\t" // set ZF if c_list[ 2 ] & 32 is zero.
    "setz           %%dh                         \n\t" // dh = ( ZF == 0 ? 1 : 0 );
	"movq       3 * 8(%%rcx),  %%r10             \n\t" // r10 = c_list[ 3 ] ( address )
    "testq     $31, %%r10                        \n\t" // set ZF if c_list[ 3 ] & 32 is zero.
    "setz           %%dl                         \n\t" // dl = ( ZF == 0 ? 1 : 0 );
    "testq     $31, %%rdi                        \n\t" // set ZF if (8*cs_c) & 32 is zero.
    "setz           %%al                         \n\t" // al = ( ZF == 0 ? 1 : 0 );
    "                                            \n\t" // and(bl,bh) followed by
    "                                            \n\t" // and(bh,al) will reveal result
    "                                            \n\t"
    //"jmp     .DCOLSTORED                         \n\t" // jump to column storage case
    //"jmp     .DGENSTORED                         \n\t" // jump to column storage case
    "                                            \n\t"
    "                                            \n\t" // now avoid loading C if beta == 0
    "                                            \n\t"
//    "vxorpd    %%ymm0,  %%ymm0,  %%ymm0          \n\t" // set ymm0 to zero.
//    "vucomisd  %%xmm0,  %%xmm2                   \n\t" // set ZF if beta == 0.
//    "je      .DBETAZERO                          \n\t" // if ZF = 1, jump to beta == 0 case
    "                                            \n\t"
    "                                            \n\t" // check if aligned/column-stored
    "andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
    "andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
    "andb     %%al, %%ah                         \n\t" // set ZF if al & ah == 1.
    "andb     %%ah, %%dl                         \n\t" // set ZF if ah & dl == 1.
    "andb     %%dl, %%dh                         \n\t" // set ZF if dl & dh == 1.
    "jne     .DCOLSTORED                         \n\t" // jump to column storage case
    "                                            \n\t"
    //"jmp     .DCOLSTORED                         \n\t" // jump to column storage case
    "                                            \n\t"
    ".DGENSTORED:                                \n\t"
    "                                            \n\t"
    "leaq        (,%%rsi,2), %%r12               \n\t" // r12 = 2*rs_c;
    "leaq   (%%r12,%%rsi,1), %%r13               \n\t" // r13 = 3*rs_c;
    "                                            \n\t"
	"movq         %[len_c], %%r10                \n\t" // i = len_c;                        ( v )
    "movq         %[coeff_list], %%rax           \n\t" // load address of alpha_list[ 0 ]
    "                                            \n\t"
    ".DGENSTORELOOP:                             \n\t"
    "                                            \n\t"
	"vbroadcastsd    (%%rax), %%ymm6             \n\t" // load alpha_list[ i ] and duplicate
    "                                            \n\t"
	"movq       0 * 8(%%rcx),  %%rbx             \n\t" // rbx = c_list[ 0 ] ( address )
    "                                            \n\t"
    "leaq   (%%rbx,%%rsi,4), %%rdx               \n\t" // load address of c + 4*rs_c;
    "                                            \n\t"
    "                                            \n\t"
    "vextractf128 $1, %%ymm9,  %%xmm1            \n\t"
    "vmovlpd    (%%rbx),       %%xmm0,  %%xmm0   \n\t" // load c00 and c10,
    "vmovhpd    (%%rbx,%%rsi), %%xmm0,  %%xmm0   \n\t"
    "vmulpd           %%xmm6,  %%xmm9,  %%xmm2   \n\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \n\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%rbx)           \n\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%rbx,%%rsi)     \n\t"
    "vmovlpd    (%%rbx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c20 and c30,
    "vmovhpd    (%%rbx,%%r13), %%xmm0,  %%xmm0   \n\t"
    "vmulpd           %%xmm6,  %%xmm1,  %%xmm2   \n\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \n\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%rbx,%%r12)     \n\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%rbx,%%r13)     \n\t"
    "addq      %%rdi, %%rbx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vextractf128 $1, %%ymm11,  %%xmm1            \n\t"
    "vmovlpd    (%%rbx),       %%xmm0,  %%xmm0   \n\t" // load c01 and c11,
    "vmovhpd    (%%rbx,%%rsi), %%xmm0,  %%xmm0   \n\t"
    "vmulpd           %%xmm6,  %%xmm11,  %%xmm2   \n\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \n\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%rbx)           \n\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%rbx,%%rsi)     \n\t"
    "vmovlpd    (%%rbx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c21 and c31,
    "vmovhpd    (%%rbx,%%r13), %%xmm0,  %%xmm0   \n\t"
    "vmulpd           %%xmm6,  %%xmm1,  %%xmm2   \n\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \n\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%rbx,%%r12)     \n\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%rbx,%%r13)     \n\t"
    "addq      %%rdi, %%rbx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vextractf128 $1, %%ymm13,  %%xmm1            \n\t"
    "vmovlpd    (%%rbx),       %%xmm0,  %%xmm0   \n\t" // load c02 and c12,
    "vmovhpd    (%%rbx,%%rsi), %%xmm0,  %%xmm0   \n\t"
    "vmulpd           %%xmm6,  %%xmm13,  %%xmm2   \n\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \n\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%rbx)           \n\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%rbx,%%rsi)     \n\t"
    "vmovlpd    (%%rbx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c22 and c32,
    "vmovhpd    (%%rbx,%%r13), %%xmm0,  %%xmm0   \n\t"
    "vmulpd           %%xmm6,  %%xmm1,  %%xmm2   \n\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \n\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%rbx,%%r12)     \n\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%rbx,%%r13)     \n\t"
    "addq      %%rdi, %%rbx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vextractf128 $1, %%ymm15,  %%xmm1            \n\t"
    "vmovlpd    (%%rbx),       %%xmm0,  %%xmm0   \n\t" // load c03 and c13,
    "vmovhpd    (%%rbx,%%rsi), %%xmm0,  %%xmm0   \n\t"
    "vmulpd           %%xmm6,  %%xmm15,  %%xmm2   \n\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \n\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%rbx)           \n\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%rbx,%%rsi)     \n\t"
    "vmovlpd    (%%rbx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c23 and c33,
    "vmovhpd    (%%rbx,%%r13), %%xmm0,  %%xmm0   \n\t"
    "vmulpd           %%xmm6,  %%xmm1,  %%xmm2   \n\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \n\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%rbx,%%r12)     \n\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%rbx,%%r13)     \n\t"
    "addq      %%rdi, %%rbx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vextractf128 $1, %%ymm8,  %%xmm1            \n\t"
    "vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load c40 and c50,
    "vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t"
    "vmulpd           %%xmm6,  %%xmm8,  %%xmm2   \n\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \n\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%rdx)           \n\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%rdx,%%rsi)     \n\t"
    "vmovlpd    (%%rdx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c60 and c70,
    "vmovhpd    (%%rdx,%%r13), %%xmm0,  %%xmm0   \n\t"
    "vmulpd           %%xmm6,  %%xmm1,  %%xmm2   \n\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \n\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t"
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vextractf128 $1, %%ymm10,  %%xmm1            \n\t"
    "vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load c41 and c51,
    "vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t"
    "vmulpd           %%xmm6,  %%xmm10,  %%xmm2   \n\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \n\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%rdx)           \n\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%rdx,%%rsi)     \n\t"
    "vmovlpd    (%%rdx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c61 and c71,
    "vmovhpd    (%%rdx,%%r13), %%xmm0,  %%xmm0   \n\t"
    "vmulpd           %%xmm6,  %%xmm1,  %%xmm2   \n\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \n\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t"
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vextractf128 $1, %%ymm12,  %%xmm1            \n\t"
    "vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load c42 and c52,
    "vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t"
    "vmulpd           %%xmm6,  %%xmm12,  %%xmm2   \n\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \n\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%rdx)           \n\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%rdx,%%rsi)     \n\t"
    "vmovlpd    (%%rdx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c62 and c72,
    "vmovhpd    (%%rdx,%%r13), %%xmm0,  %%xmm0   \n\t"
    "vmulpd           %%xmm6,  %%xmm1,  %%xmm2   \n\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \n\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t"
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "vextractf128 $1, %%ymm14,  %%xmm1            \n\t"
    "vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load c43 and c53,
    "vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t"
    "vmulpd           %%xmm6,  %%xmm14,  %%xmm2   \n\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \n\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%rdx)           \n\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%rdx,%%rsi)     \n\t"
    "vmovlpd    (%%rdx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c63 and c73,
    "vmovhpd    (%%rdx,%%r13), %%xmm0,  %%xmm0   \n\t"
    "vmulpd           %%xmm6,  %%xmm1,  %%xmm2   \n\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \n\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t"
    "addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
	"addq              $1 * 8,  %%rcx            \n\t" // c_list += 8
	"addq              $1 * 8,  %%rax            \n\t" // alpha_list += 8
	"                                            \n\t"
	"decq   %%r10                                \n\t" // i -= 1;
	"jne    .DGENSTORELOOP                          \n\t" // iterate again if i != 0.
    "                                            \n\t"
    "jmp    .DDONE                               \n\t" // jump to end.
    "                                            \n\t"
    ".DCOLSTORED:                                \n\t"
    "                                            \n\t"
    "movq         %[coeff_list], %%rax           \n\t" // load address of alpha_list[ 0 ]
	"movq         %[len_c], %%rsi                \n\t" // i = len_c;                        ( v )
    ".DSTORELOOP:                                \n\t"
    "                                            \n\t"
	"movq       0 * 8(%%rcx),  %%rdx             \n\t" // rdx = c_list[ i ] ( address )
    "                                            \n\t"
	//"movq       0 * 8(%%rax),  %%rbx             \n\t" // load address of alpha_list[ i ]
	//"vbroadcastsd    (%%rbx), %%ymm6             \n\t" // load alpha_list[ 1 ] and duplicate
	"vbroadcastsd    (%%rax), %%ymm6             \n\t" // load alpha_list[ i ] and duplicate
    "                                            \n\t"
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
	"addq              $1 * 8,  %%rcx            \n\t" // c_list += 8
	"addq              $1 * 8,  %%rax            \n\t" // alpha_list += 8
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .DSTORELOOP                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
    ".DDONE:                                     \n\t"
	"                                            \n\t"
	: // output operands (none)
	: // input operands
	  [k_iter]     "m" (k_iter),             // 0
	  [k_left]     "m" (k_left),             // 1
	  [a]          "m" (a),                  // 2
	  [b]          "m" (b),                  // 3
	  [coeff_list] "m" (coeff_list),         // 4
	  [len_c]      "m" (len_c),              // 5
	  [c_list]     "m" (c_list),             // 6
      [rs_c]       "m" (rs_c),               // 7
      [cs_c]       "m" (cs_c),               // 8
	  [b_next]     "m" (b_next)              // 9
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx",  "rdi", "rsi",
      "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	);
}

