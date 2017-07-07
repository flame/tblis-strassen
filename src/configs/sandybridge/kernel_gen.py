import sys
from common import is_one, is_negone, is_nonzero, write_line, write_break, transpose, printmat, contain_nontrivial

#Round Robin way to get the register
def get_reg( avoid_reg = '', cno = -1 ):
    get_reg.counter += 1
    res_reg = get_reg.reg_pool[ get_reg.counter % len(get_reg.reg_pool) ]

    while ( (res_reg == avoid_reg) or (res_reg in get_reg.allocated_pool) ):
        get_reg.counter += 1
        res_reg = get_reg.reg_pool[ get_reg.counter % len(get_reg.reg_pool) ]

    if ( cno != -1 ): #allocate reg for c
        get_reg.allocated_pool.append( res_reg )
        get_reg.c2reg[cno] = res_reg

    return res_reg


get_reg.counter = -1
get_reg.reg_pool = [ 'rcx', 'r8', 'rdx', 'r9', 'r10', 'r11', 'r14', 'r15' ]
#get_reg.reg_pool = [ 'rcx', 'rdx', 'rsi', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14' ]
# rdi, rax, rbx, r15, already occupied.
# 'r12', 'r13', 
# (rcx, rdx, rsi, r8, r9, r10, r11, r12, r13, r14): register allocation algorithm
#get_reg.allocated_pool = [] # for c, coeff register

get_reg.allocated_pool = [] # for c, coeff register

get_reg.c2reg = {} # c_no to c_reg mapping dictionary
#c2reg[0]='rcx'



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

def write_updatec_genstored_assembly( myfile, nnz ):

    write_line( myfile, 1, '".DGENSTORED:                                \\n\\t"' )
    write_line( myfile, 1, '"                                            \\n\\t"' )
    write_line( myfile, 1, '"leaq        (,%%rsi,2), %%r12               \\n\\t" // r12 = 2*rs_c;' )
    write_line( myfile, 1, '"leaq   (%%r12,%%rsi,1), %%r13               \\n\\t" // r13 = 3*rs_c;' )
    write_line( myfile, 1, '"                                            \\n\\t"' )

    for j in range( nnz ):

        c47_reg = get_reg()

        coeff_avx_reg = get_avx_reg()
        #coeff_avx_reg = 'ymm6'

        myfile.write( \
'''\
    "movq         %[coeff{0}], %%{3}               \\n\\t" // load address of coeff{0}
    "vbroadcastsd    (%%{3}), %%{4}             \\n\\t" // load coeff{0} and duplicate
    "leaq   (%%{1},%%rsi,4), %%{2}               \\n\\t" // load address of c{0} + 4*rs_c;'
    "                                            \\n\\t"
'''.format( j, get_reg.c2reg[j], c47_reg, get_reg(), coeff_avx_reg  ) )

        c03_ymm_list = ['ymm9', 'ymm11', 'ymm13', 'ymm15'] #c00:c33
        c47_ymm_list = ['ymm8', 'ymm10', 'ymm12', 'ymm14'] #c40:c73

#        for idx in range(4):
#            myfile.write( \
#'''\
#    "vextractf128 $1, %%{0},  %%xmm1            \\n\\t"
#    "vmovlpd    (%%{2}),       %%xmm0,  %%xmm0   \\n\\t" // load c{4}_0{1} and c{4}_1{1},
#    "vmovhpd    (%%{2},%%rsi), %%xmm0,  %%xmm0   \\n\\t"
#    "vmulpd           %%xmm{5},  %%xmm{3},  %%xmm2   \\n\\t" // scale by coeff{4},
#    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \\n\\t" // add the gemm result,
#    "vmovlpd          %%xmm2,  (%%{2})           \\n\\t" // and store back to memory.
#    "vmovhpd          %%xmm2,  (%%{2},%%rsi)     \\n\\t"
#    "vmovlpd    (%%{2},%%r12), %%xmm0,  %%xmm0   \\n\\t" // load c{4}_2{1} and c{4}_3{1},
#    "vmovhpd    (%%{2},%%r13), %%xmm0,  %%xmm0   \\n\\t"
#    "vmulpd           %%xmm{5},  %%xmm1,  %%xmm2   \\n\\t" // scale by coeff{4},
#    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \\n\\t" // add the gemm result,
#    "vmovlpd          %%xmm2,  (%%{2},%%r12)     \\n\\t" // and store back to memory.
#    "vmovhpd          %%xmm2,  (%%{2},%%r13)     \\n\\t"
#    "addq      %%rdi, %%{2}                      \\n\\t" // c += cs_c;
#    "                                            \\n\\t"
#'''.format( c03_ymm_list[idx], str(idx), get_reg.c2reg[j], c03_ymm_list[idx][3:], j, coeff_avx_reg[3:], ) )


        for idx in range(4):
            myfile.write( \
'''\
    "vextractf128 $1, %%{0},  %%xmm{7}            \\n\\t"
    "vmovlpd    (%%{2}),       %%xmm{6},  %%xmm{6}   \\n\\t" // load c{4}_0{1} and c{4}_1{1},
    "vmovhpd    (%%{2},%%rsi), %%xmm{6},  %%xmm{6}   \\n\\t"
    "vmulpd           %%xmm{5},  %%xmm{3},  %%xmm{8}   \\n\\t" // scale by coeff{4},
    "vaddpd           %%xmm{8},  %%xmm{6},  %%xmm{8}   \\n\\t" // add the gemm result,
    "vmovlpd          %%xmm{8},  (%%{2})           \\n\\t" // and store back to memory.
    "vmovhpd          %%xmm{8},  (%%{2},%%rsi)     \\n\\t"
    "vmovlpd    (%%{2},%%r12), %%xmm{6},  %%xmm{6}   \\n\\t" // load c{4}_2{1} and c{4}_3{1},
    "vmovhpd    (%%{2},%%r13), %%xmm{6},  %%xmm{6}   \\n\\t"
    "vmulpd           %%xmm{5},  %%xmm{7},  %%xmm{8}   \\n\\t" // scale by coeff{4},
    "vaddpd           %%xmm{8},  %%xmm{6},  %%xmm{8}   \\n\\t" // add the gemm result,
    "vmovlpd          %%xmm{8},  (%%{2},%%r12)     \\n\\t" // and store back to memory.
    "vmovhpd          %%xmm{8},  (%%{2},%%r13)     \\n\\t"
    "                                            \\n\\t"
'''.format( c03_ymm_list[idx], str(idx), get_reg.c2reg[j], c03_ymm_list[idx][3:], j, coeff_avx_reg[3:], (get_avx_reg(avoid_reg=coeff_avx_reg))[3:], (get_avx_reg(avoid_reg=coeff_avx_reg))[3:], (get_avx_reg(avoid_reg=coeff_avx_reg))[3:], ) )
            if ( idx != 3 ):
                write_line( myfile, 1, '"addq      %%rdi, %%{0}                      \\n\\t" // c += cs_c;'.format( get_reg.c2reg[j] ) )




#        for idx in range(4):
#            myfile.write( \
#'''\
#    "vextractf128 $1, %%{0},  %%xmm1            \\n\\t"
#    "vmovlpd    (%%{2}),       %%xmm0,  %%xmm0   \\n\\t" // load c{4}_4{1} and c{4}_5{1},
#    "vmovhpd    (%%{2},%%rsi), %%xmm0,  %%xmm0   \\n\\t"
#    "vmulpd           %%xmm{5},  %%xmm{3},  %%xmm2   \\n\\t" // scale by coeff{4},
#    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \\n\\t" // add the gemm result,
#    "vmovlpd          %%xmm2,  (%%{2})           \\n\\t" // and store back to memory.
#    "vmovhpd          %%xmm2,  (%%{2},%%rsi)     \\n\\t"
#    "vmovlpd    (%%{2},%%r12), %%xmm0,  %%xmm0   \\n\\t" // load c{4}_6{1} and c{4}_7{1},
#    "vmovhpd    (%%{2},%%r13), %%xmm0,  %%xmm0   \\n\\t"
#    "vmulpd           %%xmm{5},  %%xmm1,  %%xmm2   \\n\\t" // scale by coeff{4},
#    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \\n\\t" // add the gemm result,
#    "vmovlpd          %%xmm2,  (%%{2},%%r12)     \\n\\t" // and store back to memory.
#    "vmovhpd          %%xmm2,  (%%{2},%%r13)     \\n\\t"
#    "addq      %%rdi, %%{2}                      \\n\\t" // c += cs_c;
#    "                                            \\n\\t"
#'''.format( c47_ymm_list[idx], str(idx), c47_reg, c47_ymm_list[idx][3:], j, coeff_avx_reg[3:],  ) )

        for idx in range(4):
            myfile.write( \
'''\
    "vextractf128 $1, %%{0},  %%xmm{7}            \\n\\t"
    "vmovlpd    (%%{2}),       %%xmm{6},  %%xmm{6}   \\n\\t" // load c{4}_4{1} and c{4}_5{1},
    "vmovhpd    (%%{2},%%rsi), %%xmm{6},  %%xmm{6}   \\n\\t"
    "vmulpd           %%xmm{5},  %%xmm{3},  %%xmm{8}   \\n\\t" // scale by coeff{4},
    "vaddpd           %%xmm{8},  %%xmm{6},  %%xmm{8}   \\n\\t" // add the gemm result,
    "vmovlpd          %%xmm{8},  (%%{2})           \\n\\t" // and store back to memory.
    "vmovhpd          %%xmm{8},  (%%{2},%%rsi)     \\n\\t"
    "vmovlpd    (%%{2},%%r12), %%xmm{6},  %%xmm{6}   \\n\\t" // load c{4}_6{1} and c{4}_7{1},
    "vmovhpd    (%%{2},%%r13), %%xmm{6},  %%xmm{6}   \\n\\t"
    "vmulpd           %%xmm{5},  %%xmm{7},  %%xmm{8}   \\n\\t" // scale by coeff{4},
    "vaddpd           %%xmm{8},  %%xmm{6},  %%xmm{8}   \\n\\t" // add the gemm result,
    "vmovlpd          %%xmm{8},  (%%{2},%%r12)     \\n\\t" // and store back to memory.
    "vmovhpd          %%xmm{8},  (%%{2},%%r13)     \\n\\t"
    "                                            \\n\\t"
'''.format( c47_ymm_list[idx], str(idx), c47_reg, c47_ymm_list[idx][3:], j, coeff_avx_reg[3:], (get_avx_reg(avoid_reg=coeff_avx_reg))[3:], (get_avx_reg(avoid_reg=coeff_avx_reg))[3:], (get_avx_reg(avoid_reg=coeff_avx_reg))[3:],  ) )
            if ( idx != 3 ):
                write_line( myfile, 1, '"addq      %%rdi, %%{0}                      \\n\\t" // c += cs_c;'.format( c47_reg ) )



    write_line( myfile, 1, '"                                            \\n\\t"' )
    write_line( myfile, 1, '"jmp    .DDONE                               \\n\\t" // jump to end.' )
    write_line( myfile, 1, '"                                            \\n\\t"' )



#def write_updatec_genstored_assembly( myfile, nnz ):
#
#
#    write_line( myfile, 1, '".DGENSTORED:                                \\n\\t"' )
#    write_line( myfile, 1, '"                                            \\n\\t"' )
#    write_line( myfile, 1, '"leaq        (,%%rsi,2), %%r12               \\n\\t" // r12 = 2*rs_c;' )
#    write_line( myfile, 1, '"leaq   (%%r12,%%rsi,1), %%r13               \\n\\t" // r13 = 3*rs_c;' )
#    write_line( myfile, 1, '"                                            \\n\\t"' )
#
#    for j in range( nnz ):
#        c47_reg = get_reg()
#
#        coeff_avx_reg = get_avx_reg()
#
#        myfile.write( \
#'''\
#    "movq         %[coeff{0}], %%{3}               \\n\\t" // load address of coeff{0}
#    "vbroadcastsd    (%%{3}), %%{4}             \\n\\t" // load coeff{0} and duplicate
#    "leaq   (%%{1},%%rsi,4), %%{2}               \\n\\t" // load address of c{0} + 4*rs_c;
#    "                                            \\n\\t"
#'''.format( j, get_reg.c2reg[j], c47_reg, get_reg(), coeff_avx_reg ) )
#
#        c03_ymm_list = ['ymm9', 'ymm11', 'ymm13', 'ymm15'] #c00:c33
#        c47_ymm_list = ['ymm8', 'ymm10', 'ymm12', 'ymm14'] #c40:c73
#
#        for idx in range(4):
#            myfile.write( \
#'''\
#    "vextractf128 $1, %%{0},  %%xmm{6}            \\n\\t"
#    "vmovlpd    (%%{2}),       %%xmm{5},  %%xmm{5}   \\n\\t" // load c{8}_0{1} and c{8}_1{1},
#    "vmovhpd    (%%{2},%%rsi), %%xmm{5},  %%xmm{5}   \\n\\t"
#    "vmulpd           %%xmm{4},  %%xmm{3},  %%xmm{7}   \\n\\t" // scale by coeff{8},
#    "vaddpd           %%xmm{7},  %%xmm{5},  %%xmm{7}   \\n\\t" // add the gemm result,
#    "vmovlpd          %%xmm{7},  (%%{2})           \\n\\t" // and store back to memory.
#    "vmovhpd          %%xmm{7},  (%%{2},%%rsi)     \\n\\t"
#    "vmovlpd    (%%{2},%%r12), %%xmm{5},  %%xmm{5}   \\n\\t" // load c{8}_2{1} and c{8}_3{1},
#    "vmovhpd    (%%{2},%%r13), %%xmm{5},  %%xmm{5}   \\n\\t"
#    "vmulpd           %%xmm{4},  %%xmm{6},  %%xmm{7}   \\n\\t" // scale by coeff{8},
#    "vaddpd           %%xmm{7},  %%xmm{5},  %%xmm{7}   \\n\\t" // add the gemm result,
#    "vmovlpd          %%xmm{7},  (%%{2},%%r12)     \\n\\t" // and store back to memory.
#    "vmovhpd          %%xmm{7},  (%%{2},%%r13)     \\n\\t"
#    "addq      %%rdi, %%{2}                      \\n\\t" // c += cs_c;
#    "                                            \\n\\t"
#'''.format( c03_ymm_list[idx], str(idx), get_reg.c2reg[j], c03_ymm_list[idx][3:], coeff_avx_reg[3:], (get_avx_reg(coeff_avx_reg))[3:], (get_avx_reg(coeff_avx_reg))[3:], (get_avx_reg(coeff_avx_reg))[3:], j  ) )
#            #if ( j != nnz-1 ):
#            #    write( myfile, 1, '"addq      %%rdi, %%{2}                      \\n\\t" // c += cs_c;' )
#
#
#        for idx in range(4):
#            myfile.write( \
#'''\
#    "vextractf128 $1, %%{0},  %%xmm{6}            \\n\\t"
#    "vmovlpd    (%%{2}),       %%xmm{5},  %%xmm{5}   \\n\\t" // load c{8}_4{1} and c{8}_5{1},
#    "vmovhpd    (%%{2},%%rsi), %%xmm{5},  %%xmm{5}   \\n\\t"
#    "vmulpd           %%xmm{4},  %%xmm{3},  %%xmm{7}   \\n\\t" // scale by coeff{8},
#    "vaddpd           %%xmm{7},  %%xmm{5},  %%xmm{7}   \\n\\t" // add the gemm result,
#    "vmovlpd          %%xmm{7},  (%%{2})           \\n\\t" // and store back to memory.
#    "vmovhpd          %%xmm{7},  (%%{2},%%rsi)     \\n\\t"
#    "vmovlpd    (%%{2},%%r12), %%xmm{5},  %%xmm{5}   \\n\\t" // load c{8}_6{1} and c{8}_7{1},
#    "vmovhpd    (%%{2},%%r13), %%xmm{5},  %%xmm{5}   \\n\\t"
#    "vmulpd           %%xmm{8},  %%xmm{6},  %%xmm{7}   \\n\\t" // scale by coeff{8},
#    "vaddpd           %%xmm{7},  %%xmm{5},  %%xmm{7}   \\n\\t" // add the gemm result,
#    "vmovlpd          %%xmm{7},  (%%{2},%%r12)     \\n\\t" // and store back to memory.
#    "vmovhpd          %%xmm{7},  (%%{2},%%r13)     \\n\\t"
#    "addq      %%rdi, %%{2}                      \\n\\t" // c += cs_c;
#    "                                            \\n\\t"
#'''.format( c47_ymm_list[idx], str(idx), c47_reg, c47_ymm_list[idx][3:], coeff_avx_reg[3:], (get_avx_reg(coeff_avx_reg))[3:], (get_avx_reg(coeff_avx_reg))[3:], (get_avx_reg(coeff_avx_reg))[3:], j ) )
##format( c47_ymm_list[idx], str(idx), c47_reg, c47_ymm_list[idx][3:], j, coeff_avx_reg[3:]  ) )
#
#
#    write_line( myfile, 1, '"                                            \\n\\t"' )
#    write_line( myfile, 1, '"jmp    .DDONE                               \\n\\t" // jump to end.' )
#    write_line( myfile, 1, '"                                            \\n\\t"' )
#


def write_updatec_colstored_assembly( myfile, nnz ):
    write_line( myfile, 1, '".DCOLSTORED:                                \\n\\t"' )
    write_line( myfile, 1, '"                                            \\n\\t"' )
    for j in range( nnz ):
        coeff_avx_reg = get_avx_reg()

        myfile.write( \
'''\
    "                                            \\n\\t"
    "movq         %[coeff{0}], %%{1}               \\n\\t" // load address of coeff{0}
    "                                            \\n\\t"
	"vbroadcastsd    (%%{1}), %%{2}             \\n\\t" // load coeff{0} and duplicate
    "                                            \\n\\t"
'''.format( j, get_reg(), coeff_avx_reg ) )
        #"leaq   (%%rcx,%%rsi,4), %%r10               \\n\\t" // load address of c{0} + 4*rs_c;'


        c03_ymm_list = ['ymm9', 'ymm11', 'ymm13', 'ymm15'] #c00:c33
        c47_ymm_list = ['ymm8', 'ymm10', 'ymm12', 'ymm14'] #c40:c73

        for idx in range(4):
            myfile.write( \
'''\
    "vmovapd    0 * 32(%%{3}),  %%{5}           \\n\\t" // {5} = c{0}( 0:3, 0 )
	"vmulpd            %%{4},  %%{1},  %%{6}  \\n\\t" // scale by coeff{0}, {6} = {4}( coeff{0} ) * {1}( c{0}( 0:3, 0 ) )
    "vaddpd            %%{5},  %%{6},  %%{5}  \\n\\t" // {5} += {6}
    "vmovapd           %%{5},  0(%%{3})         \\n\\t" // c{0}( 0:3, 0 ) = {5}
    "vmovapd    1 * 32(%%{3}),  %%{7}           \\n\\t" // {7} = c{0}( 4:7, 0 )
	"vmulpd            %%{4},  %%{2},  %%{8}  \\n\\t" // scale by coeff{0}, {8} = {4}( coeff{0} ) * {2}( c{0}( 4:7, 0 ) )
    "vaddpd            %%{7},  %%{8},  %%{7}  \\n\\t" // {7} += {8}
    "vmovapd           %%{7},  32(%%{3})        \\n\\t" // c{0}( 4:7, 0 ) = {7}
'''.format(j, c03_ymm_list[idx], c47_ymm_list[idx], get_reg.c2reg[j], coeff_avx_reg, get_avx_reg(coeff_avx_reg), get_avx_reg(coeff_avx_reg), get_avx_reg(coeff_avx_reg), get_avx_reg(coeff_avx_reg) ) )
            if ( idx != 3 ):
                write_line( myfile, 1, '"addq              %%rdi,   %%{0}            \\n\\t"'.format( get_reg.c2reg[j] ) )



    #"addq              $1 * 8,  %%rax            \\n\\t" // alpha_list += 8


def write_function_name( myfile, number ):
    myfile.write( \
'''\
#include "blis.h"

void bli_dstra_{0}_asm_8x4(
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
{{
'''.format( number ) )

def write_common_start_assembly( myfile, nnz ):
    myfile.write( \
'''\
    void*   b_next = bli_auxinfo_next_b( data );

    uint64_t k_iter = k / 4;
    uint64_t k_left = k % 4;
''' )
    
    add = 'double '
    add += ', '.join( [ '*coeff%d = &coeff_list[%d]' % ( i, i ) for i in range( nnz ) ] )
    add += ';'
    write_line( myfile, 1, add )

    add = 'double '
    add += ', '.join( [ '*c%d = c_list[%d]' % ( i, i ) for i in range( nnz ) ] )
    add += ';'
    write_line( myfile, 1, add )

    write_break( myfile )

    myfile.write( \
'''\
	__asm__ volatile
	(
	"                                            \\n\\t"
	"                                            \\n\\t"
    "movq                %[a], %%rax             \\n\\t" // load address of a.              ( v )
    "movq                %[b], %%rbx             \\n\\t" // load address of b.              ( v )
    "movq                %[b_next], %%r15        \\n\\t" // load address of b_next.         ( v )
    "addq          $-4 * 64, %%r15               \\n\\t" //                                 ( ? )
    "                                            \\n\\t"
    "vmovapd   0 * 32(%%rax), %%ymm0             \\n\\t" // initialize loop by pre-loading
    "vmovapd   0 * 32(%%rbx), %%ymm2             \\n\\t" // elements of a and b.
    "vpermilpd  $0x5, %%ymm2, %%ymm3             \\n\\t"
    "                                            \\n\\t"
    "                                            \\n\\t"
    "movq                %[cs_c], %%rdi          \\n\\t" // load cs_c
    "leaq        (,%%rdi,8), %%rdi               \\n\\t" // cs_c * sizeof(double)
''' )


def getNumberName( number ):
    if ( number == 1 ):
        return "one"
    elif ( number == 2 ):
        return "two"
    elif ( number == 3 ):
        return "three"
    elif ( number == 4 ):
        return "four"
    else:
        return ""


def write_prefetch_assembly( myfile, nnz ):
    for j in range( nnz ):
        myfile.write( \
'''\
    "movq            %[c{0}], %%{1}               \\n\\t" // load address of c{0}
    "leaq   (%%{1},%%rdi,2), %%{2}               \\n\\t" // load address of c{0} + 2 * ldc;
    "prefetcht0   3 * 8(%%{1})                   \\n\\t" // prefetch c{0} + 0 * ldc
    "prefetcht0   3 * 8(%%{1},%%rdi)             \\n\\t" // prefetch c{0} + 1 * ldc
    "prefetcht0   3 * 8(%%{2})                   \\n\\t" // prefetch c{0} + 2 * ldc
    "prefetcht0   3 * 8(%%{2},%%rdi)             \\n\\t" // prefetch c{0} + 3 * ldc
'''.format( str(j), get_reg(cno=j), get_reg() ) )


def write_common_rankk_assembly( myfile ):
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
	"movq      %[k_iter], %%rsi                  \\n\\t" // i = k_iter;                     ( v )
	"testq  %%rsi, %%rsi                         \\n\\t" // check i via logical AND.        ( v )
	"je     .DCONSIDKLEFT                        \\n\\t" // if i == 0, jump to code that    ( v )
	"                                            \\n\\t" // contains the k_left loop.
	"                                            \\n\\t"
	"                                            \\n\\t"
	".DLOOPKITER:                                \\n\\t" // MAIN LOOP
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
	"jne    .DLOOPKITER                          \\n\\t" // iterate again if i != 0.
	"                                            \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t"
	".DCONSIDKLEFT:                              \\n\\t"
	"                                            \\n\\t"
	"movq      %[k_left], %%rsi                  \\n\\t" // i = k_left;
	"testq  %%rsi, %%rsi                         \\n\\t" // check i via logical AND.
	"je     .DPOSTACCUM                          \\n\\t" // if i == 0, we're done; jump to end.
	"                                            \\n\\t" // else, we prepare to enter k_left loop.
	"                                            \\n\\t"
	"                                            \\n\\t"
	".DLOOPKLEFT:                                \\n\\t" // EDGE LOOP
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
	"jne    .DLOOPKLEFT                          \\n\\t" // iterate again if i != 0.
	"                                            \\n\\t"
	"                                            \\n\\t"
	"                                            \\n\\t"
	".DPOSTACCUM:                                \\n\\t"
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
''' )



def write_common_end_assembly( myfile, nnz ):
    write_line( myfile, 1, '"                                            \\n\\t"' )
    write_line( myfile, 1, '".DDONE:                                    \\n\\t"' )
    write_line( myfile, 1, '"                                            \\n\\t"' )
    write_line( myfile, 1, ': // output operands (none)' )
    write_line( myfile, 1, ': // input operands' )
    write_line( myfile, 1, '  [k_iter]     "m" (k_iter),      // 0' )
    write_line( myfile, 1, '  [k_left]     "m" (k_left),      // 1' )
    write_line( myfile, 1, '  [a]          "m" (a),           // 2' )
    write_line( myfile, 1, '  [b]          "m" (b),           // 3' )
    write_line( myfile, 1, '  [b_next]     "m" (b_next),      // 4' )
    write_line( myfile, 1, '  [rs_c]       "m" (rs_c),        // 5' )
    write_line( myfile, 1, '  [cs_c]       "m" (cs_c),        // 6' )

    add = ''
    add += '\n    '.join( [ '  [c%d]         "m" (c%d)           // %d' % ( i, i, i+7 ) for i in range( nnz ) ] )

    add += '\n    '
    add += '\n    '.join( [ '  [coeff%d]     "m" (coeff%d)       // %d' % ( i, i,  i+7+nnz ) for i in range( nnz ) ] )

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

def write_updatec_header_branch_assembly( myfile ):
    myfile.write( \
'''\
    "                                            \\n\\t"
    "movq         %[rs_c], %%rsi                 \\n\\t" // load rs_c
    "                                            \\n\\t"
    "leaq        (,%%rsi,8), %%rsi               \\n\\t" // rsi = rs_c * sizeof(double)
    "                                            \\n\\t"
    "                                            \\n\\t"
    "                                            \\n\\t" // determine if
    "                                            \\n\\t" //    c    % 32 == 0, AND
    "                                            \\n\\t" //  8*cs_c % 32 == 0, AND
    "                                            \\n\\t" //    rs_c      == 1
    "                                            \\n\\t" // ie: aligned, ldim aligned, and
    "                                            \\n\\t" // column-stored
    "                                            \\n\\t"
    "cmpq       $8, %%rsi                        \\n\\t" // set ZF if (8*rs_c) == 8.
    "sete           %%bl                         \\n\\t" // bl = ( ZF == 1 ? 1 : 0 );
    "testq     $31, %%{0}                        \\n\\t" // set ZF if c_list[ 0 ] & 32 is zero.
    "setz           %%bh                         \\n\\t" // bh = ( ZF == 0 ? 1 : 0 );
    "testq     $31, %%rdi                        \\n\\t" // set ZF if (8*cs_c) & 32 is zero.
    "setz           %%al                         \\n\\t" // al = ( ZF == 0 ? 1 : 0 );
    "                                            \\n\\t" // and(bl,bh) followed by
    "                                            \\n\\t" // and(bh,al) will reveal result
    "                                            \\n\\t"
    //"jmp     .DCOLSTORED                         \\n\\t" // jump to column storage case
    //"jmp     .DGENSTORED                         \\n\\t" // jump to column storage case
    "                                            \\n\\t"
    "                                            \\n\\t" // now avoid loading C if beta == 0
    "                                            \\n\\t"
//    "vxorpd    %%ymm0,  %%ymm0,  %%ymm0          \\n\\t" // set ymm0 to zero.
//    "vucomisd  %%xmm0,  %%xmm2                   \\n\\t" // set ZF if beta == 0.
//    "je      .DBETAZERO                          \\n\\t" // if ZF = 1, jump to beta == 0 case
    "                                            \\n\\t"
    "                                            \\n\\t" // check if aligned/column-stored
    "andb     %%bl, %%bh                         \\n\\t" // set ZF if bl & bh == 1.
    "andb     %%bh, %%al                         \\n\\t" // set ZF if bh & al == 1.
    "jne     .DCOLSTORED                         \\n\\t" // jump to column storage case
    "                                            \\n\\t"
'''.format( get_reg.c2reg[0] ) )

def write_updatec_assembly( myfile, nnz ):
    write_updatec_header_branch_assembly( myfile )
    get_reg.reg_pool.extend( [ 'rax', 'rbx' ] )
    write_updatec_genstored_assembly( myfile, nnz )
    get_reg.reg_pool.extend( [ 'r12', 'r13' ] )
    write_updatec_colstored_assembly( myfile, nnz )


def gen_micro_kernel( outfile, nnz ):

    myfile = open( outfile, 'w' ) 
    #nonzero_coeffs=['1','-1']

    #gen_updatec_assembly( myfile )

    write_function_name( myfile, getNumberName(nnz) )
    write_common_start_assembly( myfile, nnz )

    write_prefetch_assembly( myfile, nnz )

    #write_line( myfile, 1, 'RANKK_UPDATE( %d )' % index )
    #write_common_rankk_assembly( myfile, index )
    #write_common_simple_rankk_assembly( myfile, index )

    write_common_rankk_assembly( myfile )

    write_updatec_assembly( myfile, nnz )
    
    write_common_end_assembly( myfile, nnz )

    write_line( myfile, 0, '}' )

    #write_break( myfile )

def main():
    try:
        outfile = sys.argv[1]
        nnz = int( sys.argv[2] )
    except:
        raise Exception('USAGE: python kernel_gen.py out_file nnz')

    gen_micro_kernel( outfile, nnz )

    print get_reg.c2reg


if __name__ == '__main__':

    main()

