import sys
from common import is_one, is_negone, is_nonzero, write_line, write_break, transpose, printmat, contain_nontrivial

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


def gen_updatec_assembly( myfile ):
    c03_ymm_list = ['ymm9', 'ymm11', 'ymm13', 'ymm15'] #c00:c33
    c47_ymm_list = ['ymm8', 'ymm10', 'ymm12', 'ymm14'] #c40:c73

    for idx in range(4):
        myfile.write( \
'''\
    "vextractf128 $1, %%{0},  %%xmm1            \\n\\t"
    "vmovlpd    (%%{2}),       %%xmm0,  %%xmm0   \\n\\t" // load c0{1} and c1{1},
    "vmovhpd    (%%{2},%%rsi), %%xmm0,  %%xmm0   \\n\\t"
    "vmulpd           %%xmm6,  %%xmm{3},  %%xmm2   \\n\\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \\n\\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%{2})           \\n\\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%{2},%%rsi)     \\n\\t"
    "vmovlpd    (%%{2},%%r12), %%xmm0,  %%xmm0   \\n\\t" // load c2{1} and c3{1},
    "vmovhpd    (%%{2},%%r13), %%xmm0,  %%xmm0   \\n\\t"
    "vmulpd           %%xmm6,  %%xmm1,  %%xmm2   \\n\\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \\n\\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%{2},%%r12)     \\n\\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%{2},%%r13)     \\n\\t"
    "addq      %%rdi, %%{2}                      \\n\\t" // c += cs_c;
    "                                            \\n\\t"
'''.format( c03_ymm_list[idx], str(idx), 'rbx', c03_ymm_list[idx][3:] ) )


    for idx in range(4):
        myfile.write( \
'''\
    "vextractf128 $1, %%{0},  %%xmm1            \\n\\t"
    "vmovlpd    (%%{2}),       %%xmm0,  %%xmm0   \\n\\t" // load c4{1} and c5{1},
    "vmovhpd    (%%{2},%%rsi), %%xmm0,  %%xmm0   \\n\\t"
    "vmulpd           %%xmm6,  %%xmm{3},  %%xmm2   \\n\\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \\n\\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%{2})           \\n\\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%{2},%%rsi)     \\n\\t"
    "vmovlpd    (%%{2},%%r12), %%xmm0,  %%xmm0   \\n\\t" // load c6{1} and c7{1},
    "vmovhpd    (%%{2},%%r13), %%xmm0,  %%xmm0   \\n\\t"
    "vmulpd           %%xmm6,  %%xmm1,  %%xmm2   \\n\\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \\n\\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%{2},%%r12)     \\n\\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%{2},%%r13)     \\n\\t"
    "addq      %%rdi, %%{2}                      \\n\\t" // c += cs_c;
    "                                            \\n\\t"
'''.format( c47_ymm_list[idx], str(idx), 'rdx', c47_ymm_list[idx][3:] ) )


def write_updatec_assembly( myfile, nonzero_coeffs ):
    nnz = len( nonzero_coeffs )
    write_line( myfile, 1, '"movq         %{0}, %%rax                      \\n\\t" // load address of alpha_list'.format(nnz+6) )


    for j, coeff in enumerate(nonzero_coeffs):
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


def write_updatec_two_assembly( myfile ):
    #nnz = len( nonzero_coeffs )
    nnz = 2

    write_line( myfile, 1, '"movq         %{0}, %%rax                      \\n\\t" // load address of alpha_list'.format(nnz+6) )

    for j in range( nnz ):
    #for j, coeff in enumerate(nonzero_coeffs):
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




def main():
    myfile = open( 'a.c', 'w' ) 
    nonzero_coeffs=['1','-1']
    #write_updatec_assembly( myfile, nonzero_coeffs )
    #gen_updatec_assembly( myfile )

    write_updatec_two_assembly( myfile )

if __name__ == '__main__':
    main()

