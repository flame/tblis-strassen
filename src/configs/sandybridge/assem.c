



    "vextractf128 $1, %%{0},  %%xmm1            \n\t"
    "vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load c00 and c10,
    "vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t"
    "vmulpd           %%xmm6,  %%xmm9,  %%xmm2   \n\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \n\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%rcx)           \n\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%rcx,%%rsi)     \n\t"
    "vmovlpd    (%%rcx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c20 and c30,
    "vmovhpd    (%%rcx,%%r13), %%xmm0,  %%xmm0   \n\t"
    "vmulpd           %%xmm6,  %%xmm1,  %%xmm2   \n\t" // scale by alpha,
    "vaddpd           %%xmm2,  %%xmm0,  %%xmm2   \n\t" // add the gemm result,
    "vmovlpd          %%xmm2,  (%%rcx,%%r12)     \n\t" // and store back to memory.
    "vmovhpd          %%xmm2,  (%%rcx,%%r13)     \n\t"
    "addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
    "                                            \n\t"


    "vextractf128 $1, %%{0},  %%xmm1            \n\t"
    "vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load c40 and c50,
    "vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t"
    "vmulpd           %%xmm6,  %%xmm9,  %%xmm2   \n\t" // scale by alpha,
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





{0}: ymm9

{0}: ymm8


alpha_list -> xmm6: by broadcast; need to move pointer in the end of N loop;



    "movq                %{0}, %%rsi               \n\t" // load rs_c
    "leaq        (,%%rsi,8), %%rsi               \n\t" // rsi = rs_c * sizeof(double)
    "                                            \n\t"
    "leaq   (%%rcx,%%rsi,4), %%rdx               \n\t" // load address of c + 4*rs_c;
    "                                            \n\t"
    "leaq        (,%%rsi,2), %%r12               \n\t" // r12 = 2*rs_c;
    "leaq   (%%r12,%%rsi,1), %%r13               \n\t" // r13 = 3*rs_c;
    "                                            \n\t"
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
    "testq     $31, %%rcx                        \n\t" // set ZF if c & 32 is zero.
    "setz           %%bh                         \n\t" // bh = ( ZF == 0 ? 1 : 0 );
    "testq     $31, %%rdi                        \n\t" // set ZF if (8*cs_c) & 32 is zero.
    "setz           %%al                         \n\t" // al = ( ZF == 0 ? 1 : 0 );
    "                                            \n\t" // and(bl,bh) followed by
    "                                            \n\t" // and(bh,al) will reveal result
    "                                            \n\t"
    "                                            \n\t" // now avoid loading C if beta == 0
    "                                            \n\t"
//    "vxorpd    %%ymm0,  %%ymm0,  %%ymm0          \n\t" // set ymm0 to zero.
//    "vucomisd  %%xmm0,  %%xmm2                   \n\t" // set ZF if beta == 0.
//    "je      .DBETAZERO                          \n\t" // if ZF = 1, jump to beta == 0 case
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // check if aligned/column-stored
    "andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
    "andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
    "jne     .DCOLSTORED                         \n\t" // jump to column storage case
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".DGENSTORED:                                \n\t"


{0}: rs_c

get_reg.reg_pool = [ 'rcx', 'rdx', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14' ]
#get_reg.reg_pool = [ 'rcx', 'rdx', 'rsi', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14' ]
# rdi, rax, rbx, r15, already occupied.
# (rcx, rdx, rsi, r8, r9, r10, r11, r12, r13, r14): register allocation algorithm





	"                                            \n\t"
    "movq         %{0}, %%rax                      \n\t" // load address of alpha_list[ 0 ]
    "movq         %{1}, %%rcx                      \n\t" // load address of c_list[ 0 ]
	"                                            \n\t"
    "                                            \n\t"
	"movq      %{2}, %%rsi                         \n\t" // i = len;                        ( v )
    "                                            \n\t"






    "                                            \n\t"
    ".DSTORELOOP:                                \n\t"
    "                                            \n\t"
	"movq       0 * 8(%%rcx),  %%rdx             \n\t" // rdx = c_list[ i ] ( address )
    "                                            \n\t"
	//"movq       0 * 8(%%rax),  %%rbx             \n\t" // load address of alpha_list[ i ]
	//"vbroadcastsd    (%%rbx), %%ymm6             \n\t" // load alpha_list[ 1 ] and duplicate
	"vbroadcastsd    (%%rax), %%ymm6             \n\t" // load alpha_list[ i ] and duplicate
    "                                            \n\t"









