/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef BLIS_KERNEL_H
#define BLIS_KERNEL_H


// -- LEVEL-3 MICRO-KERNEL CONSTANTS AND DEFINITIONS ---------------------------

//
// Constraints:
//
// (1) MC must be a multiple of:
//     (a) MR (for zero-padding purposes)
//     (b) NR (for zero-padding purposes when MR and NR are "swapped")
// (2) NC must be a multiple of
//     (a) NR (for zero-padding purposes)
//     (b) MR (for zero-padding purposes when MR and NR are "swapped")
//

#if 0

#define BLIS_SGEMM_UKERNEL         bli_sgemm_asm_24x4
#define BLIS_DEFAULT_MC_S          264
#define BLIS_DEFAULT_KC_S          128
#define BLIS_DEFAULT_NC_S          4080
#define BLIS_DEFAULT_MR_S          24
#define BLIS_DEFAULT_NR_S          4

#else

/*
#define BLIS_SGEMM_UKERNEL         bli_sgemm_asm_16x6
#define BLIS_DEFAULT_MC_S          144
#define BLIS_DEFAULT_KC_S          256
#define BLIS_DEFAULT_NC_S          4080
#define BLIS_DEFAULT_MR_S          16
#define BLIS_DEFAULT_NR_S          6
*/

#define BLIS_SGEMM_UKERNEL         bli_sgemm_asm_6x16
#define BLIS_DEFAULT_MC_S          144
#define BLIS_DEFAULT_KC_S          256
#define BLIS_DEFAULT_NC_S          4080
#define BLIS_DEFAULT_MR_S          6
#define BLIS_DEFAULT_NR_S          16

#define BLIS_SGEMM_UKERNEL_PREFERS_CONTIG_ROWS

#endif

#if 0

#define BLIS_DGEMM_UKERNEL         bli_dgemm_asm_12x4
#define BLIS_DEFAULT_MC_D          96
#define BLIS_DEFAULT_KC_D          192
#define BLIS_DEFAULT_NC_D          4080
#define BLIS_DEFAULT_MR_D          12
#define BLIS_DEFAULT_NR_D          4

#else

/*
#define BLIS_DGEMM_UKERNEL         bli_dgemm_asm_8x6
#define BLIS_DEFAULT_MC_D          72
#define BLIS_DEFAULT_KC_D          256
#define BLIS_DEFAULT_NC_D          4080
#define BLIS_DEFAULT_MR_D          8
#define BLIS_DEFAULT_NR_D          6
*/

#define BLIS_DGEMM_UKERNEL         bli_dgemm_asm_6x8
#define BLIS_DEFAULT_MC_D          72
#define BLIS_DEFAULT_KC_D          256
#define BLIS_DEFAULT_NC_D          4080
#define BLIS_DEFAULT_MR_D          6
#define BLIS_DEFAULT_NR_D          8

#define BLIS_DGEMM_UKERNEL_PREFERS_CONTIG_ROWS


#endif





// -- LEVEL-2 KERNEL CONSTANTS -------------------------------------------------




// -- LEVEL-1F KERNEL CONSTANTS ------------------------------------------------




// -- LEVEL-1M KERNEL DEFINITIONS ----------------------------------------------

// -- packm --

// -- unpackm --




// -- LEVEL-1F KERNEL DEFINITIONS ----------------------------------------------

// -- axpy2v --

// -- dotaxpyv --

// -- axpyf --

// -- dotxf --

// -- dotxaxpyf --




// -- LEVEL-1V KERNEL DEFINITIONS ----------------------------------------------

// -- addv --

// -- axpyv --

// -- copyv --

// -- dotv --

// -- dotxv --

// -- invertv --

// -- scal2v --

// -- scalv --

// -- setv --

// -- subv --

// -- swapv --



#endif

