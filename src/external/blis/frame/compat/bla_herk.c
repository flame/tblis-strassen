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

#include "blis.h"


//
// Define BLAS-to-BLIS interfaces.
//
#undef  GENTFUNCCO
#define GENTFUNCCO( ftype, ftype_r, ch, chr, blasname, blisname ) \
\
void PASTEF77(ch,blasname) \
     ( \
       const f77_char* uploc, \
       const f77_char* transa, \
       const f77_int*  m, \
       const f77_int*  k, \
       const ftype_r*  alpha, \
       const ftype*    a, const f77_int* lda, \
       const ftype_r*  beta, \
             ftype*    c, const f77_int* ldc  \
     ) \
{ \
	uplo_t  blis_uploc; \
	trans_t blis_transa; \
	dim_t   m0, k0; \
	inc_t   rs_a, cs_a; \
	inc_t   rs_c, cs_c; \
	err_t   init_result; \
\
	/* Initialize BLIS (if it is not already initialized). */ \
	bli_init_auto( &init_result ); \
\
	/* Perform BLAS parameter checking. */ \
	PASTEBLACHK(blasname) \
	( \
	  MKSTR(ch), \
	  MKSTR(blasname), \
	  uploc, \
	  transa, \
	  m, \
	  k, \
	  lda, \
	  ldc  \
	); \
\
	/* Map BLAS chars to their corresponding BLIS enumerated type value. */ \
	bli_param_map_netlib_to_blis_uplo( *uploc, &blis_uploc ); \
	bli_param_map_netlib_to_blis_trans( *transa, &blis_transa ); \
\
	/* Convert/typecast negative values of m and k to zero. */ \
	bli_convert_blas_dim1( *m, m0 ); \
	bli_convert_blas_dim1( *k, k0 ); \
\
	/* We emulate the BLAS early return behavior with the following
	   conditional, which returns if one of the following is true:
	   - matrix C is empty
	   - the rank-k product is empty (either because alpha is zero or k
	     is zero) AND matrix C is not scaled. */ \
	if ( m0 == 0 || \
	     ( ( PASTEMAC(chr,eq0)( *alpha ) || k0 == 0 ) \
	       && PASTEMAC(chr,eq1)( *beta ) \
         ) \
	   ) \
	{ \
		/* Finalize BLIS (if it was initialized above). */ \
		bli_finalize_auto( init_result ); \
\
		return; \
	} \
\
	/* Set the row and column strides of the matrix operands. */ \
	rs_a = 1; \
	cs_a = *lda; \
	rs_c = 1; \
	cs_c = *ldc; \
\
	/* Call BLIS interface. */ \
	PASTEMAC(ch,blisname) \
	( \
	  blis_uploc, \
	  blis_transa, \
	  m0, \
	  k0, \
	  (ftype_r*)alpha, \
	  (ftype*)a, rs_a, cs_a, \
	  (ftype_r*)beta, \
	  (ftype*)c, rs_c, cs_c, \
	  NULL  \
	); \
\
	/* Finalize BLIS (if it was initialized above). */ \
	bli_finalize_auto( init_result ); \
}

#ifdef BLIS_ENABLE_BLAS2BLIS
INSERT_GENTFUNCCO_BLAS( herk, herk )
#endif

