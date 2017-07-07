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

void bli_trsm( side_t  side,
               obj_t*  alpha,
               obj_t*  a,
               obj_t*  b )
{
	// The operation's "ind" function--its induced method front-end--will
	// call native execution for real domain problems. For complex problems,
	// it calls the highest priority induced method that is available (ie:
	// implemented and enabled), and otherwise calls native execution.
	bli_trsmind( side, alpha, a, b );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname)( \
                          side_t  side, \
                          uplo_t  uploa, \
                          trans_t transa, \
                          diag_t  diaga, \
                          dim_t   m, \
                          dim_t   n, \
                          ctype*  alpha, \
                          ctype*  a, inc_t rs_a, inc_t cs_a, \
                          ctype*  b, inc_t rs_b, inc_t cs_b  \
                        ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	obj_t       alphao, ao, bo; \
\
	dim_t       mn_a; \
\
	bli_set_dim_with_side( side, m, n, mn_a ); \
\
	bli_obj_create_1x1_with_attached_buffer( dt, alpha, &alphao ); \
\
	bli_obj_create_with_attached_buffer( dt, mn_a, mn_a, a, rs_a, cs_a, &ao ); \
	bli_obj_create_with_attached_buffer( dt, m,    n,    b, rs_b, cs_b, &bo ); \
\
	bli_obj_set_uplo( uploa, ao ); \
	bli_obj_set_diag( diaga, ao ); \
	bli_obj_set_conjtrans( transa, ao ); \
\
	bli_obj_set_struc( BLIS_TRIANGULAR, ao ); \
\
	PASTEMAC0(opname)( side, \
	                   &alphao, \
	                   &ao, \
	                   &bo ); \
}

INSERT_GENTFUNC_BASIC0( trsm )

