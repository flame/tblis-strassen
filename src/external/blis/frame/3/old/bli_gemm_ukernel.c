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

#define FUNCPTR_T gemm_ukr_fp

typedef void (*FUNCPTR_T)(
                           dim_t      k,
                           void*      alpha,
                           void*      a,
                           void*      b,
                           void*      beta,
                           void*      c, inc_t rs_c, inc_t cs_c,
                           auxinfo_t* data,
                           cntx_t*    cntx,
                           void*      ukr
                         );

static FUNCPTR_T GENARRAY(ftypes,gemm_ukernel_void);

void bli_gemm_ukernel( obj_t*  alpha,
                       obj_t*  a,
                       obj_t*  b,
                       obj_t*  beta,
                       obj_t*  c,
                       cntx_t* cntx )
{
	num_t     dt        = bli_obj_datatype( *c );

	dim_t     k         = bli_obj_width( *a );

	void*     buf_a     = bli_obj_buffer_at_off( *a );

	void*     buf_b     = bli_obj_buffer_at_off( *b );

	void*     buf_c     = bli_obj_buffer_at_off( *c );
	inc_t     rs_c      = bli_obj_row_stride( *c );
	inc_t     cs_c      = bli_obj_col_stride( *c );

	void*     buf_alpha = bli_obj_buffer_for_1x1( dt, *alpha );

	void*     buf_beta  = bli_obj_buffer_for_1x1( dt, *beta );

	auxinfo_t data;

	FUNCPTR_T f;

	void*     gemm_ukr;


	// Fill the auxinfo_t struct in case the micro-kernel uses it.
	bli_auxinfo_set_next_a( buf_a, data );
	bli_auxinfo_set_next_b( buf_b, data );

	bli_auxinfo_set_is_a( 1, data );
	bli_auxinfo_set_is_b( 1, data );

	// Query the function address from the micro-kernel func_t object.
	gemm_ukrs = bli_cntx_get_l3_ukr( BLIS_GEMM_UKR, cntx );
	gemm_ukr  = bli_func_obj_query( dt, gemm_ukrs );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt];

	// Invoke the function.
	f( k,
	   buf_alpha,
	   buf_a,
	   buf_b,
	   buf_beta,
	   buf_c, rs_c, cs_c,
	   &data,
	   gemm_ukr );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname, ukrtype ) \
\
void PASTEMAC(ch,varname)( \
                           dim_t      k, \
                           void*      alpha, \
                           void*      a, \
                           void*      b, \
                           void*      beta, \
                           void*      c, inc_t rs_c, inc_t cs_c, \
                           auxinfo_t* data, \
                           void*      ukr  \
                         ) \
{ \
	/* Cast the micro-kernel address to its function pointer type. */ \
	PASTECH(ch,ukrtype) ukr_cast = ukr; \
\
	ukr_cast( k, \
	          alpha, \
	          a, \
	          b, \
	          beta, \
	          c, rs_c, cs_c, \
	          data ); \
}

INSERT_GENTFUNC_BASIC( gemm_ukernel_void, gemm_ukr_t )

