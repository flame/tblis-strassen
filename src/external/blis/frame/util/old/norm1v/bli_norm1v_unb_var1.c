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

#define FUNCPTR_T norm1v_fp

typedef void (*FUNCPTR_T)(
                           dim_t  n,
                           void*  x, inc_t incx,
                           void*  norm
                         );

static FUNCPTR_T GENARRAY(ftypes,norm1v_unb_var1);


void bli_norm1v_unb_var1( obj_t*  x,
                          obj_t*  norm )
{
	num_t     dt_x     = bli_obj_datatype( *x );

	dim_t     n        = bli_obj_vector_dim( *x );

	inc_t     inc_x    = bli_obj_vector_inc( *x );
	void*     buf_x    = bli_obj_buffer_at_off( *x );

	void*     buf_norm = bli_obj_buffer_at_off( *norm );

	FUNCPTR_T f;

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_x];

	// Invoke the function.
	f( n,
	   buf_x, inc_x,
	   buf_norm );
}


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname)( \
                           dim_t  n, \
                           void*  x, inc_t incx, \
                           void*  norm  \
                         ) \
{ \
	ctype*   x_cast    = x; \
	ctype_r* norm_cast = norm; \
	ctype*   chi1; \
	ctype_r  abs_chi1; \
	ctype_r  absum; \
	dim_t    i; \
\
	/* NOTE: Early returns due to empty dimensions are handled by the
	   caller. */ \
\
	/* Initialize the absolute sum accumulator to zero. */ \
	PASTEMAC(chr,set0s)( absum ); \
\
	for ( i = 0; i < n; ++i ) \
	{ \
		chi1 = x_cast + (i  )*incx; \
\
		/* Compute the absolute value (or complex magnitude) of chi1. */ \
		PASTEMAC2(ch,chr,abval2s)( *chi1, abs_chi1 ); \
\
		/* Accumulate the absolute value of chi1 into absum. */ \
		PASTEMAC(chr,adds)( abs_chi1, absum ); \
	} \
\
	/* Store final value of absum to the output variable. */ \
	PASTEMAC(chr,copys)( absum, *norm_cast ); \
}

INSERT_GENTFUNCR_BASIC0( norm1v_unb_var1 )

