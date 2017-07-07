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

typedef void (*FUNCPTR_T)(
                           dim_t  n,
                           void*  x, inc_t incx
                         );

static FUNCPTR_T GENARRAY_MIN(ftypes,invertv_void);


//
// Define object-based interface.
//
void bli_invertv( obj_t*  x )
{
	num_t     dt        = bli_obj_datatype( *x );

	dim_t     n         = bli_obj_vector_dim( *x );

	inc_t     inc_x     = bli_obj_vector_inc( *x );
	void*     buf_x     = bli_obj_buffer_at_off( *x );

	FUNCPTR_T f         = ftypes[dt];

	if ( bli_error_checking_is_enabled() )
	    bli_invertv_check( x );

	// Invoke the void pointer-based function.
	f( n,
	   buf_x, inc_x );
}


//
// Define BLAS-like interfaces with void pointer operands.
//
#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kername ) \
\
void PASTEMAC(ch,opname)( \
                          dim_t  n, \
                          void*  x, inc_t incx  \
                        ) \
{ \
	PASTEMAC(ch,kername)( n, \
	                      x, incx ); \
}

INSERT_GENTFUNC_BASIC( invertv_void, invertv )


//
// Define BLAS-like interfaces with typed operands.
//
#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname)( \
                          dim_t  n, \
                          ctype* x, inc_t incx  \
                        ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
	cntx_t*     cntx; \
\
	PASTECH2(ch,opname,_ker_t) f; \
\
	PASTEMAC(opname,_cntx_init)( &cntx ); \
\
	f = bli_cntx_get_l1v_ker_dt( dt, kerid, &cntx ); \
\
	f( n, \
	   x, incx ); \
\
	PASTEMAC(opname,_cntx_finalize)( &cntx ); \
}

INSERT_GENTFUNC_BASIC( invertv, BLIS_INVERTV_KER )


