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

static hemv_vft GENARRAY(ftypes,hemv_unf_var3);

void bli_hemv_unf_var3( conj_t  conjh,
                        obj_t*  alpha,
                        obj_t*  a,
                        obj_t*  x,
                        obj_t*  beta,
                        obj_t*  y,
                        cntx_t* cntx,
                        hemv_t* cntl )
{
	num_t     dt_a      = bli_obj_datatype( *a );
	num_t     dt_x      = bli_obj_datatype( *x );
	num_t     dt_y      = bli_obj_datatype( *y );

	uplo_t    uplo      = bli_obj_uplo( *a );
	conj_t    conja     = bli_obj_conj_status( *a );
	conj_t    conjx     = bli_obj_conj_status( *x );

	dim_t     m         = bli_obj_length( *a );

	void*     buf_a     = bli_obj_buffer_at_off( *a );
	inc_t     rs_a      = bli_obj_row_stride( *a );
	inc_t     cs_a      = bli_obj_col_stride( *a );

	void*     buf_x     = bli_obj_buffer_at_off( *x );
	inc_t     incx      = bli_obj_vector_inc( *x );

	void*     buf_y     = bli_obj_buffer_at_off( *y );
	inc_t     incy      = bli_obj_vector_inc( *y );

	num_t     dt_alpha;
	void*     buf_alpha;

	num_t     dt_beta;
	void*     buf_beta;

	FUNCPTR_T f;

	// The datatype of alpha MUST be the type union of a and x. This is to
	// prevent any unnecessary loss of information during computation.
	dt_alpha  = bli_datatype_union( dt_a, dt_x );
	buf_alpha = bli_obj_buffer_for_1x1( dt_alpha, *alpha );

	// The datatype of beta MUST be the same as the datatype of y.
	dt_beta   = dt_y;
	buf_beta  = bli_obj_buffer_for_1x1( dt_beta, *beta );

#if 0
	obj_t x_copy, y_copy;

	bli_obj_create( dt_x, m, 1, 0, 0, &x_copy );
	bli_obj_create( dt_y, m, 1, 0, 0, &y_copy );
	bli_copyv( x, &x_copy );
	bli_copyv( y, &y_copy );
	buf_x = bli_obj_buffer_at_off( x_copy );
	buf_y = bli_obj_buffer_at_off( y_copy );
	incx = 1;
	incy = 1;
#endif

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_a];

	// Invoke the function.
	f( uplo,
	   conja,
	   conjx,
	   conjh,
	   m,
	   buf_alpha,
	   buf_a, rs_a, cs_a,
	   buf_x, incx,
	   buf_beta,
	   buf_y, incy );
#if 0
	bli_copyv( &y_copy, y );
	bli_obj_free( &x_copy );
	bli_obj_free( &y_copy );
#endif
}


#undef  GENTFUNC3U12
#define GENTFUNC3U12( ctype_a, ctype_x, ctype_y, ctype_ax, cha, chx, chy, chax, varname, kername ) \
\
void PASTEMAC(cha,varname) \
     ( \
       uplo_t  uplo, \
       conj_t  conja, \
       conj_t  conjx, \
       conj_t  conjh, \
       dim_t   m, \
       void*   alpha, \
       void*   a, inc_t rs_a, inc_t cs_a, \
       void*   x, inc_t incx, \
       void*   beta, \
       void*   y, inc_t incy, \
       cntx_t* cntx  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype_ax* alpha_cast = alpha; \
	ctype_y*  beta_cast  = beta; \
	ctype_a*  a_cast     = a; \
	ctype_x*  x_cast     = x; \
	ctype_y*  y_cast     = y; \
	ctype_y*  one        = PASTEMAC(chy,1); \
	ctype_y*  zero       = PASTEMAC(chy,0); \
	ctype_a*  A11; \
	ctype_a*  A21; \
	ctype_a*  a10t; \
	ctype_a*  alpha11; \
	ctype_a*  a21; \
	ctype_x*  x1; \
	ctype_x*  x2; \
	ctype_x*  chi11; \
	ctype_y*  y1; \
	ctype_y*  y2; \
	ctype_y*  y01; \
	ctype_y*  psi11; \
	ctype_y*  y21; \
	ctype_x   conjx_chi11; \
	ctype_ax  alpha_chi11; \
	ctype_a   alpha11_temp; \
	dim_t     i, k, j; \
	dim_t     b_fuse, f; \
	dim_t     n_ahead; \
	dim_t     f_ahead, f_behind; \
	inc_t     rs_at, cs_at; \
	conj_t    conj0, conj1; \
\
	if ( bli_zero_dim1( m ) ) return; \
\
	/* The algorithm will be expressed in terms of the lower triangular case;
	   the upper triangular case is supported by swapping the row and column
	   strides of A and toggling some conj parameters. */ \
	if      ( bli_is_lower( uplo ) ) \
	{ \
		rs_at = rs_a; \
		cs_at = cs_a; \
\
		conj0 = bli_apply_conj( conjh, conja ); \
		conj1 = conja; \
	} \
	else /* if ( bli_is_upper( uplo ) ) */ \
	{ \
		rs_at = cs_a; \
		cs_at = rs_a; \
\
		conj0 = conja; \
		conj1 = bli_apply_conj( conjh, conja ); \
	} \
\
	/* If beta is zero, use setv. Otherwise, scale by beta. */ \
	if ( PASTEMAC(cha,eq0)( *beta_cast ) ) \
	{ \
		/* y = 0; */ \
		PASTEMAC(cha,setv) \
		( \
		  BLIS_NO_CONJUGATE, \
		  m, \
		  zero, \
		  y_cast, incy, \
		  cntx  \
		); \
	} \
	else \
	{ \
		/* y = beta * y; */ \
		PASTEMAC(cha,scalv) \
		( \
		  BLIS_NO_CONJUGATE, \
		  m, \
		  beta_cast, \
		  y_cast, incy, \
		  cntx  \
		); \
	} \
\
	/* Query the fusing factor for the dotxaxpyf implementation. */ \
	b_fuse = PASTEMAC(chax,dotxaxpyf_fusefac); \
\
	for ( i = 0; i < m; i += f ) \
	{ \
		f        = bli_determine_blocksize_dim_f( i, m, b_fuse ); \
		n_ahead  = m - i - f; \
		A11      = a_cast + (i  )*rs_at + (i  )*cs_at; \
		A21      = a_cast + (i+f)*rs_at + (i  )*cs_at; \
		x1       = x_cast + (i  )*incx; \
		x2       = x_cast + (i+f)*incx; \
		y1       = y_cast + (i  )*incy; \
		y2       = y_cast + (i+f)*incy; \
\
		/* y1 = y1 + alpha * A11 * x1;  (variant 4) */ \
		for ( k = 0; k < f; ++k ) \
		{ \
			f_behind = k; \
			f_ahead  = f - k - 1; \
			a10t     = A11 + (k  )*rs_at + (0  )*cs_at; \
			alpha11  = A11 + (k  )*rs_at + (k  )*cs_at; \
			a21      = A11 + (k+1)*rs_at + (k  )*cs_at; \
			chi11    = x1  + (k  )*incx; \
			y01      = y1  + (0  )*incy; \
			psi11    = y1  + (k  )*incy; \
			y21      = y1  + (k+1)*incy; \
\
			/* y01 = y01 + alpha * a10t' * chi11; */ \
			PASTEMAC2(chx,chx,copycjs)( conjx, *chi11, conjx_chi11 ); \
			PASTEMAC3(chax,chx,chax,scal2s)( *alpha_cast, conjx_chi11, alpha_chi11 ); \
			if ( bli_is_conj( conj0 ) ) \
			{ \
				for ( j = 0; j < f_behind; ++j ) \
					PASTEMAC3(chax,cha,chy,axpyjs)( alpha_chi11, *(a10t + j*cs_at), *(y01 + j*incy) ); \
			} \
			else \
			{ \
				for ( j = 0; j < f_behind; ++j ) \
					PASTEMAC3(chax,cha,chy,axpys)( alpha_chi11, *(a10t + j*cs_at), *(y01 + j*incy) ); \
			} \
\
			/* For hemv, explicitly set the imaginary component of alpha11 to
			   zero. */ \
			PASTEMAC2(cha,cha,copycjs)( conja, *alpha11, alpha11_temp ); \
			if ( bli_is_conj( conjh ) ) \
				PASTEMAC(cha,seti0s)( alpha11_temp ); \
\
			/* psi11 = psi11 + alpha * alpha11 * chi11; */ \
			PASTEMAC3(chax,cha,chy,axpys)( alpha_chi11, alpha11_temp, *psi11 ); \
\
			/* y21 = y21 + alpha * a21 * chi11; */ \
			if ( bli_is_conj( conj1 ) ) \
			{ \
				for ( j = 0; j < f_ahead; ++j ) \
					PASTEMAC3(chax,cha,chy,axpyjs)( alpha_chi11, *(a21 + j*rs_at), *(y21 + j*incy) ); \
			} \
			else \
			{ \
				for ( j = 0; j < f_ahead; ++j ) \
					PASTEMAC3(chax,cha,chy,axpys)( alpha_chi11, *(a21 + j*rs_at), *(y21 + j*incy) ); \
			} \
		} \
\
		/* y1 = y1 + alpha * A21' * x2;  (dotxf) */ \
		/* y2 = y2 + alpha * A21  * x1;  (axpyf) */ \
		PASTEMAC(cha,kername) \
		( \
		  conj0, \
		  conj1, \
		  conjx, \
		  conjx, \
		  n_ahead, \
		  f, \
		  alpha_cast, \
		  A21, rs_at, cs_at, \
		  x2,  incx, \
		  x1,  incx, \
		  one, \
		  y1,  incy, \
		  y2,  incy, \
		  cntx  \
		); \
	} \
}

// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
INSERT_GENTFUNC3U12_BASIC( hemv_unf_var3, DOTXAXPYF_KERNEL )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC3U12_MIX_D( hemv_unf_var3, DOTXAXPYF_KERNEL )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC3U12_MIX_P( hemv_unf_var3, DOTXAXPYF_KERNEL )
#endif

