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

#ifndef BLIS_SCALAR_MACRO_DEFS_H
#define BLIS_SCALAR_MACRO_DEFS_H



// -- Assignment/Accessor macros --

// NOTE: This macro is defined first since some of the other scalar macros
// use it to abstract away the method used to assign complex values (ie:
// whether fields of a struct are set directly or whether native C99
// assignment is used).

#include "bli_sets.h"    // sets both real and imaginary components

// NOTE: These macros are not used by other scalar macros, but they are
// related to those defined in bli_sets.h, and so we #include them here.

#include "bli_setrs.h"   // sets real component only 
#include "bli_setis.h"   // sets imaginary component only 

// NOTE: This macro also needs to be defined early on since it determines
// how real and imaginary components are accessed (ie: whether the fields
// of a struct are read directly or whether native C99 functions are used.)

#include "bli_gets.h"


// -- Scalar constant initialization macros --

#include "bli_constants.h"


// -- Separated scalar macros (separated real/imaginary values) --

#include "bli_absq2ris.h"

#include "bli_abval2ris.h"

#include "bli_addris.h"
#include "bli_addjris.h"

#include "bli_add3ris.h"

#include "bli_axpbyris.h"
#include "bli_axpbyjris.h"

#include "bli_axpyris.h"
#include "bli_axpyjris.h"

#include "bli_axmyris.h"

#include "bli_conjris.h"

#include "bli_copyris.h"
#include "bli_copyjris.h"
#include "bli_copycjris.h"

#include "bli_eqris.h"

#include "bli_invertris.h"

#include "bli_invscalris.h"
#include "bli_invscaljris.h"

#include "bli_neg2ris.h"

#include "bli_scalris.h"
#include "bli_scaljris.h"
#include "bli_scalcjris.h"

#include "bli_scal2ris.h"
#include "bli_scal2jris.h"

#include "bli_set0ris.h"

#include "bli_sqrt2ris.h"

#include "bli_subris.h"
#include "bli_subjris.h"

#include "bli_swapris.h"

#include "bli_xpbyris.h"
#include "bli_xpbyjris.h"

// Inlined scalar macros in loops
#include "bli_scalris_mxn_uplo.h"


// -- Conventional scalar macros (paired real/imaginary values) --

#include "bli_absq2s.h"

#include "bli_abval2s.h"

#include "bli_adds.h"
#include "bli_addjs.h"

#include "bli_add3s.h"

#include "bli_axpbys.h"
#include "bli_axpbyjs.h"

#include "bli_axpys.h"
#include "bli_axpyjs.h"

#include "bli_axmys.h"

#include "bli_cast.h"

#include "bli_conjs.h"

#include "bli_copys.h"
#include "bli_copyjs.h"
#include "bli_copycjs.h"

#include "bli_dots.h"
#include "bli_dotjs.h"

#include "bli_eq.h"

#include "bli_fprints.h"

#include "bli_inverts.h"

#include "bli_invscals.h"
#include "bli_invscaljs.h"

#include "bli_neg2s.h"

#include "bli_rands.h"
#include "bli_randnp2s.h"

#include "bli_scals.h"
#include "bli_scaljs.h"
#include "bli_scalcjs.h"

#include "bli_scal2s.h"
#include "bli_scal2js.h"

#include "bli_set0s.h"

#include "bli_set1s.h"

#include "bli_seti0s.h"

#include "bli_sqrt2s.h"

#include "bli_subs.h"
#include "bli_subjs.h"

#include "bli_swaps.h"

#include "bli_xpbys.h"
#include "bli_xpbyjs.h"

// Inlined scalar macros in loops
#include "bli_adds_mxn.h"
#include "bli_adds_mxn_uplo.h"
#include "bli_copys_mxn.h"
#include "bli_set0s_mxn.h"
#include "bli_xpbys_mxn.h"
#include "bli_xpbys_mxn_uplo.h"


// -- 3m-specific scalar macros --

#include "bli_copyri3s.h"
#include "bli_copyjri3s.h"

#include "bli_scal2ri3s.h"
#include "bli_scal2jri3s.h"


// -- 4mh/3mh-specific scalar macros --

#include "bli_scal2rihs_mxn_diag.h"
#include "bli_scal2rihs_mxn_uplo.h"
#include "bli_setrihs_mxn_diag.h"

// ro
#include "bli_scal2ros.h"
#include "bli_scal2jros.h"

// io
#include "bli_scal2ios.h"
#include "bli_scal2jios.h"

// rpi
#include "bli_scal2rpis.h"
#include "bli_scal2jrpis.h"



// -- Miscellaneous macros --

// min, max, abs

#define bli_min( a, b )  ( (a) < (b) ? (a) : (b) )
#define bli_max( a, b )  ( (a) > (b) ? (a) : (b) )
#define bli_abs( a )     ( (a) <= 0 ? -(a) : (a) )

// fmin, fmax, fabs

#define bli_fmin( a, b ) bli_min( a, b )
#define bli_fmax( a, b ) bli_max( a, b )
#define bli_fabs( a )    ( (a) <= 0.0 ? -(a) : (a) )

// fminabs, fmaxabs

#define bli_fminabs( a, b ) \
\
	bli_fmin( bli_fabs( a ), \
	          bli_fabs( b ) )

#define bli_fmaxabs( a, b ) \
\
	bli_fmax( bli_fabs( a ), \
	          bli_fabs( b ) )

// round

#define bli_round( val ) \
\
	( round( val ) )

// round_to_mult

#define bli_round_to_mult( val, mult ) \
\
	( guint_t )( ( ( ( guint_t )val + \
	                 ( guint_t )mult / 2 \
	               ) / mult \
	             ) * mult \
	           )

// isnan, isinf

#define bli_isinf( a )  isinf( a )
#define bli_isnan( a )  isnan( a )

// is_odd, is_even

#define bli_is_odd( a )  ( a % 2 == 1 )
#define bli_is_even( a ) ( a % 2 == 0 )

// swap_types

#define bli_swap_types( type1, type2 ) \
{ \
	num_t temp = type1; \
	type1 = type2; \
	type2 = temp; \
}

// swap_dims

#define bli_swap_dims( dim1, dim2 ) \
{ \
	dim_t temp = dim1; \
	dim1 = dim2; \
	dim2 = temp; \
}

// swap_incs

#define bli_swap_incs( inc1, inc2 ) \
{ \
	inc_t temp = inc1; \
	inc1 = inc2; \
	inc2 = temp; \
}

// toggle_bool

#define bli_toggle_bool( b ) \
{ \
	if ( b == TRUE ) b = FALSE; \
	else             b = TRUE; \
}


#endif
