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

static void* bli_l3_ind_oper_fp[BLIS_NUM_IND_METHODS][BLIS_NUM_LEVEL3_OPS] = 
{
        /*   gemm   hemm   herk   her2k  symm   syrk,  syr2k  trmm3  trmm   trsm  */
/* 3mh  */ { bli_gemm3mh,  bli_hemm3mh,  bli_herk3mh,  bli_her2k3mh, bli_symm3mh,
             bli_syrk3mh,  bli_syr2k3mh, bli_trmm33mh, NULL,         NULL         },
/* 3m3  */ { bli_gemm3m3,  NULL,         NULL,         NULL,         NULL,         
             NULL,         NULL,         NULL,         NULL,         NULL         },
/* 3m2  */ { bli_gemm3m2,  NULL,         NULL,         NULL,         NULL,         
             NULL,         NULL,         NULL,         NULL,         NULL         },
/* 3m1  */ { bli_gemm3m1,  bli_hemm3m1,  bli_herk3m1,  bli_her2k3m1, bli_symm3m1,
             bli_syrk3m1,  bli_syr2k3m1, bli_trmm33m1, bli_trmm3m1,  bli_trsm3m1  },
/* 4mh  */ { bli_gemm4mh,  bli_hemm4mh,  bli_herk4mh,  bli_her2k4mh, bli_symm4mh,
             bli_syrk4mh,  bli_syr2k4mh, bli_trmm34mh, NULL,         NULL         },
/* 4mb  */ { bli_gemm4mb,  NULL,         NULL,         NULL,         NULL,         
             NULL,         NULL,         NULL,         NULL,         NULL         },
/* 4m1  */ { bli_gemm4m1,  bli_hemm4m1,  bli_herk4m1,  bli_her2k4m1, bli_symm4m1,
             bli_syrk4m1,  bli_syr2k4m1, bli_trmm34m1, bli_trmm4m1,  bli_trsm4m1  },
/* nat  */ { bli_gemmnat,  bli_hemmnat,  bli_herknat,  bli_her2knat, bli_symmnat,
             bli_syrknat,  bli_syr2knat, bli_trmm3nat, bli_trmmnat,  bli_trsmnat  },
};

//
// NOTE: "2" is used instead of BLIS_NUM_FP_TYPES/2.
//
static bool_t bli_l3_ind_oper_st[BLIS_NUM_IND_METHODS][BLIS_NUM_LEVEL3_OPS][2] = 
{
        /*   gemm   hemm   herk   her2k  symm   syrk,  syr2k  trmm3  trmm   trsm  */
        /*    c     z    */
/* 3mh  */ { {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE},
             {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}  },
/* 3m3  */ { {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE},
             {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}  },
/* 3m2  */ { {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE},
             {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}  },
/* 3m1  */ { {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE},
             {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}  },
/* 4mh  */ { {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE},
             {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}  },
/* 4mb  */ { {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE},
             {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}  },
/* 4m1  */ { {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE},
             {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}  },
/* nat  */ { {TRUE,TRUE},   {TRUE,TRUE},   {TRUE,TRUE},   {TRUE,TRUE},   {TRUE,TRUE},
             {TRUE,TRUE},   {TRUE,TRUE},   {TRUE,TRUE},   {TRUE,TRUE},   {TRUE,TRUE}    },
};

// -----------------------------------------------------------------------------

#undef  GENFUNC
#define GENFUNC( opname, optype ) \
\
bool_t PASTEMAC(opname,ind_has_avail)( num_t dt ) \
{ \
	return bli_ind_oper_has_avail( optype, dt ); \
} \
void*  PASTEMAC(opname,ind_get_avail)( num_t dt ) \
{ \
	return bli_ind_oper_get_avail( optype, dt ); \
}

GENFUNC( gemm, BLIS_GEMM )
GENFUNC( hemm, BLIS_HEMM )
GENFUNC( herk, BLIS_HERK )
GENFUNC( her2k, BLIS_HER2K )
GENFUNC( symm, BLIS_SYMM )
GENFUNC( syrk, BLIS_SYRK )
GENFUNC( syr2k, BLIS_SYR2K )
GENFUNC( trmm3, BLIS_TRMM3 )
GENFUNC( trmm, BLIS_TRMM )
GENFUNC( trsm, BLIS_TRSM )

// -----------------------------------------------------------------------------

#if 0
bool_t bli_l3_ind_oper_is_avail( opid_t oper, ind_t method, num_t dt )
{
	void*  func;
	bool_t stat;

	// If the datatype is real, it is never available.
	if ( !bli_is_complex( dt ) ) return FALSE;

	func = bli_l3_ind_oper_get_func( oper, method );
	stat = bli_l3_ind_oper_get_enable( oper, method, dt );

	return ( func != NULL && stat == TRUE );
}
#endif

// -----------------------------------------------------------------------------

ind_t bli_l3_ind_oper_find_avail( opid_t oper, num_t dt )
{
	ind_t im;

	// If the datatype is real, return native execution.
	if ( !bli_is_complex( dt ) ) return BLIS_NAT;

	// If the operation is not level-3, return native execution.
	if ( !bli_opid_is_level3( oper ) ) return BLIS_NAT;

	// Iterate over all induced methods and search for the first one
	// that is available (ie: both implemented and enabled) for the
	// current operation and datatype.
	for ( im = 0; im < BLIS_NUM_IND_METHODS; ++im )
	{
		void*  func = bli_l3_ind_oper_get_func( oper, im );
		bool_t stat = bli_l3_ind_oper_get_enable( oper, im, dt );

		if ( func != NULL &&
		     stat == TRUE ) return im;
	}

	// This return statement should never execute since the native index
	// should be found even if all induced methods are unavailable. We
	// include it simply to avoid a compiler warning.
	return BLIS_NAT;
}

// -----------------------------------------------------------------------------

void bli_l3_ind_set_enable_dt( ind_t method, num_t dt, bool_t status )
{
	opid_t iop;

	if ( !bli_is_complex( dt ) ) return;

	// Iterate over all level-3 operation ids.
	for ( iop = 0; iop < BLIS_NUM_LEVEL3_OPS; ++iop )
	{
		bli_l3_ind_oper_set_enable( iop, method, dt, status );
	}
}

// -----------------------------------------------------------------------------

void bli_l3_ind_oper_enable_only( opid_t oper, ind_t method, num_t dt )
{
	if ( !bli_is_complex( dt ) ) return;
	if ( !bli_opid_is_level3( oper ) ) return;

	bli_l3_ind_oper_set_enable_all( oper, dt, FALSE );
	bli_l3_ind_oper_set_enable( oper, method, dt, TRUE );
}

void bli_l3_ind_oper_set_enable_all( opid_t oper, num_t dt, bool_t status )
{
	ind_t im;

	if ( !bli_is_complex( dt ) ) return;
	if ( !bli_opid_is_level3( oper ) ) return;

	for ( im = 0; im < BLIS_NUM_IND_METHODS; ++im )
	{
		// Native execution should always stay enabled.
		if ( im != BLIS_NAT )
			bli_l3_ind_oper_set_enable( oper, im, dt, status );
	}
}

// -----------------------------------------------------------------------------

void bli_l3_ind_oper_set_enable( opid_t oper, ind_t method, num_t dt, bool_t status )
{
	num_t idt;

	if ( !bli_is_complex( dt ) ) return;
	if ( !bli_opid_is_level3( oper ) ) return;

	// Disallow changing status of native execution.
	if ( method == BLIS_NAT ) return;

	idt = bli_ind_map_cdt_to_index( dt );

	bli_l3_ind_oper_st[ method ][ oper ][ idt ] = status;
}

bool_t bli_l3_ind_oper_get_enable( opid_t oper, ind_t method, num_t dt )
{
	num_t idt = bli_ind_map_cdt_to_index( dt );

	return bli_l3_ind_oper_st[ method ][ oper ][ idt ];
}

// -----------------------------------------------------------------------------

void* bli_l3_ind_oper_get_func( opid_t oper, ind_t method )
{
	return bli_l3_ind_oper_fp[ method ][ oper ];
}

