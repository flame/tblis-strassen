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

#if 0
//
// Generate prototypes for _cntx_init(), _cntx_stage(), and _cntx_finalize()
// for each induced method (including native execution) based on gemm.
//

#undef  GENPROT
#define GENPROT( opname, imeth ) \
\
void  PASTEMAC2(opname,imeth,_cntx_init)( void ); \
void  PASTEMAC2(opname,imeth,_cntx_stage)( dim_t stage, cntx_t* cntx ); \
void  PASTEMAC2(opname,imeth,_cntx_finalize)( void );

GENPROT( gemm, nat )
GENPROT( gemm, 3mh )
GENPROT( gemm, 3m3 )
GENPROT( gemm, 3m2 )
GENPROT( gemm, 3m1 )
GENPROT( gemm, 4mh )
GENPROT( gemm, 4mb )
GENPROT( gemm, 4m1 )
#endif

void  bli_gemmnat_cntx_init( cntx_t* cntx );
void  bli_gemmnat_cntx_stage( dim_t stage, cntx_t* cntx );
void  bli_gemmnat_cntx_finalize( cntx_t* cntx );

void  bli_gemm3mh_cntx_init( cntx_t* cntx );
void  bli_gemm3mh_cntx_stage( dim_t stage, cntx_t* cntx );
void  bli_gemm3mh_cntx_finalize( cntx_t* cntx );

void  bli_gemm3m3_cntx_init( cntx_t* cntx );
void  bli_gemm3m3_cntx_stage( dim_t stage, cntx_t* cntx );
void  bli_gemm3m3_cntx_finalize( cntx_t* cntx );

void  bli_gemm3m2_cntx_init( cntx_t* cntx );
void  bli_gemm3m2_cntx_stage( dim_t stage, cntx_t* cntx );
void  bli_gemm3m2_cntx_finalize( cntx_t* cntx );

void  bli_gemm3m1_cntx_init( cntx_t* cntx );
void  bli_gemm3m1_cntx_stage( dim_t stage, cntx_t* cntx );
void  bli_gemm3m1_cntx_finalize( cntx_t* cntx );

void  bli_gemm4mh_cntx_init( cntx_t* cntx );
void  bli_gemm4mh_cntx_stage( dim_t stage, cntx_t* cntx );
void  bli_gemm4mh_cntx_finalize( cntx_t* cntx );

void  bli_gemm4mb_cntx_init( cntx_t* cntx );
void  bli_gemm4mb_cntx_stage( dim_t stage, cntx_t* cntx );
void  bli_gemm4mb_cntx_finalize( cntx_t* cntx );

void  bli_gemm4m1_cntx_init( cntx_t* cntx );
void  bli_gemm4m1_cntx_stage( dim_t stage, cntx_t* cntx );
void  bli_gemm4m1_cntx_finalize( cntx_t* cntx );

// -----------------------------------------------------------------------------

void  bli_gemmind_cntx_init_avail( num_t dt, cntx_t* cntx );
void  bli_gemmind_cntx_finalize_avail( num_t dt, cntx_t* cntx );

void  bli_gemmind_cntx_init( ind_t method, cntx_t* cntx );
void  bli_gemmind_cntx_finalize( ind_t method, cntx_t* cntx );

void* bli_gemmind_cntx_init_get_func( ind_t method );
void* bli_gemmind_cntx_finalize_get_func( ind_t method );

