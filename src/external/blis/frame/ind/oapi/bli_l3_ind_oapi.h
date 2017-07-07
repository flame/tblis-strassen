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


//
// Generate object-based prototypes for induced methods that work for
// trmm and trsm (ie: two-operand operations).
//
#undef  GENPROT
#define GENPROT( imeth ) \
\
void PASTEMAC(gemm,imeth) (              obj_t* alpha, obj_t* a, obj_t* b, obj_t* beta, obj_t* c, cntx_t* cntx ); \
void PASTEMAC(hemm,imeth) ( side_t side, obj_t* alpha, obj_t* a, obj_t* b, obj_t* beta, obj_t* c, cntx_t* cntx ); \
void PASTEMAC(herk,imeth) (              obj_t* alpha, obj_t* a,           obj_t* beta, obj_t* c, cntx_t* cntx ); \
void PASTEMAC(her2k,imeth)(              obj_t* alpha, obj_t* a, obj_t* b, obj_t* beta, obj_t* c, cntx_t* cntx ); \
void PASTEMAC(symm,imeth) ( side_t side, obj_t* alpha, obj_t* a, obj_t* b, obj_t* beta, obj_t* c, cntx_t* cntx ); \
void PASTEMAC(syrk,imeth) (              obj_t* alpha, obj_t* a,           obj_t* beta, obj_t* c, cntx_t* cntx ); \
void PASTEMAC(syr2k,imeth)(              obj_t* alpha, obj_t* a, obj_t* b, obj_t* beta, obj_t* c, cntx_t* cntx ); \
void PASTEMAC(trmm3,imeth)( side_t side, obj_t* alpha, obj_t* a, obj_t* b, obj_t* beta, obj_t* c, cntx_t* cntx ); \
void PASTEMAC(trmm,imeth) ( side_t side, obj_t* alpha, obj_t* a, obj_t* b,                        cntx_t* cntx ); \
void PASTEMAC(trsm,imeth) ( side_t side, obj_t* alpha, obj_t* a, obj_t* b,                        cntx_t* cntx );

GENPROT( nat )
GENPROT( ind )
GENPROT( 3m1 )
GENPROT( 4m1 )


//
// Generate object-based prototypes for induced methods that do NOT work
// for trmm and trsm (ie: two-operand operations).
//
#undef  GENPROT_NO2OP
#define GENPROT_NO2OP( imeth ) \
\
void PASTEMAC(gemm,imeth) (              obj_t* alpha, obj_t* a, obj_t* b, obj_t* beta, obj_t* c, cntx_t* cntx ); \
void PASTEMAC(hemm,imeth) ( side_t side, obj_t* alpha, obj_t* a, obj_t* b, obj_t* beta, obj_t* c, cntx_t* cntx ); \
void PASTEMAC(herk,imeth) (              obj_t* alpha, obj_t* a,           obj_t* beta, obj_t* c, cntx_t* cntx ); \
void PASTEMAC(her2k,imeth)(              obj_t* alpha, obj_t* a, obj_t* b, obj_t* beta, obj_t* c, cntx_t* cntx ); \
void PASTEMAC(symm,imeth) ( side_t side, obj_t* alpha, obj_t* a, obj_t* b, obj_t* beta, obj_t* c, cntx_t* cntx ); \
void PASTEMAC(syrk,imeth) (              obj_t* alpha, obj_t* a,           obj_t* beta, obj_t* c, cntx_t* cntx ); \
void PASTEMAC(syr2k,imeth)(              obj_t* alpha, obj_t* a, obj_t* b, obj_t* beta, obj_t* c, cntx_t* cntx ); \
void PASTEMAC(trmm3,imeth)( side_t side, obj_t* alpha, obj_t* a, obj_t* b, obj_t* beta, obj_t* c, cntx_t* cntx );

GENPROT_NO2OP( 3mh )
GENPROT_NO2OP( 3m3 )
GENPROT_NO2OP( 3m2 )
GENPROT_NO2OP( 4mh )
GENPROT_NO2OP( 4mb )

