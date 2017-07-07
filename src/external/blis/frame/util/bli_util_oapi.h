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
// Prototype object-based interfaces.
//

#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       obj_t*  x, \
       obj_t*  index  \
       BLIS_OAPI_CNTX_PARAM  \
     );

GENPROT( amaxv )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       obj_t*  x, \
       obj_t*  asum  \
       BLIS_OAPI_CNTX_PARAM  \
     );

GENPROT( asumv )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       obj_t*  a  \
       BLIS_OAPI_CNTX_PARAM  \
     );

GENPROT( mkherm )
GENPROT( mksymm )
GENPROT( mktrim )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       obj_t*  x, \
       obj_t*  norm  \
       BLIS_OAPI_CNTX_PARAM  \
     );

GENPROT( norm1v )
GENPROT( normfv )
GENPROT( normiv )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       obj_t*  x, \
       obj_t*  norm  \
       BLIS_OAPI_CNTX_PARAM  \
     );

GENPROT( norm1m )
GENPROT( normfm )
GENPROT( normim )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       FILE*   file, \
       char*   s1, \
       obj_t*  x, \
       char*   format, \
       char*   s2  \
       BLIS_OAPI_CNTX_PARAM  \
     );

GENPROT( fprintv )
GENPROT( fprintm )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       char*   s1, \
       obj_t*  x, \
       char*   format, \
       char*   s2  \
       BLIS_OAPI_CNTX_PARAM  \
     );

GENPROT( printv )
GENPROT( printm )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       obj_t*  x  \
       BLIS_OAPI_CNTX_PARAM  \
     );

GENPROT( randv )
GENPROT( randnv )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       obj_t*  x  \
       BLIS_OAPI_CNTX_PARAM  \
     );

GENPROT( randm )
GENPROT( randnm )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       obj_t*  x, \
       obj_t*  scale, \
       obj_t*  sumsq  \
       BLIS_OAPI_CNTX_PARAM  \
     );

GENPROT( sumsqv )

