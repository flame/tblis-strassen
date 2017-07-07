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

void bli_trmm_front( side_t  side,
                     obj_t*  alpha,
                     obj_t*  a,
                     obj_t*  b,
                     cntx_t* cntx,
                     gemm_t* cntl )
{
	obj_t   a_local;
	obj_t   b_local;
	obj_t   c_local;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_trmm_check( side, alpha, a, b, &BLIS_ZERO, b, cntx );

	// If alpha is zero, scale by beta and return.
	if ( bli_obj_equals( alpha, &BLIS_ZERO ) )
	{
		bli_scalm( alpha, b );
		return;
	}

	// Reinitialize the memory allocator to accommodate the blocksizes
	// in the current context.
	bli_mem_reinit( cntx );

	// Alias A and B so we can tweak the objects if necessary.
	bli_obj_alias_to( *a, a_local );
	bli_obj_alias_to( *b, b_local );
	bli_obj_alias_to( *b, c_local );

	// We do not explicitly implement the cases where A is transposed.
	// However, we can still handle them. Specifically, if A is marked as
	// needing a transposition, we simply induce a transposition. This
	// allows us to only explicitly implement the no-transpose cases. Once
	// the transposition is induced, the correct algorithm will be called,
	// since, for example, an algorithm over a transposed lower triangular
	// matrix A moves in the same direction (forwards) as a non-transposed
	// upper triangular matrix. And with the transposition induced, the
	// matrix now appears to be upper triangular, so the upper triangular
	// algorithm will grab the correct partitions, as if it were upper
	// triangular (with no transpose) all along.
	if ( bli_obj_has_trans( a_local ) )
	{
		bli_obj_induce_trans( a_local );
		bli_obj_set_onlytrans( BLIS_NO_TRANSPOSE, a_local );
	}

#if 0

	// If A is being multiplied from the right, transpose all operands
	// so that we can perform the computation as if A were being multiplied
	// from the left.
	if ( bli_is_right( side ) )
	{
		bli_toggle_side( side );
		bli_obj_induce_trans( a_local );
		bli_obj_induce_trans( b_local );
		bli_obj_induce_trans( c_local );
	}

#else

	// An optimization: If C is stored by rows and the micro-kernel prefers
	// contiguous columns, or if C is stored by columns and the micro-kernel
	// prefers contiguous rows, transpose the entire operation to allow the
	// micro-kernel to access elements of C in its preferred manner.
	// NOTE: We disable the optimization for 1x1 matrices since the concept
	// of row- vs. column storage breaks down.
	if ( !bli_obj_is_1x1( c_local ) )
	if ( bli_cntx_l3_nat_ukr_dislikes_storage_of( &c_local, BLIS_GEMM_UKR, cntx ) )
	{
		bli_toggle_side( side );
		bli_obj_induce_trans( a_local );
		bli_obj_induce_trans( b_local );
		bli_obj_induce_trans( c_local );
	}

	// If A is being multiplied from the right, swap A and B so that
	// the matrix will actually be on the right.
	if ( bli_is_right( side ) )
	{
		bli_obj_swap( a_local, b_local );
	}

#endif

	// Set each alias as the root object.
	// NOTE: We MUST wait until we are done potentially swapping the objects
	// before setting the root fields!
	bli_obj_set_as_root( a_local );
	bli_obj_set_as_root( b_local );
	bli_obj_set_as_root( c_local );
 

    thrinfo_t** infos = bli_l3_thrinfo_create_paths( BLIS_TRMM, side );
    dim_t n_threads = bli_thread_num_threads( infos[0] );

    // Invoke the internal back-end.
    bli_l3_thread_decorator( n_threads,
                                 (l3_int_t) bli_trmm_int, 
                                 alpha, 
                                 &a_local,  
                                 &b_local,  
                                 &BLIS_ZERO, 
                                 &c_local,  
                                 (void*) cntx, 
                                 (void*) cntl, 
                                 (void**) infos );

    bli_l3_thrinfo_free_paths( infos, n_threads );

}

