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

#ifdef BLIS_ENABLE_PTHREADS
pthread_mutex_t mem_manager_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

// Declare one memory pool structure for each block size/shape we want to
// be able to allocate.
static pool_t pools[3];



// -----------------------------------------------------------------------------

void bli_mem_acquire_m( siz_t     req_size,
                        packbuf_t buf_type,
                        mem_t*    mem )
{
	pool_t* pool;
	pblk_t* pblk;
	dim_t   pi;
	siz_t   block_size;

	// Make sure the API is initialized.
	bli_mem_init();

	if ( buf_type == BLIS_BUFFER_FOR_GEN_USE )
	{
		// For general-use buffer requests, such as those used by level-2
		// operations, dynamically allocating memory is sufficient.
		void* buf_sys = bli_malloc_pool( req_size );

		// Initialize the mem_t object with:
		// - the address of the memory block,
		// - the buffer type (a packbuf_t value), and
		// - the size of the requested region.
		// NOTE: We do not initialize the pool field since this block did not
		// come from a memory pool.
		bli_mem_set_buffer( buf_sys, mem );
		bli_mem_set_buf_sys( buf_sys, mem );
		bli_mem_set_buf_type( buf_type, mem );
		bli_mem_set_size( req_size, mem );
	}
	else
	{
		// This branch handles cases where the memory block needs to come
		// from an internal memory pool, in which blocks are allocated once
		// and then recycled.

		// Map the requested packed buffer type to a zero-based index, which
		// we then use to select the corresponding memory pool.
		pi   = bli_packbuf_index( buf_type );
		pool = &pools[ pi ];

		// Unconditionally perform error checking on the memory pool.
		{
			err_t e_val;

			// Make sure that the requested matrix size fits inside of a block
			// of the corresponding pool. If it does not, the pool was somehow
			// initialized improperly.
			e_val = bli_check_requested_block_size_for_pool( req_size, pool );
			bli_check_error_code( e_val );
		}

		// Extract the address of the pblk_t struct within the mem_t.
		pblk = bli_mem_pblk( mem );

#ifdef BLIS_ENABLE_OPENMP
		_Pragma( "omp critical (mem)" )
#endif
#ifdef BLIS_ENABLE_PTHREADS
		pthread_mutex_lock( &mem_manager_mutex );
#endif
		// BEGIN CRITICAL SECTION
		{

			// Checkout a block from the pool. If the pool is exhausted,
			// either because it is still empty or because all blocks have
			// been checked out already, additional blocks will be allocated
			// automatically, as-needed. Note that the addresses are stored
			// directly into the mem_t struct since pblk is the address of
			// the struct's pblk_t field.
			bli_pool_checkout_block( pblk, pool );

			// Query the size of the blocks in the pool so we can store it in
			// the mem_t object. At this point, it is guaranteed to be at
			// least as large as req_size. (NOTE: We must perform the query
			// within the critical section to ensure that the pool hasn't
			// changed, as unlikely as that would be.)
			block_size = bli_pool_block_size( pool );

		}
		// END CRITICAL SECTION

#ifdef BLIS_ENABLE_PTHREADS
		pthread_mutex_unlock( &mem_manager_mutex );
#endif

		// Initialize the mem_t object with:
		// - the buffer type (a packbuf_t value),
		// - the address of the memory pool to which it belongs, and
		// - the size of the contiguous memory block (NOT the size of the
		//   requested region).
		// The actual addresses (system and aligned) are already stored in
		// the mem_t struct's pblk_t field
		bli_mem_set_buf_type( buf_type, mem );
		bli_mem_set_pool( pool, mem );
		bli_mem_set_size( block_size, mem );
	}
}


void bli_mem_release( mem_t* mem )
{
	packbuf_t buf_type;
	pool_t*   pool;
	pblk_t*   pblk;
	siz_t     block_size_cur;
	siz_t     block_size_prev;

	// Make sure the API is initialized.
	bli_mem_init();

	// Extract the buffer type so we know what kind of memory was allocated.
	buf_type = bli_mem_buf_type( mem );

	if ( buf_type == BLIS_BUFFER_FOR_GEN_USE )
	{
		void* buf_sys = bli_mem_buf_sys( mem );

		// For general-use buffers, we dynamically allocate memory, and so
		// here we need to free.
		bli_free_pool( buf_sys );
	}
	else
	{
		// Extract the address of the pool from which the memory was
		// allocated.
		pool = bli_mem_pool( mem );

		// Extract the address of the pblk_t struct within the mem_t struct.
		pblk = bli_mem_pblk( mem );

		// Query the size of the blocks that were in the pool at the time
		// the pblk_t was checked out. (This is used below, in the critical
		// section.)
		block_size_prev = bli_mem_size( mem );

#ifdef BLIS_ENABLE_OPENMP
		_Pragma( "omp critical (mem)" )
#endif
#ifdef BLIS_ENABLE_PTHREADS
		pthread_mutex_lock( &mem_manager_mutex );
#endif

		// BEGIN CRITICAL SECTION
		{

			// Query the size of the blocks currently in the pool.
			block_size_cur = bli_pool_block_size( pool );

			// If the block size of the pool has changed since the pblk_t
			// was checked out, then we need to free the pblk_t rather
			// than check it back in. Why? Because the pool's block size
			// has (most likely) increased to meet changing needs (example:
			// larger cache blocksizes). Thus, the current pblk_t's smaller
			// allocated size is of no use anymore.
			if ( block_size_cur != block_size_prev )
			{
				// Free the pblk_t using the appropriate function in the
				// pool API.
				bli_pool_free_block( pblk );
			}
			else
			{
				// Check the block back into the pool.
				bli_pool_checkin_block( pblk, pool );
			}

		}
		// END CRITICAL SECTION

#ifdef BLIS_ENABLE_PTHREADS
		pthread_mutex_unlock( &mem_manager_mutex );
#endif
	}

	// Clear the mem_t object so that it appears unallocated. This clears:
	// - the pblk_t struct's fields (ie: the buffer addresses)
	// - the pool field
	// - the size field
	// NOTE: We do not clear the buf_type field since there is no
	// "uninitialized" value for packbuf_t.
	bli_mem_clear( mem );
}


void bli_mem_acquire_v( siz_t  req_size,
                        mem_t* mem )
{
	bli_mem_acquire_m( req_size,
	                   BLIS_BUFFER_FOR_GEN_USE,
	                   mem );
}


siz_t bli_mem_pool_size( packbuf_t buf_type )
{
	siz_t r_val;

	if ( buf_type == BLIS_BUFFER_FOR_GEN_USE )
	{
		// We don't (yet) track the amount of general-purpose
		// memory that is currently allocated.
		r_val = 0;
	}
	else
	{
		dim_t   index;
		pool_t* pool;

		// Acquire the pointer to the pool corresponding to the buf_type
		// provided.
		index = bli_packbuf_index( buf_type );
		pool  = &(pools[index]);

		// Compute the pool "size" as the product of the block size 
		// and the number of blocks in the pool.
		r_val = bli_pool_block_size( pool ) *
		        bli_pool_num_blocks( pool );
	}

	return r_val;
}

// -----------------------------------------------------------------------------

static bool_t bli_mem_is_init = FALSE;

void bli_mem_init( void )
{
	cntx_t cntx;

	// If the initialization flag is TRUE, we know the API is already
	// initialized, so we can return early.
	if ( bli_mem_is_init == TRUE ) return;

	// Create and initialize a context for gemm so we have something
	// to pass into bli_mem_init_pools().
	bli_gemm_cntx_init( &cntx );

#ifdef BLIS_ENABLE_OPENMP
	_Pragma( "omp critical (mem)" )
#endif
#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_lock( &mem_manager_mutex );
#endif

	// BEGIN CRITICAL SECTION
	{
		// Here, we test the initialization flag again. NOTE: THIS IS NOT
		// REDUNDANT. This additional test is needed so that other threads
		// that may be waiting to acquire the lock do not perform any
		// initialization actions once they are finally allowed into this
		// critical section.
		if ( bli_mem_is_init == FALSE )
		{
			// Initialize the memory pools.
			bli_mem_init_pools( &cntx );

			// After initialization, mark the API as initialized.
			bli_mem_is_init = TRUE;
		}
	}
	// END CRITICAL SECTION

#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_unlock( &mem_manager_mutex );
#endif

	// Finalize the temporary gemm context.
	bli_gemm_cntx_finalize( &cntx );
}

void bli_mem_reinit( cntx_t* cntx )
{
#ifdef BLIS_ENABLE_OPENMP
	_Pragma( "omp critical (mem)" )
#endif
#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_lock( &mem_manager_mutex );
#endif

	// BEGIN CRITICAL SECTION
	{
		// If for some reason the memory pools have not yet been
		// initialized (unlikely), we emulate the body of bli_mem_init().
		if ( bli_mem_is_init == FALSE )
		{
			// Initialize the memory pools.
			bli_mem_init_pools( cntx );

			// After initialization, mark the API as initialized.
			bli_mem_is_init = TRUE;
		}
		else
		{
			// Reinitialize the memory pools.
			bli_mem_reinit_pools( cntx );
		}
	}
	// END CRITICAL SECTION

#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_unlock( &mem_manager_mutex );
#endif
}

void bli_mem_finalize( void )
{
	// If the initialization flag is FALSE, we know the API is already
	// uninitialized, so we can return early.
	if ( bli_mem_is_init == FALSE ) return;

#ifdef BLIS_ENABLE_OPENMP
	_Pragma( "omp critical (mem)" )
#endif
#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_lock( &mem_manager_mutex );
#endif

	// BEGIN CRITICAL SECTION
	{
		// Here, we test the initialization flag again. NOTE: THIS IS NOT
		// REDUNDANT. This additional test is needed so that other threads
		// that may be waiting to acquire the lock do not perform any
		// finalization actions once they are finally allowed into this
		// critical section.
		if ( bli_mem_is_init == TRUE )
		{
			// Finalize the memory pools.
			bli_mem_finalize_pools();

			// After finalization, mark the API as uninitialized.
			bli_mem_is_init = FALSE;
		}
	}
	// END CRITICAL SECTION

#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_unlock( &mem_manager_mutex );
#endif
}

bool_t bli_mem_is_initialized( void )
{
	return bli_mem_is_init;
}

// -----------------------------------------------------------------------------

void bli_mem_init_pools( cntx_t* cntx )
{
	// Map each of the packbuf_t values to an index starting at zero.
	const dim_t index_a      = bli_packbuf_index( BLIS_BUFFER_FOR_A_BLOCK );
	const dim_t index_b      = bli_packbuf_index( BLIS_BUFFER_FOR_B_PANEL );
	const dim_t index_c      = bli_packbuf_index( BLIS_BUFFER_FOR_C_PANEL );

	const siz_t align_size   = BLIS_POOL_ADDR_ALIGN_SIZE;

	// Alias the pool addresses to convenient identifiers.
	pool_t*     pool_a       = &pools[ index_a ];
	pool_t*     pool_b       = &pools[ index_b ];
	pool_t*     pool_c       = &pools[ index_c ];

	// Start with empty pools.
	const dim_t num_blocks_a = 0;
	const dim_t num_blocks_b = 0;
	const dim_t num_blocks_c = 0;

	siz_t       block_size_a = 0;
	siz_t       block_size_b = 0;
	siz_t       block_size_c = 0;

	// Determine the block size for each memory pool.
	bli_mem_compute_pool_block_sizes( &block_size_a,
	                                  &block_size_b,
	                                  &block_size_c,
	                                  cntx );

	// Initialize the memory pools for A, B, and C.
	bli_pool_init( num_blocks_a, block_size_a, align_size, pool_a );
	bli_pool_init( num_blocks_b, block_size_b, align_size, pool_b );
	bli_pool_init( num_blocks_c, block_size_c, align_size, pool_c );
}

void bli_mem_reinit_pools( cntx_t* cntx )
{
	// Map each of the packbuf_t values to an index starting at zero.
	const dim_t index_a      = bli_packbuf_index( BLIS_BUFFER_FOR_A_BLOCK );
	const dim_t index_b      = bli_packbuf_index( BLIS_BUFFER_FOR_B_PANEL );
	const dim_t index_c      = bli_packbuf_index( BLIS_BUFFER_FOR_C_PANEL );

	const siz_t align_size   = BLIS_POOL_ADDR_ALIGN_SIZE;

	// Alias the pool addresses to convenient identifiers.
	pool_t*     pool_a       = &pools[ index_a ];
	pool_t*     pool_b       = &pools[ index_b ];
	pool_t*     pool_c       = &pools[ index_c ];

	// Query the number of blocks currently allocated in each pool.
	const dim_t num_blocks_a = bli_pool_num_blocks( pool_a );
	const dim_t num_blocks_b = bli_pool_num_blocks( pool_b );
	const dim_t num_blocks_c = bli_pool_num_blocks( pool_c );

	siz_t       block_size_a_new = 0;
	siz_t       block_size_b_new = 0;
	siz_t       block_size_c_new = 0;

	// Determine the context-implied block size needed for each pool.
	bli_mem_compute_pool_block_sizes( &block_size_a_new,
	                                  &block_size_b_new,
	                                  &block_size_c_new,
	                                  cntx );

	// Reinitialize the pool, but only if one of the parameters has
	// changed in such a way that reinitialization would be required.
	// In this case, the align_size is constant, as is num_blocks, so
	// what this actually boils down to is that reinitialization of a
	// pool occurs only if the block size for that pool has increased.
	bli_pool_reinit_if( num_blocks_a, block_size_a_new, align_size, pool_a );
	bli_pool_reinit_if( num_blocks_b, block_size_b_new, align_size, pool_b );
	bli_pool_reinit_if( num_blocks_c, block_size_c_new, align_size, pool_c );
}

void bli_mem_finalize_pools( void )
{
	// Map each of the packbuf_t values to an index starting at zero.
	dim_t   index_a = bli_packbuf_index( BLIS_BUFFER_FOR_A_BLOCK );
	dim_t   index_b = bli_packbuf_index( BLIS_BUFFER_FOR_B_PANEL );
	dim_t   index_c = bli_packbuf_index( BLIS_BUFFER_FOR_C_PANEL );

	// Alias the pool addresses to convenient identifiers.
	pool_t* pool_a  = &pools[ index_a ];
	pool_t* pool_b  = &pools[ index_b ];
	pool_t* pool_c  = &pools[ index_c ];

	// Finalize the memory pools for A, B, and C.
	bli_pool_finalize( pool_a );
	bli_pool_finalize( pool_b );
	bli_pool_finalize( pool_c );
}

// -----------------------------------------------------------------------------

void bli_mem_compute_pool_block_sizes( siz_t*  bs_a,
                                       siz_t*  bs_b,
                                       siz_t*  bs_c,
                                       cntx_t* cntx )
{
	const ind_t im = bli_cntx_get_ind_method( cntx );

	siz_t bs_cand_a = 0;
	siz_t bs_cand_b = 0;
	siz_t bs_cand_c = 0;

	num_t dt;

	// Compute pool block sizes for each datatype and find the maximum
	// size for each pool. This is done so that new pools do not need
	// to be allocated if the user switches datatypes.
	for ( dt = BLIS_DT_LO; dt <= BLIS_DT_HI; ++dt )
	{
		siz_t bs_dt_a;
		siz_t bs_dt_b;
		siz_t bs_dt_c;

		// Avoid considering induced methods for real datatypes.
		if ( bli_is_real( dt ) && im != BLIS_NAT ) continue;

		bli_mem_compute_pool_block_sizes_dt( dt,
		                                     &bs_dt_a,
		                                     &bs_dt_b,
		                                     &bs_dt_c,
		                                     cntx );

		bs_cand_a = bli_max( bs_dt_a, bs_cand_a );
		bs_cand_b = bli_max( bs_dt_b, bs_cand_b );
		bs_cand_c = bli_max( bs_dt_c, bs_cand_c );
	}

	// Save the results.
	*bs_a = bs_cand_a;
	*bs_b = bs_cand_b;
	*bs_c = bs_cand_c;
}

// -----------------------------------------------------------------------------

void bli_mem_compute_pool_block_sizes_dt( num_t   dt,
                                          siz_t*  bs_a,
                                          siz_t*  bs_b,
                                          siz_t*  bs_c,
                                          cntx_t* cntx )
{
	siz_t    size_dt = bli_datatype_size( dt );

	blksz_t* mr;
	blksz_t* nr;

	blksz_t* mc;
	blksz_t* kc;
	blksz_t* nc;

	dim_t    mr_dt;
	dim_t    nr_dt;
	dim_t    max_mnr_dt;

	dim_t    mc_max_dt;
	dim_t    kc_max_dt;
	dim_t    nc_max_dt;

	dim_t    packmr_dt;
	dim_t    packnr_dt;
	dim_t    max_packmnr_dt;

	dim_t    scale_num_dt;
	dim_t    scale_den_dt;

	dim_t    pool_mc_dt, left_mc_dt;
	dim_t    pool_nc_dt, left_nc_dt;
	dim_t    pool_kc_dt;

	//
	// Find the larger of the two register blocksizes.
	//

	// Query the mr and nr blksz_t objects for the given method of
	// execution.
	mr = bli_cntx_get_blksz( BLIS_MR, cntx );
	nr = bli_cntx_get_blksz( BLIS_NR, cntx );

	// Extract the mr and nr values specific to the current datatype.
	mr_dt = bli_blksz_get_def( dt, mr );
	nr_dt = bli_blksz_get_def( dt, nr );

	// Find the maximum of mr and nr.
	max_mnr_dt = bli_max( mr_dt, nr_dt );

	//
	// Define local maximum cache blocksizes.
	//

	// Query the mc, kc, and nc blksz_t objects for native execution.
	mc = bli_cntx_get_blksz( BLIS_MC, cntx );
	kc = bli_cntx_get_blksz( BLIS_KC, cntx );
	nc = bli_cntx_get_blksz( BLIS_NC, cntx );

	// Extract the maximum mc, kc, and nc values specific to the current
	// datatype.
	mc_max_dt = bli_blksz_get_max( dt, mc );
	kc_max_dt = bli_blksz_get_max( dt, kc );
	nc_max_dt = bli_blksz_get_max( dt, nc );

	// Add max(mr,nr) to kc to make room for the nudging of kc at
	// runtime to be a multiple of mr or nr for triangular operations
	// trmm, trmm3, and trsm.
	kc_max_dt += max_mnr_dt;

	//
	// Compute scaling factors.
	//

	// Compute integer scaling factors (numerator and denominator) used
	// to account for situations when the packing register blocksizes are
	// larger than the regular register blocksizes.

	// In order to compute the scaling factors, we first have to determine
	// whether ( packmr / mr ) is greater than ( packnr / nr ). This is
	// needed ONLY because the amount of space allocated for a block of A
	// and a panel of B needs to be such that MR and NR can be swapped (ie:
	// A is packed with NR and B is packed with MR). This transformation is
	// needed for right-side trsm when inducing an algorithm that (a) has
	// favorable access patterns for column-stored C and (b) allows the
	// macro-kernel to reuse the existing left-side fused gemmtrsm micro-
	// kernels. We avoid integer division by cross-multiplying:
	//
	//   ( packmr / mr )      >= ( packnr / nr )
	//   ( packmr / mr ) * nr >=   packnr
	//     packmr * nr        >=   packnr * mr
	//
	// So, if packmr * nr >= packnr * mr, then we will use packmr and mr as
	// our scaling factors. Otherwise, we'll use packnr and nr.

	packmr_dt = bli_blksz_get_max( dt, mr );
	packnr_dt = bli_blksz_get_max( dt, nr );

	if ( packmr_dt * nr_dt >=
	     packnr_dt * mr_dt ) { scale_num_dt = packmr_dt;
	                           scale_den_dt =     mr_dt; }
	else                     { scale_num_dt = packnr_dt;
	                           scale_den_dt =     nr_dt; }

	//
	// Compute pool block dimensions.
	//

	pool_mc_dt = ( mc_max_dt * scale_num_dt ) / scale_den_dt;
	left_mc_dt = ( mc_max_dt * scale_num_dt ) % scale_den_dt;

	pool_nc_dt = ( nc_max_dt * scale_num_dt ) / scale_den_dt;
	left_nc_dt = ( nc_max_dt * scale_num_dt ) % scale_den_dt;

	pool_kc_dt = ( kc_max_dt );

	if ( left_mc_dt > 0 ) pool_mc_dt += 1;
	if ( left_nc_dt > 0 ) pool_nc_dt += 1;

	//
	// Compute pool block sizes
	//

	// We add an extra micro-panel of space to the block sizes for A and B
	// just to be sure any pre-loading performed by the micro-kernel does
	// not cause a segmentation fault.
	max_packmnr_dt = bli_max( packmr_dt, packnr_dt );

	*bs_a = ( pool_mc_dt + max_packmnr_dt ) * pool_kc_dt * size_dt;
	*bs_b = ( pool_nc_dt + max_packmnr_dt ) * pool_kc_dt * size_dt;
	*bs_c = ( pool_mc_dt                  ) * pool_nc_dt * size_dt;
}
