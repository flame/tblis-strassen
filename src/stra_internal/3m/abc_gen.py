import sys

import abc_micro_kernel_gen

from common import is_one, is_negone, is_nonzero, write_line, write_break, data_access, transpose, printmat, writeCoeffs, phantomMatMul, parse_coeff, read_coeffs, writeFMM, writePartition, writeEquation, getBlockName, getName, generateCoeffs, exp_dim, contain_nontrivial

def create_macro_functions( myfile, coeffs ):
    for i, coeff_set in enumerate( transpose( coeffs[2] ) ):
        if len( coeff_set ) > 0:
            write_macro_func( myfile, coeff_set, i, 'C' )
            write_break( myfile )

def create_micro_functions( myfile, coeffs, kernel_header_filename ):
    write_line( myfile, 0, '#include "%s"' % kernel_header_filename )
    write_break( myfile )
    abc_micro_kernel_gen.write_common_rankk_macro_assembly( myfile )
    write_break( myfile )
    abc_micro_kernel_gen.macro_initialize_assembly( myfile )
    #write_break( myfile )
    #abc_micro_kernel_gen.macro_rankk_xor0_assembly( myfile )
    #write_break( myfile )
    #abc_micro_kernel_gen.macro_rankk_loopkiter_assembly( myfile )
    #write_break( myfile )
    #abc_micro_kernel_gen.macro_rankk_loopkleft_assembly( myfile )
    #write_break( myfile )
    #abc_micro_kernel_gen.macro_rankk_postaccum_assembly( myfile )
    write_break( myfile )
    for i, coeff_set in enumerate( transpose( coeffs[2] ) ):
        if len( coeff_set ) > 0:
            nonzero_coeffs = [coeff for coeff in coeff_set if is_nonzero(coeff)]
            nnz = len( nonzero_coeffs )

            if nnz <= 23:
                abc_micro_kernel_gen.generate_micro_kernel( myfile, nonzero_coeffs, i )

            write_break( myfile )

def create_kernel_header( myfile, coeffs ):
    #write_line( myfile, 0, '#include "bl_dgemm_kernel.h"' )
    write_break( myfile )
    abc_micro_kernel_gen.write_header_start( myfile )
    for i, coeff_set in enumerate( transpose( coeffs[2] ) ):
        if len( coeff_set ) > 0:
            nonzero_coeffs = [coeff for coeff in coeff_set if is_nonzero(coeff)]
            nnz = len( nonzero_coeffs )
            abc_micro_kernel_gen.generate_kernel_header( myfile, nonzero_coeffs, i )
            write_break( myfile )
    abc_micro_kernel_gen.write_header_end( myfile )


def create_packm_functions(myfile, coeffs):
    ''' Generate all of the custom add functions.

    myfile is the file to which we are writing
    coeffs is the set of all coefficients
    '''
    def all_adds(coeffs, name):
        for i, coeff_set in enumerate(coeffs):
            if len(coeff_set) > 0:
                write_packm_func(myfile, coeff_set, i, name)
                write_break(myfile)

    # S matrices formed from A subblocks
    all_adds(transpose(coeffs[0]), 'A')

    # T matrices formed from B subblocks
    all_adds(transpose(coeffs[1]), 'B')

    # Output C formed from multiplications
    ##all_adds( coeffs[2], 'M' )
    #all_adds(transpose(coeffs[2]), 'C' )

def write_macro_func( myfile, coeffs, index, mat_name ):
    ''' Write the add function for a set of coefficients.  This is a custom add
    function used for a single multiply in a single fast algorithm.

    coeffs is the set of coefficients used for the add
    '''
    nonzero_coeffs = [coeff for coeff in coeffs if is_nonzero(coeff)]
    nnz = len( nonzero_coeffs )
    # TODO(arbenson): put in a code-generated comment here
    add = 'inline void bl_macro_kernel_stra_abc%d( int m, int n, int k, double *packA, double *packB, ' % ( index )
    add += ', '.join(['double *%s%d' % ( mat_name, i ) for i in range(nnz)])
    add += ', int ld%s ) {' % (mat_name)
    write_line(myfile, 0, add)

    write_line( myfile, 1, 'int i, j;' )
    write_line( myfile, 1, 'aux_t aux;' )
    write_line( myfile, 1, 'aux.b_next = packB;' )

    write_line( myfile, 1, 'for ( j = 0; j < n; j += DGEMM_NR ) {' )
    write_line( myfile, 1, '    aux.n  = min( n - j, DGEMM_NR );' )
    write_line( myfile, 1, '    for ( i = 0; i < m; i += DGEMM_MR ) {' )
    write_line( myfile, 1, '        aux.m = min( m - i, DGEMM_MR );' )
    write_line( myfile, 1, '        if ( i + DGEMM_MR >= m ) {' )
    write_line( myfile, 1, '            aux.b_next += DGEMM_NR * k;' )
    write_line( myfile, 1, '        }' )

    #NEED to do: c_coeff -> pass in the parameters!

    #Generate the micro-kernel outside
    #abc_micro_kernel_gen.generate_kernel_header( my_kernel_header_file, nonzero_coeffs, index )
    #abc_micro_kernel_gen.generate_micro_kernel( my_micro_kernel_file, nonzero_coeffs, index )
    #generate the function caller


    #if nnz <= 23 and not contain_nontrivial( nonzero_coeffs ):
    #    add = '( bl_dgemm_micro_kernel_stra_abc%d ) ( k, &packA[ i * k ], &packB[ j * k ], ' % index
    #    add += '(unsigned long long) ld%s, ' % mat_name
    #    add += ', '.join( ['&%s%d[ j * ld%s + i ]' % ( mat_name, i, mat_name ) for i in range( nnz )] )
    #    add += ', &aux );'
    #    write_line(myfile, 3, add)
    #else:
    #    write_mulstrassen_kernel_caller( myfile, nonzero_coeffs )


    if nnz <= 23:
        if  not contain_nontrivial( nonzero_coeffs ):
            add = '( bl_dgemm_micro_kernel_stra_abc%d ) ( k, &packA[ i * k ], &packB[ j * k ], ' % index
            add += '(unsigned long long) ld%s, ' % mat_name
            add += ', '.join( ['&%s%d[ j * ld%s + i ]' % ( mat_name, i, mat_name ) for i in range( nnz )] )
            add += ', &aux );'
            write_line(myfile, 3, add)
        else:
            write_line( myfile, 3, 'double alpha_list[%d];' % nnz )
            add = '; '.join( [ 'alpha_list[%d]= (double)(%s)' % ( j, coeff ) for j, coeff in enumerate(nonzero_coeffs) ] )
            add += ';'
            write_line( myfile, 3, add )
            add = '( bl_dgemm_micro_kernel_stra_abc%d ) ( k, &packA[ i * k ], &packB[ j * k ], ' % index
            add += '(unsigned long long) ld%s, ' % mat_name
            add += ', '.join( ['&%s%d[ j * ld%s + i ]' % ( mat_name, i, mat_name ) for i in range( nnz )] )
            add += ', alpha_list , &aux );'
            write_line(myfile, 3, add)
    else:
        write_mulstrassen_kernel_caller( myfile, nonzero_coeffs )

    #write_mulstrassen_kernel_caller( myfile, nonzero_coeffs )

    write_line(myfile, 2, '}')
    write_line(myfile, 1, '}')

    write_line(myfile, 0, '}')  # end of function

def write_mulstrassen_kernel_caller( myfile, nonzero_coeffs ):
    nnz = len( nonzero_coeffs )
    write_line( myfile, 3, 'double alpha_list[%d];' % nnz )
    write_line( myfile, 3, 'double *c_list[%d];' % nnz )
    write_line( myfile, 3, 'unsigned long long len_c=%d;' % nnz )
    add = '; '.join( [ 'alpha_list[%d]= (double)(%s)' % ( j, coeff ) for j, coeff in enumerate(nonzero_coeffs) ] )
    add += ';'
    write_line( myfile, 3, add )
    add = '; '.join( [ 'c_list[%d] = &C%d[ j * ldC + i ]' % ( j, j ) for j, coeff in enumerate(nonzero_coeffs) ] )
    add += ';'
    write_line( myfile, 3, add )
    write_line( myfile, 3, '( bl_dgemm_asm_8x4_mulstrassen ) ( k, &packA[ i * k ], &packB[ j * k ], (unsigned long long) len_c, (unsigned long long) ldC, c_list, alpha_list, &aux );' )


def write_packm_func( myfile, coeffs, index, mat_name ):
    ''' Write the add function for a set of coefficients.  This is a custom add
    function used for a single multiply in a single fast algorithm.

    coeffs is the set of coefficients used for the add
    '''
    nonzero_coeffs = [coeff for coeff in coeffs if is_nonzero(coeff)]
    nnz = len( nonzero_coeffs )
    # TODO(arbenson): put in a code-generated comment here
    add = 'inline void pack%s_add_stra_abc%d( int m, int n, ' % (mat_name, index)
    add += ', '.join(['double *%s%d' % ( mat_name, i ) for i in range(nnz)])
    add += ', int ld%s, double *pack%s ' % (mat_name, mat_name)
    add += ') {'
    write_line(myfile, 0, add)

    write_line( myfile, 1, 'int i, j;' )

    add = 'double '
    add += ', '.join(['*%s%d_pntr' % ( mat_name, i ) for i in range(nnz)])
    add += ', *pack%s_pntr;' % mat_name
    write_line( myfile, 1, add )

    if ( mat_name == 'A' ):
        ldp  = 'DGEMM_MR'
        incp = '1'
        ldm  = 'ld%s' % mat_name
        incm = '1'
    elif ( mat_name == 'B' ):
        ldp  = 'DGEMM_NR'
        incp = '1'
        ldm  = '1'
        incm = 'ld%s' % mat_name
    else:
        print "Wrong mat_name!"
    #ldp = 'DGEMM_MR' if (mat_name == 'A') else 'DGEMM_NR'

    write_line( myfile, 1, 'for ( j = 0; j < n; ++j ) {' )
    write_line( myfile, 2, 'pack%s_pntr = &pack%s[ %s * j ];' % (mat_name, mat_name, ldp) )
    if ldm == '1':
        add = ''.join(['%s%d_pntr = &%s%d[ j ]; ' % ( mat_name, i, mat_name, i ) for i in range(nnz)])
    else:
        add = ''.join(['%s%d_pntr = &%s%d[ %s * j ]; ' % ( mat_name, i, mat_name, i, ldm ) for i in range(nnz)])
    write_line( myfile, 2, add )

    write_line( myfile, 2, 'for ( i = 0; i < %s; ++i ) {' % ldp )

    add = 'pack%s_pntr[ i ]' % mat_name + ' ='
    for j, coeff in enumerate(nonzero_coeffs):
        ind = j
        add += arith_expression_pntr(coeff, mat_name, ind, incm )
    
    add += ';'
    write_line(myfile, 3, add)

    write_line(myfile, 2, '}')
    write_line(myfile, 1, '}')

    write_line(myfile, 0, '}')  # end of function

def arith_expression_pntr(coeff, mat_name, ind, incm):
    ''' Return the arithmetic expression needed for multiplying coeff by value
    in a string of expressions.

    coeff is the coefficient
    value is a string representing the value to be multiplied by coeff
    place is the place in the arithmetic expression
    '''
    if incm == '1':
        value = '%s%d_pntr[ i ]'% ( mat_name, ind )
    else:
        value = '%s%d_pntr[ i * %s ]'% ( mat_name, ind, incm )
    if is_one(coeff):
         expr = ' %s' % value
    elif is_negone(coeff):
        expr = ' - %s' % value
    else:
        #print "coeff is not +-1!"
        expr = ' (double)(%s) * %s' % (coeff, value)

    if ind != 0 and not is_negone(coeff):
        return ' +' + expr
    return expr


def create_straprim_caller( myfile, coeffs, dims, num_multiplies, level=1 ):
    ''' Generate all of the recursive multiplication calls.

    myfile is the file to which we are writing
    coeffs is the set of all coefficients
    dims is a 3-tuple (m, k, n) of the dimensions of the problem
    '''
    for i in xrange(len(coeffs[0][0])):
        a_coeffs = [c[i] for c in coeffs[0]]
        b_coeffs = [c[i] for c in coeffs[1]]
        c_coeffs = [c[i] for c in coeffs[2]]
        write_straprim_caller(myfile, i, a_coeffs, b_coeffs, c_coeffs, dims, num_multiplies, level)


def create_straprim_abc_functions( myfile, coeffs, dims, level ):
    ''' Generate all of the recursive multiplication calls.

    myfile is the file to which we are writing
    coeffs is the set of all coefficients
    dims is a 3-tuple (m, k, n) of the dimensions of the problem
    '''
    for i in xrange(len(coeffs[0][0])):
        a_coeffs = [c[i] for c in coeffs[0]]
        b_coeffs = [c[i] for c in coeffs[1]]
        c_coeffs = [c[i] for c in coeffs[2]]
        write_straprim_abc_function( myfile, i, a_coeffs, b_coeffs, c_coeffs, dims, level )

def write_straprim_caller(myfile, index, a_coeffs, b_coeffs, c_coeffs, dims, num_multiplies, level=1):
    comment = '// M%d = (' % (index)
    comment += ' + '.join([str(c) + ' * %s' % getBlockName( 0, i, dims, level ) \
                               for i, c in enumerate(a_coeffs) if is_nonzero(c)])
    comment += ') * ('
    comment += ' + '.join([str(c) + ' * %s' % getBlockName( 1, i, dims, level ) \
                               for i, c in enumerate(b_coeffs) if is_nonzero(c)])
    comment += '); '
    comment += '; '.join([' %s += %s * M%d' % ( getBlockName( 2, i, dims, level ), c, index ) for i, c in enumerate(c_coeffs) if is_nonzero(c)])
    comment += ';'
    write_line(myfile, 1, comment)

    add = 'bl_dgemm_straprim_abc%d( ms, ns, ks, ' % index

    add += ', '.join(['%s' % getBlockName( 0, i, dims, level ) \
                      for i, c in enumerate(a_coeffs) if is_nonzero(c)])
    add += ', lda, '
    add += ', '.join(['%s' % getBlockName( 1, i, dims, level ) \
                      for i, c in enumerate(b_coeffs) if is_nonzero(c)])
    add += ', ldb, '
    add += ', '.join(['%s' % getBlockName( 2, i, dims, level ) \
                      for i, c in enumerate(c_coeffs) if is_nonzero(c)])
    add += ', ldc, packA, packB, bl_ic_nt );'
    write_line( myfile, 1, add )

def getNNZ ( coeffs ):
    nonzero_coeffs = [coeff for coeff in coeffs if is_nonzero(coeff)]
    nnz = len( nonzero_coeffs )
    return nnz

def write_straprim_abc_function( myfile, index, a_coeffs, b_coeffs, c_coeffs, dims, level ):
    comment = '// M%d = (' % (index)
    comment += ' + '.join([str(c) + ' * %s' % getBlockName( 0, i, dims, level ) \
                               for i, c in enumerate(a_coeffs) if is_nonzero(c)])
    comment += ') * ('
    comment += ' + '.join([str(c) + ' * %s' % getBlockName( 1, i, dims, level ) \
                               for i, c in enumerate(b_coeffs) if is_nonzero(c)])
    comment += '); '
    comment += '; '.join([' %s += %s * M%d' % ( getBlockName( 2, i, dims, level ), c, index ) for i, c in enumerate(c_coeffs) if is_nonzero(c)])
    comment += ';'
    write_line(myfile, 0, comment)

    add = 'void bl_dgemm_straprim_abc%d( int m, int n, int k, ' % index

    add += ', '.join(['double* %s%d' % ( 'a', i ) for i in range( getNNZ(a_coeffs) )])
    add += ', int lda, '
    add += ', '.join(['double* %s%d' % ( 'b', i ) for i in range( getNNZ(b_coeffs) )])
    add += ', int ldb, '
    add += ', '.join(['double* %s%d' % ( 'c', i ) for i in range( getNNZ(c_coeffs) )])
    add += ', int ldc, double *packA, double *packB, int bl_ic_nt ) {'

    write_line( myfile, 0, add )
    write_line( myfile, 1, 'int i, j, p, ic, ib, jc, jb, pc, pb;' )
    write_line( myfile, 1, 'for ( jc = 0; jc < n; jc += DGEMM_NC ) {' )
    write_line( myfile, 2, 'jb = min( n - jc, DGEMM_NC );' )
    write_line( myfile, 2, 'for ( pc = 0; pc < k; pc += DGEMM_KC ) {' )
    write_line( myfile, 3, 'pb = min( k - pc, DGEMM_KC );' )
    #write_line( myfile, 0, '#ifdef _PARALLEL_')
    #write_line( myfile, 3, '#pragma omp parallel for num_threads( bl_ic_nt ) private( j )' )
    #write_line( myfile, 0, '#endif')
    write_line( myfile, 3, '{')
    write_line( myfile, 4, 'int tid = omp_get_thread_num();' )
    write_line( myfile, 4, 'int my_start;' )
    write_line( myfile, 4, 'int my_end;' )
    write_line( myfile, 4, 'bl_get_range( jb, DGEMM_NR, &my_start, &my_end );' )
    write_line( myfile, 4, 'for ( j = my_start; j < my_end; j += DGEMM_NR ) {' )

    add = 'packB_add_stra_abc%d( min( jb - j, DGEMM_NR ), pb, ' % index
    add += ', '.join(['&%s%d[ pc + (jc+j)*ldb ]' % ( 'b', i ) for i in range( getNNZ(b_coeffs) )])
    add += ', ldb, &packB[ j * pb ] );'
    write_line( myfile, 5, add )
    write_line( myfile, 4, '}')
    write_line( myfile, 3, '}' )

    write_line( myfile, 0, '#ifdef _PARALLEL_')
    write_line( myfile, 0, '#pragma omp barrier')
    write_line( myfile, 0, '#endif')
    #write_line( myfile, 0, '#ifdef _PARALLEL_')
    #write_line( myfile, 3, '#pragma omp parallel num_threads( bl_ic_nt ) private( ic, ib, i )' )
    #write_line( myfile, 0, '#endif')
    write_line( myfile, 3, '{' )
    #write_line( myfile, 0, '#ifdef _PARALLEL_')
    write_line( myfile, 4, 'int tid = omp_get_thread_num();' )
    write_line( myfile, 4, 'int my_start;' )
    write_line( myfile, 4, 'int my_end;' )
    write_line( myfile, 4, 'bl_get_range( m, DGEMM_MR, &my_start, &my_end );' )
    #write_line( myfile, 0, '#else')
    #write_line( myfile, 4, 'int tid = 0;' )
    #write_line( myfile, 4, 'int my_start = 0;' )
    #write_line( myfile, 4, 'int my_end = m;' )
    #write_line( myfile, 0, '#endif')
    write_line( myfile, 4, 'for ( ic = my_start; ic < my_end; ic += DGEMM_MC ) {' )
    write_line( myfile, 5, 'ib = min( my_end - ic, DGEMM_MC );' )
    write_line( myfile, 5, 'for ( i = 0; i < ib; i += DGEMM_MR ) {' )

    add = 'packA_add_stra_abc%d( min( ib - i, DGEMM_MR ), pb, ' % index
    add += ', '.join(['&%s%d[ pc*lda + (ic+i) ]' % ( 'a', i ) for i in range( getNNZ(a_coeffs) )])
    add += ', lda, &packA[ tid * DGEMM_MC * pb + i * pb ] );'
    write_line( myfile, 6, add )

    write_line( myfile, 5, '}' )

    add = 'bl_macro_kernel_stra_abc%d( ib, jb, pb, packA + tid * DGEMM_MC * pb, packB, ' % index
    add += ', '.join(['&%s%d[ jc * ldc + ic ]' % ( 'c', i ) for i in range( getNNZ(c_coeffs) )])
    add += ', ldc );'
    write_line( myfile, 5, add )

    write_line( myfile, 4, '}' )
    write_line( myfile, 3, '}' )
    write_line( myfile, 0, '#ifdef _PARALLEL_')
    write_line( myfile, 0, '#pragma omp barrier')
    write_line( myfile, 0, '#endif')
    write_line( myfile, 2, '}' )
    write_line( myfile, 1, '}' )

    write_line( myfile, 0, '#ifdef _PARALLEL_')
    write_line( myfile, 0, '#pragma omp barrier')
    write_line( myfile, 0, '#endif')
    write_line( myfile, 0, '}' )
    write_break( myfile )

def write_abc_strassen_header( myfile ):
    write_line( myfile, 1, 'double *packA, *packB;' );
    write_break( myfile )
    write_line( myfile, 1, 'int bl_ic_nt = bl_read_nway_from_env( "BLISLAB_IC_NT" );' );
    write_break( myfile )
    write_line( myfile, 1, '//// Allocate packing buffers' );
    write_line( myfile, 1, '//packA  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_MC + 1 ) * bl_ic_nt, sizeof(double) );' );
    write_line( myfile, 1, '//packB  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_NC + 1 )           , sizeof(double) );' );

    write_line( myfile, 1, 'bl_malloc_packing_pool( &packA, &packB, n, bl_ic_nt );' )

    write_break( myfile )


def gen_abc_fmm( coeff_filename, dims, level, outfilename, micro_kernel_filename, kernel_header_filename ):

    coeffs = read_coeffs( coeff_filename )
    #print coeffs
    #print coeffs[0][0]

    #coeffs2= [ transpose( U2 ), transpose( V2 ), transpose( W2 ) ]

    with open( outfilename, 'w' ) as myfile:
        write_line( myfile, 0, '#include "%s"' % kernel_header_filename[10:] )
        write_line( myfile, 0, '#include "bl_dgemm.h"' )
        write_break( myfile )

        cur_coeffs = generateCoeffs( coeffs, level )
        #writeCoeffs( cur_coeffs )
        #writeEquation( cur_coeffs, dims, level )

        num_multiplies = len(cur_coeffs[0][0])

        create_packm_functions( myfile, cur_coeffs )

        my_micro_file = open( micro_kernel_filename, 'w' ) 
        create_micro_functions( my_micro_file, cur_coeffs, kernel_header_filename[10:] )

        my_kernel_header = open ( kernel_header_filename, 'w' )
        create_kernel_header( my_kernel_header, cur_coeffs )

        create_macro_functions( myfile, cur_coeffs )

        create_straprim_abc_functions( myfile, cur_coeffs, dims, level )


        write_line( myfile, 0, 'void bl_dgemm_strassen_abc( int m, int n, int k, double *XA, int lda, double *XB, int ldb, double *XC, int ldc )' )
        write_line( myfile, 0, '{' )

        write_abc_strassen_header( myfile )

        writePartition( myfile, dims, level )

        write_break( myfile )

        write_line( myfile, 0, '#ifdef _PARALLEL_')
        write_line( myfile, 1, '#pragma omp parallel num_threads( bl_ic_nt )' )
        write_line( myfile, 0, '#endif')
        write_line( myfile, 1, '{' )
        create_straprim_caller( myfile, cur_coeffs, dims, num_multiplies, level )
        write_line( myfile, 1, '}' )

        write_break( myfile )
        level_dim = exp_dim( dims, level )
        write_line( myfile, 1, 'bl_dynamic_peeling( m, n, k, XA, lda, XB, ldb, XC, ldc, %d * DGEMM_MR, %d, %d * DGEMM_NR );' % ( level_dim[0], level_dim[1], level_dim[2] ) )

        write_break( myfile )
        write_line( myfile, 1, '//free( packA );' )
        write_line( myfile, 1, '//free( packB );' )

        write_line( myfile, 0, '}' )

        #writePreamble( myfile, dims )
        #writePartition( myfile, dims, level )
        #cur_coeffs = generateCoeffs( coeffs, level )
        #writeCoeffs( cur_coeffs )
        #writeEquation( cur_coeffs, dims, level )
        #writeFMM( myfile, cur_coeffs, dims, level )

def main():
    try:
        coeff_file = sys.argv[1]
        print coeff_file
        dims = tuple([int(d) for d in sys.argv[2].split(',')])
        print dims

        outfile = 'a.c'
        micro_kernel_file = '../kernels/bl_dgemm_micro_kernel_stra.c'
        kernel_header_file = '../include/bl_dgemm_kernel.h'

        level = 1
        if len(sys.argv) > 3:
            level = int( sys.argv[3] )
        if len(sys.argv) > 4:
            outfile = sys.argv[4]
        if len(sys.argv) > 5:
            micro_kernel_file = sys.argv[5]
        if len(sys.argv) > 6:
            kernel_header_file = sys.argv[6]

        print outfile
        print micro_kernel_file

        print "level: " + str( level )
        print 'Generating code for %d x %d x %d' % dims
    except:
        raise Exception('USAGE: python abc_gen.py coeff_file m,n,p out_file micro_kernel_file kernel_header_file')

    gen_abc_fmm( coeff_file, dims, level, outfile, micro_kernel_file, kernel_header_file )

if __name__ == '__main__':
    main()

