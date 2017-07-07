import sys

from common import is_one, is_negone, is_nonzero, write_line, write_break, data_access, transpose, printmat, writeCoeffs, phantomMatMul, parse_coeff, read_coeffs, writeFMM, writePartition, writeEquation, getSubMatName, getBlockName, getName, generateCoeffs, exp_dim, contain_nontrivial

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

def write_stra_mat( myfile, coeff_idx, coeffs, idx, dim_name, dims, level ):
    mat_name = ( getName( coeff_idx ) )[ 0 ]
    nonzero_coeffs = [coeff for coeff in coeffs if is_nonzero(coeff)]
    add = 'stra_matrix_view<T,' + str(len(nonzero_coeffs)) + '> '
    add += mat_name + 'v' + str(idx) + '({' + dim_name + '}, {'
    #add += ', '.join( ['const_cast<T*>(%s)' % (getSubMatName(coeff_idx, i, dims, level) ) for i, c in enumerate(coeffs) if is_nonzero(c)] )
    add += ', '.join( ['const_cast<T*>(%s)' % (getBlockName(coeff_idx, i, dims, level) ) for i, c in enumerate(coeffs) if is_nonzero(c)] )
    add += '}, {'
    add += ', '.join( [ str(c) for i, c in enumerate(coeffs) if is_nonzero(c) ] )
    add += '}, {rs_' + mat_name + ', cs_' + mat_name +'});'

    write_line(myfile, 1, add)


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

    write_stra_mat( myfile, 0, a_coeffs, index, 'ms, ks', dims, level )
    write_stra_mat( myfile, 1, b_coeffs, index, 'ks, ns', dims, level )
    write_stra_mat( myfile, 2, c_coeffs, index, 'ms, ns', dims, level )

    #add = 'stra_gemm(comm, cfg, alpha, Av{0}, Bv{0}, beta, Cv{0});'.format( index )
    #add = 'straprim_naive(comm, cfg, alpha, Av{0}, Bv{0}, beta, Cv{0});'.format( index )
    add = 'straprim_ab(comm, cfg, alpha, Av{0}, Bv{0}, beta, Cv{0});'.format( index )
    write_line( myfile, 1, add )

    write_line( myfile, 1, 'comm.barrier();' )

    write_break( myfile )

def getNNZ ( coeffs ):
    nonzero_coeffs = [coeff for coeff in coeffs if is_nonzero(coeff)]
    nnz = len( nonzero_coeffs )
    return nnz


def gen_abc_fmm( coeff_filename, dims, level, outfilename ):

    coeffs = read_coeffs( coeff_filename )
    #print coeffs
    #print coeffs[0][0]

    #coeffs2= [ transpose( U2 ), transpose( V2 ), transpose( W2 ) ]

    with open( outfilename, 'w' ) as myfile:

        cur_coeffs = generateCoeffs( coeffs, level )
        #writeCoeffs( cur_coeffs )
        #writeEquation( cur_coeffs, dims, level )

        num_multiplies = len(cur_coeffs[0][0])

        writePartition( myfile, dims, level )


        write_break( myfile )

        create_straprim_caller( myfile, cur_coeffs, dims, num_multiplies, level )



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

        level = 1
        if len(sys.argv) > 3:
            level = int( sys.argv[3] )
        if len(sys.argv) > 4:
            outfile = sys.argv[4]

        print outfile

        print "level: " + str( level )
        print 'Generating code for %d x %d x %d' % dims
    except:
        raise Exception('USAGE: python abc_gen.py coeff_file m,n,p out_file')

    gen_abc_fmm( coeff_file, dims, level, outfile )

if __name__ == '__main__':
    main()

