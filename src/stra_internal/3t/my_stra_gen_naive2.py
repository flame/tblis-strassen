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

def getActualBlockIndex( coeff_idx, item_idx, dims, level=1 ):
    my_mat_name = ( getName( coeff_idx ) )[ 0 ]

    if( coeff_idx == 0 ):
        mm = dims[0]
        nn = dims[1]
    elif( coeff_idx == 1 ):
        mm = dims[1]
        nn = dims[2]
    elif( coeff_idx == 2 ):
        mm = dims[0]
        nn = dims[2]
    else:
        print "Wrong coeff_idx\n"

    #my_partition_ii = item_idx / nn
    #my_partition_jj = item_idx % nn
    submat_idx = ""
    dividend = item_idx
    mm_base = 1
    nn_base = 1
    ii_idx = 0
    jj_idx = 0
    for level_idx in range( level ):
        remainder = dividend % ( mm * nn )
        #remainder -> i, j (m_axis, n_axis)
        ii = remainder / nn
        jj = remainder % nn
        ii_idx = ii * mm_base + ii_idx
        jj_idx = jj * nn_base + jj_idx
        #submat_idx = str(remainder) + submat_idx
        dividend = dividend / ( mm * nn )
        mm_base = mm_base * mm
        nn_base = nn_base * nn

    #return  [ str(ii_idx), str(jj_idx) ] 

    return  str(ii_idx * nn_base + jj_idx)


def write_stra_mat( myfile, coeff_idx, coeffs, idx, dim_name, dims, level ):
    mat_name = ( getName( coeff_idx ) )[ 0 ]
    nonzero_coeffs = [coeff for coeff in coeffs if is_nonzero(coeff)]

    add = 'std::array<unsigned, ' + str(len(nonzero_coeffs)) + '> ' + mat_name + str(idx) + '_subid = {'
    add += ', '.join(['%s' % getActualBlockIndex( coeff_idx, i, dims, level ) \
                               for i, c in enumerate(coeffs) if is_nonzero(c)])
    add += '};'
    write_line(myfile, 1, add)

    add = 'std::array<T,' + str(len(nonzero_coeffs)) + '> ' + mat_name + str(idx) + '_coeff_list = {'
    add += ', '.join( [ str(c) for i, c in enumerate(coeffs) if is_nonzero(c) ] )
    add += '};'
    write_line(myfile, 1, add)

    add = 'stra_tensor_view<T,' + str(len(nonzero_coeffs)) + '> '
    add += mat_name + 'v' + str(idx)
    add += '(my_len_' + dim_name[0] +  ', '
    add += 'my_len_' + dim_name[1] + ', '
    add += mat_name + '_divisor, const_cast<T*>(' + mat_name + '), '
    add += mat_name + str(idx) + '_subid, ' + mat_name + str(idx) + '_coeff_list, '
    add += 'my_stride_' + mat_name + '_' + dim_name[0] + ','
    add += ' my_stride_' + mat_name + '_' + dim_name[1] 
    add += ');'

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

    write_stra_mat( myfile, 0, a_coeffs, index, ['AC', 'AB'], dims, level )
    write_stra_mat( myfile, 1, b_coeffs, index, ['AB', 'BC'], dims, level )
    write_stra_mat( myfile, 2, c_coeffs, index, ['AC', 'BC'], dims, level )

    myfile.write( \
'''\
    if (Cv{0}.stride(!row_major) == 1)
    {{
        Av{0}.transpose();
        Bv{0}.transpose();
        Cv{0}.transpose();
        straprim_naive2(comm, cfg, alpha, Bv{0}, Av{0}, beta, Cv{0});
    }} else {{
        straprim_naive2(comm, cfg, alpha, Av{0}, Bv{0}, beta, Cv{0});
    }}
'''.format( index ) )

    #Av{0}.swap(Bv{0});

    #add = 'stra_gemm(comm, cfg, alpha, Av{0}, Bv{0}, beta, Cv{0});'.format( index )
    #write_line( myfile, 1, add )

    write_line( myfile, 1, 'comm.barrier();' )

    write_line( myfile, 1, '//std::cout << "stra_internal/stra_mult_M{0}:" << std::endl;'.format( index ) )
    write_line( myfile, 1, '//print_tensor_matrix( ct );' )

    write_break( myfile )
               


def getNNZ ( coeffs ):
    nonzero_coeffs = [coeff for coeff in coeffs if is_nonzero(coeff)]
    nnz = len( nonzero_coeffs )
    return nnz

def write_divisor_initializer( myfile, dims, level ):
    level_dim = exp_dim( dims, level )
    write_line(myfile, 1, 'const std::array<unsigned,2> A_divisor={%d,%d};'%(level_dim[0],level_dim[1]))
    write_line(myfile, 1, 'const std::array<unsigned,2> B_divisor={%d,%d};'%(level_dim[1],level_dim[2]))
    write_line(myfile, 1, 'const std::array<unsigned,2> C_divisor={%d,%d};'%(level_dim[1],level_dim[2]))
    write_break(myfile)



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


        write_divisor_initializer( myfile, dims, level )
        
        #writePartition( myfile, dims, level )
        #write_break( myfile )

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

