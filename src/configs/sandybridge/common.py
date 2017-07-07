import sys

def is_one(x):
    return x == 1 or x == 1.0 or x == '1' or x == '1.0'

def is_negone(x):
    return x == -1 or x == -1.0 or x == '-1' or x == '-1.0'

def is_nonzero(x):
    return x != 0 and x != 0.0 and x != -0.0 and x != '0' and x != '0.0' and x != '-0.0'

def contain_nontrivial( coeffs ):
    for coeff in coeffs:
        if ( ( not is_one( coeff ) ) and ( not is_negone( coeff ) ) and ( is_nonzero( coeff ) ) ):
            return True
    return False

def write_line(myfile, num_indent, code):
    ''' Write the line of code with num_indent number of indents. '''
    myfile.write(' ' * 4 * num_indent + code + '\n')

def write_break(myfile, num_breaks=1):
    ''' Write a break (new line) in the file myfile. '''
    myfile.write('\n' * num_breaks)

def data_access( mat_name, ind="" ):
    return '%s[ i + j * ld%s ]' % ( mat_name + str(ind), mat_name )

def transpose(coeffs):
    ''' Given a list of rows, return a list of columns. '''
    return [[x[i] for x in coeffs] for i in range(len(coeffs[0]))]

#def transpose( twodim_list ):
#    result_list = []
#    for jj in range( len( twodim_list[ 0 ] ) ):
#        cur_list = []
#        for ii in range( len( twodim_list ) ):
#            cur_list.append( int(twodim_list[ ii ][ jj ]) )
#            #cur_list.append( twodim_list[ ii ][ jj ] )
#        result_list.append( cur_list )
#
#    return result_list

#def transpose( twodim_list ):
#    result_list = []
#    for jj in range( len( twodim_list[ 0 ] ) ):
#        #cur_list = []
#        #for row in twodim_list:
#        #    cur_list.append( row[ jj ] )
#        #result_list.append( cur_list )
#        result_list.append( [ row[ jj ] for row in twodim_list ] )
#
#    return result_list

#def transpose( twodim_list ):
#    return [[row[i] for row in twodim_list] for i in range( len( twodim_list[ 0 ] ) )]

def printmat( X ):
    for jj in range( len(X[0]) ):
        mystr = ""
        for ii in range( len(X) ):
            #mystr += '{:04.2f}'.format( float(X[ii][jj]) ) + " "
            mystr += '%5.2f' % ( float(X[ii][jj]) ) + " "
        print mystr

def writeCoeffs( coeffs ):
    U = transpose( coeffs[ 0 ] )
    V = transpose( coeffs[ 1 ] )
    W = transpose( coeffs[ 2 ] )
    print ( "U:" )
    printmat( U )
    print ( "V:" )
    printmat( V )
    print ( "W:" )
    printmat( W )
    print ""

def genSubmatID( submat_id_queue, split_num ):
    res_submat_id_queue = []
    for elem in submat_id_queue:
        for idx in range( split_num ):
            res_submat_id_queue.append( elem + '_' + str(idx) )
    return res_submat_id_queue

# composition operation?
def phantomMatMul( A, B ):
    m_A = len( A[0] )
    n_A = len( A    )
    m_B = len( B[0] )
    n_B = len( B    )

    m_C = m_A * m_B
    n_C = n_A * n_B

    C = [ [0 for x in range( m_C )] for y in range( n_C ) ]
    #print C

    for colid_A in range( n_A ):
        vec_A = A[ colid_A ]
        for rowid_A in range( m_A ):
            elem_A = vec_A[ rowid_A ]
            if ( elem_A != 0 ):
                for colid_B in range( n_B ):
                    vec_B = B[ colid_B ]
                    for rowid_B in range( m_B ):
                        elem_B = vec_B[ rowid_B ]
                        if ( elem_B != 0 ):
                            rowid_C = rowid_A * m_B + rowid_B
                            colid_C = colid_A * n_B + colid_B
                            elem_C = str( float(elem_A) * float(elem_B) )
                            C[ colid_C ][ rowid_C ] = elem_C

    #print C
    return C



def parse_coeff(coeff):
    ''' Parse a coefficient. The grammar is:
        
        * --> *i | -* | *p | [a-z] | [floating point number]
        p --> 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
        *i --> 1 / (*)
        -* --> -(*)
        *p --> (*)^p

        So -x2i is parsed as - (1 / ((x)^2))
    '''
    coeff = coeff.strip()
    # First try to convert to float
    try:
        val = float(coeff)
        return coeff
    except:
        pass
    
    # Parameterized coefficient
    if len(coeff) == 1:
        # Coeff is like 'x'.  We will use 'x' instead of whatever is provided.
        # For now, this means that we only support one paramterized coefficient.
        return 'x'
    elif coeff[0] == '(':
        assert(coeff[-1] == ')')
        expr = coeff[1:-1].split('+')
        return '(' + ' + '.join([parse_coeff(e) for e in expr]) + ')'
    elif coeff[0] == '-':
        return '-(%s)' % parse_coeff(coeff[1:])
    elif coeff[-1] == 'i':
        return '1.0 / (%s)' % parse_coeff(coeff[:-1])
    else:
        # Test for a multiplier out in front
        try:
            mult = float(coeff[0])
            return '%s * (%s)' % (mult, parse_coeff(coeff[1:]))
        except:
            pass

        # Test for an exponent
        try:
            exp = int(coeff[-1])
            return ' * '.join([parse_coeff(coeff[:-1]) for i in xrange(exp)])
        except:
            raise Exception('Cannot parse coefficient: %s' % coeff)


def read_coeffs(filename):
    ''' Read the coefficient file.  There is one group of coefficients for each
    of the three matrices.

    filename is the name of the file from which coefficients are read
    '''
    coeffs = []
    with open(filename, 'r') as coeff_file:
        curr_group = []
        for line in coeff_file:
            if line[0] == '#':
                if len(curr_group) > 0:
                    coeffs.append(curr_group)
                    curr_group = []
            else:
                curr_group.append([parse_coeff(val) for val in line.split()])
    coeffs.append(curr_group)
    # There should be three sets of coefficients: one for each matrix.
    if (len(coeffs) < 3):
        raise Exception('Expected three sets of coefficients!')
    return coeffs

def writeFMM( myfile, coeffs, dims, level=1 ):
    print "eq_num:"+str(len( coeffs[0][0] ))
    print "coeff_num:"+str(len( coeffs ))
    for eq_index in range( len( coeffs[0][0] ) ):
        for coeff_index in range( len(coeffs) ):
            #print "coeff_index:" + str(coeff_index)
            name_list = getName( coeff_index ) # 0: a, gamma; 1: b, delta; 2: c, alpha
            coeff_list = transpose( coeffs[ coeff_index ] )
            my_eq_coeff_list = coeff_list[ eq_index ]
            write_line( myfile, 0, "len_{0} = {1};".format( name_list[ 0 ], str(sum([ abs(int(elem_coeff)) for elem_coeff in my_eq_coeff_list ])) ) )

            nz_index = 0
            for item_index in range( len(my_eq_coeff_list) ):
                if ( my_eq_coeff_list[ item_index ] != 0 ):
                    #write_line( myfile, 0, str( coeff_index ) + " " + str( item_index ) )
                    write_line( myfile, 0, "{0}_list[{1}] = {2}; {3}_list[{1}] = {4};".format( name_list[0], str(nz_index), getBlockName( coeff_index, item_index, dims, level ), name_list[1], my_eq_coeff_list[ item_index ] ) )
                    nz_index += 1
        write_line( myfile, 0,
"""bl_dgemm_str_abc( ms, ns, ks,
                    len_a, lda,
                    a_list, gamma_list,
                    len_b, ldb,
                    b_list, delta_list,
                    len_c, ldc,
                    c_list, alpha_list,
                    packA, packB, bl_ic_nt
                    );""" )
        write_break( myfile )


def writeSubmat( myfile, mat_name, dim1, dim2, split1, split2, src_mat_id ): 
    decl = "double *"
    sep_symbol = ""
    for ii in range( split1 ):
        for jj in range( split2 ):
            decl+=sep_symbol+mat_name+str(src_mat_id)+'_'+str(ii * split2 + jj)
            sep_symbol=", *"
    decl+=";"
    write_line( myfile, 1, decl )

    for ii in range( split1 ):
        for jj in range( split2 ):
            write_line( myfile, 1, "bl_acquire_mpart( {0}, {1}, {2}, ld{3}, {4}, {5}, {6}, {7}, &{2}{8} );".format( dim1, dim2, mat_name+str(src_mat_id), mat_name, split1, split2, ii, jj, '_'+str(ii * split2 + jj) ) )

def exp_dim( dims, level ):
    res = [ 1, 1, 1 ]
    for i in range( level ):
        res[ 0 ] = res[ 0 ] * dims[ 0 ]
        res[ 1 ] = res[ 1 ] * dims[ 1 ]
        res[ 2 ] = res[ 2 ] * dims[ 2 ]
    return tuple( res )

def writePartition( myfile, dims, level=1 ):
    #write_line( myfile, 0, "assert(m % {0} == 0);".format( dims[0] ) );
    #write_line( myfile, 0, "assert(k % {0} == 0);".format( dims[1] ) );
    #write_line( myfile, 0, "assert(n % {0} == 0);".format( dims[2] ) );

    write_line( myfile, 1, "int ms, ks, ns;" )
    write_line( myfile, 1, "int md, kd, nd;" )
    write_line( myfile, 1, "int mr, kr, nr;" )
    write_line( myfile, 1, "double *a = XA, *b= XB, *c = XC;" )
    write_break( myfile )

    level_dim = exp_dim( dims, level )

    write_line( myfile, 1, "mr = m %% ( %d * DGEMM_MR ), kr = k %% ( %d ), nr = n %% ( %d * DGEMM_NR );" % ( level_dim[0], level_dim[1], level_dim[2] ) )
    write_line( myfile, 1, "md = m - mr, kd = k - kr, nd = n - nr;" )

    write_break( myfile )

    triple_combinations = [
        ( "a", "ms", "ks", dims[0], dims[1] ),
        ( "b", "ks", "ns", dims[1], dims[2] ),
        ( "c", "ms", "ns", dims[0], dims[2] )
    ]
    for ( mat_name, dim1, dim2, split1, split2 ) in triple_combinations:
        write_line( myfile, 1, "ms=md, ks=kd, ns=nd;" )
        submat_id_queue = [""]
        for level_id in range( level ):

            for src_mat_id in submat_id_queue:
                writeSubmat( myfile, mat_name, dim1, dim2, split1, split2, src_mat_id )

            #Generate next level myqueue
            submat_id_queue = genSubmatID( submat_id_queue, split1 * split2 )

            # Get the current submat size
            if ( level_id != level - 1 ):
                write_line( myfile, 1, "ms=ms/{0}, ks=ks/{1}, ns=ns/{2};".format( dims[0], dims[1], dims[2] ) )

            write_break( myfile )

        write_break( myfile )

    write_line( myfile, 1, "ms=ms/{0}, ks=ks/{1}, ns=ns/{2};".format( dims[0], dims[1], dims[2] ) )
           
    write_break( myfile )

def getActualMatName( idx ):
    if ( idx == 0 ):
        matname = "A"
    elif( idx == 1 ):
        matname = "B"
    elif( idx == 2 ):
        matname = "C"
    else:
        print "Not supported!\n"
    return matname

def getActualBlockName( coeff_index, item_index, dims, level=1 ):
    my_mat_name = getActualMatName( coeff_index )

    if( coeff_index == 0 ):
        mm = dims[0]
        nn = dims[1]
    elif( coeff_index == 1 ):
        mm = dims[1]
        nn = dims[2]
    elif( coeff_index == 2 ):
        mm = dims[0]
        nn = dims[2]
    else:
        print "Wrong coeff_index\n"

    #my_partition_ii = item_index / nn
    #my_partition_jj = item_index % nn
    submat_index = ""
    dividend = item_index
    mm_base = 1
    nn_base = 1
    ii_index = 0
    jj_index = 0
    for level_index in range( level ):
        remainder = dividend % ( mm * nn )
        #remainder -> i, j (m_axis, n_axis)
        ii = remainder / nn
        jj = remainder % nn
        ii_index = ii * mm_base + ii_index
        jj_index = jj * nn_base + jj_index
        #submat_index = str(remainder) + submat_index
        dividend = dividend / ( mm * nn )
        mm_base = mm_base * mm
        nn_base = nn_base * nn

    return my_mat_name + "(" + str( ii_index ) + "," + str( jj_index ) + ")"


def writeEquation( coeffs, dims, level ):
    for eq_index in range( len( coeffs[0][0] ) ):
        m_mat_name = "M"+str(eq_index)

        my_eq_str = ""
        for coeff_index in range( len(coeffs) ):
            #print "coeff_index:" + str(coeff_index)
            name_list = getName( coeff_index ) # 0: a, gamma; 1: b, delta; 2: c, alpha
            coeff_list = transpose( coeffs[ coeff_index ] )
            my_eq_coeff_list = coeff_list[ eq_index ]

            if ( coeff_index == 0 ): #A
                my_eq_str = my_eq_str + m_mat_name + "=( "
            elif ( coeff_index == 1 ): #B
                my_eq_str = my_eq_str + " )( "
            elif ( coeff_index == 2 ): #C
                my_eq_str += " );\n  "
            else:
                print "Coeff_index not supported!\n"

            nz_index = 0
            for item_index in range( len(my_eq_coeff_list) ):
                if ( is_nonzero( my_eq_coeff_list[ item_index ] ) ):

                    mat_name = getActualBlockName( coeff_index, item_index, dims, level )
                    if ( coeff_index == 0 or coeff_index == 1 ): # A or B
                        mat_prefix = ""
                        if ( is_negone( my_eq_coeff_list[ item_index ] ) ):
                            mat_prefix = "-"
                        elif ( is_one( my_eq_coeff_list[ item_index ] ) ):
                            if ( nz_index == 0 ):
                                mat_prefix = ""
                            else:
                                mat_prefix = "+"
                        else:
                            mat_prefix = "+(" + str( my_eq_coeff_list[ item_index ] )+")"
                            #print "%d:%s" % ( item_index, my_eq_coeff_list[ item_index ] )
                            #print "entry should be either 1 or -1!"
                        my_eq_str += mat_prefix + mat_name
                    elif ( coeff_index == 2 ):
                        mat_suffix = ""
                        if ( is_negone( my_eq_coeff_list[ item_index ] ) ):
                            mat_suffix = "-="
                        elif ( is_one( my_eq_coeff_list[ item_index ] ) ):
                            mat_suffix = "+="
                        else:
                            mat_suffix = "+=(" + str( my_eq_coeff_list[ item_index ] ) + ") "
                            #print "%d:%s" % ( item_index, my_eq_coeff_list[ item_index ] )
                            #print "entry should be either 1 or -1!"
                        my_eq_str += mat_name + mat_suffix + m_mat_name + ";"
                    else:
                        print "Coeff_index not support!\n"
                    #write_line( myfile, 0, str( coeff_index ) + " " + str( item_index ) )
                    #write_line( myfile, 0, "{0}_list[{1}] = {2}; {3}_list[{1}] = {4};".format( name_list[0], str(nz_index), getBlockName( coeff_index, item_index, dims, level ), name_list[1], my_eq_coeff_list[ item_index ] ) )
                    nz_index += 1
        print my_eq_str
        #print ""

def num_nonzero(arr):
    ''' Returns number of non-zero entries in the array arr. '''
    return len(filter(is_nonzero, arr))


def getBlockName( coeff_index, item_index, dims, level=1 ):
    my_mat_name = (getName( coeff_index )) [ 0 ]

    if( coeff_index == 0 ):
        mm = dims[0]
        nn = dims[1]
    elif( coeff_index == 1 ):
        mm = dims[1]
        nn = dims[2]
    elif( coeff_index == 2 ):
        mm = dims[0]
        nn = dims[2]
    else:
        print "Wrong coeff_index\n"

    #my_partition_ii = item_index / nn
    #my_partition_jj = item_index % nn
    submat_index = ""
    dividend = item_index
    for ii in range( level ):
        remainder = dividend % ( mm * nn )
        submat_index = '_' + str(remainder) + submat_index
        #submat_index = submat_index + str(remainder) 
        dividend = dividend / ( mm * nn )

    return my_mat_name + str( submat_index )

def getName( idx ):
    if ( idx == 0 ):
        my_list = [ 'a', 'gamma' ]
    elif( idx == 1 ):
        my_list = [ 'b', 'delta' ]
    elif( idx == 2 ):
        my_list = [ 'c', 'alpha' ]
    else:
        my_list = []
        print "Not supported!\n"
    return my_list

def generateCoeffs( coeffs, level ):
    U = transpose( coeffs[ 0 ] )
    V = transpose( coeffs[ 1 ] )
    W = transpose( coeffs[ 2 ] )
 
    UM = U
    VM = V
    WM = W

    for ii in range( level - 1 ):
        UM = phantomMatMul( UM, U )
        VM = phantomMatMul( VM, V )
        WM = phantomMatMul( WM, W )

    #print ( "U2:" )
    #printmat( U2 )
    #print ( "V2:" )
    #printmat( V2 )
    #print ( "W2:" )
    #printmat( W2 )

    res_coeffs = [ transpose( UM ), transpose( VM ), transpose( WM ) ]
    return res_coeffs

 
