// This code is a direct translation of Lawn96 Summa Algorithm, with some
// tweaks
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#include "mkl.h"

#include <algorithm>
#include <limits>
#include <stdint.h>
#include <iostream>
#include <random>
#include <numeric>
#include <getopt.h>
#include <sstream>
#include <type_traits>
#include <iomanip>
#include <functional>
#include <set>
#include <map>

#include "tblis.h"
#include "util/time.hpp"
#include "util/tensor.hpp"
#include "util/random.hpp"

#include "internal/3t/mult.hpp"


using namespace std;
using namespace tblis;
using namespace stl_ext;

#define PRINT_VECTOR( name ) \
    std::cout << #name << ": " ; \
for (auto &elem : name) \
{ \
    std::cout << elem << " "; \
} \
std::cout << std::endl;

#define min(x,y) ( (x) < (y) ? (x) : (y) )

const int i_one = 1;
const double d_one = 1.0, d_zero = 0.0;

void RING_Bcast( double *buf, int count, MPI_Datatype type, int root, MPI_Comm comm)
{
   int me, np, sendto, recvfrom;
   MPI_Status status;

   MPI_Comm_rank ( comm, &me );
   MPI_Comm_size ( comm, &np );
   recvfrom = ( me - 1 + np ) % np;
   sendto   = ( me + 1 ) % np ;
   if ( me != root )
   {
      MPI_Recv ( buf, count, type, recvfrom, MPI_ANY_TAG, comm, &status );
   }
   if ( sendto != root )
   {
      MPI_Send ( buf, count, type, sendto, 0, comm );
   }
}

void summa_pdgemm ( int mybb, int myaa, int myee, int myii, int myjj, int mymm, int nb, double alpha, tensor_view<double> AT, int lda, vector<label_type>& idx_A, tensor_view<double> BT, int ldb, vector<label_type>& idx_B, double beta, tensor_view<double> CT, int ldc, vector<label_type>& idx_C,
                    int *m_a, int *n_a, int *m_b, int *n_b, int *m_c, int *n_c, 
                    MPI_Comm comm_row, MPI_Comm comm_col, double *work1_buf, double *work2_buf )
{
   int myrow, mycol; // my row and column index: 0<=myrow<nprow
   int nprow, npcol; // number of processor rows and columns
   int i, j, kk, iwrk; // index variables
   int icurrow, icurcol; // index of row & col that hold current row & col
   int ii, jj; // local index (on icurrow/icurcol) of row and column
   double *temp; // pointer used in pdgemm_abt - which isn't present now
   double *p; // no idea what this is for. Must be important

   
   MPI_Comm_rank ( comm_row, &mycol );
   MPI_Comm_rank ( comm_col, &myrow );

   icurrow = 0;  // Assumes node (0,0) owns the first element!
   icurcol = 0;  // Assumes node (0,0) owns the first element!
   ii = 0;
   jj = 0; 
   /* Add this back if we want A*B^T
   temp = (double *) malloc ( m_c[myrow]*nb*sizeof(double) );
   if ( temp == NULL )
   {
       fprintf(stderr,"malloc error within pdgemm. This sucks\n");
       exit(-1);
   }
   */
   // In the original LAWN96 paper, iwrk was an unitialized variable
   // That isn't necessarily a bug, but it is in poor taste

   tensor_view<double> AT_sub(AT);
   tensor_view<double> BT_sub(BT);

   iwrk = nb;
   for ( kk = 0; kk < myee; kk+=iwrk )
   {
      iwrk = min ( nb,   m_b[icurrow]-ii );
      iwrk = min ( iwrk, n_a[icurcol]-jj );
      // pack current iwrk columns of A into work1

      map<char, len_type> lengths_work;

      lengths_work['b'] = m_a[myrow];
      lengths_work['a'] = n_b[mycol];

      lengths_work['e'] = iwrk;
      lengths_work['i'] = myii; lengths_work['j'] = myjj; lengths_work['m'] = mymm;

      vector<len_type> len_work1, len_work2;
      for (char c : idx_A) { len_work1.push_back(lengths_work.at(c)); }
      for (char c : idx_B) { len_work2.push_back(lengths_work.at(c)); }

      tensor_view<double> work1T(len_work1, work1_buf);
      tensor_view<double> work2T(len_work2, work2_buf);

      if ( mycol == icurcol )  
      {
          //Tensor add
          // AT -> work1T, copy along the e dimension, with the blocksize iwrk.
          int orig_length = AT_sub.length( 2 );
          AT_sub.length( 2, iwrk );
          add( 1.0, AT_sub, idx_A.data(), 0.0, work1T, idx_A.data() );
          AT_sub.shift ( 2, iwrk );
          AT_sub.length( 2, orig_length - iwrk );
      }
      if ( myrow == icurrow )
      {
         //Tensor add
         // BT -> work2T, copy along the e dimension, with the blocksize iwrk.
         int orig_length = BT_sub.length( 1 );
         BT_sub.length( 1, iwrk );
         add( 1.0, BT_sub, idx_B.data(), 0.0, work2T, idx_B.data() );
         BT_sub.shift ( 1, iwrk );
         BT_sub.length( 1, orig_length - iwrk );

      }
      // Broadcast work1 and work2
      RING_Bcast( work1_buf, m_a[myrow]*iwrk*mymm*myjj, MPI_DOUBLE, icurcol, comm_row );
      RING_Bcast( work2_buf, n_b[mycol]*iwrk*myii*mymm, MPI_DOUBLE, icurrow, comm_col );
      // Update local block
#ifdef DEBUG
      printf("(%d,%d) kk=%d DGEMM call on m n k = %d %d %d ldabc=%d %d %d\n",myrow,mycol,kk,m_c[myrow],n_c[mycol], iwrk, lda, ldb, ldc );
#endif

  #ifdef STR1ABC
      tblis::internal::impl = tblis::internal::BLIS_BASED;
      stra_mult((double)(1), work1T, idx_A.data(),
              work2T, idx_B.data(),
              (double)(1), CT, idx_C.data());
  #elif defined(STR1AB)
      tblis::internal::impl = tblis::internal::STRA_AB;
      stra_mult((double)(1), work1T, idx_A.data(),
              work2T, idx_B.data(),
              (double)(1), CT, idx_C.data());
  #elif defined(STR1N)
      tblis::internal::impl = tblis::internal::STRA_NAIVE;
      stra_mult((double)(1), work1T, idx_A.data(),
              work2T, idx_B.data(),
              (double)(1), CT, idx_C.data());
  #elif defined(STR2ABC)
      tblis::internal::impl = tblis::internal::BLIS_BASED;
      stra_mult_2level((double)(1), work1T, idx_A.data(),
              work2T, idx_B.data(),
              (double)(1), CT, idx_C.data());
  #elif defined(STR2AB)
      tblis::internal::impl = tblis::internal::STRA_AB;
      stra_mult_2level((double)(1), work1T, idx_A.data(),
              work2T, idx_B.data(),
              (double)(1), CT, idx_C.data());
  #elif defined(STR2N)
      tblis::internal::impl = tblis::internal::STRA_NAIVE;
      stra_mult_2level((double)(1), work1T, idx_A.data(),
              work2T, idx_B.data(),
              (double)(1), CT, idx_C.data());
  #elif defined(TBLIS)
      tblis::internal::impl = tblis::internal::BLIS_BASED;
      mult((double)(1), work1T, idx_A.data(),
              work2T, idx_B.data(),
              (double)(1), CT, idx_C.data());
  #else
      printf( "No such implementations!" );
      exit( -1 );
  #endif

      // Update indices: icurcol, icurrow, ii, jj 
      ii += iwrk;
      jj += iwrk;


      if ( jj >= n_a[icurcol] ) { icurcol++; jj = 0; }
      if ( ii >= m_b[icurrow] ) { icurrow++; ii = 0; }


   }
   // free ( temp );
}
