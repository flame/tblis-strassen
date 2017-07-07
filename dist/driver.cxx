#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mkl.h"
#include "mpi.h"

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

#ifndef MAX
#define MAX(x,y) ((x)<(y)?(y):(x))
#endif
#ifndef MIN
#define MIN(x,y) ((x)>(y)?(y):(x))
#endif
#ifndef ABS
#define ABS(x) MAX((x),-(x))
#endif

#ifndef ITER
   #define ITER 1
#endif

#if !defined(ILP64) && !defined(LP64)
   #define LP64
#endif
#if defined(ILP64) && defined(LP64)
   #error Both ILP64 and LP64 are set at the same time- only set one
#endif

// Get an integer input value in a given range with a given prompt
#ifdef LP64
void getivalue ( int *i, char *cinp, int lowerbnd, int upperbnd )
#else
void getivalue ( size_t *i, char *cinp, int lowerbnd, int upperbnd )
#endif
{
   int looper = 1;

   while ( looper == 1 )
   {
      printf("%s ",cinp);
#ifdef ILP64
      scanf("%ld",i);
#else
      scanf("%d",i);
#endif
      if ( lowerbnd <= upperbnd ) looper = (*i < lowerbnd) || (*i > upperbnd);
   }
}

// Find a quick mapping to get the grid information
void get_local_grid ( int *myrow, int *mycol, int *nprow, int *npcol )
{
   int me, size, nprow_tmp, npcol_tmp, foundone;

   MPI_Comm_rank ( MPI_COMM_WORLD, &me );
   MPI_Comm_size ( MPI_COMM_WORLD, &size );

   if ( size == 1 )
   {
      *myrow = 0; *mycol = 0; *nprow = 1; *npcol = 1;
      return ;
   }
   if ( size % 2 != 0 ) // Odd grid uses 1xsize mapping
   {
      *nprow = 1; *npcol = size; *myrow = 0; *mycol = me;
      return ;
   }
   for ( nprow_tmp = 1, foundone = 100000 ; nprow_tmp < size ; nprow_tmp++ )
   {
      npcol_tmp = size / nprow_tmp;
      if ( nprow_tmp * npcol_tmp == size )
      {
         // Make as square as possible
         if ( abs(nprow_tmp-npcol_tmp) < foundone ) 
         {
            foundone = abs(nprow_tmp-npcol_tmp);
            *nprow = nprow_tmp;
            *npcol = npcol_tmp;
         }
      }
   }
   if ( foundone == 100000 )
   {
      fprintf(stderr,"That's weird. Couldn't find a proc mapping\n");
      exit(-1);
   }
   // Assume column major mapping
   *mycol = (int) ( me / *nprow );
   *myrow = me - (*mycol)*(*nprow);
}

// If we want a loop over [I1,I2], this computes a local LI1 and LI2 to use
// If we don't own anything in that range, then *LI2 < *LI1
void mylocalpart ( int I1, int I2, int NB, int MYROC, int NPROCS, int ISRC,
                   int *LI1, int *LI2 )
{
    int gcpy, iblk, ROCSRC;
    int mydist, nblocks, numroc, extrablks;

    gcpy = I1-1;
    iblk = (int) ( gcpy / (NB) );
    ROCSRC = (iblk + (ISRC)) % (NPROCS);
    *LI1 = ((int)(iblk/(NPROCS)) + 1) * (NB) + 1;
    mydist = ((NPROCS)+(MYROC)-(ISRC)) % (NPROCS);
    if ( mydist >= (iblk % (NPROCS)) )
    {
       if ( (MYROC) == (ROCSRC) )
       {
          *LI1 += ((gcpy) % (NB));
       }
       *LI1 -= (NB);
    }
    nblocks = (int) ((I2) / (NB));
    numroc = ((int)(nblocks / (NPROCS))) * (NB);
    extrablks = nblocks % (NPROCS);
    if ( mydist < extrablks )
    {
       numroc += (NB);
    } else if ( mydist == extrablks ) {
       numroc += ((I2) % (NB));
    }
    *LI2 = numroc;
}

int main(int argc, char **argv)
{

    vector<len_type> len_m =
        random_product_constrained_sequence<len_type, ROUND_NEAREST>(random_number(1, 3), 10);

   int me, size;
   int ibuf[20], aa, bb, ee, ii, jj, mm, mb, nb, kb, itmp;
   int *m_a, *n_a, *m_b, *n_b, *m_c, *n_c;
   int i, j, lm, ln, lk1, lk2, lda, ldb, ldc, iter;
   int myrow, mycol, nprow, npcol;
   double *A, *B, *C, *A1, *B1, *C1, *work1, *work2;
   double alpha = 1.0, beta = 1.0;
   double dtmp, timer, dres, flops;
   double tmp_timer;
   extern double dsecnd_();
   extern double drand48();
   MPI_Comm comm_row, comm_col;
   extern void summa_pdgemm ( int bb, int aa, int ee, int ii, int jj, int mm, int nb, double alpha, tensor_view<double> AT, int lda, vector<label_type>& idx_A, tensor_view<double> BT, int ldb, vector<label_type>& idx_B, double beta, tensor_view<double> CT, int ldc, vector<label_type>& idx_C,
                    int *m_a, int *n_a, int *m_b, int *n_b, int *m_c, int *n_c, 
                    MPI_Comm comm_row, MPI_Comm comm_col, double *work1_buf, double *work2_buf );

   MPI_Init( &argc, &argv );
   MPI_Comm_rank(MPI_COMM_WORLD, &me);
   MPI_Comm_size(MPI_COMM_WORLD, &size );

   if ( me == 0 )
   {
#ifndef CSV
      printf("\nUSAGE: %s a b e i j m nb iter\n",argv[0]);
      printf("Distributed Memory SUMMA algorithm for C_{mxn} <- C + A*B\n");
      printf("Default values:...\n");
#endif

      if ( argc > 1 ) aa = atoi(argv[1]); else getivalue(&aa,"Enter aa >= 0 : ",0,100000);
      if ( argc > 2 ) bb = atoi(argv[2]); else getivalue(&bb,"Enter bb >= 0 : ",0,100000);
      if ( argc > 3 ) ee = atoi(argv[3]); else getivalue(&ee,"Enter cc >= 0 : ",0,100000);

      if ( argc > 4 ) ii = atoi(argv[4]); else getivalue(&ii,"Enter ii >= 0 : ",0,100000);
      if ( argc > 5 ) jj = atoi(argv[5]); else getivalue(&jj,"Enter jj >= 0 : ",0,100000);
      if ( argc > 6 ) mm = atoi(argv[6]); else getivalue(&mm,"Enter mm >= 0 : ",0,100000);

      if ( argc > 7 ) nb   = atoi(argv[7]); else getivalue(&nb,"Enter nb >= 1 physical blocksize for 2D wrap mapping: ",1,100000);
      if ( argc > 8 ) iter = atoi(argv[8]); else iter = MAX(ITER,1);

      aa = MAX(aa,1);
      bb = MAX(bb,1);
      ee = MAX(ee,1);
      ii = MAX(ii,1);
      jj = MAX(jj,1);
      mm = MAX(mm,1);

      nb = MAX(nb,1);
      iter = MAX(iter,1);

      ibuf[0] = aa;
      ibuf[1] = bb;
      ibuf[2] = ee;
      ibuf[3] = ii;
      ibuf[4] = jj;
      ibuf[5] = mm;
      ibuf[6] = nb;
      ibuf[7] = iter;

      if ( size > 1 )
      {
         MPI_Bcast ( ibuf, 8, MPI_INT, 0, MPI_COMM_WORLD );
      }
   } else {
      MPI_Bcast ( ibuf, 8, MPI_INT, 0, MPI_COMM_WORLD );
      aa = ibuf[0];
      bb = ibuf[1];
      ee = ibuf[2];
      ii = ibuf[3];
      jj = ibuf[4];
      mm = ibuf[5];
      nb = ibuf[6];
      iter = ibuf[7];
   }

#if defined(COMPLEX) || defined(COMPLEX16)
   flops = (8.0*(((double)aa)*((double)bb)*((double)ee)*((double)ii)*((double)jj)*((double)mm)))/1000000000.0;
#else
   flops = (2.0*(((double)aa)*((double)bb)*((double)ee)*((double)ii)*((double)jj)*((double)mm)))/1000000000.0;
#endif
  
   mb = nb; // Don't need this, but why not?
   kb = nb; // Don't need this, but why not?

   get_local_grid( &myrow, &mycol, &nprow, &npcol );   


   if ( (me == 0) || (me ==nprow*npcol-1) || (size <= 8) ) 
   {
      printf("Hello from %d = (%d,%d) of (%d,%d)\n",me,myrow,mycol,nprow,npcol);
   }

   if ( me == 0 )
   {
#ifndef CSV
      printf("%s compiled with: ",argv[0]);
  #ifdef TBLIS
      printf("-DTBLIS ");
  #endif
  #ifdef STR1ABC
      printf("-DSTR1ABC ");
  #endif
  #ifdef STR1AB
      printf("-DSTR1AB ");
  #endif
  #ifdef STR1N
      printf("-DSTR1N ");
  #endif
  #ifdef STR2ABC
      printf("-DSTR2ABC ");
  #endif
  #ifdef STR2AB
      printf("-DSTR2AB ");
  #endif
  #ifdef STR2N
      printf("-DSTR2N ");
  #endif
  #ifdef LP64
      printf("-DLP64 ");
  #endif
  #ifdef ILP64
      printf("-DILP64 ");
  #endif
  printf("-DITER=%d ",iter);
  #ifdef SERIAL_TEST
      printf("-DSERIAL_TEST\n");
      printf("WARNING: SERIAL_TEST makes a global copy of all arrays for debug\n");
      printf(" purposes. It will run slow and is meant only for debugging!\n");
  #else
      printf("\n");
  #endif
#endif
   }

   itmp = MPI_Comm_split ( MPI_COMM_WORLD, myrow, mycol, &comm_row );
   itmp = MPI_Comm_split ( MPI_COMM_WORLD, mycol, myrow, &comm_col );

   m_a = (int *) malloc ( nprow * sizeof(int) );
   m_b = (int *) malloc ( nprow * sizeof(int) );
   m_c = (int *) malloc ( nprow * sizeof(int) );
   for ( i = 0 ; i < nprow ; i++ )
   {
      mylocalpart ( 1, bb, mb, i, nprow, 0, &itmp, &lm );
      m_a[i] = lm;
      m_c[i] = lm;
      mylocalpart ( 1, ee, kb, i, nprow, 0, &itmp, &lk1 );
      m_b[i] = lk1;
      if ( me == 0 && nprow <= 2 ) printf("m_a[%d]=%d m_b[%d]=%d\n",i,m_a[i],i,m_b[i]);
   }
   n_a = (int *) malloc ( npcol * sizeof(int) );
   n_b = (int *) malloc ( npcol * sizeof(int) );
   n_c = (int *) malloc ( npcol * sizeof(int) );
   for ( i = 0 ; i < npcol ; i++ )
   {
      mylocalpart ( 1, ee, kb, i, npcol, 0, &itmp, &lk2 );
      n_a[i] = lk2;
      mylocalpart ( 1, aa, nb, i, npcol, 0, &itmp, &ln );
      n_b[i] = ln;
      n_c[i] = ln;
      if ( me == 0  && npcol <= 2 ) printf("n_a[%d]=%d n_b[%d]=%d\n",i,n_a[i],i,n_b[i]);
   }

   // Local A is lm x lk2 *(ii*jj) , Local B is lk1 x ln *(mm*jj) , Local C is lm x ln *(ii*mm)
   lm  = m_a[myrow];
   lk2 = n_a[mycol];
   lk1 = m_b[myrow];
   ln  = n_b[mycol];

   lda = lm;  // Can be bigger of course...
   ldb = lk1; // Can be bigger of course...
   ldc = lm;  // Can be bigger of course...

   // Initialize Local Tensors
   //A = (double *) mkl_malloc ( lda * lk2 * mm * jj * sizeof(double), 128 );
   //B = (double *) mkl_malloc ( ldb * ln  * ii * mm * sizeof(double), 128 );
   //C = (double *) mkl_malloc ( ldc * ln  * ii * jj * sizeof(double), 128 );

   string idx_A_, idx_B_, idx_C_;
   idx_A_="bmej"; idx_B_="aeim"; idx_C_="abij";
   vector<label_type> idx_A(idx_A_.begin(), idx_A_.end());
   vector<label_type> idx_B(idx_B_.begin(), idx_B_.end());
   vector<label_type> idx_C(idx_C_.begin(), idx_C_.end());
   map<char, len_type> lengths_all;
   // b -> m; a -> n; e -> k;
   lengths_all['a'] = ln; lengths_all['b'] = lm; lengths_all['e'] = lk1;
   lengths_all['i'] = ii; lengths_all['j'] = jj; lengths_all['m'] = mm;
   vector<len_type> len_A, len_B, len_C;
   for (char c : idx_A) { len_A.push_back(lengths_all.at(c)); }
   for (char c : idx_B) { len_B.push_back(lengths_all.at(c)); }
   for (char c : idx_C) { len_C.push_back(lengths_all.at(c)); }
   tensor<double> AT(len_A, 1.0);
   tensor<double> BT(len_B, 2.0);
   tensor<double> CT(len_C, 0.0);

   //srand48 ( (long int) me );
   //for ( i = 0 ; i < lda*lk2 * ii * jj; i++ ) A[i]= 1.0 - 2.0*drand48();
   //for ( i = 0 ; i < ldb*ln  * mm * jj; i++ ) B[i]= 1.0 - 2.0*drand48();
   //for ( i = 0 ; i < ldc*ln  * ii * mm; i++ ) C[i]= 1.0 - 2.0*drand48();

   map<char, len_type> lengths_work;
   // b -> m_a; a -> n_b; e -> k/?;
   // Set up work arrays
   for ( i = 0, itmp = 0 ; i < nprow; i++ ) itmp = MAX(itmp,m_a[i]); 
   itmp += 200;
   lengths_work['b'] = itmp;
   for ( i = 0, itmp = 0 ; i < npcol; i++ ) itmp = MAX(itmp,n_b[i]); 
   itmp += 200;
   lengths_work['a'] = itmp;
   lengths_work['e'] = MAX(MAX(mb,nb),kb);
   lengths_work['i'] = ii; lengths_work['j'] = jj; lengths_work['m'] = mm;

   vector<len_type> len_work1, len_work2;
   for (char c : idx_A) { len_work1.push_back(lengths_work.at(c)); }
   for (char c : idx_B) { len_work2.push_back(lengths_work.at(c)); }
   tensor<double> work1T(len_work1, 0.0);
   tensor<double> work2T(len_work1, 0.0);

   if ( me == 0 ) printf("Calling summa_pdgemm\n");

   for ( i = 0 ; i < iter ; i++ )
   {
       dtmp = dsecnd_(); // warm-up
       dtmp = dsecnd_();
       itmp = MPI_Barrier ( MPI_COMM_WORLD );

       summa_pdgemm ( bb, aa, ee, ii, jj, mm, nb, alpha, AT, lda, idx_A, BT, ldb, idx_B, beta, CT, ldc, idx_C,
                      m_a, n_a, m_b, n_b, m_c, n_c, comm_row, comm_col, 
                      work1T.data(), work2T.data() );

       itmp = MPI_Barrier ( MPI_COMM_WORLD );
       if ( i == 0 ) {
           timer = dsecnd_() - dtmp;
       } else {
           tmp_timer = dsecnd_() - dtmp;
           timer = timer < tmp_timer ? timer : tmp_timer;
       }
   }
                
   if ( me == 0 ) printf("Done calling summa_pdgemm: ave. %g secs %g Gflops\n",
                         timer, flops/timer );
   itmp = MPI_Barrier ( MPI_COMM_WORLD );

   MPI_Finalize();

}   
