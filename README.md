# Strassen's Algorithm for Tensor Contraction

Tensor contraction (TC) is an important computational kernel widely used in numerous applications. It is a multi-dimensional generalization of matrix multiplication (GEMM). While Strassen's algorithm for GEMM is well studied in theory and practice, extending it to accelerate TC has not been previously pursued. Thus, we believe this to be the first paper to demonstrate how one can in practice speed up tensor contraction with Strassen's algorithm. By adopting a Block-Scatter-Matrix format, a novel matrix-centric tensor layout, we can conceptually view TC as GEMM for a general stride storage, with an implicit tensor-to-matrix transformation. This insight enables us to tailor a recent state-of-the-art implementation of [Strassen's algorithm](http://dl.acm.org/citation.cfm?id=3014983) to a recent [state-of-the-art TC](https://github.com/devinamatthews/tblis), avoiding explicit transpositions (permutations) and extra workspace, and reducing the overhead of memory movement that is incurred. Performance benefits are demonstrated with a performance model as well as in practice on modern single core, multicore, and distributed memory parallel architectures, achieving up to 1.3× speedup. The resulting implementations can serve as a drop-in replacement for various applications with significant speedup.

# Related Links
* [TBLIS](https://github.com/devinamatthews/tblis/)
* [Generating Families of Practical Fast Matrix Multiplication Algorithms](https://github.com/flame/fmm-gen)
* [Tensor Contraction Code Generator](https://github.com/HPAC/tccg)

# Installation

  Information about building, installing, and using the code can be found in [artifact](./artifact.pdf).

* Compile the source code:
  Replace `$INSTALL_PATH` with the installation path the user wants to specify.
```
  $ tar -zxvf tblis-strassen.tar.gz
  $ cd tblis-strassen
  $ autoreconf -i
  $ ./configure --prefix=$INSTALL_PATH \
      --disable-shared --enable-config=sandybridge
  $ make && make install
```

* Parameters for the test routine:
  The executable `./test_strassen` is generated after the compilation in the `bin` folder. Here is the detailed specification for the arguments.
  Usage:
```
  ./test_strassen -m $NIm -n $NJn -k $NPk -niter $niter -level $L -impl $IMPL -seed $seed
```
  `$NIm`: NIm; `$NJn`: NJn; `$NPk`: NPk; `$niter`: number of iterations;
  `$L`: levels of Strassen ( `L=0` will invoke regular tblis TC routine);
  `$IMPL`: implementations [ 1: ABC Strassen, 2: AB Strassen, 3: Naive Strassen ];
  `$seed`: random seeds.

  Usage:
```
  ./test_strassen -niter $niter -level $L -impl $IMPL -seed $seed -file $filename
```
  `$niter`: number of iterations;
  `$L`: levels of Strassen ( `L=0` will invoke regular tblis TC routine);
  `$IMPL`: implementations [ 1: ABC Strassen, 2: AB Strassen, 3: Naive Strassen ];
  `$filename`: the benchmark file.
  `$seed`: random seeds.

* Options for random seed $seed:
  if `$seed == -1`: default seeding.
  if `$seed` is equal to any number other than -1, `$seed` will be used as the random seed for the synthetic tensor sizes, shapes, and permutations.
  if not specified, using time as the random seed.

* Set up environment variables:
  Replace `$core_num` with the number of cores the user wants to run.
```
  $ export OMP_NUM_THREADS=$core_num
  $ export KMP_AFFINITY=compact
```
  Note: if hyper-threading is enabled, the following alternative must be used:
```
  $ export KMP_AFFINITY=compact,1
```

* Single node experiment (synthetic tensor contractions): Replace `$NIm` with NIm, `$NJn` with NJn, `$NPk` with NPk, `$niter` with the iterations the user wants to run, `$L` with number of levels of Strassen to employ (L = 0 will invoke regular tblis TC routine), and `$IMPL` with the implementations the user wants to test. `$IMPL` has the following options, [ 1: ABC Strassen, 2: AB Strassen, 3: Naive Strassen ].
```
  $ cd bin
  $ ./test_strassen -m $NIm -n $NJn -k $NPk -niter $niter -level $L -impl $IMPL
```
  We have prepared the testing scripts in bin folder so the user can run it directly to reproduce the experiments in this paper (e.g. `square_run_1core.sh`).

* Single node experiment (real-world benchmark):
```
  $ cd bin
  $ ./test_strassen -file $benchmark_filename
```
  We have prepared the benchmark from [Tensor Contraction Benchmark](https://github.com/HPAC/tccg/tree/master/benchmark) as `tcb.txt` and the test scripts (e.g. `tcb_run_1core.sh`) to make it easier for the user to reproduce the result presented in this paper.

* Single node experiment (shape-dependence experiment):
```
  $ cd bin
  $ ./test_strassen -file $shape_filename
```
  We have prepared the test cases we use as `shape.txt` and the test scripts (e.g. `shape_run_1core.sh`) to make it easier for the user to reproduce the result presented in this paper.

* Distributed memory experiment:
  Replace `$P` with the edge length of the `P × P` mesh the user wants to test on.
  Note: replace `$INSTALL_PATH` variable in Makefile with the TBLIS installation path specified in compilation step.
```
  $ cd dist
  $ make
  $ sbatch ./submit_${P}x${P}.sh
```

* User specified benchmark format:
  index bundle of C,A,B & dimension length.
  e.g. `C(a,b,c) += A(d,c,a) * B(d,b)`:
  `abc dca db & a:4;b:8;c:2;d:8;`
  
  More example can be found in each line in `bin/tcb.txt` and `bin/shape.txt`

# Citation
For those of you looking for the appropriate article to cite regarding Strassen's algorithm for tensor contraction, we
recommend citing our
[TR](https://arxiv.org/pdf/1704.03092.pdf): 

```
@TechReport{FLAWN84,
  author = {Jianyu Huang and Devin A. Matthews and Robert A. {v}an~{d}e~{G}eijn},
  title = {{S}trassen's Algorithm for Tensor Contraction},
  institution = {The University of Texas at Austin, Department of Computer Science},
  type = {FLAME Working Note \#84,},
  number = {TR-17-02},
  year = {2017},
  url = {https://apps.cs.utexas.edu/apps/sites/default/files/tech_reports/TensorStrassen.pdf}
}
``` 

# Acknowledgement
This material was partially sponsored by grants from the National Science Foundation (Awards ACI-1550493), by Intel Corporation through an Intel Parallel Computing Center grant, and by a gift from Qualcomm Foundation.

_Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation (NSF)._

# Feedback
Bugs can be reported to jianyu@cs.utexas.edu. Any feedback, feature requests, and comments are welcome.

