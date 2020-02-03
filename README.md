# nevergrad4sf

Optimize parameters in the chess engine stockfish using nevergrad.

## short description

nevergrad4sf is a collection of python programs that interfaces between
[nevergrad](https://github.com/facebookresearch/nevergrad) (ng),
[cutechess](https://github.com/cutechess/cutechess), and
[stockfish](https://github.com/official-stockfish/Stockfish) (sf)
to optimize parameters marked for tuning in sf.

The score of a match for the given parameter set is used as the objective function,
and is optimized with the TBPSA method (
[1](https://github.com/facebookresearch/nevergrad/issues/273#issuecomment-531284285),
[2](https://homepages.fhv.at/hgb/New-Papers/PPSN16_HB16.pdf),
[3](https://www.lamsade.dauphine.fr/~cazenave/papers/games_cec.pdf)
) as implemented in ng. This method is is effective for noisy objective functions, and appears
to be robust, easy to use and relatively effective. nevergrad4sf aims to be easy to use.

To deal with the high computational cost of optimization, the implementation features
concurrency at multiple levels. At the lowest level, concurrency is available via cutechess,
which allows to concurrently play games of a match on a single computer. At the higher level,
nevergrad4sf employ MPI to allow for multi-node parallelism. This can be used to split the
games of a match in multiple batches, and to perform multiple nevergrad evaluations concurrently.
Even though the code can be used on a standard desktop, it aims to scale to hundreds of nodes,
and thousands of cores.

## requirements

The following list of software packages is needed, a hint is provided on how to obtain those packages on Ubuntu Linux:
* Python 3.6 or later is required for nevergrad
	* `sudo apt-get install python3`
* A functional mpi implementation is needed for mpi4py
	* `sudo apt-get install mpich`
	* If multi-node parallelism is wanted, explore the [mpi documentation](https://www.mpich.org/documentation/guides/) on how to enable this.
* nevergrad and mpi4py are installed
	* `pip3 install nevergrad mpi4py`
* `cutechess-cli` as well as a suitable opening book (e.g. `noob_3moves.epd`) are required
	* `git clone https://github.com/official-stockfish/books.git`
	* unzip the mentioned files, picking a cutechess version that matches your architecture
* a development version of stockfish is needed
	* `git clone https://github.com/official-stockfish/Stockfish.git`
	* merge the tuning branch:  `cd Stockfish/src; git merge origin/tune`
	* make suitable modifications inserting TUNE() commands, as detailed on the [fishtest wiki](https://github.com/glinscott/fishtest/wiki/Creating-my-first-test#tuning-with-spsa)
	* compile `cd Stockfish/src; make -j ARCH=x86-64-modern profile-build`
* nevergrad4sf is required
	* `git clone https://github.com/vondele/nevergrad4sf.git`, until a pip install-able version appears


Verify that the requirements are properly installed:
* `mpirun -np 4 hostname`, should launch 4 processes, each printing the name of the host.
* `echo "quit" | ./stockfish`, should print a list of parameters that can be tuned.
* `python3 nevergrad4sf.py --help`, should print some info on the command-line arguments.

In the above and in the following following, it is assumed that all needed files
(`cutechess-cli`, `stockfish`, `noob_3moves.epd`, `nevergrad4sf.py`, `cutechess_batches.py`)
have been copied in the working directory, use suitable paths otherwise.

## invocation

Finally, invocation should be easy, either with full option names:

```
mpirun -np 16 python3 -m mpi4py.futures nevergrad4sf.py --cutechess ./cutechess_cli --stockfish ./stockfish --book noob_3moves.epd --tc 1.0+0.01 --games_per_batch 20000 --cutechess_concurrency 8 --evaluation_concurrency 3 --ng_evals 100
```
or equivalently, using shortcuts and defaults:
```
mpirun -np 16 python3 -m mpi4py.futures nevergrad4sf.py -tc 1.0+0.01 -g 20000 -cc 8 -ec 3 --ng 100
```
This will start the optimization process, computing 100 batches each 20000 games at
time control 1.0+0.01, using 8 * 16 cores.  Some intermediate results will be printed
during optimization.  Additionally, files are written during optimization, namely:
* `optimal.json` a dictionary of best parameters so far.
* `ng_restart.pkl` a file that can be used to restart an interrupted optimization (using the `--restart` option).

To verify the quality of the obtained optimal parameters it is possible
to play a single match using the `optimal.json` parameters, for example verify Elo gain
using 80000 games at various time controls:
```
mpirun -np 16 python3 -m mpi4py.futures cutechess_batches.py -tc 1.0+0.01 -g 80000 -cc 8
mpirun -np 16 python3 -m mpi4py.futures cutechess_batches.py -tc 10.0+0.1 -g 80000 -cc 8
mpirun -np 16 python3 -m mpi4py.futures cutechess_batches.py -tc 60.0+0.6 -g 80000 -cc 8
```

## General tips and tricks

The optimizer is relatively insensitive to the parameters. Yet, the following guidelines might help to get results quickly:

* The number of games per batch should presumably be in the range 5000 - 50000. The actual number is not so important, as nevergrad can deal with noise. Yet, it makes sense to keep the estimated Elo error per point similar to or smaller than the expected Elo gain. Larger batch sizes also reduce overheads in cutechess and nevergrad. For small batch sizes, nevergrad will eventually perform more evaluations per iteration.
* Number of parameters to tune: the computational cost grows with the number of parameters. `N` parameters will at least require `5N` nevergrad evaluations (each requiring `games_per_batch` games) per nevergrad (TBPSA) iteration. Several iterations are required to converge the optimization, probably 10 or more. Count on O(1M) games for convergence, even with few parameters.
* The optimizer is relatively insensitive to the ranges specified for the parameters via the `TUNE()` macro in sf. Yet, the range of each tunable parameter should be selected suitably to speedup convergence. Obviously, the ranges given should include the expected minimum. The optimizer will explore points in the full allowed range initially, and ideally the effect on the score / Elo of changes to each parameter within its range should be similar. As the optimization proceeds, points near the optimum will be evaluated predominantly.
* The time control should typically be 1.0+0.01 or higher (shorter TC might lead to many time losses). It appears that the optimal parameters are often time sensitive, i.e. can be verified to be a gain at the VSTC used for tuning, but regress at STC or LTC. Ideally, tuning is performed at the TC that is most relevant.

## Tuning parallelism

The MPI parallelism in nevergrad4sf is of the master-slave type,
i.e. 1 process (rank) will coordinate the work, while the
remaining `np-1` processes will run batches of cutechess games.
The node running the master might be underutilized a bit.
Depending on the compute node architecture, it might be useful
to have 1 or more MPI ranks per node. For example, given 16 nodes each with 32 cores,
it might be useful to have `np = 64` (16 * 4) MPI ranks,
i.e. 4 ranks per node, and a cutechess concurrency of 8.

The evaluation concurrency should be >= 2 to make sure resources are always used fully.
As the total number of cores becomes large compared to the number of games in a batch,
it makes sense to increase the evaluation concurrency, or to increase the number of games
per batch. The total number of 'games in flight' per core
(`games_per_batch * evaluation_concurrency / ((np - 1) * cutechess_concurrency) `)
should be > ~20. `evaluation_concurrency` should presumably be kept smaller than `4 * N`,
as the optimizer efficiency could be reduced otherwise. For example, to optimize
N=10 parameters on 256 nodes with each 32 cores, the following invocation,
with 4 MPI ranks per node, is assumed to be efficient:
```
mpirun -np 1024 python3 -m mpi4py.futures nevergrad4sf.py -tc 1.0+0.01 -g 40000 -cc 8 -ec 40 --ng 10000
```

Finally, it is recommended to avoid binding ranks or threads to cores,
as thread creation in both cutechess and stockfish is relatively dynamic. Consult `mpirun --help`
for information, a typical option to mpirun might be `--bind-to none`.
[More advanced options](https://wiki.mpich.org/mpich/index.php/Using_the_Hydra_Process_Manager)
could be used to fine-tune binding by the expert.  If the reported games played per second
is surprisingly low, binding is the first aspect to investigate.
