"""
Use nevergrad to optimize tunable stockfish parameters

Interfaces between nevergrad, cutechess, and stockfish to optimize
parameters marked for tuning in sf.

See https://github.com/vondele/nevergrad4sf/blob/master/README.md for details.
"""

import math
import sys
import os.path
import shutil
import datetime
import time
import argparse
import json
import textwrap
import nevergrad as ng
from subprocess import Popen, PIPE
from cutechess_batches import CutechessExecutorBatch, calc_stats
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from concurrent.futures import ThreadPoolExecutor


def get_sf_parameters(stockfish_exe):
    """Run sf to obtain the tunable parameters"""

    process = Popen(stockfish_exe, shell=True, stdin=PIPE, stdout=PIPE)
    output = process.communicate(input=b"quit\n")[0]
    if process.returncode != 0:
        sys.stderr.write("get_sf_parameters: failed to execute command: %s\n" % command)
        sys.exit(1)

    # parse for parameter output
    params = {}
    for line in output.decode("utf-8").split("\n"):
        if "Stockfish" in line:
            continue
        if not "," in line:
            continue
        split_line = line.split(",")
        params[split_line[0]] = [
            int(split_line[1]),
            int(split_line[2]),
            int(split_line[3]),
        ]

    return params


def var2int(**variables):
    """Round variables to ints"""
    for name in variables:
        variables[name] = int(round(variables[name]))
    return variables


def ng4sf(
    stockfish,
    stockfishRef,
    cutechess,
    book,
    tc,
    nevergrad_evals,
    do_restart,
    games_per_batch,
    cutechess_concurrency,
    evaluation_concurrency,
):
    """
      nevergrad for sf: optimize parameters in a tuning enabled stockfish.

      specify binary names, tc, number of points to evaluate, restart or not,
      games per batch, cutechess concurrency, and evaluation batch concurrency
    """

    # ready to run with mpi
    size = MPI.COMM_WORLD.Get_size()
    print()
    if size > 1:
        print(
            "Launched ... with %d mpi ranks (1 master, %d workers)." % (size, size - 1)
        )
        print(flush=True)
    else:
        sys.stderr.write("ng4sf needs to run under mpi with at least 2 MPI ranks.\n")
        sys.exit(1)

    # print summary
    print("stockfish binary                          : ", stockfish)
    print("stockfish reference binary                : ", stockfishRef)
    print("cutechess binary                          : ", cutechess)
    print("book                                      : ", book)
    print("time control                              : ", tc)
    print("restart                                   : ", do_restart)
    print("nevergrad batch evaluations               : ", nevergrad_evals)
    print("batch size in games                       : ", games_per_batch)
    print("cutechess concurrency                     : ", cutechess_concurrency)
    print("batch evaluation concurrency:             : ", evaluation_concurrency)
    print(flush=True)

    # get info from sf
    sf_params = get_sf_parameters(stockfish)
    print(
        "About to optimize the following %d %s: %s"
        % (
            len(sf_params),
            "parameter" if len(sf_params) == 1 else "parameters",
            sf_params,
        )
    )
    print(flush=True)

    # our choice of making mpi_subbatches, i.e. mpi processes per batch (this could be tunable for performance?)
    mpi_subbatches = 2 * (
        (size - 1 + evaluation_concurrency - 1) // evaluation_concurrency
    )

    # creating the batch
    batch = CutechessExecutorBatch(
        cutechess=cutechess,
        stockfish=stockfish,
        stockfishRef=stockfishRef,
        book=book,
        tc=tc,
        rounds=((games_per_batch + 1) // 2 + mpi_subbatches - 1) // mpi_subbatches,
        concurrency=cutechess_concurrency,
        batches=mpi_subbatches,
        executor=MPIPoolExecutor(),
    )
    restartFileName = "ng_restart.pkl"

    # Create a dictionary describing to nevergrad the variables of our black box function
    variables = {}
    for v in sf_params:
        if (
            sf_params[v][1] != sf_params[v][2]
        ):  # Let equal bounds imply fixed not a parameter.
            variables[v] = (
                ng.p.Scalar(init=float(sf_params[v][0]))
                .set_bounds(
                    lower=float(sf_params[v][1]),
                    upper=float(sf_params[v][2]),
                    method="constraint",
                )
                .set_mutation(
                    sigma=(float(sf_params[v][2]) - float(sf_params[v][1])) / 4
                )
            )

    # init ng optimizer, restarting as hardcoded
    instrum = ng.p.Instrumentation(**variables)
    if not do_restart:
        optimizer = ng.optimizers.TBPSA(
            parametrization=instrum,
            budget=nevergrad_evals,
            num_workers=evaluation_concurrency,
        )
    else:
        if os.path.isfile(restartFileName):
            optimizer = ng.optimizers.TBPSA.load(restartFileName)
        else:
            sys.exit("Missing restart file: %s\n" % restartFileName)

    start_time = datetime.datetime.now()

    # with this executor, we can parallelize over evaluation_concurrency.
    executor = ThreadPoolExecutor(max_workers=evaluation_concurrency)
    evalpoints = []
    evalpoints_submitted = 0
    evalpoints_running = 0
    for i in range(evaluation_concurrency):
        x = optimizer.ask()
        evalpoints.append([x, executor.submit(batch.run, var2int(*x.args, **x.kwargs))])
        time.sleep(0.1)  # try to give some time to submit all batches of this point
        evalpoints_submitted = evalpoints_submitted + 1
        evalpoints_running = evalpoints_running + 1

    # optimizer loop
    ng_iter = 0
    previous_recommendation = None
    while evalpoints_running > 0:

        # find the point which is ready
        ready_batch = -1
        while ready_batch is -1:
            time.sleep(0.1)
            for i in range(evaluation_concurrency):
                if evalpoints[i][1].done():
                    ready_batch = i
                    evalpoints_running = evalpoints_running - 1
                    break

        # use this point to inform the optimizer.
        x = evalpoints[ready_batch][0]
        result = evalpoints[ready_batch][1].result()
        stats = calc_stats(result)

        optimizer.tell(x, -stats["score"])
        recommendation = var2int(**optimizer.provide_recommendation().kwargs)
        current_time = datetime.datetime.now()
        used_time = current_time - start_time
        evals_done = evalpoints_submitted - evalpoints_running

        print("evaluation  : %8d" % evals_done)
        print("   evaluated: ", var2int(**x.kwargs))
        print("   time used: %8.3f" % used_time.total_seconds())
        print(
            "   games/s  : %8.3f"
            % (batch.total_games * evals_done / used_time.total_seconds())
        )
        print(
            "   score    : %8.3f +- %8.3f"
            % (stats["score"] * 100, stats["score_error"] * 100)
        )
        print("   Elo      : %8.3f +- %8.3f" % (stats["Elo"], stats["Elo_error"]))
        print("   LOS      : %8.3f" % (100 * stats["LOS"]))

        # make a backup of the old restart and dump current state
        if os.path.exists(restartFileName):
            shutil.move(restartFileName, restartFileName + ".bak")
        optimizer.dump(restartFileName)

        if recommendation != previous_recommendation:
            ng_iter = ng_iter + 1
            print()
            print(
                "optimal at iter %d after %d %s and %d games : %s"
                % (
                    ng_iter,
                    evals_done,
                    "evaluation" if evals_done == 1 else "evaluations",
                    batch.total_games * evals_done,
                    str(recommendation),
                )
            )
            with open("optimal.json", "w") as outfile:
                json.dump(recommendation, outfile)

        # queue the next point for evaluation.
        if evalpoints_submitted < nevergrad_evals:
            x = optimizer.ask()
            evalpoints[ready_batch] = [
                x,
                executor.submit(batch.run, var2int(*x.args, **x.kwargs)),
            ]
            evalpoints_submitted = evalpoints_submitted + 1
            evalpoints_running = evalpoints_running + 1

        print(flush=True)
        previous_recommendation = recommendation

    return recommendation


if __name__ == "__main__":

    class MyFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter,
        description=textwrap.dedent(
            """\
                  Use nevergrad to optimize tunable stockfish parameters.

                  This program requires mpi to run. A typical invocation could be:
                     mpirun -np 3 python3 -m mpi4py.futures nevergrad4sf.py -tc 1.0+0.01 -g 2 -cc 2 -ec 3 --ng 10

                  More documentation at:
                     https://github.com/vondele/nevergrad4sf/blob/master/README.md

                  """
        ),
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        default="./stockfish",
        help="Name of the stockfish binary to be optimized",
    )
    parser.add_argument(
        "--stockfishRef",
        type=str,
        default=None,
        help="Optional, name of the reference stockfish binary to be used, defaults to the --stockfish argument.",
    )
    parser.add_argument(
        "--cutechess",
        type=str,
        default="./cutechess-cli",
        help="Name of the cutechess binary",
    )
    parser.add_argument(
        "--book",
        type=str,
        default="./noob_3moves.epd",
        help="opening book in epd or pgn fomat",
    )
    parser.add_argument(
        "-tc", "--tc", type=str, default="10.0+0.1", help="time control"
    )
    parser.add_argument(
        "-g",
        "--games_per_batch",
        type=int,
        default=20000,
        help="Number of games per evaluation point",
    )
    parser.add_argument(
        "-cc",
        "--cutechess_concurrency",
        type=int,
        default=8,
        help="Number of concurrent games per cutechess worker",
    )
    parser.add_argument(
        "-ec",
        "--evaluation_concurrency",
        type=int,
        default=3,
        help="Number of concurrently evaluated points",
    )
    parser.add_argument(
        "-ng",
        "--ng_evals",
        type=int,
        default=10000,
        help="Number of nevergrad evaluation points",
    )
    parser.add_argument(
        "--restart", action="store_true", help="Restart a previous optimization"
    )
    args = parser.parse_args()

    result = ng4sf(
        args.stockfish,
        args.stockfishRef if args.stockfishRef else args.stockfish,
        args.cutechess,
        args.book,
        args.tc,
        args.ng_evals,
        args.restart,
        args.games_per_batch,
        args.cutechess_concurrency,
        args.evaluation_concurrency,
    )
    print("Optimization finished with optimal parameters: ", result)
