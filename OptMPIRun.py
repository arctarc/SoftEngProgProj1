import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from OptMPI import one_energy, all_energy, get_order, MC_step

import os
os.environ["OMP_NUM_THREADS"] = str(1)

# Telling OpenMP to use one thread per process, due to errors when running.

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Importing MPI and writing out the required setup.

#=======================================================================
def initdat(nmax):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  arr (float(nmax,nmax)) = array to hold lattice.
    """
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr
#=======================================================================
def plotdat(arr,pflag,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """
    if pflag==0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
#=======================================================================
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)

    if rank == 0:
        with open(filename,"w") as FileOut:
        # Here, only the first process will be writing the output file.
            
            print("#=====================================================",file=FileOut)
            print("# File created:        {:s}".format(current_datetime),file=FileOut)
            print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
            print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
            print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
            print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
            print("#=====================================================",file=FileOut)
            print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
            print("#=====================================================",file=FileOut)
            for i in range(nsteps+1):
                print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
#=======================================================================
def main(program, nsteps, nmax, temp, pflag):

    if rank == 0:
        lattice = initdat(nmax)
    else:
        lattice = None

    lattice = comm.bcast(lattice, root=0)

    # Only the first process will be creating the lattice, the other processes will receive the lattice later.

    step_splits = np.array_split(np.arange(nsteps + 1), size)
    local_nsteps = len(step_splits[rank])

    energy = np.zeros(local_nsteps)
    ratio = np.zeros(local_nsteps)
    order = np.zeros(local_nsteps)

    # Instead of running all 'nsteps', we split it among the different processes. np.array_split() ensures an even split
    # whilst accounting for remainders.

    energy[0] = all_energy(lattice, nmax)
    ratio[0] = 0.5
    order[0] = get_order(lattice, nmax)

    initial = time.time()
    for it in range(1, local_nsteps):
        ratio[it] = MC_step(lattice, temp, nmax)
        energy[it] = all_energy(lattice, nmax)
        order[it] = get_order(lattice, nmax)

    # Start the timer.
    # Each process independently computes its respective steps.

    final = time.time()
    runtime = final - initial

    # Stop timing.

    recv_counts = np.array([len(split) for split in step_splits], dtype=np.int32)
    displacements = np.insert(np.cumsum(recv_counts[:-1]), 0, 0)
    global_energy = None
    global_ratio = None
    global_order = None
    if rank == 0:
        global_energy = np.zeros(nsteps + 1)
        global_ratio = np.zeros(nsteps + 1)
        global_order = np.zeros(nsteps + 1)
    comm.Gatherv(sendbuf=energy, recvbuf=(global_energy, recv_counts, displacements, MPI.DOUBLE), root=0)
    comm.Gatherv(sendbuf=ratio, recvbuf=(global_ratio, recv_counts, displacements, MPI.DOUBLE), root=0)
    comm.Gatherv(sendbuf=order, recvbuf=(global_order, recv_counts, displacements, MPI.DOUBLE), root=0)

    # Each process sends its results back to the first process.
    # recv_counts tell the first process how many elements it is being sent by each process.
    # displacements is the starting index for each recieved segment.
    # Gatherv() is used to combine arrays that may not be the same size.

    if rank == 0:
        print(f"{program}: Size: {nmax}, Steps: {nsteps}, T*: {temp:.3f}, Order: {global_order[-1]:.3f}, Time: {runtime:.6f} s")
        savedat(lattice, nsteps, temp, runtime, global_ratio, global_energy, global_order, nmax)
        plotdat(lattice, pflag, nmax)

    # Only the first process will print and save the results.
    
#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        if rank == 0:
            print("Usage: mpiexec -n <NUM_PROCESSES> python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))

        # Only the first process will print the usage.
        # The usage print has also slightly changed because a different command must be used for mpi4py.
    
#=======================================================================