"""
Basic Python Lebwohl-Lasher code.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  During the
time-stepping, an array containing two domains is used; these
domains alternate between old data and new data.

SH 16-Oct-23
"""

import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
#=======================================================================
def one_energy(arr,ix,iy,nmax):
    
    neighbours = np.array([
        ((ix + 1) % nmax, iy), # Right neighbour
        ((ix - 1) % nmax, iy), # Left
        (ix, (iy + 1) % nmax), # Upper neighbour
        (ix, (iy - 1) % nmax)  # Lower
    ])

    # All of the neighbouring locations are now stored in a single NumPy array.
    # There is no need to declare 'en' at the start of the function.

    theta_center = arr[ix, iy]

    # We have calculated the orientation angle once, rather than four times for each neighbour.

    angles = theta_center - arr[neighbours[:, 0], neighbours[:, 1]]
    en = 0.5 * np.sum(1.0 - 3.0 * np.cos(angles) ** 2)

    # Instead of performing four seperate calculations, or using a loop, we use NumPy array slicing.
    
    return en
#=======================================================================
def all_energy(arr,nmax):
    
    ix, iy = np.meshgrid(np.arange(nmax), np.arange(nmax), indexing='ij')

    # Instead of using two loops, to loop over the rows and columns, we can instantaneously create two matrices,
    # one for all the row indices and one for all the column indices.

    ix1D, iy1D = ix.ravel(), iy.ravel()

    # Convert our matrices into 1D-arrays so we can loop over them.

    enall = np.sum([one_energy(arr, i, j, nmax) for i, j in zip(ix1D, iy1D)])

    # We are looping over all the lattice sites, using 'zip()' to give us coordinate pairs.
    # We are calling one_energy to calculate the energy of each cell and then using np.sum() to add them all
    # together at the end.
    
    return enall
#=======================================================================
def get_order(arr,nmax):
    
    print("get_order function is being called")

    # Debugging statement added after completion. get_order did not appear in the profiling output, so this checks if
    # it is actually working/being called.

    startTime = time.perf_counter()

    # Explicitly timing get_order due to the above problem.
    
    delta = np.eye(3,3)

    # We don't need to manually initialise Qab, it will be handled later.
    
    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    
    Qab = 3 * np.tensordot(lab, lab, axes=([1, 2], [1, 2])) / (2 * nmax * nmax) - delta / 2
    # '3 * np.tensordot(lab, lab, axes=([1, 2], [1, 2]))' This part computes the summation over all the lattice
    # sites in one go.
    # '/ (2 * nmax * nmax)' This part normalises the Q tensor.
    # 'delta / 2' This part performs the identity matrix subtraction part of the formula.

    endTime = time.perf_counter()
    print (f"Execution time: {endTime - startTime:.6f} seconds")

    # Ending the timer and printing the time.
    
    return np.max(np.linalg.eigvalsh(Qab))

    # Here we are only calculating and returning the largest eigenvalue in one step since the eigenvectors are
    # not needed.

#=======================================================================
def MC_step(arr,Ts,nmax):
    
    scale=0.1+Ts # Distribution of angle changes, increases with temp.

    aran = np.random.normal(scale=scale, size=(nmax,nmax))

    en0 = np.array([[one_energy(arr, i, j, nmax) for j in range(nmax)] for i in range(nmax)])

    # Instead of us looping and calculating initial energies one at a time, we can compute all the initial
    # energies in one go.

    arr += aran

    # Instead of looping and changing the angle one at a time, we can change all of the angles in one go.

    en1 = np.array([[one_energy(arr, i, j, nmax) for j in range(nmax)] for i in range(nmax)])
    
    # Akin to the initial energy calculation, we can calculate all of the new energies in one go, without
    # looping.
    
    engDiff = en1 - en0

    # Calculate energy differences all in one go. These will be used later.

    acceptMask = (engDiff <= 0) | (np.exp(-engDiff / Ts) >= np.random.uniform(0.0, 1.0, size=(nmax, nmax)))

    # '(engDiff <= 0)' creates a Boolean array. If the new energy is lower than the initial energy (the
    # difference is lower than 0), the change is accepted and the Boolean value is True. Otherwise, it is false.
    # 'np.exp(-engDiff / Ts)' computes the Boltzmann probability for all the elements in one go.
    # '(np.exp(-engDiff / Ts) >= np.random.uniform(0.0, 1.0, size=(nmax, nmax)))' Performs the Metropolis test
    # for all the sites at once and gives us a True value if a site passes the test and is accepted.
    # OR (|) is used. If the first part is True, then the change is accepted, otherwise, if the second part
    # is True, then the change is accepted.
    # All True and False values (acceptance or rejection) are computed in one go, as opposed to doing so through
    # a list.

    arr[~acceptMask] -= aran[~acceptMask]
    
    # '[~accept]' gives us all of the false elements, where the site change is rejected.
    # This line reverts those changes where it has failed.

    acceptRatio = np.mean(acceptMask)

    # Calculate the acceptance ratio.

    # In this optimised version of the function, we don't randomly pick lattice sites and modify them one at
    # a time, we apply the Monte Carlo step sequentially across the entire lattice.
    
    return acceptRatio
#=======================================================================
def main(program, nsteps, nmax, temp, pflag):
    """
    Arguments:
	  program (string) = the name of the program;
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
	  pflag (int) = a flag to control plotting.
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """
    # Create and initialise lattice
    lattice = initdat(nmax)
    # Plot initial frame of lattice
    plotdat(lattice,pflag,nmax)
    # Create arrays to store energy, acceptance ratio and order parameter
    energy = np.zeros(nsteps+1,dtype=np.dtype)
    ratio = np.zeros(nsteps+1,dtype=np.dtype)
    order = np.zeros(nsteps+1,dtype=np.dtype)
    # Set initial values in arrays
    energy[0] = all_energy(lattice,nmax)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(lattice,nmax)

    # Begin doing and timing some MC steps.
    initial = time.time()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(lattice,temp,nmax)
        energy[it] = all_energy(lattice,nmax)
        order[it] = get_order(lattice,nmax)
    final = time.time()
    runtime = final-initial
    
    # Final outputs
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
    # Plot final frame of lattice and generate output file
    savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
    plotdat(lattice,pflag,nmax)
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
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
#=======================================================================