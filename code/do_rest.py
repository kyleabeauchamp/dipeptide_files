#!/usr/local/bin/env python -d
import os
import os.path
import sys
import numpy as np

import simtk.openmm as openmm
import simtk.unit as units
from simtk.openmm import app
import rest_generator

   
pdb = app.PDBFile("./native.pdb")

residue_names = np.array([a.residue.name for a in pdb.topology.atoms()])
water_indices = np.where(residue_names == "HOH")[0]

forcefield = app.ForceField('amber99sbildn-nmr.xml',"tip3p.xml")
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME,nonbondedCutoff=1.15*units.nanometer, constraints=app.HAngles)

store_filename = 'repex.nc'

Tmin = 450.0 * units.kelvin # minimum temperature
Tmax = 750.0 * units.kelvin # maximum temperature
ntemps = 5 # number of replicas
ngpus = 1 # number of GPUs per system

timestep = 2.0 * units.femtoseconds # timestep for simulation
nsteps = 2000 # number of timesteps per iteration (exchange attempt)
niterations = 15000 # number of iterations to complete
nequiliterations = 1 # number of equilibration iterations at Tmin with timestep/2 timestep, for nsteps*2
verbose = True # verbose output (set to False for less output)
minimize = False # minimize

platform = openmm.Platform.getPlatformByName("OpenCL") # OpenCL is pretty good on Mac OS X

if verbose: print "System has %d atoms." % system.getNumParticles()
try:
    print "Selecting MPI communicator and selecting a GPU device..."
    from mpi4py import MPI # MPI wrapper
    hostname = os.uname()[1]
    comm = MPI.COMM_WORLD # MPI communicator
    deviceid = 1+comm.rank % ngpus # select a unique GPU for this node assuming block allocation (not round-robin)
    
    platform.setPropertyDefaultValue('CudaDevice', '%d' % deviceid) # select Cuda device index
    platform.setPropertyDefaultValue('OpenCLDeviceIndex', '%d' % deviceid) # select OpenCL device index
    print "node '%s' deviceid %d / %d, MPI rank %d / %d" % (hostname, deviceid, ngpus, comm.rank, comm.size)
    # Make sure random number generators have unique seeds.
    seed = np.random.randint(sys.maxint - comm.size) + comm.rank
    np.random.seed(seed)
except:
    comm = None
    print "WARNING: Could not initialize MPI.  Running serially..."            


if verbose: print "Initializing parallel tempering simulation..."
#simulation = repex.REST2(system, water_indices, pdb.positions, store_filename, Tmin=Tmin, Tmax=Tmax, ntemps=ntemps, mpicomm=comm)
simulation = rest_generator.generate_rest2(system, Tmin,Tmax,ntemps,pdb.positions,water_indices, store_filename, mpicomm=comm)
simulation.verbose = True # write debug output
simulation.platform = platform # specify platform
simulation.number_of_equilibration_iterations = nequiliterations
simulation.number_of_iterations = niterations # number of iterations (exchange attempts)
simulation.timestep = timestep # timestep
simulation.nsteps_per_iteration = nsteps # number of timesteps per iteration
simulation.replica_mixing_scheme = 'swap-all' # better mixing scheme for exchange step
simulation.minimize = minimize

if verbose: print "Running..."
simulation.run()

