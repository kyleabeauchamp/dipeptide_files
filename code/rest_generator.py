"""This code is an implementation of the algorithm in the following paper:
Replica Exchange with Solute Scaling: A More Efficient Version of Replica 
Exchange with Solute Tempering (REST2)
"""

import math
import copy

import simtk.openmm 

from thermodynamics import ThermodynamicState
from simtk.pyopenmm.extras import repex

def generate_rest2(system,Tmin,Tmax,ntemps,coordinates,water_indices,store_filename,mpicomm=None):
    """Generate a REST2 repex system.
    
    Parameters
    ----------
    system : an OpenMM System object
    Tmin : the minimum temperature
    Tmax : the maximum temperature
    ntemps : the number of temperature states to generate
    coordinates : XYZ coordinates to initialize states
    water_indices : list of which atom indices are solvent atoms
    store_filename : filename used to output the resulting repex netcdf file
    mpicomm : an mpicomm object if you want to use MPI parallelization
        
    Returns
    -------
    A HamiltonianExchange object with desired REST2 hamiltonians 
    """
    temperatures = [ Tmin + (Tmax - Tmin) * (math.exp(float(i) / float(ntemps-1)) - 1.0) / (math.e - 1.0) for i in range(ntemps) ]        
    print(temperatures)
    
    reference_state = ThermodynamicState(temperature=temperatures[0])
           
    systems = [system]
    
    num_temp = len(temperatures)        
    
    for i in range(1,num_temp):
        
        rho = (temperatures[i]/temperatures[0])**0.5
        
        system = copy.deepcopy(systems[0])
        adjust_REST_system(system,rho,water_indices)
        systems.append(system)
        
    return repex.HamiltonianExchange(reference_state,systems,coordinates, store_filename, mpicomm=mpicomm)


def adjust_REST_system(system,rho,water_indices):
    """Adjust hamiltonian parameters to create REST2 hamiltonian.
    
    Parameters
    ----------
    system : an OpenMM System object
    rho : factor to adjust energies
    water_indices : list of atoms corresponding to solvent
        
    Notes
    -------
    If your force field uses terms different from AMBER, then you may need 
    to modify the logic to correctly generate the REST2 hamiltonian states.
    """
    num_part = system.getNumParticles()
    for k in range(system.getNumForces()):
        f = system.getForce(k)
        if type(f) == simtk.openmm.openmm.NonbondedForce:
            for j in xrange(num_part):
                if j not in water_indices:
                    q,s,e = f.getParticleParameters(j)
                    q /= rho
                    e /= rho**2.
                    f.setParticleParameters(j,q,s,e)
        elif type(f) == simtk.openmm.openmm.PeriodicTorsionForce:
            for i in xrange(f.getNumTorsions()):
                a,b,c,d,per,phase,k0 = f.getTorsionParameters(i)
                k0 /= (rho**2.)
                f.setTorsionParameters(i,a,b,c,d,per,phase,k0)

