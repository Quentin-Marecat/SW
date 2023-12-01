import numpy as np
from sw.solver.sw import SchriefferWolff
from sw.tools.tools import fidelity
from pyhub.solver.fermi_hubbard import FermiHubbard

# Fermi-Hubbard model parameters
Lx = 6        # Horizontal dimension of lattice
nb_sites  = Lx
nup = ndown = nb_sites//2
hilbert = (nup, ndown)
t_matrix = np.diag(np.full(nb_sites-1,-1.),k=1) + np.diag(np.full(nb_sites-1,-1.),k=-1)
t_matrix[0,-1] = t_matrix[-1,0] = -1.
U     = 4. # Hubbard parameter
opt_method = ['L-BFGS-B',"SLSQP","COBYLA","SPSA"][0]
theta = [1.,'exact','opt'][1]
optimisation = {'method':opt_method,'bounds':True,'min_energy':True}


FH = FermiHubbard(nb_sites,*hilbert,t_matrix,U)
FH.kernel(compute_rq=False,verbose=False)

SW = SchriefferWolff(nb_sites,*hilbert,t_matrix,U,trial_state = None,verbose=True)
SW.kernel(theta=theta,**optimisation)

print("%"*50+'\n')
print(f'time : {np.around(SW.time,4)} s')
print("Energies:")
energy = np.copy(SW.energy)
print("  Calculated: {}".format(energy))
print("  Exact: {}".format(FH.e0))
print(f"  Error: {np.around(100-100*energy/FH.e0,4)} %")
print('Eigenvetors')
print(f"  Fidelity with exact fermi-hubbard gs : {fidelity(SW.fermi_hubbard_gs,FH.psi0)}")
print(f"theta value: {SW.theta}")
print('\n',"%"*50)
