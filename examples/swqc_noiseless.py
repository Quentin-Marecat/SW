import numpy as np
from sw.solver.swqc import SchriefferWolffQC
from sw.tools.tools import fidelity

# Fermi-Hubbard model parameters
Lx = 6        # Horizontal dimension of lattice
Ly = 1        # Vertical dimension of lattice (Ly = 1 for chains, Ly = 2 for ladders)
nb_sites  = Lx * Ly  # Total number of lattice sites
t_matrix = np.diag(np.full(nb_sites-1,-1.),k=1) + np.diag(np.full(nb_sites-1,-1.),k=-1)
if nb_sites > 2:
    t_matrix[0,-1] = t_matrix[-1,0] = 0.
BC = 'OBC'
U     = 10. # Hubbard parameter
opt_method = ['L-BFGS-B',"SLSQP","COBYLA","SPSA"][0]
theta='exact'
solve_fermi_hubbard = True
S2_subspace = False
mode = '1' 



print(f'Noiseless simulation : ')
SWQC = SchriefferWolffQC(nb_sites,nb_sites,t_matrix,U,trial_state = None,verbose=True,Ly=Ly,noisy=None,BC=BC,mode=mode)
SWQC.kernel(solve_fermi_hubbard=solve_fermi_hubbard,theta=theta,S2_subspace=S2_subspace)
print("%"*50+'\n')
print(f'time : {np.around(SWQC.time,4)} s')
print("Energies:")
print("  Trotterized: {}".format(SWQC.energy))
if solve_fermi_hubbard:
    print("  Exact: {}".format(SWQC.eigval[0]))
    print(f"  Error: {np.around(100-100*SWQC.energy/SWQC.eigval[0],4)} %")
    if len(SWQC.fermi_hubbard_gs) == len(SWQC.eigvec[0]):
        print('Eigenvetors')
        print(f"  Fidelity with exact fermi-hubbard gs : {fidelity(SWQC.fermi_hubbard_gs,SWQC.eigvec[0])}")
print("theta value:")
print("  Trotterized: {}".format(SWQC.theta))
print('\n',"%"*50)