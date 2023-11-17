import numpy as np
from sw.solver.heisenbergqc import HeisenbergQC
from sw.tools.tools import fidelity

# Heisenberg model parameters
Lx = 8        # Horizontal dimension of lattice
Ly = 1        # Vertical dimension of lattice (Ly = 1 for chains, Ly = 2 for ladders)
nb_sites  = Lx * Ly  # Total number of lattice sites
J_matrix = np.diag(np.full(nb_sites-1,1.),k=1) + np.diag(np.full(nb_sites-1,1.),k=-1)
if nb_sites > 2:
    J_matrix[0,-1] = J_matrix[-1,0] = 0.
BC = 'OBC'
opt_method = ['L-BFGS-B',"SLSQP","COBYLA","SPSA"][0]
solve_model = True
S2_subspace = False
mode = '1' 



print(f'Noiseless simulation : ')
SWQC = HeisenbergQC(nb_sites,nb_sites,J_matrix,trial_state = None,verbose=True,Ly=Ly,noisy=None,BC=BC,mode=mode)
SWQC.kernel(solve_heisenberg=solve_model,S2_subspace=S2_subspace,opt_method=opt_method)
print("%"*50+'\n')
print(f'time : {np.around(SWQC.time,4)} s')
print("Energies:")
print("  Trotterized: {}".format(SWQC.energy))
if solve_model:
    print("  Exact: {}".format(SWQC.eigval[0]))
    print(f"  Error: {np.around(100-100*SWQC.energy/SWQC.eigval[0],4)} %")
    if len(SWQC.heisenberg_gs) == len(SWQC.eigvec[0]):
        print('Eigenvetors')
        print(f"  Fidelity with exact fermi-hubbard gs : {fidelity(SWQC.heisenberg_gs,SWQC.eigvec[0])}")
print('\n',"%"*50)