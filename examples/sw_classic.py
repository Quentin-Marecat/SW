import numpy as np
from sw.solver.sw import SchriefferWolff
from sw.tools.tools import fidelity

# Fermi-Hubbard model parameters
Lx = 6        # Horizontal dimension of lattice
nb_sites  = Lx
t_matrix = np.diag(np.full(nb_sites-1,-1.),k=1) + np.diag(np.full(nb_sites-1,-1.),k=-1)
t_matrix[0,-1] = t_matrix[-1,0] = -1.
U     = 10. # Hubbard parameter
opt_method = ['L-BFGS-B',"SLSQP","COBYLA","SPSA"][0]
theta='exact'
solve_fermi_hubbard = True



SW = SchriefferWolff(nb_sites,nb_sites,t_matrix,U,trial_state = None,verbose=True)
SW.kernel(solve_fermi_hubbard=solve_fermi_hubbard,theta=theta,opt_method=opt_method)
print("%"*50+'\n')
print(f'time : {np.around(SW.time,4)} s')
print("Energies:")
energy = np.copy(SW.energy)
print("  Calculated: {}".format(energy))
if solve_fermi_hubbard:
    print("  Exact: {}".format(SW.eigval[0]))
    print(f"  Error: {np.around(100-100*energy/SW.eigval[0],4)} %")
    print('Eigenvetors')
    print(f"  Fidelity with exact fermi-hubbard gs : {fidelity(SW.fermi_hubbard_gs,SW.eigvec[0])}")
print("theta value:")
print("  Trotterized: {}".format(SW.theta))
print('\n',"%"*50)
