import numpy as np
from sw.solver.swqc import SchriefferWolffQC
from sw.tools.tools import fidelity
from pyhub.solver.fermi_hubbard import FermiHubbard
from pyhub.core.basis import Basis

# Fermi-Hubbard model parameters
Lx = 2        # Horizontal dimension of lattice
Ly = 1        # Vertical dimension of lattice (Ly = 1 for chains, Ly = 2 for ladders)
nb_sites  = Lx * Ly  # Total number of lattice sites
nup = ndown = nb_sites//2
hilbert = (nup, ndown)
t_matrix = np.diag(np.full(nb_sites-1,-1.),k=1) + np.diag(np.full(nb_sites-1,-1.),k=-1)
if nb_sites > 2:
    t_matrix[0,-1] = t_matrix[-1,0] = 0.
BC = 'OBC'
U     = 4. # Hubbard parameter
opt_method = ['L-BFGS-B',"SLSQP","COBYLA","SPSA"][0]
theta='exact'
mode = '1' 
optimisation = {'method':opt_method,'bounds':True,'min_energy':True}


FH = FermiHubbard(nb_sites,*(nup,ndown),t_matrix,U,order='site')
FH.kernel(compute_rq=False,verbose=False)

fockbasis = Basis(nb_sites,(list(range(nb_sites+1)),list(range(nb_sites+1))))
psi0 = np.zeros(fockbasis.nstates)
index = []
for elem in FH.basis:
    index.append(np.where(fockbasis.basis==elem)[0][0]) 
psi0 = np.zeros(fockbasis.nstates)
psi0=FH.psi0

print(f'Noiseless simulation : ')
SWQC = SchriefferWolffQC(nb_sites,*hilbert,t_matrix,U,trial_state = None,verbose=True,Ly=Ly,noisy=None,BC=BC,mode=mode)
SWQC.kernel(theta=theta,**optimisation) 

print("%"*50+'\n')
print(f'time : {np.around(SWQC.time,4)} s')
print("Energies:")
print("  Trotterized: {}".format(SWQC.energy))
print("  Exact: {}".format(FH.e0))
print(f"  Error: {np.around(100-100*SWQC.energy/FH.e0,4)} %")
print('Eigenvetors')
print(SWQC.fermi_hubbard_gs)
print("different endian ordering which explains the change of sign:",psi0)
raise ValueError('Must convert spin basis to site basis : anti-commuation convention Error')
print(f"  Fidelity with exact fermi-hubbard gs : {fidelity(SWQC.fermi_hubbard_gs,psi0)}")
print("theta value:")
print("  Trotterized: {}".format(SWQC.theta))
print('\n',"%"*50)
