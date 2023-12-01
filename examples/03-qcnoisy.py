import numpy as np
from sw.solver.swqc import SchriefferWolffQC
from sw.tools.tools import fidelity
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from scipy.stats import norm

# Fermi-Hubbard model parameters
Lx = 4        # Horizontal dimension of lattice
Ly = 1        # Vertical dimension of lattice (Ly = 1 for chains, Ly = 2 for ladders)
nb_sites  = Lx * Ly  # Total number of lattice sites
t_matrix = np.diag(np.full(nb_sites-1,-1.),k=1) + np.diag(np.full(nb_sites-1,-1.),k=-1)
if nb_sites > 2:
    t_matrix[0,-1] = t_matrix[-1,0] = 0.
BC = 'OBC'
U     = 10. # Hubbard parameter
opt_method = ['L-BFGS-B',"SLSQP","COBYLA","SPSA"][-1]
theta='opt'
solve_fermi_hubbard = True
S2_subspace = False
mode = '1' 
## noisy
noisy={'nshots':2**10,'prob_1':1.e-4,'prob_2':1.e-3,'n_eval':50}


print(f'Noisy simulation : ')
SWQC = SchriefferWolffQC(nb_sites,nb_sites,t_matrix,U,trial_state = None,verbose=False,Ly=Ly,noisy=noisy,BC=BC,mode=mode)
SWQC.kernel(solve_fermi_hubbard=solve_fermi_hubbard,theta=theta,S2_subspace=S2_subspace)
print("%"*50+'\n')
print(f'time : {np.around(SWQC.time,4)} s')
print("Energies:")
print("  Trotterized: {}".format(SWQC.energy_avg))
if solve_fermi_hubbard:
    print("  Exact: {}".format(SWQC.eigval[0]))
    print(f"  Error: {np.around(100-100*SWQC.energy_avg/SWQC.eigval[0],4)} %")
    if len(SWQC.fermi_hubbard_gs) == len(SWQC.eigvec[0]):
        print('Eigenvetors')
        print(f"  Fidelity with exact fermi-hubbard gs : {fidelity(SWQC.fermi_hubbard_gs,SWQC.eigvec[0])}")
print("theta value : {}".format(SWQC.theta))
print(f'number of evaluations : {SWQC.n_eval}')
print('\n',"%"*50)

def normal(x,smu):
    return np.exp(-((x-smu[1])/smu[0])**2/2)/(smu[0]*np.sqrt(2*np.pi))

ICGMmarine=(0.168,0.168,0.525)
ICGMblue=(0,0.549,0.714)
ICGMorange=(0.968,0.647,0)
ICGMyellow=(1,0.804,0)
gray=(0.985,0.985,0.985)
clr_map=[gray,ICGMyellow,ICGMorange,ICGMblue,ICGMmarine]
cmap = colors.LinearSegmentedColormap.from_list('my_list', clr_map, N=100)

fig, axs = plt.subplots(1, 2, tight_layout=True)

axs[0].hist(SWQC.energy, bins=20)
axs[1].hist(SWQC.energy, bins=20, density=True,color=ICGMmarine)
axs[1].yaxis.set_major_formatter(PercentFormatter())
mu, std = norm.fit(SWQC.energy)
half_height = 2 * np.sqrt(2 * np.log(2) ) * std
print(f'normal law : \n  Variance {std**2} \n  Esperance / Avg {mu/SWQC.energy_avg}\n  Half-height / Avg {half_height/np.abs(SWQC.energy_avg)}')
x = np.linspace(np.min(SWQC.energy),np.max(SWQC.energy),1000)
axs[1].plot(x,norm.pdf(x,mu,std),color=ICGMorange,linewidth=2,label="Esp. = %.2f,  Var. = %.2f" % (mu, std**2))

N, bins, patches = axs[0].hist(SWQC.energy, bins=20)
# We'll color code by height, but you could use any scalar
fracs = N / N.max()
# we need to normalize the data to 0..1 for the full range of the colormap
norm = colors.Normalize(fracs.min(), fracs.max())
# Now, we'll loop through our objects and set the color of each accordingly
for thisfrac, thispatch in zip(fracs, patches):
    color = cmap(norm(thisfrac))
    thispatch.set_facecolor(color)

ticks=np.sort([SWQC.energy_avg,SWQC.eigval[0]])
for ax in axs:
    ax.axvline(SWQC.energy_avg,linestyle='--',color=ICGMorange)
    ax.axvline(SWQC.eigval[0],linestyle='--',color=ICGMmarine)
    ax.set_xticks(ticks,color='k')
    ax.get_xticklabels()[np.where(ticks==SWQC.energy_avg)[0][0]].set_color(ICGMorange)
    ax.get_xticklabels()[np.where(ticks==SWQC.eigval[0])[0][0]].set_color(ICGMmarine)
axs[1].legend(loc='upper right')
plt.savefig('histo')



