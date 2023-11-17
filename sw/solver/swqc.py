from warnings import WarningMessage
import numpy as np
import scipy
from time import perf_counter as pc
from sw.tools.measurements import count_ones_bitstring, evaluate_statevector, sampled_expectation_value, evaluate
from sw.tools.tools import optimized_features_RVB_inspired_ansatz_Heisenberg_model, compute_statevector
from sw.tools.circuits import RVB_inspired_ansatz, fermionic_version_of_spin_wave_function_site_ordered, CU_trotterized
from sw.tools.qc_operators import Hubbard_1D_operator, SW_operator, s2_operator
from scipy.stats import norm

# importing Qiskit
from qiskit_algorithms.optimizers import SPSA
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_aer import QasmSimulator
from qiskit.providers.fake_provider import FakeManila
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise

class SchriefferWolffQC():

    def __init__(self,nb_sites,nb_elec:int,t_matrix:np.ndarray,U:float,trial_state = None,verbose=True,Ly=None,noisy=None,BC='OBC',mode=1):
        self.nb_sites=nb_sites
        if nb_sites > 10:
            raise ValueError(f'number of sites must be lower or equal as 10, not {self.nb_sites}')
        self.nb_elec=nb_elec
        self.t_matrix=t_matrix 
        self.ek,Vk = np.linalg.eigh(self.t_matrix)
        self.U=U
        self.U_tensor = np.zeros([nb_sites]*4)
        self.verbose=verbose
        for i in range(self.nb_sites):
            self.U_tensor[i,i,i,i] = U
        self.verbose = True
        if not Ly==None:
            self.Ly=Ly 
            self.Lx = self.nb_sites//self.Ly
        else:
            self.Ly=1
            self.Lx=self.nb_sites

        if isinstance(noisy,dict) or noisy:
            self.printv('Noisy simulation')
            self.noisy = True
            self.nshots = 1024 if 'nshots' not in noisy.keys() else noisy['nshots']
            self.n_eval = 1 if 'n_eval' not in noisy.keys() else noisy['n_eval']
            self.var = None #if 'var' not in noisy.keys() else noisy['var']
            self.printv(f'     nshots : {self.nshots}')
            self.printv(f'     n_eval : {self.n_eval}')
#            self.printv(f'     Variance : {self.var}')
            if self.var is not None and self.n_eval < 100:
                print(f'     Please increase the number of max evaluation if you want correct variance')

            if  "FakeManila" in noisy.keys():
                device_backend   = FakeManila()
                device           = QasmSimulator.from_backend(device_backend)
                self.noise_model      = NoiseModel.from_backend(device)

            else:
                # Error probabilities
                prob_1 = 0.0001 if 'prob_1' not in noisy.keys() else noisy['prob_1'] # 1-qubit gate
                prob_2 = 0.001  if 'prob_2' not in noisy.keys() else noisy['prob_2'] # 2-qubit gate
                # Depolarizing quantum errors
                error_1 = noise.depolarizing_error(prob_1, 1)
                error_2 = noise.depolarizing_error(prob_2, 2)
                # Add errors to noise model
                self.noise_model = noise.NoiseModel()
                self.noise_model.add_all_qubit_quantum_error(error_1, ['u1','u2','u3','ry','rz','sx'])
                self.noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
                self.printv(f'     prob_1 : {prob_1}')
                self.printv(f'     prob_2 : {prob_2}')

        else:
            self.noisy = False
            self.noise_model = None


        self.printv(f'Set circuit')
        self.backend = QasmSimulator(method='statevector', noise_model=self.noise_model)
        self.n_qubits = 2*self.nb_sites

        self.trial_state = trial_state
        self.printv(f'Preparation of the Heisenberg state')
        if self.trial_state == None:
            self.set_spin_state(BC=BC,mode=mode)
            #self.initial_circuit = initial_circ_dimer(L,state='Neel',phase=True,varphi=np.pi/2)
        else: 
            self.initial_circuit = trial_state

#        simulator = BasicAer.get_backend('statevector_simulator')
#        self.result = execute(self.initial_circuit, simulator).result()

    def kernel(self,solve_fermi_hubbard=True,order=1,theta=1.,S2_subspace=False,opt_method="SPSA"):
        t0 = pc()
        self.theta=theta
        self.order=order
        if order>1:
            NotImplementedError(f'SW relations beyond 2nd order is not implemented')
        self.printv(f'Set Fermi-Hubbard Hamiltonian')
        hspace = []
        for i in range(2**self.n_qubits):
            if count_ones_bitstring(i) == self.L:
                hspace += [i]
    #        hspace += [i]
        self.printv(f'     Define Fermi-Hubbard operator')
        Hubbard_operator = Hubbard_1D_operator(self.L,self.U,self.t_matrix)
        self.printv(f'     JW Mapping')
        jw_mapper = JordanWignerMapper()
        jw_mapper = QubitConverter(jw_mapper)
        self.Hubbard_PauliSum = jw_mapper.convert(Hubbard_operator)
        self.printv(f'     Set matrix Hamiltonian')
        self.Hubbard_matrix = Hubbard_operator.to_matrix() # <class 'scipy.sparse._csc.csc_matrix'>
        self.printv(f'Set SW operator')
        self.lbd_sw = np.einsum('ijk->kij',np.array([[\
            [self.t_matrix[p,q]/(self.t_matrix[q,q]-self.t_matrix[p,p]) if np.abs(self.t_matrix[p,p]-self.t_matrix[q,q]) else 0., \
            self.t_matrix[p,q]/self.U_tensor[p,p,p,p], -self.t_matrix[p,q]/self.U_tensor[q,q,q,q], \
                self.t_matrix[p,q]/((self.t_matrix[q,q]-self.t_matrix[p,p]) + (self.U_tensor[p,p,p,p]-self.U_tensor[q,q,q,q])) if np.abs(self.U_tensor[p,p,p,p]-self.U_tensor[q,q,q,q]) else 0.]\
                for p in range(self.n_sites)] for q in range(self.n_sites)])\
        )
        SW_op = SW_operator(self.n_sites,self.lbd_sw)
        self.SW_PauliSum = jw_mapper.convert(SW_op)
        if theta=='exact':
            self.theta = (self.U/4)*np.arctan(4/self.U)
        elif isinstance(theta,float):
            self.theta = theta
        elif theta=='variationnal' or theta=='opt':
            self.printv(f'Classical optimization')
            self.variationnal(opt_method=opt_method,bounds=True)
        else:
            raise ValueError(f'Set correct value of theta, not {theta}')

        self.printv(f'Calculate Fermi-Hubbard ground-state from prepared intial circuit')
        self.fermi_hubbard_gs = compute_statevector([self.theta],self.initial_circuit, self.SW_PauliSum,self.backend)[hspace]
        self.printv(f'Calculate ground-state energy')
        if self.noisy:
            self.SW_PauliSum_theta = self.SW_PauliSum.mul(self.theta)
            self.total_circuit = self.initial_circuit.compose(CU_trotterized(self.SW_PauliSum_theta))
            self.energy = []
            for i in range(self.n_eval):
                self.energy.append(sampled_expectation_value(self.total_circuit,self.Hubbard_PauliSum,self.backend,nshots=self.nshots))
                if self.var is not None and i>20 and i%10==0:
                    mu, std = norm.fit(self.energy)
                    half_height = 2 * np.sqrt(2 * np.log(2) ) * std
                    if half_height < self.var*abs(np.average(self.energy)):
                        self.n_eval = i
                        break
            self.energy = np.array(self.energy)
            self.energy_avg = np.average(self.energy)
        else:
            self.energy = evaluate_statevector([self.theta],self.initial_circuit, self.SW_PauliSum,self.Hubbard_matrix,self.backend)
        #s2_matrix = Operator(s2_PauliSum).data
        if solve_fermi_hubbard:
            self.printv(f'Solve Fermi-Hubbard Hamiltonian')
            self.printv(f'     Set matrix Hamiltonian in Hilbert space')
            Hubbard_matrix_hspace = self.Hubbard_matrix[np.ix_(hspace, hspace)]
            self.printv(f'     Diagonalize the matrix')
            eigval_hspace,eigvec_hspace = self._diag(Hubbard_matrix_hspace)
            if S2_subspace : 
                self.printv(f'     Set matrix S2 in Hilbert space')
                s2_op = s2_operator(self.n_qubits)
                s2_matrix = s2_op.to_matrix()
                s2_matrix_hspace=s2_matrix[np.ix_(hspace, hspace)]
                self.eigval,self.eigvec = [],[]
                self.printv(f'     Find eigenvectors in S2 subspace')
                for i in range(len(eigval_hspace)):
                    s2 = int(np.around(eigvec_hspace[:,i].conj() @ s2_matrix_hspace @ eigvec_hspace[:,i]).real)  
                    if s2 == 0 :
                        self.eigval.append(eigval_hspace[i].real)
                        self.eigvec.append(eigvec_hspace[:,i])
                if len(self.eigval)==0:
                    raise ValueError(f'No solution founds in the S2 subspace')
            else:
                self.eigval,self.eigvec = eigval_hspace,eigvec_hspace.T
        self.time = pc() - t0

            
    def set_spin_state(self,BC='OBC',mode=1):
        # Ansatz parameters
        # Num_layers: Number of layers of ansatz; each layer has depth 6 CNOTs (3 CNOTs per eSWAP)
        # optimized_params: values of parameters of eSWAP gates previously found variationally and stored in memory
        # BC: boundary conditions of ansatz. If BC = 'PBC', there is an eSWAP between qubits 0 and L-1, otherwise there is not.
        #     BC = 'OBC' was adopted in all simulations.
        # mode: implementation of ansatz. mode = '1' is RVB-inspired version where every eSWAP has its own free parameter. 
        #       mode = '2' is adiabatic version where there are two free parameters per layer only. mode = '1' is assumed.
        self.num_layers, self.optimized_params = optimized_features_RVB_inspired_ansatz_Heisenberg_model(self.Lx, self.Ly)
        self.BC = BC
        mode = str(mode)
        self.Heisenberg_GS_RVB_ansatz_qc = RVB_inspired_ansatz(self.optimized_params, self.L, self.BC, self.num_layers, mode)
        self.printv(f'     Number of layers : {self.num_layers}')
        self.printv(f'Conversion into a fermionic state')
        fermionic_Heisenberg_GS_RVB_ansatz_site_ordered_qc = fermionic_version_of_spin_wave_function_site_ordered(self.Heisenberg_GS_RVB_ansatz_qc, self.L)
#        self.printv(fermionic_Heisenberg_GS_RVB_ansatz_site_ordered_qc.decompose())
        self.initial_circuit = fermionic_Heisenberg_GS_RVB_ansatz_site_ordered_qc

    def variationnal(self,opt_method='SPSA',bounds=True):
        if opt_method != "SPSA": 
            self.result = scipy.optimize.minimize(evaluate_statevector if not self.noisy else evaluate ,
                                                        x0=[(self.U/4)*np.arctan(4/self.U)],
                                                        args=(self.initial_circuit,
                                                            self.SW_PauliSum,
                                                            self.Hubbard_matrix if not self.noisy else self.Hubbard_PauliSum,
                                                            self.backend,
                                                            None if not self.noisy else self.nshots),
                                                        method=opt_method,
                                                        bounds=scipy.optimize.Bounds(np.full(self.order,0),np.full(self.order,1.)) if bounds else None ,
                                                        options={'ftol':1e-14,'gtol': 1e-09})
            self.theta = self.result.x[0]
            self.energy = self.result.fun
        else:
            #spsa = SPSA(maxiter=500,blocking=True,allowed_increase=0.1,last_avg=25,resamplings=2)
            spsa = SPSA(maxiter=1000,last_avg=25)
            cost_function = SPSA.wrap_function(evaluate_statevector if not self.noisy else evaluate,
                                                (self.initial_circuit,
                                                self.SW_PauliSum,
                                                self.Hubbard_matrix if not self.noisy else self.Hubbard_PauliSum,
                                                self.backend,
                                                None if not self.noisy else self.nshots))
            self.result = spsa.minimize(cost_function, x0=[(self.U/4)*np.arctan(4/self.U)])
            self.theta = self.result.x[0]
            self.energy = self.result.fun # if last_avg is not 1, it returns the callable function with the last_avg param_values as input ! This seems generally good and better than taking the mean of the last_avg function calls.
            self.stddev = spsa.estimate_stddev(cost_function, initial_point=[self.theta])


    def printv(self,string):
        if self.verbose:
            print(string)

    @property
    def L(self):
        return self.Lx*self.Ly
    @property
    def n_sites(self):
        return self.nb_sites

    def _diag(self,matrix):
        if self.nb_sites<=6:
            return np.linalg.eigh(matrix.A)
        else:
            return scipy.sparse.linalg.eigsh(matrix,wich='SA')

    @property
    def spin_bar(self):
        return 1 if self.spin==0 else 0
