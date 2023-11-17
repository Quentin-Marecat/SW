import numpy as np 
import quantnbody as qnb
import scipy
import warnings
from time import perf_counter as pc
warnings.filterwarnings("ignore", category=scipy.sparse.SparseEfficiencyWarning)

class SchriefferWolff():

    def __init__(self,nb_sites,nb_elec:int,t_matrix:np.ndarray,U:float,trial_state = None,verbose=True):
        self.nb_sites=nb_sites
        if nb_sites > 10:
            raise ValueError(f'number of sites must be lower or equal as 10, not {self.nb_sites}')
        self.nb_elec=nb_elec
        self.t_matrix=t_matrix 
        self.ek,Vk = np.linalg.eigh(self.t_matrix)
        self.U=U
        self.U_tensor = np.zeros([nb_sites]*4)
        for i in range(self.nb_sites):
            self.U_tensor[i,i,i,i] = U
        self.verbose = True
        self.printv('Define n-body basis')
        self.nbody_basis = qnb.fermionic.tools.build_nbody_basis( nb_sites, nb_elec )
        self.printv('Create ob operator')
        self.a_dagger_a = qnb.fermionic.tools.build_operator_a_dagger_a( self.nbody_basis )

        if trial_state == None:
            self.printv('Set trial state as AF spin state')
            self.trial_state = self.set_spin_state()
        else:
            self.trial_state = trial_state


    def kernel(self,theta=1.,order=1,solve_fermi_hubbard=True,opt_method='L-BFGS-B'):
        t0 = pc()
        self.H0 = scipy.sparse.csr_matrix((self.nb_states, self.nb_states))
        self.VX = scipy.sparse.csr_matrix((self.nb_states,self.nb_states))
        self.VD = scipy.sparse.csr_matrix((self.nb_states,self.nb_states))
        self.printv('Set Fermi-Hubbard Hamiltonian')
        for p in range(self.nb_sites):
            for q in range(self.nb_sites):
                if self.t_matrix[p, q] != 0 and p != q:
                    self.VX += (self.a_dagger_a[2 * p, 2 * q] + self.a_dagger_a[2 * p + 1, 2 * q + 1]) * self.t_matrix[p, q]
                elif self.t_matrix[p, q] != 0 and p == q:
                    self.VD += (self.a_dagger_a[2 * p, 2 * q] + self.a_dagger_a[2 * p + 1, 2 * q + 1]) * self.t_matrix[p, q]
                for r in range(self.nb_sites):
                    for s in range(self.nb_sites):
                        if self.U_tensor[p, q, r, s] != 0:  # if U is 0, it doesn't make sense to multiply matrices
                            self.H0 += self.a_dagger_a[2 * p, 2 * q] @ self.a_dagger_a[2 * r + 1, 2 * s + 1] * self.U_tensor[p, q, r, s]
        self.homogeneous = False if scipy.sparse.linalg.norm(self.VD)>1.e-10 else True
        if solve_fermi_hubbard:
            self.printv('Solve Fermi-Hubbard Hamiltonian')
            self.eigval,Vk = self._diag(self.H_fermi_hubbard)
            self.eigvec = Vk[:,:1].T
            del(Vk)
        self.printv('Build S operator')
        self.order=order
        self.build_S(order=order)
        if theta=='exact':
            self.theta = np.full(self.order,(self.U/4)*np.arctan(4/self.U))
        elif isinstance(theta,(np.ndarray,list)):
            self.theta = theta
        elif isinstance(theta,float):
            self.theta = np.full(self.order,theta)
        elif theta=='variationnal' or theta=='opt':
            self.printv(f'Classical optimization')
            self.variationnal(min_energy=True,bounds=True,opt_method=opt_method)
        else:
            raise ValueError(f'Set correct value of theta, not {theta}')
        self.time = pc() - t0
            

    def variationnal(self,min_energy=True,bounds=True,opt_method='L-BFGS-B'):
        if not min_energy:
            V = scipy.sparse.csr_matrix((self.nb_states,self.nb_states))
            for p in range(self.nb_sites):
                for q in range(self.nb_sites):
                    if p != q:
                        V += np.abs((self.a_dagger_a[2 * p, 2 * q] + self.a_dagger_a[2 * p + 1, 2 * q + 1]) * 1.)
        def fun(theta):
            self.theta = theta
            if min_energy:
                return self.energy
            else:
                return np.einsum('ij,ij,ij->',self.Hbar,self.Hbar , V.A)
        min = scipy.optimize.minimize(fun,x0=self.theta,opt_method='L-BFGS-B',bounds=scipy.optimize.Bounds(np.full(self.order,0),np.full(self.order,1.)) if bounds else None )
        self.theta=min.x


    def build_S(self,order=1):
        dim_S = len(self.nbody_basis)
        self.S_list = [scipy.sparse.csr_matrix((dim_S, dim_S)) for o in range(order)]
        id = scipy.sparse.identity(dim_S)
        if order >= 1 : 
            for p in range(self.nb_sites):
                for q in range(p):
                    lbd_sw = np.array([self.t_matrix[p,q]/(self.t_matrix[q,q]-self.t_matrix[p,p]) if np.abs(self.t_matrix[p,p]-self.t_matrix[q,q]) else 0., \
                        self.t_matrix[p,q]/self.U_tensor[p,p,p,p], -self.t_matrix[p,q]/self.U_tensor[q,q,q,q], \
                            self.t_matrix[p,q]/((self.t_matrix[q,q]-self.t_matrix[p,p]) + (self.U_tensor[p,p,p,p]-self.U_tensor[q,q,q,q])) if np.abs(self.U_tensor[p,p,p,p]-self.U_tensor[q,q,q,q]) else 0.])
                    for self.spin in [0,1]:
                        self.S_list[0] += ( lbd_sw[1] * self.a_dagger_a[2 * p+self.spin_bar, 2 * p+self.spin_bar] * (id - self.a_dagger_a[2 * q+self.spin_bar, 2 * q+self.spin_bar]) + \
                              lbd_sw[2] * self.a_dagger_a[2 * q+self.spin_bar, 2 * q+self.spin_bar] * (id - self.a_dagger_a[2 * p+self.spin_bar, 2 * p+self.spin_bar]) \
                                )\
                            * (self.a_dagger_a[2 * p+self.spin, 2 * q+self.spin] - self.a_dagger_a[2 * q+self.spin, 2 * p+self.spin])
                        if np.abs(lbd_sw[0]) > 1.e-10:
                            self.S_list[0] += ( lbd_sw[0] * (id - self.a_dagger_a[2 * p+self.spin_bar, 2 * p+self.spin_bar]) * (id - self.a_dagger_a[2 * p+self.spin_bar, 2 * p+self.spin_bar]) \
                                    )\
                                * (self.a_dagger_a[2 * p+self.spin, 2 * q+self.spin] - self.a_dagger_a[2 * q+self.spin, 2 * p+self.spin])
                        if np.abs(lbd_sw[3]) > 1.e-10:
                            self.S_list[0] += ( lbd_sw[3] * self.a_dagger_a[2 * p+self.spin_bar, 2 * p+self.spin_bar] * self.a_dagger_a[2 * p+self.spin_bar, 2 * p+self.spin_bar] \
                                    )\
                                * (self.a_dagger_a[2 * p+self.spin, 2 * q+self.spin] - self.a_dagger_a[2 * q+self.spin, 2 * p+self.spin])
        if not self.homogeneous and order > 1:
            raise NotImplementedError(f'SW relations beyond 1st order is implemented only for homogeneous case')
        if order >= 2 : ## S_2 = 0
            pass
        if order >= 3 :
            raise NotImplementedError(f'SW relations beyond 2nd order is not implemented')

    def set_spin_state(self,penalty=1.e5):
        dim_Heis = len(self.nbody_basis)
        H_Heis = scipy.sparse.csr_matrix((dim_Heis, dim_Heis))
        for p in range(self.nb_sites):
            ## penalty over doubly occupied states
            H_Heis += self.a_dagger_a[2 * p, 2 * p] @ self.a_dagger_a[2 * p + 1, 2 * p + 1] * penalty
            for q in range(self.nb_sites):
                if np.abs(self.t_matrix[p, q])>1.e-10:
                    H_Heis += -self.t_matrix[p, q] * (\
                        0.5 * ( self.a_dagger_a[2 * p, 2 * p + 1] @ self.a_dagger_a[2 * q, 2 * q + 1].T + self.a_dagger_a[2 * p, 2 * p + 1].T @ self.a_dagger_a[2 * q, 2 * q + 1]  + \
                        ((self.a_dagger_a[2 * p, 2 * p] - self.a_dagger_a[2 * p + 1, 2 * p + 1]) / 2.) * ((self.a_dagger_a[2 * q, 2 * q] - self.a_dagger_a[2 * q + 1, 2 * q + 1]) / 2.) )\
                    )
        ek,Vk = self._diag(H_Heis)
        del(ek,H_Heis)
        return Vk[:,0]


    def check_relrec(self,order=None,tol=1.e-8):
        if order == None:
            order = self.order
        if order >= 1 : 
            check = scipy.sparse.linalg.norm(self.commutator(self.S_list[0],self.H0)+self.VX)
            print(f'    1st order SW conditions : {True if check<tol else False}')
        if order >= 2 : 
            check = scipy.sparse.linalg.norm(self.commutator(self.S_list[1],self.H0)+self.commutator(self.S_list[0],self.VD))
            print(f'    2nd order SW conditions : {True if check<tol else False}')
        if order >= 3 : 
            check = scipy.sparse.linalg.norm(self.commutator(self.S_list[2],self.H0)+self.commutator(self.S_list[1],self.VD) + self.recursive_commutator(self.S_list[0],self.VX,2)/3.)
            print(f'    3rd order SW conditions : {True if check<tol else False}')
        if order >= 4 : 
            check = scipy.sparse.linalg.norm(self.commutator(self.S_list[3],self.H0)+self.commutator(self.S_list[2],self.VD) + \
                (self.commutator(self.S_list[1],self.commutator(self.S_list[0],self.VX)) + self.commutator(self.S_list[0],self.commutator(self.S_list[1],self.VX)) )/3.)
            print(f'    4rd order SW conditions : {True if check<tol else False}')


    def commutator(self,A,B):
        return A@B - B@A

    def recursive_commutator(self,A,B,order):
        if order == 1:
            return self.commutator(A,B)
        else:
            return self.recursive_commutator(A,self.commutator(A,B),order=order-1)

    def printv(self,string):
        if self.verbose:
            print(string)


    def _diag(self,matrix):
        if self.nb_sites<=6:
            return np.linalg.eigh(matrix.A)
        else:
            return scipy.sparse.linalg.eigsh(matrix,which='SA')

    @property
    def spin_bar(self):
        return 1 if self.spin==0 else 0

    @property
    def nb_states(self):
        return len(self.nbody_basis)

    @property
    def S(self):
        S = scipy.sparse.csr_matrix((self.nb_states,self.nb_states))
        for i,S_ in enumerate(self.S_list):
            S += self.theta[i] * S_ 
        return S

    @property
    def energy(self):
        return self.fermi_hubbard_gs.T@self.H_fermi_hubbard@self.fermi_hubbard_gs

    @property
    def fermi_hubbard_gs(self):
        return scipy.sparse.linalg.expm(-self.S)@self.trial_state
        
    @property
    def Hbar(self):
        return scipy.sparse.linalg.expm(self.S)@self.H_fermi_hubbard@scipy.sparse.linalg.expm(-self.S)

    @property
    def H_fermi_hubbard(self):
        return self.H0 + self.VD + self.VX