import numpy as np 
from pyhub.core.basis import Basis
from pyhub.tools.operators import c_dagger_c, n, _n, empty_operator, opesum, opeexp,idt
from pyhub.tools.models import heisenberg
from scipy.optimize import minimize, Bounds
from time import perf_counter as pc
from itertools import product

class SchriefferWolff():

    def __init__(self,nb_sites,n_up:int,n_down:int,t_matrix:np.ndarray,U:float,trial_state = None,verbose=True):
        self.nb_sites=nb_sites
        self.n_up = n_up
        self.n_down = n_down  
        self.nb_elec=n_up+n_down
        self.t_matrix=t_matrix
        self.mu = np.diag(self.t_matrix)
        self.ek,Vk = np.linalg.eigh(self.t_matrix)
        self.U=U
        self.verbose = verbose
        if isinstance(U,(float,int)):
            self.U = np.full(self.nb_sites,U,dtype=np.float64)
        self.printv('Define n-body basis')
        self.mbbasis = Basis(nb_sites,(n_up,n_down))

        if trial_state == None:
            self.printv('Set trial state as AF spin state')
            self.trial_state = self.set_spin_state()
        else:
            self.trial_state = trial_state


    def kernel(self,theta=1.,order=1,**kwargs_opt):
        t0 = pc()
        self.printv('Set Fermi-Hubbard Hamiltonian')
        self.H0 = opesum([self.U[i]*n((i,'up'))*n((i,'down')) for i in range(self.nb_sites)])
        self.H0.set_basis(self.mbbasis)
        self.VD = opesum([self.mu[i]*n((i,spin)) for i,spin in product(range(self.nb_sites),['up','down'])])
        self.VD.set_basis(self.mbbasis)
        t_matrix_ = self.t_matrix - np.diag(self.mu)
        self.VX = opesum([t_matrix_[i,j]*c_dagger_c((i,spin),(j,spin)) for i,j,spin in product(range(self.nb_sites),range(self.nb_sites),['up','down'])])
        self.VX.set_basis(self.mbbasis)
        self.homogeneous = False if np.linalg.norm(self.mu)>1.e-10 else True
        self.H_fermi_hubbard = self.H0 + self.VD + self.VX
        self.H_fermi_hubbard.set_basis(self.mbbasis)
        self.printv('Build S operator')
        self.order=order
        self.build_S(order=order)
        if theta=='exact':
            self.theta = np.full(self.order,(np.average(self.U)/4)*np.arctan(4/np.average(self.U)))
        elif isinstance(theta,(np.ndarray,list)):
            self.theta = theta
        elif isinstance(theta,float):
            self.theta = np.full(self.order,theta)
        elif theta=='variationnal' or theta=='opt':
            self.printv(f'Classical optimization')
            self.theta = np.full(self.order,(np.average(self.U)/4)*np.arctan(4/np.average(self.U)))#np.full(self.order,1.)#
            self.variationnal(**kwargs_opt)
        else:
            raise ValueError(f'Set correct value of theta, not {theta}')
        self.time = pc() - t0
            

    def variationnal(self,min_energy=True,bounds=True,method='L-BFGS-B'):
        if not min_energy:
            V_ope = opesum([c_dagger_c((i,spin),(j,spin)) if i!=j else empty_operator() for i,j,spin in product(range(self.nb_sites),range(self.nb_sites),['up','down'])])
            V_ope.set_basis(self.mbbasis)
            V = np.abs(V_ope.to_matrix)
        def fun(theta):
            self.theta = theta
            if min_energy:
                return self.energy
            else:
                Hmatrix = self.Hbar
                return np.einsum('ij,ij,ij->',Hmatrix,Hmatrix , V)
        min = minimize(fun,x0=self.theta,method='L-BFGS-B',bounds=Bounds(np.full(self.order,0),np.full(self.order,1.)) if bounds else None )
        self.theta=min.x


    def build_S(self,order=1):
        self.S_list = [empty_operator() for o in range(order)]
        if order >= 1 : 
            for p in range(self.nb_sites):
                for q in range(p):
                    lbd_sw = np.array([self.t_matrix[p,q]/(self.t_matrix[q,q]-self.t_matrix[p,p]) if np.abs(self.t_matrix[p,p]-self.t_matrix[q,q]) else 0., \
                        self.t_matrix[p,q]/self.U[q], -self.t_matrix[p,q]/self.U[p], \
                            self.t_matrix[p,q]/((self.t_matrix[q,q]-self.t_matrix[p,p]) + (self.U[p]-self.U[q])) if np.abs(self.U[p]-self.U[q]) else 0.])
                    for self.spin in ['up','down']:
                        if abs(lbd_sw[1])>1.e-10 or abs(lbd_sw[2])>1.e-10:
                            self.S_list[0] += ( lbd_sw[1] * n((p,self.spin_bar))* _n((q,self.spin_bar)) + \
                                    lbd_sw[2] * n((q,self.spin_bar))* _n((p,self.spin_bar)) \
                                    )\
                                * (c_dagger_c((p,self.spin),(q,self.spin)) - c_dagger_c((q,self.spin),(p,self.spin)))
                        if np.abs(lbd_sw[0]) > 1.e-10:
                            self.S_list[0] += ( lbd_sw[0] * _n((p,self.spin_bar)) * _n((q,self.spin_bar)) \
                                    )\
                                * (c_dagger_c((p,self.spin),(q,self.spin)) - c_dagger_c((q,self.spin),(p,self.spin)))
                        if np.abs(lbd_sw[3]) > 1.e-10:
                            self.S_list[0] += ( lbd_sw[3] * n((p,self.spin_bar)) * n((q,self.spin_bar)) \
                                    )\
                                * (c_dagger_c((p,self.spin),(q,self.spin)) - c_dagger_c((q,self.spin),(p,self.spin)))
            self.S_list[0].set_basis(self.mbbasis)
        if not self.homogeneous and order > 1:
            raise NotImplementedError(f'SW relations beyond 1st order is implemented only for homogeneous case')
        if order >= 2 : ## S_2 = 0
            self.S_list[1] = 0*idt()
            self.S_list[1].set_basis(self.mbbasis)
        if order >= 3 :
            raise NotImplementedError(f'SW relations beyond 2nd order is not implemented')

    def set_spin_state(self):
        H_Heis = heisenberg(-self.t_matrix)
        H_Heis.set_basis(self.mbbasis)
        ek,Vk = self._diag(H_Heis)
        del(ek,H_Heis)
        return Vk[:,0]


    def check_relrec(self,order=None,tol=1.e-8):
        if order == None:
            order = self.order
        if order >= 1 : 
            check = np.linalg.norm(self.commutator(self.S_list[0].to_matrix,self.H0.to_matrix)+self.VX.to_matrix)
            print(f'    1st order SW conditions : {True if check<tol else False}')
        if order >= 2 : 
            check = np.linalg.norm(self.commutator(self.S_list[1].to_matrix,self.H0.to_matrix)+self.commutator(self.S_list[0].to_matrix,self.VD.to_matrix))
            print(f'    2nd order SW conditions : {True if check<tol else False}')
        if order >= 3 : 
            check = np.linalg.norm(self.commutator(self.S_list[2].to_matrix,self.H0)+self.commutator(self.S_list[1].to_matrix,self.VD.to_matrix) + self.recursive_commutator(self.S_list[0].to_matrix,self.VX.to_matrix,2)/3.)
            print(f'    3rd order SW conditions : {True if check<tol else False}')
        if order >= 4 : 
            check = np.linalg.norm(self.commutator(self.S_list[3].to_matrix,self.H0)+self.commutator(self.S_list[2].to_matrix,self.VD) + \
                (self.commutator(self.S_list[1],self.commutator(self.S_list[0].to_matrix,self.VX)) + self.commutator(self.S_list[0].to_matrix,self.commutator(self.S_list[1].to_matrix,self.VX.to_matrix)) )/3.)
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


    def _diag(self,operator):
        if self.nb_sites<=6:
            return np.linalg.eigh(operator.to_matrix)
        else:
            return operator.lanczos()

    @property
    def spin_bar(self):
        return 'down' if self.spin=='up' else 'up'

    @property
    def nb_states(self):
        return self.mbbasis.nstates

    @property
    def S(self):
        S = empty_operator()
        for i,S_ in enumerate(self.S_list):
            S +=  self.theta[i] * S_ 
        S.set_basis(self.mbbasis)
        return S

    @property
    def energy(self):
        return self.H_fermi_hubbard.avg(self.fermi_hubbard_gs)

    @property
    def fermi_hubbard_gs(self):
        return opeexp(-self.S,self.trial_state,unitary=True)
        
    @property
    def Hbar(self):
        expS_matrix = opeexp(-self.S,None,unitary=True)
        return expS_matrix.T@self.H_fermi_hubbard.to_matrix@expS_matrix