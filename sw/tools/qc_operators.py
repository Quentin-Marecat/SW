import numpy as np
from qiskit_nature.operators.second_quantization import FermionicOp

def Hubbard_1D_operator(n_sites,U,t_matrix):
    '''
    1D Hubbard operator in qiskit.   
    '''
    n_qubits = 2*n_sites
    Hloc_op = 0
    V_op = 0

    for i in range(n_sites):
      Hloc_op += FermionicOp(("N_{}".format(2*i),t_matrix[i,i]),register_length=n_qubits)
      Hloc_op += FermionicOp(("N_{}".format(2*i+1),t_matrix[i,i]),register_length=n_qubits)
      Hloc_op += FermionicOp(("N_{} N_{}".format(2*i,2*i+1),U[i]),register_length=n_qubits)
    for i in range(n_sites):
      for j in range(n_sites):
        V_op += FermionicOp(("+_{} -_{}".format(2*i,2*j),t_matrix[i,j]),register_length=n_qubits)
        V_op += FermionicOp(("+_{} -_{}".format(2*i+1,2*j+1),t_matrix[i,j]),register_length=n_qubits)
    return Hloc_op + V_op

def sz_operator(n_qubits):
    s_z = 0

    for i in range(n_qubits//2):
      s_z += FermionicOp(("N_{}".format(2*i),0.5),register_length=n_qubits)
      s_z -= FermionicOp(("N_{}".format(2*i+1),0.5),register_length=n_qubits)

    return s_z

def N_operator(n_qubits):
    N = 0

    for i in range(n_qubits//2):
      N += FermionicOp(("N_{}".format(2*i),1),register_length=n_qubits)
      N += FermionicOp(("N_{}".format(2*i+1),1),register_length=n_qubits)

    return N

def Heisenberg_1D_operator(n_sites,t_matrix):
    '''
    1D Heisenberg operator in qiskit.   
    '''
    n_qubits = 2*n_sites
    V_op = 0

    for i in range(n_sites):
      for j in range(n_sites):
        s_moins_i = FermionicOp(("+_{} -_{}".format(2*i+1,2*i),1),register_length=n_qubits)
        s_plus_i = FermionicOp(("+_{} -_{}".format(2*i,2*i+1),1),register_length=n_qubits)
        s_moins_j = FermionicOp(("+_{} -_{}".format(2*j+1,2*j),1),register_length=n_qubits)
        s_plus_j = FermionicOp(("+_{} -_{}".format(2*j,2*j+1),1),register_length=n_qubits)
        sz_i = FermionicOp(("N_{}".format(2*i),0.5),register_length=n_qubits) - FermionicOp(("N_{}".format(2*i+1),0.5),register_length=n_qubits)
        sz_j = FermionicOp(("N_{}".format(2*j),0.5),register_length=n_qubits) - FermionicOp(("N_{}".format(2*j+1),0.5),register_length=n_qubits)
        V_op += t_matrix[i,j]* (.5*(s_plus_i@s_moins_j + s_moins_i@s_plus_j) + sz_i@sz_j)
    return V_op
    
def s2_operator(n_qubits):
    ''' 
    S2 = S- S+ + Sz(Sz+1)
    I use the usual sorting as in OpenFermion, i.e. 1up 1down, 2up 2down, etc...
    '''
    s2_op = 0
    s_moins = 0
    s_plus = 0
    s_z = sz_operator(n_qubits)
    
    for i in range(n_qubits//2):
      s_moins += FermionicOp(("+_{} -_{}".format(2*i+1,2*i),1),register_length=n_qubits)
      s_plus += FermionicOp(("+_{} -_{}".format(2*i,2*i+1),1),register_length=n_qubits)
      
    s2_op = s_moins @ s_plus + s_z @ s_z + s_z
    return s2_op

def SW_operator(n_sites,lbd) -> FermionicOp:
    '''
    Returns the SW operator for the inhomogeneous 1D Hubbard chain.
    I think it only works with periodic condition right now ?
    '''

    n_qubits = 2*n_sites
    SW_op = 0

    for i in range(n_sites):
      for j in range(n_sites):
        Pij0_up   = lbd[0][i,j]*(FermionicOp(("I",1),register_length=n_qubits)\
                            + FermionicOp("N_{} N_{}".format(2*i,2*j),register_length=n_qubits)\
                            - FermionicOp("N_{}".format(2*i),register_length=n_qubits)\
                            - FermionicOp("N_{}".format(2*j),register_length=n_qubits))
        Pij1_up   = lbd[1][i,j]*(FermionicOp("N_{}".format(2*i),register_length=n_qubits)\
                            - FermionicOp("N_{} N_{}".format(2*i,2*j),register_length=n_qubits))
        Pij2_up   = lbd[2][i,j]*(FermionicOp("N_{}".format(2*j),register_length=n_qubits)\
                            - FermionicOp("N_{} N_{}".format(2*i,2*j),register_length=n_qubits))
        Pij3_up   = lbd[3][i,j]*(FermionicOp("N_{} N_{}".format(2*i,2*j),register_length=n_qubits))
        Pij0_down = lbd[0][i,j]*(FermionicOp(("I",1),register_length=n_qubits)\
                            + FermionicOp("N_{} N_{}".format(2*i+1,2*j+1),register_length=n_qubits)\
                            - FermionicOp("N_{}".format(2*i+1),register_length=n_qubits)\
                            - FermionicOp("N_{}".format(2*j+1),register_length=n_qubits))
        Pij1_down = lbd[1][i,j]*(FermionicOp("N_{}".format(2*i+1),register_length=n_qubits)\
                            - FermionicOp("N_{} N_{}".format(2*i+1,2*j+1),register_length=n_qubits))
        Pij2_down = lbd[2][i,j]*(FermionicOp("N_{}".format(2*j+1),register_length=n_qubits)\
                            - FermionicOp("N_{} N_{}".format(2*i+1,2*j+1),register_length=n_qubits))
        Pij3_down = lbd[3][i,j]*(FermionicOp("N_{} N_{}".format(2*i+1,2*j+1),register_length=n_qubits))

        cicj_up   = FermionicOp("+_{} -_{}".format(2*i  ,2*j  ),register_length=n_qubits)
        cicj_down = FermionicOp("+_{} -_{}".format(2*i+1,2*j+1),register_length=n_qubits)
        cjci_up   = FermionicOp("+_{} -_{}".format(2*j  ,2*i  ),register_length=n_qubits)
        cjci_down = FermionicOp("+_{} -_{}".format(2*j+1,2*i+1),register_length=n_qubits)

        Pij_up = Pij0_up + Pij1_up + Pij2_up + Pij3_up
        Pij_down = Pij0_down + Pij1_down + Pij2_down + Pij3_down

        SW_op += 0.5*(Pij_up@(cicj_down - cjci_down) + Pij_down@(cicj_up - cjci_up))

    return SW_op