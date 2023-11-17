import sys
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


def CU_trotterized(operator):
    """
    Implement e^operator on a quantum computer.
    Compared to Cirq, the identities are written in the Pauli operators,
    and one has to be careful about the ordering of the Pauli operator !
    Indeed, operator[i].to_pauli_op().primitive[nqubits-j-1] actually correspond to qubit j !
    Arguments:
     - operator: the qubit operator to be exponentiated and trotterized.
    Returns the circuit.
    """
    rot_dic = { 'X' : lambda qubit, sign : circuit.h(qubit),
                'Y' : lambda qubit, sign : circuit.rx(sign*np.pi/2., qubit)}
    nterms=len(operator)
    nqubits=operator.num_qubits
    circuit = QuantumCircuit(nqubits)
    for i in range(nterms):
        cnot_qubit_pairs = []
        # Begining of the circuit fragment: we detect if Pauli op is X, Y or Z
        # and apply a rotation gate accordingly on the associated qubits
        for j in range(nqubits):
            if str(operator[i].to_pauli_op().primitive[nqubits-j-1]) == 'I':
                continue
            elif str(operator[i].to_pauli_op().primitive[nqubits-j-1]) == 'Z':
                cnot_qubit_pairs.append(j)
            else:
                rot_dic[str(operator[i].to_pauli_op().primitive[nqubits-j-1])](j,1)
                cnot_qubit_pairs.append(j)
        # Middle of the circuit fragment
        # First, We extend the chain of Z operators thoughout the required
        # qubits to match the length of the Pauli string
        if len(cnot_qubit_pairs) > 1 and nqubits > 1:
            for j in range(len(cnot_qubit_pairs)-1):
                circuit.cx(cnot_qubit_pairs[j],cnot_qubit_pairs[j+1])
        # Second, we apply a Z rotation gate on the qubit with the last non-identity gate of the Pauli string.
        if len(cnot_qubit_pairs) > 0: # the length is zero only if there were only identities.
            circuit.rz((+2.0*operator[i].to_pauli_op().coeff.imag),cnot_qubit_pairs[-1])
            # factor 2 because Rz(lambda) ---> e^[i lambda/2].
        # Then, we extend again the chain of Z operators thoughtout the required
        # qubits to match the length of the Pauli string
        if len(cnot_qubit_pairs) > 1 and nqubits > 1:
            for j in range(len(cnot_qubit_pairs)-2,-1,-1):
                circuit.cx(cnot_qubit_pairs[j],cnot_qubit_pairs[j+1])
        # Finally,  we detect again if Pauli op is X, Y or Z
        # and apply a rotation gate accordingly on the associated qubits
        for j in range(nqubits-1,-1,-1):
            if str(operator[i].to_pauli_op().primitive[nqubits-j-1]) == 'I' or str(operator[i].to_pauli_op().primitive[nqubits-j-1]) == 'Z':
                continue
            else:
                rot_dic[str(operator[i].to_pauli_op().primitive[nqubits-j-1])](j,-1)

    return circuit

def initial_circ_dimer(n_sites,state='Neel',phase=False,varphi=np.pi):
    '''
    Create the circuit associated to the AntiFerroMagnetic Neel state, or Ionic state.
    '''
    initial_circuit = QuantumCircuit(2*n_sites)

    if state == "Neel": # cos(varphi/2) |100110011001...>  - sin(varphi/2) |0110011001100110...>) (Check that S^2 = 0)
       initial_circuit.ry(varphi,0)
       initial_circuit.x(1)
       initial_circuit.x(3)
       initial_circuit.cx(0,1)
       initial_circuit.cx(1,2)
       initial_circuit.cx(2,3)
       if phase: initial_circuit.z(3)
       for i in range(1,n_sites//2):
         initial_circuit.cx(3,4*i)
         initial_circuit.cx(3,4*i+3)
         initial_circuit.cx(2,4*i+1)
         initial_circuit.cx(2,4*i+2)
    elif state == "Ionic": # cos(varphi/2) |110011001100...> + sin(varphi/2) |001100110011...>
       initial_circuit.ry(varphi,0)
       initial_circuit.x(2)
       initial_circuit.cx(0,1)
       initial_circuit.cx(1,2)
       initial_circuit.cx(2,3)
       if phase: initial_circuit.z(3)
       for i in range(1,n_sites//2):
         initial_circuit.cx(0,4*i)
         initial_circuit.cx(0,4*i+1)
         initial_circuit.cx(2,4*i+2)
         initial_circuit.cx(2,4*i+3)
    else:
       sys.exit('Wrong initial state, not implemented.')

    return initial_circuit


def RVB_inspired_ansatz(given_params, N, BC, num_layers, mode):
    # Function that generates wave function corresponding to output
    # of quantum circuit that implements RVB-inspired ansatz with
    # given parameters to be applied to eSWAP gates
    
    # Fixed parameters: N, BC, mode, num_layers
    # N: number of sites (i.e., number of qubits)
    # BC = 'OBC' or 'PBC'
    # num_layers = number of layers of eSWAP, including even and odd pairs
    # mode = '1', '2' (see description of each below)
    
    # Free parameters to be optimized: given_params
    
    # mode = '1': Each eSWAP has its own free parameter, so the
    #             given_params input should be an array with
    #             num_layers x N-1 entries for OBCs or 
    #             num_layers x N entries for PBCs. For each layer,
    #             there are N-1 (OBC) or N parameters. The first
    #             N/2 (-1) are for the odd-pair eSWAPs. The following
    #             N/2 are for the even-pair eSWAPs.
    # mode = '2': All eSWAPs within the same even or odd pairs sublayer
    #             have the same free parameter, so given_params should
    #             be array with num_layers x 2 entries. For each layer,
    #             the first entry is for the odd-pairs layer and the
    #             second entry is for the even-pair layer
    
    if mode == '1':
        params = given_params
        
    if mode == '2':
        params = []
        for layer in range(num_layers):
            if BC == 'OBC':
                params = np.concatenate((params, [given_params[2*layer]]*(int(N/2)-1)))
            else:
                params = np.concatenate((params, [given_params[2*layer]]*(int(N/2))))
            params = np.concatenate((params, [given_params[2*layer+1]]*(int(N/2))))
                
    q = QuantumRegister(N)
    qc = QuantumCircuit(q)

    # Initial state: spin singlets initialized at even pairs
    # (0,1), (2,3), (4,5), ...
    for i in range(int(N/2)):
        qc.x(q[2*i])
        qc.x(q[2*i+1])
        qc.h(q[2*i])
        qc.cx(q[2*i],q[2*i+1])
    
    # Variational layers
    for layer in range(num_layers):
        if BC == 'OBC':
            # Odd pairs: (1,2), (3,4), (5,6), ...
            start_odd = layer*(N-1)
            end_odd   = layer*(N-1)+(int(N/2)-1)
            theta_odd = params[start_odd:end_odd]
            for i in range(int(N/2)-1):
                qc.cx(q[2*i+1], q[2*i+2])
                qc.crx(theta_odd[i], q[2*i+2], q[2*i+1])
                qc.x(q[2*i+2])
                qc.rz(-theta_odd[i]/2, q[2*i+2])
                qc.x(q[2*i+2])
                qc.cx(q[2*i+1],q[2*i+2])

            # Even pairs: (0,1), (2,3), (4,5), ...
            start_even = layer*(N-1)+(int(N/2)-1)
            end_even   = (layer+1)*(N-1)
            theta_even = params[start_even:end_even]
            for i in range(int(N/2)):
                qc.cx(q[2*i], q[2*i+1])
                qc.crx(theta_even[i], q[2*i+1], q[2*i])
                qc.x(q[2*i+1])
                qc.rz(-theta_even[i]/2, q[2*i+1])
                qc.x(q[2*i+1])
                qc.cx(q[2*i],q[2*i+1])
        else:
            # Odd pairs: (1,2), (3,4), (5,6), ...
            start_odd = layer*(N)
            end_odd   = layer*(N)+(int(N/2))
            theta_odd = params[start_odd:end_odd]
            for i in range(int(N/2)-1):
                qc.cx(q[2*i+1], q[2*i+2])
                qc.crx(theta_odd[i], q[2*i+2], q[2*i+1])
                qc.x(q[2*i+2])
                qc.rz(-theta_odd[i]/2, q[2*i+2])
                qc.x(q[2*i+2])
                qc.cx(q[2*i+1],q[2*i+2])
            ## Boundary conditions
            qc.cx(q[N-1], q[0])
            qc.crx(theta_odd[int(N/2)-1], q[0], q[N-1])
            qc.x(q[0])
            qc.rz(-theta_odd[int(N/2)-1]/2, q[0])
            qc.x(q[0])
            qc.cx(q[N-1],q[0])

            # Even pairs: (0,1), (2,3), (4,5), ...
            start_even = layer*(N)+(int(N/2))
            end_even   = (layer+1)*(N)
            theta_even = params[start_even:end_even]
            for i in range(int(N/2)):
                qc.cx(q[2*i], q[2*i+1])
                qc.crx(theta_even[i], q[2*i+1], q[2*i])
                qc.x(q[2*i+1])
                qc.rz(-theta_even[i]/2, q[2*i+1])
                qc.x(q[2*i+1])
                qc.cx(q[2*i],q[2*i+1])

    return qc


def fermionic_version_of_spin_wave_function_spin_ordered(spin_wf_qc, N):
    q = QuantumRegister(2*N)
    qc = QuantumCircuit(q)
    
    spin_qubits = []
    for i in range(N):
        spin_qubits.append(q[i])
    
    state_preparation_subcirc = spin_wf_qc.to_instruction()
    qc.append(state_preparation_subcirc, spin_qubits)
    
    for i in range(0,N,2):
        qc.x(q[i])
        qc.cx(q[i],q[N+i])
        qc.x(q[i])
    
    for i in range(1,N,2):
        qc.x(q[N+i])
        qc.x(q[i])
        qc.h(q[N+i])
        qc.cx(q[i],q[N+i])
        qc.h(q[N+i])
        qc.x(q[N+i])
        qc.cx(q[i],q[N+i])
        qc.x(q[i])
        
    
    return qc


def fermionic_version_of_spin_wave_function_site_ordered(spin_wf_qc, N):
    q = QuantumRegister(2*N)
    qc = QuantumCircuit(q)
    
    spin_qubits = []
    for i in range(N):
        spin_qubits.append(q[2*i])
    
    state_preparation_subcirc = spin_wf_qc.to_instruction()
    qc.append(state_preparation_subcirc, spin_qubits)
    
    for i in range(N):
        qc.x(q[2*i])
        qc.cx(q[2*i],q[2*i+1])
        qc.x(q[2*i])
    
    return qc
