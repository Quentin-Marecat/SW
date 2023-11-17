import numpy as np
from sw.tools.circuits import  CU_trotterized
from qiskit import transpile

def list_of_ones(computational_basis_state: int, n_qubits):
    '''
    Indices of ones in the binary expansion of an integer in big endian
    order. e.g. 010110 -> [1, 3, 4] (which is the reverse of the qubit ordering...)
    '''

    bitstring = format(computational_basis_state, 'b').zfill(n_qubits)

    return [abs(j-n_qubits+1) for j in range(len(bitstring)) if bitstring[j] == '1']


def count_ones_bitstring(n):
   '''
   Count the number of ones in bitstring given by integer n
   '''
   n = str(bin(n))
   one_count = 0
   for i in n:
      if i == "1":
         one_count+=1
   return one_count


def evaluate_statevector(theta,initial_circuit,SW_PauliSum,Hmatrix,backend,nshots=None):
    '''
    This function returns the energy.
    Careful about the endian-ordering... For the moment, I only managed to make it work always with Hmatrix = Hfermion.to_matrix().A, and reverse ordering.
    '''

    if isinstance(theta,(float,int,list,np.ndarray)):
      SW_PauliSum_theta = SW_PauliSum.mul(theta[0])
      total_circuit = initial_circuit.compose(CU_trotterized(SW_PauliSum_theta))
    else:
      total_circuit = initial_circuit
    # Reverse the circuit which is sorted with a different endian.
    total_circuit = total_circuit.reverse_bits()
    total_circuit.save_statevector()
    # Transpile for simulator
    total_circuit = transpile(total_circuit, backend)
    result = backend.run(total_circuit).result()
    trotterized_state = result.get_statevector(total_circuit)
    energy = (trotterized_state @ Hmatrix @ trotterized_state.conj()).real

    return energy


def evaluate_SWiterative_statevector(circuit,SW_PauliSum_list,Hmatrix,backend):
    '''
    This function returns the energy.
    Careful about the endian-ordering... For the moment, I only managed to make it work always with Hmatrix = Hfermion.to_matrix().A, and reverse ordering.
    '''

    for SW_PauliSum in SW_PauliSum_list:
      circuit = circuit.compose(CU_trotterized(SW_PauliSum))
    # Reverse the circuit which is sorted with a different endian.
    circuit = circuit.reverse_bits()
    circuit.save_statevector()
    # Transpile for simulator
    circuit = transpile(circuit, backend)
    result = backend.run(circuit).result()
    trotterized_state = result.get_statevector(circuit)
    energy = (trotterized_state @ Hmatrix @ trotterized_state.conj()).real

    return energy

def evaluate(theta,initial_circuit,SW_PauliSum,observable_PauliSum,backend,nshots):
    '''
    This function returns the expectation value of the observable.
    '''
    if isinstance(theta,(float,int,list,np.ndarray)):
      SW_PauliSum_theta = SW_PauliSum.mul(theta[0])
      total_circuit = initial_circuit.compose(CU_trotterized(SW_PauliSum_theta))
    else:
      total_circuit = initial_circuit
    energy = sampled_expectation_value(total_circuit,observable_PauliSum,backend,nshots=nshots)

    return energy

def evaluate_SWiterative(circuit,SW_PauliSum_list,observable_PauliSum,backend,nshots):
    '''
    This function returns the expectation value of the observable.
    '''

    for SW_PauliSum in SW_PauliSum_list:
      circuit = circuit.compose(CU_trotterized(SW_PauliSum))
    energy = sampled_expectation_value(circuit,observable_PauliSum,backend,nshots=nshots)

    return energy

def sampled_expectation_value(original_circuit,operator,backend,nshots=1024):
    rot_dic = { 'X' : lambda qubit : circuit.h(qubit),
                'Y' : lambda qubit : circuit.rx(np.pi/2., qubit)}

    nterms = len(operator)
    nqubits = operator.num_qubits
    nshots_per_pauli = int(nshots/nterms)
    expectation_value = 0
    for i in range(nterms):
        circuit = original_circuit.copy()
        # Begining of the circuit fragment: we detect if Pauli op is X, Y or Z
        # and apply a rotation gate accordingly on the associated qubits
        # store the places where there is a X, Y or Z operators as well (all rotated to Z)
        list_Z = []
        for j in reversed(range(nqubits)):
            # For some reason, if str(operator[i].to_pauli_op().primitive) --> ZII, we have
            # str(operator[i].to_pauli_op().primitive[0]) = "I" and
            # str(operator[i].to_pauli_op().primitive[2]) = "Z".............. 
            # This messed up so much with my brain.
            if str(operator[i].to_pauli_op().primitive[j]) == 'I':
                continue
            elif str(operator[i].to_pauli_op().primitive[j]) == 'Z':
                list_Z.append(j)
            else:
                rot_dic[str(operator[i].to_pauli_op().primitive[j])](j)
                list_Z.append(j)

        circuit.measure_all()

        # Transpile for simulator
        circuit = transpile(circuit, backend)

        # Run and get counts
        result = backend.run(circuit,shots=nshots).result()
        counts = result.get_counts(circuit) # The string is little-endian (cr[0] on the right hand side).

        # transform the dictionary with binary to integer:
        proba_computational_basis = {}
        for item in counts.items():
          proba_computational_basis[str(int(item[0],2))] = item[1]/nshots

        # Compute the energy by combining the proba:
        for integer_bitstring in range(2**nqubits):
          if str(integer_bitstring) in proba_computational_basis:
            phase = 1

            # Determine the phase:
            for qubit in list_Z:
              if qubit in list_of_ones(integer_bitstring,nqubits):
                phase *= -1
            # Determine the expectation value:
            expectation_value += phase * proba_computational_basis[str(integer_bitstring)] * operator[i].to_pauli_op().coeff

    return expectation_value
