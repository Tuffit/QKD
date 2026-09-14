import numpy as np
import scipy 
from scipy.sparse import csr_matrix
from functools import reduce

def pauli_matrices() -> dict:
    I=np.array([[1,0],[0,1]], dtype=complex)
    X=np.array([[0,1],[1,0]], dtype=complex)
    Y=np.array([[0, -1j],[1j,0]], dtype=complex)
    Z=np.array([[1,0],[0,-1]], dtype=complex) 
    return {"I":csr_matrix(I), "X":csr_matrix(X),"Y":csr_matrix(Y), "Z":csr_matrix(Z) }

#not using it anymore, but Ill keep it
def pauli_on_site(op: scipy.sparse.csr_matrix, site: int, n_qubits: int) -> scipy.sparse.csr_matrix:
    paulis=pauli_matrices()
    I=paulis["I"]
    chain=[I]*n_qubits
    chain[site]=op
    result=reduce(scipy.sparse.kron,chain)
    return result

def jw_string(site: int, op: str, n_qubits: int) -> csr_matrix:
    paulis = pauli_matrices()
    I, Z = paulis["I"], paulis["Z"]
    terminal = paulis[op]
    chain = [Z if k < site else (terminal if k == site else I) for k in range(n_qubits)]
    return reduce(scipy.sparse.kron, chain)


def majorana_operators(n_qubits:int) -> list[scipy.sparse.csr_matrix]:
    chi=[]
    for i in range(n_qubits):
        chi_2j =  1 / np.sqrt(2) * (jw_string(i, "X", n_qubits))
        chi_2j1 = 1 / np.sqrt(2) * (jw_string(i, "Y", n_qubits))
        chi.append(chi_2j)
        chi.append(chi_2j1)        
    return chi